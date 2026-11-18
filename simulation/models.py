"""학습 모델 모음.

1) PowerState    : 전력만 보고 운전 상태(정지/냉각/제상)를 가려내는 비지도 모델 (가우시안 혼합)
2) ThermalModel  : 1차 열 모델 파라미터를 최소제곱으로 추정 + 남은 오차를 보정하는 회귀 모델
3) VirtualTemp   : 압력·운전 상태로 고내 온도를 추정하는 '가상 온도 센서' (그래디언트 부스팅)
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

from sim import tsat_r134a


# ---------------------------------------------------------------- 1) 전력 → 운전 상태
class PowerState:
    """전력 분포에 가우시안 3개를 맞춘 뒤 평균이 낮은 순서로 정지·냉각·제상으로 이름 붙인다.
    라벨이 필요 없으므로 실제 현장 데이터에도 그대로 쓸 수 있다."""

    def fit(self, W):
        self.gmm = GaussianMixture(3, random_state=0).fit(np.asarray(W).reshape(-1, 1))
        order = np.argsort(self.gmm.means_.ravel())
        self.map = np.empty(3, int)
        self.map[order] = [0, 1, 2]  # 0=정지, 1=냉각, 2=제상
        # 제상 히터 전력은 거의 일정하다. 제상으로 분류된 값의 중앙값보다 30W 이상 낮으면
        # 컴프레서 기동 돌입 전력으로 본다 (하한도 데이터에서 정한다).
        self.def_floor = -np.inf
        W = np.asarray(W, float)
        s = self.predict(W)
        if (s == 2).any():
            self.def_floor = float(np.median(W[s == 2]) - 30)
        return self

    def predict(self, W, min_defrost=5):
        W = np.asarray(W, float)
        s = self.map[self.gmm.predict(W.reshape(-1, 1))]
        s[(s == 2) & (W < getattr(self, "def_floor", -np.inf))] = 1
        # 컴프레서 기동 직후 1~2분의 돌입 전력은 제상 히터처럼 보일 수 있다.
        # 실제 제상은 수십 분 이어지므로, 짧게 끝나는 '제상'은 냉각으로 바로잡는다.
        i, n = 0, len(s)
        while i < n:
            if s[i] == 2:
                j = i
                while j < n and s[j] == 2:
                    j += 1
                if j - i < min_defrost:
                    s[i:j] = 1
                i = j
            else:
                i += 1
        return s


def state_features(state):
    """상태 배열에서 u, d, '현재 상태가 몇 분째인지'를 만든다."""
    u = (state == 1).astype(float)
    d = (state == 2).astype(float)
    since = np.zeros(len(state))
    for i in range(1, len(state)):
        since[i] = since[i - 1] + 1 if state[i] == state[i - 1] else 0
    return u, d, np.minimum(since, 600)


# ---------------------------------------------------------------- 2) 열 모델
class ThermalModel:
    """다음 분 온도 = 현재 온도 + a(Ta-T) - b*u + h*d + c + 보정(g)

    물리층(a, b, h, c)은 최소제곱으로, 보정층(g)은 물리층이 남긴 오차를 그래디언트 부스팅으로 학습한다.
    a·b·h는 해석 가능한 값이라 발표에서 '단열 시정수', '냉각 능력'으로 설명할 수 있다."""

    def __init__(self, use_correction=True):
        self.use_correction = use_correction

    @staticmethod
    def _corr_X(T, Ta, u, d, since):
        return np.column_stack([u, d, since, Ta, Ta - T])

    def fit(self, T, Ta, u, d, since):
        # 상수항은 넣지 않는다. 주변온도-고내온도 차이가 거의 일정해서 상수항과 a가 서로 구별되지 않기 때문이다.
        # 문 열림은 드물고 큰 '튀는 값'이므로, 보통의 최소제곱 대신 튀는 값에 덜 끌려가는 Huber 회귀를 쓴다.
        # 이렇게 해야 모델이 '문 열림이 없는 설비 자체의 물리'를 배우고, 문 열림은 예측보다 높은 쪽 잔차로 남는다.
        dT = T[1:] - T[:-1]
        X = np.column_stack([Ta[:-1] - T[:-1], -u[:-1], d[:-1]])
        hub = HuberRegressor(fit_intercept=False, epsilon=1.35, alpha=0.0, max_iter=500).fit(X, dT)
        self.a, self.b, self.h = hub.coef_
        self.c = 0.0
        resid = dT - X @ hub.coef_
        # 문 열림 등 모델에 없는 열 유입의 평균 (안전 쪽으로 보수적인 예측에만 사용)
        self.extra_heat = float(max(np.mean(resid), 0.0))
        # 보정층을 믿어도 되는 범위: 학습 데이터에서 본 (주변온도-고내온도) 범위와 상태 지속시간
        gap = Ta[:-1] - T[:-1]
        self.dom = (float(np.quantile(gap, 0.005)), float(np.quantile(gap, 0.995)))
        # 상태별(정지/냉각/제상)로 학습 때 본 최대 지속시간. 이보다 길게 이어지면 보정층을 끈다.
        st = (u > 0).astype(int) + 2 * (d > 0).astype(int)
        self.since_max = {k: float(since[st == k].max()) if (st == k).any() else 0.0 for k in (0, 1, 2)}
        self.g = None
        if self.use_correction:
            # 보정층도 중앙값 기준(절대오차 손실)으로 학습해 같은 이유로 문 열림에 끌려가지 않게 한다.
            self.g = HistGradientBoostingRegressor(loss="absolute_error", max_iter=200,
                                                   learning_rate=0.05, random_state=0)
            self.g.fit(self._corr_X(T[:-1], Ta[:-1], u[:-1], d[:-1], since[:-1]), resid)
        return self

    def step(self, T, Ta, u, d, since):
        nxt = T + self.a * (Ta - T) - self.b * u + self.h * d + self.c
        if self.g is not None:
            # 학습 때 본 적 없는 상황(예: 수십 분 넘게 이어진 제상, 주변보다 뜨거운 고내)에서는
            # 트리 모델의 보정이 엉뚱한 값을 낼 수 있으므로 끄고 물리층만 쓴다.
            gap = Ta - T
            st = (u > 0).astype(int) + 2 * (d > 0).astype(int)
            smax = np.choose(st, [self.since_max[0], self.since_max[1], self.since_max[2]])
            inside = (gap >= self.dom[0]) & (gap <= self.dom[1]) & (since <= smax)
            corr = self.g.predict(self._corr_X(T, Ta, u, d, since))
            nxt = nxt + np.where(inside, corr, 0.0)
        return nxt

    def fit_bootstrap(self, T, Ta, u, d, n_boot=30, block=360, seed=0):
        """물리 파라미터의 불확실성: 6시간 블록을 복원추출해 Huber 회귀를 다시 맞춘다."""
        rng = np.random.default_rng(seed)
        dT = T[1:] - T[:-1]
        X = np.column_stack([Ta[:-1] - T[:-1], -u[:-1], d[:-1]])
        nb = len(dT) // block
        coefs = []
        for _ in range(n_boot):
            pick = rng.integers(0, nb, nb)
            idx = np.concatenate([np.arange(b * block, (b + 1) * block) for b in pick])
            hub = HuberRegressor(fit_intercept=False, epsilon=1.35, alpha=0.0, max_iter=500).fit(X[idx], dT[idx])
            coefs.append(hub.coef_)
        self.boot = np.array(coefs)
        return self

    def rollout_spread(self, T0, Ta, u, d):
        """부트스트랩 파라미터마다 물리층만으로 굴린 결과의 표준편차 (외삽 불확실성)."""
        a, b, h = self.boot[:, 0], self.boot[:, 1], self.boot[:, 2]
        T = np.full(len(a), float(T0))
        out = np.empty(len(Ta))
        for i in range(len(Ta)):
            out[i] = T.std()
            T = T + a * (Ta[i] - T) - b * u[i] + h * d[i]
        return out

    def open_loop(self, T, Ta, u, d, since, H):
        """모든 시점 t에 대해 t-H의 보고 온도에서 출발해 H분 동안 '전력이 말해 주는 운전 상태'로
        온도를 굴려 본 예측값. 보고 온도 T가 진짜라면 T[t]와 비슷해야 한다."""
        n = len(T)
        out = np.full(n, np.nan)
        if n <= H:
            return out
        sim = T[:n - H].copy()
        for k in range(H):
            sl = slice(k, n - H + k)
            ok = ~np.isnan(sim)
            nxt = np.full_like(sim, np.nan)
            nxt[ok] = self.step(sim[ok], Ta[sl][ok], u[sl][ok], d[sl][ok], since[sl][ok])
            sim = nxt
        out[H:] = sim
        return out

    def rollout(self, T0, Ta, u, d, since, extra=0.0):
        """한 시작점에서 끝까지 굴린다 (복원·남은 시간 예측용).
        extra: 매 분 더할 열(°C/분). 보수적 예측에서 평균 문 열림 열을 더할 때 쓴다."""
        out = np.empty(len(Ta))
        T = np.array([T0], float)
        for i in range(len(Ta)):
            out[i] = T[0]
            T = self.step(T, Ta[i:i + 1], u[i:i + 1], d[i:i + 1], since[i:i + 1]) + extra
        return out


# ---------------------------------------------------------------- 3) 가상 온도 센서
def pressure_features(P, u, d, since):
    P = np.asarray(P, float)
    lag = lambda x, k: np.concatenate([np.full(k, x[0]), x[:-k]])
    Ts = tsat_r134a(P)
    return np.column_stack([Ts, lag(Ts, 1), lag(Ts, 3), Ts - lag(Ts, 3), u, d, since])


class VirtualTemp:
    """압력(포화온도로 변환)과 운전 상태로 고내 온도를 추정한다."""

    def fit(self, P, u, d, since, T):
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        self.m.fit(pressure_features(P, u, d, since), T)
        return self

    def predict(self, P, u, d, since):
        return self.m.predict(pressure_features(P, u, d, since))
