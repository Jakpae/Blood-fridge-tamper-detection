"""탐지 모듈: 통계 기준선과 학습 모델 잔차를 같은 틀(표준화 → CUSUM)로 다룬다."""
import numpy as np

from config import WINDOW_H, CUSUM_K, Z_CLIP, THERMAL_SCALE_FLOOR
from models import state_features, pressure_features  # noqa: F401  (다른 단계에서 재사용)


def robust_scale(x):
    x = x[~np.isnan(x)]
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return med, max(mad, 1e-6)


def cusum(z, k=CUSUM_K):
    """양방향 CUSUM. 값이 없는 시점(NaN)에서는 누적값을 그대로 유지한다."""
    sp = np.zeros(len(z))
    sn = np.zeros(len(z))
    for i in range(1, len(z)):
        if np.isnan(z[i]):
            sp[i], sn[i] = sp[i - 1], sn[i - 1]
            continue
        sp[i] = max(0.0, sp[i - 1] + z[i] - k)
        sn[i] = max(0.0, sn[i - 1] - z[i] - k)
    return np.maximum(sp, sn), sp, sn


def switch_points(state):
    """전력으로 본 컴프레서 켜짐(정지→냉각)·꺼짐(냉각→정지) 시점. 제상 전후 전환은 제외."""
    on = np.flatnonzero((state[1:] == 1) & (state[:-1] == 0)) + 1
    off = np.flatnonzero((state[1:] == 0) & (state[:-1] == 1)) + 1
    return on, off


def raw_signals(df, art):
    """각 모듈의 원신호(표준화 전)를 계산한다.

    art: step2에서 저장한 학습 결과(dict)
    """
    T, P, W, Ta = (df[c].to_numpy(float) for c in ("T_rep", "P_rep", "W", "Ta"))
    state = art["power"].predict(W)
    u, d, since = state_features(state)

    sig = {}
    # --- 통계 기준선
    sig["z_T"] = T                                  # 개별값 (z-score는 표준화 단계에서)
    sig["z_P"] = P
    X = np.column_stack([T, P]) - art["mu_TP"]
    sig["maha"] = np.sqrt(np.einsum("ij,jk,ik->i", X, art["icov_TP"], X))
    sig["cusum_T"] = T                              # 원시 온도에 CUSUM
    # --- 학습 모델
    T_virtual = art["virtual"].predict(P, u, d, since)
    sig["virtual"] = T - T_virtual                  # 보고 온도 vs 압력으로 추정한 온도 (누적 검사)
    sig["virtual_pt"] = T - T_virtual               # 같은 값을 한 시점씩 검사 (순간 튐)
    T_sim = art["thermal"].open_loop(T, Ta, u, d, since, WINDOW_H)
    sig["thermal"] = T - T_sim                      # 보고 온도 vs 전력으로 굴린 열 모델
    Tv_sim = art["thermal"].open_loop(T_virtual, Ta, u, d, since, WINDOW_H)
    sig["thermal_P"] = T_virtual - Tv_sim           # 압력 쪽 이야기 vs 전력 쪽 이야기 (원인 판별용)
    # --- 제어 논리 정합성: 온도조절기는 약 5°C에서 켜고 약 3°C에서 끈다.
    #     전력으로 잡은 전환 시점의 보고 온도가 학습한 전환 온도와 맞는지 본다.
    on, off = switch_points(state)
    sw = np.full(len(T), np.nan)
    sw[on] = T[on - 1] - art["switch_T"][0]
    sw[off] = T[off - 1] - art["switch_T"][1]
    sig["switch"] = sw
    # 원인 판별용: 압력으로 추정한 온도도 전환 시점에 맞는지 (경보 모듈로는 쓰지 않음)
    swp = np.full(len(T), np.nan)
    swp[on] = T_virtual[on - 1] - art["switch_T"][0]
    swp[off] = T_virtual[off - 1] - art["switch_T"][1]
    sig["switch_P"] = swp
    extra = dict(state=state, u=u, d=d, since=since, T_virtual=T_virtual, T_sim=T_sim)
    return sig, extra


POINTWISE = {"z_T", "z_P", "maha", "virtual_pt"}   # 한 시점 값으로 바로 판정
# 문 열림은 언제나 실제 온도를 예측보다 높인다. 그래서 '보고값 > 예측' 방향은 정상에서도 흔하다.
# 위험한 위조(과열 은폐)는 '보고값 < 예측' 방향이므로 열 모델 검사는 이 방향만 누적한다.
ONE_SIDED_LOW = {"thermal", "thermal_P"}
CUSUM_MODULES = {"cusum_T", "virtual", "thermal", "thermal_P", "switch", "switch_P"}

CONFIGS = {
    "1) 통계만": ["z_T", "z_P", "maha", "cusum_T"],
    "2) 통계 + 가상 온도 센서": ["z_T", "z_P", "maha", "cusum_T", "virtual", "virtual_pt"],
    "3) 통계 + 가상 센서 + 전력 기반 검사": ["z_T", "z_P", "maha", "cusum_T", "virtual", "virtual_pt", "thermal", "switch"],
}


def scores(sig, art):
    """원신호 → 모듈별 점수(클수록 이상). 기준값은 검증 정상 데이터에서 구한 것."""
    out = {}
    for name, x in sig.items():
        med, sc = art["scale"][name]
        if name in ONE_SIDED_LOW:
            # 정상 데이터가 아주 조용하면(문을 짧게 여는 현장) 표준화 단위가 0.1°C까지 작아진다.
            # 그러면 상한(Z_CLIP)에 막혀 누적 속도는 그대로인데 임계값만 커져 경보가 늦어진다.
            # 온도계 정확도 수준인 0.2°C보다 작은 단위는 쓰지 않는다.
            sc = max(sc, THERMAL_SCALE_FLOOR)
        z = (x - med) / sc
        if name in POINTWISE:
            out[name] = np.abs(np.nan_to_num(z, nan=0.0))
        else:
            if name in ONE_SIDED_LOW:
                # 열 모델은 문 열림·날씨 차이로 0.1~0.2°C 정도의 작은 치우침이 남는다.
                # 그만큼은 봐주도록 여유값을 2로 크게 둔다. 과열 은폐는 이보다 훨씬 크게 어긋난다.
                out[name] = cusum(np.clip(z, -Z_CLIP, Z_CLIP), k=2.0)[2]
            else:
                out[name] = cusum(np.clip(z, -Z_CLIP, Z_CLIP))[0]
    return out


def alarms(sc, art, modules):
    a = np.zeros(len(next(iter(sc.values()))), bool)
    for m in modules:
        a |= sc[m] > art["thr"][m]
    return a


def onset_estimate(sc, art, module, alarm_idx):
    """경보 시점에서 거꾸로 거슬러 CUSUM이 0이었던 마지막 시점 = 조작 시작 추정."""
    s = sc[module]
    i = alarm_idx
    while i > 0 and s[i - 1] > 0:
        i -= 1
    return i


def attribute(sig, art, t0, t1):
    """[t0, t1) 구간에서 어떤 센서가 거짓인지 판정한다.

    세 가지 이야기를 맞춰 본다.
      ① 보고 온도         ② 압력으로 추정한 온도         ③ 전력(독립 채널)이 말하는 온도
    ①≠②, ②=③  → 온도 센서만 거짓                      → 'T'
    ①≠②, ②≠③  → 온도와 압력 모두 거짓(서로도 안 맞음)  → 'T+P'
    ①=②, ①≠③  → 온도와 압력이 '일관되게' 함께 거짓      → 'T+P'
    그 밖                                                → '?' (순간 잡음 등)
    """
    def level(name):
        med, sc = art["scale"][name]
        z = (sig[name][t0:t1] - med) / sc
        z = z[~np.isnan(z)]
        return float(np.mean(np.abs(z))) if z.size else 0.0
    lim = art["attr_limit"]
    v = level("virtual") > lim                                   # ①≠②
    pw = max(level("thermal_P"), level("switch_P")) > lim        # ②≠③
    tw = max(level("thermal"), level("switch")) > lim            # ①≠③
    if v and not pw:
        return "T"
    if v and pw:
        return "T+P"
    if not v and tw:
        return "T+P"
    return "?"
