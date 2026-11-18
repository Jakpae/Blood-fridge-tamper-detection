"""공격·오류 주입기.

모든 에피소드는 2일(2880분)이며, 공격은 둘째 날 끝 3~6시간에 시작해 에피소드 끝까지 이어진다.
'label' 열은 보고값이 위조된 지점을 1로 표시한다.
"""
import numpy as np

from config import DAY, PLANT
from sim import simulate, psat_r134a, tsat_r134a

N = 2 * DAY

# 이름: (설명, 실제 설비 조작, 위조 대상)
ATTACKS = {
    "offset":          ("온도 값을 1.5°C 낮게 조작", None, "T"),
    "spike":           ("온도 값을 순간적으로 튀게 조작", None, "T"),
    "drift":           ("온도 값을 서서히 낮게 조작", None, "T"),
    "multi":           ("온도·압력 값을 함께 낮게 조작", None, "TP"),
    "defrost_freeze":  ("히터 강제 가동 + 화면 숫자 멈추기", "defrost", "TP"),
    "defrost_replay":  ("히터 강제 가동 + 어제 기록 틀어 주기", "defrost", "TP"),
    "compoff_replay":  ("냉각 장치 끄기 + 어제 기록 틀어 주기", "comp_off", "TP"),
    "missing":         ("온도 기록 끊김 (30~120분)", None, "-"),
}
DETECTION_ATTACKS = [k for k in ATTACKS if k != "missing"]


def make_episode(kind, seed, plant=PLANT):
    rng = np.random.default_rng(seed)
    dur = int(rng.integers(180, 361))
    start = N - dur
    _, actuation, _ = ATTACKS[kind]
    force = {"start": start, "mode": actuation} if actuation else None
    df = simulate(N, seed + 100_000, force=force, start_min=int(rng.integers(0, DAY)), p=plant)
    df["label"] = 0
    idx = slice(start, N)
    T = df["T_rep"].to_numpy().copy()
    P = df["P_rep"].to_numpy().copy()
    k = np.arange(N - start)

    if kind == "offset":
        T[idx] -= 1.5
        df.loc[start:, "label"] = 1
    elif kind == "spike":
        hit = start + np.flatnonzero(rng.random(N - start) < 0.03)
        T[hit] += rng.choice([-4.0, 4.0], hit.size)
        df.loc[hit, "label"] = 1
    elif kind == "drift":
        T[idx] -= 0.01 * k
        df.loc[start:, "label"] = 1
    elif kind == "multi":
        T[idx] -= 1.5
        P[idx] = psat_r134a(tsat_r134a(P[idx]) - 1.5)  # 압력도 온도와 맞게 일관 변조
        df.loc[start:, "label"] = 1
    elif kind == "defrost_freeze":
        T[idx] = T[start - 1] + rng.normal(0, 0.05, N - start)
        P[idx] = P[start - 1] + rng.normal(0, 0.02, N - start)
        df.loc[start:, "label"] = 1
    elif kind in ("defrost_replay", "compoff_replay"):
        T[idx] = df["T_rep"].to_numpy()[start - DAY:N - DAY]
        P[idx] = df["P_rep"].to_numpy()[start - DAY:N - DAY]
        df.loc[start:, "label"] = 1
    elif kind == "missing":
        g0 = int(rng.integers(DAY + 200, N - 200))
        glen = int(rng.integers(30, 121))
        T[g0:g0 + glen] = np.nan
        df.loc[g0:g0 + glen - 1, "label"] = 1   # 여기서 label은 '누락' 표시
        start = g0

    df["T_rep"], df["P_rep"] = T, P
    df["attack"] = kind
    df["onset"] = start
    return df
