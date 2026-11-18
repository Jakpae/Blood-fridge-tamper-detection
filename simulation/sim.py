"""물리 기반 가상 냉장 보관실 시뮬레이터.

출력 열
  T_true : 실제 고내 온도 (정답, 평가용)
  T_rep  : 제어기가 보고하는 온도 (제어망, 위조 대상)
  P_rep  : 제어기가 보고하는 흡입 압력 [bar] (제어망, 위조 대상)
  W      : 독립 계측 상자가 잰 소비 전력 [W] (제어망과 분리)
  Ta     : 독립 계측 상자가 잰 주변 온도 [°C]
  u, d, door : 컴프레서/제상/문 열림의 실제 상태 (정답, 평가용)
"""
import numpy as np
import pandas as pd

from config import PLANT, DAY


def psat_r134a(t_c):
    """R-134a 포화압력 근사 [bar abs]. -10°C≈2.0, 0°C≈2.9, 10°C≈4.1 bar에 맞춘 2점 근사식."""
    return np.exp(10.975 - 2705.0 / (np.asarray(t_c) + 273.15))


def tsat_r134a(p_bar):
    """psat_r134a의 역함수: 압력 → 포화온도 [°C]."""
    return 2705.0 / (10.975 - np.log(np.asarray(p_bar))) - 273.15


import os
DOOR_MODE = os.environ.get("DOOR_MODE", "real")


def _ambient(n, rng, start_min):
    t_h = (np.arange(n) + start_min) / 60.0
    walk = np.cumsum(rng.normal(0, 0.02, n))
    walk -= np.linspace(0, walk[-1], n)
    return 22 + 3 * np.sin(2 * np.pi * (t_h - 9) / 24) + walk


def _doors(n, rng, start_min):
    """분마다 문이 열려 있던 비율(0~1)을 돌려준다.

    DOOR_MODE=real (기본): 현장 자문값. 주간(08~20시) 시간당 3~4회, 야간 1~2회,
        한 번 열면 10초 이내(3~10초).
    DOOR_MODE=stress: 이전 가혹 조건. 주간 시간당 2회, 야간 0.2회, 한 번에 1~3분.
    """
    hour = ((np.arange(n) + start_min) / 60.0) % 24
    day = (hour >= 8) & (hour < 20)
    door = np.zeros(n)
    if DOOR_MODE == "stress":
        rate = np.where(day, 2 / 60, 0.2 / 60)
        for i in np.flatnonzero(rng.random(n) < rate):
            door[i:i + rng.integers(1, 4)] = 1.0
    else:
        rate = np.where(day, 3.5 / 60, 1.5 / 60)
        for i in np.flatnonzero(rng.random(n) < rate):
            door[i] = min(1.0, door[i] + rng.uniform(3, 10) / 60)
    return door


def simulate(n_min, seed, force=None, start_min=0, p=PLANT):
    """n_min분 동안 시뮬레이션한다.

    force: {'start': int, 'mode': 'defrost' | 'comp_off'}
           start 이후 공격자가 설정을 바꿔 제상 고정 또는 컴프레서 정지 상태로 만든다.
    """
    rng = np.random.default_rng(seed)
    Ta = _ambient(n_min, rng, start_min)
    door = _doors(n_min, rng, start_min)

    T = p.setpoint + rng.normal(0, 0.3)
    Te = T
    u_prev, since_on = 0, 99
    out = np.zeros((n_min, 7))
    for i in range(n_min):
        clock = i + start_min
        T_ctrl = T + rng.normal(0, p.noise_T)  # 제어기가 보는 자기 센서 값
        defrost = ((clock - p.defrost_offset_min) % p.defrost_every_min) < p.defrost_len_min
        if defrost:
            u, d = 0, 1
        else:
            d = 0
            if T_ctrl > p.setpoint + p.band:
                u = 1
            elif T_ctrl < p.setpoint - p.band:
                u = 0
            else:
                u = u_prev
        if force is not None and i >= force["start"]:
            if force["mode"] == "defrost":
                u, d = 0, 1
            elif force["mode"] == "comp_off":
                u, d = 0, 0
        since_on = since_on + 1 if (u and u_prev) else (0 if u else 99)

        ramp = min(1.0, (since_on + 1) / p.ramp_min) if u else 0.0
        b_eff = p.b * (1 - p.b_ta_slope * (Ta[i] - 22)) * ramp
        dT = p.a * (Ta[i] - T) - b_eff + p.h * d + p.door_heat * door[i]
        T = T + dT + rng.normal(0, p.process_noise)

        target = T - p.evap_dt if u else (T + 3 if d else T)
        Te += (target - Te) / p.p_tau
        P = psat_r134a(Te)

        if u:
            W = p.w_comp + 3 * (Ta[i] - 20) + 40 * (P - 2.0) + (p.w_inrush if since_on < 2 else 0)
        else:
            W = p.w_idle
        if d:
            W += p.w_heater
        W += rng.normal(0, 6 if u else 3)

        out[i] = [T, P, W, u, d, door[i], 0]
        u_prev = u

    df = pd.DataFrame({
        "minute": np.arange(n_min) + start_min,
        "T_true": out[:, 0],
        "T_rep": out[:, 0] + rng.normal(0, p.noise_T, n_min),
        "P_rep": out[:, 1] + rng.normal(0, p.noise_P, n_min),
        "W": out[:, 2],
        "Ta": Ta + rng.normal(0, p.noise_Ta, n_min),
        "u": out[:, 3].astype(int),
        "d": out[:, 4].astype(int),
        "door": out[:, 5].round(3),
    })
    return df


def normal_days(days, seed):
    """공격이 없는 정상 운전 데이터."""
    return simulate(days * DAY, seed, start_min=0)
