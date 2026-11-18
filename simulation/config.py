"""실험 전체에서 함께 쓰는 설정값.

가상 설비는 병원 혈액은행의 적혈구 보관 냉장고(보관 온도 1~6°C)를 본뜬 것이다.
모든 수치는 가정이며 특정 제품의 사양이 아니다.
발표 자료에 이 값들을 그대로 공개한다.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for _p in (DATA, MODELS, RESULTS):
    _p.mkdir(exist_ok=True)

DAY = 1440  # 하루 = 1440분 (샘플 간격 1분)


@dataclass(frozen=True)
class Plant:
    # 고내 온도 1차 열 모델: dT/dt = a(Ta - T) - b*u + h*d + 문 열림 열
    a: float = 1 / 300        # 단열 손실 계수 [1/분] (시정수 300분)
    b: float = 0.25           # 컴프레서 냉각 능력 [°C/분] (기준 주변온도 22°C)
    b_ta_slope: float = 0.015  # 주변온도 1°C 상승당 냉각 능력 감소 비율 (선형 모델이 놓치는 부분)
    ramp_min: float = 3.0     # 컴프레서 기동 후 정격 냉각까지 걸리는 시간 [분]
    h: float = 0.07           # 제상 히터 가열 [°C/분] (짧고 약하게: 제상 중에도 6°C를 넘지 않도록)
    door_heat: float = 0.30   # 문 열림 중 열 유입 [°C/분]
    process_noise: float = 0.01

    # 제어기
    setpoint: float = 4.0
    band: float = 0.5                 # 3.5~4.5°C 사이에서 켜고 끔 (혈액 냉장고는 폭을 좁게 둔다)
    defrost_every_min: int = 8 * 60   # 8시간마다
    defrost_offset_min: int = 120     # 02:00, 10:00, 18:00
    defrost_len_min: int = 10

    # 냉매(R-134a 근사)와 압력
    evap_dt: float = 12.0     # 가동 중 증발온도 = 고내온도 - 12°C
    p_tau: float = 3.0        # 증발기 온도 응답 시정수 [분]

    # 전력 [W]
    w_idle: float = 35.0       # 정지 중 (팬·제어기)
    w_comp: float = 300.0      # 컴프레서 가동 기본
    w_inrush: float = 120.0    # 기동 직후 2분 돌입 전력
    w_heater: float = 550.0    # 제상 히터

    # 센서 잡음
    noise_T: float = 0.05
    noise_P: float = 0.02
    noise_Ta: float = 0.10


PLANT = Plant()

# ---------------------------------------------------------------- 냉장고 여러 대
# 같은 기준(1~6°C)을 지키지만 단열·냉각 능력·히터·전력이 조금씩 다른 냉장고들.
# 0번은 기준 냉장고이고, 나머지는 기준값에서 무작위로 벗어나게 만든다.
N_FRIDGES = 8


def make_fridges(n=N_FRIDGES, seed=2026):
    import dataclasses
    import numpy as np
    rng = np.random.default_rng(seed)
    out = [PLANT]
    for _ in range(n - 1):
        f = lambda lo, hi: float(rng.uniform(lo, hi))
        out.append(dataclasses.replace(
            PLANT,
            a=PLANT.a * f(0.8, 1.2),              # 단열 ±20%
            b=PLANT.b * f(0.85, 1.15),            # 냉각 능력 ±15%
            h=PLANT.h * f(0.9, 1.1),              # 제상 히터 ±10%
            b_ta_slope=PLANT.b_ta_slope * f(0.7, 1.3),
            ramp_min=f(2.0, 4.0),
            door_heat=PLANT.door_heat * f(0.8, 1.2),
            evap_dt=f(10.0, 14.0),
            defrost_offset_min=int(rng.integers(0, 8 * 60)),
            w_idle=PLANT.w_idle * f(0.8, 1.2),
            w_comp=PLANT.w_comp * f(0.8, 1.2),
            w_heater=PLANT.w_heater * f(0.85, 1.15),
        ))
    return out


FRIDGES = make_fridges()

SAFE_LIMIT = 6.0    # 적혈구 보관 상한 [°C] (보관 기준 1~6°C)
SAFE_LOW = 1.0      # 적혈구 보관 하한 [°C] (이번 공격들은 모두 과열 방향이라 평가에는 상한만 쓴다)
WINDOW_H = 30       # 열 모델 개루프 검사 길이 [분]
CUSUM_K = 1.0       # CUSUM 허용 여유값 (표준화 단위)
Z_CLIP = 4.0        # 표준화 잔차 상한 (문 열림 같은 짧은 튐이 누적을 지배하지 않게)
THR_MARGIN = 1.1    # 임계값 = 검증 정상 데이터에서 나온 최댓값 × 1.1

SEED_TRAIN, SEED_VAL, SEED_CAL = 11, 22, 33
TRAIN_DAYS, VAL_DAYS, CAL_DAYS = 10, 30, 20
EPISODES_PER_ATTACK = 50   # 공격 유형마다 50회, 냉장고 8대에 돌아가며 배정
THERMAL_SCALE_FLOOR = 0.2  # 열 모델 잔차 표준화 단위 하한 [°C] (온도계 정확도 수준)
