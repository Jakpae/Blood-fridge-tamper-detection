"""1단계: 가상 데이터 만들기 (냉장고 8대).

data/fridges.csv  : 냉장고별 실제 물리값 (정답, 평가용)
data/train.csv    : 냉장고별 10일 연속 정상 운전 (학습)
data/val.csv      : 냉장고별 2일 × 15회 정상 운전 (경보 임계값)
data/cal.csv      : 냉장고별 2일 × 10회 정상 운전 (복원 신뢰 구간)
data/episodes.csv : 공격 유형별 50회, 각 2일. 냉장고 8대에 돌아가며 배정
"""
import dataclasses
import time

import pandas as pd

from config import (DATA, DAY, SEED_TRAIN, SEED_VAL, SEED_CAL, TRAIN_DAYS, VAL_DAYS, CAL_DAYS,
                    EPISODES_PER_ATTACK, FRIDGES)
from sim import simulate
from attacks import ATTACKS, make_episode


def main():
    t0 = time.time()
    pd.DataFrame([{"fridge": f, **dataclasses.asdict(p)} for f, p in enumerate(FRIDGES)]).to_csv(
        DATA / "fridges.csv", index=False)

    trains, vals, cals = [], [], []
    for f, plant in enumerate(FRIDGES):
        df = simulate(TRAIN_DAYS * DAY, SEED_TRAIN + 1000 * f, p=plant)
        df["fridge"], df["run"] = f, 0
        trains.append(df)
        # 검증·보정용은 날씨와 문 열림이 서로 다른 2일짜리 운전을 여러 개 모은다.
        # 한 번의 긴 운전으로 임계값을 정하면 다른 날씨에서 오경보가 늘어나기 때문이다.
        for name, days, seed, bucket in (("val", VAL_DAYS, SEED_VAL, vals), ("cal", CAL_DAYS, SEED_CAL, cals)):
            for r in range(days // 2):
                x = simulate(2 * DAY, seed * 1000 + 100 * f + r, start_min=(r * 377) % DAY, p=plant)
                x["fridge"], x["run"] = f, r
                bucket.append(x)
        print(f"냉장고 {f}: 학습 {TRAIN_DAYS}일, 검증 2일×{VAL_DAYS // 2}, 보정 2일×{CAL_DAYS // 2}")
    pd.concat(trains).to_csv(DATA / "train.csv", index=False)
    pd.concat(vals).to_csv(DATA / "val.csv", index=False)
    pd.concat(cals).to_csv(DATA / "cal.csv", index=False)

    eps = []
    for ai, kind in enumerate(ATTACKS):
        for j in range(EPISODES_PER_ATTACK):
            f = j % len(FRIDGES)
            ep = make_episode(kind, seed=1000 * (ai + 1) + j, plant=FRIDGES[f])
            ep.insert(0, "episode", f"{kind}_{j:02d}")
            ep.insert(1, "fridge", f)
            eps.append(ep)
        print(f"공격 {kind:15s}: {EPISODES_PER_ATTACK}회")
    pd.concat(eps).to_csv(DATA / "episodes.csv", index=False)
    print(f"완료 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
