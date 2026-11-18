"""2단계: 냉장고마다 모델 학습, 임계값·신뢰 구간 보정.

냉장고마다 자기 정상 데이터로 자기 모델을 따로 만든다.
(현장에 설치할 때 며칠 동안 정상 운전을 보고 그 냉장고의 물리값을 배우는 방식)

train : 전력 상태 모델, 열 모델, 가상 온도 센서, 통계 기준선 학습 (10일 연속 운전)
val   : 모듈별 표준화 기준과 경보 임계값 (2일 × 15회, 서로 다른 날씨)
cal   : 복원값 신뢰 구간 (2일 × 10회)
결과  : models/artifacts.joblib (냉장고 번호 → 학습 결과), results/fridge_params.csv
"""
import time

import joblib
import numpy as np
import pandas as pd

from config import DATA, MODELS, RESULTS, THR_MARGIN, WINDOW_H
from models import PowerState, ThermalModel, VirtualTemp, state_features
from detect import raw_signals, robust_scale, scores, switch_points

HORIZONS = [10, 30, 60, 120, 180, 240, 300, 360]


def rmse(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def runs(df):
    return [g.reset_index(drop=True) for _, g in df.groupby("run")]


def train_one(tr, va_runs, ca_runs):
    """냉장고 한 대의 모델을 학습하고 보정한다. (학습 결과 dict, 요약 dict)"""
    art, info = {}, {}

    # ---- 1) 전력 → 운전 상태 (라벨 없이 학습)
    art["power"] = PowerState().fit(tr["W"])
    acc = [np.mean(art["power"].predict(v["W"]) == np.where(v["d"] == 1, 2, v["u"])) for v in va_runs]
    info["power_acc"] = float(np.mean(acc))

    st_tr = art["power"].predict(tr["W"])
    u, d, since = state_features(st_tr)
    T, Ta = tr["T_rep"].to_numpy(), tr["Ta"].to_numpy()

    # ---- 2) 열 모델 (물리층만 vs 물리층+보정층) + 파라미터 부트스트랩
    plain = ThermalModel(use_correction=False).fit(T, Ta, u, d, since)
    art["thermal"] = ThermalModel(use_correction=True).fit(T, Ta, u, d, since).fit_bootstrap(T, Ta, u, d)
    th = art["thermal"]
    info.update(a=th.a, b=th.b, h=th.h)
    for name, m in (("rmse30_phys", plain), ("rmse30_hybrid", th)):
        errs = []
        for v in va_runs:
            uv, dv, snv = state_features(art["power"].predict(v["W"]))
            ol = m.open_loop(v["T_rep"].to_numpy(), v["Ta"].to_numpy(), uv, dv, snv, WINDOW_H)
            errs.append(rmse(ol, v["T_rep"].to_numpy()))
        info[name] = float(np.mean(errs))

    # ---- 3) 가상 온도 센서 (압력 + 운전 상태 → 온도)
    art["virtual"] = VirtualTemp().fit(tr["P_rep"].to_numpy(), u, d, since, T)

    # ---- 3-1) 제어 논리: 전력으로 잡은 켜짐/꺼짐 직전의 보고 온도 평균
    on, off = switch_points(st_tr)
    art["switch_T"] = (float(np.mean(T[on - 1])), float(np.mean(T[off - 1])))

    # ---- 4) 통계 기준선
    TP = tr[["T_rep", "P_rep"]].to_numpy()
    art["mu_TP"] = TP.mean(0)
    art["icov_TP"] = np.linalg.inv(np.cov(TP.T) + 1e-6 * np.eye(2))

    # ---- 5) 표준화 기준과 임계값 (검증 정상 데이터, 여러 날씨)
    sigs = [raw_signals(v, art) for v in va_runs]
    pooled = {k: np.concatenate([s[k][WINDOW_H:] for s, _ in sigs]) for k in sigs[0][0]}
    art["scale"] = {k: robust_scale(v) for k, v in pooled.items()}
    mx = {}
    for s, _ in sigs:
        for k, v in scores(s, art).items():
            mx[k] = max(mx.get(k, 0.0), float(np.max(v[WINDOW_H:])))
    art["thr"] = {k: v * THR_MARGIN for k, v in mx.items()}
    art["attr_limit"] = 2.0
    info["virtual_mae"] = float(np.mean([np.mean(np.abs(e["T_virtual"] - v["T_rep"].to_numpy()))
                                         for (_, e), v in zip(sigs, va_runs)]))

    # ---- 6) 복원 신뢰 구간 (보정용 정상 데이터)
    err_v, err_h = [], {H: [] for H in HORIZONS}
    for c in ca_runs:
        uc, dc, snc = state_features(art["power"].predict(c["W"]))
        Tc, Tac, Pc = c["T_rep"].to_numpy(), c["Ta"].to_numpy(), c["P_rep"].to_numpy()
        err_v.append(np.abs(art["virtual"].predict(Pc, uc, dc, snc) - Tc))
        for H in HORIZONS:
            e = np.abs(th.open_loop(Tc, Tac, uc, dc, snc, H) - Tc)
            err_h[H].append(e[~np.isnan(e)])
    art["q_virtual"] = float(np.quantile(np.concatenate(err_v), 0.95))
    qh = [float(np.quantile(np.concatenate(err_h[H]), 0.95)) for H in HORIZONS]
    art["q_horizon"] = (np.array([0] + HORIZONS, float), np.array([0.1] + qh))
    info["q_virtual"] = art["q_virtual"]
    info["q_60"], info["q_360"] = qh[2], qh[-1]
    return art, info


def main():
    t0 = time.time()
    tr_all = pd.read_csv(DATA / "train.csv")
    va_all = pd.read_csv(DATA / "val.csv")
    ca_all = pd.read_csv(DATA / "cal.csv")
    truth = pd.read_csv(DATA / "fridges.csv").set_index("fridge")

    arts, rows = {}, []
    for f in sorted(tr_all["fridge"].unique()):
        t1 = time.time()
        tr = tr_all[tr_all.fridge == f].reset_index(drop=True)
        art, info = train_one(tr, runs(va_all[va_all.fridge == f]), runs(ca_all[ca_all.fridge == f]))
        arts[int(f)] = art
        tp = truth.loc[f]
        rows.append({"fridge": int(f), **info,
                     "a_true": tp.a, "b_true": tp.b, "h_true": tp.h})
        print(f"[냉장고 {f}] 전력 상태 정확도 {info['power_acc']:.4f} | "
              f"단열 시정수 {1 / info['a']:.0f}분 (정답 {1 / tp.a:.0f}) | "
              f"냉각 {info['b']:.3f} (정답 {tp.b:.3f}) | 히터 {info['h']:.3f} (정답 {tp.h:.3f}) | "
              f"30분 굴림 RMSE {info['rmse30_phys']:.2f}→{info['rmse30_hybrid']:.2f} °C | "
              f"가상 센서 MAE {info['virtual_mae']:.3f} °C ({time.time() - t1:.0f}초)")

    joblib.dump(arts, MODELS / "artifacts.joblib")
    pd.DataFrame(rows).to_csv(RESULTS / "fridge_params.csv", index=False, encoding="utf-8-sig")
    print(f"저장: models/artifacts.joblib, results/fridge_params.csv ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
