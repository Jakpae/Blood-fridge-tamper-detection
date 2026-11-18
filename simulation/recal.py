"""학습된 모델은 그대로 두고 표준화 기준과 임계값만 다시 계산한다 (step2의 5번 단계)."""
import joblib, numpy as np, pandas as pd
from config import DATA, MODELS, THR_MARGIN, WINDOW_H
from detect import raw_signals, robust_scale, scores
A = joblib.load(MODELS / "artifacts.joblib")
va_all = pd.read_csv(DATA / "val.csv")
for f, art in A.items():
    va = [g.reset_index(drop=True) for _, g in va_all[va_all.fridge == f].groupby("run")]
    sigs = [raw_signals(v, art) for v in va]
    pooled = {k: np.concatenate([s[k][WINDOW_H:] for s, _ in sigs]) for k in sigs[0][0]}
    art["scale"] = {k: robust_scale(v) for k, v in pooled.items()}
    mx = {}
    for s, _ in sigs:
        for k, v in scores(s, art).items():
            mx[k] = max(mx.get(k, 0.0), float(np.max(v[WINDOW_H:])))
    art["thr"] = {k: v * THR_MARGIN for k, v in mx.items()}
    print(f, round(art["scale"]["thermal"][1], 3), round(art["thr"]["thermal"], 1))
joblib.dump(A, MODELS / "artifacts.joblib")
