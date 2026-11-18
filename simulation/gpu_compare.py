"""선택 과제 (GPU): 순수 딥러닝(GRU) 온도 예측 모델 vs 냉장고별 물리 기반 열 모델.

질문 1: 냉장고 8대의 정상 데이터를 모두 모아 학습한 '범용' 딥러닝 모델이,
        냉장고마다 따로 배운 물리 모델(파라미터 3개)보다 정확한가?
질문 2: 정상 데이터만 본 딥러닝 모델이 학습에 없던 과열(제상 고정, 컴프레서 정지)을 따라가는가?

입력은 열 모델과 똑같이 [고내 온도, 전력, 주변 온도]만 쓴다 (압력은 위조될 수 있으므로 제외).
step1, step2를 먼저 실행해야 한다.

실행: python gpu_compare.py
"""
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import DATA, MODELS, RESULTS, WINDOW_H
from models import state_features

# ---- 여기 값을 키우면 GPU를 더 많이 쓴다
L = 30           # 입력 길이 (분)
HIDDEN = 64      # GRU 은닉 크기 (256으로 키워 볼 것)
EPOCHS = 30      # 학습 반복
BATCH = 512
DEV = "cuda" if torch.cuda.is_available() else "cpu"
COLS = ["T_rep", "W", "Ta"]


class GRUNet(nn.Module):
    def __init__(self, n_in=3, hidden=HIDDEN):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):              # x: (배치, L, 3)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # 다음 분 온도 변화량(표준화)


def make_xy(df, mu, sd):
    """한 냉장고의 연속 데이터에서 (L분 입력, 다음 분 변화량) 쌍을 만든다."""
    X = (df[COLS].to_numpy(np.float32) - mu) / sd
    T = df["T_rep"].to_numpy(np.float32)
    dT = np.diff(T)
    idx = np.arange(L - 1, len(T) - 1)
    xs = np.stack([X[i - L + 1:i + 1] for i in idx])
    return xs, dT[idx]


def rollout(model, df, starts, H, mu, sd, dT_sd):
    """여러 시작점에서 동시에 H분 굴린다. 온도는 모델 자신의 예측을 이어 붙이고, 전력·주변온도는 실측을 쓴다."""
    T = df["T_rep"].to_numpy(np.float32)
    WTa = df[["W", "Ta"]].to_numpy(np.float32)
    hist_T = np.stack([T[s - L:s] for s in starts])            # (B, L)
    hist_WTa = np.stack([WTa[s - L:s] for s in starts])        # (B, L, 2)
    out = np.empty((len(starts), H), np.float32)
    model.eval()
    with torch.no_grad():
        for k in range(H):
            x = (np.concatenate([hist_T[..., None], hist_WTa], axis=2) - mu) / sd
            dt = model(torch.from_numpy(x.astype(np.float32)).to(DEV)).cpu().numpy() * dT_sd
            nxt = hist_T[:, -1] + dt
            out[:, k] = nxt
            new_WTa = np.stack([WTa[s + k] for s in starts])
            hist_T = np.concatenate([hist_T[:, 1:], nxt[:, None]], axis=1)
            hist_WTa = np.concatenate([hist_WTa[:, 1:], new_WTa[:, None, :]], axis=1)
    return out


def main():
    print("장치:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "(GPU 없음, CPU로 실행)")
    t0 = time.time()
    tr = pd.read_csv(DATA / "train.csv")
    mu = tr[COLS].mean().to_numpy(np.float32)
    sd = tr[COLS].std().to_numpy(np.float32)
    parts = [make_xy(g.reset_index(drop=True), mu, sd) for _, g in tr.groupby("fridge")]  # 냉장고 경계를 넘지 않게
    xs = np.concatenate([p[0] for p in parts])
    ys = np.concatenate([p[1] for p in parts])
    dT_sd = float(ys.std())
    X = torch.from_numpy(xs).to(DEV)
    Y = torch.from_numpy((ys / dT_sd).astype(np.float32)).to(DEV)
    print(f"학습 샘플 {len(X):,}개 (냉장고 {tr.fridge.nunique()}대)")

    torch.manual_seed(0)
    model = GRUNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n = len(X)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEV)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(X[b]), Y[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3d}  loss {tot / n:.4f}")
    print(f"학습 {time.time() - t0:.0f}초")

    arts = joblib.load(MODELS / "artifacts.joblib")

    # (1) 정상 운전: 냉장고별 30분 굴림 오차
    va = pd.read_csv(DATA / "val.csv")
    rows = []
    for (f, _), v in va.groupby(["fridge", "run"]):
        v = v.reset_index(drop=True)
        th, ps = arts[int(f)]["thermal"], arts[int(f)]["power"]
        starts = np.arange(L, len(v) - WINDOW_H, 10)
        pred = rollout(model, v, starts, WINDOW_H, mu, sd, dT_sd)[:, -1]
        truth = v["T_rep"].to_numpy()[starts + WINDOW_H - 1]
        u, d, s = state_features(ps.predict(v["W"]))
        ol = th.open_loop(v["T_rep"].to_numpy(), v["Ta"].to_numpy(), u, d, s, WINDOW_H)
        rows.append({"fridge": int(f), "gru_se": np.nanmean((pred - truth) ** 2),
                     "th_se": np.nanmean((ol[starts + WINDOW_H - 1] - truth) ** 2)})
    nr = pd.DataFrame(rows).groupby("fridge").mean() ** 0.5
    nr.columns = ["GRU_RMSE", "열모델_RMSE"]
    print(f"[정상 운전 {WINDOW_H}분 굴림 RMSE, °C]")
    print(nr.round(3))
    nr.to_csv(RESULTS / "gpu_normal_rmse.csv", encoding="utf-8-sig")

    # (2) 학습에 없던 과열: 공격 시작부터 끝까지 굴림
    eps = pd.read_csv(DATA / "episodes.csv")
    rows = []
    for ep_id, df in eps[eps.attack.isin(["defrost_freeze", "compoff_replay"])].groupby("episode"):
        df = df.reset_index(drop=True)
        f = int(df.fridge.iat[0])
        th, ps = arts[f]["thermal"], arts[f]["power"]
        on = int(df.onset.iat[0])
        g = rollout(model, df, np.array([on]), len(df) - on, mu, sd, dT_sd)[0]
        u, d, s = state_features(ps.predict(df["W"]))
        p = th.rollout(df.T_rep.iat[on - 1], df.Ta.to_numpy()[on:], u[on:], d[on:], s[on:])  # GRU와 같은 출발점
        truth = df.T_true.to_numpy()[on:]
        rows.append({"episode": ep_id, "fridge": f, "attack": df.attack.iat[0],
                     "GRU_MAE": float(np.mean(np.abs(g - truth))), "열모델_MAE": float(np.mean(np.abs(p - truth))),
                     "_g": g, "_p": p, "_t": truth})
    res = pd.DataFrame(rows)
    print("[학습에 없던 과열, 공격 구간 전체 굴림 MAE (°C)]")
    print(res.groupby("attack")[["GRU_MAE", "열모델_MAE"]].mean().round(2))
    res.drop(columns=["_g", "_p", "_t"]).to_csv(RESULTS / "gpu_gru_vs_thermal.csv", index=False, encoding="utf-8-sig")

    for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"):
        if any(name == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    r = res[res.attack == "defrost_freeze"].iloc[0]
    t = np.arange(len(r["_t"])) / 60
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.plot(t, r["_t"], color="#2563eb", lw=1.8, label="실제 온도")
    ax.plot(t, r["_p"], color="#059669", lw=1.8, ls="--", label="물리식 모델 (냉장고별)")
    ax.plot(t, r["_g"], color="#dc2626", lw=1.4, ls=":", label="딥러닝 GRU (정상 데이터만 학습)")
    ax.set_xlabel("히터 강제 가동 공격 시작 후 시간 (시간)")
    ax.set_ylabel("온도 (°C)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color="#e5e7eb")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_gpu_gru_vs_physics.png")
    print("저장: results/fig_gpu_gru_vs_physics.png, gpu_gru_vs_thermal.csv, gpu_normal_rmse.csv")


if __name__ == "__main__":
    main()
