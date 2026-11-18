"""선택 과제 ② (GPU): 물리식 + 신경망 보정, 굴림 오차로 학습.

앞의 gpu_compare.py에서 본 것:
  - GRU(순수 딥러닝)는 평소엔 물리 모델보다 정확하지만, 학습에 없던 과열에서는 6°C 근처에 멈춰 버린다.
  - 물리 모델은 과열을 따라가지만 평소 정확도가 조금 떨어진다.
이 스크립트의 질문: 둘의 장점을 합칠 수 있는가?

모델: 다음 분 온도 = 현재 온도 + a(Ta-T) - b*u + h*d        ← 물리층 (냉장고마다 a, b, h)
                    + 신경망 보정 × [학습 때 본 범위 안일 때만]  ← AI층 (8대가 함께 쓰는 신경망 1개)

학습은 두 단계다.
  1단계: 물리층(a, b, h)만, '60분 굴렸을 때의 오차'로 다시 맞춘다. (step2는 1분 앞 오차로 맞췄다)
  2단계: 물리층을 얼린 뒤 신경망만 학습한다. 물리층이 설명 못 한 나머지만 AI가 배우게 하려는 것이다.
신경망 보정은 크기를 ±CORR_MAX °C/분으로 묶고, 학습 범위 밖(오래 이어진 제상·정지, 과열)에서는 끈다.
그래서 처음 보는 상황에서는 자동으로 물리식만 남는다.

입력은 [고내 온도, 전력에서 읽은 운전 상태, 주변 온도]뿐이다 (압력은 위조될 수 있으므로 쓰지 않는다).
step1, step2를 먼저 실행해야 한다. gpu_compare.py를 먼저 돌려 두면 GRU 결과도 함께 비교한다.

실행: python gpu_hybrid.py
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
H_TRAIN = 60        # 학습 때 한 번에 굴리는 길이 (분)
HIDDEN = 64         # 신경망 크기 (128~256으로 키워 볼 것)
PHYS_EPOCHS = 0     # 1단계 반복 (0이면 step2의 물리값을 그대로 쓴다. 10으로 두면 물리값이 무너지는 것을 볼 수 있다)
NN_EPOCHS = 30      # 2단계 반복
STEPS = 200         # 반복 한 번당 미니배치 수
BATCH = 1024        # 미니배치 하나에 들어가는 60분 구간 수
CORR_MAX = 0.05     # 신경망 보정의 최대 크기 (°C/분)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
F32 = np.float32
ATTACKS = ["defrost_freeze", "defrost_replay", "compoff_replay"]


# ---------------------------------------------------------------- 데이터 준비
def prep(df, art):
    """보고 온도, 주변 온도, 그리고 '전력이 말해 주는 운전 상태'(u, d, 지속시간)를 꺼낸다."""
    u, d, s = state_features(art["power"].predict(df["W"]))
    return {"T": df["T_rep"].to_numpy(F32), "Ta": df["Ta"].to_numpy(F32),
            "u": u.astype(F32), "d": d.astype(F32), "s": s.astype(F32)}


def to_dev(x):
    return torch.as_tensor(np.asarray(x), device=DEV)


# ---------------------------------------------------------------- 모델
class Hybrid(nn.Module):
    def __init__(self, arts, fridges):
        super().__init__()
        th = [arts[f]["thermal"] for f in fridges]
        abh = np.array([[t.a, t.b, t.h] for t in th], F32)
        self.log_abh = nn.Parameter(torch.log(torch.as_tensor(abh)))       # 양수가 되도록 로그로 둔다
        # 보정을 믿어도 되는 범위 (step2의 열 모델과 같은 기준)
        self.register_buffer("dom", torch.as_tensor(np.array([t.dom for t in th], F32)))
        self.register_buffer("smax", torch.as_tensor(
            np.array([[t.since_max[0], t.since_max[1], t.since_max[2]] for t in th], F32)))
        self.emb = nn.Embedding(len(fridges), 8)                            # 냉장고마다 다른 특성을 담는 칸
        self.net = nn.Sequential(nn.Linear(5 + 8, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, 1))
        nn.init.zeros_(self.net[-1].weight)                                 # 처음엔 보정 0 = 물리식 그대로
        nn.init.zeros_(self.net[-1].bias)

    def abh(self):
        return torch.exp(self.log_abh)

    def step(self, T, Ta, u, d, s, f, use_corr=True):
        a, b, h = torch.exp(self.log_abh[f]).unbind(-1)
        nxt = T + a * (Ta - T) - b * u + h * d
        if not use_corr:
            return nxt
        lo, hi = self.dom[f, 0], self.dom[f, 1]
        st = (u > 0.5).long() + 2 * (d > 0.5).long()
        smax = self.smax[f].gather(1, st[:, None]).squeeze(1)
        gap = Ta - T
        inside = (gap >= lo) & (gap <= hi) & (s <= smax)
        x = torch.stack([u, d,
                         torch.minimum(s, smax) / (smax + 1.0),
                         (gap.clamp(lo, hi) - (lo + hi) / 2) / ((hi - lo) / 2 + 1e-6),
                         (Ta - 20.0) / 5.0], dim=-1)
        x = torch.cat([x, self.emb(f)], dim=-1)
        corr = CORR_MAX * torch.tanh(self.net(x).squeeze(-1))
        return nxt + corr * inside.float()

    def roll(self, T0, Ta, u, d, s, f, use_corr=True):
        """T0: (B,), 나머지: (B, H). 각 분의 입력으로 한 걸음씩 굴린 H개의 예측 (B, H)를 낸다."""
        T, outs = T0, []
        for k in range(Ta.shape[1]):
            T = self.step(T, Ta[:, k], u[:, k], d[:, k], s[:, k], f, use_corr)
            outs.append(T)
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------- 학습
def train(model, D, n_f, epochs, params, lr, use_corr, label):
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=0.3)      # 문 열림 같은 튀는 값에 덜 끌려가도록
    N = D["T"].shape[1]
    ar = torch.arange(H_TRAIN + 1, device=DEV)
    for ep in range(epochs):
        tot = 0.0
        for _ in range(STEPS):
            f = torch.randint(0, n_f, (BATCH,), device=DEV)
            st = torch.randint(0, N - H_TRAIN - 1, (BATCH,), device=DEV)
            idx = st[:, None] + ar[None, :]
            g = {k: v[f[:, None], idx] for k, v in D.items()}
            pred = model.roll(g["T"][:, 0], g["Ta"][:, :-1], g["u"][:, :-1], g["d"][:, :-1],
                              g["s"][:, :-1], f, use_corr)
            loss = loss_fn(pred, g["T"][:, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep % 5 == 0 or ep == epochs - 1:
            a, b, h = model.abh()[0].tolist()
            print(f"  [{label}] epoch {ep:3d}  loss {tot / STEPS:.5f}   냉장고0: 시정수 {1 / a:.0f}분, 냉각 {b:.3f}, 히터 {h:.3f}")


@torch.no_grad()
def batch_roll(model, segs, use_corr=True):
    """길이가 다른 구간들을 한꺼번에 굴린다. segs: [(fridge_idx, T0, prep 딕셔너리 조각)]"""
    Hm = max(len(p["Ta"]) for _, _, p in segs)
    pad = lambda k: np.stack([np.pad(p[k], (0, Hm - len(p[k])), mode="edge") for _, _, p in segs])
    out = model.roll(to_dev(np.array([t0 for _, t0, _ in segs], F32)),
                     to_dev(pad("Ta")), to_dev(pad("u")), to_dev(pad("d")), to_dev(pad("s")),
                     to_dev(np.array([f for f, _, _ in segs])), use_corr).cpu().numpy()
    return [out[i, :len(p["Ta"])] for i, (_, _, p) in enumerate(segs)]


def cut(p, i, j):
    return {k: v[i:j] for k, v in p.items()}


# ---------------------------------------------------------------- 실행
def main():
    print("장치:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "(GPU 없음, CPU로 실행)")
    t0 = time.time()
    arts = joblib.load(MODELS / "artifacts.joblib")
    fridges = sorted(arts)
    fi = {f: i for i, f in enumerate(fridges)}
    tr = pd.read_csv(DATA / "train.csv")
    preps = [prep(tr[tr.fridge == f].reset_index(drop=True), arts[f]) for f in fridges]
    N = min(len(p["T"]) for p in preps)
    D = {k: to_dev(np.stack([p[k][:N] for p in preps])) for k in preps[0]}      # (냉장고 수, 분)
    print(f"학습 데이터: 냉장고 {len(fridges)}대 × {N:,}분")

    torch.manual_seed(0)
    model = Hybrid(arts, fridges).to(DEV)
    print("1단계: 물리층(a, b, h)을 60분 굴림 오차로 다시 맞춘다")
    train(model, D, len(fridges), PHYS_EPOCHS, [model.log_abh], 1e-2, False, "물리층")
    model.log_abh.requires_grad_(False)
    print("2단계: 물리층을 얼리고 신경망 보정만 학습한다")
    train(model, D, len(fridges), NN_EPOCHS, list(model.emb.parameters()) + list(model.net.parameters()),
          2e-3, True, "신경망")
    model.eval()
    print(f"학습 {time.time() - t0:.0f}초")
    torch.save(model.state_dict(), MODELS / "hybrid.pt")

    # (0) 배운 물리값: step2(1분 앞 오차) vs 이번 1단계(60분 굴림 오차) vs 정답
    true = pd.read_csv(DATA / "fridges.csv").set_index("fridge")
    abh = model.abh().cpu().numpy()
    pr = pd.DataFrame([{"fridge": f,
                        "시정수_정답": 1 / true.loc[f, "a"], "시정수_step2": 1 / arts[f]["thermal"].a, "시정수_이번": 1 / abh[i, 0],
                        "냉각_정답": true.loc[f, "b"], "냉각_step2": arts[f]["thermal"].b, "냉각_이번": abh[i, 1],
                        "히터_정답": true.loc[f, "h"], "히터_step2": arts[f]["thermal"].h, "히터_이번": abh[i, 2]}
                       for f, i in fi.items()]).set_index("fridge")
    print("[배운 물리값]")
    print(pr.round(3).to_string())
    pr.to_csv(RESULTS / "hybrid_params.csv", encoding="utf-8-sig")

    # (1) 정상 운전: 30분 굴림 오차
    va = pd.read_csv(DATA / "val.csv")
    rows = []
    for (f, _), v in va.groupby(["fridge", "run"]):
        v = v.reset_index(drop=True)
        f = int(f)
        p = prep(v, arts[f])
        starts = np.arange(0, len(v) - WINDOW_H - 1, 10)
        segs = [(fi[f], p["T"][s], cut(p, s, s + WINDOW_H)) for s in starts]
        hy = np.array([r[-1] for r in batch_roll(model, segs)])
        ph = np.array([r[-1] for r in batch_roll(model, segs, use_corr=False)])
        truth = p["T"][starts + WINDOW_H]
        ol = arts[f]["thermal"].open_loop(p["T"].astype(float), p["Ta"], p["u"], p["d"], p["s"], WINDOW_H)
        rows.append({"fridge": f, "물리층만_이번": np.mean((ph - truth) ** 2),
                     "하이브리드": np.mean((hy - truth) ** 2),
                     "열모델_step2": np.nanmean((ol[starts + WINDOW_H] - truth) ** 2)})
    nr = pd.DataFrame(rows).groupby("fridge").mean() ** 0.5
    gru_n = RESULTS / "gpu_normal_rmse.csv"
    if gru_n.exists():
        nr = nr.join(pd.read_csv(gru_n).set_index("fridge")[["GRU_RMSE"]].rename(columns={"GRU_RMSE": "GRU"}))
    print(f"[정상 운전 {WINDOW_H}분 굴림 RMSE, °C]")
    print(nr.round(3).to_string())
    print("평균:", nr.mean().round(3).to_dict())
    nr.to_csv(RESULTS / "hybrid_normal_rmse.csv", encoding="utf-8-sig")

    # (2) 학습에 없던 과열: 공격 시작 직전의 보고 온도에서 출발해 끝까지 굴림
    eps = pd.read_csv(DATA / "episodes.csv")
    meta, segs = [], []
    for ep_id, df in eps[eps.attack.isin(ATTACKS)].groupby("episode"):
        df = df.reset_index(drop=True)
        f = int(df.fridge.iat[0])
        on = int(df.onset.iat[0])
        p = prep(df, arts[f])
        seg = cut(p, on - 1, len(df) - 1)
        segs.append((fi[f], p["T"][on - 1], seg))
        meta.append((ep_id, f, df.attack.iat[0], on, df, p))
    t1 = time.time()
    hys = batch_roll(model, segs)
    print(f"공격 {len(segs)}회 하이브리드 굴림 {time.time() - t1:.0f}초 (한꺼번에)")
    rows, keep = [], {}
    for (ep_id, f, atk, on, df, p), hy in zip(meta, hys):
        th = arts[f]["thermal"]
        thr = th.rollout(p["T"][on - 1], p["Ta"][on - 1:], p["u"][on - 1:], p["d"][on - 1:], p["s"][on - 1:])[1:]
        truth = df["T_true"].to_numpy()[on:]
        rows.append({"episode": ep_id, "fridge": f, "attack": atk,
                     "하이브리드_MAE": float(np.mean(np.abs(hy - truth))),
                     "열모델_MAE": float(np.mean(np.abs(thr - truth)))})
        if atk not in keep:
            keep[atk] = (truth, thr, hy)
    res = pd.DataFrame(rows)
    gru_a = RESULTS / "gpu_gru_vs_thermal.csv"
    if gru_a.exists():
        res = res.merge(pd.read_csv(gru_a)[["episode", "GRU_MAE"]], on="episode", how="left")
    print("[학습에 없던 과열, 공격 구간 전체 굴림 MAE 중앙값 (°C)]")
    print(res.groupby("attack").median(numeric_only=True).drop(columns="fridge").round(2).to_string())
    res.to_csv(RESULTS / "hybrid_attack_mae.csv", index=False, encoding="utf-8-sig")

    # 그림: 공격 두 종류에서 실제 / 열 모델 / 하이브리드
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"):
        if any(name == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    show = [a for a in ("defrost_freeze", "compoff_replay") if a in keep]
    xl = {"defrost_freeze": "히터 강제 가동 공격 시작 후 시간 (시간)", "compoff_replay": "냉각 장치 끄기 공격 시작 후 시간 (시간)"}
    fig, axes = plt.subplots(1, len(show), figsize=(11, 4), dpi=130, squeeze=False)
    for ax, atk in zip(axes[0], show):
        truth, thr, hy = keep[atk]
        t = np.arange(len(truth)) / 60
        ax.plot(t, truth, color="#2563eb", lw=1.8, label="실제 온도")
        ax.plot(t, thr, color="#059669", lw=1.6, ls="--", label="물리식 모델")
        ax.plot(t, hy, color="#d97706", lw=1.6, ls="-.", label="물리식 + AI 보정 (GPU)")
        ax.axhline(6.0, color="#6b7280", lw=0.8, ls=":")
        ax.set_xlabel(xl[atk])
        ax.set_ylabel("온도 (°C)")
        ax.grid(True, color="#e5e7eb")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_gpu_hybrid.png")
    print("저장: models/hybrid.pt, results/hybrid_params.csv, hybrid_normal_rmse.csv, hybrid_attack_mae.csv, fig_gpu_hybrid.png")
    print(f"전체 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
