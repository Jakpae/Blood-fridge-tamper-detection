"""4단계: 발표·게시물용 그림 (results/*.png)."""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from config import DATA, MODELS, RESULTS, SAFE_LIMIT
from attacks import DETECTION_ATTACKS, ATTACKS
from detect import raw_signals, scores, alarms, onset_estimate, CONFIGS
from models import ThermalModel

# 한글 글꼴: 윈도우는 맑은 고딕, 없으면 다른 한글 글꼴을 찾는다.
for f in ("Malgun Gothic", "Noto Sans CJK KR", "NanumGothic", "AppleGothic"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
C_TRUE, C_REP, C_REC, C_ALARM = "#2563eb", "#dc2626", "#059669", "#7c3aed"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def episode_figure(ep_id, arts, eps, fname, title):
    df = eps[eps.episode == ep_id].reset_index(drop=True)
    art = arts[int(df.fridge.iat[0])]
    sig, ex = raw_signals(df, art)
    sc = scores(sig, art)
    mods = CONFIGS[list(CONFIGS)[-1]]
    a = alarms(sc, art, mods)
    onset = int(df.onset.iat[0])
    hit = onset + int(np.flatnonzero(a[onset:])[0])
    first = [m for m in mods if sc[m][hit] > art["thr"][m]][0]
    on_est = onset_estimate(sc, art, first, hit) if first not in ("z_T", "z_P", "maha", "virtual_pt") else hit
    th = art["thermal"]
    seg = slice(on_est, len(df))
    args = (df.Ta.to_numpy()[seg], ex["u"][seg], ex["d"][seg])
    rec = th.rollout(df.T_rep.iat[on_est - 1], *args, ex["since"][seg])
    hs, qs = art["q_horizon"]
    half = np.interp(np.arange(1, len(rec) + 1), hs, qs) + 1.96 * th.rollout_spread(df.T_rep.iat[on_est - 1], *args)

    lo = onset - 8 * 60
    t = (np.arange(len(df)) - onset) / 60.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(t[lo:], df.T_true[lo:], color=C_TRUE, lw=1.6, label="실제 냉장고 안 온도")
    ax.plot(t[lo:], df.T_rep[lo:], color=C_REP, lw=1.2, label="화면에 보인 온도 (조작됨)")
    tt = t[seg]
    ax.fill_between(tt, rec - half, rec + half, color=C_REC, alpha=0.18, lw=0)
    ax.plot(tt, rec, color=C_REC, lw=1.8, ls="--", label="되살린 온도 (전력으로 계산)")
    ax.axhline(SAFE_LIMIT, color=MUTED, lw=1, ls=":")
    ax.text(t[lo], SAFE_LIMIT + 0.5, f"혈액 보관 한계 {SAFE_LIMIT:.0f}°C", color=MUTED, fontsize=8)
    ax.axvline(t[hit], color=C_ALARM, lw=1.2)
    ax.text(t[hit] + 0.08, ax.get_ylim()[1] * 0.92, f"경보 (+{hit - onset}분)", color=C_ALARM, fontsize=9)
    ax.axvline(0, color=INK, lw=0.8, ls="--")
    ax.text(0.08, ax.get_ylim()[1] * 0.80, "공격 시작", color=INK, fontsize=9)
    ax.set_ylabel("온도 (°C)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    style(ax)
    ax2.plot(t[lo:], df.W[lo:], color=INK, lw=0.9)
    ax2.set_ylabel("전력 (W)")
    ax2.set_xlabel("공격 시작 기준 시간 (시간)")
    style(ax2)
    fig.tight_layout()
    fig.savefig(RESULTS / fname)
    plt.close(fig)


CFG_LABEL = {"1) 통계만": "① 통계 검사만",
             "2) 통계 + 가상 온도 센서": "② + 압력으로 온도 다시 계산",
             "3) 통계 + 가상 센서 + 전력 기반 검사": "③ + 전력으로 온도 다시 계산"}


def detection_figure():
    det = pd.read_csv(RESULTS / "detection_by_config.csv")
    piv = det.pivot(index="attack", columns="config", values="탐지율(에피소드)").loc[DETECTION_ATTACKS]
    names = [ATTACKS[k][0] for k in piv.index]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(piv))
    colors = ["#cbd5e1", "#93c5fd", "#1d4ed8"]
    h = 0.26
    for i, (cfg, c) in enumerate(zip(piv.columns, colors)):
        ax.barh(y + (i - 1) * h, piv[cfg] * 100, height=h, color=c, label=CFG_LABEL.get(cfg, cfg))
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("탐지한 공격 비율 (%), 공격 유형별 50회 (냉장고 8대)")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_detection_by_config.png")
    plt.close(fig)


def tp_scatter(arts):
    art = arts[0]
    tr = pd.read_csv(DATA / "train.csv")
    tr = tr[tr.fridge == 0].iloc[:4 * 1440]
    st = art["power"].predict(tr.W)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k, name, c in ((0, "쉬는 중", "#94a3b8"), (1, "냉각 중", "#2563eb"), (2, "성에 제거 히터 켜짐", "#dc2626")):
        m = st == k
        ax.scatter(tr.T_rep[m], tr.P_rep[m], s=4, color=c, alpha=0.5, label=f"{name} (전력으로 구분)")
    ax.set_xlabel("냉장고 안 온도 (°C)")
    ax.set_ylabel("냉매 압력 (bar)")
    ax.legend(fontsize=8, frameon=False, markerscale=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_T_P_by_state.png")
    plt.close(fig)


def hybrid_figure(arts, eps):
    """학습에 없던 과열 상황에서 보정층을 무조건 쓰면 어떻게 되는가."""
    df = eps[eps.episode == "defrost_freeze_00"].reset_index(drop=True)
    art = arts[int(df.fridge.iat[0])]
    _, ex = raw_signals(df, art)
    on = int(df.onset.iat[0])
    seg = slice(on, len(df))
    args = (df.Ta.to_numpy()[seg], ex["u"][seg], ex["d"][seg], ex["since"][seg])
    th = art["thermal"]
    gated = th.rollout(df.T_true.iat[on - 1], *args)
    keep = (th.dom, th.since_max)
    th.dom, th.since_max = (-1e9, 1e9), {0: 1e9, 1: 1e9, 2: 1e9}
    always = th.rollout(df.T_true.iat[on - 1], *args)
    th.dom, th.since_max = keep
    t = np.arange(len(gated)) / 60
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, df.T_true.to_numpy()[seg], color=C_TRUE, lw=1.8, label="실제 온도")
    ax.plot(t, gated, color=C_REC, lw=1.8, ls="--", label="물리식 + 익숙한 범위에서만 AI 보정")
    ax.plot(t, always, color=C_REP, lw=1.4, ls=":", label="AI 보정을 항상 사용")
    ax.set_xlabel("히터 강제 가동 공격 시작 후 시간 (시간)")
    ax.set_ylabel("온도 (°C)")
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_hybrid_extrapolation.png")
    plt.close(fig)


def fridge_params_figure():
    """냉장고 8대의 실제 물리값과 모델이 배운 값 비교."""
    fp = pd.read_csv(RESULTS / "fridge_params.csv")
    panels = (("단열 성능 (분, 클수록 열이 덜 들어옴)", 1 / fp.a_true, 1 / fp.a),
              ("냉각 능력 (°C/분)", fp.b_true, fp.b),
              ("히터 세기 (°C/분)", fp.h_true, fp.h))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (name, t, l) in zip(axes, panels):
        lo, hi = min(t.min(), l.min()), max(t.max(), l.max())
        pad = (hi - lo) * 0.15
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID, lw=1.2, zorder=0)
        ax.scatter(t, l, color=C_TRUE, s=28, zorder=2)
        for f, x, y in zip(fp.fridge, t, l):
            ax.annotate(str(f), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=MUTED)
        ax.set_xlabel(f"실제 {name}", fontsize=8)
        ax.set_ylabel(f"모델이 배운 {name}", fontsize=8)
        style(ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_fridge_params.png")
    plt.close(fig)


def main():
    arts = joblib.load(MODELS / "artifacts.joblib")
    eps = pd.read_csv(DATA / "episodes.csv")
    episode_figure("defrost_replay_00", arts, eps, "fig_example_defrost_replay.png", "")
    episode_figure("compoff_replay_00", arts, eps, "fig_example_compoff_replay.png", "")
    detection_figure()
    tp_scatter(arts)
    hybrid_figure(arts, eps)
    fridge_params_figure()
    print("저장:", ", ".join(p.name for p in sorted(RESULTS.glob("fig_*.png"))))


if __name__ == "__main__":
    main()
