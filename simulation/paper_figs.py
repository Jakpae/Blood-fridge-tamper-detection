"""보고서·README용 그림 (실험 결과만 사용). step2~3 실행 후 실행 → results/paper_figs/

글꼴은 Pretendard가 있으면 쓰고, 없으면 step4_plots.py가 고른 한글 글꼴을 쓴다."""
import os, glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import step4_plots as S  # 데이터 처리 함수 재사용
from config import DATA, MODELS, RESULTS, SAFE_LIMIT
from attacks import DETECTION_ATTACKS
from detect import raw_signals, scores, alarms, onset_estimate, CONFIGS

OUT = str(RESULTS / "paper_figs")
os.makedirs(OUT, exist_ok=True)
_has_pretendard = any("Pretendard" in x.name for x in font_manager.fontManager.ttflist)
plt.rcParams.update({**({"font.family": "Pretendard"} if _has_pretendard else {}), "font.size": 9, "axes.unicode_minus": False,
                     "figure.dpi": 300, "savefig.dpi": 300, "axes.titlesize": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

INK, MUTED, GRID = "#1D1D1F", "#6E6E73", "#E5E5EA"
BLUE, RED, GRAY1, GRAY2 = "#0066CC", "#D70015", "#D1D1D6", "#8E8E93"

NAMES = {"offset": "온도 낮게 (−1.5°C)", "spike": "순간 튐", "drift": "서서히 어긋남", "multi": "온도·압력 함께",
         "defrost_freeze": "히터 + 숫자 멈춤 (위장)", "defrost_replay": "히터 + 전날 기록 (위장)",
         "compoff_replay": "냉각 정지 + 전날 기록 (위장)"}
CFG = {"1) 통계만": "① 통계 검사만", "2) 통계 + 가상 온도 센서": "② + 압력 가상 센서",
       "3) 통계 + 가상 센서 + 전력 기반 검사": "③ + 전력 기반 검사 (제안)"}


def style(ax, grid_axis="both"):
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, width=0.6)


def fig_params():
    fp = pd.read_csv(RESULTS / "fridge_params.csv")
    panels = (("(a) 시정수 1/a (시간)", 1 / fp.a_true / 60, 1 / fp.a / 60),
              ("(b) 냉각 속도 b (°C/분)", fp.b_true, fp.b),
              ("(c) 가열 속도 h (°C/분)", fp.h_true, fp.h))
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.3))
    for ax, (name, t, l) in zip(axes, panels):
        lo, hi = min(t.min(), l.min()), max(t.max(), l.max())
        pad = (hi - lo) * 0.25
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRAY2, lw=0.8, ls="--", zorder=1)
        ax.scatter(t, l, color=BLUE, s=14, zorder=2)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(name, loc="left", color=INK)
        ax.set_xlabel("시뮬레이터 설정값"); ax.set_ylabel("추정값")
        style(ax)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(f"{OUT}/fig_params.png"); plt.close(fig)


def fig_detection():
    det = pd.read_csv(RESULTS / "detection_by_config.csv")
    piv = det.pivot(index="attack", columns="config", values="탐지율(에피소드)").loc[DETECTION_ATTACKS]
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    y = np.arange(len(piv)); h = 0.26
    cols = [GRAY1, GRAY2, BLUE]
    for i, (cfg, c) in enumerate(zip(piv.columns, cols)):
        v = piv[cfg].to_numpy() * 100
        ax.barh(y + (i - 1) * h, v, height=h, color=c, label=CFG[cfg])
        if i == 2:
            continue
        for yy, vv in zip(y + (i - 1) * h, v):
            if vv < 100:
                ax.text(vv + 1.2, yy, f"{vv:.0f}%", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y, [NAMES[k] for k in piv.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 108); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("탐지한 공격 비율 (%)  ·  공격 유형별 50회, 냉장고 8대")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.35, -0.18), ncol=3, handlelength=1.2)
    style(ax, "x")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_detection.png"); plt.close(fig)


def fig_restore(ep_id="defrost_replay_00"):
    arts = joblib.load(MODELS / "artifacts.joblib")
    eps = pd.read_csv(DATA / "episodes.csv")
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
    lo = onset - 3 * 60
    t = (np.arange(len(df)) - onset) / 60.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.6, 3.9), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    ax.axvspan(0, t[-1], color=RED, alpha=0.04, lw=0)
    ax.axhline(SAFE_LIMIT, color=RED, lw=0.8, ls="--")
    ax.text(t[-1], SAFE_LIMIT + 0.6, f"보관 상한 {SAFE_LIMIT:.0f}°C", color=RED, fontsize=7.5, ha="right")
    ax.plot(t[lo:], df.T_true[lo:], color=INK, lw=1.5, label="실제 고내 온도")
    ax.plot(t[lo:], df.T_rep[lo:], color=RED, lw=1.0, label="화면 온도 (조작됨)")
    ax.plot(t[seg], rec, color=BLUE, lw=1.5, ls="--", label="복원 온도 (전력 열 모델)")
    ax.axvline(0, color=GRAY2, lw=0.8, ls=":")
    ax.axvline(t[hit], color=BLUE, lw=0.8)
    ymax = ax.get_ylim()[1]
    ax.text(0.05, ymax * 0.97, "공격 시작", color=MUTED, fontsize=7.5, va="top")
    ax.text(t[hit] + 0.05, ymax * 0.86, f"경보 (+{hit - onset}분)", color=BLUE, fontsize=7.5, va="top")
    ax.set_ylabel("온도 (°C)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=False)
    style(ax)
    ax2.plot(t[lo:], df.W[lo:], color=INK, lw=0.7)
    ax2.set_ylabel("전력 (W)")
    ax2.set_xlabel("공격 시작 기준 시간 (시간)")
    style(ax2)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(f"{OUT}/fig_restore.png"); plt.close(fig)
    print("restore: onset", onset, "hit", hit, "on_est", on_est, "fridge", int(df.fridge.iat[0]),
          "MAE", float(np.mean(np.abs(rec - df.T_true.to_numpy()[seg]))))


if __name__ == "__main__":
    fig_params(); fig_detection(); fig_restore()
    print(os.listdir(OUT))
