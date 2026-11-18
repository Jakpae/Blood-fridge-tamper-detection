"""3단계: 공격 에피소드에서 탐지·조작 구간 식별·원인 판별·복원·남은 시간을 평가한다.

결과 파일 (results/)
  detection_by_config.csv   구성별(통계만 / +가상센서 / +전력 기반) 공격 유형별 탐지 성능
  episode_detail.csv        에피소드별 상세 (경보 시각, 시작 추정, 원인 판별, 복원 오차 등)
  reconstruction.csv        복원 방법별 오차와 신뢰 구간 포함률
  missing_reconstruction.csv  누락 구간 복원 방법 비교
  hmac_test.csv             HMAC 판정 결과
"""
import time

import joblib
import numpy as np
import pandas as pd

from config import DATA, MODELS, RESULTS, DAY, SAFE_LIMIT, WINDOW_H
from attacks import ATTACKS, DETECTION_ATTACKS
from detect import raw_signals, scores, alarms, onset_estimate, attribute, CONFIGS
import integrity


def interval_at(art, horizon):
    hs, qs = art["q_horizon"]
    return np.interp(horizon, hs, qs)


def first_true(mask, start=0):
    idx = np.flatnonzero(mask[start:])
    return start + idx[0] if idx.size else None


def rising_edges(mask):
    return int(np.sum(mask[1:] & ~mask[:-1]) + mask[0])


def main():
    t0 = time.time()
    arts = joblib.load(MODELS / "artifacts.joblib")   # 냉장고 번호 → 학습 결과
    eps = pd.read_csv(DATA / "episodes.csv")

    det_rows, ep_rows, rec_rows, miss_rows = [], [], [], []
    for ep_id, df in eps.groupby("episode", sort=False):
        df = df.reset_index(drop=True)
        art = arts[int(df["fridge"].iat[0])]     # 이 에피소드 냉장고의 모델
        th = art["thermal"]
        kind, onset = df["attack"].iat[0], int(df["onset"].iat[0])
        label = df["label"].to_numpy().astype(bool)
        sig, ex = raw_signals(df, art)
        sc = scores(sig, art)
        T_true, T_rep = df["T_true"].to_numpy(), df["T_rep"].to_numpy()
        n = len(df)

        # ------------------------------------------------ 누락 에피소드: 복원 방법만 비교
        if kind == "missing":
            gap = label
            lin = pd.Series(T_rep).interpolate(limit_direction="both").to_numpy()
            g0 = onset
            th_roll = th.rollout(T_rep[g0 - 1], df["Ta"].to_numpy()[g0:], ex["u"][g0:], ex["d"][g0:], ex["since"][g0:])
            th_full = np.full(n, np.nan); th_full[g0:] = th_roll
            for name, est in (("선형 보간", lin), ("가상 온도 센서(압력)", ex["T_virtual"]), ("열 모델(전력)", th_full)):
                miss_rows.append({"episode": ep_id, "gap_min": int(gap.sum()), "method": name,
                                  "MAE": float(np.mean(np.abs(est[gap] - T_true[gap]))),
                                  "max_err": float(np.max(np.abs(est[gap] - T_true[gap])))})
            continue

        normal = np.arange(n) < onset  # 공격 전 구간 (오탐 측정용)
        normal[:WINDOW_H] = False

        # ------------------------------------------------ 구성별 탐지 성능
        for cfg, mods in CONFIGS.items():
            a = alarms(sc, art, mods)
            hit = first_true(a, onset)
            det_rows.append({
                "episode": ep_id, "fridge": int(df["fridge"].iat[0]), "attack": kind, "config": cfg,
                "detected": hit is not None,
                "delay_min": (hit - onset) if hit is not None else np.nan,
                "TP": int(np.sum(a & label)), "FN": int(np.sum(~a & label)),
                "FP": int(np.sum(a & normal)), "TN": int(np.sum(~a & normal)),
                "false_alarm_events": rising_edges(a & normal),
                "normal_days": normal.sum() / DAY,
            })

        # ------------------------------------------------ 최종 구성(3)으로 사후 처리
        mods = CONFIGS[list(CONFIGS)[-1]]
        a = alarms(sc, art, mods)
        hit = first_true(a, onset)
        row = {"episode": ep_id, "fridge": int(df["fridge"].iat[0]), "attack": kind, "onset": onset, "alarm": hit}
        if hit is None:
            ep_rows.append(row)
            continue
        fired = [m for m in mods if sc[m][hit] > art["thr"][m]]
        first_mod = fired[0]
        on_est = hit if first_mod in ("z_T", "z_P", "maha", "virtual_pt") else onset_estimate(sc, art, first_mod, hit)
        cause = attribute(sig, art, hit, n)
        row.update(first_module=first_mod, onset_est=on_est, onset_err_min=on_est - onset, cause=cause)

        # 복원: 온도만 거짓 → 가상 온도 센서 / 온도·압력 모두 거짓 → 전력 기반 열 모델
        seg = slice(on_est, n)
        if cause in ("T", "?"):
            est = ex["T_virtual"][seg]
            half = np.full(len(est), art["q_virtual"])
            method = "가상 온도 센서(압력)"
        else:
            args = (df["Ta"].to_numpy()[seg], ex["u"][seg], ex["d"][seg])
            est = th.rollout(T_rep[on_est - 1], *args, ex["since"][seg])
            # 신뢰 구간 = 정상 운전에서 잰 굴림 오차 + 물리 파라미터 불확실성(부트스트랩, 외삽 구간에서 커짐)
            half = interval_at(art, np.arange(1, len(est) + 1)) + 1.96 * th.rollout_spread(T_rep[on_est - 1], *args)
            method = "열 모델(전력)"
        truth = T_true[seg]
        lab = label[seg]
        if lab.any():
            err = np.abs(est[lab] - truth[lab])
            rec_rows.append({
                "episode": ep_id, "attack": kind, "method": method,
                "MAE_restored": float(err.mean()),
                "MAE_reported": float(np.mean(np.abs(T_rep[seg][lab] - truth[lab]))),
                "coverage95": float(np.mean(err <= half[lab])),
                "mean_halfwidth": float(half[lab].mean()),
            })
            row["restore_MAE"] = float(err.mean())

        # 남은 시간: 경보 시점의 복원 온도에서, 지금 운전 상태(전력)가 이어진다고 보고 안전 한계 도달 시각 예측
        if ATTACKS[kind][1] is not None:
            T_now = est[hit - on_est]
            k_true = first_true(T_true > SAFE_LIMIT, hit)
            Ta_now = df["Ta"].iat[hit]
            hold = 24 * 60
            st = ex["state"][hit]
            u_h = np.full(hold, float(st == 1)); d_h = np.full(hold, float(st == 2))
            sin_h = np.minimum(ex["since"][hit] + np.arange(hold), 600)
            fut = th.rollout(T_now, np.full(hold, Ta_now), u_h, d_h, sin_h)
            k_pred = first_true(fut > SAFE_LIMIT)
            # 보수적 예측: 현재 온도를 신뢰 구간 상단으로 두고, 평균 문 열림 열을 더해 굴린다.
            #             그리고 앞으로 굴릴수록 넓어지는 신뢰 구간의 '상단'이 한계를 넘는 시각을 쓴다.
            T_hi = T_now + half[hit - on_est]
            fut_c = th.rollout(T_hi, np.full(hold, Ta_now), u_h, d_h, sin_h, extra=th.extra_heat)
            grow = interval_at(art, np.arange(hold)) + 1.96 * th.rollout_spread(T_hi, np.full(hold, Ta_now), u_h, d_h)
            k_cons = first_true(fut_c + grow > SAFE_LIMIT)
            row.update(T_now_est=T_now, T_now_true=T_true[hit],
                       limit_true_min=(k_true - hit) if k_true is not None else np.nan,
                       limit_pred_min=k_pred if k_pred is not None else np.nan,
                       limit_cons_min=k_cons if k_cons is not None else np.nan)
        ep_rows.append(row)

    det = pd.DataFrame(det_rows)
    epd = pd.DataFrame(ep_rows)
    rec = pd.DataFrame(rec_rows)
    miss = pd.DataFrame(miss_rows)

    # ------------------------------------------------ 요약 표
    g = det.groupby(["config", "attack"])
    summ = pd.DataFrame({
        "탐지율(에피소드)": g["detected"].mean(),
        "탐지 지연 중앙값(분)": g["delay_min"].median(),
        "재현율(시점)": g["TP"].sum() / (g["TP"].sum() + g["FN"].sum()),
    }).reset_index()
    order = {k: i for i, k in enumerate(DETECTION_ATTACKS)}
    summ = summ.sort_values(["config", "attack"], key=lambda s: s.map(order) if s.name == "attack" else s)
    summ.to_csv(RESULTS / "detection_by_config.csv", index=False, encoding="utf-8-sig")

    fp = det.groupby("config").agg(FP=("FP", "sum"), TN=("TN", "sum"),
                                   events=("false_alarm_events", "sum"), days=("normal_days", "sum"))
    fp["오탐률(시점)"] = fp["FP"] / (fp["FP"] + fp["TN"])
    fp["하루당 오경보"] = fp["events"] / fp["days"]
    fp.to_csv(RESULTS / "false_alarms.csv", encoding="utf-8-sig")

    epd.to_csv(RESULTS / "episode_detail.csv", index=False, encoding="utf-8-sig")
    rec.to_csv(RESULTS / "reconstruction.csv", index=False, encoding="utf-8-sig")
    miss.to_csv(RESULTS / "missing_reconstruction.csv", index=False, encoding="utf-8-sig")

    h = integrity.self_test()
    pd.DataFrame([{"type": k, "count": v[0], "flagged": v[1]} for k, v in h.items()]).to_csv(
        RESULTS / "hmac_test.csv", index=False)

    # ------------------------------------------------ 화면 출력
    pd.set_option("display.width", 200)
    print("\n=== 구성별 탐지율 (에피소드 기준) ===")
    print(summ.pivot(index="attack", columns="config", values="탐지율(에피소드)").loc[DETECTION_ATTACKS].round(2))
    print("\n=== 구성별 탐지 지연 중앙값 (분) ===")
    print(summ.pivot(index="attack", columns="config", values="탐지 지연 중앙값(분)").loc[DETECTION_ATTACKS])
    print("\n=== 정상 구간 오경보 ===")
    print(fp[["오탐률(시점)", "하루당 오경보"]].round(4))
    print("\n=== 최종 구성: 조작 시작 추정 오차·원인 판별 ===")
    print(epd.groupby("attack").agg(onset_err_median=("onset_err_min", "median"),
                                    onset_err_abs_mean=("onset_err_min", lambda x: np.nanmean(np.abs(x))),
                                    cause=("cause", lambda x: x.value_counts().to_dict())).loc[DETECTION_ATTACKS])
    print("\n=== 복원 (공격 구간, °C) ===")
    print(rec.groupby(["attack", "method"])[["MAE_reported", "MAE_restored", "coverage95", "mean_halfwidth"]].mean().round(2))
    if "limit_true_min" in epd:
        lt = epd.dropna(subset=["limit_true_min", "limit_pred_min"])
        lt = lt[lt["limit_true_min"] > 0]
        if len(lt):
            print("\n=== 안전 한계까지 남은 시간 (경보 시점) ===")
            print(lt.groupby("attack").apply(lambda x: pd.Series({
                "건수": len(x), "실제 중앙값(분)": x.limit_true_min.median(),
                "중앙 예측(분)": x.limit_pred_min.median(),
                "보수적 예측(분)": x.limit_cons_min.median(),
                "보수적 예측이 늦지 않은 비율": np.mean(x.limit_cons_min <= x.limit_true_min)}),
                include_groups=False).round(2))
    print("\n=== 누락 구간 복원 MAE (°C) ===")
    print(miss.groupby("method")[["MAE", "max_err"]].mean().round(3))
    last = list(CONFIGS)[-1]
    by_f = det[det.config == last].groupby("fridge").agg(
        탐지율=("detected", "mean"), 지연중앙값=("delay_min", "median"),
        오경보_하루당=("false_alarm_events", "sum"), 정상일수=("normal_days", "sum"))
    by_f["오경보_하루당"] = by_f["오경보_하루당"] / by_f["정상일수"]
    by_f.drop(columns="정상일수").to_csv(RESULTS / "detection_by_fridge.csv", encoding="utf-8-sig")
    print("\n=== 최종 구성: 냉장고별 ===")
    print(by_f.drop(columns="정상일수").round(3))
    print("\n=== HMAC ===", h)
    print(f"\n완료 ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
