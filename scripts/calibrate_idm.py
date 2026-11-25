"""Fit one shared IDM parameter set on pairs_full.csv (Lane A deliverable).

Minimizes step RMSE of predicted acceleration vs observed accel_mps2.
delta is fixed at 4 (Treiber et al.). Extreme |accel| > 8 m/s^2 rows are
dropped as tracking noise (see pairs_full.meta.json).

Usage (from repo root, with data/full/pairs_full.csv present):

    python scripts/calibrate_idm.py
    python scripts/calibrate_idm.py --max-rows 250000 --seed 517

Writes lanes/A_data_idm/idm_calibrated.json and prints the numbers to paste
into IDM.calibrated() in carfollow/baselines.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parents[1]
FULL_CSV = REPO / "data" / "full" / "pairs_full.csv"
OUT_JSON = REPO / "lanes" / "A_data_idm" / "idm_calibrated.json"

# (v0, T, s0, a_max, b); delta fixed at 4
# Bounds are kept inside typical NGSIM/IDM ranges so the fit stays interpretable
# (very wide bounds push a_max→0 and v0→highway free-flow and look like noise fitting).
PARAM_NAMES = ("v0", "T", "s0", "a_max", "b")
TEXTBOOK = np.array([30.0, 1.5, 2.0, 1.0, 1.5], dtype=float)
BOUNDS_LO = np.array([10.0, 0.5, 1.0, 0.5, 1.0], dtype=float)
BOUNDS_HI = np.array([40.0, 3.0, 5.0, 2.5, 3.5], dtype=float)
DELTA = 4.0
MAX_BRAKE = 9.0
ACCEL_CLIP = 8.0  # drop noisier rows beyond this |accel|


def idm_accel_vec(gap, v, dv, params: np.ndarray) -> np.ndarray:
    v0, T, s0, a_max, b = params
    gap = np.maximum(gap, 1e-6)
    v = np.maximum(v, 0.0)
    s_star = s0 + np.maximum(0.0, v * T + v * dv / (2.0 * np.sqrt(a_max * b)))
    a = a_max * (1.0 - (v / v0) ** DELTA - (s_star / gap) ** 2)
    return np.maximum(a, -MAX_BRAKE)


def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


def load_features(path: Path, max_rows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usecols = ["driver_group", "gap_m", "speed_mps", "closing_speed_mps", "accel_mps2"]
    print(f"loading {path} ...")
    t0 = time.perf_counter()
    df = pd.read_csv(path, usecols=usecols)
    print(f"  {len(df):,} rows in {time.perf_counter() - t0:.1f}s")

    mask = (df["gap_m"] > 0) & (df["accel_mps2"].abs() <= ACCEL_CLIP)
    df = df.loc[mask]
    print(f"  {len(df):,} rows after dropping gap<=0 or |accel|>{ACCEL_CLIP}")

    if max_rows is not None and len(df) > max_rows:
        # Stratify by driver_group so each fold stays represented.
        rng = np.random.default_rng(seed)
        parts = []
        per = max_rows // 5
        for g in range(5):
            sub = df[df["driver_group"] == g]
            take = min(per, len(sub))
            idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            parts.append(df.loc[idx])
        df = pd.concat(parts, ignore_index=True)
        print(f"  subsampled to {len(df):,} rows (seed={seed})")

    gap = df["gap_m"].to_numpy(dtype=float)
    speed = df["speed_mps"].to_numpy(dtype=float)
    closing = df["closing_speed_mps"].to_numpy(dtype=float)
    accel = df["accel_mps2"].to_numpy(dtype=float)
    group = df["driver_group"].to_numpy(dtype=int)
    return gap, speed, closing, accel, group


def fit(gap, speed, closing, accel) -> tuple[np.ndarray, dict]:
    def residual(params: np.ndarray) -> np.ndarray:
        return idm_accel_vec(gap, speed, closing, params) - accel

    t0 = time.perf_counter()
    result = least_squares(
        residual,
        x0=TEXTBOOK.copy(),
        bounds=(BOUNDS_LO, BOUNDS_HI),
        method="trf",
        loss="linear",
        max_nfev=120,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0
    params = result.x
    y_hat = idm_accel_vec(gap, speed, closing, params)
    y_tb = idm_accel_vec(gap, speed, closing, TEXTBOOK)
    info = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
        "fit_seconds": round(elapsed, 2),
        "step_rmse_calibrated": round(rmse(y_hat, accel), 6),
        "step_rmse_textbook": round(rmse(y_tb, accel), 6),
    }
    return params, info


def leave_one_group_out(gap, speed, closing, accel, group) -> list[dict]:
    rows = []
    for g in range(5):
        train = group != g
        test = group == g
        params, info = fit(gap[train], speed[train], closing[train], accel[train])
        y_hat = idm_accel_vec(gap[test], speed[test], closing[test], params)
        y_tb = idm_accel_vec(gap[test], speed[test], closing[test], TEXTBOOK)
        rows.append(
            {
                "test_group": g,
                "params": {n: round(float(v), 6) for n, v in zip(PARAM_NAMES, params)},
                "test_step_rmse_calibrated": round(rmse(y_hat, accel[test]), 6),
                "test_step_rmse_textbook": round(rmse(y_tb, accel[test]), 6),
                "fit_seconds": info["fit_seconds"],
                "nfev": info["nfev"],
            }
        )
        print(
            f"  LOGO test_group={g}: "
            f"calibrated RMSE={rows[-1]['test_step_rmse_calibrated']:.4f} "
            f"vs textbook {rows[-1]['test_step_rmse_textbook']:.4f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=FULL_CSV)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--max-rows", type=int, default=400_000, help="stratified subsample size; 0 = use all")
    parser.add_argument("--seed", type=int, default=517)
    parser.add_argument("--skip-logo", action="store_true", help="skip leave-one-group-out report")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"missing {args.csv}; put pairs_full.csv under data/full/")

    max_rows = None if args.max_rows == 0 else args.max_rows
    gap, speed, closing, accel, group = load_features(args.csv, max_rows, args.seed)

    print("fitting one shared parameter set on all sampled rows ...")
    params, info = fit(gap, speed, closing, accel)
    params_dict = {n: round(float(v), 6) for n, v in zip(PARAM_NAMES, params)}
    print("  params:", params_dict)
    print(
        f"  step RMSE calibrated={info['step_rmse_calibrated']:.4f} "
        f"textbook={info['step_rmse_textbook']:.4f} "
        f"({info['fit_seconds']}s, nfev={info['nfev']})"
    )

    logo = []
    if not args.skip_logo:
        print("leave-one-group-out (fit on 4 groups, score on held-out) ...")
        logo = leave_one_group_out(gap, speed, closing, accel, group)

    payload = {
        "model": "IDM",
        "name": "idm_calibrated",
        "source_csv": str(args.csv.relative_to(REPO)).replace("\\", "/"),
        "delta_fixed": DELTA,
        "accel_clip_mps2": ACCEL_CLIP,
        "max_rows": max_rows,
        "seed": args.seed,
        "n_rows_used": int(len(accel)),
        "loss": "linear least squares on accel residual",
        "bounds": {
            "lo": {n: float(v) for n, v in zip(PARAM_NAMES, BOUNDS_LO)},
            "hi": {n: float(v) for n, v in zip(PARAM_NAMES, BOUNDS_HI)},
        },
        "params": params_dict,
        "textbook_params": {n: float(v) for n, v in zip(PARAM_NAMES, TEXTBOOK)},
        "fit": info,
        "leave_one_group_out": logo,
        "notes": [
            "One shared parameter set for everyone (Lane A milestone).",
            "Fit on pooled sampled rows from all five driver groups.",
            "LOGO block is diagnostic only; IDM.calibrated() uses `params`.",
            "delta kept at 4; not fitted.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO)}")
    print("paste into IDM.calibrated():")
    print(
        "        return cls(\n"
        f"            v0={params_dict['v0']},\n"
        f"            T={params_dict['T']},\n"
        f"            s0={params_dict['s0']},\n"
        f"            a_max={params_dict['a_max']},\n"
        f"            b={params_dict['b']},\n"
        "            name=\"idm_calibrated\",\n"
        "        )"
    )


if __name__ == "__main__":
    main()
