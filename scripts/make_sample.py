"""Build data/sample/pairs_sample.csv from a single-lane NGSIM US-101 window.

Input: CSV with columns vehicle_id, global_time, local_y (ft), v_vel (ft/s),
lane_id, v_class, all rows from one lane, 10 Hz.

    python scripts/make_sample.py path/to/ngsim_us101_lane_window.csv
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from carfollow.schema import COLUMNS, DT, N_GROUPS, validate  # noqa: E402

FT_TO_M = 0.3048
STEP_MS = int(DT * 1000)
SMOOTH_WINDOW = 21          # rows = 2.1 s
SMOOTH_ORDER = 2
NOMINAL_LEADER_LENGTH_M = 4.5   # the lane window has no vehicle length column
MIN_PAIR_ROWS = 100         # 10 s
SEED = 517


def smooth_trajectories(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values(["vehicle_id", "global_time"]).reset_index(drop=True)
    df["position_m"] = df["local_y"] * FT_TO_M
    new_traj = df.groupby("vehicle_id")["global_time"].diff().ne(STEP_MS)
    df["traj_no"] = new_traj.groupby(df["vehicle_id"]).cumsum()
    df["traj_id"] = df["vehicle_id"].astype(str) + "_" + df["traj_no"].astype(str)

    parts = []
    for _, t in df.groupby("traj_id", sort=False):
        if len(t) < SMOOTH_WINDOW:
            continue
        raw_x = t["position_m"].to_numpy()
        v = np.clip(savgol_filter(raw_x, SMOOTH_WINDOW, SMOOTH_ORDER, deriv=1, delta=DT), 0.0, None)
        # Position is integrated from speed and accel is a forward difference of speed,
        # so carfollow.kinematics.advance replays the data exactly.
        x = savgol_filter(raw_x, SMOOTH_WINDOW, SMOOTH_ORDER)[0] + np.concatenate([[0.0], np.cumsum(v[:-1]) * DT])
        t = t.iloc[:-1].copy()
        t["position_m"] = x[:-1]
        t["speed_mps"] = v[:-1]
        t["accel_mps2"] = np.diff(v) / DT
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def attach_leaders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["global_time", "position_m"]).reset_index(drop=True)
    ahead = df.groupby("global_time").shift(-1)
    df["leader_id"] = ahead["traj_id"]
    df["leader_position_m"] = ahead["position_m"]
    df["leader_speed_mps"] = ahead["speed_mps"]
    df["leader_accel_mps2"] = ahead["accel_mps2"]
    return df.dropna(subset=["leader_id"]).reset_index(drop=True)


def cut_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["traj_id", "global_time"]).reset_index(drop=True)
    df["gap_m"] = df["leader_position_m"] - df["position_m"] - NOMINAL_LEADER_LENGTH_M
    same_run = (
        df["traj_id"].eq(df["traj_id"].shift())
        & df["leader_id"].eq(df["leader_id"].shift())
        & df["global_time"].diff().eq(STEP_MS)
    )
    df["run"] = (~same_run).cumsum()
    run_size = df.groupby("run")["run"].transform("size")
    run_min_gap = df.groupby("run")["gap_m"].transform("min")
    df = df[(run_size >= MIN_PAIR_ROWS) & (run_min_gap > 0)].copy()

    df["driver_id"] = df["vehicle_id"].astype(str)
    df["leader_id"] = df["leader_id"].str.split("_").str[0]
    df["pair_id"] = df["driver_id"] + "->" + df["leader_id"] + "@" + df.groupby("run")["global_time"].transform("min").astype(str)
    df["global_time_ms"] = df["global_time"]
    df["t_s"] = (df["global_time"] - df.groupby("run")["global_time"].transform("min")) / 1000.0
    df["closing_speed_mps"] = df["speed_mps"] - df["leader_speed_mps"]

    drivers = np.array(sorted(df["driver_id"].unique()))
    np.random.default_rng(SEED).shuffle(drivers)
    group_of = {d: i % N_GROUPS for i, d in enumerate(drivers)}
    df["driver_group"] = df["driver_id"].map(group_of)

    out = df[list(COLUMNS)].astype(COLUMNS)
    return out.sort_values(["pair_id", "t_s"], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lane_window_csv", type=Path)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "sample" / "pairs_sample.csv")
    args = ap.parse_args()

    raw = pd.read_csv(args.lane_window_csv)
    if raw["lane_id"].nunique() != 1:
        raise ValueError("input must contain a single lane")
    table = cut_pairs(attach_leaders(smooth_trajectories(raw)))
    validate(table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {args.out.resolve()}")
    table.to_csv(args.out, index=False, float_format="%.4f")

    meta = {
        "source": "NGSIM US-101 vehicle trajectories, data.transportation.gov (dataset 8ect-6jqj), CC BY-SA 3.0",
        "input_file": args.lane_window_csv.name,
        "lane_id": int(raw["lane_id"].iloc[0]),
        "global_time_ms_range": [int(raw["global_time"].min()), int(raw["global_time"].max())],
        "built": date.today().isoformat(),
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "drivers": int(table["driver_id"].nunique()),
        "rows_per_group": {int(k): int(v) for k, v in table["driver_group"].value_counts().sort_index().items()},
        "notes": [
            "Units converted from ft to m.",
            f"Speed is the derivative of a Savitzky-Golay fit to position ({SMOOTH_WINDOW} rows, order {SMOOTH_ORDER}).",
            "position(t+1) = position(t) + speed(t) * dt and accel(t) = (speed(t+1) - speed(t)) / dt, so carfollow.kinematics.advance replays every pair exactly.",
            "driver_id is the physical vehicle, so a car that leaves and re-enters the lane stays in one driver_group.",
            f"gap_m assumes every leader is {NOMINAL_LEADER_LENGTH_M} m long; the full dataset will use real vehicle lengths.",
            "Leader = next car ahead in the same lane at the same time; a pair ends when the leader changes or either car leaves the lane.",
            f"Pairs shorter than {MIN_PAIR_ROWS * DT:.0f} s are dropped.",
        ],
    }
    meta_path = args.out.with_suffix(".meta.json")
    print(f"writing {meta_path.resolve()}")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("rows", "pairs", "drivers", "rows_per_group")}, indent=2))


if __name__ == "__main__":
    main()
