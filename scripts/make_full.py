"""Build data/full/pairs_full.csv from the downloaded NGSIM US-101 pages (main lanes 1-5).

Same steps and columns as scripts/make_sample.py, plus:
  - all five main lanes and all three 15-minute periods;
  - gap_m uses the leader's real length (v_length) instead of 4.5 m;
  - a vehicle_id reused by a different car later in the day becomes a separate driver_id.

    python scripts/download_ngsim.py --out data/raw/us101
    python scripts/make_full.py data/raw/us101
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from carfollow.schema import COLUMNS, DT, N_GROUPS, validate  # noqa: E402
from make_sample import FT_TO_M, MIN_PAIR_ROWS, SEED, SMOOTH_ORDER, SMOOTH_WINDOW, STEP_MS, smooth_trajectories  # noqa: E402

REUSE_GAP_MS = 60_000   # same vehicle_id absent for over 60 s = a different car


def load_pages(raw_dir: Path) -> pd.DataFrame:
    pages = sorted(raw_dir.glob("page_*.csv"))
    if not pages:
        raise FileNotFoundError(f"no page_*.csv in {raw_dir}; run scripts/download_ngsim.py first")
    raw = pd.concat((pd.read_csv(p) for p in pages), ignore_index=True)
    before = len(raw)
    raw = raw.drop_duplicates(["vehicle_id", "global_time"]).reset_index(drop=True)
    return raw, before - len(raw)


def assign_physical_ids(raw: pd.DataFrame) -> pd.DataFrame:
    """vehicle_id 'N' becomes 'N' for the first car using it and 'N.k' for the k-th reuse."""
    raw = raw.sort_values(["vehicle_id", "global_time"]).reset_index(drop=True)
    new_car = raw.groupby("vehicle_id")["global_time"].diff().gt(REUSE_GAP_MS)
    k = new_car.groupby(raw["vehicle_id"]).cumsum()
    base = raw["vehicle_id"].astype(str)
    raw["vehicle_id"] = base.where(k.eq(0), base + "." + k.astype(str))
    return raw


def attach_leaders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["global_time", "lane_id", "position_m"]).reset_index(drop=True)
    ahead = df.groupby(["global_time", "lane_id"]).shift(-1)
    df["leader_id"] = ahead["traj_id"]
    df["leader_position_m"] = ahead["position_m"]
    df["leader_speed_mps"] = ahead["speed_mps"]
    df["leader_accel_mps2"] = ahead["accel_mps2"]
    df["leader_length_m"] = ahead["v_length"] * FT_TO_M
    df["leader_vehicle_raw"] = ahead["vehicle_id_raw"]
    return df.dropna(subset=["leader_id"]).reset_index(drop=True)


def cut_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["traj_id", "global_time"]).reset_index(drop=True)
    df["gap_m"] = df["leader_position_m"] - df["position_m"] - df["leader_length_m"]
    same_run = (
        df["traj_id"].eq(df["traj_id"].shift())
        & df["leader_id"].eq(df["leader_id"].shift())
        & df["global_time"].diff().eq(STEP_MS)
    )
    df["run"] = (~same_run).cumsum()
    run_size = df.groupby("run")["run"].transform("size")
    run_min_gap = df.groupby("run")["gap_m"].transform("min")
    df = df[(run_size >= MIN_PAIR_ROWS) & (run_min_gap > 0)].copy()

    df["driver_id"] = df["traj_id"].str.rsplit("_", n=1).str[0]
    df["leader_id"] = df["leader_id"].str.rsplit("_", n=1).str[0]
    df["pair_id"] = df["driver_id"] + "->" + df["leader_id"] + "@" + df.groupby("run")["global_time"].transform("min").astype(str)
    df["global_time_ms"] = df["global_time"]
    df["t_s"] = (df["global_time"] - df.groupby("run")["global_time"].transform("min")) / 1000.0
    df["closing_speed_mps"] = df["speed_mps"] - df["leader_speed_mps"]

    drivers = np.array(sorted(df["driver_id"].unique()))
    np.random.default_rng(SEED).shuffle(drivers)
    group_of = {d: i % N_GROUPS for i, d in enumerate(drivers)}
    df["driver_group"] = df["driver_id"].map(group_of)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", type=Path)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "full" / "pairs_full.csv")
    args = ap.parse_args()

    raw, n_dupes = load_pages(args.raw_dir)
    print(f"raw rows: {len(raw):,} (dropped {n_dupes:,} duplicate vehicle/time rows)")
    raw["vehicle_id_raw"] = raw["vehicle_id"]
    raw = assign_physical_ids(raw)
    n_reused = int(raw["vehicle_id"].str.contains(r"\.").groupby(raw["vehicle_id"]).any().sum())
    print(f"physical vehicles: {raw['vehicle_id'].nunique():,} ({n_reused:,} from reused IDs)")

    smoothed = smooth_trajectories(raw)
    print(f"smoothed rows: {len(smoothed):,}")
    led = attach_leaders(smoothed)
    agree = (led["leader_vehicle_raw"] == led["preceding"]).mean()
    print(f"rows with a leader: {len(led):,}; leader matches NGSIM 'preceding' in {agree:.1%} of rows")
    full = cut_pairs(led)
    table = full[list(COLUMNS)].astype(COLUMNS).sort_values(["pair_id", "t_s"], ignore_index=True)
    validate(table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {args.out.resolve()}")
    table.to_csv(args.out, index=False, float_format="%.4f")

    lane_rows = full["lane_id"].value_counts().sort_index()
    meta = {
        "source": "NGSIM US-101 vehicle trajectories, data.transportation.gov (dataset 8ect-6jqj), CC BY-SA 3.0",
        "version": 1,
        "lanes": [1, 2, 3, 4, 5],
        "global_time_ms_range": [int(raw["global_time"].min()), int(raw["global_time"].max())],
        "built": date.today().isoformat(),
        "raw_rows": int(len(raw)),
        "duplicate_rows_dropped": int(n_dupes),
        "rows": len(table),
        "pairs": int(table["pair_id"].nunique()),
        "drivers": int(table["driver_id"].nunique()),
        "rows_per_group": {int(k): int(v) for k, v in table["driver_group"].value_counts().sort_index().items()},
        "rows_per_lane": {int(k): int(v) for k, v in lane_rows.items()},
        "leader_matches_ngsim_preceding": round(float(agree), 4),
        "notes": [
            "The portal copy has exact duplicate rows (every column identical), all between 08:06 and 08:16; one copy of each is kept.",
            "Main lanes 1-5 only. Lane 6 (auxiliary) and lanes 7-8 (ramps) are left out because most movement there is merging, not car following.",
            "Same processing as the sample: ft to m, Savitzky-Golay smoothing "
            f"({SMOOTH_WINDOW} rows, order {SMOOTH_ORDER}), position re-integrated from speed, accel = forward difference of speed.",
            "gap_m = leader rear bumper - follower front bumper, using the leader's real length (NGSIM v_length).",
            "Leader = next car ahead in the same lane at the same time; a pair ends when the leader changes or either car leaves lanes 1-5.",
            f"A vehicle_id absent for more than {REUSE_GAP_MS // 1000} s is treated as a new car (NGSIM reuses IDs); its driver_id gets a suffix, e.g. 1234.1.",
            f"Pairs shorter than {MIN_PAIR_ROWS * DT:.0f} s or with any gap <= 0 are dropped.",
            "driver_group: drivers shuffled with seed 517 and dealt into 5 groups.",
        ],
    }
    meta_path = args.out.with_suffix(".meta.json")
    print(f"writing {meta_path.resolve()}")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("rows", "pairs", "drivers", "rows_per_group", "rows_per_lane")}, indent=2))


if __name__ == "__main__":
    main()
