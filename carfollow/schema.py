"""Agreement 1: the shared table.

One row per following car per 0.1 s. All units are SI (m, m/s, m/s^2).
"""

import pandas as pd

DT = 0.1
HISTORY_STEPS = 30
HISTORY_FEATURES = ("gap_m", "speed_mps", "closing_speed_mps")
N_GROUPS = 5

COLUMNS = {
    "pair_id": "string",            # one continuous stretch of the same driver behind the same leader
    "driver_id": "string",          # the following car
    "leader_id": "string",          # the car directly ahead in the same lane
    "driver_group": "int64",        # 0..4, used for the driver-grouped split
    "global_time_ms": "int64",
    "t_s": "float64",               # seconds since the start of this pair_id
    "position_m": "float64",        # follower front bumper along the road
    "leader_position_m": "float64",
    "gap_m": "float64",             # leader rear bumper minus follower front bumper
    "speed_mps": "float64",
    "leader_speed_mps": "float64",
    "closing_speed_mps": "float64",  # speed_mps - leader_speed_mps; positive = getting closer
    "accel_mps2": "float64",        # the target the models predict
    "leader_accel_mps2": "float64",
}


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError listing every way `df` breaks the agreement."""
    problems = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    else:
        if df[list(COLUMNS)].isna().any().any():
            problems.append("table contains NaN")
        if not df["driver_group"].between(0, N_GROUPS - 1).all():
            problems.append(f"driver_group must be in 0..{N_GROUPS - 1}")
        if (df["gap_m"] <= 0).any():
            problems.append("gap_m must be positive")
        if (df["speed_mps"] < 0).any():
            problems.append("speed_mps must be non-negative")
        steps = df.sort_values(["pair_id", "t_s"]).groupby("pair_id")["t_s"].diff().dropna()
        if not ((steps - DT).abs() < 1e-6).all():
            problems.append(f"rows inside a pair_id must be exactly {DT} s apart")
        if df.groupby("pair_id")["driver_group"].nunique().gt(1).any():
            problems.append("a pair_id must belong to a single driver_group")
    if problems:
        raise ValueError("; ".join(problems))
