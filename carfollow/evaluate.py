"""Agreement 3: the one scorer everyone uses. Owned by lane D.

    evaluate(model, drivers) -> dict of scores

`drivers` is a table in the shared format (usually one held-out driver group).
Every model is scored by this function only, so the comparison stays fair.
"""

import pandas as pd

from .plug import Model

SCORE_KEYS = (
    "step_rmse",        # Step: error of the predicted acceleration for the next 0.1 s
    "drive_gap_rmse",   # Drive: gap error after a 30 s closed-loop rollout behind the real leader
    "drive_crashes",    # Drive: number of rollouts where the gap reached zero
    "spread_distance",  # Spread: distance between simulated and real time-headway distributions
)


def evaluate(model: Model, drivers: pd.DataFrame) -> dict[str, float]:
    raise NotImplementedError("Lane D: implement the Step, Drive and Spread scores here.")
