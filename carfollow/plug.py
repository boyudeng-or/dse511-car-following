"""Agreement 2: the plug every model fits.

    accel(history) -> acceleration (m/s^2)

`history` is a float array of shape (HISTORY_STEPS, 3) = the last 3 s,
oldest row first, newest row last. Columns, in order:
gap_m, speed_mps, closing_speed_mps.

The returned value is the follower's acceleration at the newest row; the
simulator uses it to move the car forward by 0.1 s. Memoryless models and
IDM read only history[-1].
"""

from typing import Protocol

import numpy as np
import pandas as pd

from .schema import HISTORY_FEATURES, HISTORY_STEPS


class Model(Protocol):
    name: str

    def accel(self, history: np.ndarray) -> float: ...


def history_at(pair: pd.DataFrame, i: int) -> np.ndarray:
    """History ending at row `i` of one pair (rows sorted by t_s). Needs i >= HISTORY_STEPS - 1."""
    if i < HISTORY_STEPS - 1:
        raise IndexError(f"need at least {HISTORY_STEPS} rows before predicting; got index {i}")
    rows = pair.iloc[i - HISTORY_STEPS + 1 : i + 1]
    return rows[list(HISTORY_FEATURES)].to_numpy(dtype=float)


def check_model(model: Model, n_trials: int = 50, seed: int = 0) -> None:
    """Fail loudly if `model` does not fit the plug."""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        history = np.column_stack(
            [
                rng.uniform(2.0, 80.0, HISTORY_STEPS),   # gap_m
                rng.uniform(0.0, 35.0, HISTORY_STEPS),   # speed_mps
                rng.uniform(-5.0, 5.0, HISTORY_STEPS),   # closing_speed_mps
            ]
        )
        a = model.accel(history)
        if not isinstance(a, (float, np.floating)) or not np.isfinite(a):
            raise TypeError(f"{getattr(model, 'name', model)}.accel must return a finite float, got {a!r}")
