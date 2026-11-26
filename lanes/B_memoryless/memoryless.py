"""Lane B: memoryless models wrapped in the shared plug (carfollow/plug.py).

The wrapper and its feature builders live here, not in the notebook, so a
saved model can be loaded anywhere in the repo:

    from lanes.B_memoryless.memoryless import load_model
    model = load_model("lanes/B_memoryless/outputs/gradient_boosting_physics_full.joblib")
    model.accel(history)   # history: (30, 3) = gap_m, speed_mps, closing_speed_mps

Run from the repository root (or put it on sys.path) so `lanes` and
`carfollow` are importable.
"""

from pathlib import Path

import joblib
import numpy as np

from carfollow import IDM
from carfollow.baselines import MAX_BRAKE_MPS2

PHYSICS_FEATURES = [
    "gap_m", "speed_mps", "closing_speed_mps",
    "inv_gap", "inv_headway", "inv_ttc", "brake_need", "idm_textbook",
]
# Monotonic constraints for Gradient Boosting, one per PHYSICS_FEATURES entry.
PHYSICS_MONOTONIC = [1, -1, -1, -1, -1, -1, -1, 1]

_idm = IDM.textbook()


def raw_features(states):
    """states: (n, 3) array of [gap_m, speed_mps, closing_speed_mps]."""
    return np.asarray(states, dtype=float)


def physics_features(states):
    """The 3 raw inputs plus 5 features computed from the same row only (still memoryless)."""
    states = np.asarray(states, dtype=float)
    gap = np.maximum(states[:, 0], 0.1)
    speed, closing = states[:, 1], states[:, 2]
    s_star = _idm.s0 + np.maximum(
        0.0, speed * _idm.T + speed * closing / (2.0 * np.sqrt(_idm.a_max * _idm.b))
    )
    idm_accel = _idm.a_max * (1.0 - (speed / _idm.v0) ** _idm.delta - (s_star / gap) ** 2)
    return np.column_stack([
        states,
        1.0 / gap,
        speed / gap,
        closing / gap,
        closing * np.abs(closing) / (2.0 * gap),
        np.clip(idm_accel, -MAX_BRAKE_MPS2, _idm.a_max),
    ])


FEATURE_BUILDERS = {"raw": raw_features, "physics": physics_features}


class MemorylessAccelerationModel:
    """Lane B plug: reads only history[-1] and returns one acceleration in m/s^2."""

    def __init__(self, name, estimator, feature_set):
        self.name = name
        self.estimator = estimator
        self.feature_set = feature_set

    def accel_batch(self, histories):
        """histories: (n, 30, 3) -> (n,) accelerations. Optional; speeds up rollouts."""
        states = np.asarray(histories, dtype=float)[:, -1, :]
        X = FEATURE_BUILDERS[self.feature_set](states)
        return np.clip(self.estimator.predict(X), -MAX_BRAKE_MPS2, MAX_BRAKE_MPS2)

    def accel(self, history):
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] == 0:
            raise ValueError("history must be a 2D array with 3 columns: [gap_m, speed_mps, closing_speed_mps]")
        return float(self.accel_batch(history[None])[0])


def save_model(model, path):
    joblib.dump(model, Path(path))


def load_model(path):
    return joblib.load(Path(path))
