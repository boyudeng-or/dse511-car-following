"""Agreement 3: the one scorer everyone uses. Owned by lane D.

    evaluate(model, drivers) -> dict of scores

`drivers` is a table in the shared format (usually one held-out driver group).
Every model is scored by this function only, so the comparison stays fair.

Step    Predict the next 0.1 s from the real last 3 s. Every row with a full
        3 s history whose next step is still in the pair (rows 29 .. n-2, the
        rows lanes B and C report on) is compared with the real accel_mps2.

Drive   Drive behind the real leader. One rollout per pair: start from the
        real state at row 29 with the real 3 s history; then, for up to 30 s,
        ask the model for an acceleration, move the car with
        kinematics.advance, and append the new gap, speed and closing speed
        to the history. A rollout ends early when the gap reaches zero (a
        crash). The gap error pools every step of every rollout. Replaying the
        real accelerations (`replay`) reproduces the real gaps.

Spread  Reproduce real headways. Wasserstein-1 distance, in seconds, between
        the simulated and the real time headway (gap / speed) over the same
        rollout steps, counting steps where both cars move faster than 1 m/s.

Error bars are 95% bootstrap intervals over drivers (the unit the groups are
split by), returned as `<score>_lo` and `<score>_hi`. The seed is fixed, so
every model is resampled the same way.

Optional speed-up: a model may also define `accel_batch(histories)`, taking an
(n, 30, 3) array and returning n accelerations. The scorer then calls it
instead of `accel`, after checking on real histories that both agree.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .kinematics import advance
from .plug import Model
from .schema import HISTORY_FEATURES, HISTORY_STEPS, validate

SCORE_KEYS = (
    "step_rmse",        # Step: RMSE of the predicted acceleration for the next 0.1 s (m/s^2)
    "drive_gap_rmse",   # Drive: RMSE between simulated and real gap over every rollout step (m)
    "drive_crashes",    # Drive: number of rollouts where the gap reached zero
    "spread_distance",  # Spread: Wasserstein-1 distance between simulated and real time headways (s)
)
# Scores that come with a 95% interval, returned as <key>_lo and <key>_hi.
INTERVAL_KEYS = ("step_rmse", "drive_gap_rmse", "drive_crash_rate", "spread_distance")

DRIVE_STEPS = 300              # 30 s at 0.1 s
HEADWAY_MIN_SPEED_MPS = 1.0    # gap / speed blows up near standstill
N_BOOT = 200
SEED = 511

FIRST = HISTORY_STEPS - 1      # first row of a pair with a full 3 s history
_OFFSETS = np.arange(-FIRST, 1)
_CHUNK = 50_000                # histories per model call in the Step score
_BATCH_CHECKS = 25


def evaluate(
    model: Model,
    drivers: pd.DataFrame,
    *,
    steps: int = DRIVE_STEPS,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, float]:
    """Every score for one model on one table, with 95% intervals (see the module docstring)."""
    table = _Table(drivers)
    rows = table.score_rows()
    if rows.size == 0:
        raise ValueError(f"no pair in `drivers` has more than {HISTORY_STEPS} rows; nothing to score")
    _check_batch(model, table, seed)
    n_drivers = len(table.driver_names)

    # Step
    pred = np.concatenate(
        [_accel_many(model, table.histories(rows[i : i + _CHUNK])) for i in range(0, rows.size, _CHUNK)]
    )
    real = table.accel[rows]
    err = pred - real
    step_sse = np.bincount(table.driver[rows], weights=err**2, minlength=n_drivers)
    step_n = np.bincount(table.driver[rows], minlength=n_drivers)

    # Drive
    drives = _Drives(table, steps)
    roll = drives.run(lambda history, idx, k: _accel_many(model, history))
    gap_err = roll.sim_gap[:, 1:] - roll.real_gap[:, 1:]
    driven = np.isfinite(gap_err)
    gap_sse = np.bincount(drives.driver, weights=np.nansum(gap_err**2, axis=1), minlength=n_drivers)
    gap_n = np.bincount(drives.driver, weights=driven.sum(axis=1), minlength=n_drivers)
    crashes = np.bincount(drives.driver, weights=roll.crashed, minlength=n_drivers)
    n_drives = np.bincount(drives.driver, minlength=n_drivers)

    # Spread
    sim_speed, real_speed = roll.sim_speed[:, 1:], roll.real_speed[:, 1:]
    moving = driven & (sim_speed > HEADWAY_MIN_SPEED_MPS) & (real_speed > HEADWAY_MIN_SPEED_MPS)
    spread = _wasserstein_by_driver(
        roll.sim_gap[:, 1:][moving] / sim_speed[moving],
        roll.real_gap[:, 1:][moving] / real_speed[moving],
        np.broadcast_to(drives.driver[:, None], moving.shape)[moving],
    )

    # Each score as a function of driver weights: all ones for the estimate, bootstrap counts for the interval.
    stats = {
        "step_rmse": lambda w: np.sqrt((w @ step_sse) / (w @ step_n)),
        "drive_gap_rmse": lambda w: np.sqrt((w @ gap_sse) / (w @ gap_n)),
        "drive_crash_rate": lambda w: (w @ crashes) / (w @ n_drives),
        "spread_distance": spread,
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        point = {key: float(f(np.ones(n_drivers))) for key, f in stats.items()}
        interval = _bootstrap(stats, n_drivers, n_boot, seed)

    def with_interval(key: str) -> dict[str, float]:
        return {key: point[key], f"{key}_lo": interval[key][0], f"{key}_hi": interval[key][1]}

    return {
        **with_interval("step_rmse"),
        "step_mae": float(np.mean(np.abs(err))),
        "step_r2": float(1.0 - np.sum(err**2) / np.sum((real - real.mean()) ** 2)),
        **with_interval("drive_gap_rmse"),
        "drive_crashes": int(roll.crashed.sum()),
        **with_interval("drive_crash_rate"),
        **with_interval("spread_distance"),
        "n_rows": int(rows.size),
        "n_drives": int(roll.crashed.size),
        "n_drivers": n_drivers,
    }


def compare(models: Iterable[Model], drivers: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """evaluate() for several models on the same table: the comparison table, one row per model."""
    rows = [{"model": _name(m), **evaluate(m, drivers, **kwargs)} for m in models]
    return pd.DataFrame(rows).set_index("model")


@dataclass
class Rollouts:
    """Closed-loop drives from `simulate` or `replay`, one row per pair.

    Column k of each (pair, step) array is k * 0.1 s after the start, which is
    row 29 of the pair. Simulated entries after a drive ended (pair over, or
    crash) are NaN; real entries are NaN only after the pair is over.
    """

    pair_ids: np.ndarray
    driver_ids: np.ndarray
    sim_gap: np.ndarray
    real_gap: np.ndarray
    sim_speed: np.ndarray
    real_speed: np.ndarray
    crashed: np.ndarray


def simulate(model: Model, drivers: pd.DataFrame, steps: int = DRIVE_STEPS) -> Rollouts:
    """Let `model` drive every pair of `drivers` behind its real leader: the Drive rollouts."""
    table = _Table(drivers)
    _check_batch(model, table, SEED)
    return _Drives(table, steps).run(lambda history, idx, k: _accel_many(model, history))


def replay(drivers: pd.DataFrame, steps: int = DRIVE_STEPS) -> Rollouts:
    """The same rollouts driven by the real accelerations: a sanity check that must match the data."""
    drives = _Drives(_Table(drivers), steps)
    return drives.run(lambda history, idx, k: drives.accel[idx, k])


class _Table:
    """The shared table as arrays sorted by pair and time, with each pair's rows [start, end)."""

    def __init__(self, drivers: pd.DataFrame):
        if len(drivers) == 0:
            raise ValueError("`drivers` is empty")
        validate(drivers)
        df = drivers.sort_values(["pair_id", "t_s"], kind="mergesort", ignore_index=True)
        self.feats = df[list(HISTORY_FEATURES)].to_numpy(dtype=float)
        self.accel = df["accel_mps2"].to_numpy(dtype=float)
        self.position = df["position_m"].to_numpy(dtype=float)
        self.gap = df["gap_m"].to_numpy(dtype=float)
        self.speed = df["speed_mps"].to_numpy(dtype=float)
        self.leader_speed = df["leader_speed_mps"].to_numpy(dtype=float)
        pair = df["pair_id"].to_numpy(dtype=object)
        self.starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
        self.ends = np.r_[self.starts[1:], len(df)]
        self.pair_ids = pair[self.starts]
        codes, names = pd.factorize(df["driver_id"])
        self.driver = codes                       # driver of each row, as 0..n_drivers-1
        self.driver_names = np.asarray(names, dtype=object)

    def score_rows(self) -> np.ndarray:
        """Rows with a full real 3 s history whose next step is still in the pair."""
        parts = [np.arange(s + FIRST, e - 1) for s, e in zip(self.starts, self.ends)]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)

    def histories(self, rows: np.ndarray) -> np.ndarray:
        """The (len(rows), 30, 3) histories ending at `rows`, as history_at() cuts them."""
        return self.feats[np.asarray(rows)[:, None] + _OFFSETS]


class _Drives:
    """One rollout per pair with more than HISTORY_STEPS rows, laid out as (pair, step) grids."""

    def __init__(self, table: _Table, steps: int):
        if steps < 1:
            raise ValueError("steps must be at least 1")
        keep = np.flatnonzero(table.ends - table.starts > HISTORY_STEPS)
        start = table.starts[keep] + FIRST
        self.steps = steps
        self.length = np.minimum(table.ends[keep] - 1 - start, steps)   # steps each rollout can take
        self.pair_ids = table.pair_ids[keep]
        self.driver = table.driver[start]
        self.history = table.histories(start)
        col = np.arange(steps + 1)
        inside = col <= self.length[:, None]
        rows = np.where(inside, start[:, None] + col, 0)

        def grid(values: np.ndarray) -> np.ndarray:
            return np.where(inside, values[rows], np.nan)

        self.position = grid(table.position)
        self.gap = grid(table.gap)
        self.speed = grid(table.speed)
        self.leader_speed = grid(table.leader_speed)
        self.accel = grid(table.accel)
        self.driver_ids = table.driver_names[self.driver]

    def run(self, accel_for: Callable[[np.ndarray, np.ndarray, int], np.ndarray]) -> Rollouts:
        """accel_for(histories, rollout indices, step) gives the accelerations for one step."""
        n, width = self.gap.shape
        x, v = self.position[:, 0].copy(), self.speed[:, 0].copy()
        history = self.history.copy()
        sim_gap = np.full((n, width), np.nan)
        sim_speed = np.full((n, width), np.nan)
        sim_gap[:, 0], sim_speed[:, 0] = self.gap[:, 0], v
        crashed = np.zeros(n, dtype=bool)
        for k in range(self.steps):
            idx = np.flatnonzero(~crashed & (k < self.length))
            if idx.size == 0:
                break
            accel = accel_for(history[idx], idx, k)
            moved = [advance(p, s, a) for p, s, a in zip(x[idx].tolist(), v[idx].tolist(), accel.tolist())]
            x[idx], v[idx] = np.array(moved).T
            # Same leader as in the data, so the gap changes only by how far our car is from the real one.
            gap = self.gap[idx, k + 1] + (self.position[idx, k + 1] - x[idx])
            sim_gap[idx, k + 1] = gap
            sim_speed[idx, k + 1] = v[idx]
            newest = np.column_stack([gap, v[idx], v[idx] - self.leader_speed[idx, k + 1]])
            history[idx] = np.concatenate([history[idx, 1:], newest[:, None, :]], axis=1)
            crashed[idx[gap <= 0.0]] = True
        return Rollouts(
            pair_ids=self.pair_ids,
            driver_ids=self.driver_ids,
            sim_gap=sim_gap,
            real_gap=self.gap.copy(),
            sim_speed=sim_speed,
            real_speed=self.speed.copy(),
            crashed=crashed,
        )


def _name(model: Model) -> str:
    return getattr(model, "name", type(model).__name__)


def _accel_many(model: Model, histories: np.ndarray) -> np.ndarray:
    """One acceleration per history: one accel_batch call if the model has it, else accel per history."""
    batch = getattr(model, "accel_batch", None)
    if batch is not None:
        accel = np.asarray(batch(histories), dtype=float).reshape(-1)
    else:
        accel = np.fromiter((model.accel(h) for h in histories), dtype=float, count=len(histories))
    if accel.shape != (len(histories),):
        raise ValueError(f"{_name(model)}: got {accel.size} accelerations for {len(histories)} histories")
    if not np.isfinite(accel).all():
        raise ValueError(f"{_name(model)}: returned a non-finite acceleration")
    return accel


def _check_batch(model: Model, table: _Table, seed: int) -> None:
    """A model's accel_batch must agree with its accel (the plug) on real histories."""
    batch = getattr(model, "accel_batch", None)
    rows = table.score_rows()
    if batch is None or rows.size == 0:
        return
    rng = np.random.default_rng([seed, 1])   # its own stream: the bootstrap draws stay the same for every model
    histories = table.histories(rng.choice(rows, min(_BATCH_CHECKS, rows.size), replace=False))
    one_by_one = np.array([model.accel(h) for h in histories], dtype=float)
    together = np.asarray(batch(histories), dtype=float).reshape(-1)
    if together.shape != one_by_one.shape or not np.allclose(together, one_by_one, rtol=1e-4, atol=1e-4):
        raise ValueError(f"{_name(model)}: accel_batch disagrees with accel on real histories")


def _wasserstein_by_driver(sim: np.ndarray, real: np.ndarray, owner: np.ndarray) -> Callable[[np.ndarray], float]:
    """Wasserstein-1 distance between paired samples, as a function of driver weights.

    sim[i] and real[i] come from the same rollout step of driver owner[i], so a
    driver's weight counts in both samples alike. All ones gives the plain
    distance (scipy.stats.wasserstein_distance); bootstrap counts give a resample.
    """
    values = np.concatenate([sim, real])
    order = np.argsort(values, kind="stable")
    widths = np.diff(values[order])
    from_sim = order < sim.size
    owner = np.concatenate([owner, owner])[order]

    def distance(weights: np.ndarray) -> float:
        w = weights[owner]
        total = w.sum() / 2.0          # the weight of each sample
        if total == 0:
            return np.nan
        cdf_gap = np.cumsum(np.where(from_sim, w, -w))[:-1]
        return float(np.abs(cdf_gap) @ widths / total)

    return distance


def _bootstrap(stats: dict, n_drivers: int, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% percentile intervals from resampling drivers with replacement."""
    if n_boot < 1:
        return {key: (np.nan, np.nan) for key in stats}
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(stats)))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_drivers, n_drivers), minlength=n_drivers).astype(float)
        draws[b] = [f(w) for f in stats.values()]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return {key: (float(lo[j]), float(hi[j])) for j, key in enumerate(stats)}
