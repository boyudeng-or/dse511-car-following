"""The shared scorer (lane D): a model that drives exactly like the data must score zero."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import wasserstein_distance

from carfollow import (
    DT,
    HISTORY_STEPS,
    IDM,
    KeepSpeed,
    advance,
    compare,
    evaluate,
    history_at,
    iter_pairs,
    load_table,
    replay,
    simulate,
)
from carfollow.data import FULL_PATH, SAMPLE_PATH
from carfollow.evaluate import INTERVAL_KEYS, SCORE_KEYS, _wasserstein_by_driver
from carfollow.schema import COLUMNS


def driven_table(model, n_pairs=10, n_rows=200, seed=0) -> pd.DataFrame:
    """Pairs in the shared format whose follower is driven by `model` (memoryless) behind a wavy leader."""
    rng = np.random.default_rng(seed)
    leader_length = 4.5
    frames = []
    for p in range(n_pairs):
        t = np.arange(n_rows) * DT
        lead_a = 1.2 * np.sin(2 * np.pi * t / rng.uniform(8.0, 20.0) + rng.uniform(0.0, 2 * np.pi))
        lead_x, lead_v = np.empty(n_rows), np.empty(n_rows)
        lead_x[0], lead_v[0] = 100.0, rng.uniform(10.0, 15.0)
        x, v, a = np.empty(n_rows), np.empty(n_rows), np.empty(n_rows)
        x[0], v[0] = lead_x[0] - leader_length - rng.uniform(15.0, 30.0), lead_v[0]
        gap, closing = np.empty(n_rows), np.empty(n_rows)
        for i in range(n_rows):
            gap[i], closing[i] = lead_x[i] - x[i] - leader_length, v[i] - lead_v[i]
            a[i] = model.accel(np.tile([gap[i], v[i], closing[i]], (HISTORY_STEPS, 1)))
            if i + 1 < n_rows:
                x[i + 1], v[i + 1] = advance(x[i], v[i], a[i])
                lead_x[i + 1], lead_v[i + 1] = advance(lead_x[i], lead_v[i], lead_a[i])
        frames.append(
            pd.DataFrame(
                {
                    "pair_id": f"{p}->L{p}@0",
                    "driver_id": str(p),
                    "leader_id": f"L{p}",
                    "driver_group": p % 5,
                    "global_time_ms": np.arange(n_rows) * 100,
                    "t_s": t,
                    "position_m": x,
                    "leader_position_m": lead_x,
                    "gap_m": gap,
                    "speed_mps": v,
                    "leader_speed_mps": lead_v,
                    "closing_speed_mps": closing,
                    "accel_mps2": a,
                    "leader_accel_mps2": lead_a,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).astype(COLUMNS)


class Constant:
    """Always the same acceleration."""

    def __init__(self, accel, name="constant"):
        self.value, self.name = accel, name

    def accel(self, history):
        return float(self.value)


class Recorder:
    """Returns zero and remembers every history it was shown."""

    name = "recorder"

    def __init__(self):
        self.seen = []

    def accel(self, history):
        self.seen.append(np.array(history))
        return 0.0


class BatchIDM(IDM):
    """IDM with the optional accel_batch speed-up."""

    def accel_batch(self, histories):
        return np.array([IDM.accel(self, h) for h in histories])


class WrongBatchIDM(IDM):
    def accel_batch(self, histories):
        return np.array([IDM.accel(self, h) + 1.0 for h in histories])


@pytest.fixture(scope="module")
def idm_table():
    return driven_table(IDM.textbook())


def test_the_true_model_scores_zero(idm_table):
    scores = evaluate(IDM.textbook(), idm_table)
    assert scores["step_rmse"] < 1e-12
    assert scores["drive_gap_rmse"] < 1e-9
    assert scores["drive_crashes"] == 0
    assert scores["spread_distance"] < 1e-9


def test_another_model_scores_worse(idm_table):
    scores = evaluate(KeepSpeed(), idm_table)
    assert scores["step_rmse"] > 0.05
    assert scores["drive_gap_rmse"] > 0.5
    assert scores["spread_distance"] > 0.01


def test_every_score_has_its_key_and_interval(idm_table):
    scores = evaluate(IDM.calibrated(), idm_table)
    for key in SCORE_KEYS:
        assert key in scores
    for key in INTERVAL_KEYS:
        assert scores[f"{key}_lo"] <= scores[f"{key}_hi"]
    assert scores["n_drives"] == idm_table["pair_id"].nunique()
    assert scores["n_rows"] == sum(len(p) - HISTORY_STEPS for p in iter_pairs(idm_table))
    assert all(np.isfinite(v) for v in scores.values())


def test_scores_repeat_with_the_same_seed(idm_table):
    assert evaluate(IDM.calibrated(), idm_table) == evaluate(IDM.calibrated(), idm_table)
    no_boot = evaluate(IDM.calibrated(), idm_table, n_boot=0)
    assert np.isnan(no_boot["step_rmse_lo"]) and np.isfinite(no_boot["step_rmse"])


def test_replay_reproduces_the_real_gaps(idm_table):
    roll = replay(idm_table)
    driven = np.isfinite(roll.sim_gap)
    assert np.array_equal(driven, np.isfinite(roll.real_gap))
    assert np.abs(roll.sim_gap[driven] - roll.real_gap[driven]).max() < 1e-9
    assert not roll.crashed.any()


def test_the_model_sees_the_same_history_as_history_at(idm_table):
    recorder = Recorder()
    evaluate(recorder, idm_table, n_boot=0)
    pairs = list(iter_pairs(idm_table))
    first = pairs[0]
    n_step = len(first) - HISTORY_STEPS
    for j in range(n_step):
        assert np.array_equal(recorder.seen[j], history_at(first, HISTORY_STEPS - 1 + j))
    # The first Drive step of every pair starts from its real history at row 29.
    n_rows = sum(len(p) - HISTORY_STEPS for p in pairs)
    for j, pair in enumerate(pairs):
        assert np.array_equal(recorder.seen[n_rows + j], history_at(pair, HISTORY_STEPS - 1))


def test_a_crash_is_counted_and_ends_the_drive(idm_table):
    full_throttle = Constant(3.0, "full_throttle")
    roll = simulate(full_throttle, idm_table)
    assert roll.crashed.all()
    for sim in roll.sim_gap:
        driven = sim[np.isfinite(sim)]
        assert driven[-1] <= 0.0 and (driven[:-1] > 0.0).all()
        assert np.isnan(sim[len(driven):]).all()
    assert evaluate(full_throttle, idm_table, n_boot=0)["drive_crashes"] == len(roll.crashed)


def test_rollouts_last_at_most_30_s(idm_table):
    roll = simulate(KeepSpeed(), idm_table, steps=50)
    assert roll.sim_gap.shape == (idm_table["pair_id"].nunique(), 51)
    long_table = driven_table(IDM.textbook(), n_pairs=2, n_rows=400)
    assert simulate(IDM.textbook(), long_table).sim_gap.shape == (2, 301)


def test_accel_batch_gives_the_same_scores():
    table = driven_table(IDM.calibrated(), n_pairs=6)
    one_by_one = evaluate(IDM.textbook(), table)
    batched = evaluate(BatchIDM(name="idm_textbook"), table)
    for key, value in one_by_one.items():
        assert np.isclose(batched[key], value, rtol=1e-12, atol=1e-12), key


def test_an_accel_batch_that_disagrees_is_rejected(idm_table):
    with pytest.raises(ValueError, match="disagrees"):
        evaluate(WrongBatchIDM(), idm_table)
    with pytest.raises(ValueError, match="disagrees"):
        simulate(WrongBatchIDM(), idm_table)


def test_a_non_finite_acceleration_is_rejected(idm_table):
    with pytest.raises(ValueError, match="non-finite"):
        evaluate(Constant(np.nan), idm_table)


def test_a_table_that_breaks_the_agreement_is_rejected(idm_table):
    with pytest.raises(ValueError):
        evaluate(KeepSpeed(), idm_table.drop(columns="gap_m"))
    with pytest.raises(ValueError):
        evaluate(KeepSpeed(), idm_table.iloc[:0])


def test_compare_gives_one_row_per_model(idm_table):
    table = compare([KeepSpeed(), IDM.textbook()], idm_table, n_boot=0)
    assert list(table.index) == ["keep_speed", "idm_textbook"]
    assert table.loc["idm_textbook", "step_rmse"] < table.loc["keep_speed", "step_rmse"]


def test_wasserstein_matches_scipy():
    rng = np.random.default_rng(1)
    sim, real = rng.gamma(2.0, 1.0, 500), rng.gamma(2.5, 0.8, 500)
    owner = rng.integers(0, 40, 500)
    distance = _wasserstein_by_driver(sim, real, owner)
    assert np.isclose(distance(np.ones(40)), wasserstein_distance(sim, real))
    weights = rng.integers(0, 3, 40).astype(float)
    assert np.isclose(distance(weights), wasserstein_distance(sim, real, weights[owner], weights[owner]))


REAL_PATH = SAMPLE_PATH if SAMPLE_PATH.exists() else FULL_PATH
needs_data = pytest.mark.skipif(not REAL_PATH.exists(), reason="no pair table under data/ (sample or full)")


@pytest.fixture(scope="module")
def real_pairs():
    table = load_table(REAL_PATH)
    first = table["pair_id"].drop_duplicates().iloc[:300]
    return table[table["pair_id"].isin(first)].reset_index(drop=True)


@needs_data
def test_replay_matches_the_real_data(real_pairs):
    roll = replay(real_pairs)
    driven = np.isfinite(roll.sim_gap)
    assert np.abs(roll.sim_gap[driven] - roll.real_gap[driven]).max() < 0.05
    assert not roll.crashed.any()


@needs_data
def test_baselines_score_on_real_data(real_pairs):
    table = compare([KeepSpeed(), IDM.textbook(), IDM.calibrated()], real_pairs, n_boot=20)
    assert np.isfinite(table[list(SCORE_KEYS)].to_numpy(dtype=float)).all()
    assert (table["drive_crashes"] <= table["n_drives"]).all()
