"""Lane C: the GRU plug must follow the shared contract (accel and accel_batch agree, save/load round-trips)."""

import numpy as np
import pytest
import torch

from carfollow.baselines import MAX_BRAKE_MPS2
from carfollow.plug import check_model
from lanes.C_memory.gru import GRUDriver, GRUFollower, load_model, save_model

MEAN = np.array([30.0, 10.0, 0.0])
STD = np.array([20.0, 5.0, 2.0])


def random_histories(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(2.0, 80.0, (n, 30)),   # gap_m
            rng.uniform(0.0, 35.0, (n, 30)),   # speed_mps
            rng.uniform(-5.0, 5.0, (n, 30)),   # closing_speed_mps
        ]
    ).reshape(n, 30, 3)


@pytest.fixture
def driver():
    torch.manual_seed(42)
    return GRUDriver(GRUFollower(), MEAN, STD, lookback=30, name="gru_w30_test")


def test_driver_fits_the_plug(driver):
    check_model(driver)


def test_accel_and_accel_batch_agree(driver):
    histories = random_histories(20)
    one_by_one = np.array([driver.accel(h) for h in histories])
    together = driver.accel_batch(histories)
    assert together.shape == (20,)
    assert np.allclose(one_by_one, together, atol=1e-5)


def test_accel_is_clipped_to_the_brake_limit(driver):
    histories = random_histories(200)
    a = driver.accel_batch(histories)
    assert a.min() >= -MAX_BRAKE_MPS2 and a.max() <= MAX_BRAKE_MPS2


def test_short_lookback_ignores_older_rows():
    torch.manual_seed(42)
    driver = GRUDriver(GRUFollower(), MEAN, STD, lookback=2, name="gru_w2_test")
    histories = random_histories(10)
    changed = histories.copy()
    changed[:, :-2, :] = random_histories(10, seed=99)[:, :-2, :]   # rewrite everything except the last 2 rows
    assert np.allclose(driver.accel_batch(histories), driver.accel_batch(changed), atol=1e-6)


def test_save_and_load_round_trip(driver, tmp_path):
    path = tmp_path / "gru_test.pt"
    save_model(driver, path)
    loaded = load_model(path)
    histories = random_histories(10)
    assert loaded.name == driver.name
    assert np.allclose(loaded.accel_batch(histories), driver.accel_batch(histories), atol=1e-6)
