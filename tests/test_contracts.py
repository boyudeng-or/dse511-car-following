import numpy as np
import pytest

from carfollow import (
    DT,
    HISTORY_STEPS,
    IDM,
    KeepSpeed,
    N_GROUPS,
    advance,
    check_model,
    history_at,
    iter_pairs,
    load_table,
    split_by_group,
)
from carfollow.data import SAMPLE_PATH


def make_history(gap, speed, closing):
    return np.tile([gap, speed, closing], (HISTORY_STEPS, 1)).astype(float)


@pytest.mark.parametrize("model", [KeepSpeed(), IDM.textbook()])
def test_baselines_fit_the_plug(model):
    check_model(model)


def test_idm_speeds_up_on_an_empty_road():
    assert IDM.textbook().accel(make_history(gap=200.0, speed=10.0, closing=0.0)) > 0


def test_idm_brakes_when_closing_in():
    assert IDM.textbook().accel(make_history(gap=8.0, speed=20.0, closing=5.0)) < 0


def test_idm_settles_at_equilibrium_gap():
    idm = IDM.textbook()
    v = 15.0
    s_eq = (idm.s0 + v * idm.T) / np.sqrt(1 - (v / idm.v0) ** idm.delta)
    assert abs(idm.accel(make_history(gap=s_eq, speed=v, closing=0.0))) < 1e-9


needs_sample = pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="data/sample/pairs_sample.csv not downloaded")


@pytest.fixture(scope="module")
def table():
    return load_table()


@needs_sample
def test_sample_follows_the_table_agreement(table):
    assert set(table["driver_group"].unique()) == set(range(N_GROUPS))
    first_pair = next(iter_pairs(table))
    assert np.allclose(np.diff(first_pair["t_s"]), DT)


@needs_sample
def test_history_has_the_agreed_shape(table):
    pair = next(iter_pairs(table))
    assert history_at(pair, HISTORY_STEPS - 1).shape == (HISTORY_STEPS, 3)
    with pytest.raises(IndexError):
        history_at(pair, HISTORY_STEPS - 2)


@needs_sample
def test_replaying_real_accel_reproduces_every_pair(table):
    for pair in iter_pairs(table):
        x, v = pair["position_m"].iloc[0], pair["speed_mps"].iloc[0]
        for a in pair["accel_mps2"].iloc[:-1]:
            x, v = advance(x, v, a)
        assert abs(x - pair["position_m"].iloc[-1]) < 0.05
        assert abs(v - pair["speed_mps"].iloc[-1]) < 0.01


@needs_sample
def test_split_keeps_drivers_apart(table):
    for g in range(N_GROUPS):
        train, test = split_by_group(table, g)
        assert not set(train["driver_id"]) & set(test["driver_id"])
        assert len(train) + len(test) == len(table)
