"""Loading the shared table and splitting it by driver group."""

from pathlib import Path
from typing import Iterator

import pandas as pd

from .schema import COLUMNS, validate

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "data" / "sample" / "pairs_sample.csv"


def load_table(path: Path = SAMPLE_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download pairs_sample.csv from the Teams channel into data/sample/."
        )
    df = pd.read_csv(path, dtype={"pair_id": "string", "driver_id": "string", "leader_id": "string"})
    df = df.astype(COLUMNS).sort_values(["pair_id", "t_s"], ignore_index=True)
    validate(df)
    return df


def split_by_group(df: pd.DataFrame, test_group: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on four driver groups, test on the fifth. No driver appears on both sides."""
    test = df["driver_group"] == test_group
    return df[~test].reset_index(drop=True), df[test].reset_index(drop=True)


def iter_pairs(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per pair_id, rows sorted by time."""
    for _, pair in df.groupby("pair_id", sort=True):
        yield pair.sort_values("t_s").reset_index(drop=True)
