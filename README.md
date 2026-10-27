# When does a learned driver break?

DSE 511 capstone, Fall 2026. We compare car-following models on real highway
data (NGSIM US-101): a textbook rule (IDM) against machine-learning models,
scored three ways.

| Question | In plain words |
|---|---|
| Q1 | Does the model that best predicts the next 0.1 s also drive best for 30 s and reproduce real headways? |
| Q2 | Does memory (the last 2–3 s) help? |
| Q3 | Does the ranking hold on a new road? (cut first if time is short) |

## Lanes

| Lane | Owner | Folder | Owns |
|---|---|---|---|
| A · Data + IDM | Boyu | `lanes/A_data_idm/`, `carfollow/` | Pair table, 5 driver groups, IDM, one-command rebuild |
| B · ML-1 memoryless | Pantho | `lanes/B_memoryless/` | Linear regression, gradient boosting |
| C · ML-2 with memory | Daniel | `lanes/C_memory/` | Small GRU on the last 3 s |
| D · Evaluation | Mehmed | `lanes/D_evaluation/`, `carfollow/evaluate.py` | Scorer, 30 s simulator, figures |

Code reviewers: B ↔ C and A ↔ D.

## The three agreements

Everything else can change; these three only change if all four of us agree.

**1. One table.** One row per following car per 0.1 s. Units are SI.

| Column | Meaning |
|---|---|
| `pair_id` | One continuous stretch of the same driver behind the same leader |
| `driver_id`, `leader_id` | Following car, car directly ahead |
| `driver_group` | 0–4. Train on four groups, test on the fifth |
| `global_time_ms`, `t_s` | Clock time; seconds since the pair started |
| `position_m`, `leader_position_m` | Front bumper along the road |
| `gap_m` | Space between the leader's rear bumper and the follower's front bumper |
| `speed_mps`, `leader_speed_mps` | Speeds |
| `closing_speed_mps` | `speed_mps - leader_speed_mps`; positive = getting closer |
| `accel_mps2` | **What the models predict** |
| `leader_accel_mps2` | Leader's acceleration (for the simulator, not a model input) |

Defined in `carfollow/schema.py`; `validate(df)` checks a table.

**2. One plug.** Every model is an object with

```python
def accel(self, history: np.ndarray) -> float: ...
```

`history` has shape `(30, 3)`: the last 3 s, oldest row first, columns
`gap_m, speed_mps, closing_speed_mps`. It returns the acceleration the driver
applies over the next 0.1 s. Memoryless models and IDM read only `history[-1]`.
`check_model(model)` tells you if your model fits. See `carfollow/plug.py` and
the baselines in `carfollow/baselines.py`.

**3. One scorer.** `evaluate(model, drivers)` in `carfollow/evaluate.py`
returns all scores for one model on one table. No private metrics.

| Score | Measures |
|---|---|
| Step | Error of the predicted acceleration for the next 0.1 s |
| Drive | Gap error and crashes when the model drives 30 s behind the real leader |
| Spread | Distance between simulated and real time-headway distributions |

## Getting started

```bash
git clone <repo-url>
cd dse511-car-following
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Data is **not** in Git. Download `pairs_sample.csv` from the Teams channel and
put it at `data/sample/pairs_sample.csv`. Then:

```bash
python -m pytest -q
```

All tests should pass. Using the data in Python:

```python
from carfollow import load_table, split_by_group, iter_pairs, history_at, IDM

table = load_table()                          # validated sample
train, test = split_by_group(table, test_group=0)
pair = next(iter_pairs(test))
IDM.textbook().accel(history_at(pair, 29))    # first prediction needs 30 rows
```

## The sample (week 1)

`pairs_sample.csv`: NGSIM US-101, lane 2, 6 minutes at 10 Hz. About 80,000 rows,
239 pairs, 214 drivers. Built by `scripts/make_sample.py`; details in
`pairs_sample.meta.json`. Known limits, fixed in the full data (30 Oct):

- one lane and 6 minutes only;
- `gap_m` assumes every leader is 4.5 m long;
- position is smoothed (2.1 s Savitzky–Golay window) before taking speed and
  acceleration, because raw NGSIM positions are noisy.

The column names will not change between the sample and the full data, so
code written on the sample keeps working.

## How we work

- Work in your own lane folder. To change `carfollow/` or another lane's
  folder, open a pull request and ask that lane's owner to review.
- One branch per task, pull request into `main`, reviewed within 2 days.
- Fix and log random seeds. Never commit data or trained model files.
- Friday update in the Teams chat: done / stuck / next.

## Key dates

| Date | What happens |
|---|---|
| Fri 30 Oct | Full US-101 data, version 1 |
| Fri 6 Nov | Milestone: data frozen, first comparison table |
| Fri 13 Nov | Results frozen |
| Thu 19 Nov | Course checkpoint: all analysis done |
| Tue 1 Dec | Final presentations |

## Data license

NGSIM vehicle trajectories, U.S. Department of Transportation
(data.transportation.gov, dataset `8ect-6jqj`), CC BY-SA 3.0.
