# Lane B · ML-1 memoryless (Pantho)

**Job:** a standard regression. Input: gap, own speed and closing speed right
now (`history[-1]`). Output: `accel_mps2`.

**Hands off:** two memoryless models, each wrapped as a model with
`accel(history)` (see `carfollow/plug.py`):

- **Linear Regression (physics)**
- **Gradient Boosting (physics, monotonic)**: `HistGradientBoostingRegressor`,
  `max_leaf_nodes=25`, `min_samples_leaf=400`, `learning_rate=0.05`,
  `max_iter=250`, `l2_regularization=1.0`

Both use the 3 raw inputs plus 5 physics features computed from the same row
(`inv_gap`, `inv_headway`, `inv_ttc`, `brake_need`, `idm_textbook`). Gradient
Boosting is constrained so each feature pushes acceleration in the physically
expected direction. Outputs are clipped to ±9 m/s² (IDM's braking limit). The
raw-feature versions are kept only as comparison results.

| By | Deliverable | Status |
|---|---|---|
| week 1 | First models trained on the sample | done |
| 1 Nov | Trained on the full data | done |
| 8 Nov | Tuned; running in D's simulator | tuned (nested group CV); local 30 s drive test done; waiting on D's scorer |
| 13 Nov | Results frozen | done for local scoring; re-check with D's scorer |
| 19 Nov | Folder documented | this README |

**Done when:** both models pass D's scorer on all 5 driver groups.

## Files

| File | What it is |
|---|---|
| `Lane_B_Memoryless_ML.ipynb` | Training, nested group cross-validation, local drive test, results and plots |
| `memoryless.py` | The plug wrapper (`MemorylessAccelerationModel`), feature builders, `save_model` / `load_model` |
| `outputs/` | Results CSVs and saved `.joblib` models. **Not in Git** (see `.gitignore`); shared through Teams |

Main outputs (written by the last notebook cell):

- `lane_b_fold_results.csv`: every model's scores for each held-out group
- `lane_b_summary.csv`: averages over the 5 held-out groups
- `lane_b_grid_search_by_outer_fold.csv`: Gradient Boosting settings picked in each outer fold
- `linear_regression_physics_full.joblib`, `gradient_boosting_physics_full.joblib`: the hand-off models

## Using the models

Run from the repository root so `lanes` and `carfollow` are importable:

```python
from lanes.B_memoryless.memoryless import load_model

model = load_model("lanes/B_memoryless/outputs/gradient_boosting_physics_full.joblib")
model.accel(history)   # history: (30, 3) array of [gap_m, speed_mps, closing_speed_mps]
```

The models read only the last row of `history`. `model.accel_batch(histories)`
takes an `(n, 30, 3)` array for faster rollouts.

> **Caveat:** the saved hand-off models are refit on **all five groups**,
> including group 0. Do not score them on group 0 (or any group) as if it were
> unseen. For held-out numbers, use the cross-validation results below.

## How it was evaluated

- **Nested group cross-validation.** Each of the 5 driver groups is held out
  once (`split_by_group`). No driver is on both sides. Gradient Boosting
  (physics, monotonic) is tuned with `GridSearchCV` + `GroupKFold` on the four
  training groups only. The other models use fixed settings.
- **Step score** (MAE, RMSE, R²) uses the same rows as the other lanes: the first
  29 rows of each pair (no full 3 s history yet) and the last row (its
  acceleration can't be checked) are dropped. For group 0 that is 685,851 rows.
- **Local drive test.** Each held-out pair starts from its real state at row 29,
  and the model drives up to 30 s behind the real leader using
  `carfollow.kinematics.advance`. We report gap RMSE, crashes and a headway-spread
  proxy (`spread_w1`, Wasserstein distance between simulated and real time
  headways). This is a local check until D's official scorer is ready.

## Results

**Group 0 (shared test group), same rows as the other lanes:**

| Model | MAE | RMSE | R² | Drive gap RMSE (m) | Crashes / 1640 | spread_w1 (s) |
|---|---|---|---|---|---|---|
| KeepSpeed | 0.5377 | 0.7420 | −0.006 | 31.26 | 676 | 2.691 |
| IDM textbook | 1.1993 | 1.9778 | −6.144 | 6.55 | 0 | 0.464 |
| Linear Regression (raw) | 0.4837 | 0.6497 | 0.229 | 7.04 | 400 | 0.118 |
| **Linear Regression (physics)** | 0.4778 | 0.6365 | 0.260 | 6.66 | 28 | 0.076 |
| Gradient Boosting (raw) | 0.4649 | 0.6219 | 0.294 | 6.32 | 145 | 0.073 |
| **Gradient Boosting (physics, monotonic)** | 0.4693 | 0.6257 | 0.285 | 6.30 | 80 | 0.087 |

**Average over all 5 held-out groups** (crashes summed over 8,301 drives):

| Model | MAE | RMSE (± sd) | R² | Drive gap RMSE (± sd) | Crashes / 8301 | spread_w1 (s) |
|---|---|---|---|---|---|---|
| KeepSpeed | 0.5377 | 0.7401 ± 0.0023 | −0.006 | 31.38 ± 0.38 | 3437 | 2.909 |
| IDM textbook | 1.1646 | 1.9169 ± 0.0438 | −5.751 | 6.67 ± 0.14 | 0 | 0.470 |
| Linear Regression (raw) | 0.4849 | 0.6501 ± 0.0024 | 0.224 | 6.89 ± 0.10 | 2020 | 0.112 |
| **Linear Regression (physics)** | 0.4787 | 0.6367 ± 0.0019 | 0.256 | 6.57 ± 0.11 | 155 | 0.086 |
| Gradient Boosting (raw) | 0.4660 | 0.6222 ± 0.0020 | 0.289 | 6.30 ± 0.12 | 664 | 0.084 |
| **Gradient Boosting (physics, monotonic)** | 0.4704 | 0.6261 ± 0.0021 | 0.280 | 6.25 ± 0.13 | 399 | 0.101 |

**Main finding (early answer to Q1):** all four Lane B models are almost equally
good at one-step prediction (R² 0.22–0.29), but they drive very differently.
Over the 5 groups, the raw Linear Regression crashed in about 1 in 4 drives,
Gradient Boosting (physics) in about 1 in 20, and Linear Regression (physics) in
about 1 in 50. Good one-step prediction does not guarantee good driving. The
physics features mainly help in closed loop.

**Limits:** these models see only the current moment, while real drivers react
with a delay. Lane C's memory models are expected to do better on driving.

## Reproducing

1. Put the full pair table where `CSV_PATH` in the notebook's setup cell points
   (download from Teams; data is never committed).
2. Run `Lane_B_Memoryless_ML.ipynb` top to bottom. The nested cross-validation
   cell is the slow one.
3. The last cell writes the CSVs and hand-off models to `outputs/` and checks
   that each reloaded model passes `check_model` and gives the same answer.
