# Lane B · ML-1 memoryless (Pantho)

**Job:** a standard regression. Input: gap, own speed and closing speed right
now (`history[-1]`). Output: `accel_mps2`.

**Hands off:** linear regression and gradient boosting, each wrapped as a
model with `accel(history)` (see `carfollow/plug.py`).

| By | Deliverable |
|---|---|
| week 1 | First models trained on the sample |
| 1 Nov | Trained on the full data |
| 8 Nov | Tuned; running in D's simulator |
| 13 Nov | Results frozen |
| 19 Nov | Folder documented |

**Done when:** both models pass D's scorer on all 5 driver groups.

Tips: train with `split_by_group` so no driver is in both train and test.
`check_model(your_model)` confirms the plug fits.
