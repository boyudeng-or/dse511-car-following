# Lane C: ML-2 with memory (Daniel)

**Goal**: Same prediction as Lane B, but using memory (GRU) to improve the predictionhe last 3 seconds of 'history'.

**Handoff**: Three trained GRU models wrapped in the shared plug (`accel(history)`, `accel_batch(histories)`). The models differ by the number of seconds/rows they look back (1,2,30 rows).

## Files

- `gru_train.ipynb`: This includes the training script for the GRU models including loading data, splitting into windows, showing step scores and plots

- `gru.py`: The network/model, the plug (`GRUDriver`) and `save/load_model`

- `outputs/`: Where the trained model files will be saved (`.pt`, `.npy`).

## Handoff files

These will be trained on `1`, `2` and `30` rows and trained on groups `1-3`, so they can be scored on group `0`. The files will be named as follows with `0` indicating the test group:

- `gru_w1_0_handoff.pt`

- `gru_w2_0_handoff.pt`

- `gru_w30_0_handoff.pt`

## To run the model

In the project root run the following code

```python
from lanes.C_memory.gru import load_model

model = load_model("lanes/C_memory/outputs/gru_w30_test0_handoff.pt")
model.accel(history) 
```

**NOTE**: Each file holds its weights and training normalization. `history` will be an array of size (30,3) consisting of [gap_m, speed_mps, closing_speed_mps].

## Training Details

- Trained on groups 1-3, validated on group 4 and tested on group 0

- Step scores use rows 29 to n-2 of each pair (685,851 on group 0), the same rows as lane B and scorer

- To retrain: run `gru_train.ipynb` top to bottom. Seeds are fixed (42) and trained models are saved to `outputs/`
