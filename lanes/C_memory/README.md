# Lane C · ML-2 with memory (Daniel)

**Job:** the same prediction as lane B, but the model sees the whole
`history` (last 3 s, shape `(30, 3)`), like a driver who remembers that the
car ahead has been braking.

**Hands off:** a small GRU (PyTorch) wrapped as a model with `accel(history)`.
Answers Q2: does memory help?

| By | Deliverable |
|---|---|
| week 1 | GRU skeleton trained on the sample |
| 1 Nov | Trained on the full data |
| 8 Nov | Tuned; running in D's simulator |
| 13 Nov | Results frozen |
| 19 Nov | Folder documented |

Extra, cut first if time is short: an IDM + neural-network hybrid.

Tips: `history_at(pair, i)` cuts the window for row `i`; training samples are
every `i >= 29` of every pair. Install PyTorch from pytorch.org (not in
`requirements.txt`). Do not commit `.pt` files.
