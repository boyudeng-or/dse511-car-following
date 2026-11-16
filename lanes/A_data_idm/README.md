# Lane A · Data + IDM (Boyu)

**Job:** turn the NGSIM download into the shared pair table, and provide the
textbook rule everyone must beat.

**Hands off:** the pair table and 5 driver groups (to everyone); IDM as a plug (to D).

| By | Deliverable | Status |
|---|---|---|
| 25 Oct | Sample table, textbook IDM, `KeepSpeed` | done (`scripts/make_sample.py`, `carfollow/baselines.py`) |
| 30 Oct | Full US-101 data, version 1 (main lanes 1–5, real vehicle lengths) | done 4 Nov (`scripts/download_ngsim.py`, `scripts/make_full.py`) |
| 6 Nov | Data frozen; IDM calibrated (one parameter set for everyone) | |
| 13 Nov | One command rebuilds every result | |
| 19 Nov | Clean-machine test passes | |

**Done when:** a teammate rebuilds everything with one command on a clean machine.
