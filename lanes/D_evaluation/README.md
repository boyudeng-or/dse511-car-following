# Lane D · Evaluation (Mehmed)

**Job:** the referee. Let every model drive behind a real leader for 30 s and
score all models the same way.

**Hands off:** `evaluate(model, drivers)` in `carfollow/evaluate.py`, the
comparison table and the final figures.

| By | Deliverable |
|---|---|
| week 1 | Scorer v1 (Step score); simulator |
| 1 Nov | Drive score works |
| 8 Nov | Spread score; first comparison table |
| 15 Nov | Q3 scoring; which inputs each model relies on |
| 19 Nov | README; checkpoint slides |

**Done when:** one call gives all scores with error bars for any model.

Start with `KeepSpeed()` and `IDM.textbook()` from `carfollow.baselines`:
no training needed. The sample has `leader_position_m` and
`leader_speed_mps` to replay the real leader during a rollout. Simulator
update for one 0.1 s step (suggested): `v += a*dt`, `x += v*dt` (speed
clipped at 0), with a crash when `gap_m <= 0`.
