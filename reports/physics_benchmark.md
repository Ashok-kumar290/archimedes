# Physics lobe benchmark

Held-out physics problems (`scripts/benchmark_physics.py`, seed 99991,
disjoint from the training seed), 100 problems per family, 1000 total. Every
answer is produced by the checkpoint's own step-by-step reasoning — pick the
formula, substitute, compute — with no tool or calculator in the path. The
answer key is computed independently by the benchmark; the model only sees the
word problem.

## v3 vs v1 — the arithmetic-drill result

Both checkpoints scored on the identical n=1000 benchmark. v1 (`_reasoning_v1`)
was trained on formula families only; v3 (`_reasoning_v3`) added arithmetic
drill families (division-heavy) reusing the math lobe's verified digit-level
traces, with explicit per-family weighting so the small division families are
not starved.

| family | v1 | v3 |
|---|---|---|
| density | 52% | **100%** |
| ohm_i (V/R) | 61% | **100%** |
| power (W/t) | 41% | **100%** |
| momentum | 100% | 100% |
| newton | 99% | 99% |
| ohm_v (I*R) | 100% | 100% |
| weight | 100% | 100% |
| work | 100% | 100% |
| kinematics | 66% | 66% |
| potential | 37% | 41% |
| **overall** | **75.6%** | **90.6%** |

## Reading it honestly

- The **+15 point overall gain is entirely in the three division families**
  (density, power, ohm_i: 41-61% -> 100%). Directly drilling digit-level
  division fixed the lobe's measured weakness. Nothing regressed.
- The physics base model was pretrained on physics *prose*, so it started
  arithmetically weaker than the math base (which saw computation-rich text).
  The drills supply the arithmetic practice the pretraining lacked.
- The **two multi-step families did not improve** (kinematics 66->66,
  potential 37->41). This is the honest boundary: these chain two operations
  (a*t then +u; m*h then *10, with a 5-digit result). Single-operation
  families are all ~100%; the model handles one operation reliably but
  degrades across a chain. This is a decomposition problem, addressed by the
  orchestrator (compute each step separately), not by more curriculum.
- Methodology note: an earlier n=25 benchmark was too noisy to iterate on
  (families swung ±12%); a v2 attempt appeared to regress purely from that
  noise. The move to n=1000 gave a stable signal, and re-scoring v1 on the
  same benchmark (75.6%, not the noisy earlier "80%") makes the v1-vs-v3
  delta a clean measurement rather than an artifact of a changed test.

## Reproduce

```bash
python3 scripts/benchmark_physics.py \
  --checkpoint <checkpoint>.pt \
  --tokenizer <archimedes_physics_bpe_32768_digitsplit>/tokenizer.json \
  --per-family 100
```
