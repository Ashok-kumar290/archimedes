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

## Baseline ladder — specialist (117M) vs open general models

All models scored on the **identical** held-out benchmark (seed 99991, 50
problems/family, n=500), same fixed extraction (final number in the
completion). Baselines got a 4-example physics few-shot for answer *format*
only (no worked physics), and instruct models used their chat template.

| model | params | overall |
|---|---|---|
| **Archimedes-Physics (ours, from scratch)** | **117M** | **91.6%** |
| Qwen2.5-1.5B-Instruct | 1.5B | 59.4% |
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 44.6% |
| Qwen2.5-0.5B-Instruct | 0.5B | 13.2% |
| SmolLM2-135M | 135M | 0.6% |
| Galactica-1.3B (science specialist) | 1.3B | 0.8% |
| Galactica-125M (science specialist) | 125M | 0.6% |

### Reading it honestly

- **The headline is data efficiency, not raw capability.** Archimedes is a
  *specialist* trained on exactly these 10 formula families; the Qwen models
  are *generalists* meeting them cold. A 117M specialist beating a 13x-larger
  general instruct model (91.6 vs 59.4) is the divided-brain thesis working —
  but it is a specialist-vs-generalist comparison and must be stated as one.
- **Same-size science specialist (the comparison that matters):** Galactica-125M,
  Meta's science model at our exact size, scored **0.6%** — and Galactica-1.3B
  (11x bigger) scored **0.8%**. Caveat: Galactica is a 2022 *base completion*
  model, not instruction-tuned, so much of that ~0 is format mismatch, not an
  inability to do the physics. It is not a clean capability test. What it does
  show: off-the-shelf, a same-size science LM produces nothing usable on these
  prompts, whereas a task-shaped 117M model answers cleanly.
- **Multi-step is hard for everyone.** Kinematics (v = u + a*t, a two-op chain)
  was the great leveller: Qwen2.5-1.5B scored **2%** on it, Qwen-0.5B **0%**,
  us **66%**, Qwen-Math **66%**. Potential energy (three-factor) was low across
  the board (us 41%, Qwen-1.5B 12%). This is the same decomposition ceiling the
  v1-vs-v3 table found from the inside — it shows up in the frontier models too,
  and is the orchestrator's job to fix.
- **Qwen-Math < Qwen-general (44.6 vs 59.4)** on these word problems: the
  math-tuned model over-works simple substitutions and drifts on format, while
  the general instruct model answers plainly. Consistent with the arithmetic
  benchmark's finding.
- **Extraction-fix audit:** re-running the two verbose baselines with
  last-number extraction moved them <2 points (Qwen-1.5B 61.0->59.4,
  Qwen-Math 45.2->44.6), confirming the earlier scores were not materially
  biased. The fix makes the method defensible, not the result different.

### Scope disclaimer

This benchmark is a **narrow 10-formula integer calculator suite**, not a test
of general physics understanding. Every answer is an exact integer; there is no
unit reasoning, no conceptual/multiple-choice content, no symbolic derivation,
no multi-paragraph problem. The numbers above say Archimedes-Physics is an
excellent *calculator for these ten relations* — nothing broader. Claims must
stay inside that boundary.

## Reproduce

```bash
# our checkpoint
python3 scripts/benchmark_physics.py \
  --checkpoint <checkpoint>.pt \
  --tokenizer <archimedes_physics_bpe_32768_digitsplit>/tokenizer.json \
  --per-family 100

# an open baseline (same benchmark)
python3 scripts/benchmark_physics.py \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct --hf-chat --per-family 50 --out bench/phys_qwen15.jsonl
```
