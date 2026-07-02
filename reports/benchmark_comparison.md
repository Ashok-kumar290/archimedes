# Arithmetic benchmark comparison

Same seeded held-out problem set for every model (`scripts/benchmark_arithmetic.py`, seed 777, 25 problems per family). Archimedes answers with step-by-step neural reasoning, no tools; baselines get few-shot direct-answer prompting. Run 2026-07-02, checkpoint `archimedes_math_small_reasoning_v9/step_006000.pt`. The 249-problem symbolic set below is byte-identical to the one the baselines were scored on.

| family | Archimedes-117M | GPT-2-124M | Pythia-160M |
|---|---|---|---|
| add | 92% (23/25) | 0% (0/25) | 0% (0/25) |
| div | 96% (24/25) | 0% (0/25) | 0% (0/25) |
| fraction | 100% (25/25) | 0% (0/25) | 0% (0/25) |
| gcd | 72% (18/25) | 0% (0/25) | 0% (0/25) |
| linear | 100% (25/25) | 8% (2/25) | 12% (3/25) |
| mul | 80% (20/25) | 0% (0/25) | 0% (0/25) |
| odd_sum | 100% (24/24) | 0% (0/24) | 0% (0/24) |
| percent | 68% (17/25) | 0% (0/25) | 0% (0/25) |
| sub | 96% (24/25) | 0% (0/25) | 0% (0/25) |
| triangular | 100% (25/25) | 0% (0/25) | 0% (0/25) |
| **overall** | **90%** (225/249) | **1%** (2/249) | **1%** (3/249) |

Worded operator families (spoken forms like "Compute 707 plus 5754."),
25 problems each, same harness; baselines not rerun on these:

| family | v9 (reference) | v8 |
|---|---|---|
| add_worded | 96% (24/25) | 92% (23/25) |
| sub_worded | 80% (20/25) | 92% (23/25) |
| overall incl. worded (299) | 90.0% | 84.6% |

Earlier reference runs on the identical 249 symbolic problems:
`..._v8` 83% (fraction 64%, percent 44%); `..._v6` 82% (percent 28%,
failed worded operator prompts entirely).

Notes:

- Archimedes solves each problem by explicit digit-level computation
  (column addition with carries, place-value partial products, long
  division) generated entirely by the neural net; no tool or routing code
  exists in the generation path.
- Train/test overlap, stated honestly: add and sub draw from a ~10^8
  problem space of which training saw under 0.05% — accuracy there is true
  held-out generalization. mul/div/linear/sum-formula spaces were largely
  covered during training, so those scores measure learned execution of
  practiced procedures.
- The weak families track per-family training exposure, not algorithm
  failures: fraction went 64% -> 100% and percent 44% -> 68% after their
  curriculum share was raised (v8 -> v9). gcd (72%) has not responded to
  upweighting yet and is the current open gap, likely because Euclidean
  steps embed multi-digit division inside each line.
- Baselines score near zero on multi-digit arithmetic, consistent with
  the known behavior of general-purpose models at this scale.
- Reproduce with:
  `python3 scripts/benchmark_arithmetic.py --checkpoint <ckpt> --tokenizer <tok>`
  and `--hf-model gpt2` / `--hf-model EleutherAI/pythia-160m`.
