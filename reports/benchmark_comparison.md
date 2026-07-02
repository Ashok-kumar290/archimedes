# Arithmetic benchmark comparison

Same seeded held-out problem set for every model (`scripts/benchmark_arithmetic.py`, seed 777, 25 problems per family). Archimedes answers with step-by-step neural reasoning, no tools; baselines get few-shot direct-answer prompting. Run 2026-07-02, checkpoint `archimedes_math_small_reasoning_v6/step_004000.pt`.

| family | Archimedes-117M | GPT-2-124M | Pythia-160M |
|---|---|---|---|
| add | 84% (21/25) | 0% (0/25) | 0% (0/25) |
| div | 80% (20/25) | 0% (0/25) | 0% (0/25) |
| fraction | 80% (20/25) | 0% (0/25) | 0% (0/25) |
| gcd | 84% (21/25) | 0% (0/25) | 0% (0/25) |
| linear | 100% (25/25) | 8% (2/25) | 12% (3/25) |
| mul | 88% (22/25) | 0% (0/25) | 0% (0/25) |
| odd_sum | 100% (24/24) | 0% (0/24) | 0% (0/24) |
| percent | 28% (7/25) | 0% (0/25) | 0% (0/25) |
| sub | 84% (21/25) | 0% (0/25) | 0% (0/25) |
| triangular | 96% (24/25) | 0% (0/25) | 0% (0/25) |
| **overall** | **82%** (205/249) | **1%** (2/249) | **1%** (3/249) |

Notes:

- Archimedes solves each problem by explicit digit-level computation
  (column addition with carries, place-value partial products, long
  division) generated entirely by the neural net; no tool routing.
- The weak family, percent, had the smallest training pool (~800 unique
  examples) — a data-coverage gap, not an algorithmic one.
- Baselines score near zero on multi-digit arithmetic, consistent with
  the known behavior of general-purpose models at this scale.
- Reproduce with:
  `python3 scripts/benchmark_arithmetic.py --checkpoint <ckpt> --tokenizer <tok>`
  and `--hf-model gpt2` / `--hf-model EleutherAI/pythia-160m`.
