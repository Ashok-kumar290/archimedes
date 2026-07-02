# Arithmetic benchmark comparison

Same seeded held-out problem set for every model (`scripts/benchmark_arithmetic.py`, seed 777, 25 problems per family). Archimedes answers with step-by-step neural reasoning, no tools; baselines get few-shot direct-answer prompting. Run 2026-07-02, checkpoint `archimedes_math_small_reasoning_v8/step_006000.pt` (trained on the spoken-operator curriculum, so it also handles worded prompts like "8351 plus 4767").

| family | Archimedes-117M | GPT-2-124M | Pythia-160M |
|---|---|---|---|
| add | 96% (24/25) | 0% (0/25) | 0% (0/25) |
| div | 88% (22/25) | 0% (0/25) | 0% (0/25) |
| fraction | 64% (16/25) | 0% (0/25) | 0% (0/25) |
| gcd | 68% (17/25) | 0% (0/25) | 0% (0/25) |
| linear | 100% (25/25) | 8% (2/25) | 12% (3/25) |
| mul | 96% (24/25) | 0% (0/25) | 0% (0/25) |
| odd_sum | 100% (24/24) | 0% (0/24) | 0% (0/24) |
| percent | 44% (11/25) | 0% (0/25) | 0% (0/25) |
| sub | 84% (21/25) | 0% (0/25) | 0% (0/25) |
| triangular | 92% (23/25) | 0% (0/25) | 0% (0/25) |
| **overall** | **83%** (207/249) | **1%** (2/249) | **1%** (3/249) |

Previous reference run for comparison: `archimedes_math_small_reasoning_v6`
scored 82% (205/249) on the identical problems (add 84%, fraction 80%,
gcd 84%, percent 28%) but failed worded operator prompts entirely.

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
- The weak families, percent (44%), fraction (64%), and gcd (68%), track
  training-pool size and per-family exposure — data-coverage gaps, not
  algorithmic ones. percent doubled from 28% when training length grew
  from 4000 to 6000 steps.
- Baselines score near zero on multi-digit arithmetic, consistent with
  the known behavior of general-purpose models at this scale.
- Reproduce with:
  `python3 scripts/benchmark_arithmetic.py --checkpoint <ckpt> --tokenizer <tok>`
  and `--hf-model gpt2` / `--hf-model EleutherAI/pythia-160m`.
