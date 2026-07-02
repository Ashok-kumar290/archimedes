# Archimedes Math

Archimedes Math is a math-domain data and model training pipeline:

- provenance-tracked data ingestion
- math corpus curation and mixture building
- tokenizer training
- token shard generation
- decoder-only Transformer pretraining

The data itself is intentionally not stored in git. Keep large artifacts on the
Archimedes HDD or in a separate Colab/Drive payload.

Default local data root:

```text
/home/seyominaoto/archimedes-data
```

## Current Artifacts

The local HDD currently contains:

```text
math_tokens_v1:           163.3M actual tokens
math_tokens_v2_balanced:  126.3M actual tokens
tokenizers:               v1 and v2 32k BPE tokenizers
colab payload:            archimedes_math_colab_payload_v2.tar.gz
```

## Current Results

This repo is an experimental training pipeline, not a finished benchmark model.
The current math expert has learned mathematical formatting and some common
solution patterns, but it is still undertrained for reliable arithmetic and
multi-step proof reasoning.

Current measured state:

- corpus snapshot: 163.3M token IDs, 17 train shards, 1 validation shard
- balanced math mixture target: 25% textbooks/notes, 25% worked problems, 20%
  formal proofs, 15% research, 10% symbolic math, 5% references
- trained configs: 44M parameter tiny model and 117M parameter small model;
  a larger A100 base config is included but not yet fully exercised
- best small pretraining run observed in Colab: validation perplexity improved
  to about 4.1-5.8 depending on run/checkpoint and data split
- SFT works mechanically with prompt-loss masking, but small SFT runs can
  overfit quickly and should be evaluated with held-out prompts
- held-out eval of the small model after reasoning SFT (run
  `archimedes_math_small_reasoning_v3`, 2026-07-02): output format, procedure
  selection, and both template proofs are correct on all six held-out prompts;
  all four numeric final answers are wrong. The model learned what to compute
  but not digit-level arithmetic, because training traces stated multi-digit
  results in a single step.
- after retraining on the digit-decomposed curriculum (run
  `archimedes_math_small_reasoning_v4`, 2026-07-02): 3 of 4 numeric answers
  correct on held-out prompts (linear equation, 17^2 via four partial products
  and chained column additions, 25*26/2), both proofs still correct. The one
  failure (247 + 389) shows correct single-digit facts and carry mechanics but
  wrong digit extraction from the operands: the BPE tokenizer chunks numbers
  into multi-digit tokens, so digit identity must be learned per token. The
  curriculum now adds explicit digit-reading steps and a digit-listing drill
  family; a digit-split tokenizer is the planned fix at the next pretraining
  run.
- with digit-reading drills (run `archimedes_math_small_reasoning_v5`,
  2026-07-02): still 3 of 4 numeric, but the failure narrowed to a single
  misread digit (hundreds of 247 read as 3); the model now reads operand
  digits aloud before computing and every downstream carry is correct. The
  drill pool now enumerates all 2-3 digit numbers exhaustively instead of
  sampling them.
- with exhaustive digit drills (run `archimedes_math_small_reasoning_v6`,
  2026-07-02): all six held-out eval prompts correct, and 82% overall on a
  249-problem held-out benchmark across 10 task families, versus 1% for
  GPT-2 (124M) and 1% for Pythia-160M on the identical problems. Every
  answer is produced by step-by-step neural computation with no tool
  assistance. Full table: `reports/benchmark_comparison.md`. Known gap:
  percent problems (28%), the smallest training family.

Evaluation honesty rule: there is no tool or routing path anywhere in
generation. `scripts/generate_math.py`, `scripts/eval_math_prompts.py`, and
`scripts/benchmark_arithmetic.py` produce every answer from the checkpoint's
forward passes alone; the historical arithmetic router and the legacy SFT
builders that trained on eval prompts have been deleted from the codebase.

Generalization vs. recall, stated plainly: the add and sub benchmark
families draw from a ~10^8 problem space of which training saw under 0.05%,
so accuracy there (84% each) is true held-out generalization of the learned
digit algorithms. Smaller families (mul, div, linear, sum formulas) have
problem spaces that training largely covered, so their scores measure
learned execution of procedures the model has practiced, not novel
generalization.

Process charts are in `reports/current_process/`:

- `tokens_by_source.png`
- `storage_by_stage.png`
- `token_shard_progress.png`

## Colab

See [COLAB.md](COLAB.md).

## Data Commands

Seed retrieval:

```bash
python3 scripts/retrieve_seed.py --sources configs/seed_sources.yaml
```

Run curation:

```bash
python3 scripts/curate_math_v2.py --target-tokens 120000000 --validation-frac 0.005
```

Build token shards:

```bash
python3 scripts/build_token_shards.py   --tokenizer /home/seyominaoto/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json   --train-manifest /home/seyominaoto/archimedes-data/metadata/math_train_manifest_v2.jsonl   --val-manifest /home/seyominaoto/archimedes-data/metadata/math_val_manifest_v2.jsonl   --out-dir /home/seyominaoto/archimedes-data/shards/math_tokens_v2_balanced
```

## Generate From A Checkpoint

After training in Colab, generate from a saved checkpoint. By default this uses
the model only, without deterministic math tools:

```bash
python3 scripts/generate_math.py \
  --checkpoint /content/archimedes-data/checkpoints/archimedes_math_small_v2_long/step_006000.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json \
  --prompt "Prove that the sum of the first n odd numbers is n^2." \
  --prompt-style solution \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-k 50 \
  --tool-mode off
```

## Supervised Fine-Tuning

The pretrained checkpoints are base next-token models. To make them answer in worked-solution format, build a seed SFT set and fine-tune from the best pretrained checkpoint:

```bash
python3 scripts/build_sft_seed.py \
  --out /content/archimedes-data/sft/math_seed_sft.jsonl \
  --count 20000

python3 scripts/train_sft_math.py \
  --init-checkpoint /content/archimedes-data/checkpoints/archimedes_math_small_v2_long/step_006000.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json \
  --sft-jsonl /content/archimedes-data/sft/math_seed_sft.jsonl \
  --data-root /content/archimedes-data \
  --run-name archimedes_math_small_sft_seed \
  --batch-size 16 \
  --grad-accum 2 \
  --max-steps 1000 \
  --lr 5e-5 \
  --eval-interval 100 \
  --eval-batches 20 \
  --save-interval 250 \
  --compile
```


For SFT runs after this revision, keep the explicit solution end marker enabled. It teaches the model when to stop instead of drifting back into pretraining text. A safer rerun command is:

```bash
python3 scripts/train_sft_math.py \
  --init-checkpoint /content/archimedes-data/checkpoints/archimedes_math_small_v2_long/step_006000.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json \
  --sft-jsonl /content/archimedes-data/sft/math_seed_sft.jsonl \
  --data-root /content/archimedes-data \
  --run-name archimedes_math_small_sft_seed_stop \
  --batch-size 16 \
  --grad-accum 2 \
  --max-steps 1000 \
  --lr 5e-5 \
  --eval-interval 50 \
  --eval-batches 20 \
  --save-interval 50 \
  --compile
```

Compare checkpoints with:

```bash
python3 scripts/eval_math_prompts.py \
  --checkpoint /content/archimedes-data/checkpoints/archimedes_math_small_sft_seed_stop/step_001000.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json \
  --out /content/archimedes-data/checkpoints/archimedes_math_small_sft_seed_stop/eval_prompts.jsonl
```

To train reasoning traces rather than benchmark-specific routes, build the
reasoning curriculum and fine-tune from the best available small checkpoint:

```bash
python3 scripts/build_reasoning_curriculum.py \
  --out /content/archimedes-data/sft/math_reasoning_curriculum_v1.jsonl \
  --count 200000

python3 scripts/train_sft_math.py \
  --init-checkpoint /content/archimedes-data/checkpoints/archimedes_math_small_sft_curriculum_v2_rebuild/step_002000.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json \
  --sft-jsonl /content/archimedes-data/sft/math_reasoning_curriculum_v1.jsonl \
  --data-root /content/archimedes-data \
  --run-name archimedes_math_small_reasoning_v1 \
  --batch-size 16 \
  --grad-accum 2 \
  --max-steps 1500 \
  --lr 5e-6 \
  --eval-interval 100 \
  --eval-batches 30 \
  --save-interval 250 \
  --compile
```

