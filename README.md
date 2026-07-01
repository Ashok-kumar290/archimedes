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

After training in Colab, generate from a saved checkpoint:

```bash
python3 scripts/generate_math.py \
  --checkpoint /content/archimedes-data/checkpoints/archimedes_math_small_v2_long/step_006000.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_math_bpe_32768_v2/tokenizer.json \
  --prompt "Prove that the sum of the first n odd numbers is n^2." \
  --prompt-style solution \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-k 50
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
