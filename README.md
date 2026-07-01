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
