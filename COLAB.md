# Archimedes Math Colab Setup

This repo contains code, configs, and scripts. The tokenized dataset is stored
separately as a payload archive because it should not be committed to git.

## 1. Clone

```bash
git clone https://github.com/Ashok-kumar290/archimedes.git /content/archimedes
cd /content/archimedes
pip install -e .
```

## 2. Upload Dataset Payload

Upload this local file to Colab or Google Drive:

```text
/home/seyominaoto/archimedes-data/colab/archimedes_math_colab_payload_v2.tar.gz
```

Then extract in Colab. Extract into a staging directory, never into
`/content/archimedes`: the payload archive carries snapshots of some repo
files, and extracting it into the repo silently overwrites the cloned code
with stale versions.

```bash
mkdir -p /content/archimedes-data /content/payload
tar -xzf /content/archimedes_math_colab_payload_v2.tar.gz -C /content/payload
mv /content/payload/shards /content/archimedes-data/
mv /content/payload/tokenizers /content/archimedes-data/
mv /content/payload/metadata /content/archimedes-data/ || true
```

If a payload was ever extracted into the repo by mistake, restore the code
with `cd /content/archimedes && git checkout -- .` before running anything.

## 3. Train Tiny On V2 Balanced Data

```bash
cd /content/archimedes
python3 scripts/train_math_model.py \
  --config configs/model/archimedes_math_tiny.json \
  --data-root /content/archimedes-data \
  --shard-dir /content/archimedes-data/shards/math_tokens_v2_balanced \
  --run-name archimedes_math_tiny_v2 \
  --batch-size 32 \
  --grad-accum 4 \
  --max-steps 2000 \
  --warmup-steps 100 \
  --eval-interval 100 \
  --eval-batches 20 \
  --save-interval 500 \
  --compile
```

## 4. Train Tiny On V1 Baseline

```bash
python3 scripts/train_math_model.py \
  --config configs/model/archimedes_math_tiny.json \
  --data-root /content/archimedes-data \
  --shard-dir /content/archimedes-data/shards/math_tokens_v1 \
  --run-name archimedes_math_tiny_v1 \
  --batch-size 32 \
  --grad-accum 4 \
  --max-steps 2000 \
  --warmup-steps 100 \
  --eval-interval 100 \
  --eval-batches 20 \
  --save-interval 500 \
  --compile
```
