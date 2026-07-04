# Training a lobe on Colab (the repeatable recipe)

Every expert lobe trains the same way: pretrain a 117M base for fluency, pick
the checkpoint at the val-perplexity **floor** (later steps overfit), then
reasoning-SFT on the verified curriculum, and pick the SFT checkpoint by the
**held-out benchmark** (SFT val loss is uninformative — it just memorizes the
format). Worked example below is **chemistry**; code notes at the end.

## Key Colab gotcha
Colab's local disk is wiped on teardown. Read shards from **local** disk (fast)
but write checkpoints to **Drive** (persistent):
- `--shard-dir /content/archimedes-data/shards/<lobe>_tokens_v1`  (local)
- `--data-root /content/drive/MyDrive/archimedes_runs`            (Drive → checkpoints persist)

Checkpoints land at `<data-root>/checkpoints/<run-name>/step_*.pt`.

---

## Chemistry (complete this first)

```bash
# 0. (NOTEBOOK CELL) mount Drive
#    from google.colab import drive; drive.mount('/content/drive')

# 1. clone + deps + extract the payload (tokenizer + shards) to LOCAL disk
cd /content && rm -rf archimedes && git clone https://github.com/Ashok-kumar290/archimedes.git
cd /content/archimedes && pip -q install torch tokenizers transformers >/dev/null 2>&1
mkdir -p /content/archimedes-data
tar xzf /content/drive/MyDrive/archimedes_chemistry_payload_v1.tar.gz -C /content/archimedes-data

# 2. generate the verified curriculum (fast, ~seconds)
python3 scripts/build_chemistry_curriculum.py --out /content/chem_curriculum.jsonl --count 120000

# 3. PRETRAIN the 117M base — checkpoints to Drive, shards from local
PYTHONPATH=src python3 src/archimedes_model/train.py \
  --config configs/model/archimedes_chemistry_small.json \
  --data-root /content/drive/MyDrive/archimedes_runs \
  --shard-dir /content/archimedes-data/shards/chemistry_tokens_v1 \
  --run-name archimedes_chem_small_v1 \
  --batch-size 24 --grad-accum 2 \
  --max-steps 3000 --warmup-steps 100 \
  --eval-interval 250 --save-interval 500

# 4. pick the base checkpoint at the val_ppl FLOOR (physics overfit past ~2500)
cat /content/drive/MyDrive/archimedes_runs/checkpoints/archimedes_chem_small_v1/metrics.jsonl
#   -> choose the step with the lowest val_ppl, e.g. step_002500.pt

# 5. reasoning SFT from that base (override tokenizer to the CHEM tokenizer!)
PYTHONPATH=src python3 src/archimedes_model/sft.py \
  --init-checkpoint /content/drive/MyDrive/archimedes_runs/checkpoints/archimedes_chem_small_v1/step_002500.pt \
  --config configs/model/archimedes_chemistry_small.json \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_chemistry_bpe_32768_digitsplit/tokenizer.json \
  --sft-jsonl /content/chem_curriculum.jsonl \
  --data-root /content/drive/MyDrive/archimedes_runs \
  --run-name archimedes_chem_reasoning_v1 \
  --max-steps 2500 --save-interval 500

# 6. HELD-OUT benchmark — pick the SFT checkpoint with the best OVERALL
python3 scripts/benchmark_chemistry.py \
  --checkpoint /content/drive/MyDrive/archimedes_runs/checkpoints/archimedes_chem_reasoning_v1/step_002500.pt \
  --tokenizer /content/archimedes-data/tokenizers/archimedes_chemistry_bpe_32768_digitsplit/tokenizer.json \
  --per-family 50
```

Expect single-step relations near ~100% and the multi-step ones (molar_mass_two,
dilution, heat) weaker — the same multi-step ceiling physics has. If arithmetic
families lag, the curriculum already includes d_div/d_mul/d_add drills (the fix
that took physics 75.6%→90.6%).

---

## Python code (do after chemistry)

Same shape, different payload/curriculum/benchmark:
- payload: `archimedes_code_payload_v1.tar.gz` → shards `code_tokens_v1`, tokenizer `archimedes_code_bpe_32768_digitsplit`
- config: `configs/model/archimedes_code_small.json`
- curriculum: `python3 scripts/build_code_curriculum.py --out /content/code_curriculum.jsonl --count 120000` (write + debug)
- benchmark: `python3 scripts/benchmark_code.py --checkpoint <sft>.pt --tokenizer <code tokenizer> --mode write` (also `--mode debug`)

Note: the code corpus is ~10M tokens (lighter), so cap pretrain lower (~2000
steps) and watch val_ppl — it will hit the floor sooner.

---

## Every subsequent lobe
Identical recipe. Only four inputs change: the **payload** (corpus shards +
tokenizer), the **model config**, the **curriculum builder**, and the **benchmark**.
That's the lobe factory.
