from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from tokenizers import Tokenizer

from archimedes_model.model import ArchimedesMathModel, ModelConfig


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
DEFAULT_END_MARKER = "<|endofsolution|>"


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    completion: str


class SFTDataset:
    def __init__(self, path: Path, tokenizer: Tokenizer, context_length: int, seed: int = 1234, end_marker: str = DEFAULT_END_MARKER) -> None:
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rng = random.Random(seed)
        self.examples: list[tuple[list[int], list[int]]] = []
        eos_id = tokenizer.token_to_id("<|eos|>")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row["prompt"].strip()
                completion = row["completion"].strip()
                formatted_prompt = f"Problem: {prompt}\nSolution:"
                full_text = f"{formatted_prompt} {completion} {end_marker}\n"
                prompt_ids = tokenizer.encode(formatted_prompt).ids
                if eos_id is not None and prompt_ids and prompt_ids[-1] == eos_id:
                    # encode() appends <|eos|>; counting it would extend the
                    # prompt-loss mask over the first completion token
                    prompt_ids = prompt_ids[:-1]
                full_ids = tokenizer.encode(full_text).ids
                if len(full_ids) < 2:
                    continue
                if len(full_ids) > context_length:
                    full_ids = full_ids[:context_length]
                labels = full_ids[1:].copy()
                prompt_loss_tokens = min(max(len(prompt_ids) - 1, 0), len(labels))
                labels[:prompt_loss_tokens] = [-100] * prompt_loss_tokens
                if all(label == -100 for label in labels):
                    continue
                self.examples.append((full_ids[:-1], labels))
        if not self.examples:
            raise ValueError(f"no usable SFT examples loaded from {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        batch = [self.examples[self.rng.randrange(len(self.examples))] for _ in range(batch_size)]
        max_len = max(len(x) for x, _ in batch)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        targets = torch.full((batch_size, max_len), -100, dtype=torch.long)
        for idx, (x, y) in enumerate(batch):
            input_ids[idx, : len(x)] = torch.tensor(x, dtype=torch.long)
            targets[idx, : len(y)] = torch.tensor(y, dtype=torch.long)
        return input_ids.to(device), targets.to(device)


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.device("cpu"), torch.float32


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("_orig_mod.") for key in state):
        return {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    return state


def load_model(init_checkpoint: Path, config_path: Path | None, device: torch.device) -> ArchimedesMathModel:
    checkpoint = torch.load(init_checkpoint, map_location="cpu")
    if config_path is not None:
        cfg = ModelConfig.from_json(config_path)
    else:
        raw_cfg = checkpoint.get("config")
        if raw_cfg is None:
            raise ValueError("checkpoint has no embedded config; pass --config")
        cfg = ModelConfig(**raw_cfg)
    model = ArchimedesMathModel(cfg)
    model.load_state_dict(normalize_state_dict(checkpoint["model"]))
    return model.to(device)


@torch.no_grad()
def evaluate(model: ArchimedesMathModel, dataset: SFTDataset, batch_size: int, eval_batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = dataset.sample_batch(batch_size, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, val_loss: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": raw_model.cfg.__dict__,
            "step": step,
            "val_loss": val_loss,
            "stage": "sft",
        },
        path,
    )


def split_examples(path: Path, train_path: Path, val_path: Path, val_frac: float, seed: int) -> None:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_count = max(1, int(len(rows) * val_frac))
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervised fine-tune Archimedes-Math on worked math solutions.")
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_DATA_ROOT / "tokenizers" / "archimedes_math_bpe_32768_v2" / "tokenizer.json")
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-name", default="archimedes_math_sft")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--end-marker", default=DEFAULT_END_MARKER)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device, amp_dtype = device_and_dtype()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.init_checkpoint, args.config, device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    ckpt_dir = args.data_root / "checkpoints" / args.run_name
    metrics_path = ckpt_dir / "metrics.jsonl"
    split_dir = ckpt_dir / "sft_split"
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "val.jsonl"
    split_examples(args.sft_jsonl, train_path, val_path, args.val_frac, args.seed)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    train_data = SFTDataset(train_path, tokenizer, raw_model.cfg.context_length, seed=args.seed, end_marker=args.end_marker)
    val_data = SFTDataset(val_path, tokenizer, raw_model.cfg.context_length, seed=args.seed + 1, end_marker=args.end_marker)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "stage": "sft",
        "init_checkpoint": str(args.init_checkpoint),
        "parameters": raw_model.num_parameters(),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "end_marker": args.end_marker,
    }), flush=True)

    model.train()
    t0 = time.time()
    last_val = None
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        supervised_tokens = 0
        for _ in range(args.grad_accum):
            x, y = train_data.sample_batch(args.batch_size, device)
            supervised_tokens += int((y != -100).sum().item())
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                assert loss is not None
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == 1:
            elapsed = max(1e-6, time.time() - t0)
            msg = {
                "step": step,
                "train_loss": loss_accum,
                "lr": args.lr,
                "supervised_tokens_per_sec": supervised_tokens * args.log_interval / elapsed if step % args.log_interval == 0 else supervised_tokens / elapsed,
            }
            print(json.dumps(msg), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(msg) + "\n")
            t0 = time.time()

        if step % args.eval_interval == 0 or step == args.max_steps:
            last_val = evaluate(model, val_data, args.batch_size, args.eval_batches, device)
            msg = {"step": step, "val_loss": last_val, "val_ppl": math.exp(min(20.0, last_val))}
            print(json.dumps(msg), flush=True)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(msg) + "\n")

        if step % args.save_interval == 0 or step == args.max_steps:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, last_val)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
