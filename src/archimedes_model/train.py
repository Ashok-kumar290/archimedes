from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from archimedes_model.data import TokenShardDataset
from archimedes_model.model import ArchimedesMathModel, ModelConfig


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.device("cpu"), torch.float32


def cosine_lr(step: int, max_steps: int, warmup_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(model: ArchimedesMathModel, dataset: TokenShardDataset, batch_size: int, eval_batches: int, device: torch.device) -> float:
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = dataset.sample_batch(batch_size, device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path: Path, model: ArchimedesMathModel, optimizer: torch.optim.Optimizer, cfg: ModelConfig, step: int, val_loss: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.__dict__,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Archimedes-Math decoder model.")
    parser.add_argument("--config", type=Path, default=Path("configs/model/archimedes_math_tiny.json"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--run-name", default="archimedes_math_tiny")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    cfg = ModelConfig.from_json(args.config)
    device, amp_dtype = device_and_dtype()
    shard_dir = args.shard_dir or (args.data_root / "shards" / "math_tokens_v1")
    train_data = TokenShardDataset(shard_dir, "train", cfg.context_length)
    val_data = TokenShardDataset(shard_dir, "val", cfg.context_length, seed=2718)

    model = ArchimedesMathModel(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    ckpt_dir = args.data_root / "checkpoints" / args.run_name
    metrics_path = ckpt_dir / "metrics.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "model": cfg.name,
        "parameters": model.num_parameters() if hasattr(model, "num_parameters") else "compiled",
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "train_tokens": train_data.total_tokens,
        "val_tokens": val_data.total_tokens,
    }), flush=True)

    model.train()
    t0 = time.time()
    last_val = None
    for step in range(1, args.max_steps + 1):
        lr = cosine_lr(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        tokens = 0
        for _ in range(args.grad_accum):
            x, y = train_data.sample_batch(args.batch_size, device)
            tokens += x.numel()
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
                "lr": lr,
                "tokens_per_sec": tokens * args.log_interval / elapsed if step % args.log_interval == 0 else tokens / elapsed,
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
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step, last_val)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

