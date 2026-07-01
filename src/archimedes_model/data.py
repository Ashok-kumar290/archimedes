from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TokenShardDataset:
    shard_dir: Path
    split: str
    context_length: int
    seed: int = 1729

    def __post_init__(self) -> None:
        self.files = sorted(self.shard_dir.glob(f"{self.split}-*.u32"))
        if not self.files:
            raise FileNotFoundError(f"no {self.split} shards found in {self.shard_dir}")
        self.arrays = [np.memmap(path, dtype=np.uint32, mode="r") for path in self.files]
        self.rng = random.Random(self.seed)

    @property
    def total_tokens(self) -> int:
        return sum(len(arr) for arr in self.arrays)

    def sample_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        xs = []
        ys = []
        for _ in range(batch_size):
            arr = self.rng.choice(self.arrays)
            if len(arr) <= self.context_length + 1:
                raise ValueError("token shard is shorter than context_length + 1")
            start = self.rng.randint(0, len(arr) - self.context_length - 1)
            chunk = np.asarray(arr[start : start + self.context_length + 1], dtype=np.int64)
            xs.append(torch.from_numpy(chunk[:-1].copy()))
            ys.append(torch.from_numpy(chunk[1:].copy()))
        x = torch.stack(xs).to(device=device, dtype=torch.long)
        y = torch.stack(ys).to(device=device, dtype=torch.long)
        return x, y

