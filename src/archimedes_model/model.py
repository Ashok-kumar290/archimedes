from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    context_length: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.n_heads != 0:
            raise ValueError("hidden_size must be divisible by n_heads")
        return self.hidden_size // self.n_heads

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x: [batch, heads, seq, head_dim]
    x_float = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_float)
    freqs = freqs_cis[: x.shape[-2]].to(x.device)
    while freqs.ndim < x_complex.ndim:
        freqs = freqs.unsqueeze(0)
    rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return rotated.type_as(x)


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.cfg.n_kv_heads, self.cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)
        repeats = self.cfg.n_heads // self.cfg.n_kv_heads
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.cfg.hidden_size)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class ArchimedesMathModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.output = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output.weight = self.tok_embeddings.weight
        self.register_buffer(
            "freqs_cis",
            precompute_rope_frequencies(cfg.head_dim, cfg.context_length, cfg.rope_theta),
            persistent=False,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        if input_ids.shape[1] > self.cfg.context_length:
            raise ValueError("sequence length exceeds context_length")
        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        logits = self.output(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx = input_ids[:, -self.cfg.context_length :]
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            if repetition_penalty != 1.0:
                for batch_idx in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[batch_idx])
                    logits[batch_idx, seen] = torch.where(
                        logits[batch_idx, seen] < 0,
                        logits[batch_idx, seen] * repetition_penalty,
                        logits[batch_idx, seen] / repetition_penalty,
                    )
            if temperature <= 0.0 or top_k == 1:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                continue
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                remove = cumulative_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits[remove] = -float("inf")
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


