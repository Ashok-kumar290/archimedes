from __future__ import annotations

import argparse
import json
import sys
from array import array
from pathlib import Path

from tokenizers import Tokenizer


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")


def load_docs(manifest: Path):
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def flush_tokens(path: Path, tokens: list[int]) -> None:
    arr = array("I", tokens)
    with path.open("wb") as handle:
        arr.tofile(handle)


def build_split(
    tokenizer: Tokenizer,
    manifest: Path,
    out_dir: Path,
    split: str,
    shard_tokens: int,
    limit_docs: int,
) -> tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    total_tokens = 0
    docs = 0
    buffer: list[int] = []
    end_doc = tokenizer.token_to_id("<|endofdoc|>")

    for doc in load_docs(manifest):
        if limit_docs and docs >= limit_docs:
            break
        text = Path(doc["cleaned_path"]).read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text).ids
        if end_doc is not None:
            ids.append(end_doc)
        buffer.extend(ids)
        total_tokens += len(ids)
        docs += 1
        while len(buffer) >= shard_tokens:
            shard_idx += 1
            flush_tokens(out_dir / f"{split}-{shard_idx:05d}.u32", buffer[:shard_tokens])
            del buffer[:shard_tokens]

    if buffer:
        shard_idx += 1
        flush_tokens(out_dir / f"{split}-{shard_idx:05d}.u32", buffer)
    return docs, total_tokens, shard_idx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build uint32 token-id shards.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    parser.add_argument("--limit-docs", type=int, default=0)
    args = parser.parse_args(argv)

    tokenizer_path = args.tokenizer or (args.data_root / "tokenizers" / "archimedes_math_bpe_32768" / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    out_dir = args.out_dir or (args.data_root / "shards" / "math_tokens_v1")
    train_manifest = args.train_manifest or (args.data_root / "metadata" / "math_train_manifest_v1.jsonl")
    val_manifest = args.val_manifest or (args.data_root / "metadata" / "math_val_manifest_v1.jsonl")

    train_docs, train_tokens, train_shards = build_split(tokenizer, train_manifest, out_dir, "train", args.shard_tokens, args.limit_docs)
    val_docs, val_tokens, val_shards = build_split(tokenizer, val_manifest, out_dir, "val", args.shard_tokens, 0)
    print(f"train docs={train_docs} tokens={train_tokens} shards={train_shards}")
    print(f"val docs={val_docs} tokens={val_tokens} shards={val_shards}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

