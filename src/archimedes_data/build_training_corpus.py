from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
EXCLUDED_SOURCES = {
    "project_gutenberg_public_domain_seed",
    "wikipedia_science_seed",
    "arxiv_science_seed",
}
LOWER_WEIGHT_SOURCES = {
    "math_arxiv_metadata_core",
}


def quality_score(source_name: str, text: str) -> float:
    if not text:
        return 0.0
    chars = len(text)
    lines = text.count("\n") + 1
    math_signal = len(re.findall(r"(?i)\b(theorem|proof|lemma|corollary|definition|equation|integral|matrix|algebra|calculus|geometry|topology|probability)\b", text))
    symbol_signal = len(re.findall(r"[=+\-*/∑∫√≤≥∈∀∃]", text))
    repeated_space_penalty = min(0.25, len(re.findall(r" {8,}", text)) / max(1, lines) * 0.02)
    too_short_penalty = 0.35 if chars < 1_000 else 0.0
    too_long_line_penalty = 0.15 if max((len(line) for line in text.splitlines()), default=0) > 20_000 else 0.0
    source_penalty = 0.1 if source_name in LOWER_WEIGHT_SOURCES else 0.0
    signal = min(0.35, (math_signal / max(1, chars / 2000)) * 0.04)
    symbols = min(0.20, (symbol_signal / max(1, chars / 3000)) * 0.03)
    score = 0.65 + signal + symbols - repeated_space_penalty - too_short_penalty - too_long_line_penalty - source_penalty
    return max(0.0, min(1.0, score))


def init_output(data_root: Path) -> tuple[Path, Path]:
    manifest_dir = data_root / "metadata"
    shard_dir = data_root / "shards" / "math_text_v1"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir, shard_dir


def load_docs(conn: sqlite3.Connection, min_score: float) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT doc_id, source_name, source_url, domain, license, cleaned_path, token_estimate
        FROM documents
        WHERE status = 'ok'
        ORDER BY source_name, doc_id
        """
    )
    docs: list[dict[str, object]] = []
    for doc_id, source_name, source_url, domain, license_name, cleaned_path, token_estimate in rows:
        if source_name in EXCLUDED_SOURCES:
            continue
        path = Path(cleaned_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        score = quality_score(source_name, text)
        if score < min_score:
            continue
        docs.append(
            {
                "doc_id": doc_id,
                "source_name": source_name,
                "source_url": source_url,
                "domain": domain,
                "license": license_name,
                "cleaned_path": str(path),
                "token_estimate": int(token_estimate or 0),
                "quality_score": round(score, 4),
                "char_count": len(text),
            }
        )
    return docs


def write_jsonl(path: Path, docs: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")


def write_text_shards(shard_dir: Path, split: str, docs: list[dict[str, object]], target_chars: int) -> int:
    shard_idx = 0
    chars = 0
    out = None
    try:
        for doc in docs:
            if out is None or chars >= target_chars:
                if out is not None:
                    out.close()
                shard_idx += 1
                chars = 0
                out = (shard_dir / f"{split}-{shard_idx:05d}.txt").open("w", encoding="utf-8")
            text = Path(str(doc["cleaned_path"])).read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            out.write(f"<|doc|>{doc['doc_id']} source={doc['source_name']} quality={doc['quality_score']}\n")
            out.write(text)
            out.write("\n<|endofdoc|>\n")
            chars += len(text)
    finally:
        if out is not None:
            out.close()
    return shard_idx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build quality-filtered math text shards.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument("--validation-frac", type=float, default=0.005)
    parser.add_argument("--target-chars", type=int, default=80_000_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    manifest_dir, shard_dir = init_output(args.data_root)
    conn = sqlite3.connect(args.data_root / "metadata" / "documents.sqlite")
    docs = load_docs(conn, args.min_score)

    rng = random.Random(args.seed)
    rng.shuffle(docs)
    val_count = max(1, int(len(docs) * args.validation_frac))
    val_docs = docs[:val_count]
    train_docs = docs[val_count:]

    write_jsonl(manifest_dir / "math_train_manifest_v1.jsonl", train_docs)
    write_jsonl(manifest_dir / "math_val_manifest_v1.jsonl", val_docs)
    train_shards = write_text_shards(shard_dir, "train", train_docs, args.target_chars)
    val_shards = write_text_shards(shard_dir, "val", val_docs, args.target_chars)

    train_tokens = sum(int(doc["token_estimate"]) for doc in train_docs)
    val_tokens = sum(int(doc["token_estimate"]) for doc in val_docs)
    print(f"selected_docs={len(docs)} train_docs={len(train_docs)} val_docs={len(val_docs)}")
    print(f"estimated_tokens train={train_tokens} val={val_tokens} total={train_tokens + val_tokens}")
    print(f"text_shards train={train_shards} val={val_shards} dir={shard_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

