from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from archimedes_data.curate_math import duplicate_key, write_manifest

DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")

# Permissive Python sources for the code lobe (PSF stdlib + MIT repos).
CODE_SOURCES = (
    "python_stdlib",
    "thealgorithms_python",
    "keon_algorithms",
    "geekcomputers_python",
)

CODE_TOPICS = {
    "sorting_searching": r"\b(sort|bubble|merge|quick|heap|binary[_ ]search|bisect|insertion)\b",
    "graphs": r"\b(graph|dijkstra|bfs|dfs|adjacency|topological|spanning|node|edge)\b",
    "strings": r"\b(str|string|substring|palindrome|anagram|regex|split|join|char)\b",
    "math_numeric": r"\b(prime|factorial|fibonacci|gcd|modulo|matrix|sqrt|combinatoric|sieve)\b",
    "data_structures": r"\b(stack|queue|linked[_ ]?list|tree|heap|hash|deque|trie|class )\b",
    "dynamic_prog": r"\b(dynamic|memo|knapsack|subsequence|dp\b|recursion|recursive)\b",
    "io_system": r"\b(open\(|read|write|os\.|sys\.|argv|path|file|socket|json)\b",
}


def source_family(source: str) -> str:
    if "stdlib" in source:
        return "stdlib"
    return "repo"


def topic_for(text: str) -> str:
    sample = text[:80_000].lower()
    scores = {t: len(re.findall(p, sample)) for t, p in CODE_TOPICS.items()}
    topic, score = max(scores.items(), key=lambda kv: kv[1])
    return topic if score > 0 else "general_python"


def quality_metrics(text: str) -> tuple[float, float]:
    """Return (code_structure_density, extraction_score) — both in [0, 1].

    A file is high-signal code if it has real Python structure (def/class/
    control flow) rather than being mostly comments or data. We also down-rate
    files that are >70% comment/blank lines."""
    chars = max(1, len(text))
    struct = len(re.findall(r"(?m)^\s*(def |class |return |for |while |if |import |with |try:)", text))
    struct_density = min(1.0, struct / (chars / 400))
    lines = text.splitlines()
    nonblank = [ln for ln in lines if ln.strip()]
    comment = sum(1 for ln in nonblank if ln.lstrip().startswith("#"))
    comment_frac = comment / max(1, len(nonblank))
    extraction = 0.7
    if comment_frac < 0.7:
        extraction += 0.2
    if chars > 200:
        extraction += 0.1
    return struct_density, min(1.0, extraction)


def collect_eligible(data_root: Path, min_chars: int) -> tuple[list[dict], dict[str, int]]:
    conn = sqlite3.connect(data_root / "metadata" / "documents.sqlite")
    placeholders = ",".join("?" for _ in CODE_SOURCES)
    rows = conn.execute(
        f"""
        SELECT doc_id, source_name, source_url, license, cleaned_path, token_estimate
        FROM documents
        WHERE status='ok' AND source_name IN ({placeholders})
        ORDER BY source_name, doc_id
        """,
        CODE_SOURCES,
    ).fetchall()
    conn.close()

    seen: set[str] = set()
    docs: list[dict] = []
    reasons: Counter[str] = Counter()
    for doc_id, source, url, license_text, cleaned_path, tok in rows:
        try:
            text = Path(cleaned_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            reasons["unreadable"] += 1
            continue
        if len(text) < min_chars:
            reasons["too_short"] += 1
            continue
        dkey = duplicate_key(text)
        if dkey in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(dkey)
        struct_density, extraction = quality_metrics(text)
        if struct_density < 0.02:  # near-zero real code structure = data/config file
            reasons["low_structure"] += 1
            continue
        docs.append({
            "doc_id": doc_id,
            "source_name": source,
            "source_url": url,
            "license": license_text,
            "cleaned_path": cleaned_path,
            "token_estimate": int(tok or 0),
            "source_family": source_family(source),
            "topic": topic_for(text),
            "struct_density": round(struct_density, 4),
            "extraction_score": round(extraction, 4),
        })
        reasons["kept"] += 1
    return docs, dict(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build code train/val manifests from the ingested Python corpus.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--validation-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    docs, reasons = collect_eligible(args.data_root, args.min_chars)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    val_count = max(1, int(len(docs) * args.validation_frac))
    val_docs, train_docs = docs[:val_count], docs[val_count:]

    meta = args.data_root / "metadata"
    write_manifest(meta / "code_train_manifest_v1.jsonl", train_docs)
    write_manifest(meta / "code_val_manifest_v1.jsonl", val_docs)

    total_tok = sum(d["token_estimate"] for d in docs)
    by_family = Counter()
    by_topic = Counter()
    for d in docs:
        by_family[d["source_family"]] += d["token_estimate"]
        by_topic[d["topic"]] += d["token_estimate"]
    print(f"classification: {reasons}")
    print(f"kept_docs={len(docs)} train={len(train_docs)} val={len(val_docs)} est_tokens={total_tok:,}")
    print(f"tokens_by_family={dict(by_family.most_common())}")
    print(f"tokens_by_topic={dict(by_topic.most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
