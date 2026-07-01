from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")

TOPIC_PATTERNS = {
    "algebra": r"\b(algebra|polynomial|ring|field|group|module|linear|matrix|vector)\b",
    "analysis": r"\b(analysis|limit|derivative|integral|measure|functional|operator|sobolev|banach|hilbert)\b",
    "calculus": r"\b(calculus|derivative|integral|series|taylor|differential equation)\b",
    "geometry_topology": r"\b(geometry|topology|manifold|knot|homology|cohomology|metric|surface|curve)\b",
    "number_theory": r"\b(number theory|prime|integer|diophantine|modular|galois|zeta)\b",
    "probability_statistics": r"\b(probability|statistics|stochastic|random|distribution|bayesian)\b",
    "logic_foundations": r"\b(logic|set theory|category theory|type theory|proof assistant|theorem prover)\b",
    "combinatorics_graphs": r"\b(combinatorics|graph|enumeration|partition|matching|coloring)\b",
    "optimization": r"\b(optimization|convex|gradient|lagrangian|linear programming|control)\b",
}


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_features (
            doc_id TEXT PRIMARY KEY,
            source_family TEXT NOT NULL,
            format TEXT NOT NULL,
            level TEXT NOT NULL,
            topic TEXT NOT NULL,
            mixture_bucket TEXT NOT NULL,
            quality_score REAL NOT NULL,
            duplicate_key TEXT NOT NULL,
            math_density REAL NOT NULL,
            symbol_density REAL NOT NULL,
            repetition_score REAL NOT NULL,
            extraction_score REAL NOT NULL,
            license_confidence REAL NOT NULL,
            eligible INTEGER NOT NULL,
            reason TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_bucket ON document_features(mixture_bucket)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_topic ON document_features(topic)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_features_eligible ON document_features(eligible)")


def load_mixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_for_dedupe(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 =+\-*/^().,;:]", "", text)
    return text[:20_000]


def duplicate_key(text: str) -> str:
    return hashlib.sha256(normalize_for_dedupe(text).encode("utf-8")).hexdigest()[:24]


def source_family(source: str) -> str:
    if source.startswith("formal_"):
        return "formal_proof"
    if source.startswith("contest_"):
        return "worked_problem"
    if source.startswith("textbook_"):
        return "textbook_notes"
    if source.startswith("arxiv_") or source.startswith("math_arxiv"):
        return "research"
    if source.startswith("open_notes") or source.startswith("math_wikipedia"):
        return "textbook_notes"
    if source.startswith("synthetic_"):
        return "synthetic"
    return "other"


def format_for(source: str, source_type: str) -> str:
    if source.startswith("formal_"):
        return "formal_code"
    if source.startswith("contest_"):
        return "problem_solution"
    if source.startswith("textbook_"):
        return "textbook_source"
    if source == "arxiv_math_papers":
        return "paper"
    if "metadata" in source:
        return "abstract_metadata"
    if source.startswith("open_notes") or source.startswith("math_wikipedia"):
        return "expository"
    return source_type


def level_for(source: str, text: str) -> str:
    if source.startswith("formal_"):
        return "formal_proof"
    if source == "contest_gsm8k":
        return "elementary"
    if source == "contest_hendrycks_math":
        return "high_school_olympiad"
    if source.startswith("open_notes"):
        return "high_school_undergraduate"
    if source.startswith("textbook_"):
        return "undergraduate_graduate"
    if source == "arxiv_math_papers":
        return "research"
    if "graduate" in text[:5000].lower():
        return "graduate"
    return "mixed"


def topic_for(text: str) -> str:
    sample = text[:80_000].lower()
    scores = {
        topic: len(re.findall(pattern, sample, flags=re.IGNORECASE))
        for topic, pattern in TOPIC_PATTERNS.items()
    }
    topic, score = max(scores.items(), key=lambda item: item[1])
    return topic if score > 0 else "general_math"


def bucket_for(source: str, mixture: dict) -> str:
    for bucket, spec in mixture["buckets"].items():
        if source in spec["sources"]:
            return bucket
    family = source_family(source)
    if family == "formal_proof":
        return "formal_proofs"
    if family == "worked_problem":
        return "worked_problems"
    if family == "research":
        return "research"
    if family == "textbook_notes":
        return "textbook_notes"
    if family == "synthetic":
        return "synthetic_symbolic"
    return "unassigned"


def density_metrics(text: str) -> tuple[float, float, float, float]:
    chars = max(1, len(text))
    math_terms = len(re.findall(r"(?i)\b(theorem|proof|lemma|definition|integral|matrix|algebra|geometry|topology|probability|equation|solution)\b", text))
    symbols = len(re.findall(r"[=+\-*/^_{}\\()[\]∑∫√≤≥∈∀∃]", text))
    repeated_lines = 0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        counts = Counter(lines)
        repeated_lines = sum(count for line, count in counts.items() if count > 5)
    long_line = max((len(line) for line in lines), default=0)
    math_density = min(1.0, math_terms / (chars / 1500))
    symbol_density = min(1.0, symbols / (chars / 25))
    repetition_score = min(1.0, repeated_lines / max(1, len(lines)))
    extraction_score = 0.65
    if long_line < 20_000:
        extraction_score += 0.2
    if "\ufffd" not in text:
        extraction_score += 0.1
    if len(text) > 1000:
        extraction_score += 0.05
    return math_density, symbol_density, repetition_score, min(1.0, extraction_score)


def license_confidence(license_text: str) -> float:
    lower = license_text.lower()
    if any(term in lower for term in ("apache", "mit", "bsd", "public domain", "cc-by", "lgpl")):
        return 0.95
    if "see-repository-license" in lower:
        return 0.80
    if "verify" in lower or "varies" in lower:
        return 0.45
    return 0.60


def quality_score(source: str, text: str, license_text: str) -> tuple[float, dict[str, float]]:
    math_density, symbol_density, repetition_score, extraction_score = density_metrics(text)
    lic = license_confidence(license_text)
    source_base = {
        "formal_proof": 0.82,
        "worked_problem": 0.78,
        "textbook_notes": 0.80,
        "research": 0.68,
        "synthetic": 0.75,
    }.get(source_family(source), 0.60)
    score = (
        source_base
        + 0.12 * math_density
        + 0.08 * symbol_density
        + 0.10 * extraction_score
        + 0.08 * lic
        - 0.20 * repetition_score
    )
    if len(text) < 700:
        score -= 0.25
    if source == "math_arxiv_metadata_core":
        score -= 0.10
    return max(0.0, min(1.0, score)), {
        "math_density": math_density,
        "symbol_density": symbol_density,
        "repetition_score": repetition_score,
        "extraction_score": extraction_score,
        "license_confidence": lic,
    }


def classify_all(data_root: Path, mixture: dict) -> dict[str, int]:
    conn = sqlite3.connect(data_root / "metadata" / "documents.sqlite")
    ensure_tables(conn)
    excluded = set(mixture.get("exclude_sources", []))
    rows = conn.execute(
        """
        SELECT doc_id, source_name, source_url, domain, license, source_type, cleaned_path
        FROM documents
        WHERE status='ok'
        ORDER BY source_name, doc_id
        """
    )
    seen_dupes: set[str] = set()
    counts: Counter[str] = Counter()
    for doc_id, source, _url, _domain, license_text, source_type, cleaned_path in rows:
        path = Path(cleaned_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        bucket = bucket_for(source, mixture)
        q, metrics = quality_score(source, text, license_text)
        dkey = duplicate_key(text)
        duplicate = dkey in seen_dupes
        seen_dupes.add(dkey)
        reason = "ok"
        eligible = 1
        if source in excluded:
            eligible, reason = 0, "excluded_source"
        elif duplicate:
            eligible, reason = 0, "near_duplicate"
        elif bucket == "unassigned":
            eligible, reason = 0, "unassigned_bucket"
        else:
            min_quality = mixture["buckets"].get(bucket, {}).get("min_quality", 0.5)
            if q < min_quality:
                eligible, reason = 0, "below_quality_threshold"
        conn.execute(
            """
            INSERT OR REPLACE INTO document_features (
                doc_id, source_family, format, level, topic, mixture_bucket, quality_score,
                duplicate_key, math_density, symbol_density, repetition_score, extraction_score,
                license_confidence, eligible, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                source_family(source),
                format_for(source, source_type),
                level_for(source, text),
                topic_for(text),
                bucket,
                round(q, 4),
                dkey,
                round(metrics["math_density"], 4),
                round(metrics["symbol_density"], 4),
                round(metrics["repetition_score"], 4),
                round(metrics["extraction_score"], 4),
                round(metrics["license_confidence"], 4),
                eligible,
                reason,
            ),
        )
        counts[f"{bucket}:{reason}"] += 1
    conn.commit()
    conn.close()
    return dict(counts)


def load_eligible_docs(data_root: Path) -> list[dict]:
    conn = sqlite3.connect(data_root / "metadata" / "documents.sqlite")
    rows = conn.execute(
        """
        SELECT d.doc_id, d.source_name, d.source_url, d.license, d.cleaned_path,
               d.token_estimate, f.mixture_bucket, f.topic, f.level, f.format, f.quality_score
        FROM documents d
        JOIN document_features f ON d.doc_id = f.doc_id
        WHERE d.status='ok' AND f.eligible=1
        ORDER BY d.source_name, d.doc_id
        """
    )
    docs = []
    for row in rows:
        docs.append(
            {
                "doc_id": row[0],
                "source_name": row[1],
                "source_url": row[2],
                "license": row[3],
                "cleaned_path": row[4],
                "token_estimate": int(row[5] or 0),
                "mixture_bucket": row[6],
                "topic": row[7],
                "level": row[8],
                "format": row[9],
                "quality_score": float(row[10]),
            }
        )
    conn.close()
    return docs


def weighted_sample(docs: list[dict], mixture: dict, target_tokens: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        by_bucket[doc["mixture_bucket"]].append(doc)
    selected: list[dict] = []
    for bucket, spec in mixture["buckets"].items():
        candidates = by_bucket.get(bucket, [])
        if not candidates:
            continue
        bucket_target = int(target_tokens * float(spec["target_fraction"]))
        candidates = sorted(candidates, key=lambda d: (d["quality_score"], d["token_estimate"]), reverse=True)
        # Shuffle within quality bands to avoid fixed source ordering.
        top_pool = candidates[:]
        rng.shuffle(top_pool)
        top_pool.sort(key=lambda d: d["quality_score"], reverse=True)
        total = 0
        i = 0
        while total < bucket_target and i < len(top_pool):
            doc = dict(top_pool[i])
            doc["selected_bucket_target"] = bucket_target
            selected.append(doc)
            total += int(doc["token_estimate"])
            i += 1
    rng.shuffle(selected)
    return selected


def write_manifest(path: Path, docs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(data_root: Path, docs: list[dict], counts: dict[str, int]) -> None:
    report_dir = data_root / "metadata" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    by_bucket = Counter()
    by_topic = Counter()
    by_format = Counter()
    for doc in docs:
        by_bucket[doc["mixture_bucket"]] += int(doc["token_estimate"])
        by_topic[doc["topic"]] += int(doc["token_estimate"])
        by_format[doc["format"]] += int(doc["token_estimate"])
    report = {
        "selected_docs": len(docs),
        "selected_estimated_tokens": sum(int(doc["token_estimate"]) for doc in docs),
        "classification_counts": counts,
        "tokens_by_bucket": dict(by_bucket.most_common()),
        "tokens_by_topic": dict(by_topic.most_common()),
        "tokens_by_format": dict(by_format.most_common()),
    }
    (report_dir / "math_curation_v2_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify, score, and build weighted math v2 manifests.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--mixture", type=Path, default=Path("configs/data/math_mixture_v2.json"))
    parser.add_argument("--target-tokens", type=int, default=120_000_000)
    parser.add_argument("--validation-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    mixture = load_mixture(args.mixture)
    counts = classify_all(args.data_root, mixture)
    eligible_docs = load_eligible_docs(args.data_root)
    selected = weighted_sample(eligible_docs, mixture, args.target_tokens, args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(selected)
    val_count = max(1, int(len(selected) * args.validation_frac))
    val_docs = selected[:val_count]
    train_docs = selected[val_count:]
    write_manifest(args.data_root / "metadata" / "math_train_manifest_v2.jsonl", train_docs)
    write_manifest(args.data_root / "metadata" / "math_val_manifest_v2.jsonl", val_docs)
    write_report(args.data_root, selected, counts)
    print(f"eligible_docs={len(eligible_docs)} selected_docs={len(selected)}")
    print(f"estimated_tokens train={sum(int(d['token_estimate']) for d in train_docs)} val={sum(int(d['token_estimate']) for d in val_docs)} total={sum(int(d['token_estimate']) for d in selected)}")
    print(f"report={args.data_root / 'metadata' / 'reports' / 'math_curation_v2_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

