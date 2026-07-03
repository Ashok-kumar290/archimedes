from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Reuse the generic, domain-agnostic helpers from the math curator so the two
# pipelines stay consistent (same dedup and manifest format).
from archimedes_data.curate_math import duplicate_key, write_manifest

DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")

# Sources that make up the physics corpus (see configs/physics_*.yaml).
PHYSICS_SOURCES = (
    "wikipedia_physics_core",
    "wikipedia_physics_expanded",
    "wikibooks_physics",
    "physics_arxiv_papers",
    "physics_arxiv_metadata_bulk",
    "arxiv_physics_education",
    "gutenberg_physics_classics",
)

PHYSICS_TOPICS = {
    "mechanics": r"\b(force|newton|momentum|velocity|acceleration|torque|friction|collision|kinematic|gravit)\b",
    "energy_work": r"\b(energy|work|power|kinetic|potential|conservation|joule|watt)\b",
    "thermodynamics": r"\b(thermodynam|heat|temperature|entropy|enthalpy|gas law|carnot|thermal)\b",
    "waves_optics": r"\b(wave|frequency|wavelength|diffraction|interference|refraction|lens|optic|sound|resonance)\b",
    "electromagnetism": r"\b(electric|magnetic|charge|current|voltage|circuit|capacit|induct|field|maxwell|coulomb|ohm)\b",
    "modern_physics": r"\b(quantum|relativ|photon|electron|nucle|particle|spin|uncertainty|schr[oö]dinger|planck)\b",
    "astro": r"\b(orbit|planet|star|galaxy|cosmo|kepler|astro)\b",
}


def source_family(source: str) -> str:
    if "arxiv" in source and "metadata" not in source:
        return "research_paper"
    if "metadata" in source:
        return "abstract_metadata"
    if "wikipedia" in source or "wikibooks" in source:
        return "expository"
    if "gutenberg" in source:
        return "book"
    return "other"


def topic_for(text: str) -> str:
    sample = text[:80_000].lower()
    scores = {t: len(re.findall(p, sample)) for t, p in PHYSICS_TOPICS.items()}
    topic, score = max(scores.items(), key=lambda kv: kv[1])
    return topic if score > 0 else "general_physics"


def quality_metrics(text: str) -> tuple[float, float]:
    """Return (physics_term_density, extraction_score) — both in [0, 1]."""
    chars = max(1, len(text))
    terms = len(re.findall(
        r"(?i)\b(force|energy|velocity|acceleration|momentum|field|charge|current|wave|"
        r"frequency|mass|equation|magnetic|electric|quantum|particle|heat|temperature|"
        r"pressure|density|voltage|photon|relativ)\b", text))
    term_density = min(1.0, terms / (chars / 1500))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    long_line = max((len(ln) for ln in lines), default=0)
    extraction = 0.65
    if long_line < 20_000:
        extraction += 0.20
    if "�" not in text:
        extraction += 0.10
    if chars > 1000:
        extraction += 0.05
    return term_density, min(1.0, extraction)


def collect_eligible(data_root: Path, min_chars: int) -> tuple[list[dict], dict[str, int]]:
    conn = sqlite3.connect(data_root / "metadata" / "documents.sqlite")
    placeholders = ",".join("?" for _ in PHYSICS_SOURCES)
    rows = conn.execute(
        f"""
        SELECT doc_id, source_name, source_url, license, cleaned_path, token_estimate
        FROM documents
        WHERE status='ok' AND source_name IN ({placeholders})
        ORDER BY source_name, doc_id
        """,
        PHYSICS_SOURCES,
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
        term_density, extraction = quality_metrics(text)
        # abstract-metadata dumps are low value per token; keep but flag topic
        docs.append({
            "doc_id": doc_id,
            "source_name": source,
            "source_url": url,
            "license": license_text,
            "cleaned_path": cleaned_path,
            "token_estimate": int(tok or 0),
            "source_family": source_family(source),
            "topic": topic_for(text),
            "term_density": round(term_density, 4),
            "extraction_score": round(extraction, 4),
        })
        reasons["kept"] += 1
    return docs, dict(reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build physics train/val manifests from the retrieved corpus.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--min-chars", type=int, default=400,
                        help="drop cleaned docs shorter than this (garbled/empty extractions)")
    parser.add_argument("--validation-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    docs, reasons = collect_eligible(args.data_root, args.min_chars)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    val_count = max(1, int(len(docs) * args.validation_frac))
    val_docs, train_docs = docs[:val_count], docs[val_count:]

    meta = args.data_root / "metadata"
    write_manifest(meta / "physics_train_manifest_v1.jsonl", train_docs)
    write_manifest(meta / "physics_val_manifest_v1.jsonl", val_docs)

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
