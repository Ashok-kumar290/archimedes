from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# Reuse the generic, domain-agnostic helpers so all lobe pipelines stay
# consistent (same dedup and manifest format). Mirror of curate_physics.py.
from archimedes_data.curate_math import duplicate_key, write_manifest

DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")

# Sources that make up the chemistry corpus (see configs/chemistry_*.yaml).
CHEMISTRY_SOURCES = (
    "wikipedia_chemistry_core",
    "wikipedia_chemistry_expanded",
    "wikibooks_chemistry",
    "openstax_chemistry_textbooks",
    "chemistry_arxiv_papers",
    "chemistry_arxiv_metadata_bulk",
    "gutenberg_chemistry_classics",
)

CHEMISTRY_TOPICS = {
    "stoichiometry": r"\b(mole|molar|stoichiometr|molecular mass|molar mass|limiting reagent|yield|formula unit|avogadro)\b",
    "atomic_structure": r"\b(atom|electron|proton|neutron|orbital|isotope|nucleus|quantum number|valence|shell)\b",
    "bonding": r"\b(bond|covalent|ionic|metallic|electronegativit|lewis|vsepr|hybrid|polarity|lattice)\b",
    "thermochemistry": r"\b(enthalpy|entropy|gibbs|thermochem|heat of|calorimet|exotherm|endotherm|spontaneous)\b",
    "kinetics_equilibrium": r"\b(rate|kinetic|equilibrium|le chatelier|catalys|activation energy|reaction order|collision)\b",
    "acids_bases": r"\b(acid|base|ph\b|pka|titrat|buffer|neutraliz|hydronium|hydroxide|conjugate)\b",
    "electrochem_redox": r"\b(redox|oxidation|reduction|electrochem|electrode|galvanic|electrolys|cell potential|voltaic)\b",
    "organic": r"\b(organic|hydrocarbon|alkane|alkene|alkyne|functional group|isomer|aromatic|benzene|polymer|ester|amine)\b",
    "solutions_gases": r"\b(solution|solubilit|concentration|molarit|gas law|ideal gas|partial pressure|colligative|osmosis)\b",
    "periodic": r"\b(periodic|element|group|period|transition metal|halogen|noble gas|alkali|metalloid)\b",
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
    if "openstax" in source:
        return "textbook"
    return "other"


def topic_for(text: str) -> str:
    sample = text[:80_000].lower()
    scores = {t: len(re.findall(p, sample)) for t, p in CHEMISTRY_TOPICS.items()}
    topic, score = max(scores.items(), key=lambda kv: kv[1])
    return topic if score > 0 else "general_chemistry"


def quality_metrics(text: str) -> tuple[float, float]:
    """Return (chemistry_term_density, extraction_score) — both in [0, 1]."""
    chars = max(1, len(text))
    terms = len(re.findall(
        r"(?i)\b(mole|molar|atom|molecule|ion|electron|bond|reaction|acid|base|"
        r"compound|element|concentration|equilibrium|oxidation|reduction|enthalpy|"
        r"catalyst|solution|periodic|valence|stoichiometr|organic|polymer)\b", text))
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
    placeholders = ",".join("?" for _ in CHEMISTRY_SOURCES)
    rows = conn.execute(
        f"""
        SELECT doc_id, source_name, source_url, license, cleaned_path, token_estimate
        FROM documents
        WHERE status='ok' AND source_name IN ({placeholders})
        ORDER BY source_name, doc_id
        """,
        CHEMISTRY_SOURCES,
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
    parser = argparse.ArgumentParser(description="Build chemistry train/val manifests from the retrieved corpus.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--min-chars", type=int, default=400)
    parser.add_argument("--validation-frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    docs, reasons = collect_eligible(args.data_root, args.min_chars)
    rng = random.Random(args.seed)
    rng.shuffle(docs)
    val_count = max(1, int(len(docs) * args.validation_frac))
    val_docs, train_docs = docs[:val_count], docs[val_count:]

    meta = args.data_root / "metadata"
    write_manifest(meta / "chemistry_train_manifest_v1.jsonl", train_docs)
    write_manifest(meta / "chemistry_val_manifest_v1.jsonl", val_docs)

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
