#!/usr/bin/env python3
"""Combine benchmark_arithmetic.py result files into one comparison table.

  python3 scripts/make_bench_report.py \
      Archimedes-117M=bench_archimedes_v6.jsonl \
      GPT-2-124M=bench_gpt2.jsonl \
      Pythia-160M=bench_pythia160m.jsonl \
      --out reports/benchmark_comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[bool]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            stats.setdefault(row["family"], []).append(bool(row["hit"]))
    return {fam: (sum(hits), len(hits)) for fam, hits in stats.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="LABEL=results.jsonl pairs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    models: list[tuple[str, dict[str, tuple[int, int]]]] = []
    for spec in args.runs:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=path, got: {spec}")
        models.append((label, load(Path(path))))

    families = sorted({fam for _, stats in models for fam in stats})
    lines = ["| family | " + " | ".join(label for label, _ in models) + " |"]
    lines.append("|" + "---|" * (len(models) + 1))
    for fam in families:
        cells = []
        for _, stats in models:
            hit, n = stats.get(fam, (0, 0))
            cells.append(f"{hit / n:.0%} ({hit}/{n})" if n else "-")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    overall = []
    for _, stats in models:
        hit = sum(h for h, _ in stats.values())
        n = sum(c for _, c in stats.values())
        overall.append(f"**{hit / n:.0%}** ({hit}/{n})" if n else "-")
    lines.append("| **overall** | " + " | ".join(overall) + " |")

    table = "\n".join(lines)
    print(table)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Arithmetic benchmark comparison\n\n"
            "Same seeded held-out problem set for every model "
            "(`scripts/benchmark_arithmetic.py`, seed 777, 25 problems per family). "
            "Archimedes answers with step-by-step neural reasoning, no tools; "
            "baselines get few-shot direct-answer prompting.\n\n"
        )
        args.out.write_text(header + table + "\n", encoding="utf-8")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
