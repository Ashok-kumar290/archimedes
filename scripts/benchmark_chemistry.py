#!/usr/bin/env python3
"""Held-out chemistry benchmark: novel seeded problems (disjoint from training),
solved by the checkpoint's own step-by-step reasoning. Mirrors
benchmark_physics.py exactly — same rigor, same backends, same scoring.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_arithmetic as ba
from benchmark_arithmetic import ArchimedesBackend, HFBackend
from benchmark_physics import observed_number  # exponent-safe last-number extraction


def gen_problems(per_family: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    problems: list[dict] = []

    def add(family: str, prompt: str, expected: int) -> None:
        problems.append({"family": family, "prompt": prompt, "expected": str(expected)})

    for _ in range(per_family):
        n, m = rng.randint(2, 30), rng.randint(2, 40)
        add("molar_mass_mono", f"A molecule is made of {n} atoms each of atomic mass {m}. Find its molar mass.", n * m)

        a, m1 = rng.randint(1, 6), rng.randint(1, 20)
        b, m2 = rng.randint(1, 6), rng.randint(1, 20)
        add("molar_mass_two", f"A molecule has {a} atoms of atomic mass {m1} and {b} atoms of atomic mass {m2}. Find its molar mass.", a * m1 + b * m2)

        M, q = rng.randint(2, 9), rng.randint(2, 60)  # single-digit molar mass divisor
        add("moles", f"How many moles are in {M * q} g of a substance with molar mass {M} g/mol?", q)

        n, M = rng.randint(2, 40), rng.randint(2, 99)
        add("mass", f"Find the mass of {n} mol of a substance with molar mass {M} g/mol.", n * M)

        V, c = rng.randint(2, 9), rng.randint(2, 40)  # single-digit volume divisor
        add("molarity", f"A solution has {c * V} mol of solute in {V} L. Find its molarity.", c)

        nA, k = rng.randint(2, 40), rng.randint(2, 9)
        add("stoich", f"In a reaction, 1 mol of A yields B in a 1:{k} ratio. How many moles of B form from {nA} mol of A?", nA * k)

        d, V = rng.randint(2, 20), rng.randint(2, 9)  # single-digit volume divisor
        add("density", f"A sample has mass {d * V} g and volume {V} mL. Find its density.", d)

        while True:
            C1, V1, V2 = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 9)
            if (C1 * V1) % V2 == 0:
                break
        add("dilution", f"A {C1} mol/L solution of volume {V1} L is adjusted to a volume of {V2} L. Find the new concentration.", C1 * V1 // V2)

        m, c, dT = rng.randint(2, 20), rng.randint(1, 9), rng.randint(2, 20)
        add("heat", f"How much heat is absorbed when {m} g of a substance with specific heat {c} J/(g°C) is heated by {dT} °C?", m * c * dT)
    return problems


CHEM_FEW_SHOT = (
    ("A molecule is made of 3 atoms each of atomic mass 4. Find its molar mass.", "12"),
    ("How many moles are in 30 g of a substance with molar mass 5 g/mol?", "6"),
    ("Find the mass of 4 mol of a substance with molar mass 8 g/mol.", "32"),
    ("A solution has 20 mol of solute in 4 L. Find its molarity.", "5"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-chat", action="store_true")
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=99991)  # disjoint from training seed 2718
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    problems = gen_problems(args.per_family, args.seed)
    if args.hf_model is not None:
        ba.FEW_SHOT_PAIRS = CHEM_FEW_SHOT
        ba.FEW_SHOT = "".join(f"Problem: {p}\nAnswer: {a}\n\n" for p, a in CHEM_FEW_SHOT)
        backend = HFBackend(args.hf_model, chat=args.hf_chat)
    else:
        if args.checkpoint is None or args.tokenizer is None:
            raise SystemExit("pass either --hf-model, or both --checkpoint and --tokenizer")
        backend = ArchimedesBackend(args.checkpoint, args.tokenizer, args.max_new_tokens)

    results = []
    per_family: dict[str, list[bool]] = {}
    for i, p in enumerate(problems):
        answer_text = backend.answer(p["prompt"])
        obs = observed_number(answer_text)
        hit = obs is not None and obs == int(p["expected"])
        per_family.setdefault(p["family"], []).append(hit)
        results.append({**p, "observed": obs, "answer_text": answer_text,
                        "hit": hit, "raw": getattr(backend, "last_raw", None)})
        if (i + 1) % 25 == 0:
            done = sum(1 for r in results if r["hit"])
            print(f"[{i + 1}/{len(problems)}] running accuracy {done / (i + 1):.1%}", flush=True)

    print(f"\nmodel: {backend.name}")
    print(f"{'family':<16} {'accuracy':>9}  n")
    total = 0
    for fam in sorted(per_family):
        hits = per_family[fam]
        total += sum(hits)
        print(f"{fam:<16} {sum(hits) / len(hits):>8.1%}  {len(hits)}")
    print(f"{'OVERALL':<16} {total / len(results):>8.1%}  {len(results)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as h:
            for row in results:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"detailed results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
