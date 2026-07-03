#!/usr/bin/env python3
"""Held-out physics benchmark: novel problems (seeded, disjoint from training)
solved by the checkpoint's own step-by-step reasoning — the true generalization
test for the physics lobe, analogous to benchmark_arithmetic.py for math.

Answers are computed here independently; the model only sees the word problem.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# reuse the neural-only backend from the math benchmark; there is no tool path —
# every answer is a forward pass. answer() already returns the "Final answer:"
# text, so we only need to pull the number (and drop the unit) from it.
from benchmark_arithmetic import ArchimedesBackend

G = 10


def gen_problems(per_family: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    problems: list[dict] = []

    def add(family: str, prompt: str, expected: int) -> None:
        problems.append({"family": family, "prompt": prompt, "expected": str(expected)})

    for _ in range(per_family):
        m, a = rng.randint(2, 99), rng.randint(2, 50)
        add("newton", f"A {m} kg object accelerates at {a} m/s^2. Find the net force.", m * a)
        m, v = rng.randint(2, 99), rng.randint(2, 99)
        add("momentum", f"An object of mass {m} kg moves at {v} m/s. Find its momentum.", m * v)
        m = rng.randint(2, 400)
        add("weight", f"Find the weight of a {m} kg object. Use g = {G} m/s^2.", m * G)
        m, h = rng.randint(2, 80), rng.randint(2, 80)
        add("potential", f"A {m} kg object is lifted to a height of {h} m. Find its potential energy. Use g = {G} m/s^2.", m * G * h)
        u, a, t = rng.randint(2, 90), rng.randint(2, 30), rng.randint(2, 30)
        add("kinematics", f"A car travels at {u} m/s and accelerates at {a} m/s^2 for {t} s. Find its final velocity.", u + a * t)
        F, d = rng.randint(2, 99), rng.randint(2, 99)
        add("work", f"A force of {F} N moves an object {d} m in the direction of the force. Find the work done.", F * d)
        I, R = rng.randint(2, 50), rng.randint(2, 99)
        add("ohm_v", f"A current of {I} A flows through a {R} ohm resistor. Find the voltage.", I * R)
        I, R = rng.randint(2, 30), rng.randint(2, 12)
        add("ohm_i", f"A voltage of {I * R} V is applied across a {R} ohm resistor. Find the current.", I)
        rho, V = rng.randint(2, 60), rng.randint(2, 12)
        add("density", f"An object has mass {rho * V} kg and volume {V} m^3. Find its density.", rho)
        P, t = rng.randint(2, 80), rng.randint(2, 12)
        add("power", f"An engine does {P * t} J of work in {t} s. Find its power output.", P)
    return problems


def observed_number(ans: str | None) -> int | None:
    if ans is None:
        return None
    # answer text looks like "36 N" / "150 J"; collapse spaces (digit-split
    # can space out digits) then take the leading integer, ignoring the unit
    m = re.match(r"-?\d[\d,]*", ans.replace(" ", ""))
    return int(m.group().replace(",", "")) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=25)
    parser.add_argument("--seed", type=int, default=99991)  # disjoint from training seed 2718
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    problems = gen_problems(args.per_family, args.seed)
    backend = ArchimedesBackend(args.checkpoint, args.tokenizer, args.max_new_tokens)

    results = []
    per_family: dict[str, list[bool]] = {}
    for i, p in enumerate(problems):
        raw = backend.answer(p["prompt"])  # returns the extracted final-answer text
        obs = observed_number(raw)
        hit = obs is not None and obs == int(p["expected"])
        per_family.setdefault(p["family"], []).append(hit)
        results.append({**p, "observed": obs, "hit": hit})
        if (i + 1) % 25 == 0:
            done = sum(1 for r in results if r["hit"])
            print(f"[{i + 1}/{len(problems)}] running accuracy {done / (i + 1):.1%}", flush=True)

    print(f"\nmodel: {backend.name}")
    print(f"{'family':<12} {'accuracy':>9}  n")
    total = 0
    for fam in sorted(per_family):
        hits = per_family[fam]
        total += sum(hits)
        print(f"{fam:<12} {sum(hits) / len(hits):>8.1%}  {len(hits)}")
    print(f"{'OVERALL':<12} {total / len(results):>8.1%}  {len(results)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as h:
            for row in results:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"detailed results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
