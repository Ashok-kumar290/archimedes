#!/usr/bin/env python3
"""Chemistry reasoning curriculum: word problem -> pick relation -> substitute ->
compute with digit-level arithmetic (reusing the math/physics narration).

Same rigor as build_physics_curriculum.py: every input is chosen so the answer
is an exact integer, and every answer is re-derived independently by the family
function. All quantities the problem needs (atomic masses, molar masses) are
STATED in the prompt, so the task is pure computation from given numbers — no
hidden chemical knowledge, fully verifiable, exactly like the physics lobe.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reasoning_curriculum import (  # reuse the verified arithmetic narration
    add_narration,
    mul_narration,
    longdiv_narration,
    add_trace,
    mul_trace,
    div_trace,
    digits_trace,
    emit,
)


# Fixed eval prompts — never allowed in training (held out for honest eval).
EVAL_HOLDOUT = {
    "A molecule has 2 atoms of atomic mass 1 and 1 atom of atomic mass 16. Find its molar mass.",
    "How many moles are in 44 g of a substance with molar mass 4 g/mol?",
    "Find the mass of 7 mol of a substance with molar mass 18 g/mol.",
    "A solution has 12 mol of solute in 3 L. Find its molarity.",
    "In a reaction, 1 mol of A yields B in a 1:4 ratio. How many moles of B form from 9 mol of A?",
}


def mul_step(a: int, b: int) -> tuple[str, int]:
    steps, ans = mul_narration(a, b)
    return " ".join(steps), ans


def div_step(a: int, b: int) -> tuple[str, int]:
    narration, q = longdiv_narration(a, b)
    return narration, q


# --- problem families: each returns (prompt, completion, answer_int) ---

def f_molar_mass_mono(rng):
    # single-element molecule: molar mass = count * atomic_mass (one multiply)
    n, m = rng.randint(2, 30), rng.randint(2, 40)
    narr, M = mul_step(n, m)
    prompt = f"A molecule is made of {n} atoms each of atomic mass {m}. Find its molar mass."
    completion = (
        f"Plan: molar mass is the sum of the atomic masses; with one element it is count * atomic mass. "
        f"Step 1: substitute count = {n} and atomic mass = {m}, so M = {n} * {m}. "
        f"Step 2: {narr}. Final answer: {M} g/mol."
    )
    return prompt, completion, M


def f_molar_mass_two(rng):
    # two-element molecule: M = a*m1 + b*m2 (two multiplies + add) — the multi-step
    # family, deliberately included as the honest hard case (like physics potential)
    a, m1 = rng.randint(1, 6), rng.randint(1, 20)
    b, m2 = rng.randint(1, 6), rng.randint(1, 20)
    n1, p1 = mul_step(a, m1)
    n2, p2 = mul_step(b, m2)
    na, M = add_narration(p1, p2)
    prompt = (f"A molecule has {a} atoms of atomic mass {m1} and {b} atoms of atomic mass {m2}. "
              f"Find its molar mass.")
    completion = (
        f"Plan: molar mass = (count1 * mass1) + (count2 * mass2). "
        f"Step 1: first part, {a} * {m1}: {n1}, giving {p1}. "
        f"Step 2: second part, {b} * {m2}: {n2}, giving {p2}. "
        f"Step 3: add them, {p1} + {p2}: {na}. Final answer: {M} g/mol."
    )
    return prompt, completion, M


def f_moles(rng):
    # n = mass / molar_mass ; single-digit molar mass keeps division tractable
    M, n = rng.randint(2, 9), rng.randint(2, 60)
    mass = M * n
    narr, ans = div_step(mass, M)
    prompt = f"How many moles are in {mass} g of a substance with molar mass {M} g/mol?"
    completion = (
        f"Plan: moles = mass / molar mass. "
        f"Step 1: substitute mass = {mass} g and molar mass = {M} g/mol, so n = {mass} / {M}. "
        f"Step 2: {narr}. Final answer: {ans} mol."
    )
    return prompt, completion, ans


def f_mass(rng):
    # mass = moles * molar_mass (one multiply)
    n, M = rng.randint(2, 40), rng.randint(2, 99)
    narr, mass = mul_step(n, M)
    prompt = f"Find the mass of {n} mol of a substance with molar mass {M} g/mol."
    completion = (
        f"Plan: mass = moles * molar mass. "
        f"Step 1: substitute moles = {n} mol and molar mass = {M} g/mol, so m = {n} * {M}. "
        f"Step 2: {narr}. Final answer: {mass} g."
    )
    return prompt, completion, mass


def f_molarity(rng):
    # c = moles / volume ; single-digit volume divisor
    V, c = rng.randint(2, 9), rng.randint(2, 40)
    n = c * V
    narr, ans = div_step(n, V)
    prompt = f"A solution has {n} mol of solute in {V} L. Find its molarity."
    completion = (
        f"Plan: molarity = moles / volume. "
        f"Step 1: substitute moles = {n} mol and volume = {V} L, so c = {n} / {V}. "
        f"Step 2: {narr}. Final answer: {ans} mol/L."
    )
    return prompt, completion, ans


def f_stoich(rng):
    # n_B = n_A * ratio (one multiply), ratio 1:k
    nA, k = rng.randint(2, 40), rng.randint(2, 9)
    narr, nB = mul_step(nA, k)
    prompt = (f"In a reaction, 1 mol of A yields B in a 1:{k} ratio. "
              f"How many moles of B form from {nA} mol of A?")
    completion = (
        f"Plan: moles of B = moles of A * ratio. "
        f"Step 1: substitute moles of A = {nA} and ratio = {k}, so n_B = {nA} * {k}. "
        f"Step 2: {narr}. Final answer: {nB} mol."
    )
    return prompt, completion, nB


# --- arithmetic drill families (same rationale as physics: reinforce the
# digit-level ops the relations rest on, especially division) ---

def d_mul(rng):
    a, b = rng.randint(12, 99), rng.randint(2, 99)
    return f"Compute {a} * {b}.", mul_trace(a, b), a * b


def d_div(rng):
    b, q = rng.randint(2, 9), rng.randint(12, 999)
    a = b * q
    return f"Compute {a} / {b}.", div_trace(a, b), q


def d_add(rng):
    a, b = rng.randint(10, 9999), rng.randint(10, 9999)
    return f"Compute {a} + {b}.", add_trace(a, b), a + b


# Explicit per-family weights (see build_physics_curriculum for rationale):
# division-bearing relations + the d_div drill are weighted heavy because
# division is the lobe's measured weak spot. Single-step relations dominate;
# the two-element molar mass is the one multi-step family, kept modest.
WEIGHTED = [
    (f_molar_mass_mono, 6), (f_mass, 6), (f_stoich, 6),
    (f_moles, 9), (f_molarity, 9),
    (f_molar_mass_two, 6),
    (d_div, 14), (d_mul, 8), (d_add, 5),
]


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    total_weight = sum(w for _, w in WEIGHTED)
    rows: list[str] = []
    for fn, weight in WEIGHTED:
        target = int(count * weight / total_weight)
        seen: set[str] = set()
        pool: list[str] = []
        stale = 0
        while len(pool) < target and stale < 40_000:
            prompt, completion, _ = fn(rng)
            if prompt in EVAL_HOLDOUT or prompt in seen:
                stale += 1
                continue
            stale = 0
            seen.add(prompt)
            pool.append(emit(prompt, completion))
        if pool:
            rows.extend(pool[i % len(pool)] for i in range(target))
    rng.shuffle(rows)
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args(argv)
    rows = build(args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
