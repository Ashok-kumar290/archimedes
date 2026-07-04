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


def f_density(rng):
    # density = mass / volume ; single-digit volume divisor
    d, V = rng.randint(2, 20), rng.randint(2, 9)
    m = d * V
    narr, ans = div_step(m, V)
    prompt = f"A sample has mass {m} g and volume {V} mL. Find its density."
    completion = (
        f"Plan: density = mass / volume. "
        f"Step 1: substitute mass = {m} g and volume = {V} mL, so density = {m} / {V}. "
        f"Step 2: {narr}. Final answer: {ans} g/mL."
    )
    return prompt, completion, ans


def f_dilution(rng):
    # C1*V1 = C2*V2 -> C2 = (C1 * V1) / V2 (multiply then divide) — multi-step
    while True:
        C1, V1, V2 = rng.randint(2, 20), rng.randint(2, 20), rng.randint(2, 9)
        if (C1 * V1) % V2 == 0:
            break
    n1, prod = mul_step(C1, V1)
    narr2, C2 = div_step(prod, V2)
    prompt = (f"A {C1} mol/L solution of volume {V1} L is adjusted to a volume of {V2} L. "
              f"Find the new concentration.")
    completion = (
        f"Plan: moles are conserved, C1 * V1 = C2 * V2, so C2 = (C1 * V1) / V2. "
        f"Step 1: C1 * V1 = {C1} * {V1}: {n1}, giving {prod}. "
        f"Step 2: divide by V2 = {V2}, {prod} / {V2}: {narr2}. Final answer: {C2} mol/L."
    )
    return prompt, completion, C2


def f_heat(rng):
    # q = m * c * ΔT computed as (m * c) * ΔT — multi-step multiply chain
    m, c, dT = rng.randint(2, 20), rng.randint(1, 9), rng.randint(2, 20)
    n1, mc = mul_step(m, c)
    n2, q = mul_step(mc, dT)
    prompt = (f"How much heat is absorbed when {m} g of a substance with specific heat "
              f"{c} J/(g°C) is heated by {dT} °C?")
    completion = (
        f"Plan: heat is q = m * c * ΔT. "
        f"Step 1: m * c = {m} * {c}: {n1}, giving {mc}. "
        f"Step 2: multiply by ΔT = {dT}, {mc} * {dT}: {n2}. Final answer: {q} J."
    )
    return prompt, completion, q


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
    (f_molar_mass_mono, 5), (f_mass, 5), (f_stoich, 5),
    (f_moles, 8), (f_molarity, 8), (f_density, 8),
    (f_molar_mass_two, 5), (f_dilution, 6), (f_heat, 5),
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
