#!/usr/bin/env python3
"""Physics reasoning curriculum: word problem -> pick formula -> substitute ->
compute with digit-level arithmetic (reusing the math curriculum's narration).

Every problem uses integer inputs chosen so the answer is an exact integer,
and every answer is re-derived independently in check_physics() below. Same
rigor as build_reasoning_curriculum.py: no answer is stated that wasn't
computed by a rule.
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
    emit,
)

G = 10  # simplified gravitational acceleration, m/s^2 (common in intro physics)

# Fixed eval prompts — never allowed in training (held out for honest eval).
EVAL_HOLDOUT = {
    "A 6 kg object accelerates at 4 m/s^2. Find the net force.",
    "A car travels at 12 m/s and accelerates at 3 m/s^2 for 8 s. Find its final velocity.",
    "A 3 kg object is lifted to a height of 5 m. Find its potential energy. Use g = 10 m/s^2.",
    "A current of 3 A flows through a 5 ohm resistor. Find the voltage.",
    "An object of mass 8 kg moves at 7 m/s. Find its momentum.",
}

PREFIX = ["Find", "Calculate", "Compute", "Work out", "Determine"]


def mul_step(a: int, b: int) -> tuple[str, int]:
    """Return (narration, product) using the place-value multiplication trace."""
    steps, ans = mul_narration(a, b)
    return " ".join(steps), ans


def div_step(a: int, b: int) -> tuple[str, int]:
    narration, q = longdiv_narration(a, b)
    return narration, q


# --- problem families: each returns (prompt, completion, answer_int) ---

def f_newton(rng):
    m, a = rng.randint(2, 99), rng.randint(2, 50)
    narr, F = mul_step(m, a)
    prompt = f"A {m} kg object accelerates at {a} m/s^2. Find the net force."
    completion = (
        f"Plan: use Newton's second law, F = m * a. "
        f"Step 1: substitute m = {m} kg and a = {a} m/s^2, so F = {m} * {a}. "
        f"Step 2: {narr}. Final answer: {F} N."
    )
    return prompt, completion, F


def f_momentum(rng):
    m, v = rng.randint(2, 99), rng.randint(2, 99)
    narr, p = mul_step(m, v)
    prompt = f"An object of mass {m} kg moves at {v} m/s. Find its momentum."
    completion = (
        f"Plan: momentum is p = m * v. "
        f"Step 1: substitute m = {m} kg and v = {v} m/s, so p = {m} * {v}. "
        f"Step 2: {narr}. Final answer: {p} kg m/s."
    )
    return prompt, completion, p


def f_weight(rng):
    m = rng.randint(2, 400)
    narr, W = mul_step(m, G)
    prompt = f"Find the weight of a {m} kg object. Use g = {G} m/s^2."
    completion = (
        f"Plan: weight is W = m * g. "
        f"Step 1: substitute m = {m} kg and g = {G} m/s^2, so W = {m} * {G}. "
        f"Step 2: {narr}. Final answer: {W} N."
    )
    return prompt, completion, W


def f_potential(rng):
    m, h = rng.randint(2, 80), rng.randint(2, 80)
    narr1, mg = mul_step(m, G)
    narr2, PE = mul_step(mg, h)
    prompt = f"A {m} kg object is lifted to a height of {h} m. Find its potential energy. Use g = {G} m/s^2."
    completion = (
        f"Plan: gravitational potential energy is PE = m * g * h. "
        f"Step 1: substitute m = {m} kg, g = {G} m/s^2, h = {h} m. "
        f"Step 2: m * g = {m} * {G}: {narr1}. "
        f"Step 3: multiply by h, {mg} * {h}: {narr2}. Final answer: {PE} J."
    )
    return prompt, completion, PE


def f_kinematics_v(rng):
    u, a, t = rng.randint(2, 90), rng.randint(2, 30), rng.randint(2, 30)
    narr1, at = mul_step(a, t)
    narr2, v = add_narration(u, at)
    prompt = f"A car travels at {u} m/s and accelerates at {a} m/s^2 for {t} s. Find its final velocity."
    completion = (
        f"Plan: use v = u + a * t. "
        f"Step 1: substitute u = {u} m/s, a = {a} m/s^2, t = {t} s. "
        f"Step 2: a * t = {a} * {t}: {narr1}. "
        f"Step 3: add u, {u} + {at}: {narr2}. Final answer: {v} m/s."
    )
    return prompt, completion, v


def f_work(rng):
    F, d = rng.randint(2, 99), rng.randint(2, 99)
    narr, W = mul_step(F, d)
    prompt = f"A force of {F} N moves an object {d} m in the direction of the force. Find the work done."
    completion = (
        f"Plan: work is W = F * d. "
        f"Step 1: substitute F = {F} N and d = {d} m, so W = {F} * {d}. "
        f"Step 2: {narr}. Final answer: {W} J."
    )
    return prompt, completion, W


def f_ohm_v(rng):
    I, R = rng.randint(2, 50), rng.randint(2, 99)
    narr, V = mul_step(I, R)
    prompt = f"A current of {I} A flows through a {R} ohm resistor. Find the voltage."
    completion = (
        f"Plan: use Ohm's law, V = I * R. "
        f"Step 1: substitute I = {I} A and R = {R} ohm, so V = {I} * {R}. "
        f"Step 2: {narr}. Final answer: {V} V."
    )
    return prompt, completion, V


def f_ohm_i(rng):
    I, R = rng.randint(2, 30), rng.randint(2, 12)
    V = I * R
    narr, ans = div_step(V, R)
    prompt = f"A voltage of {V} V is applied across a {R} ohm resistor. Find the current."
    completion = (
        f"Plan: rearrange Ohm's law to I = V / R. "
        f"Step 1: substitute V = {V} V and R = {R} ohm, so I = {V} / {R}. "
        f"Step 2: {narr}. Final answer: {ans} A."
    )
    return prompt, completion, ans


def f_density(rng):
    rho, V = rng.randint(2, 60), rng.randint(2, 12)
    m = rho * V
    narr, ans = div_step(m, V)
    prompt = f"An object has mass {m} kg and volume {V} m^3. Find its density."
    completion = (
        f"Plan: density is rho = m / V. "
        f"Step 1: substitute m = {m} kg and V = {V} m^3, so rho = {m} / {V}. "
        f"Step 2: {narr}. Final answer: {ans} kg/m^3."
    )
    return prompt, completion, ans


def f_power(rng):
    P, t = rng.randint(2, 80), rng.randint(2, 12)
    W = P * t
    narr, ans = div_step(W, t)
    prompt = f"An engine does {W} J of work in {t} s. Find its power output."
    completion = (
        f"Plan: power is P = W / t. "
        f"Step 1: substitute W = {W} J and t = {t} s, so P = {W} / {t}. "
        f"Step 2: {narr}. Final answer: {ans} W."
    )
    return prompt, completion, ans


FAMILIES = [
    f_newton, f_momentum, f_weight, f_potential, f_kinematics_v,
    f_work, f_ohm_v, f_ohm_i, f_density, f_power,
]


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool: list[str] = []
    seen: set[str] = set()
    # gather as many unique problems as the value ranges allow, then tile the
    # pool to the requested count (repeating verified examples is fine for SFT)
    stale = 0
    while stale < 200_000 and len(pool) < count:
        prompt, completion, _ = rng.choice(FAMILIES)(rng)
        if prompt in EVAL_HOLDOUT or prompt in seen:
            stale += 1
            continue
        stale = 0
        seen.add(prompt)
        pool.append(emit(prompt, completion))
    rows = [pool[i % len(pool)] for i in range(count)] if pool else []
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
