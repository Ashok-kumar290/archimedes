#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from math import gcd
from pathlib import Path


def emit(prompt: str, completion: str) -> str:
    return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


def fmt_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def add_trace(a: int, b: int) -> str:
    ans = a + b
    check = ans - b
    return (
        f"Plan: add the two integers and check by subtracting one addend. "
        f"Step 1: compute {a} + {b} = {ans}. "
        f"Check: {ans} - {b} = {check}, which recovers {a}. "
        f"Final answer: {ans}."
    )


def sub_trace(a: int, b: int) -> str:
    ans = a - b
    check = ans + b
    return (
        f"Plan: subtract and check by adding back the subtracted number. "
        f"Step 1: compute {a} - {b} = {ans}. "
        f"Check: {ans} + {b} = {check}. "
        f"Final answer: {ans}."
    )


def mul_trace(a: int, b: int) -> str:
    ans = a * b
    tens, ones = divmod(abs(b), 10)
    sign = -1 if b < 0 else 1
    parts = []
    if tens:
        parts.append(f"{a} * {sign * tens * 10} = {a * sign * tens * 10}")
    if ones:
        parts.append(f"{a} * {sign * ones} = {a * sign * ones}")
    if not parts:
        parts.append(f"{a} * 0 = 0")
    return (
        f"Plan: break the multiplication into place-value parts. "
        f"Step 1: {'; '.join(parts)}. "
        f"Step 2: add the partial products to get {ans}. "
        f"Final answer: {ans}."
    )


def linear_trace(a: int, b: int, x: int, sign: int) -> tuple[str, str]:
    c = a * x + sign * b
    op = "+" if sign > 0 else "-"
    move = c - sign * b
    prompt = f"Solve for x: {a}x {op} {b} = {c}."
    completion = (
        f"Plan: isolate x using inverse operations. "
        f"Step 1: start with {a}x {op} {b} = {c}. "
        f"Step 2: move the constant term to get {a}x = {move}. "
        f"Step 3: divide by {a}: x = {x}. "
        f"Check: {a} * {x} {op} {b} = {c}. "
        f"Final answer: x = {x}."
    )
    return prompt, completion


def fraction_add_trace(a: int, b: int, c: int, d: int) -> tuple[str, str]:
    ans = Fraction(a, b) + Fraction(c, d)
    lcm = b * d // gcd(b, d)
    left = a * (lcm // b)
    right = c * (lcm // d)
    raw_num = left + right
    prompt = f"Compute {a}/{b} + {c}/{d}."
    completion = (
        f"Plan: use a common denominator. "
        f"Step 1: a common denominator for {b} and {d} is {lcm}. "
        f"Step 2: rewrite {a}/{b} as {left}/{lcm} and {c}/{d} as {right}/{lcm}. "
        f"Step 3: add numerators: ({left} + {right})/{lcm} = {raw_num}/{lcm}. "
        f"Step 4: simplify to {fmt_fraction(ans)}. "
        f"Final answer: {fmt_fraction(ans)}."
    )
    return prompt, completion


def odd_sum_trace(n: int) -> str:
    ans = n * n
    return (
        f"Plan: use the identity that the first n odd positive integers sum to n^2. "
        f"Step 1: here n = {n}. "
        f"Step 2: compute {n}^2 = {ans}. "
        f"Check: the answer is a square, as expected for an odd-number sum. "
        f"Final answer: {ans}."
    )


def triangular_trace(n: int) -> str:
    ans = n * (n + 1) // 2
    return (
        f"Plan: use the formula 1 + 2 + ... + n = n(n + 1)/2. "
        f"Step 1: substitute n = {n}. "
        f"Step 2: compute {n} * {n + 1} / 2 = {ans}. "
        f"Final answer: {ans}."
    )


def even_square_proof(var: str) -> tuple[str, str]:
    prompt = f"Prove that if {var} is even, then {var}^2 is even."
    completion = (
        f"Plan: use the definition of an even integer. "
        f"Step 1: since {var} is even, write {var} = 2k for some integer k. "
        f"Step 2: square both sides: {var}^2 = (2k)^2 = 4k^2. "
        f"Step 3: rewrite 4k^2 as 2(2k^2). "
        f"Check: 2k^2 is an integer, so {var}^2 is divisible by 2. "
        f"Final answer: {var}^2 is even."
    )
    return prompt, completion


def odd_square_proof(var: str) -> tuple[str, str]:
    prompt = f"Prove that if {var} is odd, then {var}^2 is odd."
    completion = (
        f"Plan: use the definition of an odd integer. "
        f"Step 1: since {var} is odd, write {var} = 2k + 1 for some integer k. "
        f"Step 2: square: {var}^2 = (2k + 1)^2 = 4k^2 + 4k + 1. "
        f"Step 3: rewrite this as 2(2k^2 + 2k) + 1. "
        f"Check: 2k^2 + 2k is an integer, so {var}^2 has the form 2m + 1. "
        f"Final answer: {var}^2 is odd."
    )
    return prompt, completion


def odd_sum_proof(var: str) -> tuple[str, str]:
    prompt = f"Prove that 1 + 3 + 5 + ... + (2{var} - 1) = {var}^2."
    completion = (
        f"Plan: prove the identity by induction on {var}. "
        f"Step 1: base case {var} = 1 gives 1 = 1^2, true. "
        f"Step 2: assume 1 + 3 + ... + (2{var} - 1) = {var}^2. "
        f"Step 3: the next odd term is 2({var} + 1) - 1 = 2{var} + 1. "
        f"Step 4: adding it gives {var}^2 + 2{var} + 1 = ({var} + 1)^2. "
        f"Check: the statement holds for {var} + 1 whenever it holds for {var}. "
        f"Final answer: proved by induction."
    )
    return prompt, completion


def triangular_proof(var: str) -> tuple[str, str]:
    prompt = f"Prove that 1 + 2 + ... + {var} = {var}({var} + 1)/2."
    completion = (
        f"Plan: pair the sum with itself in reverse order. "
        f"Step 1: let S = 1 + 2 + ... + {var}. "
        f"Step 2: also S = {var} + ({var} - 1) + ... + 1. "
        f"Step 3: adding the two lines gives 2S = ({var} + 1) + ({var} + 1) + ... + ({var} + 1), with {var} terms. "
        f"Step 4: therefore 2S = {var}({var} + 1), so S = {var}({var} + 1)/2. "
        f"Final answer: proved."
    )
    return prompt, completion


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []
    vars_ = ["n", "m", "r", "t"]
    prompt_variants = ["Compute", "Find", "Evaluate"]

    anchors = [
        ("Compute 247 + 389.", add_trace(247, 389)),
        linear_trace(7, 5, 6, 1),
        ("Find the sum of the first 17 odd positive integers.", odd_sum_trace(17)),
        ("Find 1 + 2 + ... + 25.", triangular_trace(25)),
        even_square_proof("n"),
        odd_sum_proof("n"),
    ]
    for prompt, completion in anchors:
        rows.extend([emit(prompt, completion)] * 20)

    seen: set[str] = set()
    while len(rows) < count:
        kind = rng.randrange(11)
        if kind == 0:
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            prompt = f"{rng.choice(prompt_variants)} {a} + {b}."
            completion = add_trace(a, b)
        elif kind == 1:
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            prompt = f"{rng.choice(prompt_variants)} {a} - {b}."
            completion = sub_trace(a, b)
        elif kind == 2:
            a, b = rng.randint(2, 999), rng.randint(2, 99)
            prompt = f"{rng.choice(prompt_variants)} {a} * {b}."
            completion = mul_trace(a, b)
        elif kind == 3:
            a = rng.choice([x for x in range(2, 18)])
            b = rng.randint(1, 80)
            x = rng.randint(-40, 60)
            sign = rng.choice([1, -1])
            prompt, completion = linear_trace(a, b, x, sign)
        elif kind == 4:
            n = rng.randint(2, 300)
            prompt = rng.choice([
                f"Find the sum of the first {n} odd positive integers.",
                f"What is 1 + 3 + 5 + ... through the first {n} odd positive integers?",
            ])
            completion = odd_sum_trace(n)
        elif kind == 5:
            n = rng.randint(2, 400)
            prompt = rng.choice([f"Find 1 + 2 + ... + {n}.", f"Find the sum of the first {n} positive integers."])
            completion = triangular_trace(n)
        elif kind == 6:
            a, b = rng.randint(1, 25), rng.randint(2, 25)
            c, d = rng.randint(1, 25), rng.randint(2, 25)
            prompt, completion = fraction_add_trace(a, b, c, d)
        elif kind == 7:
            prompt, completion = even_square_proof(rng.choice(vars_))
        elif kind == 8:
            prompt, completion = odd_square_proof(rng.choice(vars_))
        elif kind == 9:
            prompt, completion = odd_sum_proof(rng.choice(vars_))
        else:
            prompt, completion = triangular_proof(rng.choice(vars_))

        key = prompt + "\n" + completion
        if key in seen and rng.random() < 0.95:
            continue
        seen.add(key)
        rows.append(emit(prompt, completion))

    rng.shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build varied SFT examples that teach step-by-step math reasoning, not benchmark routing.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
