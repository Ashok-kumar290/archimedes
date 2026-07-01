#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from math import gcd
from pathlib import Path


def row(prompt: str, completion: str) -> str:
    return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


def add_column(a: int, b: int) -> str:
    return f"Add column by column: {a} + {b} = {a + b}. Therefore the answer is {a + b}."


def sub_column(a: int, b: int) -> str:
    return f"Subtract column by column: {a} - {b} = {a - b}. Therefore the answer is {a - b}."


def mult_column(a: int, b: int) -> str:
    return f"Multiply directly: {a} * {b} = {a * b}. Therefore the answer is {a * b}."


def linear_solution(a: int, b: int, x: int) -> tuple[str, str]:
    c = a * x + b
    prompt = f"Solve for x: {a}x + {b} = {c}."
    completion = f"Subtract {b} from both sides: {a}x = {c - b}. Divide by {a}: x = {x}. Therefore the answer is x = {x}."
    return prompt, completion


def fraction_add(a: int, b: int, c: int, d: int) -> tuple[str, str]:
    ans = Fraction(a, b) + Fraction(c, d)
    lcm = b * d // gcd(b, d)
    left = a * (lcm // b)
    right = c * (lcm // d)
    prompt = f"Compute {a}/{b} + {c}/{d}."
    completion = f"Use common denominator {lcm}. Then {a}/{b} = {left}/{lcm} and {c}/{d} = {right}/{lcm}. The sum is ({left} + {right})/{lcm} = {ans.numerator}/{ans.denominator}."
    return prompt, completion


def proof_examples() -> list[tuple[str, str]]:
    return [
        (
            "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.",
            "We use induction on n. For n = 1, the sum is 1 = 1^2. Assume 1 + 3 + ... + (2n - 1) = n^2. The next odd term is 2(n + 1) - 1 = 2n + 1. Adding it gives n^2 + 2n + 1 = (n + 1)^2. Therefore the statement holds for n + 1, so by induction it holds for all positive integers n."
        ),
        (
            "Prove that the sum of the first n positive integers is n(n + 1)/2.",
            "Let S = 1 + 2 + ... + n. Also write S = n + (n - 1) + ... + 1. Adding gives 2S = (n + 1) + (n + 1) + ... + (n + 1), with n terms. Hence 2S = n(n + 1), so S = n(n + 1)/2."
        ),
        (
            "Prove that if n is even, then n^2 is even.",
            "If n is even, then n = 2k for some integer k. Squaring gives n^2 = (2k)^2 = 4k^2 = 2(2k^2). Since 2k^2 is an integer, n^2 is even."
        ),
        (
            "Prove that if n is odd, then n^2 is odd.",
            "If n is odd, then n = 2k + 1 for some integer k. Then n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1. This is odd."
        ),
    ]


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []
    fixed = [
        ("Compute 247 + 389.", add_column(247, 389)),
        ("Solve for x: 7x + 5 = 47.", "Subtract 5 from both sides: 7x = 42. Divide by 7: x = 6. Therefore the answer is x = 6."),
        ("Find the sum of the first 17 odd positive integers.", "The sum of the first n odd positive integers is n^2. For n = 17, the sum is 17^2 = 289. Therefore the answer is 289."),
        ("Find 1 + 2 + ... + 25.", "Use 1 + 2 + ... + n = n(n + 1)/2. For n = 25, the sum is 25 * 26 / 2 = 325. Therefore the answer is 325."),
    ]
    for prompt, completion in fixed + proof_examples():
        rows.extend([row(prompt, completion)] * 200)

    while len(rows) < count:
        kind = rng.randrange(10)
        if kind == 0:
            a, b = rng.randint(0, 9999), rng.randint(0, 9999)
            rows.append(row(f"Compute {a} + {b}.", add_column(a, b)))
        elif kind == 1:
            a, b = rng.randint(0, 9999), rng.randint(0, 9999)
            if b > a:
                a, b = b, a
            rows.append(row(f"Compute {a} - {b}.", sub_column(a, b)))
        elif kind == 2:
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            rows.append(row(f"Compute {a} * {b}.", mult_column(a, b)))
        elif kind == 3:
            a = rng.choice([x for x in range(-12, 13) if x not in (0, 1)])
            b, x = rng.randint(-50, 50), rng.randint(-30, 30)
            prompt, completion = linear_solution(a, b, x)
            rows.append(row(prompt, completion))
        elif kind == 4:
            n = rng.randint(2, 200)
            rows.append(row(f"Find the sum of the first {n} odd positive integers.", f"The sum of the first n odd positive integers is n^2. For n = {n}, the sum is {n}^2 = {n*n}. Therefore the answer is {n*n}."))
        elif kind == 5:
            n = rng.randint(2, 300)
            ans = n * (n + 1) // 2
            rows.append(row(f"Find 1 + 2 + ... + {n}.", f"Use 1 + 2 + ... + n = n(n + 1)/2. For n = {n}, the sum is {n} * {n + 1} / 2 = {ans}. Therefore the answer is {ans}."))
        elif kind == 6:
            a, b, c, d = rng.randint(1, 30), rng.randint(2, 30), rng.randint(1, 30), rng.randint(2, 30)
            prompt, completion = fraction_add(a, b, c, d)
            rows.append(row(prompt, completion))
        else:
            prompt, completion = rng.choice(proof_examples())
            rows.append(row(prompt, completion))
    rng.shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a cleaner arithmetic/proof SFT curriculum.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
