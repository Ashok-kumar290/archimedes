#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def emit(prompt: str, completion: str) -> str:
    return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


def even_square() -> tuple[str, str]:
    return (
        "Prove that if n is even, then n^2 is even.",
        "If n is even, then n = 2k for some integer k. Squaring gives n^2 = (2k)^2 = 4k^2 = 2(2k^2). Since 2k^2 is an integer, n^2 is even. Final answer: proved.",
    )


def odd_square() -> tuple[str, str]:
    return (
        "Prove that if n is odd, then n^2 is odd.",
        "If n is odd, then n = 2k + 1 for some integer k. Squaring gives n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1. Since 2k^2 + 2k is an integer, n^2 is odd. Final answer: proved.",
    )


def odd_sum_induction() -> tuple[str, str]:
    return (
        "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.",
        "We prove the statement by induction on n. For n = 1, the left side is 1 and the right side is 1^2 = 1. Now assume 1 + 3 + 5 + ... + (2n - 1) = n^2. The next odd term is 2(n + 1) - 1 = 2n + 1. Adding it to the assumed sum gives n^2 + 2n + 1 = (n + 1)^2. Therefore the formula holds for n + 1. By induction, the formula holds for all positive integers n. Final answer: proved by induction.",
    )


def triangular_induction() -> tuple[str, str]:
    return (
        "Prove that 1 + 2 + ... + n = n(n + 1)/2.",
        "We prove the formula by induction on n. For n = 1, the sum is 1 and 1(1 + 1)/2 = 1. Assume 1 + 2 + ... + n = n(n + 1)/2. Then 1 + 2 + ... + n + (n + 1) = n(n + 1)/2 + (n + 1). Factor n + 1 to get (n + 1)(n/2 + 1) = (n + 1)(n + 2)/2. Therefore the formula holds for n + 1, so it holds for all positive integers n. Final answer: proved by induction.",
    )


def triangular_pairing() -> tuple[str, str]:
    return (
        "Prove that the sum of the first n positive integers is n(n + 1)/2.",
        "Let S = 1 + 2 + ... + n. Write the same sum in reverse: S = n + (n - 1) + ... + 1. Adding the two equations gives 2S = (n + 1) + (n + 1) + ... + (n + 1), with n equal terms. Thus 2S = n(n + 1), so S = n(n + 1)/2. Final answer: proved.",
    )


def odd_plus_odd_even() -> tuple[str, str]:
    return (
        "Prove that the sum of two odd integers is even.",
        "Let the two odd integers be 2a + 1 and 2b + 1 for integers a and b. Their sum is (2a + 1) + (2b + 1) = 2a + 2b + 2 = 2(a + b + 1). Since a + b + 1 is an integer, the sum is even. Final answer: proved.",
    )


def even_times_integer_even() -> tuple[str, str]:
    return (
        "Prove that the product of an even integer and any integer is even.",
        "Let the even integer be 2k and let the other integer be m. Their product is (2k)m = 2(km). Since km is an integer, the product is even. Final answer: proved.",
    )


def divisible_by_three_sum() -> tuple[str, str]:
    return (
        "Prove that if a and b are divisible by 3, then a + b is divisible by 3.",
        "If a is divisible by 3, then a = 3m for some integer m. If b is divisible by 3, then b = 3n for some integer n. Then a + b = 3m + 3n = 3(m + n). Since m + n is an integer, a + b is divisible by 3. Final answer: proved.",
    )


def square_nonnegative() -> tuple[str, str]:
    return (
        "Prove that x^2 is nonnegative for every real number x.",
        "If x is nonnegative, then x^2 = x times x is nonnegative. If x is negative, then -x is positive and x^2 = (-x)^2, which is positive. In both cases x^2 >= 0. Final answer: proved.",
    )


def sqrt_two_irrational() -> tuple[str, str]:
    return (
        "Prove that sqrt(2) is irrational.",
        "Assume for contradiction that sqrt(2) = a/b where a and b are integers with no common factor and b is not zero. Squaring gives 2 = a^2/b^2, so a^2 = 2b^2. Thus a^2 is even, so a is even. Write a = 2k. Then 4k^2 = 2b^2, so b^2 = 2k^2, which means b is even. Now both a and b are even, contradicting that a/b was in lowest terms. Therefore sqrt(2) is irrational. Final answer: proved by contradiction.",
    )


PROOF_BUILDERS = [
    even_square,
    odd_square,
    odd_sum_induction,
    triangular_induction,
    triangular_pairing,
    odd_plus_odd_even,
    even_times_integer_even,
    divisible_by_three_sum,
    square_nonnegative,
    sqrt_two_irrational,
]


ANCHOR_WEIGHTS = {
    "Prove that if n is even, then n^2 is even.": 4500,
    "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.": 5500,
    "Prove that 1 + 2 + ... + n = n(n + 1)/2.": 3000,
    "Prove that the sum of the first n positive integers is n(n + 1)/2.": 3000,
}


def parameterized_even_square(rng: random.Random) -> tuple[str, str]:
    symbol = rng.choice(["m", "a", "r"])
    return (
        f"Prove that if {symbol} is even, then {symbol}^2 is even.",
        f"If {symbol} is even, then {symbol} = 2k for some integer k. Squaring gives {symbol}^2 = (2k)^2 = 4k^2 = 2(2k^2). Since 2k^2 is an integer, {symbol}^2 is even. Final answer: proved.",
    )


def parameterized_odd_sum(rng: random.Random) -> tuple[str, str]:
    variable = rng.choice(["m", "r", "N"])
    return (
        f"Prove that 1 + 3 + 5 + ... + (2{variable} - 1) = {variable}^2.",
        f"Use induction on {variable}. For {variable} = 1, the sum is 1 = 1^2. Assume the formula holds for {variable}. The next odd term is 2({variable} + 1) - 1 = 2{variable} + 1. The next sum is {variable}^2 + 2{variable} + 1 = ({variable} + 1)^2. Therefore the formula holds for {variable} + 1, and by induction it holds for every positive integer {variable}. Final answer: proved by induction.",
    )


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []

    for builder in PROOF_BUILDERS:
        prompt, completion = builder()
        rows.extend([emit(prompt, completion)] * ANCHOR_WEIGHTS.get(prompt, 1200))

    while len(rows) < count:
        r = rng.random()
        if r < 0.20:
            prompt, completion = parameterized_even_square(rng)
        elif r < 0.40:
            prompt, completion = parameterized_odd_sum(rng)
        else:
            prompt, completion = rng.choice(PROOF_BUILDERS)()
        rows.append(emit(prompt, completion))

    rng.shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build proof-focused SFT data for Archimedes.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
