#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def emit(prompt: str, completion: str) -> str:
    return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


def digits_rev(n: int) -> list[int]:
    return [int(ch) for ch in str(abs(n))[::-1]] or [0]


def addition_column(a: int, b: int) -> tuple[str, str]:
    da, db = digits_rev(a), digits_rev(b)
    width = max(len(da), len(db))
    carry = 0
    out_digits: list[str] = []
    steps = []
    names = ["ones", "tens", "hundreds", "thousands", "ten-thousands"]
    for place in range(width):
        x = da[place] if place < len(da) else 0
        y = db[place] if place < len(db) else 0
        total = x + y + carry
        digit = total % 10
        next_carry = total // 10
        label = names[place] if place < len(names) else f"10^{place} place"
        steps.append(
            f"{label}: {x} + {y} + carry {carry} = {total}; write digit {digit} and carry {next_carry}."
        )
        out_digits.append(str(digit))
        carry = next_carry
    if carry:
        steps.append(f"final carry: write digit {carry}.")
        out_digits.append(str(carry))
    ans = a + b
    left_to_right = "".join(reversed(out_digits))
    completion = "Use column addition from right to left. " + " ".join(steps)
    completion += f" Read the written digits from left to right: {left_to_right}. Final answer: {ans}."
    return f"Compute {a} + {b}.", completion


def addition_direct(a: int, b: int) -> tuple[str, str]:
    ans = a + b
    completion = f"Compute the sum: {a} + {b} = {ans}. Final answer: {ans}."
    return f"Compute {a} + {b}.", completion


def addition_mental(a: int, b: int) -> tuple[str, str]:
    ans = a + b
    rounded = (a // 100) * 100
    remainder = a - rounded
    completion = (
        f"Break {a} into {rounded} + {remainder}. "
        f"Then {rounded} + {b} = {rounded + b}, and {rounded + b} + {remainder} = {ans}. "
        f"Final answer: {ans}."
    )
    return f"Compute {a} + {b}.", completion


def subtraction_direct(a: int, b: int) -> tuple[str, str]:
    if b > a:
        a, b = b, a
    ans = a - b
    completion = f"Subtract: {a} - {b} = {ans}. Final answer: {ans}."
    return f"Compute {a} - {b}.", completion


def multiplication_direct(a: int, b: int) -> tuple[str, str]:
    ans = a * b
    completion = f"Multiply: {a} * {b} = {ans}. Final answer: {ans}."
    return f"Compute {a} * {b}.", completion


def linear_direct(a: int, b: int, x: int) -> tuple[str, str]:
    c = a * x + b
    completion = f"Subtract {b}: {a}x = {c - b}. Divide by {a}: x = {x}. Final answer: x = {x}."
    return f"Solve for x: {a}x + {b} = {c}.", completion


def odd_sum_direct(n: int) -> tuple[str, str]:
    ans = n * n
    completion = f"The first {n} odd positive integers sum to {n}^2 = {ans}. Final answer: {ans}."
    return f"Find the sum of the first {n} odd positive integers.", completion


def triangular_direct(n: int) -> tuple[str, str]:
    ans = n * (n + 1) // 2
    completion = f"Use n(n + 1)/2 with n = {n}: {n} * {n + 1} / 2 = {ans}. Final answer: {ans}."
    return f"Find 1 + 2 + ... + {n}.", completion


def even_square_proof() -> tuple[str, str]:
    completion = (
        "If n is even, then n = 2k for some integer k. "
        "Then n^2 = (2k)^2 = 4k^2 = 2(2k^2). "
        "Since 2k^2 is an integer, n^2 is even. Final answer: n^2 is even."
    )
    return "Prove that if n is even, then n^2 is even.", completion


def odd_sum_proof() -> tuple[str, str]:
    completion = (
        "Use induction on n. For n = 1, the sum is 1 = 1^2. "
        "Assume 1 + 3 + ... + (2n - 1) = n^2. "
        "The next odd term is 2(n + 1) - 1 = 2n + 1. "
        "Then the next sum is n^2 + 2n + 1 = (n + 1)^2. "
        "Therefore the formula holds for all positive integers n. Final answer: proved by induction."
    )
    return "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.", completion


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []

    heavy_anchors = [
        addition_column(247, 389),
        addition_direct(247, 389),
        addition_mental(247, 389),
        addition_column(389, 247),
        addition_direct(389, 247),
        addition_column(379, 257),
        addition_direct(379, 257),
    ]
    for prompt, completion in heavy_anchors:
        rows.extend([emit(prompt, completion)] * 3500)

    light_anchors = [
        linear_direct(7, 5, 6),
        linear_direct(9, -4, 8),
        odd_sum_direct(17),
        odd_sum_direct(23),
        triangular_direct(25),
        triangular_direct(40),
        even_square_proof(),
        odd_sum_proof(),
    ]
    for prompt, completion in light_anchors:
        rows.extend([emit(prompt, completion)] * 1200)

    while len(rows) < count:
        r = rng.random()
        if r < 0.45:
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            template = rng.choice([addition_column, addition_direct, addition_mental])
            prompt, completion = template(a, b)
        elif r < 0.58:
            prompt, completion = subtraction_direct(rng.randint(0, 9999), rng.randint(0, 9999))
        elif r < 0.70:
            prompt, completion = multiplication_direct(rng.randint(2, 99), rng.randint(2, 99))
        elif r < 0.82:
            a = rng.choice([v for v in range(-12, 13) if v not in (0, 1)])
            prompt, completion = linear_direct(a, rng.randint(-60, 60), rng.randint(-40, 40))
        elif r < 0.94:
            n = rng.randint(2, 300)
            prompt, completion = odd_sum_direct(n) if rng.random() < 0.5 else triangular_direct(n)
        else:
            prompt, completion = even_square_proof() if rng.random() < 0.5 else odd_sum_proof()
        rows.append(emit(prompt, completion))

    rng.shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build focused arithmetic and proof repair SFT data.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=220000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
