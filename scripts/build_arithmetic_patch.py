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


def addition_scratch(a: int, b: int) -> tuple[str, str]:
    da, db = digits_rev(a), digits_rev(b)
    width = max(len(da), len(db))
    carry = 0
    steps = []
    result_digits = []
    for place in range(width):
        x = da[place] if place < len(da) else 0
        y = db[place] if place < len(db) else 0
        total = x + y + carry
        digit = total % 10
        next_carry = total // 10
        steps.append(
            f"place {place}: {x} + {y} + carry {carry} = {total}, write {digit}, carry {next_carry}."
        )
        result_digits.append(str(digit))
        carry = next_carry
    if carry:
        steps.append(f"final carry: write {carry}.")
        result_digits.append(str(carry))
    ans = a + b
    completion = "Use digit-by-digit addition. " + " ".join(steps)
    completion += f" Reading the digits gives {ans}. Final answer: {ans}."
    return f"Compute {a} + {b}.", completion


def addition_direct(a: int, b: int) -> tuple[str, str]:
    ans = a + b
    completion = f"Add the two integers: {a} + {b} = {ans}. Final answer: {ans}."
    return f"Compute {a} + {b}.", completion


def subtraction_scratch(a: int, b: int) -> tuple[str, str]:
    if b > a:
        a, b = b, a
    da, db = digits_rev(a), digits_rev(b)
    width = max(len(da), len(db))
    borrow = 0
    steps = []
    result_digits = []
    for place in range(width):
        x = da[place] if place < len(da) else 0
        y = db[place] if place < len(db) else 0
        current = x - borrow
        if current < y:
            current += 10
            next_borrow = 1
        else:
            next_borrow = 0
        digit = current - y
        steps.append(
            f"place {place}: {current} - {y} = {digit}, next borrow {next_borrow}."
        )
        result_digits.append(str(digit))
        borrow = next_borrow
    ans = a - b
    completion = "Use digit-by-digit subtraction. " + " ".join(steps)
    completion += f" Reading the digits gives {ans}. Final answer: {ans}."
    return f"Compute {a} - {b}.", completion


def subtraction_direct(a: int, b: int) -> tuple[str, str]:
    if b > a:
        a, b = b, a
    ans = a - b
    completion = f"Subtract the second integer from the first: {a} - {b} = {ans}. Final answer: {ans}."
    return f"Compute {a} - {b}.", completion


def multiplication_scratch(a: int, b: int) -> tuple[str, str]:
    ans = a * b
    completion = f"Compute the product directly: {a} * {b} = {ans}. Final answer: {ans}."
    return f"Compute {a} * {b}.", completion


def linear_scratch(a: int, b: int, x: int) -> tuple[str, str]:
    c = a * x + b
    rhs = c - b
    completion = (
        f"Start with {a}x + {b} = {c}. "
        f"Subtract {b} from both sides: {a}x = {rhs}. "
        f"Divide both sides by {a}: x = {x}. Final answer: x = {x}."
    )
    return f"Solve for x: {a}x + {b} = {c}.", completion


def linear_direct(a: int, b: int, x: int) -> tuple[str, str]:
    c = a * x + b
    completion = f"Subtract {b}: {a}x = {c - b}. Divide by {a}: x = {x}. Final answer: x = {x}."
    return f"Solve for x: {a}x + {b} = {c}.", completion


def odd_sum_scratch(n: int) -> tuple[str, str]:
    ans = n * n
    completion = (
        f"The sum of the first n odd positive integers is n^2. "
        f"Here n = {n}. Compute {n}^2 = {ans}. Final answer: {ans}."
    )
    return f"Find the sum of the first {n} odd positive integers.", completion


def odd_sum_direct(n: int) -> tuple[str, str]:
    ans = n * n
    completion = f"The first {n} odd positive integers sum to {n}^2 = {ans}. Final answer: {ans}."
    return f"Find the sum of the first {n} odd positive integers.", completion


def triangular_scratch(n: int) -> tuple[str, str]:
    ans = n * (n + 1) // 2
    completion = (
        f"Use the formula 1 + 2 + ... + n = n(n + 1)/2. "
        f"Here n = {n}, so the sum is {n} * {n + 1} / 2 = {ans}. Final answer: {ans}."
    )
    return f"Find 1 + 2 + ... + {n}.", completion


def triangular_direct(n: int) -> tuple[str, str]:
    ans = n * (n + 1) // 2
    completion = f"Use n(n + 1)/2 with n = {n}: {n} * {n + 1} / 2 = {ans}. Final answer: {ans}."
    return f"Find 1 + 2 + ... + {n}.", completion


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []
    anchors = [
        addition_scratch(247, 389),
        addition_direct(247, 389),
        addition_scratch(389, 247),
        addition_direct(389, 247),
        addition_scratch(379, 257),
        addition_direct(379, 257),
        linear_scratch(7, 5, 6),
        linear_direct(7, 5, 6),
        linear_scratch(9, -4, 8),
        linear_direct(9, -4, 8),
        odd_sum_scratch(17),
        odd_sum_direct(17),
        odd_sum_scratch(23),
        odd_sum_direct(23),
        triangular_scratch(25),
        triangular_direct(25),
        triangular_scratch(40),
        triangular_direct(40),
    ]
    for prompt, completion in anchors:
        rows.extend([emit(prompt, completion)] * 1000)

    while len(rows) < count:
        kind = rng.randrange(5)
        if kind == 0:
            a, b = rng.randint(0, 9999), rng.randint(0, 9999)
            prompt, completion = addition_scratch(a, b) if rng.random() < 0.5 else addition_direct(a, b)
        elif kind == 1:
            a, b = rng.randint(0, 9999), rng.randint(0, 9999)
            prompt, completion = subtraction_scratch(a, b) if rng.random() < 0.5 else subtraction_direct(a, b)
        elif kind == 2:
            prompt, completion = multiplication_scratch(rng.randint(2, 99), rng.randint(2, 99))
        elif kind == 3:
            a = rng.choice([v for v in range(-12, 13) if v not in (0, 1)])
            b, x = rng.randint(-60, 60), rng.randint(-40, 40)
            prompt, completion = linear_scratch(a, b, x) if rng.random() < 0.5 else linear_direct(a, b, x)
        else:
            n = rng.randint(2, 300)
            if rng.random() < 0.5:
                prompt, completion = odd_sum_scratch(n) if rng.random() < 0.5 else odd_sum_direct(n)
            else:
                prompt, completion = triangular_scratch(n) if rng.random() < 0.5 else triangular_direct(n)
        rows.append(emit(prompt, completion))
    rng.shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build focused scratchpad arithmetic patch SFT data.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=160000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
