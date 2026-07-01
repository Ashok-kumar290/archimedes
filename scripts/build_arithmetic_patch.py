#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def emit(prompt: str, completion: str) -> str:
    return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


def addition(a: int, b: int) -> tuple[str, str]:
    s = a + b
    return (
        f"Compute {a} + {b}.",
        f"Add the two integers: {a} + {b} = {s}. Therefore the answer is {s}.",
    )


def subtraction(a: int, b: int) -> tuple[str, str]:
    if b > a:
        a, b = b, a
    d = a - b
    return (
        f"Compute {a} - {b}.",
        f"Subtract the second integer from the first: {a} - {b} = {d}. Therefore the answer is {d}.",
    )


def multiplication(a: int, b: int) -> tuple[str, str]:
    p = a * b
    return (
        f"Compute {a} * {b}.",
        f"Multiply the two integers: {a} * {b} = {p}. Therefore the answer is {p}.",
    )


def linear(a: int, b: int, x: int) -> tuple[str, str]:
    c = a * x + b
    return (
        f"Solve for x: {a}x + {b} = {c}.",
        f"Subtract {b} from both sides: {a}x = {c - b}. Divide by {a}: x = {x}. Therefore the answer is x = {x}.",
    )


def odd_sum(n: int) -> tuple[str, str]:
    ans = n * n
    return (
        f"Find the sum of the first {n} odd positive integers.",
        f"The sum of the first n odd positive integers is n^2. Here n = {n}, so the sum is {n}^2 = {ans}. Therefore the answer is {ans}.",
    )


def triangular(n: int) -> tuple[str, str]:
    ans = n * (n + 1) // 2
    return (
        f"Find 1 + 2 + ... + {n}.",
        f"Use the formula 1 + 2 + ... + n = n(n + 1)/2. For n = {n}, this is {n} * {n + 1} / 2 = {ans}. Therefore the answer is {ans}.",
    )


def build(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []

    anchors = [
        addition(247, 389),
        linear(7, 5, 6),
        odd_sum(17),
        triangular(25),
        addition(379, 257),
        linear(9, -4, 8),
        odd_sum(23),
        triangular(40),
    ]
    for prompt, completion in anchors:
        rows.extend([emit(prompt, completion)] * 500)

    while len(rows) < count:
        kind = rng.randrange(5)
        if kind == 0:
            a = rng.randint(0, 9999)
            b = rng.randint(0, 9999)
            prompt, completion = addition(a, b)
        elif kind == 1:
            a = rng.randint(0, 9999)
            b = rng.randint(0, 9999)
            prompt, completion = subtraction(a, b)
        elif kind == 2:
            a = rng.randint(2, 99)
            b = rng.randint(2, 99)
            prompt, completion = multiplication(a, b)
        elif kind == 3:
            a = rng.choice([v for v in range(-12, 13) if v not in (0, 1)])
            b = rng.randint(-60, 60)
            x = rng.randint(-40, 40)
            prompt, completion = linear(a, b, x)
        else:
            n = rng.randint(2, 300)
            prompt, completion = odd_sum(n) if rng.random() < 0.5 else triangular(n)
        rows.append(emit(prompt, completion))

    rng.shuffle(rows)
    return rows[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build focused arithmetic patch SFT data.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
