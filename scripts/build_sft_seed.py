#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from pathlib import Path


def write_row(handle, prompt: str, completion: str) -> None:
    handle.write(json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False) + "\n")


def build_examples(count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []

    proof_templates = [
        (
            "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.",
            "We prove the claim by induction on n. For n = 1, the left side is 1 and the right side is 1^2 = 1, so the claim holds. Assume that 1 + 3 + ... + (2n - 1) = n^2. The next odd number is 2(n + 1) - 1 = 2n + 1. Therefore 1 + 3 + ... + (2n - 1) + (2n + 1) = n^2 + 2n + 1 = (n + 1)^2. Hence the formula holds for n + 1, and by induction it holds for all positive integers n."
        ),
        (
            "Prove that the sum of the first n positive integers is n(n + 1)/2.",
            "Let S = 1 + 2 + ... + n. Write the same sum in reverse: S = n + (n - 1) + ... + 1. Adding the two equations term by term gives 2S = (n + 1) + (n + 1) + ... + (n + 1), with n copies. Thus 2S = n(n + 1), so S = n(n + 1)/2."
        ),
        (
            "Prove that if n is even, then n^2 is even.",
            "Since n is even, there is an integer k such that n = 2k. Then n^2 = (2k)^2 = 4k^2 = 2(2k^2). Since 2k^2 is an integer, n^2 is divisible by 2. Therefore n^2 is even."
        ),
        (
            "Prove that if n is odd, then n^2 is odd.",
            "Since n is odd, there is an integer k such that n = 2k + 1. Then n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1. This has the form 2m + 1 for an integer m, so n^2 is odd."
        ),
    ]
    examples.extend(proof_templates * max(1, count // 80))

    for _ in range(count):
        kind = rng.randrange(8)
        if kind == 0:
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            prompt = f"Compute {a} + {b}."
            completion = f"Add the numbers directly: {a} + {b} = {a + b}. Therefore the answer is {a + b}."
        elif kind == 1:
            a, b = rng.randint(10, 999), rng.randint(2, 99)
            prompt = f"Compute {a} - {b}."
            completion = f"Subtract {b} from {a}: {a} - {b} = {a - b}. Therefore the answer is {a - b}."
        elif kind == 2:
            a, b = rng.randint(2, 30), rng.randint(2, 30)
            prompt = f"Compute {a} * {b}."
            completion = f"Multiplying gives {a} * {b} = {a * b}. Therefore the answer is {a * b}."
        elif kind == 3:
            x = rng.randint(-20, 20)
            a = rng.choice([i for i in range(-9, 10) if i not in (0, 1)])
            b = rng.randint(-20, 20)
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}."
            completion = f"Subtract {b} from both sides to get {a}x = {c - b}. Divide by {a}: x = ({c - b})/{a} = {x}. Therefore x = {x}."
        elif kind == 4:
            r1, r2 = rng.randint(-10, 10), rng.randint(-10, 10)
            b = -(r1 + r2)
            c = r1 * r2
            prompt = f"Find the roots of x^2 + ({b})x + ({c}) = 0."
            completion = f"We factor the quadratic using numbers whose sum is {-b} and product is {c}. These numbers are {r1} and {r2}. Thus x^2 + ({b})x + ({c}) = (x - ({r1}))(x - ({r2})). Therefore the roots are x = {r1} and x = {r2}."
        elif kind == 5:
            a, b, c, d = rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 12)
            f1, f2 = Fraction(a, b), Fraction(c, d)
            ans = f1 + f2
            prompt = f"Compute {a}/{b} + {c}/{d}."
            completion = f"Use a common denominator {b*d}. We have {a}/{b} = {a*d}/{b*d} and {c}/{d} = {c*b}/{b*d}. Therefore the sum is ({a*d} + {c*b})/{b*d} = {ans.numerator}/{ans.denominator}."
        elif kind == 6:
            n = rng.randint(2, 50)
            prompt = f"Find the sum of the first {n} odd positive integers."
            completion = f"The sum of the first n odd positive integers is n^2. Here n = {n}, so the sum is {n}^2 = {n*n}."
        else:
            n = rng.randint(2, 50)
            prompt = f"Find 1 + 2 + ... + {n}."
            completion = f"Use the formula 1 + 2 + ... + n = n(n + 1)/2. With n = {n}, the sum is {n}({n} + 1)/2 = {n*(n+1)//2}."
        examples.append((prompt, completion))
    rng.shuffle(examples)
    return examples[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a seed worked-solution SFT JSONL for Archimedes-Math.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples = build_examples(args.count, args.seed)
    with args.out.open("w", encoding="utf-8") as handle:
        for prompt, completion in examples:
            write_row(handle, prompt, completion)
    print(json.dumps({"out": str(args.out), "examples": len(examples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
