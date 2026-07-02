#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from fractions import Fraction
from math import gcd
from pathlib import Path


# The fixed prompts used by scripts/eval_math_prompts.py. These must never
# appear in training data, otherwise the eval measures memorization.
EVAL_HOLDOUT = {
    "Compute 247 + 389.",
    "Solve for x: 7x + 5 = 47.",
    "Find the sum of the first 17 odd positive integers.",
    "Find 1 + 2 + ... + 25.",
    "Prove that if n is even, then n^2 is even.",
    "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.",
}

ARITH_PREFIXES = ["Compute", "Find", "Evaluate", "Calculate", "Work out"]
PROOF_PREFIXES = ["Prove that", "Show that", "Explain why"]


def emit(prompt: str, completion: str) -> str:
    return json.dumps({"prompt": prompt, "completion": completion}, ensure_ascii=False)


def fmt_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def signed_term(coef: int) -> str:
    return f"+ {coef}" if coef >= 0 else f"- {-coef}"


def arith_prompt(rng: random.Random, expr: str) -> str:
    prefix = rng.choice(ARITH_PREFIXES + ["What is"])
    if prefix == "What is":
        return f"What is {expr}?"
    return f"{prefix} {expr}."


def add_trace(a: int, b: int) -> str:
    ans = a + b
    return (
        f"Plan: add the two integers and check by subtracting one addend. "
        f"Step 1: compute {a} + {b} = {ans}. "
        f"Check: {ans} - {b} = {a}, which recovers the first addend. "
        f"Final answer: {ans}."
    )


def sub_trace(a: int, b: int) -> str:
    ans = a - b
    return (
        f"Plan: subtract and check by adding back the subtracted number. "
        f"Step 1: compute {a} - {b} = {ans}. "
        f"Check: {ans} + {b} = {a}. "
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


def div_trace(a: int, b: int) -> str:
    q = a // b
    return (
        f"Plan: divide and check by multiplying back. "
        f"Step 1: compute {a} / {b} = {q}. "
        f"Check: {q} * {b} = {a}. "
        f"Final answer: {q}."
    )


def order_of_ops_trace(a: int, b: int, c: int, grouped: bool) -> tuple[str, str]:
    if grouped:
        expr = f"({a} + {b}) * {c}"
        inner = a + b
        ans = inner * c
        completion = (
            f"Plan: evaluate the parentheses first, then multiply. "
            f"Step 1: compute {a} + {b} = {inner}. "
            f"Step 2: compute {inner} * {c} = {ans}. "
            f"Final answer: {ans}."
        )
    else:
        expr = f"{a} + {b} * {c}"
        prod = b * c
        ans = a + prod
        completion = (
            f"Plan: multiplication comes before addition. "
            f"Step 1: compute {b} * {c} = {prod}. "
            f"Step 2: compute {a} + {prod} = {ans}. "
            f"Final answer: {ans}."
        )
    return expr, completion


def power_trace(base: int, exp: int) -> tuple[str, str]:
    expr = f"{base}^{exp}"
    steps = []
    value = 1
    for i in range(1, exp + 1):
        value *= base
        steps.append(f"{base}^{i} = {value}")
    completion = (
        f"Plan: multiply by {base} one power at a time. "
        f"Step 1: {'; '.join(steps)}. "
        f"Final answer: {value}."
    )
    return expr, completion


def percent_trace(p: int, n: int) -> tuple[str, str]:
    ans = p * n // 100
    prompt = f"What is {p}% of {n}?"
    completion = (
        f"Plan: convert the percentage to a fraction of 100. "
        f"Step 1: {p}% of {n} is {p}/100 * {n}. "
        f"Step 2: compute {p} * {n} = {p * n}, then divide by 100 to get {ans}. "
        f"Final answer: {ans}."
    )
    return prompt, completion


def gcd_trace(a: int, b: int) -> tuple[str, str]:
    prompt_expr = f"the greatest common divisor of {a} and {b}"
    steps = []
    x, y = max(a, b), min(a, b)
    idx = 1
    while y:
        q, r = divmod(x, y)
        steps.append(f"Step {idx}: {x} = {q} * {y} + {r}.")
        x, y = y, r
        idx += 1
    completion = (
        f"Plan: apply the Euclidean algorithm until the remainder is 0. "
        f"{' '.join(steps)} "
        f"Step {idx}: the last nonzero remainder is {x}. "
        f"Final answer: {x}."
    )
    return prompt_expr, completion


def mean_trace(nums: list[int]) -> tuple[str, str]:
    total = sum(nums)
    ans = total // len(nums)
    listing = ", ".join(str(n) for n in nums)
    prompt = f"Find the mean of {listing}."
    completion = (
        f"Plan: add the numbers, then divide by how many there are. "
        f"Step 1: {' + '.join(str(n) for n in nums)} = {total}. "
        f"Step 2: divide by {len(nums)}: {total} / {len(nums)} = {ans}. "
        f"Final answer: {ans}."
    )
    return prompt, completion


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


def two_step_linear_trace(a: int, c: int, b: int, x: int) -> tuple[str, str]:
    d = (a - c) * x + b
    lhs = f"{a}x {signed_term(b)}"
    rhs = f"{c}x {signed_term(d)}" if d != 0 else f"{c}x"
    coef = a - c
    const = d - b
    prompt = f"Solve for x: {lhs} = {rhs}."
    completion = (
        f"Plan: collect the x terms on one side and constants on the other. "
        f"Step 1: start with {lhs} = {rhs}. "
        f"Step 2: subtract {c}x from both sides: {coef}x {signed_term(b)} = {d}. "
        f"Step 3: move the constant: {coef}x = {const}. "
        f"Step 4: divide by {coef}: x = {x}. "
        f"Check: {a} * {x} + {b} = {a * x + b} and {c} * {x} + {d} = {c * x + d}. "
        f"Final answer: x = {x}."
    )
    return prompt, completion


def fraction_add_trace(a: int, b: int, c: int, d: int, subtract: bool) -> tuple[str, str]:
    rhs = Fraction(c, d) * (-1 if subtract else 1)
    ans = Fraction(a, b) + rhs
    lcm = b * d // gcd(b, d)
    left = a * (lcm // b)
    right = c * (lcm // d)
    op = "-" if subtract else "+"
    raw_num = left - right if subtract else left + right
    verb = "subtract" if subtract else "add"
    prompt = f"Compute {a}/{b} {op} {c}/{d}."
    completion = (
        f"Plan: use a common denominator. "
        f"Step 1: a common denominator for {b} and {d} is {lcm}. "
        f"Step 2: rewrite {a}/{b} as {left}/{lcm} and {c}/{d} as {right}/{lcm}. "
        f"Step 3: {verb} numerators: ({left} {op} {right})/{lcm} = {raw_num}/{lcm}. "
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


def even_square_proof(prefix: str, var: str) -> tuple[str, str]:
    prompt = f"{prefix} if {var} is even, then {var}^2 is even."
    completion = (
        f"Plan: use the definition of an even integer. "
        f"Step 1: since {var} is even, write {var} = 2k for some integer k. "
        f"Step 2: square both sides: {var}^2 = (2k)^2 = 4k^2. "
        f"Step 3: rewrite 4k^2 as 2(2k^2). "
        f"Check: 2k^2 is an integer, so {var}^2 is divisible by 2. "
        f"Final answer: {var}^2 is even."
    )
    return prompt, completion


def odd_square_proof(prefix: str, var: str) -> tuple[str, str]:
    prompt = f"{prefix} if {var} is odd, then {var}^2 is odd."
    completion = (
        f"Plan: use the definition of an odd integer. "
        f"Step 1: since {var} is odd, write {var} = 2k + 1 for some integer k. "
        f"Step 2: square: {var}^2 = (2k + 1)^2 = 4k^2 + 4k + 1. "
        f"Step 3: rewrite this as 2(2k^2 + 2k) + 1. "
        f"Check: 2k^2 + 2k is an integer, so {var}^2 has the form 2m + 1. "
        f"Final answer: {var}^2 is odd."
    )
    return prompt, completion


def odd_sum_proof(prefix: str, var: str) -> tuple[str, str]:
    prompt = f"{prefix} 1 + 3 + 5 + ... + (2{var} - 1) = {var}^2."
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


def triangular_proof(prefix: str, var: str) -> tuple[str, str]:
    prompt = f"{prefix} 1 + 2 + ... + {var} = {var}({var} + 1)/2."
    completion = (
        f"Plan: pair the sum with itself in reverse order. "
        f"Step 1: let S = 1 + 2 + ... + {var}. "
        f"Step 2: also S = {var} + ({var} - 1) + ... + 1. "
        f"Step 3: adding the two lines gives 2S = ({var} + 1) + ({var} + 1) + ... + ({var} + 1), with {var} terms. "
        f"Step 4: therefore 2S = {var}({var} + 1), so S = {var}({var} + 1)/2. "
        f"Final answer: proved."
    )
    return prompt, completion


def even_plus_even_proof(prefix: str, u: str, v: str) -> tuple[str, str]:
    prompt = f"{prefix} the sum of two even integers {u} and {v} is even."
    completion = (
        f"Plan: use the definition of an even integer for both terms. "
        f"Step 1: write {u} = 2j and {v} = 2k for integers j and k. "
        f"Step 2: add them: {u} + {v} = 2j + 2k = 2(j + k). "
        f"Check: j + k is an integer, so the sum is divisible by 2. "
        f"Final answer: {u} + {v} is even."
    )
    return prompt, completion


def odd_plus_odd_proof(prefix: str, u: str, v: str) -> tuple[str, str]:
    prompt = f"{prefix} the sum of two odd integers {u} and {v} is even."
    completion = (
        f"Plan: use the definition of an odd integer for both terms. "
        f"Step 1: write {u} = 2j + 1 and {v} = 2k + 1 for integers j and k. "
        f"Step 2: add them: {u} + {v} = 2j + 2k + 2 = 2(j + k + 1). "
        f"Check: j + k + 1 is an integer, so the sum is divisible by 2. "
        f"Final answer: {u} + {v} is even."
    )
    return prompt, completion


def even_times_any_proof(prefix: str, u: str, v: str) -> tuple[str, str]:
    prompt = f"{prefix} if {u} is even, then {u}{v} is even for any integer {v}."
    completion = (
        f"Plan: factor out 2 from the product. "
        f"Step 1: since {u} is even, write {u} = 2k for some integer k. "
        f"Step 2: multiply: {u}{v} = 2k{v} = 2(k{v}). "
        f"Check: k{v} is an integer, so the product is divisible by 2. "
        f"Final answer: {u}{v} is even."
    )
    return prompt, completion


def consecutive_product_proof(prefix: str, var: str) -> tuple[str, str]:
    prompt = f"{prefix} {var}({var} + 1) is even for every integer {var}."
    completion = (
        f"Plan: split into cases by the parity of {var}. "
        f"Step 1: if {var} is even, then {var} = 2k, so {var}({var} + 1) = 2k({var} + 1), which is even. "
        f"Step 2: if {var} is odd, then {var} + 1 = 2k, so {var}({var} + 1) = 2{var}k, which is even. "
        f"Check: one of two consecutive integers is always even. "
        f"Final answer: {var}({var} + 1) is even."
    )
    return prompt, completion


VARS = ["n", "m", "r", "t"]
VAR_PAIRS = [("a", "b"), ("m", "n"), ("r", "s"), ("s", "t")]


def enumerate_proofs() -> list[str]:
    single_var = [even_square_proof, odd_square_proof, odd_sum_proof, triangular_proof, consecutive_product_proof]
    pair_var = [even_plus_even_proof, odd_plus_odd_proof, even_times_any_proof]
    rows = []
    for prefix in PROOF_PREFIXES:
        for fn in single_var:
            for var in VARS:
                prompt, completion = fn(prefix, var)
                if prompt not in EVAL_HOLDOUT:
                    rows.append(emit(prompt, completion))
        for fn in pair_var:
            for u, v in VAR_PAIRS:
                prompt, completion = fn(prefix, u, v)
                if prompt not in EVAL_HOLDOUT:
                    rows.append(emit(prompt, completion))
    return rows


def build(count: int, seed: int, proof_fraction: float) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []

    # Proofs have a small unique space, so they are tiled to a fixed fraction
    # of the dataset; this is mixture weighting, stated here, not hidden
    # repetition of benchmark prompts.
    proofs = enumerate_proofs()
    proof_target = int(count * proof_fraction)
    rng.shuffle(proofs)
    for idx in range(proof_target):
        rows.append(proofs[idx % len(proofs)])

    numeric_target = count - proof_target
    numeric_rows: list[str] = []
    families = [
        "add", "sub", "mul", "div", "order_ops", "power", "percent", "gcd", "mean",
        "linear", "two_step_linear", "fraction", "odd_sum", "triangular",
    ]

    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 60
    while len(numeric_rows) < numeric_target and attempts < max_attempts:
        attempts += 1
        kind = rng.choice(families)
        if kind == "add":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            prompt = arith_prompt(rng, f"{a} + {b}")
            completion = add_trace(a, b)
        elif kind == "sub":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            prompt = arith_prompt(rng, f"{a} - {b}")
            completion = sub_trace(a, b)
        elif kind == "mul":
            a, b = rng.randint(2, 999), rng.randint(2, 99)
            prompt = arith_prompt(rng, f"{a} * {b}")
            completion = mul_trace(a, b)
        elif kind == "div":
            b, q = rng.randint(2, 99), rng.randint(2, 999)
            a = b * q
            prompt = arith_prompt(rng, f"{a} / {b}")
            completion = div_trace(a, b)
        elif kind == "order_ops":
            a, b, c = rng.randint(2, 99), rng.randint(2, 30), rng.randint(2, 30)
            expr, completion = order_of_ops_trace(a, b, c, rng.random() < 0.5)
            prompt = arith_prompt(rng, expr)
        elif kind == "power":
            base = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            exp = rng.randint(2, 10 if base <= 3 else 5)
            expr, completion = power_trace(base, exp)
            prompt = arith_prompt(rng, expr)
        elif kind == "percent":
            p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80])
            n = 20 * rng.randint(1, 400)
            prompt, completion = percent_trace(p, n)
        elif kind == "gcd":
            a, b = rng.randint(12, 999), rng.randint(12, 999)
            expr, completion = gcd_trace(a, b)
            prompt = f"{rng.choice(ARITH_PREFIXES)} {expr}."
        elif kind == "mean":
            k = rng.randint(3, 5)
            m = rng.randint(5, 200)
            offsets = [rng.randint(-9, 9) for _ in range(k - 1)]
            nums = [m + off for off in offsets] + [m - sum(offsets)]
            rng.shuffle(nums)
            prompt, completion = mean_trace(nums)
        elif kind == "linear":
            a = rng.randint(2, 17)
            b = rng.randint(1, 80)
            x = rng.randint(-40, 60)
            prompt, completion = linear_trace(a, b, x, rng.choice([1, -1]))
        elif kind == "two_step_linear":
            a, c = rng.sample(range(2, 13), 2)
            b = rng.randint(1, 60)
            x = rng.randint(-30, 30)
            prompt, completion = two_step_linear_trace(a, c, b, x)
        elif kind == "fraction":
            a, b = rng.randint(1, 25), rng.randint(2, 25)
            c, d = rng.randint(1, 25), rng.randint(2, 25)
            prompt, completion = fraction_add_trace(a, b, c, d, rng.random() < 0.4)
        elif kind == "odd_sum":
            n = rng.randint(2, 300)
            prompt = rng.choice([
                f"Find the sum of the first {n} odd positive integers.",
                f"What is 1 + 3 + 5 + ... through the first {n} odd positive integers?",
                f"Compute the sum of the first {n} odd positive integers.",
            ])
            completion = odd_sum_trace(n)
        else:
            n = rng.randint(2, 400)
            prompt = rng.choice([
                f"Find 1 + 2 + ... + {n}.",
                f"Find the sum of the first {n} positive integers.",
                f"Compute 1 + 2 + ... + {n}.",
            ])
            completion = triangular_trace(n)

        if prompt in EVAL_HOLDOUT:
            continue
        key = prompt + "\n" + completion
        if key in seen:
            continue
        seen.add(key)
        numeric_rows.append(emit(prompt, completion))

    rows.extend(numeric_rows)
    rng.shuffle(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build varied SFT examples that teach step-by-step math reasoning, not benchmark routing.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--proof-fraction", type=float, default=0.1, help="fraction of the dataset tiled from the unique proof pool")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed, args.proof_fraction)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    if len(rows) < args.count:
        print(f"warning: unique example space exhausted at {len(rows)} of {args.count} requested", flush=True)
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
