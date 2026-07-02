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
PLACES = ["units", "tens", "hundreds", "thousands", "ten-thousands"]
VARS = ["n", "m", "r", "t"]
VAR_PAIRS = [("a", "b"), ("m", "n"), ("r", "s"), ("s", "t")]


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


def digits_rev(n: int) -> list[int]:
    return [int(c) for c in str(n)][::-1]


def add_narration(a: int, b: int) -> tuple[str, int]:
    ans = a + b
    da, db = digits_rev(a), digits_rev(b)
    parts = []
    carry = 0
    for i in range(max(len(da), len(db))):
        x = da[i] if i < len(da) else 0
        y = db[i] if i < len(db) else 0
        s = x + y + carry
        seg = f"{PLACES[i]}: {x} + {y}" + (f" + {carry} carried" if carry else "") + f" = {s}"
        carry, write = divmod(s, 10)
        seg += f", write {write}" + (f", carry {carry}" if carry else "")
        parts.append(seg)
    if carry:
        parts.append(f"write the final carry {carry}")
    return "; ".join(parts) + f"; the digits give {ans}", ans


def sub_narration(a: int, b: int) -> tuple[str, int]:
    ans = a - b
    da, db = digits_rev(a), digits_rev(b)
    parts = []
    borrow = 0
    for i in range(len(da)):
        x = da[i]
        y = (db[i] if i < len(db) else 0) + borrow
        if x < y:
            parts.append(f"{PLACES[i]}: {x} - {y} needs a borrow, so {x + 10} - {y} = {x + 10 - y}")
            borrow = 1
        else:
            parts.append(f"{PLACES[i]}: {x} - {y} = {x - y}")
            borrow = 0
    return "; ".join(parts) + f"; the digits give {ans}", ans


def mul_narration(a: int, b: int) -> tuple[list[str], int]:
    ans = a * b
    partial_texts = []
    partials = []
    for i, x in enumerate(digits_rev(a)):
        for j, y in enumerate(digits_rev(b)):
            if x == 0 or y == 0:
                continue
            p = x * y * 10 ** (i + j)
            partial_texts.append(f"{x * 10 ** i} * {y * 10 ** j} = {p} since {x} * {y} = {x * y}")
            partials.append(p)
    steps = ["place-value products: " + "; ".join(partial_texts)]
    total = partials[0]
    for p in partials[1:]:
        narration, total = add_narration(total, p)
        steps.append(f"add the partials so far, {total - p} + {p}: {narration}")
    return steps, ans


def longdiv_narration(a: int, b: int) -> tuple[str, int]:
    q = a // b
    parts = []
    r = 0
    for ch in str(a):
        cur = r * 10 + int(ch)
        qd = cur // b
        r = cur - qd * b
        parts.append(f"bring down {ch} making {cur}; {b} * {qd} = {qd * b}, remainder {r}")
    return "; ".join(parts) + f"; the quotient digits give {q}", q


def digit_listing(n: int) -> str:
    ds = digits_rev(n)
    return ", ".join(f"{d} ({PLACES[i]})" for i, d in enumerate(ds))


def digits_trace(n: int) -> str:
    ds = digits_rev(n)
    reading = "; ".join(f"the {PLACES[i]} digit of {n} is {d}" for i, d in enumerate(ds))
    listing = ", ".join(str(d) for d in ds)
    return (
        f"Plan: read the number place by place starting at the units. "
        f"Step 1: {reading}. "
        f"Final answer: {listing}."
    )


def add_trace(a: int, b: int) -> str:
    narration, ans = add_narration(a, b)
    return (
        f"Plan: read off the digits of each number, then add column by column with carries. "
        f"Step 1: the digits of {a} from the units are {digit_listing(a)}; "
        f"the digits of {b} from the units are {digit_listing(b)}. "
        f"Step 2: {narration}. "
        f"Final answer: {ans}."
    )


def sub_trace(a: int, b: int) -> str:
    listing = (
        f"the digits of {a} from the units are {digit_listing(a)}; "
        f"the digits of {b} from the units are {digit_listing(b)}"
    )
    if a >= b:
        narration, ans = sub_narration(a, b)
        return (
            f"Plan: read off the digits of each number, then subtract column by column with borrows. "
            f"Step 1: {listing}. "
            f"Step 2: {narration}. "
            f"Final answer: {ans}."
        )
    narration, diff = sub_narration(b, a)
    return (
        f"Plan: since {b} is larger than {a}, compute {b} - {a} and negate the result. "
        f"Step 1: {listing}. "
        f"Step 2: {narration}. "
        f"Step 3: therefore {a} - {b} = -{diff}. "
        f"Final answer: {-diff}."
    )


def mul_trace(a: int, b: int) -> str:
    steps, ans = mul_narration(a, b)
    numbered = " ".join(f"Step {i + 1}: {s}." for i, s in enumerate(steps))
    return (
        f"Plan: multiply using place-value partial products, then add them. "
        f"{numbered} "
        f"Final answer: {ans}."
    )


def div_trace(a: int, b: int) -> str:
    narration, q = longdiv_narration(a, b)
    return (
        f"Plan: use long division digit by digit. "
        f"Step 1: {narration}. "
        f"Check: the remainder is 0, so {b} divides {a} exactly. "
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
    value = base
    idx = 1
    for _ in range(exp - 1):
        if value <= 12:
            steps.append(f"Step {idx}: {value} * {base} = {value * base}.")
            value *= base
        else:
            mul_steps, value = mul_narration(value, base)
            joined = " ".join(f"{s};" for s in mul_steps)
            steps.append(f"Step {idx}: multiply by {base}: {joined} giving {value}.")
        idx += 1
    return expr, (
        f"Plan: multiply by {base} one power at a time. "
        f"{' '.join(steps)} "
        f"Final answer: {value}."
    )


def percent_trace(p: int, n: int) -> tuple[str, str]:
    ans = p * n // 100
    prompt = f"What is {p}% of {n}?"
    if p == 50:
        narr, _ = longdiv_narration(n, 2)
        body = f"Step 1: 50% is one half. Step 2: divide by 2: {narr}."
    elif p == 25:
        half = n // 2
        n1, _ = longdiv_narration(n, 2)
        n2, _ = longdiv_narration(half, 2)
        body = f"Step 1: 25% is one quarter. Step 2: half of {n}: {n1}. Step 3: half of {half}: {n2}."
    elif p == 10:
        body = f"Step 1: 10% is one tenth; dividing by 10 shifts each digit one place: {n} / 10 = {ans}."
    else:
        tenth = n // 10
        narr, _ = add_narration(tenth, tenth)
        body = f"Step 1: 10% of {n} is {tenth}. Step 2: 20% is twice that; add {tenth} + {tenth}: {narr}."
    return prompt, (
        f"Plan: convert the percentage to a simple fraction. "
        f"{body} "
        f"Final answer: {ans}."
    )


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
    steps = []
    running = nums[0]
    idx = 1
    for n in nums[1:]:
        narration, running = add_narration(running, n)
        steps.append(f"Step {idx}: add {running - n} + {n}: {narration}.")
        idx += 1
    div_narr, _ = longdiv_narration(total, len(nums))
    steps.append(f"Step {idx}: divide by {len(nums)}: {div_narr}.")
    return prompt, (
        f"Plan: add the numbers one at a time, then divide by how many there are. "
        f"{' '.join(steps)} "
        f"Final answer: {ans}."
    )


def linear_trace(a: int, b: int, x: int, sign: int) -> tuple[str, str]:
    c = a * x + sign * b
    op = "+" if sign > 0 else "-"
    move = c - sign * b
    if sign > 0:
        narration, _ = sub_narration(c, b)
        move_step = f"subtract {b} from both sides: {c} - {b}: {narration}"
    else:
        narration, _ = add_narration(c, b)
        move_step = f"add {b} to both sides: {c} + {b}: {narration}"
    prompt = f"Solve for x: {a}x {op} {b} = {c}."
    completion = (
        f"Plan: isolate x using inverse operations. "
        f"Step 1: start with {a}x {op} {b} = {c}. "
        f"Step 2: {move_step}, so {a}x = {move}. "
        f"Step 3: divide by {a}: since {a} * {x} = {move}, x = {x}. "
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
        f"Step 3: subtract {b} from both sides: {coef}x = {const}. "
        f"Step 4: divide by {coef}: since {coef} * {x} = {const}, x = {x}. "
        f"Final answer: x = {x}."
    )
    return prompt, completion


def fraction_trace(a: int, b: int, c: int, d: int, subtract: bool) -> tuple[str, str]:
    rhs = Fraction(c, d) * (-1 if subtract else 1)
    ans = Fraction(a, b) + rhs
    lcm = b * d // gcd(b, d)
    left = a * (lcm // b)
    right = c * (lcm // d)
    op = "-" if subtract else "+"
    raw_num = left - right if subtract else left + right
    if subtract:
        if left >= right:
            narration, _ = sub_narration(left, right)
        else:
            narration, diff = sub_narration(right, left)
            narration += f"; since {right} is larger, the result is -{diff}"
    else:
        narration, _ = add_narration(left, right)
    prompt = f"Compute {a}/{b} {op} {c}/{d}."
    completion = (
        f"Plan: use a common denominator. "
        f"Step 1: a common denominator for {b} and {d} is {lcm}. "
        f"Step 2: rewrite {a}/{b} as {left}/{lcm} since {a} * {lcm // b} = {left}, "
        f"and {c}/{d} as {right}/{lcm} since {c} * {lcm // d} = {right}. "
        f"Step 3: combine numerators, {left} {op} {right}: {narration}; this gives {raw_num}/{lcm}. "
        f"Step 4: simplify to {fmt_fraction(ans)}. "
        f"Final answer: {fmt_fraction(ans)}."
    )
    return prompt, completion


def odd_sum_trace(n: int) -> str:
    steps, ans = mul_narration(n, n)
    numbered = " ".join(f"Step {i + 2}: {s}." for i, s in enumerate(steps))
    return (
        f"Plan: use the identity that the first n odd positive integers sum to n^2. "
        f"Step 1: here n = {n}, so compute {n} * {n}. "
        f"{numbered} "
        f"Final answer: {ans}."
    )


def triangular_trace(n: int) -> str:
    if n % 2 == 0:
        f1, f2 = n // 2, n + 1
        halving = f"half of {n} is {f1}, so the sum is {f1} * {n + 1}"
    else:
        f1, f2 = n, (n + 1) // 2
        halving = f"half of {n + 1} is {f2}, so the sum is {n} * {f2}"
    steps, ans = mul_narration(f1, f2)
    numbered = " ".join(f"Step {i + 3}: {s}." for i, s in enumerate(steps))
    return (
        f"Plan: use the formula 1 + 2 + ... + n = n(n + 1)/2. "
        f"Step 1: substitute n = {n}. "
        f"Step 2: {halving}. "
        f"{numbered} "
        f"Final answer: {ans}."
    )


def fact_pool() -> list[str]:
    rows = []
    for x in range(2, 10):
        for y in range(2, 10):
            for prefix in ARITH_PREFIXES:
                rows.append(emit(
                    f"{prefix} {x} + {y}.",
                    f"Plan: recall the basic addition fact. Step 1: {x} + {y} = {x + y}. Final answer: {x + y}.",
                ))
                rows.append(emit(
                    f"{prefix} {x} * {y}.",
                    f"Plan: recall the multiplication table. Step 1: {x} * {y} = {x * y}. Final answer: {x * y}.",
                ))
                rows.append(emit(
                    f"{prefix} {x * y} / {y}.",
                    f"Plan: use the multiplication table in reverse. Step 1: {y} * {x} = {x * y}, so {x * y} / {y} = {x}. Final answer: {x}.",
                ))
    for a in range(11, 19):
        for y in range(2, 10):
            if 1 <= a - y <= 9:
                for prefix in ARITH_PREFIXES:
                    rows.append(emit(
                        f"{prefix} {a} - {y}.",
                        f"Plan: recall the basic subtraction fact. Step 1: {a} - {y} = {a - y}. Final answer: {a - y}.",
                    ))
    return [r for r in rows if json.loads(r)["prompt"] not in EVAL_HOLDOUT]


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


def sum_formula_pools() -> tuple[list[str], list[str]]:
    odd_rows, tri_rows = [], []
    for n in range(2, 100):
        completion = odd_sum_trace(n)
        for prompt in (
            f"Find the sum of the first {n} odd positive integers.",
            f"What is 1 + 3 + 5 + ... through the first {n} odd positive integers?",
            f"Compute the sum of the first {n} odd positive integers.",
        ):
            if prompt not in EVAL_HOLDOUT:
                odd_rows.append(emit(prompt, completion))
        completion = triangular_trace(n)
        for prompt in (
            f"Find 1 + 2 + ... + {n}.",
            f"Find the sum of the first {n} positive integers.",
            f"Compute 1 + 2 + ... + {n}.",
        ):
            if prompt not in EVAL_HOLDOUT:
                tri_rows.append(emit(prompt, completion))
    return odd_rows, tri_rows


def build(count: int, seed: int, proof_fraction: float, facts_fraction: float) -> list[str]:
    rng = random.Random(seed)
    rows: list[str] = []

    # Proofs, single-digit facts, and the fully enumerated sum-formula
    # families have small unique spaces, so they are tiled to fixed fractions
    # of the dataset; this is stated mixture weighting, not hidden repetition
    # of benchmark prompts.
    odd_pool, tri_pool = sum_formula_pools()
    pools = (
        (enumerate_proofs(), proof_fraction),
        (fact_pool(), facts_fraction),
        (odd_pool, 0.015),
        (tri_pool, 0.015),
    )
    for pool, fraction in pools:
        target = int(count * fraction)
        rng.shuffle(pool)
        for idx in range(target):
            rows.append(pool[idx % len(pool)])

    numeric_target = count - len(rows)
    numeric_rows: list[str] = []
    # digit reading and column add/sub are weighted up: digit extraction from
    # multi-digit BPE tokens is the observed bottleneck
    families = [
        "digits", "digits", "add", "add", "sub", "sub",
        "mul", "div", "order_ops", "power", "percent", "gcd", "mean",
        "linear", "two_step_linear", "fraction",
    ]

    seen: set[str] = set()
    attempts = 0
    max_attempts = count * 60
    while len(numeric_rows) < numeric_target and attempts < max_attempts:
        attempts += 1
        kind = rng.choice(families)
        if kind == "digits":
            n = rng.randint(10, 9999)
            prompt = rng.choice([
                f"List the digits of {n} from the units place.",
                f"What are the digits of {n}, starting from the units?",
                f"Read off the digits of {n} from the units upward.",
            ])
            completion = digits_trace(n)
        elif kind == "add":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            prompt = arith_prompt(rng, f"{a} + {b}")
            completion = add_trace(a, b)
        elif kind == "sub":
            a, b = rng.randint(10, 9999), rng.randint(10, 9999)
            prompt = arith_prompt(rng, f"{a} - {b}")
            completion = sub_trace(a, b)
        elif kind == "mul":
            a, b = rng.randint(12, 99), rng.randint(2, 99)
            prompt = arith_prompt(rng, f"{a} * {b}")
            completion = mul_trace(a, b)
        elif kind == "div":
            b, q = rng.randint(2, 9), rng.randint(12, 999)
            a = b * q
            prompt = arith_prompt(rng, f"{a} / {b}")
            completion = div_trace(a, b)
        elif kind == "order_ops":
            a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 12)
            expr, completion = order_of_ops_trace(a, b, c, rng.random() < 0.5)
            prompt = arith_prompt(rng, expr)
        elif kind == "power":
            base = rng.randint(2, 12)
            max_exp = {2: 8, 3: 5, 4: 4, 5: 4, 6: 4, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3}[base]
            exp = rng.randint(2, max_exp)
            expr, completion = power_trace(base, exp)
            prompt = arith_prompt(rng, expr)
        elif kind == "percent":
            p = rng.choice([10, 20, 25, 50])
            n = 20 * rng.randint(1, 200)
            prompt, completion = percent_trace(p, n)
        elif kind == "gcd":
            a, b = rng.randint(12, 144), rng.randint(12, 144)
            expr, completion = gcd_trace(a, b)
            prompt = f"{rng.choice(ARITH_PREFIXES)} {expr}."
        elif kind == "mean":
            k = rng.randint(3, 4)
            m = rng.randint(5, 50)
            offsets = [rng.randint(-4, 4) for _ in range(k - 1)]
            nums = [m + off for off in offsets] + [m - sum(offsets)]
            if min(nums) < 1:
                continue
            rng.shuffle(nums)
            prompt, completion = mean_trace(nums)
        elif kind == "linear":
            a = rng.randint(2, 9)
            b = rng.randint(1, 20)
            x = rng.randint(2, 12)
            sign = rng.choice([1, -1])
            if sign < 0 and a * x - b <= 0:
                sign = 1
            prompt, completion = linear_trace(a, b, x, sign)
        elif kind == "two_step_linear":
            a = rng.randint(3, 9)
            c = rng.randint(2, a - 1)
            b = rng.randint(1, 20)
            x = rng.randint(2, 12)
            prompt, completion = two_step_linear_trace(a, c, b, x)
        else:
            a, b = rng.randint(1, 11), rng.randint(2, 12)
            c, d = rng.randint(1, 11), rng.randint(2, 12)
            prompt, completion = fraction_trace(a, b, c, d, rng.random() < 0.4)

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
    parser = argparse.ArgumentParser(description="Build SFT examples that teach digit-level arithmetic and step-by-step math reasoning.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--proof-fraction", type=float, default=0.1, help="fraction of the dataset tiled from the unique proof pool")
    parser.add_argument("--facts-fraction", type=float, default=0.12, help="fraction of the dataset tiled from the single-digit fact pool")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = build(args.count, args.seed, args.proof_fraction, args.facts_fraction)
    if len(rows) < args.count:
        print(f"warning: unique example space exhausted at {len(rows)} of {args.count} requested", flush=True)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
