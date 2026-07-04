#!/usr/bin/env python3
"""Code reasoning curriculum (expert #4): task -> Python function that solves it.

Unlike the arithmetic lobes, code is verified by EXECUTION. Each task carries an
INDEPENDENT oracle (a builtin or a separate implementation). At build time every
taught solution is executed against the oracle on random inputs — if a taught
solution ever disagrees with its oracle, the build fails loudly (the analog of
the "4800/4800 internally correct" check). The benchmark then runs the MODEL's
generated code against fresh oracle-checked inputs (execution-based pass@1).

Narrow by design: a 117M from-scratch model learns a fixed set of algorithmic
task types, exactly as the chemistry lobe learns a fixed set of relations.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_reasoning_curriculum import emit


@dataclass
class Task:
    name: str
    call: str                       # function name the solution must define
    prompt: str                     # natural-language spec
    solution: str                   # taught reference implementation (source)
    oracle: Callable                # INDEPENDENT ground truth
    gen_input: Callable             # rng -> tuple of positional args


def _rand_ints(rng, lo=-50, hi=50, n_lo=0, n_hi=8):
    return [rng.randint(lo, hi) for _ in range(rng.randint(n_lo, n_hi))]


def _rand_word(rng, n_lo=1, n_hi=12):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(n_lo, n_hi)))


def _rand_sentence(rng):
    return " ".join(_rand_word(rng) for _ in range(rng.randint(1, 8)))


# --- task definitions: (spec, taught solution, independent oracle, input gen) ---

TASKS: list[Task] = [
    Task("sum_list", "sum_list",
         "Write a Python function `sum_list(nums)` that returns the sum of the integers in the list `nums`.",
         "def sum_list(nums):\n    total = 0\n    for x in nums:\n        total += x\n    return total\n",
         lambda nums: sum(nums),
         lambda rng: (_rand_ints(rng),)),
    Task("max_list", "max_list",
         "Write a Python function `max_list(nums)` that returns the largest integer in the non-empty list `nums`.",
         "def max_list(nums):\n    best = nums[0]\n    for x in nums:\n        if x > best:\n            best = x\n    return best\n",
         lambda nums: max(nums),
         lambda rng: (_rand_ints(rng, n_lo=1),)),
    Task("reverse_string", "reverse_string",
         "Write a Python function `reverse_string(s)` that returns the string `s` reversed.",
         "def reverse_string(s):\n    return s[::-1]\n",
         lambda s: s[::-1],
         lambda rng: (_rand_word(rng, 0, 14),)),
    Task("count_vowels", "count_vowels",
         "Write a Python function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u) in the lowercase string `s`.",
         "def count_vowels(s):\n    n = 0\n    for c in s:\n        if c in 'aeiou':\n            n += 1\n    return n\n",
         lambda s: sum(c in "aeiou" for c in s),
         lambda rng: (_rand_word(rng, 0, 16),)),
    Task("is_palindrome", "is_palindrome",
         "Write a Python function `is_palindrome(s)` that returns True if the string `s` reads the same forwards and backwards, else False.",
         "def is_palindrome(s):\n    return s == s[::-1]\n",
         lambda s: s == s[::-1],
         lambda rng: (rng.choice([_rand_word(rng, 0, 8), (lambda w: w + w[::-1])(_rand_word(rng, 1, 5))]),)),
    Task("factorial", "factorial",
         "Write a Python function `factorial(n)` that returns n! (the factorial of the non-negative integer `n`).",
         "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
         lambda n: math.factorial(n),
         lambda rng: (rng.randint(0, 12),)),
    Task("fib", "fib",
         "Write a Python function `fib(n)` that returns the n-th Fibonacci number, with fib(0)=0 and fib(1)=1.",
         "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
         lambda n: _fib_oracle(n),
         lambda rng: (rng.randint(0, 20),)),
    Task("is_prime", "is_prime",
         "Write a Python function `is_prime(n)` that returns True if the integer `n` is prime, else False.",
         "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n",
         lambda n: _is_prime_oracle(n),
         lambda rng: (rng.randint(-2, 60),)),
    Task("gcd", "gcd",
         "Write a Python function `gcd(a, b)` that returns the greatest common divisor of the positive integers `a` and `b`.",
         "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
         lambda a, b: math.gcd(a, b),
         lambda rng: (rng.randint(1, 99), rng.randint(1, 99))),
    Task("filter_even", "filter_even",
         "Write a Python function `filter_even(nums)` that returns a list of the even integers from `nums`, in order.",
         "def filter_even(nums):\n    return [x for x in nums if x % 2 == 0]\n",
         lambda nums: [x for x in nums if x % 2 == 0],
         lambda rng: (_rand_ints(rng),)),
    Task("sum_range", "sum_range",
         "Write a Python function `sum_range(n)` that returns the sum of all integers from 1 to `n` inclusive.",
         "def sum_range(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total\n",
         lambda n: n * (n + 1) // 2,
         lambda rng: (rng.randint(1, 100),)),
    Task("count_words", "count_words",
         "Write a Python function `count_words(s)` that returns the number of whitespace-separated words in the string `s`.",
         "def count_words(s):\n    return len(s.split())\n",
         lambda s: len(s.split()),
         lambda rng: (_rand_sentence(rng),)),
    Task("digit_sum", "digit_sum",
         "Write a Python function `digit_sum(n)` that returns the sum of the digits of the non-negative integer `n`.",
         "def digit_sum(n):\n    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total\n",
         lambda n: sum(int(d) for d in str(n)),
         lambda rng: (rng.randint(0, 99999),)),
    Task("second_largest", "second_largest",
         "Write a Python function `second_largest(nums)` that returns the second-largest distinct value in `nums` (which has at least two distinct values).",
         "def second_largest(nums):\n    uniq = sorted(set(nums))\n    return uniq[-2]\n",
         lambda nums: sorted(set(nums))[-2],
         lambda rng: (_two_distinct(rng),)),
    Task("count_char", "count_char",
         "Write a Python function `count_char(s, ch)` that returns how many times the character `ch` appears in the string `s`.",
         "def count_char(s, ch):\n    n = 0\n    for c in s:\n        if c == ch:\n            n += 1\n    return n\n",
         lambda s, ch: s.count(ch),
         lambda rng: (_rand_word(rng, 3, 16), rng.choice(string.ascii_lowercase))),
]


def _fib_oracle(n):
    # recursive — independent from the taught iterative solution
    if n < 2:
        return n
    return _fib_oracle(n - 1) + _fib_oracle(n - 2)


def _is_prime_oracle(n):
    if n < 2:
        return False
    return all(n % i for i in range(2, int(n ** 0.5) + 1))


def _two_distinct(rng):
    xs = _rand_ints(rng, n_lo=2, n_hi=8)
    while len(set(xs)) < 2:
        xs = _rand_ints(rng, n_lo=2, n_hi=8)
    return xs


PREFIXES = ["", "Plan: define the function, then implement it directly.\n"]

# Single-edit bug injections. Each is applied once to a correct reference; the
# result is KEPT as a debugging example only if it actually fails the oracle
# (so every "bug" is a real, execution-confirmed bug — not a cosmetic change).
MUTATIONS = [
    (" + ", " - "), (" - ", " + "), (" > ", " < "), (" < ", " > "),
    (" >= ", " > "), (" <= ", " < "), (" == ", " != "), (" != ", " == "),
    (" += ", " -= "), ("n + 1", "n"), ("[0]", "[1]"), ("[-2]", "[-1]"),
    ("% 2 == 0", "% 2 != 0"), ("range(2,", "range(1,"), ("i * i", "i"),
]


# Hand-authored bugs for tasks whose one-line solutions have no auto-mutation
# surface. All are hang-safe (no infinite loops) and wrong-by-construction.
MANUAL_BUGS = {
    "reverse_string": ["def reverse_string(s):\n    return s[1:][::-1]\n"],
    "gcd": ["def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return b\n"],
    "count_words": ["def count_words(s):\n    return len(s.split()) + 1\n"],
}


def make_buggy_variants(task: Task, rng: random.Random, trials: int = 40) -> list[str]:
    """Return reference-derived variants that are EXECUTION-CONFIRMED broken:
    they define the function but disagree with the oracle on >=1 input."""
    variants: list[str] = []
    candidates = [task.solution.replace(o, n, 1) for o, n in MUTATIONS if o in task.solution]
    candidates += MANUAL_BUGS.get(task.name, [])
    for buggy in candidates:
        if buggy == task.solution:
            continue
        try:
            ns: dict = {}
            exec(buggy, ns)
            fn = ns.get(task.call)
            if fn is None:
                continue
        except Exception:
            continue
        wrong = False
        for _ in range(trials):
            args = task.gen_input(rng)
            try:
                if fn(*args) != task.oracle(*args):
                    wrong = True
                    break
            except Exception:
                wrong = True
                break
        if wrong:
            variants.append(buggy)
    return variants


def debug_prompt(task: Task, buggy: str) -> str:
    return (f"The function below is meant to solve this task but contains a bug:\n"
            f"{task.prompt}\n\n{buggy.rstrip()}\n\nReturn a corrected version of the function.")


def verify_task(task: Task, rng: random.Random, trials: int = 200) -> int:
    """Execute the taught solution against the independent oracle. Returns the
    number of mismatches (must be 0)."""
    ns: dict = {}
    exec(task.solution, ns)  # taught solution is our own trusted source
    fn = ns[task.call]
    bad = 0
    for _ in range(trials):
        args = task.gen_input(rng)
        try:
            got = fn(*args)
            want = task.oracle(*args)
        except Exception:
            bad += 1
            continue
        if got != want:
            bad += 1
    return bad


def build(count: int, seed: int) -> tuple[list[str], dict]:
    """Emit a mix of WRITE (spec->code) and DEBUG (buggy->fixed) examples, both
    execution-verified. Returns (rows, stats)."""
    rng = random.Random(seed)
    per_task = max(2, count // len(TASKS))
    rows: list[str] = []
    stats = {"write": 0, "debug": 0, "bugs_per_task": {}}
    for task in TASKS:
        variants = make_buggy_variants(task, rng)
        stats["bugs_per_task"][task.name] = len(variants)
        n_debug = per_task // 2 if variants else 0
        n_write = per_task - n_debug
        for _ in range(n_write):
            rows.append(emit(task.prompt, rng.choice(PREFIXES) + task.solution))
        for i in range(n_debug):
            buggy = variants[i % len(variants)]
            rows.append(emit(debug_prompt(task, buggy), task.solution))
        stats["write"] += n_write
        stats["debug"] += n_debug
    rng.shuffle(rows)
    return rows, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--count", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify_only and args.out is None:
        parser.error("--out is required unless --verify-only")

    # build-time verification: every taught solution must match its oracle
    vrng = random.Random(1234)
    total_bad = 0
    for task in TASKS:
        bad = verify_task(task, vrng)
        total_bad += bad
        print(f"  {task.name:16} mismatches={bad}/200", file=sys.stderr)
    if total_bad:
        raise SystemExit(f"VERIFICATION FAILED: {total_bad} taught-solution/oracle mismatches")
    print(f"verification: all {len(TASKS)} tasks 0 mismatches", file=sys.stderr)
    if args.verify_only:
        return 0

    rows, stats = build(args.count, args.seed)
    zero_bug = [t for t, n in stats["bugs_per_task"].items() if n == 0]
    print(f"verified bugs/task: {stats['bugs_per_task']}", file=sys.stderr)
    if zero_bug:
        print(f"WARNING: no confirmed bug variants for: {zero_bug}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "examples": len(rows), "tasks": len(TASKS),
                      "write": stats["write"], "debug": stats["debug"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
