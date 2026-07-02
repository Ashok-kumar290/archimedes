from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class RoutedAnswer:
    kind: str
    prompt: str
    completion: str

    @property
    def output(self) -> str:
        return f"Problem: {self.prompt}\nSolution: {self.completion}"


def _fmt_number(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _parse_int(text: str) -> int:
    return int(text.replace(",", ""))


def _compute_binary(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(r"\s*Compute\s+(-?\d[\d,]*)\s*([+\-*x×])\s*(-?\d[\d,]*)\s*\.\s*", prompt, re.IGNORECASE)
    if not match:
        return None
    a = _parse_int(match.group(1))
    op = match.group(2)
    b = _parse_int(match.group(3))
    if op == "+":
        ans = a + b
        completion = f"Add the integers: {a} + {b} = {ans}. Final answer: {ans}."
    elif op == "-":
        ans = a - b
        completion = f"Subtract the integers: {a} - {b} = {ans}. Final answer: {ans}."
    else:
        ans = a * b
        completion = f"Multiply the integers: {a} * {b} = {ans}. Final answer: {ans}."
    return RoutedAnswer("binary_arithmetic", prompt.strip(), completion)


def _solve_linear(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(
        r"\s*Solve\s+for\s+x:\s*(-?\d+)\s*x\s*([+-])\s*(\d+)\s*=\s*(-?\d+)\s*\.\s*",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    a = int(match.group(1))
    sign = match.group(2)
    b = int(match.group(3))
    c = int(match.group(4))
    signed_b = b if sign == "+" else -b
    rhs = c - signed_b
    x = Fraction(rhs, a)
    x_text = _fmt_number(x)
    completion = (
        f"Start with {a}x {'+' if signed_b >= 0 else '-'} {abs(signed_b)} = {c}. "
        f"Move the constant term: {a}x = {rhs}. "
        f"Divide by {a}: x = {x_text}. Final answer: x = {x_text}."
    )
    return RoutedAnswer("linear_equation", prompt.strip(), completion)


def _triangular_sum(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(r"\s*Find\s+1\s*\+\s*2\s*\+\s*\.\.\.\s*\+\s*(\d+)\s*\.\s*", prompt, re.IGNORECASE)
    if not match:
        return None
    n = int(match.group(1))
    ans = n * (n + 1) // 2
    completion = f"Use n(n + 1)/2 with n = {n}: {n} * {n + 1} / 2 = {ans}. Final answer: {ans}."
    return RoutedAnswer("triangular_sum", prompt.strip(), completion)


def _odd_sum(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(
        r"\s*Find\s+the\s+sum\s+of\s+the\s+first\s+(\d+)\s+odd\s+positive\s+integers\s*\.\s*",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    n = int(match.group(1))
    ans = n * n
    completion = f"The first {n} odd positive integers sum to {n}^2 = {ans}. Final answer: {ans}."
    return RoutedAnswer("odd_sum", prompt.strip(), completion)


def _even_square_proof(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(r"\s*Prove\s+that\s+if\s+n\s+is\s+even,\s+then\s+n\^2\s+is\s+even\s*\.\s*", prompt, re.IGNORECASE)
    if not match:
        return None
    completion = (
        "If n is even, then n = 2k for some integer k. "
        "Squaring gives n^2 = (2k)^2 = 4k^2 = 2(2k^2). "
        "Since 2k^2 is an integer, n^2 is even. Final answer: proved."
    )
    return RoutedAnswer("even_square_proof", prompt.strip(), completion)


def _odd_square_proof(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(r"\s*Prove\s+that\s+if\s+n\s+is\s+odd,\s+then\s+n\^2\s+is\s+odd\s*\.\s*", prompt, re.IGNORECASE)
    if not match:
        return None
    completion = (
        "If n is odd, then n = 2k + 1 for some integer k. "
        "Then n^2 = (2k + 1)^2 = 4k^2 + 4k + 1 = 2(2k^2 + 2k) + 1. "
        "Since 2k^2 + 2k is an integer, n^2 is odd. Final answer: proved."
    )
    return RoutedAnswer("odd_square_proof", prompt.strip(), completion)


def _odd_sum_proof(prompt: str) -> RoutedAnswer | None:
    match = re.fullmatch(
        r"\s*Prove\s+that\s+1\s*\+\s*3\s*\+\s*5\s*\+\s*\.\.\.\s*\+\s*\(2n\s*-\s*1\)\s*=\s*n\^2\s*\.\s*",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    completion = (
        "We prove the formula by induction on n. "
        "For n = 1, the left side is 1 and the right side is 1^2 = 1. "
        "Assume 1 + 3 + 5 + ... + (2n - 1) = n^2. "
        "The next odd term is 2(n + 1) - 1 = 2n + 1. "
        "Adding it gives n^2 + 2n + 1 = (n + 1)^2. "
        "Therefore the formula holds for n + 1, so by induction it holds for all positive integers n. "
        "Final answer: proved by induction."
    )
    return RoutedAnswer("odd_sum_proof", prompt.strip(), completion)


def _triangular_sum_proof(prompt: str) -> RoutedAnswer | None:
    patterns = [
        r"\s*Prove\s+that\s+1\s*\+\s*2\s*\+\s*\.\.\.\s*\+\s*n\s*=\s*n\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2\s*\.\s*",
        r"\s*Prove\s+that\s+the\s+sum\s+of\s+the\s+first\s+n\s+positive\s+integers\s+is\s+n\s*\(\s*n\s*\+\s*1\s*\)\s*/\s*2\s*\.\s*",
    ]
    if not any(re.fullmatch(pattern, prompt, re.IGNORECASE) for pattern in patterns):
        return None
    completion = (
        "Let S = 1 + 2 + ... + n. "
        "Write the same sum in reverse: S = n + (n - 1) + ... + 1. "
        "Adding the two equations gives 2S = (n + 1) + (n + 1) + ... + (n + 1), with n terms. "
        "Thus 2S = n(n + 1), so S = n(n + 1)/2. Final answer: proved."
    )
    return RoutedAnswer("triangular_sum_proof", prompt.strip(), completion)


def route(prompt: str) -> RoutedAnswer | None:
    for solver in (
        _compute_binary,
        _solve_linear,
        _triangular_sum,
        _odd_sum,
        _even_square_proof,
        _odd_square_proof,
        _odd_sum_proof,
        _triangular_sum_proof,
    ):
        answer = solver(prompt)
        if answer is not None:
            return answer
    return None
