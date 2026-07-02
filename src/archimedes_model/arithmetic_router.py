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


def route(prompt: str) -> RoutedAnswer | None:
    for solver in (_compute_binary, _solve_linear, _triangular_sum, _odd_sum):
        answer = solver(prompt)
        if answer is not None:
            return answer
    return None
