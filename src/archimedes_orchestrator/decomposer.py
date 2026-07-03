"""Symbolic decomposer (v1) + lobe-call executor.

The decomposer holds *symbolic structure* — "kinematics means multiply then
add" — as a small, auditable rule set. The lobe holds *arithmetic skill* — the
actual `3 * 8 = 24`, in a neural forward pass. This split is the whole point:

    THE HONESTY INVARIANT
    The executor NEVER computes an arithmetic answer itself. Every operation is
    a single-op question handed to the lobe; the result is parsed from the
    lobe's text output. No answer number ever comes from a table or from Python
    arithmetic. If we ever let the executor compute `a * b`, we have cheated.

A Plan is a straight-line sequence of Steps (a DAG in general; these formulas
are linear chains). Each Step's args are either integer literals pulled from the
problem, or a Ref(i) pointing at the output of an earlier step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# --- plan representation ---

@dataclass(frozen=True)
class Ref:
    """Reference to the output of step `index` in the same plan."""
    index: int


Arg = int | Ref

OP_SYMBOL = {"mul": "*", "add": "+", "div": "/", "sub": "-"}


@dataclass(frozen=True)
class Step:
    op: str          # one of OP_SYMBOL
    a: Arg
    b: Arg


@dataclass(frozen=True)
class Plan:
    family: str
    steps: tuple[Step, ...]
    unit: str        # unit of the final answer, for presentation only


# --- number parsing (mirrors benchmark_physics.observed_number) ---

def parse_number(text: str | None) -> int | None:
    """Pull the final integer from a lobe's text answer. Take the LAST number:
    a step-by-step trace writes the working before the answer."""
    if text is None:
        return None
    nums = re.findall(r"-?\d[\d,]*", text.replace(" ", ""))
    return int(nums[-1].replace(",", "")) if nums else None


# --- executor ---

@dataclass
class ExecResult:
    answer: int | None
    steps: list[dict]        # per-step audit: question asked, raw reply, parsed value
    ok: bool


def _resolve(arg: Arg, values: list[int]) -> int:
    return values[arg.index] if isinstance(arg, Ref) else arg


def execute(plan: Plan, lobe) -> ExecResult:
    """Run a plan by asking `lobe` one single-op question per step.

    `lobe(question: str) -> str` is any callable that maps an arithmetic
    question to a text answer — the real Archimedes checkpoint in production,
    or a MockLobe in tests. This function performs NO arithmetic of its own; it
    only substitutes prior results into later questions and parses replies.
    """
    values: list[int] = []
    audit: list[dict] = []
    for step in plan.steps:
        a = _resolve(step.a, values)
        b = _resolve(step.b, values)
        question = f"Compute {a} {OP_SYMBOL[step.op]} {b}."
        reply = lobe(question)               # <-- the lobe computes; we do not
        value = parse_number(reply)
        audit.append({"question": question, "reply": reply, "value": value})
        if value is None:
            return ExecResult(answer=None, steps=audit, ok=False)
        values.append(value)
    return ExecResult(answer=values[-1], steps=audit, ok=True)


# --- formula templates: regex on the problem -> Plan ---
# Each entry pairs a pattern (matching the physics word-problem phrasing) with a
# builder that names the captured groups and returns the operation tree. The
# numbers are extracted from the problem; the STRUCTURE is the symbolic
# knowledge. g is read from the prompt (not hardcoded) where the family states it.

def _newton(m: int, a: int) -> Plan:
    return Plan("newton", (Step("mul", m, a),), "N")


def _momentum(m: int, v: int) -> Plan:
    return Plan("momentum", (Step("mul", m, v),), "kg m/s")


def _weight(m: int, g: int) -> Plan:
    return Plan("weight", (Step("mul", m, g),), "N")


def _potential(m: int, h: int, g: int) -> Plan:
    # PE = m * g * h, computed as (m * h) * g: one clean multiply, then * g
    return Plan("potential", (Step("mul", m, h), Step("mul", Ref(0), g)), "J")


def _kinematics(u: int, a: int, t: int) -> Plan:
    # v = u + a * t: multiply, then add
    return Plan("kinematics", (Step("mul", a, t), Step("add", u, Ref(0))), "m/s")


def _work(F: int, d: int) -> Plan:
    return Plan("work", (Step("mul", F, d),), "J")


def _ohm_v(I: int, R: int) -> Plan:
    return Plan("ohm_v", (Step("mul", I, R),), "V")


def _ohm_i(V: int, R: int) -> Plan:
    return Plan("ohm_i", (Step("div", V, R),), "A")


def _density(m: int, V: int) -> Plan:
    return Plan("density", (Step("div", m, V),), "kg/m^3")


def _power(W: int, t: int) -> Plan:
    return Plan("power", (Step("div", W, t),), "W")


# (compiled_pattern, builder). Order doesn't matter — patterns are disjoint.
_TEMPLATES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"A (\d+) kg object accelerates at (\d+) m/s\^2\. Find the net force\."),
     lambda m: _newton(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"An object of mass (\d+) kg moves at (\d+) m/s\. Find its momentum\."),
     lambda m: _momentum(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"Find the weight of a (\d+) kg object\. Use g = (\d+) m/s\^2\."),
     lambda m: _weight(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"A (\d+) kg object is lifted to a height of (\d+) m\. Find its potential energy\. Use g = (\d+) m/s\^2\."),
     lambda m: _potential(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
    (re.compile(r"A car travels at (\d+) m/s and accelerates at (\d+) m/s\^2 for (\d+) s\. Find its final velocity\."),
     lambda m: _kinematics(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
    (re.compile(r"A force of (\d+) N moves an object (\d+) m in the direction of the force\. Find the work done\."),
     lambda m: _work(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"A current of (\d+) A flows through a (\d+) ohm resistor\. Find the voltage\."),
     lambda m: _ohm_v(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"A voltage of (\d+) V is applied across a (\d+) ohm resistor\. Find the current\."),
     lambda m: _ohm_i(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"An object has mass (\d+) kg and volume (\d+) m\^3\. Find its density\."),
     lambda m: _density(int(m.group(1)), int(m.group(2)))),
    (re.compile(r"An engine does (\d+) J of work in (\d+) s\. Find its power output\."),
     lambda m: _power(int(m.group(1)), int(m.group(2)))),
]


def decompose(problem: str) -> Plan | None:
    """Return a Plan for a recognized physics word problem, or None if no
    template matches (the router would then fall back to a direct lobe call)."""
    for pattern, builder in _TEMPLATES:
        m = pattern.search(problem)
        if m is not None:
            return builder(m)
    return None
