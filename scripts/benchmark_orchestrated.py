#!/usr/bin/env python3
"""Orchestrated vs monolithic physics benchmark.

Holds the model FIXED and changes ONLY the orchestration:
  - monolithic:   ask the lobe the whole word problem in one shot (the existing
                  benchmark_physics path)
  - orchestrated: decompose into single-op sub-questions, ask the lobe each one,
                  chain the outputs (archimedes_orchestrator)

Same checkpoint, same weights, same held-out problems (seed 99991). Any lift is
attributable to decomposition alone — no new training, no confound.

Run modes:
  --self-test                      local plan-correctness check, no checkpoint
  --checkpoint X --tokenizer Y     real run on an Archimedes lobe (Colab)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark_physics import gen_problems, observed_number
from archimedes_orchestrator import decompose, execute


class MockLobe:
    """A test double that returns the EXACT arithmetic for a single-op question.

    Used only by --self-test to prove the templates + executor are structurally
    correct (right ops, right order, right operand wiring). It is NOT a model
    and says nothing about accuracy — a perfect lobe isolates plan logic from
    model error. The real accuracy lift is measured on the checkpoint.
    """
    name = "mock:exact-arithmetic"

    def __call__(self, question: str) -> str:
        # parse "Compute A <op> B." and evaluate — allowed HERE because this is
        # a stand-in for the model, not the orchestrator's executor.
        import re
        m = re.match(r"Compute (-?\d+) ([*+/-]) (-?\d+)\.", question)
        if not m:
            return "?"
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        val = {"*": a * b, "+": a + b, "-": a - b, "/": a // b}[op]
        return f"{a} {op} {b} = {val}"


def self_test(per_family: int, seed: int) -> int:
    """Every recognized problem must (a) decompose and (b) produce the correct
    final answer through a perfect lobe. Fails loudly on any gap."""
    problems = gen_problems(per_family, seed)
    lobe = MockLobe()
    unmatched: list[str] = []
    wrong: list[tuple[str, int, int | None]] = []
    per_family_ok: dict[str, list[bool]] = {}
    for p in problems:
        plan = decompose(p["prompt"])
        if plan is None:
            unmatched.append(p["prompt"])
            per_family_ok.setdefault(p["family"], []).append(False)
            continue
        result = execute(plan, lobe)
        ok = result.ok and result.answer == int(p["expected"])
        per_family_ok.setdefault(p["family"], []).append(ok)
        if not ok:
            wrong.append((p["prompt"], int(p["expected"]), result.answer))

    print(f"{'family':<12} {'plan-correct':>12}  n")
    all_ok = True
    for fam in sorted(per_family_ok):
        oks = per_family_ok[fam]
        rate = sum(oks) / len(oks)
        all_ok &= all(oks)
        print(f"{fam:<12} {rate:>11.0%}  {len(oks)}")

    if unmatched:
        print(f"\n{len(unmatched)} problems did NOT match any template, e.g.:")
        print(f"  {unmatched[0]}")
    if wrong:
        print(f"\n{len(wrong)} plans produced the WRONG answer, e.g.:")
        for prompt, exp, got in wrong[:3]:
            print(f"  expected {exp}, got {got}: {prompt}")
    print(f"\nSELF-TEST {'PASSED' if all_ok and not unmatched else 'FAILED'}")
    return 0 if (all_ok and not unmatched) else 1


def run_real(args) -> int:
    from benchmark_arithmetic import ArchimedesBackend
    backend = ArchimedesBackend(args.checkpoint, args.tokenizer, args.max_new_tokens)
    lobe = backend.answer   # (question:str) -> extracted answer text

    problems = gen_problems(args.per_family, args.seed)
    mono: dict[str, list[bool]] = {}
    orch: dict[str, list[bool]] = {}
    rows = []
    for i, p in enumerate(problems):
        expected = int(p["expected"])

        # --- monolithic: whole problem, one shot ---
        mono_text = backend.answer(p["prompt"])
        mono_hit = observed_number(mono_text) == expected
        mono.setdefault(p["family"], []).append(mono_hit)

        # --- orchestrated: decompose -> per-op lobe calls ---
        plan = decompose(p["prompt"])
        if plan is None:
            orch_answer, orch_steps = None, []
        else:
            res = execute(plan, lobe)
            orch_answer, orch_steps = res.answer, res.steps
        orch_hit = orch_answer == expected
        orch.setdefault(p["family"], []).append(orch_hit)

        rows.append({**p, "mono_text": mono_text, "mono_hit": mono_hit,
                     "orch_answer": orch_answer, "orch_hit": orch_hit,
                     "orch_steps": orch_steps})
        if (i + 1) % 25 == 0:
            m = sum(r["mono_hit"] for r in rows) / len(rows)
            o = sum(r["orch_hit"] for r in rows) / len(rows)
            print(f"[{i + 1}/{len(problems)}] mono {m:.1%}  orch {o:.1%}", flush=True)

    print(f"\nmodel: {backend.name}")
    print(f"{'family':<12} {'mono':>7} {'orch':>7} {'delta':>7}  n")
    tm = to = 0
    for fam in sorted(mono):
        mh, oh = mono[fam], orch[fam]
        tm += sum(mh); to += sum(oh)
        mr, orr = sum(mh) / len(mh), sum(oh) / len(oh)
        print(f"{fam:<12} {mr:>6.1%} {orr:>6.1%} {orr - mr:>+6.1%}  {len(mh)}")
    n = len(rows)
    print(f"{'OVERALL':<12} {tm / n:>6.1%} {to / n:>6.1%} {(to - tm) / n:>+6.1%}  {n}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as h:
            for row in rows:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"detailed results written to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=99991)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.self_test:
        return self_test(args.per_family, args.seed)
    if args.checkpoint is None or args.tokenizer is None:
        raise SystemExit("pass --self-test, or both --checkpoint and --tokenizer")
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
