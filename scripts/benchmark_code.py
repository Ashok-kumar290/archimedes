#!/usr/bin/env python3
"""Execution-based held-out code benchmark (expert #4).

The model generates a Python function from the task spec; we EXECUTE it against
fresh random inputs (seed 99991, disjoint from training) whose expected outputs
come from the task's independent oracle. A task counts as solved (pass@1) only
if the generated function is correct on ALL held-out inputs — a memorized output
can't pass. Greedy decode, so one function per task; score = tasks solved / 15.

Safety: generated code runs in a subprocess with a timeout, on our own
arithmetic/string tasks (no file/network use in any reference solution). Treat
as a controlled harness, not a general sandbox.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_code_curriculum import TASKS


def raw_generate(backend, prompt: str) -> str:
    """Greedy generation returning the RAW decoded text (code needs the whole
    body, not the arithmetic 'Final answer' extraction)."""
    torch = backend.torch
    ids = backend.tokenizer.encode(f"Problem: {prompt}\nSolution:").ids
    if backend.eos_id is not None and ids and ids[-1] == backend.eos_id:
        ids = ids[:-1]
    x = torch.tensor([ids], dtype=torch.long, device=backend.device)
    with torch.autocast(device_type=backend.device.type, dtype=backend.amp_dtype,
                        enabled=backend.device.type == "cuda"):
        y = backend.model.generate(x, max_new_tokens=backend.max_new_tokens,
                                   temperature=0.0, top_k=1, eos_token_id=backend.eos_id)
    text = backend.tokenizer.decode(y[0].detach().cpu().tolist())
    return text.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")


def extract_code(text: str, call: str) -> str | None:
    """Pull the function definition out of a completion (strip markdown fences,
    take from `def <call>` to the next top-level 'Problem:'/prose boundary)."""
    text = text.replace("```python", "```").replace("```", "\n")
    idx = text.find(f"def {call}")
    if idx == -1:
        return None
    body = text[idx:]
    # cut at the start of a new problem or obvious prose continuation
    for marker in ["\nProblem:", "\nWrite a Python", "\nQuestion:", "\n#"]:
        p = body.find(marker)
        if p != -1:
            body = body[:p]
    return body.rstrip()


HARNESS = """{code}

import json, sys
_inputs = json.loads(sys.argv[1])
_out = []
for _a in _inputs:
    try:
        _out.append(repr({call}(*_a)))
    except Exception as _e:
        _out.append("ERR:" + type(_e).__name__)
print(json.dumps(_out))
"""


def run_solution(code: str, call: str, inputs: list) -> list[str] | None:
    src = HARNESS.format(code=code, call=call)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        res = subprocess.run([sys.executable, path, json.dumps(inputs)],
                             capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            return None
        return json.loads(res.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-chat", action="store_true")
    parser.add_argument("--trials", type=int, default=20, help="held-out inputs per task")
    parser.add_argument("--seed", type=int, default=99991)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.hf_model is not None:
        from benchmark_arithmetic import HFBackend
        backend = HFBackend(args.hf_model, chat=args.hf_chat)
        def gen(prompt): return getattr(backend, "last_raw", None) or backend.answer(prompt)
    else:
        if args.checkpoint is None or args.tokenizer is None:
            raise SystemExit("pass either --hf-model, or both --checkpoint and --tokenizer")
        from benchmark_arithmetic import ArchimedesBackend
        backend = ArchimedesBackend(args.checkpoint, args.tokenizer, args.max_new_tokens)
        def gen(prompt):
            return raw_generate(backend, prompt)

    rng = random.Random(args.seed)
    results = []
    solved = 0
    print(f"{'task':16} {'pass@1':>7} {'inputs_ok':>10}")
    for task in TASKS:
        inputs = [list(task.gen_input(rng)) for _ in range(args.trials)]
        expected = [repr(task.oracle(*a)) for a in inputs]
        raw = gen(task.prompt)
        code = extract_code(raw, task.call)
        if code is None:
            passed = [False] * len(inputs)
        else:
            got = run_solution(code, task.call, inputs)
            passed = [False] * len(inputs) if got is None else [
                g == e for g, e in zip(got, expected)]
        ok = sum(passed)
        task_solved = ok == len(inputs)
        solved += task_solved
        print(f"{task.name:16} {('YES' if task_solved else 'no'):>7} {ok:>7}/{len(inputs)}")
        results.append({"task": task.name, "solved": task_solved,
                        "inputs_passed": ok, "n": len(inputs),
                        "code": code, "raw": raw[:500]})

    print(f"\nmodel: {backend.name}")
    print(f"SOLVED (pass@1, execution-verified): {solved}/{len(TASKS)}  ({solved/len(TASKS):.1%})")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as h:
            for row in results:
                h.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"detailed results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
