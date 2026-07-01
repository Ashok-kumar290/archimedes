#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_PROMPTS = [
    "Compute 247 + 389.",
    "Solve for x: 7x + 5 = 47.",
    "Find the sum of the first 17 odd positive integers.",
    "Find 1 + 2 + ... + 25.",
    "Prove that if n is even, then n^2 is even.",
    "Prove that 1 + 3 + 5 + ... + (2n - 1) = n^2.",
]

EXPECTED_FINALS = {
    "Compute 247 + 389.": "636",
    "Solve for x: 7x + 5 = 47.": "x=6",
    "Find the sum of the first 17 odd positive integers.": "289",
    "Find 1 + 2 + ... + 25.": "325",
}

ARTIFACTS = ["Copyright", "theory", "Require", "import", "/-", "(*"]


def compact_math(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def numeric_hit(expected: str, observed: str) -> bool:
    if not re.fullmatch(r"-?\d+", expected):
        return False
    return expected in re.findall(r"-?\d+", observed)


def final_answer(output: str) -> str | None:
    matches = re.findall(r"final\s+answer\s*:\s*([^\n.]+(?:\.[^\n]*)?)", output, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"therefore\s+the\s+answer\s+is\s+([^\n.]+)", output, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def score(prompt: str, output: str) -> dict[str, object]:
    expected = EXPECTED_FINALS.get(prompt)
    observed = final_answer(output)
    if expected is None:
        expected_hit = None
    elif observed is None:
        expected_hit = False
    elif re.fullmatch(r"-?\d+", expected):
        expected_hit = numeric_hit(expected, observed)
    else:
        expected_hit = compact_math(expected) in compact_math(observed)
    artifacts = [marker for marker in ARTIFACTS if marker in output]
    return {"expected": expected, "observed_final": observed, "expected_hit": expected_hit, "artifacts": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed prompt suite against an Archimedes checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    args = parser.parse_args()

    rows = []
    for prompt in DEFAULT_PROMPTS:
        cmd = [
            sys.executable,
            "scripts/generate_math.py",
            "--checkpoint", str(args.checkpoint),
            "--tokenizer", str(args.tokenizer),
            "--prompt", prompt,
            "--prompt-style", "solution",
            "--max-new-tokens", str(args.max_new_tokens),
            "--temperature", str(args.temperature),
            "--top-k", str(args.top_k),
            "--top-p", str(args.top_p),
            "--repetition-penalty", str(args.repetition_penalty),
        ]
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        row = json.loads(result.stdout)
        row["score"] = score(prompt, row["output"])
        rows.append(row)
        print("=" * 80)
        print(row["prompt"])
        print(row["output"])
        print("SCORE", json.dumps(row["score"], ensure_ascii=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
