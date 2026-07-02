#!/usr/bin/env python3
"""Benchmark arithmetic/procedural math accuracy over generated held-out problems.

Runs the same problem set through either an Archimedes checkpoint or a
HuggingFace causal LM, so size-matched models can be compared fairly:

  python3 scripts/benchmark_arithmetic.py --checkpoint <ckpt> --tokenizer <tok>
  python3 scripts/benchmark_arithmetic.py --hf-model gpt2

Each backend gets its native prompt format (Problem/Solution reasoning for
Archimedes, few-shot direct answers for baselines). Scoring is exact-match on
the normalized final answer.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path

EVAL_HOLDOUT = {
    "Compute 247 + 389.",
    "Solve for x: 7x + 5 = 47.",
    "Find the sum of the first 17 odd positive integers.",
    "Find 1 + 2 + ... + 25.",
}


def gen_problems(per_family: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    problems: list[dict] = []

    def add(family: str, prompt: str, expected: str) -> None:
        if prompt not in EVAL_HOLDOUT:
            problems.append({"family": family, "prompt": prompt, "expected": expected})

    for _ in range(per_family):
        a, b = rng.randint(10, 9999), rng.randint(10, 9999)
        add("add", f"Compute {a} + {b}.", str(a + b))
        a, b = rng.randint(10, 9999), rng.randint(10, 9999)
        add("sub", f"Compute {a} - {b}.", str(a - b))
        a, b = rng.randint(12, 99), rng.randint(2, 99)
        add("mul", f"Compute {a} * {b}.", str(a * b))
        b, q = rng.randint(2, 9), rng.randint(12, 999)
        add("div", f"Compute {b * q} / {b}.", str(q))
        a, x, c0 = rng.randint(2, 9), rng.randint(2, 12), rng.randint(1, 20)
        add("linear", f"Solve for x: {a}x + {c0} = {a * x + c0}.", f"x = {x}")
        n = rng.randint(2, 99)
        add("odd_sum", f"Find the sum of the first {n} odd positive integers.", str(n * n))
        n = rng.randint(2, 99)
        add("triangular", f"Find 1 + 2 + ... + {n}.", str(n * (n + 1) // 2))
        p = rng.choice([10, 20, 25, 50])
        n = 20 * rng.randint(1, 200)
        add("percent", f"What is {p}% of {n}?", str(p * n // 100))
        a, b = rng.randint(12, 144), rng.randint(12, 144)
        add("gcd", f"Compute the greatest common divisor of {a} and {b}.", str(gcd(a, b)))
        a, b = rng.randint(1, 11), rng.randint(2, 12)
        c, d = rng.randint(1, 11), rng.randint(2, 12)
        val = Fraction(a, b) + Fraction(c, d)
        expected = str(val.numerator) if val.denominator == 1 else f"{val.numerator}/{val.denominator}"
        add("fraction", f"Compute {a}/{b} + {c}/{d}.", expected)

    # worded operator families use a separate rng stream so the original
    # 249 seeded problems above stay byte-identical across script versions
    worded = random.Random(seed + 1)
    for _ in range(per_family):
        a, b = worded.randint(10, 9999), worded.randint(10, 9999)
        add("add_worded", f"Compute {a} plus {b}.", str(a + b))
        a, b = worded.randint(10, 9999), worded.randint(10, 9999)
        add("sub_worded", f"Compute {a} minus {b}.", str(a - b))
    return problems


def norm(text: str) -> str:
    # drop whitespace and thousands separators so "13,118" == "13118"
    return re.sub(r"[\s,]+", "", text).strip(".").lower()


def as_value(text: str) -> Fraction | None:
    # parse "x = 6", "7/4", "1.75", "\( -3 \)", "Answer: 55" into an exact value
    cleaned = clean_answer(text)
    cleaned = re.sub(r"[*]", "", cleaned)
    if "=" in cleaned:
        cleaned = cleaned.split("=")[-1]
    cleaned = norm(cleaned)
    try:
        return Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        return None


def is_hit(expected: str, observed: str | None) -> bool:
    if observed is None:
        return False
    exp, obs = norm(expected), norm(observed)
    if obs == exp:
        return True
    # generous to baselines: accept any mathematically equal form —
    # a bare "6" for "x = 6", "50/24" or "1.75" for a fraction, LaTeX noise
    exp_val, obs_val = as_value(expected), as_value(observed)
    return exp_val is not None and obs_val is not None and exp_val == obs_val


def extract_final(output: str) -> str | None:
    matches = re.findall(r"final\s+answer\s*:\s*([^\n]+)", output, flags=re.IGNORECASE)
    if matches:
        # the end-of-solution marker decodes as visible text on the same
        # line; cut the answer at its first character
        return matches[-1].split("<")[0].strip()
    return None


class ArchimedesBackend:
    name_prefix = "archimedes"

    def __init__(self, checkpoint: Path, tokenizer: Path, max_new_tokens: int) -> None:
        import torch
        from tokenizers import Tokenizer

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from archimedes_model.generate import device_and_dtype, load_model

        self.torch = torch
        self.device, self.amp_dtype = device_and_dtype()
        self.tokenizer = Tokenizer.from_file(str(tokenizer))
        self.model = load_model(checkpoint, None, self.device)
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        self.max_new_tokens = max_new_tokens
        self.name = f"archimedes:{checkpoint.parent.name}/{checkpoint.name}"

    def answer(self, prompt: str) -> str | None:
        ids = self.tokenizer.encode(f"Problem: {prompt}\nSolution:").ids
        if self.eos_id is not None and ids and ids[-1] == self.eos_id:
            ids = ids[:-1]
        x = self.torch.tensor([ids], dtype=self.torch.long, device=self.device)
        with self.torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            y = self.model.generate(x, max_new_tokens=self.max_new_tokens, temperature=0.0, top_k=1, eos_token_id=self.eos_id)
        text = self.tokenizer.decode(y[0].detach().cpu().tolist())
        text = text.replace("Ġ", " ").replace("Ċ", "\n")
        return extract_final(text)


FEW_SHOT_PAIRS = (
    ("Compute 46 + 58.", "104"),
    ("Solve for x: 3x + 4 = 19.", "x = 5"),
    ("What is 25% of 80?", "20"),
    ("Compute 1/2 + 1/3.", "5/6"),
)
FEW_SHOT = "".join(f"Problem: {p}\nAnswer: {a}\n\n" for p, a in FEW_SHOT_PAIRS)


def clean_answer(answer: str) -> str:
    # peel repeated "Answer:" prefixes ("Answer: Answer: 55") and LaTeX
    # wrappers ("\( x = 8 \)", "$\frac{7}{4}$") off an extracted answer
    answer = re.sub(r"^\s*(?:answer\s*(?:is|:)[\s*]*)+", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", r"\1/\2", answer)
    answer = re.sub(r"\\d?frac(-?\d)(-?\d)", r"\1/\2", answer)
    answer = re.sub(r"[\\()\[\]$]", "", answer)
    return answer.strip().strip("*").strip().rstrip(".")


def extract_boxed(completion: str) -> str | None:
    # regex can't handle nested braces in \boxed{\frac{7}{4}}; scan instead
    idx = completion.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 0
    for i in range(start, len(completion)):
        if completion[i] == "{":
            depth += 1
        elif completion[i] == "}":
            if depth == 0:
                return completion[start:i]
            depth -= 1
    return completion[start:]


def extract_chat_answer(completion: str) -> str | None:
    # generous extraction for chain-of-thought outputs: prefer \boxed{},
    # then an explicit "Answer:" line, then the last number/fraction anywhere
    boxed = extract_boxed(completion)
    if boxed is not None:
        return clean_answer(boxed)
    marked = re.findall(r"answer\s*(?:is|:)\s*([^\n]+)", completion, flags=re.IGNORECASE)
    if marked:
        return clean_answer(marked[-1])
    numbers = re.findall(r"-?\d[\d,]*(?:/\d+)?(?:\.\d+)?", completion)
    if numbers:
        return numbers[-1]
    return completion.strip().split("\n")[-1].strip() or None


class HFBackend:
    def __init__(self, model_name: str, max_new_tokens: int = 24, chat: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.chat = chat
        # chat models reason step by step before answering; give them room
        # (verbose math models were getting truncated before the answer at 512)
        self.max_new_tokens = max(max_new_tokens, 768) if chat else max_new_tokens
        self.last_raw: str | None = None
        self.name = f"hf:{model_name}" + (":chat" if chat else "")

    def answer(self, prompt: str) -> str | None:
        if self.chat:
            messages = []
            for shot_prompt, shot_answer in FEW_SHOT_PAIRS:
                messages.append({"role": "user", "content": shot_prompt})
                messages.append({"role": "assistant", "content": f"Answer: {shot_answer}"})
            messages.append({
                "role": "user",
                "content": f"{prompt}\nEnd your reply with the final result as 'Answer: <result>'.",
            })
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = FEW_SHOT + f"Problem: {prompt}\nAnswer:"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_raw = completion
        if self.chat:
            return extract_chat_answer(completion)
        return completion.split("\n")[0].strip() or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-chat", action="store_true",
                        help="use the model's chat template (for instruct models)")
    parser.add_argument("--per-family", type=int, default=25)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print generated problems and exit")
    args = parser.parse_args()

    problems = gen_problems(args.per_family, args.seed)
    if args.dry_run:
        for p in problems[:12]:
            print(json.dumps(p, ensure_ascii=False))
        print(f"... {len(problems)} problems total")
        return 0

    if args.checkpoint is not None:
        if args.tokenizer is None:
            raise SystemExit("--tokenizer is required with --checkpoint")
        backend = ArchimedesBackend(args.checkpoint, args.tokenizer, args.max_new_tokens)
    elif args.hf_model is not None:
        backend = HFBackend(args.hf_model, chat=args.hf_chat)
    else:
        raise SystemExit("pass either --checkpoint/--tokenizer or --hf-model")

    results = []
    per_family: dict[str, list[bool]] = {}
    for i, p in enumerate(problems):
        observed = backend.answer(p["prompt"])
        hit = is_hit(p["expected"], observed)
        per_family.setdefault(p["family"], []).append(hit)
        # keep the raw completion so scoring disputes can be audited later
        results.append({**p, "observed": observed, "hit": hit,
                        "raw": getattr(backend, "last_raw", None)})
        if (i + 1) % 25 == 0:
            done = sum(1 for r in results if r["hit"])
            print(f"[{i + 1}/{len(problems)}] running accuracy {done / (i + 1):.1%}", flush=True)

    print(f"\nmodel: {backend.name}")
    print(f"{'family':<12} {'accuracy':>9}  n")
    total_hits = 0
    for family in sorted(per_family):
        hits = per_family[family]
        total_hits += sum(hits)
        print(f"{family:<12} {sum(hits) / len(hits):>8.1%}  {len(hits)}")
    print(f"{'OVERALL':<12} {total_hits / len(results):>8.1%}  {len(results)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for row in results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"detailed results written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
