from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from archimedes_model.arithmetic_router import route
from archimedes_model.model import ArchimedesMathModel, ModelConfig


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
DEFAULT_STOP_TEXT = "<|endofsolution|>"


def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.device("cpu"), torch.float32


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("_orig_mod.") for key in state):
        return {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    return state


def load_model(checkpoint_path: Path, config_path: Path | None, device: torch.device) -> ArchimedesMathModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if config_path is not None:
        cfg = ModelConfig.from_json(config_path)
    else:
        raw_cfg = checkpoint.get("config")
        if raw_cfg is None:
            raise ValueError("checkpoint has no embedded config; pass --config")
        cfg = ModelConfig(**raw_cfg)
    model = ArchimedesMathModel(cfg)
    model.load_state_dict(normalize_state_dict(checkpoint["model"]))
    return model.to(device).eval()


def trim_at_stop(text: str, stop_text: str) -> str:
    if not stop_text:
        return text
    variants = [
        stop_text,
        stop_text.replace("endofsolution", " end of solution "),
        "<| end of solution |>",
        "<| end of solution | >",
        "< | end of solution | >",
    ]
    cut = len(text)
    for variant in variants:
        pos = text.find(variant)
        if pos >= 0:
            cut = min(cut, pos)
    artifact_markers = ["Copyright", "theory ", "Require ", "/-", "(*", "import "]
    for marker in artifact_markers:
        pos = text.find(marker)
        if pos >= 0:
            cut = min(cut, pos)
    return text[:cut]


def clean_decoded_text(text: str) -> str:
    return (
        text.replace("Ġ", " ")
        .replace("Ċ", "\n")
        .replace("ĉ", "\t")
        .replace(" :", ":")
        .replace(" .", ".")
        .replace(" ,", ",")
    )


def build_prompt(user_prompt: str, style: str) -> str:
    if style == "plain":
        return user_prompt
    if style == "qa":
        return f"Question: {user_prompt}\nAnswer:"
    if style == "solution":
        return f"Problem: {user_prompt}\nSolution:"
    raise ValueError(f"unknown prompt style: {style}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from an Archimedes-Math checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_DATA_ROOT / "tokenizers" / "archimedes_math_bpe_32768_v2" / "tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-style", choices=("plain", "qa", "solution"), default="solution")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--raw-output", action="store_true")
    parser.add_argument("--stop-text", default=DEFAULT_STOP_TEXT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tool-mode", choices=("auto", "off", "only"), default="auto")
    args = parser.parse_args()

    routed = None if args.tool_mode == "off" else route(args.prompt)
    prompt = build_prompt(args.prompt, args.prompt_style)
    if routed is not None:
        print(json.dumps({
            "checkpoint": str(args.checkpoint),
            "device": "tool",
            "prompt": prompt,
            "output": routed.output,
            "tool": routed.kind,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.tool_mode == "only":
        raise ValueError("no deterministic arithmetic route matched this prompt")

    torch.manual_seed(args.seed)
    device, amp_dtype = device_and_dtype()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = load_model(args.checkpoint, args.config, device)

    input_ids = tokenizer.encode(prompt).ids
    if not input_ids:
        raise ValueError("prompt produced no tokens")
    if len(input_ids) > model.cfg.context_length:
        input_ids = input_ids[-model.cfg.context_length :]
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
        y = model.generate(
            x,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    text = tokenizer.decode(y[0].detach().cpu().tolist())
    text = trim_at_stop(text, args.stop_text)
    if not args.raw_output:
        text = clean_decoded_text(text)
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "prompt": prompt,
        "output": text,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
