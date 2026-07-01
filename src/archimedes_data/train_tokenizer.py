from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|doc|>", "<|endofdoc|>"]


def iter_text(manifest: Path):
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            yield Path(doc["cleaned_path"]).read_text(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train Archimedes math BPE tokenizer.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args(argv)

    manifest = args.manifest or (args.data_root / "metadata" / "math_train_manifest_v1.jsonl")
    out_dir = args.data_root / "tokenizers" / f"archimedes_math_bpe_{args.vocab_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_text(manifest), trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )
    path = out_dir / "tokenizer.json"
    tokenizer.save(str(path))
    print(f"tokenizer={path}")
    print(f"vocab_size={tokenizer.get_vocab_size()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

