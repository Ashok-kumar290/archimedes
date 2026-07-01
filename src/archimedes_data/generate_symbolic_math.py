from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import sympy as sp


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            domain TEXT NOT NULL,
            license TEXT NOT NULL,
            source_type TEXT NOT NULL,
            raw_path TEXT NOT NULL,
            cleaned_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            byte_count INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            token_estimate INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            retrieved_at INTEGER NOT NULL
        )
        """
    )
    return conn


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def random_poly(rng: random.Random, x: sp.Symbol, degree: int = 4) -> sp.Expr:
    expr = 0
    for power in range(degree + 1):
        coeff = rng.randint(-9, 9)
        if coeff:
            expr += coeff * x**power
    return expr or x


def derivative_example(rng: random.Random, x: sp.Symbol) -> str:
    expr = random_poly(rng, x, rng.randint(2, 5)) * sp.sin(rng.randint(1, 5) * x)
    result = sp.diff(expr, x)
    return f"Problem: Differentiate f(x) = {sp.sstr(expr)}.\nSolution: Use product, chain, and power rules.\nf'(x) = {sp.sstr(result)}.\n"


def integral_example(rng: random.Random, x: sp.Symbol) -> str:
    inner = random_poly(rng, x, rng.randint(1, 3))
    expr = sp.diff(inner, x) * sp.exp(inner)
    result = sp.integrate(expr, x)
    return f"Problem: Compute the indefinite integral ∫ {sp.sstr(expr)} dx.\nSolution: Let u = {sp.sstr(inner)}. Then du = ({sp.sstr(sp.diff(inner, x))}) dx.\nIntegral = {sp.sstr(result)} + C.\n"


def equation_example(rng: random.Random, x: sp.Symbol) -> str:
    roots = [rng.randint(-9, 9) for _ in range(rng.randint(2, 4))]
    expr = sp.expand(sp.prod(x - r for r in roots))
    sol = sorted(set(roots))
    return f"Problem: Solve {sp.sstr(expr)} = 0 over the real numbers.\nSolution: Factor the polynomial: {sp.sstr(sp.factor(expr))} = 0.\nTherefore x ∈ {sol}.\n"


def matrix_example(rng: random.Random) -> str:
    a = sp.Matrix([[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)])
    b = sp.Matrix([[rng.randint(-5, 5) for _ in range(2)] for _ in range(2)])
    product = a * b
    det = a.det()
    return f"Problem: For A = {a.tolist()} and B = {b.tolist()}, compute AB and det(A).\nSolution: Matrix multiplication gives AB = {product.tolist()}.\nThe determinant is det(A) = {det}.\n"


def limit_example(rng: random.Random, x: sp.Symbol) -> str:
    a = rng.randint(1, 6)
    expr = sp.sin(a * x) / x
    result = sp.limit(expr, x, 0)
    return f"Problem: Evaluate lim_(x→0) {sp.sstr(expr)}.\nSolution: Since sin({a}x)/x = {a} * sin({a}x)/({a}x), the limit is {result}.\n"


def generate_chunk(rng: random.Random, examples: int, chunk_id: int) -> str:
    x = sp.Symbol("x")
    funcs = [derivative_example, integral_example, equation_example, matrix_example, limit_example]
    parts = [f"# Synthetic Symbolic Math Chunk {chunk_id}\n"]
    for i in range(examples):
        fn = rng.choice(funcs)
        if fn is matrix_example:
            text = fn(rng)
        else:
            text = fn(rng, x)
        parts.append(f"Example {i + 1}\n{text}")
    return "\n".join(parts)


def write_doc(conn: sqlite3.Connection, data_root: Path, chunk_id: int, text: str) -> None:
    source = "synthetic_symbolic_math_v1"
    raw_dir = data_root / "raw" / source
    cleaned_dir = data_root / "cleaned" / source
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    doc_id = hashlib.sha256(f"{source}:{chunk_id}".encode("utf-8")).hexdigest()[:24]
    raw_path = raw_dir / f"{doc_id}.json"
    cleaned_path = cleaned_dir / f"{doc_id}.txt"
    payload = json.dumps({"chunk_id": chunk_id, "text": text}, ensure_ascii=False, indent=2).encode("utf-8")
    raw_path.write_bytes(payload)
    cleaned_path.write_text(text + "\n", encoding="utf-8")
    conn.execute(
        """
        INSERT OR REPLACE INTO documents (
            doc_id, source_name, source_url, domain, license, source_type,
            raw_path, cleaned_path, sha256, byte_count, char_count,
            token_estimate, status, error, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            source,
            "generated://sympy/synthetic_symbolic_math_v1",
            "synthetic_symbolic_mathematics",
            "Generated locally by Archimedes pipeline",
            "generated_text",
            str(raw_path),
            str(cleaned_path),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            len(text),
            token_estimate(text),
            "ok",
            None,
            int(time.time()),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic symbolic math examples with SymPy.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--chunks", type=int, default=200)
    parser.add_argument("--examples-per-chunk", type=int, default=500)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    conn = init_db(args.data_root / "metadata" / "documents.sqlite")
    total_tokens = 0
    for chunk_id in range(args.chunks):
        text = generate_chunk(rng, args.examples_per_chunk, chunk_id)
        write_doc(conn, args.data_root, chunk_id, text)
        total_tokens += token_estimate(text)
        if (chunk_id + 1) % 25 == 0:
            conn.commit()
            print(f"chunks={chunk_id + 1} token_estimate={total_tokens}", flush=True)
    conn.commit()
    print(f"synthetic_done chunks={args.chunks} token_estimate={total_tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

