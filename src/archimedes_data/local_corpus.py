from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sqlite3
import sys
import time
from pathlib import Path


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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents(domain)")
    return conn


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def doc_id_for_path(source_name: str, path: Path) -> str:
    return hashlib.sha256(f"{source_name}:{path}".encode("utf-8")).hexdigest()[:24]


def matches(path: Path, include: list[str], exclude: list[str]) -> bool:
    rel = path.as_posix()
    if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
        return False
    return any(fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(rel, pattern) for pattern in include)


def ingest_file(
    conn: sqlite3.Connection,
    data_root: Path,
    source_root: Path,
    path: Path,
    source_name: str,
    domain: str,
    license_name: str,
    source_url: str,
    force: bool,
) -> str:
    doc_id = doc_id_for_path(source_name, path.relative_to(source_root))
    if not force:
        row = conn.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row and row[0] == "ok":
            return "skip"

    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")
    cleaned = normalize_text(text)

    cleaned_dir = data_root / "cleaned" / source_name
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_dir / f"{doc_id}.txt"
    cleaned_path.write_text(cleaned + "\n", encoding="utf-8")

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
            source_name,
            source_url,
            domain,
            license_name,
            "local_text",
            str(path),
            str(cleaned_path),
            digest,
            len(raw_bytes),
            len(cleaned),
            token_estimate(cleaned),
            "ok",
            None,
            int(time.time()),
        ),
    )
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest cloned/local math corpora.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--domain", default="mathematics")
    parser.add_argument("--license", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    include = args.include or ["*.txt", "*.md", "*.tex", "*.lean", "*.v", "*.thy", "*.mm", "*.json"]
    exclude = args.exclude or [".git/*", "*/.git/*", "__pycache__/*", "*/__pycache__/*"]
    conn = init_db(args.data_root / "metadata" / "documents.sqlite")

    ok = skipped = errors = 0
    for path in sorted(p for p in args.root.rglob("*") if p.is_file()):
        rel = path.relative_to(args.root)
        if not matches(rel, include, exclude):
            continue
        try:
            result = ingest_file(
                conn,
                args.data_root,
                args.root,
                path,
                args.source_name,
                args.domain,
                args.license,
                args.source_url,
                args.force,
            )
            if result == "ok":
                ok += 1
            else:
                skipped += 1
        except OSError as exc:
            errors += 1
            print(f"err {path} :: {exc}", flush=True)
        if args.limit and ok >= args.limit:
            break

    conn.commit()
    print(f"local_ingest_done source={args.source_name} ok={ok} skipped={skipped} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

