from __future__ import annotations

import argparse
import hashlib
import html
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
USER_AGENT = "ArchimedesDataEngine/0.1 (+local research dataset builder)"


@dataclass(frozen=True)
class Source:
    name: str
    domain: str
    license: str
    source_type: str
    urls: tuple[str, ...]


def parse_seed_sources(path: Path) -> list[Source]:
    """Parse the small YAML subset used by configs/seed_sources.yaml."""
    sources: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    in_urls = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()

        if stripped == "sources:":
            continue
        if stripped.startswith("- name:"):
            if current:
                sources.append(current)
            current = {"name": stripped.split(":", 1)[1].strip(), "urls": []}
            in_urls = False
            continue
        if current is None:
            raise ValueError(f"Field before source entry in {path}: {raw_line}")
        if stripped == "urls:":
            in_urls = True
            continue
        if in_urls and stripped.startswith("- "):
            current_urls = current.setdefault("urls", [])
            assert isinstance(current_urls, list)
            current_urls.append(stripped[2:].strip())
            continue
        if ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = value.strip()
            in_urls = False
            continue
        raise ValueError(f"Unsupported config line in {path}: {raw_line}")

    if current:
        sources.append(current)

    parsed: list[Source] = []
    for source in sources:
        parsed.append(
            Source(
                name=required_str(source, "name"),
                domain=required_str(source, "domain"),
                license=required_str(source, "license"),
                source_type=required_str(source, "source_type"),
                urls=tuple(required_list(source, "urls")),
            )
        )
    return parsed


def required_str(source: dict[str, object], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing required source field: {key}")
    return value


def required_list(source: dict[str, object], key: str) -> list[str]:
    value = source.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing required source list: {key}")
    return [str(item) for item in value]


def ensure_layout(data_root: Path) -> None:
    for rel in ("raw", "parsed", "cleaned", "metadata", "shards", "indexes", "logs"):
        (data_root / rel).mkdir(parents=True, exist_ok=True)


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


def url_to_doc_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def extension_for(source_type: str, url: str) -> str:
    path = urllib.parse.urlparse(url).path.lower()
    suffix = Path(path).suffix
    if suffix in {".txt", ".html", ".htm", ".xml", ".pdf", ".json", ".md"}:
        return suffix
    return {
        "text": ".txt",
        "html": ".html",
        "xml": ".xml",
        "pdf": ".pdf",
    }.get(source_type, ".bin")


def download(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_with_retries(url: str, timeout: int, retries: int, retry_sleep: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return download(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt >= retries:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt >= retries:
                raise
        time.sleep(retry_sleep * (attempt + 1))
    assert last_error is not None
    raise last_error


def decode_bytes(payload: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def extract_text(payload: bytes, source_type: str, raw_path: Path) -> str:
    if source_type == "pdf" or raw_path.suffix.lower() == ".pdf":
        return extract_pdf_text(raw_path)
    text = decode_bytes(payload)
    if source_type == "html" or raw_path.suffix.lower() in {".html", ".htm"}:
        return html_to_text(text)
    if source_type == "xml" or raw_path.suffix.lower() == ".xml":
        return xml_to_text(text)
    return normalize_text(text)


def extract_pdf_text(raw_path: Path) -> str:
    # Keep v0 dependency-light. If pdftotext is unavailable, retain metadata and
    # make the parse failure explicit.
    import subprocess

    result = subprocess.run(
        ["pdftotext", "-layout", str(raw_path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pdftotext failed")
    return normalize_text(result.stdout)


def html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|h[1-6]|li|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return normalize_text(html.unescape(text))


def xml_to_text(text: str) -> str:
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return normalize_text(html.unescape(text))


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_estimate(text: str) -> int:
    # Approximation for early data accounting before tokenizer training.
    return max(1, len(text) // 4) if text else 0


def write_document(
    conn: sqlite3.Connection,
    data_root: Path,
    source: Source,
    url: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
    force: bool,
) -> str:
    doc_id = url_to_doc_id(url)
    if not force:
        exists = conn.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if exists:
            return f"skip {doc_id} {url}"

    raw_dir = data_root / "raw" / source.name
    cleaned_dir = data_root / "cleaned" / source.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    ext = extension_for(source.source_type, url)
    raw_path = raw_dir / f"{doc_id}{ext}"
    cleaned_path = cleaned_dir / f"{doc_id}.txt"
    retrieved_at = int(time.time())

    try:
        payload = download_with_retries(url, timeout=timeout, retries=retries, retry_sleep=retry_sleep)
        raw_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        cleaned = extract_text(payload, source.source_type, raw_path)
        cleaned_path.write_text(cleaned + "\n", encoding="utf-8")
        status = "ok"
        error = None
    except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
        payload = b""
        digest = ""
        cleaned = ""
        status = "error"
        error = str(exc)

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
            source.name,
            url,
            source.domain,
            source.license,
            source.source_type,
            str(raw_path),
            str(cleaned_path),
            digest,
            len(payload),
            len(cleaned),
            token_estimate(cleaned),
            status,
            error,
            retrieved_at,
        ),
    )
    conn.commit()

    if status == "ok":
        return f"ok   {doc_id} {len(payload):>9} bytes {token_estimate(cleaned):>8} tok_est {url}"
    return f"err  {doc_id} {url} :: {error}"


def iter_jobs(sources: Iterable[Source]) -> Iterable[tuple[Source, str]]:
    for source in sources:
        for url in source.urls:
            yield source, url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrieve Archimedes seed documents.")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    ensure_layout(args.data_root)
    sources = parse_seed_sources(args.sources)
    conn = init_db(args.data_root / "metadata" / "documents.sqlite")

    total = 0
    for source, url in iter_jobs(sources):
        print(
            write_document(
                conn,
                args.data_root,
                source,
                url,
                args.timeout,
                args.retries,
                args.retry_sleep,
                args.force,
            ),
            flush=True,
        )
        total += 1
        if args.sleep:
            time.sleep(args.sleep)

    rows = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(token_estimate), 0) FROM documents WHERE status = 'ok'"
    ).fetchone()
    print(f"retrieval_done jobs={total} ok_docs={rows[0]} token_estimate={rows[1]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

