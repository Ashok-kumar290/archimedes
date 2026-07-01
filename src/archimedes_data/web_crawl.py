from __future__ import annotations

import argparse
import hashlib
import html
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
USER_AGENT = "ArchimedesDataEngine/0.1 (+local research dataset builder)"


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


def doc_id_for_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    parsed = parsed._replace(fragment="")
    return urllib.parse.urlunsplit(parsed)


def download(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type and "text" not in content_type:
            raise ValueError(f"unsupported content-type: {content_type}")
        return response.read()


def decode(payload: bytes) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return payload.decode(enc)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|h[1-6]|li|tr)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_links(base_url: str, text: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r'(?is)<a\s+[^>]*href=["\']([^"\']+)["\']', text):
        href = html.unescape(match.group(1)).strip()
        if not href or href.startswith(("mailto:", "javascript:", "#")):
            continue
        links.append(normalize_url(urllib.parse.urljoin(base_url, href)))
    return links


def allowed(url: str, allowed_domains: set[str], allowed_prefixes: list[str]) -> bool:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.netloc not in allowed_domains:
        return False
    if allowed_prefixes and not any(url.startswith(prefix) for prefix in allowed_prefixes):
        return False
    return True


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def save_page(
    conn: sqlite3.Connection,
    data_root: Path,
    source_name: str,
    domain: str,
    license_name: str,
    url: str,
    payload: bytes,
    cleaned: str,
) -> None:
    doc_id = doc_id_for_url(url)
    raw_dir = data_root / "raw" / source_name
    cleaned_dir = data_root / "cleaned" / source_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{doc_id}.html"
    cleaned_path = cleaned_dir / f"{doc_id}.txt"
    raw_path.write_bytes(payload)
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
            url,
            domain,
            license_name,
            "html",
            str(raw_path),
            str(cleaned_path),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            len(cleaned),
            token_estimate(cleaned),
            "ok",
            None,
            int(time.time()),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crawl public math HTML pages.")
    parser.add_argument("--seed", action="append", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--domain", default="mathematics_textbook_notes")
    parser.add_argument("--license", required=True)
    parser.add_argument("--allowed-domain", action="append", required=True)
    parser.add_argument("--allowed-prefix", action="append", default=[])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)

    conn = init_db(args.data_root / "metadata" / "documents.sqlite")
    queue = deque(normalize_url(seed) for seed in args.seed)
    seen: set[str] = set()
    allowed_domains = set(args.allowed_domain)
    ok = errors = 0

    while queue and ok < args.max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)
        if not allowed(url, allowed_domains, args.allowed_prefix):
            continue
        try:
            payload = download(url, args.timeout)
            raw_text = decode(payload)
            cleaned = html_to_text(raw_text)
            if len(cleaned) >= 500:
                save_page(conn, args.data_root, args.source_name, args.domain, args.license, url, payload, cleaned)
                ok += 1
                print(f"ok   {ok:>4} {token_estimate(cleaned):>7} tok_est {url}", flush=True)
            for link in extract_links(url, raw_text):
                if link not in seen and allowed(link, allowed_domains, args.allowed_prefix):
                    queue.append(link)
        except Exception as exc:  # noqa: BLE001 - crawler records errors and keeps going.
            errors += 1
            print(f"err       {url} :: {exc}", flush=True)
        conn.commit()
        if args.sleep:
            time.sleep(args.sleep)

    print(f"crawl_done source={args.source_name} ok={ok} errors={errors} queued={len(queue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

