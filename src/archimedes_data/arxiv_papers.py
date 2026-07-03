from __future__ import annotations

import argparse
import hashlib
import html
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/seyominaoto/archimedes-data")
USER_AGENT = "ArchimedesDataEngine/0.1 (+local research dataset builder)"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


@dataclass(frozen=True)
class Paper:
    arxiv_id: str
    title: str
    pdf_url: str
    categories: tuple[str, ...]
    primary_category: str


def normalize_arxiv_id(entry_id: str) -> str:
    return entry_id.rsplit("/", 1)[-1]


def doc_id_for_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def token_estimate(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def parse_papers(xml_path: Path) -> list[Paper]:
    root = ET.parse(xml_path).getroot()
    papers: list[Paper] = []
    for entry in root.findall(f"{ATOM}entry"):
        entry_id = entry.findtext(f"{ATOM}id", default="")
        title = clean_text(html.unescape(entry.findtext(f"{ATOM}title", default="")))
        pdf_url = ""
        for link in entry.findall(f"{ATOM}link"):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not entry_id or not pdf_url:
            continue
        categories = tuple(
            cat.attrib.get("term", "")
            for cat in entry.findall(f"{ATOM}category")
            if cat.attrib.get("term")
        )
        primary = ""
        primary_node = entry.find(f"{ARXIV}primary_category")
        if primary_node is not None:
            primary = primary_node.attrib.get("term", "")
        papers.append(
            Paper(
                arxiv_id=normalize_arxiv_id(entry_id),
                title=title,
                pdf_url=pdf_url,
                categories=categories,
                primary_category=primary,
            )
        )
    return papers


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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS arxiv_papers (
            arxiv_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            title TEXT NOT NULL,
            pdf_url TEXT NOT NULL,
            categories TEXT NOT NULL,
            primary_category TEXT NOT NULL
        )
        """
    )
    return conn


def download_pdf(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_pdf_with_retries(url: str, timeout: int, retries: int, retry_sleep: float) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return download_pdf(url, timeout)
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


def extract_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "pdftotext failed")
    return clean_text(result.stdout)


def extract_images(pdf_path: Path, image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    prefix = image_dir / pdf_path.stem
    subprocess.run(["pdfimages", "-png", str(pdf_path), str(prefix)], check=False)


def existing_ok(conn: sqlite3.Connection, doc_id: str) -> bool:
    row = conn.execute("SELECT status FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    return bool(row and row[0] == "ok")


def ingest_paper(
    conn: sqlite3.Connection,
    data_root: Path,
    paper: Paper,
    timeout: int,
    retries: int,
    retry_sleep: float,
    extract_image_files: bool,
    force: bool,
    source_name: str = "arxiv_math_papers",
    domain: str = "mathematics_research_full_paper",
) -> str:
    doc_id = doc_id_for_url(paper.pdf_url)
    if not force and existing_ok(conn, doc_id):
        return f"skip {paper.arxiv_id} {paper.pdf_url}"

    raw_dir = data_root / "raw" / source_name
    cleaned_dir = data_root / "cleaned" / source_name
    image_dir = data_root / "parsed" / f"{source_name}_images" / paper.arxiv_id.replace("/", "_")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = raw_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    cleaned_path = cleaned_dir / f"{paper.arxiv_id.replace('/', '_')}.txt"
    retrieved_at = int(time.time())
    status = "ok"
    error = None
    payload = b""
    text = ""
    digest = ""

    try:
        payload = download_pdf_with_retries(paper.pdf_url, timeout, retries, retry_sleep)
        pdf_path.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        text = extract_pdf_text(pdf_path)
        cleaned_path.write_text(text + "\n", encoding="utf-8")
        if extract_image_files:
            extract_images(pdf_path, image_dir)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
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
            source_name,
            paper.pdf_url,
            domain,
            "arXiv paper license varies by record; verify before redistribution/training release",
            "pdf",
            str(pdf_path),
            str(cleaned_path),
            digest,
            len(payload),
            len(text),
            token_estimate(text),
            status,
            error,
            retrieved_at,
        ),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO arxiv_papers (
            arxiv_id, doc_id, title, pdf_url, categories, primary_category
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            paper.arxiv_id,
            doc_id,
            paper.title,
            paper.pdf_url,
            ",".join(paper.categories),
            paper.primary_category,
        ),
    )
    conn.commit()

    if status == "ok":
        return f"ok   {paper.arxiv_id:<16} {len(payload):>9} bytes {token_estimate(text):>8} tok_est {paper.title[:70]}"
    return f"err  {paper.arxiv_id:<16} {paper.pdf_url} :: {error}"


def collect_papers(data_root: Path, metadata_sources: list[str]) -> list[Paper]:
    seen: set[str] = set()
    papers: list[Paper] = []
    for source in metadata_sources:
        metadata_dir = data_root / "raw" / source
        for xml_path in sorted(metadata_dir.glob("*.xml")):
            for paper in parse_papers(xml_path):
                if paper.arxiv_id in seen:
                    continue
                seen.add(paper.arxiv_id)
                papers.append(paper)
    return papers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract math arXiv PDFs.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=60.0)
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--metadata-source", action="append", default=None,
                        help="raw/ subdir(s) holding arXiv query XML to draw the paper pool "
                             "from; repeatable. Default: math_arxiv_metadata_core")
    parser.add_argument("--source-name", default="arxiv_math_papers",
                        help="source_name to catalogue downloaded papers under")
    parser.add_argument("--domain", default="mathematics_research_full_paper")
    args = parser.parse_args(argv)

    conn = init_db(args.data_root / "metadata" / "documents.sqlite")
    metadata_sources = args.metadata_source or ["math_arxiv_metadata_core"]
    papers = collect_papers(args.data_root, metadata_sources)
    selected = papers[args.offset : args.offset + args.limit]
    print(f"available_papers={len(papers)} selected={len(selected)} offset={args.offset}", flush=True)

    for paper in selected:
        result = ingest_paper(
            conn,
            args.data_root,
            paper,
            args.timeout,
            args.retries,
            args.retry_sleep,
            args.extract_images,
            args.force,
            args.source_name,
            args.domain,
        )
        print(result, flush=True)
        # only throttle when we actually hit the network; skips are just a DB
        # lookup, so a resume can churn past already-downloaded papers fast
        if args.sleep and not result.startswith("skip"):
            time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

