#!/usr/bin/env python3
from pathlib import Path
import sqlite3


DB = Path("/home/seyominaoto/archimedes-data/metadata/documents.sqlite")


def main() -> int:
    if not DB.exists():
        print(f"missing database: {DB}")
        return 1

    conn = sqlite3.connect(DB)
    print("Archimedes data stats")
    print(f"database: {DB}")
    print()

    row = conn.execute(
        """
        SELECT
          COUNT(*),
          COALESCE(SUM(byte_count), 0),
          COALESCE(SUM(char_count), 0),
          COALESCE(SUM(token_estimate), 0)
        FROM documents
        WHERE status = 'ok'
        """
    ).fetchone()
    print(f"ok_docs:        {row[0]}")
    print(f"raw_bytes:      {row[1]}")
    print(f"cleaned_chars:  {row[2]}")
    print(f"token_estimate: {row[3]}")
    print()

    print("by_source:")
    for item in conn.execute(
        """
        SELECT source_name, COUNT(*), COALESCE(SUM(token_estimate), 0)
        FROM documents
        WHERE status = 'ok'
        GROUP BY source_name
        ORDER BY source_name
        """
    ):
        print(f"  {item[0]} docs={item[1]} token_estimate={item[2]}")

    errors = conn.execute("SELECT COUNT(*) FROM documents WHERE status != 'ok'").fetchone()[0]
    print()
    print(f"errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

