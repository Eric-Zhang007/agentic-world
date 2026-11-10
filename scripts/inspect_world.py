import argparse
import json
from pathlib import Path
import sqlite3


def inspect(path):
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    tables = connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    result = {"path": str(path), "tables": {}}
    for name, sql in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        columns = [r[1] for r in connection.execute(f"PRAGMA table_info({quoted})")]
        count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
        rows = connection.execute(f"SELECT * FROM {quoted} LIMIT 1").fetchall()
        samples = [dict(zip(columns, row)) for row in rows]
        result["tables"][name] = {"schema": sql, "count": count, "sample": samples}
        print(name, count, sql, flush=True)
        for sample in samples:
            print(json.dumps(sample, ensure_ascii=True, default=str)[:3500], flush=True)
    connection.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output")
    args = parser.parse_args()
    path = Path(args.path)
    paths = sorted(path.rglob("*.sqlite3")) if path.is_dir() else [path]
    result = [inspect(p) for p in paths]
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
