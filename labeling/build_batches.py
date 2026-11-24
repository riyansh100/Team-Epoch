"""Build relevance-labelling batches: every query x a few dataset cards per prompt.

Reads the benchmark and profiled tables produced by notebook 01 (the downloaded zip or an unpacked folder) and writes
one JSON file per batch. Each batch holds the user message for `run_labels.py`. The labeller sees only the original
metadata and the data (profile, sample rows, frequent values), never generated descriptions (UFD/SFD).

    python labeling/build_batches.py --source results/ntcir_ddg_intermediate.zip --out results/labels --per-batch 10
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

csv.field_size_limit(sys.maxsize)

DESC_CHARS = 1200
PROFILE_CHARS = 1500
MAX_COLUMNS = 40
SAMPLE_ROWS = 5
SAMPLE_COLS = 25
CELL_CHARS = 60
TOP_COLUMNS = 15
TOP_K = 5
TOP_VALUE_CHARS = 40


class Source:
    """Read files from the notebook-01 zip or from an unpacked folder with benchmark/ and intermediate/."""

    def __init__(self, path: Path):
        self.path = path
        self.zip = zipfile.ZipFile(path) if path.suffix == ".zip" else None

    def text(self, rel: str) -> str:
        if self.zip:
            return self.zip.read(rel).decode("utf-8")
        return (self.path / rel).read_text(encoding="utf-8")


def clip(s, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def sample_rows(sample_csv: str) -> str:
    rows = list(csv.reader(io.StringIO(sample_csv)))[: SAMPLE_ROWS + 1]
    if not rows:
        return ""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    for r in rows:
        cells = [clip(c, CELL_CHARS) for c in r[:SAMPLE_COLS]]
        if len(r) > SAMPLE_COLS:
            cells.append(f"(+{len(r) - SAMPLE_COLS} more columns)")
        w.writerow(cells)
    return out.getvalue().rstrip("\n")


def frequent_values(top_values: dict) -> str:
    lines = []
    for col, info in list(top_values.items())[:TOP_COLUMNS]:
        top = info.get("top") or []
        if not top:
            continue
        vals = ", ".join(f"{clip(v, TOP_VALUE_CHARS)} ({c})" for v, c in top[:TOP_K])
        lines.append(f"- {clip(col, 60)} [{info.get('n_unique', '?')} distinct]: {vals}")
    return "\n".join(lines)


def card(handle: str, doc: dict, table: dict | None) -> str:
    parts = [f"### {handle}", f"Title: {clip(doc.get('title'), 300)}"]
    if doc.get("publisher"):
        parts.append(f"Publisher: {clip(doc['publisher'], 150)}")
    parts.append(f"Description: {clip(doc.get('description'), DESC_CHARS) or '(none)'}")
    if not table:
        parts.append("Data: (not available; judge from the metadata above)")
        return "\n".join(parts)
    cols = table.get("columns") or []
    parts.append(f"Size: {table.get('n_rows')} rows x {table.get('n_cols')} columns")
    col_txt = ", ".join(clip(c, 60) for c in cols[:MAX_COLUMNS])
    if len(cols) > MAX_COLUMNS:
        col_txt += f", ... (+{len(cols) - MAX_COLUMNS} more)"
    parts.append(f"Columns: {col_txt}")
    profile = (table.get("basic_profile") or "").replace(
        "The key data profile information for this dataset includes:", ""
    )
    if profile.strip():
        parts.append(f"Profile: {clip(profile, PROFILE_CHARS)}")
    if table.get("top_values"):
        fv = frequent_values(table["top_values"])
        if fv:
            parts.append("Frequent values:\n" + fv)
    if table.get("sample_csv"):
        parts.append("Sample rows (CSV):\n" + sample_rows(table["sample_csv"]))
    return "\n".join(parts)


def load(src: Source):
    queries = {}
    for line in src.text("benchmark/queries.tsv").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] != "qid":
            queries[p[0]] = p[1]
    docs = {}
    for line in src.text("benchmark/docs.jsonl").splitlines():
        if line.strip():
            d = json.loads(line)
            docs[d["id"]] = d
    tables = {}
    for line in src.text("intermediate/tables.jsonl").splitlines():
        if line.strip():
            t = json.loads(line)
            if t.get("sample_csv") and not t.get("error"):
                tables[t["id"]] = t
    return queries, docs, tables


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="notebook-01 zip or unpacked folder")
    ap.add_argument("--out", type=Path, default=Path("results/labels"))
    ap.add_argument("--per-batch", type=int, default=10, help="dataset cards per prompt")
    ap.add_argument("--include-unprofiled", action="store_true",
                    help="also label datasets without a profiled table (metadata-only cards)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N datasets (pilot)")
    args = ap.parse_args()

    queries, docs, tables = load(Source(args.source))
    qids = sorted(queries)
    qmap = {f"Q{i + 1:02d}": q for i, q in enumerate(qids)}
    qblock = "\n".join(f"{h}: {queries[q]}" for h, q in qmap.items())
    ids = sorted(d for d in docs if args.include_unprofiled or d in tables)
    if args.limit:
        ids = ids[: args.limit]

    bdir = args.out / "batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for old in bdir.glob("batch_*.json"):
        old.unlink()
    system = (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8")
    total_chars = 0
    n_batches = 0
    for b, start in enumerate(range(0, len(ids), args.per_batch)):
        chunk = ids[start: start + args.per_batch]
        dmap = {f"D{i + 1}": d for i, d in enumerate(chunk)}
        cards = "\n\n".join(card(h, docs[d], tables.get(d)) for h, d in dmap.items())
        user = f"Queries:\n{qblock}\n\nDatasets:\n\n{cards}\n\nReturn the JSON for datasets {', '.join(dmap)}."
        (bdir / f"batch_{b:04d}.json").write_text(
            json.dumps({"batch": f"batch_{b:04d}", "datasets": dmap, "queries": qmap, "user": user},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        total_chars += len(system) + len(user)
        n_batches += 1

    n_prof = sum(d in tables for d in ids)
    print(f"{len(qids)} queries; {len(ids)} datasets ({n_prof} with a profiled table) of {len(docs)} in docs.jsonl")
    print(f"{n_batches} batches of <= {args.per_batch} -> {bdir}")
    if n_batches:
        print(f"prompt size: ~{total_chars / n_batches / 4:,.0f} tokens per batch, "
              f"~{total_chars / 4:,.0f} tokens in total (rough: 4 characters per token)")


if __name__ == "__main__":
    main()
