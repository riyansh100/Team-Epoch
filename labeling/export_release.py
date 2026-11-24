"""Export the NTCIR-DDG relevance labels (NTCIR humans, DSEBench humans, Claude Opus 5.5) as a self-contained folder.

    python3 labeling/export_release.py --source results/ntcir_ddg_intermediate.zip \
        --llm-labels results/labels_final/labels.jsonl --dsebench results/dsebench --out ntcir_ddg_labels

Writes queries.tsv, datasets.jsonl, judgments.tsv (one row per query x dataset, all three label sources side by side)
and qrels/*.txt in TREC format. README.md, the prompt copy and the licence files are kept by hand in the folder.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
import zipfile
from pathlib import Path


def read_source(path: Path, rel: str) -> str:
    if path.suffix == ".zip":
        return zipfile.ZipFile(path).read(rel).decode("utf-8")
    return (path / rel).read_text(encoding="utf-8")


def iter_source_lines(path: Path, rel: str):
    """Stream a large file line by line (tables.jsonl is ~50 MB)."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path).open(rel) as f:
            for line in f:
                yield line.decode("utf-8")
    else:
        with open(path / rel, encoding="utf-8") as f:
            yield from f


def load_queries(src: Path) -> dict[str, str]:
    out = {}
    for line in read_source(src, "benchmark/queries.tsv").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] != "qid":
            out[p[0]] = p[1].strip()
    return out


def load_ntcir(src: Path) -> dict[tuple[str, str], int]:
    out = {}
    for line in read_source(src, "benchmark/qrels.tsv").splitlines():  # no header line in this file
        p = line.split("\t")
        if len(p) >= 3 and p[2].strip().isdigit():
            out[(p[0], p[1])] = int(p[2])
    return out


def load_llm(path: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r["qid"], r["doc_id"])] = r
    return out


def load_dsebench(dse: Path, queries: dict[str, str]) -> dict[tuple[str, str], list[int]]:
    """All human query_rel votes per (our qid, dataset), matching DSEBench queries to ours by normalised text."""
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    ours = {norm(t): q for q, t in queries.items()}
    rev = {}
    for line in (dse / "queries.tsv").read_text(encoding="utf-8").splitlines():
        i, t = line.split("\t", 1)
        if norm(t) in ours:
            rev[i] = ours[norm(t)]
    votes = collections.defaultdict(list)
    for j in json.loads((dse / "human_annotated_judgments.json").read_text(encoding="utf-8")):
        if j["query_id"] in rev:
            votes[(rev[j["query_id"]], j["candidate_dataset_id"])].append(int(j["query_rel"]))
    return votes


def majority(v: list[int]) -> int:
    """Most frequent vote; ties go to the lower grade (same rule as labeling/agreement.py)."""
    return max(sorted(set(v)), key=v.count)


def load_datasets(src: Path) -> list[dict]:
    docs = [json.loads(x) for x in read_source(src, "benchmark/docs.jsonl").splitlines() if x.strip()]
    tables = {}
    for line in iter_source_lines(src, "intermediate/tables.jsonl"):
        if line.strip():
            t = json.loads(line)
            if t.get("sample_csv") and not t.get("error"):
                tables[t["id"]] = t  # last good row wins, as in build_batches.py
    out = []
    for d in sorted(docs, key=lambda d: d["id"]):
        t = tables.get(d["id"])
        files = d.get("csv_files") or []
        used = next((f for f in files if t and f["data_filename"] == t["file_used"]), None)
        out.append({
            "id": d["id"],
            "title": d["title"],
            "description": d["description"],
            "publisher": d.get("publisher"),
            "url": d["url"],
            "license": d.get("license"),
            "attribution": d.get("attribution"),
            "n_csv_files": len(files),
            "profiled": t is not None,
            "table_file": t["file_used"] if t else None,
            "table_url": used["data_url"] if used else None,
            "n_rows": t["n_rows"] if t else None,
            "n_cols": t["n_cols"] if t else None,
            "columns": t["columns"] if t else None,
        })
    return out


def write_trec(path: Path, qrels: dict[tuple[str, str], int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for (q, d), g in sorted(qrels.items()):
            f.write(f"{q} 0 {d} {g}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True, help="notebook-01 zip or unpacked folder")
    ap.add_argument("--llm-labels", type=Path, required=True, help="labels.jsonl from run_labels.py")
    ap.add_argument("--dsebench", type=Path, required=True, help="unpacked DSEBench_data folder")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    queries = load_queries(args.source)
    datasets = load_datasets(args.source)
    ids = [d["id"] for d in datasets]
    corpus = set(ids)
    ntcir = load_ntcir(args.source)
    llm = load_llm(args.llm_labels)
    dse_votes = load_dsebench(args.dsebench, queries)
    dse_in = {k: v for k, v in dse_votes.items() if k[1] in corpus}
    dse = {k: majority(v) for k, v in dse_in.items()}

    # sanity checks: every pair is ours, NTCIR pairs are exactly the pools, the LLM graded every pair
    pools = json.loads(read_source(args.source, "benchmark/pools.json"))
    assert set(ntcir) == {(q, d) for q, ds in pools.items() for d in ds}, "NTCIR qrels differ from the pools"
    assert all(q in queries and d in corpus for q, d in ntcir), "NTCIR pair outside queries x datasets"
    assert set(llm) == {(q, d) for q in queries for d in ids}, "LLM labels do not cover every query x dataset"
    models = {r["model"] for r in llm.values()}
    assert len(models) == 1, f"several labelling models: {models}"

    out = args.out
    (out / "qrels").mkdir(parents=True, exist_ok=True)
    with open(out / "queries.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["qid", "query"])
        w.writerows(sorted(queries.items()))
    with open(out / "datasets.jsonl", "w", encoding="utf-8") as f:
        for d in datasets:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(out / "judgments.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["qid", "doc_id", "in_ntcir_pool", "ntcir", "dsebench", "dsebench_votes", "llm", "llm_reason"])
        for q in sorted(queries):
            for d in ids:
                k = (q, d)
                w.writerow([q, d, int(k in ntcir), ntcir.get(k, ""), dse.get(k, ""),
                            ",".join(map(str, sorted(dse_in[k]))) if k in dse_in else "",
                            llm[k]["grade"], llm[k]["why"]])
    write_trec(out / "qrels" / "ntcir.txt", ntcir)
    write_trec(out / "qrels" / "dsebench.txt", dse)
    write_trec(out / "qrels" / "llm_opus55.txt", {k: r["grade"] for k, r in llm.items()})

    def dist(q):
        return dict(sorted(collections.Counter(q.values()).items()))

    print(f"queries {len(queries)}, datasets {len(datasets)} ({sum(d['profiled'] for d in datasets)} profiled)")
    print(f"ntcir    {len(ntcir):>6} pairs, grades {dist(ntcir)}")
    print(f"dsebench {len(dse):>6} pairs in {len({q for q, _ in dse})} queries, grades {dist(dse)}, "
          f"{sum(k not in ntcir for k in dse)} outside the NTCIR pools; "
          f"{sum(1 for k in dse_votes if k[1] not in corpus)} more pairs fall outside the 437 datasets (dropped)")
    llm_g = {k: r["grade"] for k, r in llm.items()}
    print(f"llm      {len(llm):>6} pairs, grades {dist(llm_g)}, model {models.pop()}")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
