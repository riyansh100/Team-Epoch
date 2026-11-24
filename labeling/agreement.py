"""Compare LLM relevance labels with human labels (NTCIR qrels, and optionally DSEBench's human judgments).

    python labeling/agreement.py --labels results/labels/llm_qrels.tsv --source results/ntcir_ddg_intermediate.zip \
        [--dsebench path/to/DSEBench_data]

Only pairs labelled by both sides are compared. Prints the 3x3 grade table, exact and relevant/not agreement, and
Cohen's kappa (binary and quadratic-weighted), plus the same between the two human sources for reference.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import zipfile
from pathlib import Path


def read_source(path: Path, rel: str) -> str:
    if path.suffix == ".zip":
        return zipfile.ZipFile(path).read(rel).decode("utf-8")
    return (path / rel).read_text(encoding="utf-8")


def ntcir_qrels(src: Path) -> dict:
    q = {}
    for line in read_source(src, "benchmark/qrels.tsv").splitlines():  # no header line in this file
        p = line.split("\t")
        if len(p) >= 3 and p[2].strip().isdigit():
            q[(p[0], p[1])] = int(p[2])
    return q


def dsebench_human(dse: Path, src: Path) -> dict:
    """Majority query_rel per (our qid, doc) from DSEBench's human judgments, matched to our queries by text."""
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    ours = {}
    for line in read_source(src, "benchmark/queries.tsv").splitlines():
        p = line.split("\t")
        if len(p) >= 2 and p[0] != "qid":
            ours[norm(p[1])] = p[0]
    rev = {}
    for line in (dse / "queries.tsv").read_text(encoding="utf-8").splitlines():
        i, t = line.split("\t", 1)
        if norm(t) in ours:
            rev[i] = ours[norm(t)]
    votes = collections.defaultdict(list)
    for j in json.loads((dse / "human_annotated_judgments.json").read_text(encoding="utf-8")):
        if j["query_id"] in rev:
            votes[(rev[j["query_id"]], j["candidate_dataset_id"])].append(int(j["query_rel"]))
    return {k: max(sorted(set(v)), key=v.count) for k, v in votes.items()}


def kappa(pairs, weights=None) -> float:
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    if n == 0 or len(cats) < 2:
        return float("nan")
    w = weights or (lambda a, b: 0.0 if a == b else 1.0)
    pa = collections.Counter(a for a, _ in pairs)
    pb = collections.Counter(b for _, b in pairs)
    obs = sum(w(a, b) for a, b in pairs) / n
    exp = sum(pa[a] * pb[b] * w(a, b) for a in cats for b in cats) / (n * n)
    return 1 - obs / exp if exp else float("nan")


def report(name: str, a: dict, b: dict, a_name: str, b_name: str):
    keys = sorted(set(a) & set(b))
    print(f"\n== {name}: {len(keys)} pairs labelled by both")
    if not keys:
        return
    pairs = [(a[k], b[k]) for k in keys]
    cm = collections.Counter(pairs)
    print(f"rows = {a_name} grade, columns = {b_name} grade")
    print("      " + "".join(f"{g:>6}" for g in (0, 1, 2)))
    for r in (0, 1, 2):
        print(f"{r:>6}" + "".join(f"{cm.get((r, c), 0):>6}" for c in (0, 1, 2)))
    exact = sum(x == y for x, y in pairs) / len(pairs)
    binp = [(int(x > 0), int(y > 0)) for x, y in pairs]
    binary = sum(x == y for x, y in binp) / len(binp)
    quad = kappa(pairs, lambda x, y: (x - y) ** 2 / 4)
    print(f"exact agreement {exact:.0%} | relevant/not agreement {binary:.0%} | "
          f"kappa binary {kappa(binp):.2f} | kappa quadratic-weighted {quad:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("results/labels/llm_qrels.tsv"))
    ap.add_argument("--source", type=Path, required=True, help="notebook-01 zip or unpacked folder")
    ap.add_argument("--dsebench", type=Path, help="unpacked DSEBench_data folder (optional)")
    args = ap.parse_args()

    llm = {}
    for line in args.labels.read_text(encoding="utf-8").splitlines()[1:]:
        q, d, g = line.split("\t")
        llm[(q, d)] = int(g)
    print(f"LLM labels: {len(llm)} pairs, {sum(g > 0 for g in llm.values())} relevant")
    ntcir = ntcir_qrels(args.source)
    report("LLM vs NTCIR", llm, ntcir, "LLM", "NTCIR")
    if args.dsebench:
        dse = dsebench_human(args.dsebench, args.source)
        report("LLM vs DSEBench humans", llm, dse, "LLM", "DSEBench")
        report("reference: NTCIR vs DSEBench humans", ntcir, dse, "NTCIR", "DSEBench")
        new = [k for k in llm if k not in ntcir]
        print(f"\nLLM pairs outside NTCIR's labels: {len(new)}, {sum(llm[k] > 0 for k in new)} graded relevant; "
              f"{sum(k in dse for k in new)} of them also have a DSEBench human label")


if __name__ == "__main__":
    main()
