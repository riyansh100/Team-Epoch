"""Label query-dataset relevance with headless Claude Code (`claude -p`), on your Claude subscription.

Runs every batch from `build_batches.py` that has no saved result yet, so you can stop it and run it again at any
time: when a usage limit is hit it stops, and the next run continues after the limit resets.

    claude auth login                                    # once, with your Claude Pro account
    python labeling/run_labels.py --out results/labels --max-batches 2      # pilot
    python labeling/run_labels.py --out results/labels                      # the rest

Outputs in --out: raw/<batch>.json (full CLI result incl. token usage), labels.jsonl (one row per query x dataset),
llm_qrels.tsv (qid, doc_id, grade) and usage.tsv (tokens per batch).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "limit will reset", "429", "overloaded")
AUTH_WORDS = ("authenticate", "login", "logged in", "oauth", "api key")


def call_claude(system: str, user: str, model: str | None, effort: str | None, timeout: int) -> dict:
    cmd = ["claude", "-p", "--tools", "", "--strict-mcp-config", "--setting-sources", "",
           "--no-session-persistence", "--disable-slash-commands", "--output-format", "json",
           "--system-prompt", system]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    p = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"is_error": True, "result": (p.stdout + "\n" + p.stderr).strip()[:2000]}


def parse_labels(text: str, batch: dict) -> dict:
    """{doc_id: {qid: (grade, why)}} with every dataset of the batch; raises ValueError on a malformed answer."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the answer")
    obj = json.loads(m.group(0))
    out = {}
    for item in obj.get("labels", []):
        h = item.get("dataset")
        if h not in batch["datasets"]:
            raise ValueError(f"unknown dataset handle {h!r}")
        grades = {}
        why = item.get("why") or {}
        for qh, g in (item.get("grades") or {}).items():
            if qh not in batch["queries"]:
                raise ValueError(f"unknown query handle {qh!r}")
            g = int(g)
            if g not in (0, 1, 2):
                raise ValueError(f"bad grade {g!r}")
            if g:
                grades[batch["queries"][qh]] = (g, str(why.get(qh, ""))[:200])
        out[batch["datasets"][h]] = grades
    missing = set(batch["datasets"].values()) - set(out)
    if missing:
        raise ValueError(f"{len(missing)} datasets missing from the answer")
    return out


def usage_of(res: dict) -> dict:
    u = res.get("usage") or {}
    return {
        "input": u.get("input_tokens", 0),
        "cache_write": u.get("cache_creation_input_tokens", 0),
        "cache_read": u.get("cache_read_input_tokens", 0),
        "output": u.get("output_tokens", 0),
        "models": ",".join(sorted(res.get("modelUsage") or {})) or "?",
        "seconds": round((res.get("duration_ms") or 0) / 1000, 1),
    }


def rebuild_outputs(out: Path):
    """Write labels.jsonl, llm_qrels.tsv and usage.tsv from all saved raw results."""
    rows, usage = [], []
    for f in sorted((out / "raw").glob("batch_*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        batch, labels = rec["batch"], rec["labels"]
        u = usage_of(rec["cli"])
        usage.append((batch["batch"], u))
        for doc in batch["datasets"].values():
            for qid in batch["queries"].values():
                g, why = labels.get(doc, {}).get(qid, (0, ""))
                rows.append({"qid": qid, "doc_id": doc, "grade": g, "why": why,
                             "batch": batch["batch"], "model": u["models"]})
    with open(out / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "llm_qrels.tsv", "w", encoding="utf-8") as fh:
        fh.write("qid\tdoc_id\tgrade\n")
        for r in rows:
            fh.write(f"{r['qid']}\t{r['doc_id']}\t{r['grade']}\n")
    keys = ["input", "cache_write", "cache_read", "output", "seconds"]
    with open(out / "usage.tsv", "w", encoding="utf-8") as fh:
        fh.write("batch\t" + "\t".join(keys) + "\tmodels\n")
        for b, u in usage:
            fh.write(b + "\t" + "\t".join(str(u[k]) for k in keys) + f"\t{u['models']}\n")
    return rows, usage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/labels"), help="folder with batches/ from build_batches")
    ap.add_argument("--model", default=None, help="claude model alias, e.g. opus or sonnet (default: CLI default)")
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-batches", type=int, default=0, help="stop after N new batches (0 = all)")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between calls")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    system = (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8")
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    batches = sorted((args.out / "batches").glob("batch_*.json"))
    if not batches:
        sys.exit(f"no batches in {args.out / 'batches'}; run build_batches.py first")
    todo = [b for b in batches if not (raw / b.name).exists()]
    print(f"{len(batches)} batches, {len(batches) - len(todo)} done, {len(todo)} to do")

    done_now = 0
    for bf in todo:
        if args.max_batches and done_now >= args.max_batches:
            break
        batch = json.loads(bf.read_text(encoding="utf-8"))
        for attempt in (1, 2):
            res = call_claude(system, batch["user"], args.model, args.effort, args.timeout)
            text = str(res.get("result", ""))
            if res.get("is_error"):
                low = text.lower()
                if any(w in low for w in AUTH_WORDS):
                    sys.exit(f"not logged in: {text}\nrun `claude auth login` with your Claude account, then rerun")
                if any(w in low for w in LIMIT_WORDS):
                    print(f"stopping at {bf.name}: {text}\nrerun this command after the limit resets")
                    rebuild_outputs(args.out)
                    return
                print(f"{bf.name} attempt {attempt}: error: {text[:300]}")
                continue
            try:
                labels = parse_labels(text, batch)
            except (ValueError, json.JSONDecodeError) as e:
                (raw / f"{bf.stem}.bad{attempt}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
                print(f"{bf.name} attempt {attempt}: unusable answer ({e})")
                continue
            ser = {d: {q: list(v) for q, v in g.items()} for d, g in labels.items()}
            (raw / bf.name).write_text(json.dumps({"batch": batch, "labels": ser, "cli": res},
                                                  ensure_ascii=False, indent=1), encoding="utf-8")
            u = usage_of(res)
            n_rel = sum(len(g) for g in labels.values())
            print(f"{bf.name}: {n_rel} relevant pairs | tokens in {u['input']:,} + cache write {u['cache_write']:,}"
                  f" + cache read {u['cache_read']:,}, out {u['output']:,} | {u['seconds']}s | {u['models']}")
            done_now += 1
            break
        time.sleep(args.pause)

    rows, usage = rebuild_outputs(args.out)
    if usage:
        tot = {k: sum(u[k] for _, u in usage) for k in ("input", "cache_write", "cache_read", "output")}
        n = len(usage)
        print(f"\n{n}/{len(batches)} batches labelled; {sum(r['grade'] > 0 for r in rows)} relevant of {len(rows)} pairs")
        print(f"tokens so far: input {tot['input']:,}, cache write {tot['cache_write']:,}, cache read "
              f"{tot['cache_read']:,}, output {tot['output']:,}")
        per = sum(tot.values()) / n
        print(f"average {per:,.0f} tokens per batch -> all {len(batches)} batches ~{per * len(batches):,.0f} tokens")


if __name__ == "__main__":
    main()
