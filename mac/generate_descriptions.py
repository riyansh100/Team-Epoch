#!/usr/bin/env python3
"""Batch-generate AutoDDG descriptions (UFD + SFD) for every table in tables.jsonl.

Follows AutoDDG's quick-start notebook exactly (see zen_common.describe_table):
    sample_df        = pd.read_csv(io.StringIO(sample_csv), dtype=str, keep_default_na=False)
    semantic_details = analyze_semantics(sample_df, use_group_prompting=True, group_size=...)
    semantic_profile = structural_profile + "\\n" + semantic_details
    topic            = generate_topic(title, original_description, sample_csv)
    ufd              = describe_dataset(sample_csv, basic_profile, semantic_profile, topic, use_*=True)
    sfd              = expand_description_for_search(ufd, topic)

Every completion is cached in SQLite (re-runs cost nothing; a crash loses at most the call in flight), every finished
table is appended to the output immediately, and tables that already succeeded are skipped on the next run.

Example:
    python mac/generate_descriptions.py --tables data/intermediate/tables.jsonl \
        --out data/descriptions/descriptions.jsonl --model big-pickle --limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from importlib.metadata import version as pkg_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zen_common as zc  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut by a crash
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", required=True, type=Path, help="intermediate/tables.jsonl from notebook 01")
    ap.add_argument("--out", required=True, type=Path, help="descriptions.jsonl (appended to)")
    ap.add_argument("--model", required=True, help="exact model id; pin ONE model for all experiments")
    ap.add_argument("--base-url", default=os.environ.get("ZEN_BASE_URL", zc.DEFAULT_BASE_URL))
    ap.add_argument("--api-key-env", default="OPENCODE_API_KEY")
    ap.add_argument("--cache", type=Path, help="SQLite cache (default: llm_cache.sqlite next to --out)")
    ap.add_argument("--concurrency", type=int, default=1, help="tables processed in parallel (free tier: 1-2)")
    ap.add_argument("--limit", type=int, help="process at most N pending tables (pilot runs)")
    ap.add_argument("--ids", help="comma-separated table ids (or @file with one id per line) to (re)run")
    ap.add_argument("--retry-errors", action="store_true", help="also rerun tables whose last attempt failed")
    ap.add_argument("--group-size", type=int, default=0, help="semantic group size; 0 = all columns in one prompt")
    ap.add_argument("--wide-cols", type=int, default=40, help="tables wider than this use --wide-group-size")
    ap.add_argument("--wide-group-size", type=int, default=10)
    ap.add_argument("--description-words", type=int, default=100, help="AutoDDG default: 100")
    ap.add_argument("--temperature", choices=["auto", "send", "omit"], default="auto",
                    help="send temperature=0 (auto: unless /models says the model rejects it)")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--allow-mixed-models", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"WARNING: ${args.api_key_env} is not set; Zen rejects keyless calls (429 FreeUsageLimitError).")

    tables = [t for t in read_jsonl(args.tables)]
    by_id = {t["id"]: t for t in tables}
    usable = {i: t for i, t in by_id.items() if not t.get("error") and t.get("sample_csv")}
    print(f"tables.jsonl: {len(by_id)} tables, {len(usable)} usable (others failed profiling and are skipped)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    for row in read_jsonl(args.out):
        existing[row["id"]] = row  # last row per id wins
    done_models = {r["model"] for r in existing.values() if not r.get("error")}
    if done_models - {args.model} and not args.allow_mixed_models:
        print(f"ERROR: {args.out} already holds results from {sorted(done_models)}; this run uses {args.model!r}. "
              "All experiments must use one pinned model: pick another --out or pass --allow-mixed-models.")
        return zc.EXIT_GENERIC

    if args.ids:
        ids = ([s.strip() for s in Path(args.ids[1:]).read_text().split()] if args.ids.startswith("@")
               else [s.strip() for s in args.ids.split(",") if s.strip()])
        unknown = [i for i in ids if i not in usable]
        if unknown:
            print(f"WARNING: {len(unknown)} id(s) not usable/unknown, skipped: {unknown[:5]}")
        todo = [i for i in ids if i in usable]
    else:
        transient = ("network", "rate_limited", "quota_exhausted", "retries_exhausted")

        def pending(i):  # never run, or failed (transient failures are always retried)
            r = existing.get(i)
            return r is None or (bool(r.get("error")) and (args.retry_errors or r.get("error_kind") in transient))
        todo = [i for i in sorted(usable) if pending(i)]
    n_done = sum(1 for i in usable if existing.get(i) and not existing[i].get("error"))
    n_err = sum(1 for i in usable if existing.get(i) and existing[i].get("error"))
    if args.limit is not None:
        todo = todo[: args.limit]

    # ---- models / temperature
    models, model_info = [], None
    try:
        models = zc.fetch_models(args.base_url, api_key)
        model_info = next((m for m in models if m.get("id") == args.model), None)
        if model_info is None:
            print(f"WARNING: {args.model!r} is not in GET /models ({len(models)} models listed).")
        else:
            print(f"model {args.model}: free={zc.is_free(model_info)} api={model_info.get('recommended_api')} "
                  f"temperature={zc.supports_temperature(model_info)}")
            if not zc.chat_compatible(model_info):
                print("ERROR: this model is not served on chat-completions; pick another (see zen_smoke_test.py).")
                return zc.EXIT_MODEL
    except Exception as e:  # noqa: BLE001
        print(f"note: GET /models failed ({e}); continuing")
    send_temp = {"send": True, "omit": False}.get(args.temperature, zc.supports_temperature(model_info) is not False)
    if not send_temp:
        print("temperature is NOT sent (model rejects it); outputs are recorded with temperature='omitted'")

    # ---- plan
    calls = tokens = 0
    for i in todo:
        c, t = zc.estimate_table(usable[i], args.group_size, args.wide_cols, args.wide_group_size,
                                 args.description_words)
        calls, tokens = calls + c, tokens + t
    print(f"\nPLAN: {len(todo)} tables to run ({n_done} already done, {n_err} previous errors"
          f"{'' if args.retry_errors else ' not retried without --retry-errors'}), concurrency {args.concurrency}")
    print(f"      ~{calls} LLM calls (min; semantic fallbacks/retries add more), ~{tokens / 1e6:.2f} M tokens "
          "(rough: 4 chars/token; cached calls cost nothing)")
    print(f"      model {args.model!r} @ {args.base_url}; output {args.out}")
    if args.dry_run or not todo:
        return zc.EXIT_OK
    if not args.yes and sys.stdin.isatty():
        if input("proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            return zc.EXIT_OK

    cache = args.cache or args.out.parent / "llm_cache.sqlite"
    llm = zc.CachingLLMClient(zc.make_openai_client(args.base_url, api_key), str(cache),
                              send_temperature=send_temp, max_retries=args.max_retries)
    pipe = zc.build_autoddg(llm, args.model, args.description_words)
    autoddg_version = pkg_version("autoddg")

    write_lock, stop = threading.Lock(), threading.Event()
    fatal: list[zc.FatalAPIError] = []
    counters = {"ok": 0, "error": 0}
    t_start = time.time()

    def work(doc_id: str) -> None:
        if stop.is_set():
            return
        t = usable[doc_id]
        row = {"id": doc_id, "model": args.model, "base_url": args.base_url,
               "temperature": 0.0 if send_temp else "omitted", "description_words": args.description_words,
               "autoddg_version": autoddg_version, "title": t.get("title"), "topic": None, "semantic_profile": None,
               "ufd": None, "sfd": None, "error": None, "error_kind": None}
        t0 = time.time()
        try:
            res = zc.describe_table(pipe, llm, t, group_size=args.group_size, wide_cols=args.wide_cols,
                                    wide_group_size=args.wide_group_size)
            row.update({k: res[k] for k in ("topic", "semantic_profile", "ufd", "sfd", "n_cols", "group_size_used",
                                            "semantic_columns_covered", "usage", "usage_total")})
            if res["semantic_columns_covered"] < res["n_cols"]:
                row["warning"] = (f"semantic profile covers {res['semantic_columns_covered']}/{res['n_cols']} "
                                  "columns")
        except zc.FatalAPIError as e:
            row["error"], row["error_kind"] = str(e), e.kind
            if e.kind in ("auth", "model_not_found", "unsupported_parameter", "quota_exhausted"):
                fatal.append(e)
                stop.set()
        except zc.RetriesExhausted as e:
            row["error"], row["error_kind"] = str(e), "retries_exhausted"
        except Exception as e:  # noqa: BLE001
            row["error"], row["error_kind"] = f"{type(e).__name__}: {e}", "pipeline"
        row["latency_s"] = round(time.time() - t0, 2)
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        with write_lock:
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            counters["error" if row["error"] else "ok"] += 1
            n = counters["ok"] + counters["error"]
            ut = row.get("usage_total") or {}
            print(f"[{n}/{len(todo)}] {doc_id[:8]} {row['latency_s']:6.1f}s "
                  f"calls={ut.get('calls', '-')} cached={ut.get('cached_calls', '-')} "
                  f"tok={ut.get('total_tokens', '-')} {'ERROR ' + row['error'][:120] if row['error'] else 'ok'}"
                  f"{'  (' + row['warning'] + ')' if row.get('warning') else ''}", flush=True)

    interrupted = False

    def on_sigint(signum, frame):  # finish in-flight tables, start no new ones
        nonlocal interrupted
        interrupted = True
        stop.set()
        print("\ninterrupt: finishing tables in flight, then stopping (results so far are saved)", flush=True)

    signal.signal(signal.SIGINT, on_sigint)
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        queue = list(todo)
        running = set()
        while queue or running:
            while queue and len(running) < max(1, args.concurrency) and not stop.is_set():
                running.add(ex.submit(work, queue.pop(0)))
            if not running:
                break
            finished, running = wait(running, return_when=FIRST_COMPLETED)
            running = set(running)

    el = time.time() - t_start
    print(f"\nfinished in {el / 60:.1f} min: {counters['ok']} ok, {counters['error']} errors; "
          f"API calls {llm.stats['api_calls']}, cache hits {llm.stats['cache_hits']}, retries {llm.stats['retries']}")
    if fatal:
        e = fatal[0]
        print(f"STOPPED on a fatal API error ({e.kind}): {e}")
        return e.exit_code
    if interrupted:
        return 130
    return zc.EXIT_OK if counters["error"] == 0 else zc.EXIT_GENERIC


if __name__ == "__main__":
    sys.exit(main())
