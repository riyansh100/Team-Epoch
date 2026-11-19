#!/usr/bin/env python3
"""Check that OpenCode Zen works, then run AutoDDG end-to-end on ONE table.

Steps
  1. GET /models (with and without the key); list models, highlight free ones (id ends in "-free" or $0 price).
  2. Probe a chat call WITHOUT a key and report what happens.
  3. Pick the model: --model, else the first free model served on chat-completions.
  4. One-line chat completion with temperature=0: latency, token usage, reply.
  5. AutoDDG on one table (first usable record of --tables, or AutoDDG's bundled example CSV): semantic profile
     (group prompting), topic, UFD, SFD. Prints each output plus total tokens and time.

Exit codes: 0 ok | 2 auth | 3 model not found | 4 rate limit / free quota exhausted | 5 unsupported parameter
            (e.g. temperature) | 6 other API error | 7 AutoDDG pipeline failure | 8 network | 1 other
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zen_common as zc  # noqa: E402
import httpx  # noqa: E402

EXAMPLE_CSV_URL = ("https://raw.githubusercontent.com/VIDA-NYU/AutoDDG/"
                   "f286ac697bdb0a199eac1ce0778401f30341b4bb/examples/clark_dataset.csv")
EXAMPLE_TITLE = "Renal Cell Carcinoma"
EXAMPLE_DESCRIPTION = ("This study reports a large-scale proteogenomic analysis of ccRCC to discern the functional "
                       "impact of genomic alterations and provides evidence for rational treatment selection stemming "
                       "from ccRCC pathobiology")
FALLBACK_CSV = """Case_ID,Gender,Age,BMI,Tumor_Site,Tumor_Size_cm,Grade,Tumor_Stage_Pathological
C3L-00004,Male,72,22.8,Lower pole,12.0,G3,Stage III
C3L-00010,Male,30,34.15,Upper pole,6.5,G3,Stage I
C3L-00011,Female,77,24.69,Lower pole,8.5,G4,Stage IV
C3L-00026,Female,74,31.35,Upper pole,4.5,G2,Stage I
C3L-00079,Male,64,26.22,Middle,11.0,G4,Stage III
"""


def fail(code: int, msg: str) -> int:
    print(f"\nFAIL ({code}): {msg}")
    return code


def banner(s: str) -> None:
    print(f"\n{'=' * 8} {s} {'=' * max(0, 70 - len(s))}")


def load_table(args) -> dict:
    """A tables.jsonl record, or one built from AutoDDG's example CSV exactly like the quick-start notebook."""
    if args.tables:
        for line in open(args.tables, encoding="utf-8"):
            rec = json.loads(line)
            if (args.id is None or rec["id"] == args.id) and not rec.get("error") and rec.get("sample_csv"):
                print(f"table {rec['id']}: {rec.get('title')!r} ({rec.get('n_rows')} rows x {rec.get('n_cols')} cols)")
                return rec
        raise SystemExit(fail(zc.EXIT_GENERIC, f"no usable record{' with id ' + args.id if args.id else ''} in "
                                               f"{args.tables}"))
    import io

    import httpx
    import pandas as pd
    from autoddg.profiling import profile_dataset
    from autoddg.utils import get_sample

    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "autoddg_clark_dataset.csv"
    try:
        if not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(httpx.get(EXAMPLE_CSV_URL, timeout=30, follow_redirects=True).raise_for_status().text)
        text, src = cache.read_text(), "AutoDDG examples/clark_dataset.csv"
    except Exception as e:  # noqa: BLE001
        text, src = FALLBACK_CSV, f"built-in 5-row excerpt (download failed: {e})"
    df = pd.read_csv(io.StringIO(text))
    _, sample_csv = get_sample(df, sample_size=100)
    basic_profile, structural_profile = profile_dataset(df)
    print(f"table: {src} ({len(df)} rows x {df.shape[1]} cols)")
    return {"id": "example", "title": EXAMPLE_TITLE, "original_description": EXAMPLE_DESCRIPTION,
            "sample_csv": sample_csv, "basic_profile": basic_profile, "structural_profile": structural_profile}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model id (default: first free chat-completions model)")
    ap.add_argument("--base-url", default=os.environ.get("ZEN_BASE_URL", zc.DEFAULT_BASE_URL))
    ap.add_argument("--api-key-env", default="OPENCODE_API_KEY")
    ap.add_argument("--tables", type=Path, help="tables.jsonl; the first usable record is used")
    ap.add_argument("--id", help="use this table id from --tables")
    ap.add_argument("--temperature", choices=["auto", "send", "omit"], default="auto")
    ap.add_argument("--skip-autoddg", action="store_true", help="only test the API")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    banner("1. API key and models")
    print(f"base URL : {args.base_url}")
    print(f"API key  : {'$' + args.api_key_env + ' is set (' + api_key[:4] + '…)' if api_key else 'NOT SET'}")

    models, models_error = [], None
    for label, key in (("with key", api_key), ("without key", None)) if api_key else (("without key", None),):
        try:
            got = zc.fetch_models(args.base_url, key)
            print(f"GET /models {label}: {len(got)} models")
            models = models or got
        except Exception as e:  # noqa: BLE001
            models_error = e
            print(f"GET /models {label}: failed ({type(e).__name__}: {str(e)[:200]})")
    if models:
        print(f"\n{'model id':38s} {'free':5s} {'api':18s} {'temp':5s}")
        for m in sorted(models, key=lambda m: (not zc.is_free(m), m.get("id", ""))):
            if zc.is_free(m) or len(models) <= 15:
                star = "*" if zc.is_free(m) else " "
                print(f"{star}{m.get('id', ''):37s} {str(zc.is_free(m)):5s} {str(m.get('recommended_api', '?')):18s} "
                      f"{str(zc.supports_temperature(m)):5s}")
        print(f"(* = free; {sum(not zc.is_free(m) for m in models)} paid models not shown)" if len(models) > 15 else "")

    model_info = None
    if args.model:
        model_info = next((m for m in models if m.get("id") == args.model), None)
        if models and model_info is None:
            return fail(zc.EXIT_MODEL, f"model {args.model!r} is not listed by GET /models")
        model = args.model
    elif not models:
        code = zc.EXIT_NETWORK if isinstance(models_error, httpx.TransportError) else zc.EXIT_API
        return fail(code, f"GET /models failed ({models_error}); cannot pick a default model. Check the network / "
                          "--base-url, or pass --model")
    else:
        model_info = zc.pick_default_model(models)
        if model_info is None:
            return fail(zc.EXIT_MODEL, "no free chat-completions model found; pass --model")
        model = model_info["id"]
    if model_info is not None and not zc.chat_compatible(model_info):
        return fail(zc.EXIT_MODEL, f"{model!r} is served on {model_info.get('recommended_api')!r}, not "
                                   "chat-completions; pick a model whose api is chat-completions")
    send_temp = {"send": True, "omit": False}.get(args.temperature, zc.supports_temperature(model_info) is not False)
    print(f"\nusing model: {model}   (temperature {'sent as 0' if send_temp else 'omitted: model rejects it'})")

    banner("2. Keyless probe")
    try:
        r = httpx.post(args.base_url.rstrip("/") + "/chat/completions", timeout=60,
                       json={"model": model, "messages": [{"role": "user", "content": "Reply with: pong"}]})
        print(f"chat without a key -> HTTP {r.status_code}: {r.text[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"chat without a key -> {type(e).__name__}: {e}")
    if not api_key:
        return fail(zc.EXIT_AUTH, f"${args.api_key_env} is not set. Create a key at https://opencode.ai/zen and "
                                  f"`export {args.api_key_env}=...`")

    banner("3. One-line chat completion")
    llm = zc.CachingLLMClient(zc.make_openai_client(args.base_url, api_key, timeout=120), None,
                              send_temperature=send_temp, max_retries=args.max_retries)
    try:
        with llm.step("ping") as rec:
            t0 = time.time()
            resp = llm.chat_completions_create(model=model, messages=[
                {"role": "user", "content": "Reply with exactly one word: pong"}], temperature=0.0)
            dt = time.time() - t0
    except zc.FatalAPIError as e:
        hint = {"unsupported_parameter": " -> retry with --temperature omit",
                "quota_exhausted": " -> free quota used up; wait or check your Zen plan",
                "auth": " -> check the key"}.get(e.kind, "")
        return fail(e.exit_code, f"{e}{hint}")
    except zc.RetriesExhausted as e:
        return fail(e.exit_code, str(e))
    print(f"reply   : {resp['choices'][0]['message']['content'].strip()[:200]!r}")
    print(f"latency : {dt:.2f}s   usage: {resp.get('usage')}   served model: {resp.get('model')}")
    if args.skip_autoddg:
        print("\nOK (API only)")
        return zc.EXIT_OK

    banner("4. AutoDDG end-to-end on one table")
    table = load_table(args)
    pipe = zc.build_autoddg(llm, model)
    t0 = time.time()
    try:
        res = zc.describe_table(pipe, llm, table)
    except zc.FatalAPIError as e:
        return fail(e.exit_code, str(e))
    except zc.RetriesExhausted as e:
        return fail(e.exit_code, str(e))
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return fail(zc.EXIT_PIPELINE, f"AutoDDG pipeline failed: {type(e).__name__}: {e}")
    total_s = time.time() - t0
    print(f"\n--- semantic profile ({res['semantic_columns_covered']}/{res['n_cols']} columns described, "
          f"group_size={res['group_size_used']}) ---\n{res['semantic_profile'][:3000]}")
    print(f"\n--- topic ---\n{res['topic']}")
    print(f"\n--- UFD ({len(res['ufd'].split())} words) ---\n{res['ufd']}")
    print(f"\n--- SFD ({len(res['sfd'].split())} words) ---\n{res['sfd']}")
    print("\n--- usage per step ---")
    for step, u in res["usage"].items():
        print(f"  {step:9s} calls={u['calls']} prompt={u['prompt_tokens']:6d} completion={u['completion_tokens']:6d} "
              f"latency={u['latency_s']:.1f}s retries={u['retries']}")
    t = res["usage_total"]
    print(f"  TOTAL     calls={t['calls']} tokens={t['total_tokens']} time={total_s:.1f}s")
    problems = []
    if res["semantic_columns_covered"] < res["n_cols"]:
        problems.append(f"semantic profile covers only {res['semantic_columns_covered']}/{res['n_cols']} columns")
    if not res["topic"].strip() or not res["ufd"].strip() or not res["sfd"].strip():
        problems.append("an output is empty")
    if problems:
        print("\nWARNING: " + "; ".join(problems) + " (the model may struggle with AutoDDG's JSON prompts)")
    print(f"\nOK: {model} works end-to-end.")
    return zc.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
