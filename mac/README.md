# Mac side: LLM generation with OpenCode Zen

The Mac only runs the LLM steps (semantic profile, topic, UFD, SFD) over the small files that notebook 01 produced on
Colab. Everything here stays well under 1 GB (see [Disk use](#disk-use)).

```
Colab: notebooks/01_build_ntcir_ddg.ipynb  ->  Drive: ntcir_ddg/ntcir_ddg_intermediate.zip
Mac:   generate_descriptions.py            ->  descriptions.jsonl  ->  Drive: ntcir_ddg/descriptions/
Colab or Mac: notebooks/02_eval_retrieval.ipynb
```

All commands below are run from the repository root (`Team-Epoch/`).

## 1. Python environment (Python 3.12)

Use **Python 3.10–3.12**. Python 3.13 does not work: AutoDDG pins `fastembed==0.5.1`, which needs `mmh3==4.1.0`, and
that has no 3.13 wheel (it would need a C compiler). `uv` downloads a 3.12 interpreter for you.

```bash
brew install uv
```

```bash
uv venv -p 3.12 .venv-mac
```

```bash
source .venv-mac/bin/activate
```

```bash
uv pip install -r requirements-mac.txt
```

Without uv: `python3.12 -m venv .venv-mac && source .venv-mac/bin/activate && pip install -r requirements-mac.txt`.
Run `source .venv-mac/bin/activate` again in every new terminal.

## 2. API key

Create a key at <https://opencode.ai/zen>, then:

```bash
export OPENCODE_API_KEY="paste-your-key-here"
```

(Add that line to `~/.zshrc` to keep it.) Never commit the key.

## 3. Smoke test

Lists the models (free ones are starred), probes a keyless call, sends one chat completion, then runs AutoDDG
end-to-end on AutoDDG's own example table:

```bash
python mac/zen_smoke_test.py
```

It picks the first free model served on the chat-completions API. **Choose one model and pin it** for every experiment:

```bash
export ZEN_MODEL="big-pickle"
```

```bash
python mac/zen_smoke_test.py --model "$ZEN_MODEL"
```

Exit codes: `2` auth, `3` model not found / not on chat-completions, `4` rate limit or free quota used up,
`5` unsupported parameter (e.g. the model rejects `temperature`: add `--temperature omit`), `6` other API error,
`7` AutoDDG pipeline error, `8` network. The models listed in Nov 2026 included `big-pickle`,
`mimo-v2.6-flash-free`, `deepseek-v4.1-flash-free` and `longcat-2.5-free` (which rejects `temperature`); the free list
changes, so trust the smoke test's listing over this README.

## 4. Get the intermediate files from Drive

Download `MyDrive/ntcir_ddg/ntcir_ddg_intermediate.zip` (a few MB) to `~/Downloads`, then:

```bash
mkdir -p ~/ntcir_ddg && unzip -o ~/Downloads/ntcir_ddg_intermediate.zip -d ~/ntcir_ddg
```

This gives `~/ntcir_ddg/benchmark/` and `~/ntcir_ddg/intermediate/tables.jsonl`. Try one real table:

```bash
python mac/zen_smoke_test.py --model "$ZEN_MODEL" --tables ~/ntcir_ddg/intermediate/tables.jsonl
```

## 5. Pilot (5 tables), then the full run

```bash
python mac/generate_descriptions.py --tables ~/ntcir_ddg/intermediate/tables.jsonl --out ~/ntcir_ddg/descriptions/descriptions.jsonl --model "$ZEN_MODEL" --limit 5
```

It prints the plan first (number of LLM calls, rough token count) and asks for confirmation (`--yes` skips the
question, `--dry-run` only prints the plan). Look at the 5 rows of `descriptions.jsonl` (topic, ufd, sfd, warnings).
If they look right, run everything. `caffeinate -i` keeps the Mac awake:

```bash
caffeinate -i python mac/generate_descriptions.py --tables ~/ntcir_ddg/intermediate/tables.jsonl --out ~/ntcir_ddg/descriptions/descriptions.jsonl --model "$ZEN_MODEL" --concurrency 2
```

* **Resume**: run the same command again. Tables that already succeeded are skipped. Every completion is cached in
  `~/ntcir_ddg/descriptions/llm_cache.sqlite`, so even re-running a finished table costs no API calls.
* **Ctrl-C** finishes the tables in flight, then stops. Results are written as each table finishes.
* **Rate limits**: 429/5xx/network errors are retried with exponential backoff (honouring `Retry-After`). If the free
  quota is exhausted the run stops cleanly; re-run later. Keep `--concurrency` at 1–2 on the free tier.
* **Rerun specific tables**: `--ids id1,id2` (or `--ids @ids.txt`). `--retry-errors` also reruns tables whose last
  attempt failed for a non-transient reason.
* **One pinned model**: every row records `model`; the script refuses to append rows from a different model to the same
  file (use a different `--out` per model). Temperature is 0 (omitted only for models that reject it, and then recorded
  as `"omitted"`).
* Wide tables (> 40 columns) use semantic group prompts of 10 columns; if the model leaves columns out, the step is
  retried with groups of 5. Rows whose semantic profile is still incomplete carry a `warning`.

## 6. Copy the results back for notebook 02

Upload `~/ntcir_ddg/descriptions/descriptions.jsonl` to `MyDrive/ntcir_ddg/descriptions/` in the Drive web UI, or with
Google Drive for desktop:

```bash
mkdir -p ~/Library/CloudStorage/GoogleDrive-*/My\ Drive/ntcir_ddg/descriptions && cp ~/ntcir_ddg/descriptions/descriptions.jsonl ~/Library/CloudStorage/GoogleDrive-*/My\ Drive/ntcir_ddg/descriptions/
```

Then run `notebooks/02_eval_retrieval.ipynb` on Colab. To run it on the Mac instead:

```bash
uv pip install jupyter matplotlib
```

```bash
NTCIR_DDG_WORK_DIR=~/ntcir_ddg jupyter lab notebooks/02_eval_retrieval.ipynb
```

## Disk use

| What | Size |
|---|---|
| `.venv-mac` (AutoDDG + deps, Python 3.12) | ~330 MB |
| uv download cache (`uv cache clean` frees it) | a few hundred MB |
| `~/ntcir_ddg` (zip contents) | < 20 MB |
| `descriptions.jsonl` | ~5 MB |
| `llm_cache.sqlite` (prompts + answers) | ~50–150 MB |
| NLTK data (notebook 02 only) | ~40 MB |

No raw tables ever come to the Mac.

## Tests (no API calls)

`tests/test_pipeline.py` runs both scripts against a mock Zen server (`tests/mock_openai_server.py`) that injects
429s, 500s and empty answers. It checks caching, resume, backoff, the output format and the exit codes:

```bash
uv pip install -r tests/requirements-test.txt && python -m pytest -q tests/test_pipeline.py
```
