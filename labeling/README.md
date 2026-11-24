# LLM relevance labels (Claude subscription, no API key)

These scripts fill in the query–dataset pairs that NTCIR never judged, so retrieval can be scored over the whole corpus
instead of each query's ~19-dataset pool. The labeller reads only the original metadata and the data (profile, sample
rows, frequent values), never the generated UFD/SFD descriptions. It runs on headless Claude Code (`claude -p`), so
usage counts against your Claude plan's limits rather than paid API credits.

Each prompt holds all 37 queries and 10 dataset cards, about 18k tokens in all (numbers and CSV rows take more tokens
than plain text). 437 datasets come to 44 prompts and about 0.8M tokens.

All commands run from the repository root.

## 1. Log in once

```bash
claude auth login
```

```bash
claude auth status
```

The model below needs Claude Code 2.1.280 or newer (`claude update`).

## 2. Build the batches

Use the zip from notebook 01's merged run (`SHARD=None` after all four shards), so every card has the archive version
of the table and its frequent values:

```bash
python3 labeling/build_batches.py --source results/ntcir_ddg_intermediate.zip --out results/labels_final --include-unprofiled
```

Without `--include-unprofiled` only datasets with a profiled table are included; with it, the rest are labelled from
metadata alone (their cards say so). `--limit N` keeps the first N datasets.

## 3. Label

```bash
python3 labeling/run_labels.py --out results/labels_final --model claude-opus-5-5 --effort high
```

It prints the tokens each prompt used and projects the total. When a usage limit is hit the script stops; running the
same command again continues with the batches that have no result yet. `--max-batches N` stops after N prompts, for a
pilot. The model actually used is recorded per batch.

## 4. Compare with the human labels

```bash
python3 labeling/agreement.py --labels results/labels_final/llm_qrels.tsv --source results/ntcir_ddg_intermediate.zip --dsebench results/dsebench
```

DSEBench is at https://zenodo.org/records/17805089 (`DSEBench_data.zip`, 38 MB; unpacked into `results/dsebench/`). The
script prints LLM vs NTCIR, LLM vs DSEBench humans, and NTCIR vs DSEBench humans as the reference for how much two
groups of human assessors agree (about 62% relevant/not agreement, kappa 0.23).

## 5. Export for the team

```bash
python3 labeling/export_release.py --source results/ntcir_ddg_intermediate.zip --llm-labels results/labels_final/labels.jsonl --dsebench results/dsebench --out ntcir_ddg_labels
```

It rewrites the data files in `ntcir_ddg_labels/` (queries, datasets, all three label sources, TREC qrels) and checks
that the LLM covered every pair. The data card there (`README.md`) is kept by hand.

## Why Opus 5.5 with high effort (pilot)

Pilot on the first notebook-01 run (172 profiled tables, no frequent values), kept in `results/labels_opus55/`:

| Labeller | Tokens per prompt | Agreement with NTCIR, first batch |
|---|---|---|
| Sonnet 5 (default) | ~21k | 15 of 21 judged pairs |
| Opus 5, high | ~20k | 17 of 21 |
| Opus 5.5, high | ~19k | 15 of 21 |

The three models disagreed on only 4 of 370 pairs in that batch. Over all 172 tables (6,364 pairs, ~306k tokens, no
failed prompts), Opus 5.5 agreed with NTCIR on 61% of pairs (relevant/not, kappa 0.27) and with DSEBench's humans on 71%
(kappa 0.36), at least as well as the two human sources agree with each other. It is stricter than NTCIR: many NTCIR
grade-1 pairs it grades 0. It found 57 relevant pairs that NTCIR never judged.

## Outputs (`results/labels_final/`)

| File | Content |
|---|---|
| `batches/batch_NNNN.json` | the prompt and the handle → id maps for each batch |
| `raw/batch_NNNN.json` | parsed labels plus the full CLI result (token usage, model) |
| `labels.jsonl` | one row per query × dataset: grade 0/1/2, short reason, batch, model |
| `llm_qrels.tsv` | `qid, doc_id, grade` |
| `usage.tsv` | tokens per batch |
