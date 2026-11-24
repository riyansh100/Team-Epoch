# NTCIR-DDG relevance labels

Relevance labels for **37 NTCIR-15 Data Search queries × 437 Data.gov datasets with CSV tables** (16,169 pairs), from
three sources:

| Source | Who judged | Pairs | Grades 0 / 1 / 2 | Coverage |
|---|---|---|---|---|
| `ntcir` | NTCIR-15 human assessors | 695 | 326 / 326 / 43 | each query's pool only (10–42 datasets, 18.8 on average) |
| `dsebench` | DSEBench human annotators | 467 | 299 / 121 / 47 | 17 of the 37 queries; 269 pairs outside the NTCIR pools |
| `llm` | Claude Opus 5.5 (`claude-opus-5-5`, high effort) | 16,169 | 15,975 / 174 / 20 | every query × every dataset |

Grades in all three: **0** = not relevant, **1** = partially relevant, **2** = highly relevant.

This is the benchmark we rebuilt to reproduce AutoDDG (Zhang et al., SIGMOD 2026), whose NTCIR-DDG benchmark was not
released. NTCIR only judged each query's pool, so the paper's evaluation ranks about 19 datasets per query. The LLM
labels fill in every other pair, so retrieval can also be scored over all 437 datasets (our "open" setting).

## Files

| File | Rows | Content |
|---|---|---|
| `queries.tsv` | 37 | `qid`, `query` (with header) |
| `datasets.jsonl` | 437 | one dataset per line; fields below |
| `judgments.tsv` | 16,169 | one row per query × dataset, all sources side by side; columns below |
| `qrels/ntcir.txt` | 695 | TREC qrels: `qid 0 doc_id grade`, space-separated, no header |
| `qrels/dsebench.txt` | 467 | same format |
| `qrels/llm_opus55.txt` | 16,169 | same format |
| `llm_labelling_prompt.md` | | the system prompt the LLM was given |
| `licenses/DSEBench-Apache-2.0.txt` | | licence of the DSEBench judgements |

The `qrels/` files work directly with `trec_eval`, `pytrec_eval` and `ir_measures`.

### `judgments.tsv`

| Column | Meaning |
|---|---|
| `qid` | NTCIR query id, e.g. `DS1-E-0005` |
| `doc_id` | Data.gov dataset id (the NTCIR collection id) |
| `in_ntcir_pool` | `1` if NTCIR judged this pair, else `0` |
| `ntcir` | NTCIR grade; empty when not judged |
| `dsebench` | DSEBench grade (majority of `dsebench_votes`, ties go to the lower grade); empty when not judged |
| `dsebench_votes` | all DSEBench `query_rel` judgements for this pair, comma-separated |
| `llm` | LLM grade; never empty |
| `llm_reason` | the LLM's reason (12 words or fewer), given for grades 1 and 2 only |

An empty cell means "not judged", **not** "not relevant".

### `datasets.jsonl`

| Field | Meaning |
|---|---|
| `id`, `title`, `description`, `publisher`, `url` | original Data.gov metadata from the NTCIR collection |
| `license`, `attribution` | the dataset's licence and the attribution line NTCIR supplies |
| `n_csv_files` | number of CSV files the dataset has |
| `profiled` | whether we read and profiled a table (true for all 437) |
| `table_file`, `table_url` | the CSV used for this dataset (the first CSV file) and where it came from |
| `n_rows`, `n_cols`, `columns` | size and header of that table |

## Loading

```python
import pandas as pd

j = pd.read_csv("ntcir_ddg_labels/judgments.tsv", sep="\t", dtype={"ntcir": "Int64", "dsebench": "Int64"},
                keep_default_na=False, na_values={"ntcir": [""], "dsebench": [""]})
queries = pd.read_csv("ntcir_ddg_labels/queries.tsv", sep="\t")
datasets = pd.read_json("ntcir_ddg_labels/datasets.jsonl", lines=True)

pool = j[j.in_ntcir_pool == 1]          # the paper's setting: rank each query's pool
qrels = {q: dict(zip(g.doc_id, g.llm)) for q, g in j.groupby("qid")}   # open setting, LLM labels
```

```python
import ir_measures
from ir_measures import nDCG

qrels = list(ir_measures.read_trec_qrels("ntcir_ddg_labels/qrels/llm_opus55.txt"))
has_rel = {q.query_id for q in qrels if q.relevance > 0}
qrels = [q for q in qrels if q.query_id in has_rel]   # drop queries with no relevant dataset (see Label quality)
# run = {qid: {doc_id: score}}
ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
```

`evaluation/open_corpus.py` in this repository runs the full evaluation: pool and open settings, with the `human`,
`human+llm` and `llm` label sets.

## How the benchmark was built

- **Queries:** NTCIR-15 Data Search English training and test topics (`DS1-E-*`) whose judged pools contain at least
  5 relevant and 10 judged datasets with a CSV table (MIME type `text/csv`). The paper used 32 queries; this rule gives
  37.
- **Datasets:** every dataset in those pools, 437 in all. Datasets with several CSV files are represented by the first
  one.
- **NTCIR grades:** `L0/L1/L2` mapped to 0/1/2. Built by `notebooks/01_build_ntcir_ddg.ipynb`.
- **DSEBench grades:** DSEBench's 169 NTCIR queries include all 37 of ours, matched by query text; only pairs whose
  dataset is among the 437 are kept, and 749 other pairs are dropped. DSEBench judges a candidate once per case (a
  query plus a target dataset), so one pair can have several judgements; `dsebench_votes` lists them all. Its
  annotators read metadata fields (title, description, tags, author, summary), not table contents.
- **LLM grades:** `labeling/build_batches.py` and `labeling/run_labels.py`, run 2026-11-22 through headless Claude
  Code 2.1.330 (`claude -p --model claude-opus-5-5 --effort high`). Each prompt held all 37 queries and 10 dataset
  cards: 44 prompts, ~0.93M tokens, no failures. A card has the original title, description and publisher, the
  table's size and columns, a data profile, 5 sample rows and frequent values. The LLM never saw any generated
  description (UFD/SFD) or any other label source. The prompt asked for grades 1 and 2 only; every pair left out
  counts as 0.
- **Export:** `labeling/export_release.py` writes every file except this README, the prompt copy and `licenses/`.

## Label quality

Relevant (1 or 2) vs not relevant, on the pairs both sources judged:

| Comparison | Pairs | Agreement | Cohen's kappa |
|---|---|---|---|
| LLM vs NTCIR | 695 | 61% | 0.25 |
| LLM vs DSEBench | 467 | 76% | 0.40 |
| NTCIR vs DSEBench (human vs human) | 198 | 62% | 0.23 |

The LLM agrees with each human source at least as well as the two human sources agree with each other. Things to
know before using the labels:

- **The LLM is much stricter than NTCIR.** It grades 0 for 240 NTCIR grade-1 pairs and 22 NTCIR grade-2 pairs; spot
  checks suggest NTCIR is the lenient one. 10 of the 37 queries have no LLM-relevant dataset at all (e.g.
  `trend number police officer 1945`). `ir_measures` scores such queries 0 and keeps them in the average (a perfect
  ranking gets NDCG@10 0.73 on `llm_opus55.txt`); drop them before averaging, as `evaluation/open_corpus.py` does.
  DSEBench has one such query among its 17.
- **The labels find datasets NTCIR missed:** 76 LLM-relevant pairs outside the NTCIR pools, across 19 queries (DSEBench
  humans add 66 relevant pairs outside the pools).
- **Do not mix standards silently.** "NTCIR inside the pools, LLM outside" combines lenient and strict judgements. Use
  the LLM labels alone for a single consistent standard, and the human labels as robustness checks.
- **One run, one model.** A rerun would not give exactly the same grades; treat these files as the reference copy.
- **This is a subset.** It covers 37 queries and 437 CSV datasets, not the full NTCIR-15 collection of 46,615
  datasets.

## Licences and citation

- **NTCIR queries and relevance judgements:** CC BY 4.0, from the NTCIR Data Search Task test collection
  (<https://huggingface.co/datasets/mpkato/ntcir_data_search>). The authors ask to be credited by citing their papers
  (listed at <https://ntcir.datasearch.jp/>), e.g. Kato, Ohshima, Liu and Chen, *Overview of the NTCIR-15 Data Search
  Task*, Proceedings of NTCIR-15, 2020.
- **Dataset metadata** (`datasets.jsonl`): each dataset keeps its own licence, given in its `license` and `attribution`
  fields: U.S. Government Work (148), CC0 1.0 (140), CC BY 3.0 (77), PDDL (61) or CC BY 4.0 (11).
- **DSEBench judgements:** Apache License 2.0 (`licenses/DSEBench-Apache-2.0.txt`), from the DSEBench test collection
  (<https://zenodo.org/records/17805089>; see the record for its paper). We changed them as described above: restricted
  to our queries and datasets, and reduced to a majority grade per pair.
- **LLM labels:** produced by Team Epoch with Claude Opus 5.5; released under CC BY 4.0.
- **AutoDDG:** Haoxiang Zhang, Yurong Liu, Aécio Santos, Wei-Lun (Allen) Hung and Juliana Freire. *AutoDDG: Automated
  Dataset Description Generation using Large Language Models.* Proc. ACM Manag. Data 4(1), Article 12, 2026.
  doi:10.1145/3786626.
