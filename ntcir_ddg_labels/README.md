# NTCIR-DDG relevance labels

37 search queries × 437 Data.gov datasets, with relevance grades: **0** = not relevant, **1** = partially relevant,
**2** = highly relevant.

## 1. Where it comes from and what we changed

**The original data.** NTCIR-15 Data Search is a dataset-search benchmark: short queries (e.g. `births by month`) over
46,615 Data.gov datasets. Human assessors graded only a small pool of datasets per query (the ones that participating
systems returned). The AutoDDG paper built "NTCIR-DDG" from it but did not release that benchmark.

**What we changed:**

1. **Rebuilt the benchmark.** We kept queries with at least 5 relevant and 10 graded datasets that have a CSV table:
   37 queries and 437 datasets. Each query's pool has 10–42 datasets (about 19 on average). Only these 695
   query–dataset pairs have NTCIR grades.
2. **Labelled every other pair with an LLM.** Claude Opus 5.5 graded all 37 × 437 = 16,169 pairs from each dataset's
   title, description, columns and sample rows. It never saw any generated description. Now retrieval can be scored
   over all 437 datasets, not only each query's pool.
3. **Added DSEBench's human grades** where they exist: 467 pairs across 17 of our queries.

**How good are the LLM labels?** They agree with NTCIR on 61% of pairs and with DSEBench on 76%. The two human sources
agree with each other on only 62%. The LLM is **stricter** than NTCIR: it often gives 0 where NTCIR gave 1, and 10
queries have no relevant dataset under the LLM labels.

## 2. How to use it

The main file is **`judgments.tsv`**, one row per query × dataset:

| qid | doc_id | in_ntcir_pool | ntcir | dsebench | dsebench_votes | llm | llm_reason |
|---|---|---|---|---|---|---|---|
| DS1-E-0005 | 1560c507-… | 1 | 1 | 1 | 1 | 1 | US live births, but annual rather than monthly |
| DS1-E-0005 | 000f1c44-… | 0 | | | | 0 | |

An empty cell means "not graded by that source", not "not relevant". `in_ntcir_pool = 1` marks the pairs NTCIR
graded.

```python
import pandas as pd

j = pd.read_csv("ntcir_ddg_labels/judgments.tsv", sep="\t", dtype={"ntcir": "Int64", "dsebench": "Int64"},
                keep_default_na=False, na_values={"ntcir": [""], "dsebench": [""]})
queries = pd.read_csv("ntcir_ddg_labels/queries.tsv", sep="\t")          # qid, query
datasets = pd.read_json("ntcir_ddg_labels/datasets.jsonl", lines=True)   # id, title, description, columns, ...

# grades as {qid: {doc_id: grade}}
llm = {q: dict(zip(g.doc_id, g.llm)) for q, g in j.groupby("qid")}                 # all 437 datasets per query
pool = j[j.in_ntcir_pool == 1]
ntcir = {q: dict(zip(g.doc_id, g.ntcir)) for q, g in pool.groupby("qid")}         # NTCIR pool only
```

**Which labels to use:**

- **To compare with the AutoDDG paper:** rank each query's pool (`in_ntcir_pool == 1`) and score with `ntcir`.
- **To evaluate over all 437 datasets:** score with `llm`.
- **Skip queries that have no relevant dataset** under the labels you use, or they pull every average down.

`qrels/` has the same grades as TREC qrels files (`qid 0 doc_id grade`), for `trec_eval`, `pytrec_eval` or
`ir_measures`. `evaluation/open_corpus.py` in this repository runs the whole evaluation.

Other files: `llm_labelling_prompt.md` is the prompt the LLM was given; `labeling/export_release.py` regenerates this
folder.

## Credits and licences

- NTCIR queries and grades: CC BY 4.0, from <https://huggingface.co/datasets/mpkato/ntcir_data_search>. Cite Kato et
  al., *Overview of the NTCIR-15 Data Search Task*, NTCIR-15, 2020.
- DSEBench grades: Apache 2.0 (`licenses/DSEBench-Apache-2.0.txt`), from <https://zenodo.org/records/17805089>. We kept
  only our queries and datasets and took the majority grade per pair.
- Dataset metadata: each dataset's own licence, in the `license` and `attribution` fields of `datasets.jsonl`.
- LLM labels: Team Epoch, made with Claude Opus 5.5 on 2026-11-22; CC BY 4.0.
- AutoDDG: Zhang et al., *AutoDDG: Automated Dataset Description Generation using Large Language Models*, SIGMOD 2026,
  doi:10.1145/3786626.
