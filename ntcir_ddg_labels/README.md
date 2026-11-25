# NTCIR-DDG relevance labels

37 search queries × 437 Data.gov datasets. Grades: **0** = not relevant, **1** = partially relevant, **2** = highly
relevant.

| File | Rows | What it is |
|---|---|---|
| `queries.csv` | 37 | `qid, query`: the original NTCIR queries |
| `datasets.csv` | 437 | `doc_id, title, description, …`: one row per dataset (columns below) |
| `human_labels.csv` | 695 | `qid, query, doc_id, title, grade`: the original NTCIR human grades |
| `llm_labels.csv` | 16,169 | `qid, query, doc_id, title, grade, reason`: our LLM grades for every query × dataset |
| `extra/` | | DSEBench's human grades (467 pairs), the LLM prompt, the DSEBench licence. You can ignore it |

The queries are always the original NTCIR queries. Only the labels in `llm_labels.csv` come from an LLM.

## 1. Where it comes from and what we changed

**The original data.** NTCIR-15 Data Search is a dataset-search benchmark: short queries (e.g. `births by month`) over
46,615 Data.gov datasets. Human assessors graded only a small pool of datasets per query. The AutoDDG paper built
"NTCIR-DDG" from it but did not release that benchmark.

**What we changed:**

1. **Rebuilt the benchmark.** We kept queries with at least 5 relevant and 10 graded datasets that have a CSV table:
   37 queries and 437 datasets. Each query's pool has 10–42 datasets (about 19 on average). Those 695 pairs are
   `human_labels.csv`.
2. **Labelled every pair with an LLM.** Claude Opus 5.5 graded all 37 × 437 pairs from each dataset's title,
   description, columns and sample rows; it never saw any generated description. That is `llm_labels.csv`, so
   retrieval can be scored over all 437 datasets, not only each query's pool.

**How good are the LLM labels?** They agree with NTCIR on 61% of pairs and with DSEBench's human grades on 76%; the two
human sources agree with each other on only 62%. The LLM is **stricter** than NTCIR: it often gives 0 where NTCIR gave
1, and 10 queries have no relevant dataset under the LLM labels.

## 2. How to use it

```python
import pandas as pd

queries = pd.read_csv("ntcir_ddg_labels/queries.csv")
datasets = pd.read_csv("ntcir_ddg_labels/datasets.csv")
human = pd.read_csv("ntcir_ddg_labels/human_labels.csv")
llm = pd.read_csv("ntcir_ddg_labels/llm_labels.csv")

# grades as {qid: {doc_id: grade}}
human_grades = {q: dict(zip(g.doc_id, g.grade)) for q, g in human.groupby("qid")}
llm_grades = {q: dict(zip(g.doc_id, g.grade)) for q, g in llm.groupby("qid")}
```

**Which labels to use:**

- **To compare with the AutoDDG paper:** for each query, rank only the datasets in `human_labels.csv` and score with
  those grades.
- **To evaluate over all 437 datasets:** rank all of them and score with `llm_labels.csv`.
- **Skip queries that have no relevant dataset** under the labels you use, or they pull every average down.

`datasets.csv` columns: `doc_id`, `title`, `description`, `publisher`, `url` (Data.gov page), `license`,
`attribution`, `table_url` (the CSV file we used), `n_rows`, `n_cols`, `table_header` (the table's column names, as a JSON list:
`json.loads(row.table_header)`).

## Credits and licences

- NTCIR queries and grades: CC BY 4.0, from <https://huggingface.co/datasets/mpkato/ntcir_data_search>. Cite Kato et
  al., *Overview of the NTCIR-15 Data Search Task*, NTCIR-15, 2020.
- DSEBench grades (`extra/`): Apache 2.0, from <https://zenodo.org/records/17805089>. We kept only our queries and
  datasets and took the majority of each pair's votes.
- Dataset metadata: each dataset's own licence, in the `license` and `attribution` columns.
- LLM labels: Team Epoch, made with Claude Opus 5.5 on 2026-11-22; CC BY 4.0. Regenerate this folder with
  `labeling/export_release.py`.
- AutoDDG: Zhang et al., *AutoDDG: Automated Dataset Description Generation using Large Language Models*, SIGMOD 2026,
  doi:10.1145/3786626.
