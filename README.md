# Team-Epoch

**Project No.: 2 — Dataset Description Generation**

## Group Members

1. **Riyansh Sachdev** — `rs10248` — [rs10248@nyu.edu](mailto:rs10248@nyu.edu)
2. **Prateek Saxena** — `ps5838` — [ps5838@nyu.edu](mailto:ps5838@nyu.edu)
3. **Yu-Fei Shih** — `ys7237` — [ys7237@nyu.edu](mailto:ys7237@nyu.edu)
4. **Zachary Wang** — `zw5361` — [zw5361@nyu.edu](mailto:zw5361@nyu.edu)

## About

LLM-powered automatic generation of concise and informative descriptions for tabular datasets.

**Team Epoch** 

## Reproducing AutoDDG on NTCIR-DDG

The AutoDDG paper's NTCIR-DDG benchmark was never released, so we rebuild it from the public NTCIR-15 Data Search
collection ([`mpkato/ntcir_data_search`](https://huggingface.co/datasets/mpkato/ntcir_data_search)).

| Step | Where | File |
|---|---|---|
| Explore data, build the benchmark, acquire + profile tables (no LLM) | Colab | [`notebooks/01_build_ntcir_ddg.ipynb`](notebooks/01_build_ntcir_ddg.ipynb) |
| Generate UFD/SFD descriptions with OpenCode Zen | Mac | [`mac/README.md`](mac/README.md) |
| BM25 retrieval evaluation (NDCG@5/10/15/20) | Colab or Mac | [`notebooks/02_eval_retrieval.ipynb`](notebooks/02_eval_retrieval.ipynb) |

AutoDDG is pinned to GitHub commit `f286ac6` everywhere (PyPI 0.1.2 is older). Every code path is tested offline on a
synthetic fixture and a mock LLM server: `python -m pytest -q tests/test_pipeline.py`.
