"""End-to-end tests: notebooks 01/02 on the synthetic fixture, Mac scripts against a mock Zen server.

    pip install -r requirements-mac.txt -r tests/requirements-test.txt
    pytest -q tests/test_pipeline.py

Nothing here touches Hugging Face, Google Drive or the real Zen API.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import make_fixture  # noqa: E402
import mock_openai_server  # noqa: E402

PY = sys.executable
MOCK_PORT = 18766
MOCK_URL = f"http://127.0.0.1:{MOCK_PORT}/v1"
MANUAL = {"topic_set": "ntcir16_train+test", "test_qrels": "test", "rel_min": 1, "csv_mode": "mime"}


def run_notebook(name: str, out: Path, env: dict) -> str:
    r = subprocess.run([PY, str(ROOT / "tests" / "run_notebook.py"), str(ROOT / "notebooks" / name), str(out)],
                       env={**os.environ, **env}, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-3000:]
    return r.stdout


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_by_id(rows):
    out = {}
    for r in rows:
        out[r["id"]] = r
    return out


# ------------------------------------------------------------------------------------------------ notebook 01 / 02
@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fixture")
    meta = make_fixture.build(d)
    return d, meta


def nb01_env(fixture, work, raw, **overrides):
    return {"NTCIR_DDG_FIXTURE": "1", "NTCIR_DDG_FIXTURE_DIR": str(fixture), "NTCIR_DDG_WORK_DIR": str(work),
            "NTCIR_DDG_RAW_DIR": str(raw), "NTCIR_DDG_OVERRIDES": json.dumps(overrides)}


@pytest.fixture(scope="session")
def nb01_first(fixture_dir, tmp_path_factory):
    fx, meta = fixture_dir
    base = tmp_path_factory.mktemp("nb01")
    env = nb01_env(fx, base / "work", base / "raw", AUTO_SELECT_CONFIG=False, MANUAL_CONFIG=MANUAL)
    out1 = run_notebook("01_build_ntcir_ddg.ipynb", base / "run1.ipynb", env)
    tables_after_1 = jsonl(base / "work" / "intermediate" / "tables.jsonl")
    out2 = run_notebook("01_build_ntcir_ddg.ipynb", base / "run2.ipynb", env)
    return {"base": base, "work": base / "work", "out1": out1, "out2": out2, "meta": meta,
            "rows_after_1": len(tables_after_1)}


def test_nb01_edge_cases(nb01_first):
    meta, work = nb01_first["meta"], nb01_first["work"]
    tables = last_by_id(jsonl(work / "intermediate" / "tables.jsonl"))
    ids = meta["ids"]
    t = lambda name: tables[ids[name]]  # noqa: E731
    assert t("latin1")["encoding"] == "latin-1" and not t("latin1")["error"]
    assert "Montréal" in t("latin1")["sample_csv"]
    assert t("semicolon")["delimiter"] == ";" and t("semicolon")["n_cols"] == 4
    assert any("malformed" in n for n in t("malformed")["read_notes"]) and t("malformed")["n_rows"] == 20
    assert t("bom")["encoding"] == "utf-8-sig" and t("bom")["columns"][0] == "incident_id"
    assert t("quotednl")["n_rows"] == 29
    big = t("large")
    assert big["streamed"] and big["n_rows"] == 12000 and "Rows: 12000" in big["basic_profile"]
    assert t("htmlpage")["error"].startswith("html")
    assert t("empty")["error"].startswith("empty")
    assert t("missing")["error"].startswith("not_acquired")
    assert not t("slowurl")["error"] and t("slowurl")["file_source"] == "archive"
    m = t("multi")
    assert not m["error"] and m["n_csv_files"] == 2 and m["n_rows"] == 10  # policy "first"
    # every successful table carries the fields the Mac script needs
    for r in tables.values():
        if not r["error"]:
            assert r["sample_csv"] and r["basic_profile"].startswith("The key data profile") and r["n_cols"] >= 3
            assert r["structural_profile"] is not None


def test_nb01_outputs_and_acquisition(nb01_first):
    work, out1 = nb01_first["work"], nb01_first["out1"]
    bench = work / "benchmark"
    for f in ["queries.tsv", "qrels.tsv", "pools.json", "docs.jsonl", "needed_files.txt", "needed_files.tsv",
              "config.json", "config_grid.csv", "queries_acquired.tsv", "qrels_acquired.tsv", "pools_acquired.json",
              "coverage.tsv"]:
        assert (bench / f).exists(), f
    rels = {line.split("\t")[2].strip() for line in (bench / "qrels.tsv").read_text().splitlines()}
    assert rels <= {"0", "1", "2"}
    assert json.loads((bench / "config.json").read_text())["chosen"] == MANUAL
    assert "simulated connection drop" not in out1  # the drop is logged as a resume, not an error
    assert "resuming with a Range request" in out1
    assert "liveness on" in out1 and "failure reasons (this run)" in out1
    manifest = jsonl(work / "acquisition" / "manifest.jsonl")
    assert {r["source"] for r in manifest} == {"url", "archive"}
    fails = (work / "acquisition" / "url_failures.tsv").read_text()
    assert "http_404" in fails and "html_response" in fails and "ReadTimeout" in fails
    state = json.loads((work / "acquisition" / "archive_pass.json").read_text())
    assert len(state["not_found"]) == 1
    assert (work / "ntcir_ddg_intermediate.zip").stat().st_size < 5_000_000


def test_nb01_is_idempotent(nb01_first):
    work, out2 = nb01_first["work"], nb01_first["out2"]
    rows = jsonl(work / "intermediate" / "tables.jsonl")
    # second run: only the not-acquired table is retried; nothing re-downloaded or re-streamed
    assert len(rows) == nb01_first["rows_after_1"] + 1
    assert "files to take from the archive: 0" in out2
    assert "profiling 1 tables" in out2


def test_nb01_concat_and_url_only(fixture_dir, tmp_path):
    fx, meta = fixture_dir
    env = nb01_env(fx, tmp_path / "work", tmp_path / "raw", AUTO_SELECT_CONFIG=False, MANUAL_CONFIG=MANUAL,
                   MULTI_FILE_POLICY="concat", ARCHIVE_EXTRACT="missing")
    run_notebook("01_build_ntcir_ddg.ipynb", tmp_path / "run.ipynb", env)
    tables = last_by_id(jsonl(tmp_path / "work" / "intermediate" / "tables.jsonl"))
    multi = tables[meta["ids"]["multi"]]
    assert multi["n_rows"] == 50 and multi["file_used"].count(";") == 1
    sources = {s for r in tables.values() if not r["error"] for s in r["file_source"].split(";")}
    assert sources == {"url", "archive"}
    assert any("live source URL" in n for r in tables.values() for n in r["read_notes"])


def test_nb01_auto_config_runs(fixture_dir, tmp_path):
    fx, _ = fixture_dir
    out = run_notebook("01_build_ntcir_ddg.ipynb", tmp_path / "run.ipynb",
                       nb01_env(fx, tmp_path / "work", tmp_path / "raw"))
    assert "closest to the paper" in out and "(auto)" in out


# ------------------------------------------------------------------------------------------------ Mac scripts
@pytest.fixture(scope="session")
def mock():
    srv = mock_openai_server.serve(MOCK_PORT)
    yield MOCK_URL
    srv.shutdown()


def reset(**kw):
    httpx.post(f"http://127.0.0.1:{MOCK_PORT}/reset", json=kw).raise_for_status()


def stats():
    return httpx.get(f"http://127.0.0.1:{MOCK_PORT}/stats").json()


def mac(script, *args, key="test-key", base=MOCK_URL, timeout=300):
    env = {**os.environ, "ZEN_BASE_URL": base, "XDG_CACHE_HOME": str(ROOT / "tests" / ".cache")}
    env.pop("OPENCODE_API_KEY", None)
    if key:
        env["OPENCODE_API_KEY"] = key
    return subprocess.run([PY, str(ROOT / "mac" / script), *args], env=env, capture_output=True, text=True,
                          timeout=timeout)


@pytest.mark.parametrize("args,key,code", [
    ([], "test-key", 0),
    (["--skip-autoddg"], None, 2),
    (["--skip-autoddg"], "wrong-key", 2),
    (["--model", "no-such-model"], "test-key", 3),
    (["--model", "mock-messages-free"], "test-key", 3),
    (["--model", "mock-notemp-free", "--temperature", "send", "--skip-autoddg"], "test-key", 5),
    (["--model", "mock-notemp-free"], "test-key", 0),
])
def test_smoke_test_exit_codes(mock, nb01_first, args, key, code):
    reset()
    tables = nb01_first["work"] / "intermediate" / "tables.jsonl"
    r = mac("zen_smoke_test.py", "--tables", str(tables), *args, key=key)
    assert r.returncode == code, r.stdout[-3000:] + r.stderr[-2000:]
    if code == 0:
        assert "OK:" in r.stdout and "--- SFD" in r.stdout and "TOTAL" in r.stdout


def test_smoke_test_bundled_example(mock):
    reset()
    r = mac("zen_smoke_test.py")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "Renal" in r.stdout or "built-in" in r.stdout


@pytest.fixture(scope="session")
def generated(mock, nb01_first, tmp_path_factory):
    d = tmp_path_factory.mktemp("gen")
    tables = nb01_first["work"] / "intermediate" / "tables.jsonl"
    out = d / "descriptions.jsonl"
    common = ["--tables", str(tables), "--out", str(out), "--model", "mock-free", "--yes"]
    res = {}
    # pilot with injected 429s, 500s and an empty completion: must still succeed
    reset(fail_429=3, fail_500=2, empty=1)
    res["pilot"] = mac("generate_descriptions.py", *common, "--limit", "3")
    res["pilot_stats"] = stats()
    reset()
    res["full"] = mac("generate_descriptions.py", *common, "--concurrency", "2")
    res["full_stats"] = stats()
    reset()
    res["noop"] = mac("generate_descriptions.py", *common)
    res["noop_stats"] = stats()
    # lose the output file entirely: the SQLite cache replays every call
    lost = d / "descriptions_lost.jsonl"
    reset()
    res["replay"] = mac("generate_descriptions.py", "--tables", str(tables), "--out", str(lost), "--model",
                        "mock-free", "--cache", str(d / "llm_cache.sqlite"), "--yes")
    res["replay_stats"] = stats()
    return {"dir": d, "out": out, "lost": lost, "tables": tables, **res}


def test_generate_output_format(generated):
    assert generated["pilot"].returncode == 0, generated["pilot"].stdout[-3000:] + generated["pilot"].stderr[-2000:]
    assert generated["full"].returncode == 0, generated["full"].stdout[-3000:]
    rows = jsonl(generated["out"])
    ok_tables = [t for t in jsonl(generated["tables"]) if not t.get("error")]
    ok_ids = {t["id"] for t in ok_tables}
    assert {r["id"] for r in rows} == ok_ids and len(rows) == len(ok_ids)  # each table exactly once
    for r in rows:
        assert not r["error"]
        assert r["model"] == "mock-free" and r["temperature"] == 0.0
        for k in ("topic", "semantic_profile", "ufd", "sfd", "usage", "usage_total", "latency_s", "timestamp"):
            assert r[k] not in (None, ""), k
        assert set(r["usage"]) == {"semantic", "topic", "ufd", "sfd"}
        assert r["semantic_columns_covered"] == r["n_cols"]
        assert "The key semantic information" in r["semantic_profile"]


def test_generate_retries_and_plan(generated):
    out = generated["pilot"].stdout
    assert "PLAN: 3 tables" in out and "LLM calls" in out and "M tokens" in out
    assert "[rate_limited] retry" in out and "[http_500] retry" in out and "[empty_completion] retry" in out
    assert generated["pilot_stats"]["chat_requests"] == 3 * 4 + 3 + 2 + 1


def test_generate_resume_and_cache(generated):
    n_ok = len([t for t in jsonl(generated["tables"]) if not t.get("error")])
    assert f"PLAN: {n_ok - 3} tables" in generated["full"].stdout  # pilot tables skipped
    assert generated["full_stats"]["chat_requests"] == (n_ok - 3) * 4
    assert "PLAN: 0 tables" in generated["noop"].stdout and generated["noop_stats"]["chat_requests"] == 0
    assert generated["replay"].returncode == 0
    assert generated["replay_stats"]["chat_requests"] == 0  # everything served from the cache
    assert len(jsonl(generated["lost"])) == n_ok


def test_generate_ids_rerun_and_model_pin(generated, mock):
    one = jsonl(generated["out"])[0]["id"]
    reset()
    r = mac("generate_descriptions.py", "--tables", str(generated["tables"]), "--out", str(generated["out"]),
            "--model", "mock-free", "--ids", one, "--yes")
    assert r.returncode == 0 and "PLAN: 1 tables" in r.stdout
    assert stats()["chat_requests"] == 0  # cached
    r = mac("generate_descriptions.py", "--tables", str(generated["tables"]), "--out", str(generated["out"]),
            "--model", "mock-notemp-free", "--yes")
    assert r.returncode == 1 and "one pinned model" in r.stdout


def test_generate_fatal_errors(generated, mock, tmp_path):
    reset()
    common = ["--tables", str(generated["tables"]), "--model", "mock-free", "--yes"]
    r = mac("generate_descriptions.py", *common, "--out", str(tmp_path / "a.jsonl"), key="wrong-key")
    assert r.returncode == 2 and "STOPPED on a fatal API error (auth)" in r.stdout
    rows = jsonl(tmp_path / "a.jsonl")
    assert rows and all(x["error_kind"] == "auth" for x in rows)
    # a model that rejects temperature: auto mode omits it and succeeds
    r = mac("generate_descriptions.py", "--tables", str(generated["tables"]), "--model", "mock-notemp-free",
            "--out", str(tmp_path / "b.jsonl"), "--limit", "2", "--yes")
    assert r.returncode == 0, r.stdout[-2000:]
    assert all(x["temperature"] == "omitted" for x in jsonl(tmp_path / "b.jsonl"))
    r = mac("generate_descriptions.py", "--tables", str(generated["tables"]), "--model", "mock-notemp-free",
            "--out", str(tmp_path / "c.jsonl"), "--limit", "2", "--temperature", "send", "--yes")
    assert r.returncode == 5


def test_nb02_with_descriptions(nb01_first, generated, tmp_path):
    out = run_notebook("02_eval_retrieval.ipynb", tmp_path / "nb02.ipynb",
                       {"NTCIR_DDG_WORK_DIR": str(nb01_first["work"]),
                        "NTCIR_DDG_DESCRIPTIONS": str(generated["out"])})
    assert "models: {'mock-free'" in out
    import pandas as pd

    res = nb01_first["work"] / "results"
    summary = pd.read_csv(res / "summary_acquired_mock-free.csv")
    assert set(summary.variant) == {"Original", "UFD", "SFD", "random ranking"}
    assert summary["ndcg@20"].between(0, 1).all()
    per_q = pd.read_csv(res / "per_query_acquired_mock-free.csv")
    assert len(per_q) == 3 * 2 * per_q.qid.nunique()


def test_nb02_original_only_before_any_llm(nb01_first, tmp_path):
    out = run_notebook("02_eval_retrieval.ipynb", tmp_path / "nb02.ipynb",
                       {"NTCIR_DDG_WORK_DIR": str(nb01_first["work"]),
                        "NTCIR_DDG_DESCRIPTIONS": str(tmp_path / "none.jsonl"),
                        "NTCIR_DDG_OVERRIDES": json.dumps({"POOL_SET": "planned"})})
    assert "evaluating Original only" in out and "benchmark: planned" in out
