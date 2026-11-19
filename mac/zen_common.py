"""Shared code for the Mac-side LLM scripts (zen_smoke_test.py, generate_descriptions.py).

* Zen model listing (`GET /models`), free-model detection, capability checks
* `CachingLLMClient`: an `autoddg.llm.base.LLMClient` that wraps an `openai.OpenAI` client with an SQLite completion
  cache, exponential backoff on 429/5xx/network errors, and per-step usage/latency accounting
* `describe_table`: AutoDDG's quick-start flow (semantic profile -> topic -> UFD -> SFD) for one table record

How AutoDDG picks a client (checked against commit f286ac6):
* `AutoDDG.__init__` uses any `LLMClient` instance as is (`elif isinstance(client, LLMClient)`); anything else is
  wrapped in `OpenAICompatibleClient`. PyPI autoddg==0.1.2 lacks that branch, so `build_autoddg` also assigns the
  wrapper to every component explicitly; that works on both versions.
* `SemanticProfiler` only distinguishes `LocalLLMClient` (batch mode; multi-threading refused). Our wrapper is *not* a
  `LocalLLMClient`, so it gets the API code paths: group prompting and multi-threading both work; batch mode raises,
  exactly as for `OpenAICompatibleClient`.
* In group mode `SemanticProfiler._get_semantic_types_group` swallows every exception from the client and retries the
  missing columns up to 3 times. So this wrapper does its own retrying/backoff *inside* `chat_completions_create`, and
  `describe_table` checks how many columns actually got a semantic description.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import sqlite3
import threading
import time
import warnings
from contextlib import contextmanager
from typing import Any

warnings.filterwarnings("ignore", message="transformers and torch not available")

import httpx  # noqa: E402  (installed with openai)
import openai  # noqa: E402
import pandas as pd  # noqa: E402
from autoddg import AutoDDG  # noqa: E402
from autoddg.llm.base import LLMClient  # noqa: E402

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"

# exit codes shared by the scripts
EXIT_OK, EXIT_GENERIC, EXIT_AUTH, EXIT_MODEL, EXIT_RATE, EXIT_PARAM, EXIT_API, EXIT_PIPELINE, EXIT_NETWORK = (
    0, 1, 2, 3, 4, 5, 6, 7, 8)


# ----------------------------------------------------------------------------------------------------------- errors
class FatalAPIError(RuntimeError):
    """An error retrying cannot fix (auth, unknown model, bad parameter, exhausted free quota)."""

    def __init__(self, kind: str, message: str, exit_code: int) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind, self.exit_code = kind, exit_code


class RetriesExhausted(RuntimeError):
    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class EmptyCompletion(RuntimeError):
    pass


def _body_text(err: Exception) -> str:
    resp = getattr(err, "response", None)
    try:
        return resp.text if resp is not None else str(err)
    except Exception:  # noqa: BLE001
        return str(err)


def classify(err: Exception) -> tuple[str, int, bool]:
    """(kind, exit_code, retryable) for an exception raised by the openai client."""
    text = _body_text(err).lower()
    if isinstance(err, openai.AuthenticationError | openai.PermissionDeniedError):
        return "auth", EXIT_AUTH, False
    if isinstance(err, openai.NotFoundError):
        return "model_not_found", EXIT_MODEL, False
    if isinstance(err, openai.RateLimitError):
        if "freeusagelimit" in text or "usage limit" in text or "quota" in text or "credit" in text:
            return "quota_exhausted", EXIT_RATE, False
        return "rate_limited", EXIT_RATE, True
    if isinstance(err, openai.BadRequestError | openai.UnprocessableEntityError):
        if "temperature" in text or "unsupported" in text or "not supported" in text:
            return "unsupported_parameter", EXIT_PARAM, False
        if "model" in text and ("unavailable" in text or "not found" in text or "does not exist" in text):
            return "model_not_found", EXIT_MODEL, False
        return "bad_request", EXIT_API, False
    if isinstance(err, openai.APIStatusError):
        code = err.status_code
        if code in (408, 409, 425) or code >= 500:
            return f"http_{code}", EXIT_API, True
        return f"http_{code}", EXIT_API, False
    if isinstance(err, openai.APITimeoutError | openai.APIConnectionError):
        return "network", EXIT_NETWORK, True
    if isinstance(err, EmptyCompletion):
        return "empty_completion", EXIT_API, True
    return type(err).__name__, EXIT_GENERIC, False


def retry_after_seconds(err: Exception) -> float | None:
    resp = getattr(err, "response", None)
    if resp is None:
        return None
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------------------------- models
def fetch_models(base_url: str, api_key: str | None, timeout: float = 30.0) -> list[dict[str, Any]]:
    """Raw `GET {base_url}/models` (keeps Zen's extra fields: cost, capabilities, recommended_api)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    r = httpx.get(base_url.rstrip("/") + "/models", headers=headers, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("data", data) if isinstance(data, dict) else data


def _price(m: dict, side: str) -> float | None:
    try:
        return float(m["cost"]["text"][side]["price"])
    except (KeyError, TypeError, ValueError):
        pass
    for key in ("pricing", "price"):  # other OpenAI-compatible gateways
        p = m.get(key)
        if isinstance(p, dict):
            for k in (side, f"{side}_price", "prompt" if side == "input" else "completion"):
                try:
                    return float(p[k])
                except (KeyError, TypeError, ValueError):
                    continue
    return None


def is_free(m: dict) -> bool:
    if str(m.get("id", "")).endswith("-free"):
        return True
    pin, pout = _price(m, "input"), _price(m, "output")
    return pin == 0 and pout == 0


def chat_compatible(m: dict) -> bool:
    api = m.get("recommended_api")
    if api is None:
        return True
    return api == "chat-completions" or "chat-completions" in (m.get("api") or {})


def supports_temperature(m: dict | None) -> bool | None:
    if not m:
        return None
    cap = (m.get("capabilities") or {}).get("temperature")
    return cap if isinstance(cap, bool) else None


def pick_default_model(models: list[dict]) -> dict | None:
    free = [m for m in models if is_free(m) and chat_compatible(m)]
    free.sort(key=lambda m: (supports_temperature(m) is False,))  # prefer models that accept temperature
    return free[0] if free else None


# ----------------------------------------------------------------------------------------------------------- client
class CachingLLMClient(LLMClient):
    """LLMClient with an SQLite cache, retries with exponential backoff, and usage accounting.

    Cache key = sha256(model + messages + params actually sent). Only successful, non-empty completions are cached,
    so a crash or an error never poisons the cache and a re-run replays finished calls for free.
    """

    def __init__(self, client: Any, cache_path: str | None = None, *, send_temperature: bool = True,
                 max_retries: int = 8, backoff_base: float = 2.0, backoff_max: float = 120.0,
                 extra_params: dict[str, Any] | None = None, verbose: bool = True) -> None:
        self.client = client
        self.send_temperature = send_temperature
        self.max_retries, self.backoff_base, self.backoff_max = max_retries, backoff_base, backoff_max
        self.extra_params = dict(extra_params or {})
        self.verbose = verbose
        self._lock = threading.Lock()
        self._local = threading.local()
        self.stats = {"api_calls": 0, "cache_hits": 0, "retries": 0}
        self.db = None
        if cache_path:
            self.db = sqlite3.connect(cache_path, check_same_thread=False, isolation_level=None)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS completions (key TEXT PRIMARY KEY, model TEXT, "
                            "request TEXT, response TEXT, created REAL)")

    # -- usage accounting (per thread, per named step)
    @contextmanager
    def step(self, name: str):
        rec = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
               "latency_s": 0.0, "retries": 0}
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        stack.append(rec)
        t0 = time.time()
        try:
            yield rec
        finally:
            rec["wall_s"] = round(time.time() - t0, 3)
            rec["latency_s"] = round(rec["latency_s"], 3)
            stack.pop()

    def _record(self, usage: dict | None, latency: float, cached: bool, retries: int) -> None:
        for rec in getattr(self._local, "stack", None) or []:
            rec["calls"] += 1
            rec["cached_calls"] += int(cached)
            rec["latency_s"] += latency
            rec["retries"] += retries
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                rec[k] += int((usage or {}).get(k) or 0)

    # -- cache
    @staticmethod
    def cache_key(model: str, messages: list[dict[str, str]], params: dict[str, Any]) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True,
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _get(self, key: str) -> dict | None:
        if self.db is None:
            return None
        with self._lock:
            row = self.db.execute("SELECT response FROM completions WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key: str, model: str, request: dict, response: dict) -> None:
        if self.db is None:
            return
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO completions VALUES (?,?,?,?,?)",
                            (key, model, json.dumps(request, ensure_ascii=False),
                             json.dumps(response, ensure_ascii=False), time.time()))

    def is_cached(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0, **kwargs: Any) -> bool:
        return self._get(self.cache_key(model, messages, self._params(temperature, kwargs))) is not None

    def _params(self, temperature: float, kwargs: dict) -> dict:
        params = {**self.extra_params, **kwargs}
        if self.send_temperature:
            params["temperature"] = temperature
        return params

    # -- the LLMClient API
    def chat_completions_create(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0,
                                **kwargs: Any) -> dict[str, Any]:
        params = self._params(temperature, kwargs)
        key = self.cache_key(model, messages, params)
        hit = self._get(key)
        if hit is not None:
            with self._lock:
                self.stats["cache_hits"] += 1
            self._record(hit.get("usage"), 0.0, True, 0)
            return hit

        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                with self._lock:
                    self.stats["api_calls"] += 1
                resp = self.client.chat.completions.create(model=model, messages=messages, **params)
                latency = time.time() - t0
                content = resp.choices[0].message.content if resp.choices else None
                if not content or not content.strip():
                    raise EmptyCompletion("model returned an empty message (reasoning-only output?)")
                usage = None
                if getattr(resp, "usage", None):
                    usage = {"prompt_tokens": resp.usage.prompt_tokens,
                             "completion_tokens": resp.usage.completion_tokens,
                             "total_tokens": resp.usage.total_tokens}
                result = {"choices": [{"message": {"role": resp.choices[0].message.role or "assistant",
                                                   "content": content}}],
                          "usage": usage, "model": getattr(resp, "model", None) or model}
                self._put(key, model, {"messages": messages, "params": params}, result)
                self._record(usage, latency, False, attempt)
                return result
            except Exception as err:  # noqa: BLE001
                kind, code, retryable = classify(err)
                if not retryable:
                    raise FatalAPIError(kind, _body_text(err)[:500], code) from err
                last = err
                if attempt == self.max_retries:
                    break
                wait = retry_after_seconds(err)
                if wait is None:
                    wait = min(self.backoff_max, self.backoff_base * (2 ** attempt))
                    wait *= 0.5 + random.random()  # jitter
                with self._lock:
                    self.stats["retries"] += 1
                if self.verbose:
                    print(f"    [{kind}] retry {attempt + 1}/{self.max_retries} in {wait:.1f}s", flush=True)
                time.sleep(wait)
        kind, code, _ = classify(last)
        raise RetriesExhausted(f"{kind} after {self.max_retries} retries: {_body_text(last)[:300]}", code)


def make_openai_client(base_url: str, api_key: str | None, timeout: float = 300.0) -> openai.OpenAI:
    # retries are handled by CachingLLMClient (so they are visible and use Retry-After)
    return openai.OpenAI(base_url=base_url, api_key=api_key or "missing-key", max_retries=0, timeout=timeout)


def build_autoddg(llm: CachingLLMClient, model: str, description_words: int = 100) -> AutoDDG:
    pipe = AutoDDG(client=llm, model_name=model, description_words=description_words,
                   description_temperature=0.0, topic_temperature=0.0)
    for comp in (pipe.description_generator, pipe.semantic_profiler, pipe.topic_generator, pipe.search_description):
        comp.llm_client = llm  # see module docstring (needed on PyPI 0.1.2, harmless on the pinned commit)
    return pipe


# ----------------------------------------------------------------------------------------------------------- flow
def semantic_lines(details: str) -> int:
    return sum(1 for line in details.splitlines() if line.startswith("**"))


def describe_table(pipe: AutoDDG, llm: CachingLLMClient, table: dict, *, group_size: int = 0, wide_cols: int = 40,
                   wide_group_size: int = 10, fallback_group_size: int = 5) -> dict[str, Any]:
    """Quick-start flow for one tables.jsonl record. Raises on API failure."""
    sample_csv = table["sample_csv"]
    # rebuild the sample exactly as written (no NA conversion, no type inference)
    sample_df = pd.read_csv(io.StringIO(sample_csv), dtype=str, keep_default_na=False)
    n_cols = sample_df.shape[1]
    usage: dict[str, dict] = {}

    with llm.step("semantic") as rec:
        sizes = [group_size if n_cols <= wide_cols else wide_group_size]
        if sizes[0] == 0 or sizes[0] > fallback_group_size:
            sizes.append(fallback_group_size)
        best, best_cov, used = "", -1, None
        for gs in sizes:
            details = pipe.analyze_semantics(sample_df, use_group_prompting=True, group_size=gs)
            cov = semantic_lines(details)
            if cov > best_cov:
                best, best_cov, used = details, cov, gs
            if cov >= n_cols:
                break
    usage["semantic"] = rec
    semantic_details = best
    structural_profile = table.get("structural_profile") or ""
    semantic_profile = "\n".join(s for s in [structural_profile, semantic_details] if s)

    with llm.step("topic") as rec:
        topic = pipe.generate_topic(title=table.get("title") or "",
                                    original_description=table.get("original_description"),
                                    dataset_sample=sample_csv)
    usage["topic"] = rec

    with llm.step("ufd") as rec:
        _, ufd = pipe.describe_dataset(dataset_sample=sample_csv, dataset_profile=table.get("basic_profile"),
                                       use_profile=True, semantic_profile=semantic_profile, use_semantic_profile=True,
                                       data_topic=topic, use_topic=True)
    usage["ufd"] = rec

    with llm.step("sfd") as rec:
        _, sfd = pipe.expand_description_for_search(description=ufd, topic=topic)
    usage["sfd"] = rec

    total = {k: sum(u[k] for u in usage.values()) for k in
             ("calls", "cached_calls", "prompt_tokens", "completion_tokens", "total_tokens", "retries")}
    total["latency_s"] = round(sum(u["latency_s"] for u in usage.values()), 3)
    return {"topic": topic, "semantic_details": semantic_details, "semantic_profile": semantic_profile, "ufd": ufd,
            "sfd": sfd, "n_cols": n_cols, "group_size_used": used, "semantic_columns_covered": best_cov,
            "usage": usage, "usage_total": total}


def estimate_table(table: dict, group_size: int, wide_cols: int, wide_group_size: int,
                   description_words: int) -> tuple[int, int]:
    """Rough (calls, tokens) for one table; ~4 characters per token."""
    try:
        df = pd.read_csv(io.StringIO(table["sample_csv"]), dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001
        return 0, 0
    n_cols = df.shape[1]
    gs = group_size if n_cols <= wide_cols else wide_group_size
    sem_calls = 1 if gs == 0 else -(-n_cols // gs)
    head = df.head(5)
    col_chars = sum(len(str(c)) + sum(len(v) for v in head[c]) + 40 for c in df.columns)
    sample = len(table["sample_csv"])
    sem_in = sem_calls * 1800 + col_chars
    sem_out = n_cols * 220
    topic_in = 300 + len(table.get("title") or "") + len(table.get("original_description") or "") + sample
    ufd_in = 900 + sample + len(table.get("basic_profile") or "") + len(table.get("structural_profile") or "") + n_cols * 160
    ufd_out = int(description_words * 7)
    sfd_in = 3300 + ufd_out
    sfd_out = 3000
    chars = sem_in + sem_out + topic_in + 20 + ufd_in + ufd_out + sfd_in + sfd_out
    return sem_calls + 3, chars // 4
