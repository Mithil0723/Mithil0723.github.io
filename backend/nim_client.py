"""
nim_client.py — Shared NVIDIA NIM API client for embeddings and reranking.

Handles:
- Embedding via /v1/embeddings (with input_type for asymmetric encoding)
- Reranking via /v1/ranking
- Exponential backoff with jitter on 429/5xx
- Explicit timeouts
- Latency and status logging

All model names and the base URL are env-driven.

Task 1 (embeddings), Task 2 (reranker), Task 6 (network resilience).
"""

import os
import time
import random
import logging
import httpx
import numpy as np

logger = logging.getLogger(__name__)

NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

EMBED_MODEL = os.getenv("EMBED_MODEL", "nvidia/nv-embed-v1")
RERANK_MODEL = os.getenv("RERANK_MODEL", "nvidia/nv-rerankqa-mistral-4b-v3")

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0
JITTER_MAX = 0.5  # seconds


def _get_api_key() -> str:
    """Return the NVIDIA API key, reading from env each time for lazy-load safety."""
    key = os.getenv("NVIDIA_API_KEY", "")
    if not key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set. Get one from https://build.nvidia.com"
        )
    return key


def _should_retry(status_code: int) -> bool:
    """Return True for retryable HTTP status codes."""
    return status_code == 429 or status_code >= 500


def _request_with_retry(
    method: str,
    url: str,
    *,
    json_body: dict,
    timeout: float,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """
    Make an HTTP request with exponential backoff on 429/5xx.
    Returns the parsed JSON response.
    Raises RuntimeError on terminal failure.
    """
    api_key = _get_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        start_time = time.monotonic()
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method, url, json=json_body, headers=headers
                )
            elapsed = time.monotonic() - start_time

            logger.info(
                f"NIM {url.split('/')[-1]} status={response.status_code} "
                f"latency={elapsed:.2f}s attempt={attempt}/{max_retries}"
            )

            if response.status_code == 403:
                # Account entitlement issue — do not retry
                logger.error(
                    "NIM API returned 403 Forbidden. This is likely an account "
                    "entitlement issue (missing 'Public API Endpoints' permission), "
                    "not a code bug. Check your NVIDIA API key at build.nvidia.com."
                )
                raise RuntimeError(
                    f"NIM API 403 Forbidden: {response.text}"
                )

            if _should_retry(response.status_code) and attempt < max_retries:
                backoff = (
                    INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** (attempt - 1))
                    + random.uniform(0, JITTER_MAX)
                )
                logger.warning(
                    f"NIM API returned {response.status_code}, "
                    f"retrying in {backoff:.1f}s (attempt {attempt}/{max_retries})"
                )
                time.sleep(backoff)
                last_error = RuntimeError(
                    f"NIM API {response.status_code}: {response.text}"
                )
                continue

            response.raise_for_status()
            return response.json()

        except httpx.TimeoutException as e:
            elapsed = time.monotonic() - start_time
            logger.warning(
                f"NIM request timed out after {elapsed:.2f}s "
                f"(attempt {attempt}/{max_retries}): {e}"
            )
            last_error = e
            if attempt < max_retries:
                backoff = (
                    INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** (attempt - 1))
                    + random.uniform(0, JITTER_MAX)
                )
                time.sleep(backoff)
                continue

        except httpx.HTTPStatusError as e:
            last_error = e
            if attempt < max_retries and _should_retry(e.response.status_code):
                backoff = (
                    INITIAL_BACKOFF * (BACKOFF_MULTIPLIER ** (attempt - 1))
                    + random.uniform(0, JITTER_MAX)
                )
                logger.warning(
                    f"NIM API error {e.response.status_code}, "
                    f"retrying in {backoff:.1f}s"
                )
                time.sleep(backoff)
                continue
            raise RuntimeError(f"NIM API error: {e}") from e

    raise RuntimeError(
        f"NIM API failed after {max_retries} attempts: {last_error}"
    )


# ─────────────────────────────────────────────
# Embeddings — Task 1
# ─────────────────────────────────────────────

def nim_embed(
    texts: list[str],
    input_type: str = "query",
    model: str | None = None,
    timeout: float = 10.0,
) -> list[np.ndarray]:
    """
    Embed texts via NIM /v1/embeddings endpoint.

    Args:
        texts: List of strings to embed.
        input_type: "query" for search queries, "passage" for documents.
                    NV-Embed is a bi-encoder trained with different treatment
                    for queries vs passages. Getting these backwards degrades
                    retrieval silently. (Task 1a)
        model: Model ID override. Defaults to EMBED_MODEL env var.
        timeout: Request timeout in seconds.

    Returns:
        List of L2-normalized numpy arrays (float32).
        Normalization is explicit regardless of whether the API returns
        normalized vectors. (Task 1b)
    """
    model = model or os.getenv("EMBED_MODEL", EMBED_MODEL)
    url = f"{os.getenv('NIM_BASE_URL', NIM_BASE_URL)}/embeddings"

    body = {
        "input": texts,
        "model": model,
        "input_type": input_type,
        "encoding_format": "float",
    }

    result = _request_with_retry("POST", url, json_body=body, timeout=timeout)

    # Parse embeddings from response (OpenAI-compatible format)
    embeddings = []
    for item in sorted(result["data"], key=lambda x: x["index"]):
        vec = np.asarray(item["embedding"], dtype=np.float32)
        # Explicit L2 normalization (Task 1b)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec)

    return embeddings


def nim_embed_single(
    text: str,
    input_type: str = "query",
    model: str | None = None,
    timeout: float = 10.0,
) -> np.ndarray:
    """Convenience wrapper for embedding a single text."""
    results = nim_embed([text], input_type=input_type, model=model, timeout=timeout)
    return results[0]


# ─────────────────────────────────────────────
# Reranking — Task 2
# ─────────────────────────────────────────────

def nim_rerank(
    query: str,
    passages: list[str],
    model: str | None = None,
    timeout: float = 10.0,
) -> list[dict]:
    """
    Rerank passages against a query via NIM /v1/ranking endpoint.

    Args:
        query: The search query.
        passages: List of passage texts to rerank.
        model: Model ID override. Defaults to RERANK_MODEL env var.
        timeout: Request timeout in seconds.

    Returns:
        List of dicts with keys 'index' and 'logit', sorted by logit descending.
        Logit scores are raw (unbounded, can be negative). Higher = more relevant.
        No threshold is applied — caller decides how many to take.
    """
    model = model or os.getenv("RERANK_MODEL", RERANK_MODEL)
    url = f"{os.getenv('NIM_BASE_URL', NIM_BASE_URL)}/ranking"

    body = {
        "model": model,
        "query": {"text": query},
        "passages": [{"text": p} for p in passages],
    }

    result = _request_with_retry("POST", url, json_body=body, timeout=timeout)

    # Parse rankings from response
    rankings = result.get("rankings", [])
    # Sort by logit descending
    rankings.sort(key=lambda x: x.get("logit", 0), reverse=True)

    return rankings
