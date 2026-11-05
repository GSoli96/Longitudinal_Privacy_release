"""Thin wrapper around the Ollama client.

Works with a local Ollama server or an Ollama-Cloud-compatible endpoint.
The endpoint and the API key are read from the environment, never
hard-coded:

    OLLAMA_HOST      e.g. http://localhost:11434  or  https://ollama.com
    OLLAMA_API_KEY   bearer token for a cloud endpoint (omit for local)

The original experiment scripts contained hard-coded API keys. Those keys
have been removed; configure your own through the environment or a local
.env file (see configs/example.env). Never commit a key.
"""

from __future__ import annotations

import os
import time

try:
    from ollama import Client
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the Ollama client:  pip install ollama") from exc


def make_client(host: str | None = None, api_key: str | None = None) -> "Client":
    host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    api_key = api_key if api_key is not None else os.environ.get("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    return Client(host=host, headers=headers)


def generate(
    client: "Client",
    model: str,
    prompt: str,
    temperature: float = 0.0,
    system: str | None = None,
    max_retries: int = 3,
    backoff: float = 10.0,
) -> str:
    """Single-turn chat completion with simple exponential back-off retry."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature},
                stream=False,
            )
            return resp["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - network/model errors are retried
            last_exc = exc
            if attempt < max_retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_exc}")
