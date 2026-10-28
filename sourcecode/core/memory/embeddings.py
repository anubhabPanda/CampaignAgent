"""Embedding seam: production can use local Ollama or OpenAI; tests inject deterministic vectors."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Protocol
from urllib.request import Request, urlopen


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class OllamaNomicEmbedder:
    """Small dependency-free client for a locally running Ollama instance."""
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model, self.base_url = model, base_url.rstrip("/")

    @property
    def fingerprint(self) -> str:
        return f"ollama:{self.base_url}:{self.model}:nomic-retrieval-v1"

    def embed(self, text: str) -> list[float]:
        """Embed text for clustering/chunk-boundary comparison."""
        return self._embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(f"search_document: {text}")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(f"search_query: {text}")

    def _embed(self, text: str) -> list[float]:
        # Ollama's current endpoint accepts batched input; older installs use
        # /api/embeddings. Supporting both keeps local workshop setup simple.
        body = json.dumps({"model": self.model, "input": text}).encode()
        request = Request(self.base_url + "/api/embed", data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=30) as response:  # nosec B310: local configurable service
                data = json.load(response)
            return list(data["embeddings"][0])
        except Exception:
            old_body = json.dumps({"model": self.model, "prompt": text}).encode()
            old = Request(self.base_url + "/api/embeddings", data=old_body, headers={"Content-Type": "application/json"})
            with urlopen(old, timeout=30) as response:  # nosec B310: local configurable service
                return list(json.load(response)["embedding"])

    def release(self) -> None:
        """Unload the embedding model before a larger answer model runs.

        Ollama otherwise keeps both models resident. On student laptops that
        can terminate the answer connection under memory pressure even though
        embedding and generation each work independently.
        """
        body = json.dumps({"model": self.model, "input": "", "keep_alive": 0}).encode()
        request = Request(self.base_url + "/api/embed", data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=10):  # nosec B310: local configurable service
                pass
        except Exception:
            pass


class OpenAIEmbedder:
    """Embeds text via OpenAI's embeddings API.

    Dependency-free (no `openai` SDK import) so the module keeps working even
    when that package is absent; it talks directly to the REST endpoint,
    mirroring ``OpenAIGatewayClient``'s env-var conventions (``OPENAI_API_KEY``,
    ``OPENAI_BASE_URL``).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")

    @property
    def fingerprint(self) -> str:
        return f"openai:{self.model}"

    def embed(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        body = json.dumps({"model": self.model, "input": text}).encode()
        request = Request(
            self.base_url + "/embeddings",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key or ''}",
            },
        )
        with urlopen(request, timeout=30) as response:  # nosec B310: configurable, trusted API endpoint
            data = json.load(response)
        return list(data["data"][0]["embedding"])


class DeterministicEmbedder:
    """Stable hashed bag-of-words vectors for unit tests and offline demos."""
    def __init__(self, dimensions: int = 96):
        self.dimensions = dimensions

    @property
    def fingerprint(self) -> str:
        return f"deterministic:{self.dimensions}"

    def embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token, count in Counter(re.findall(r"[a-z0-9]+", text.lower())).items():
            # stable, unlike Python's randomized hash()
            slot = sum((i + 1) * ord(c) for i, c in enumerate(token)) % self.dimensions
            values[slot] += float(count)
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values] if norm else values

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right))
