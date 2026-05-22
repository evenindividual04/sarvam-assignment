"""
V3.1 — Singleton wrapper over fastembed's BAAI/bge-small-en-v1.5 (384-dim, ~33MB ONNX, CPU only).

fastembed is lazy-imported inside the singleton so importing this module does
NOT trigger model load unless hybrid retrieval is on.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384
_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model = None  # cached fastembed.TextEmbedding instance


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding  # lazy import — avoids load unless used
        logger.info("Loading embedding model: %s", _MODEL_NAME, extra={"component": "embedder"})
        _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def _embed_sync(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    # fastembed returns a generator of np.ndarray; coerce to plain list[float].
    return [list(map(float, vec)) for vec in model.embed(texts)]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Async wrapper. Runs the CPU-bound embed call in a thread."""
    if not texts:
        return []
    return await asyncio.to_thread(_embed_sync, texts)


def is_warm() -> bool:
    """True if the model has been instantiated (and ONNX session initialized)."""
    return _model is not None


async def warm() -> None:
    """Pre-warm the embedding model + ONNX session so the *first* user query
    doesn't pay the 2-4s cold-start. Called from `main.py:lifespan` as a
    background task immediately after `init_db()`.

    The Dockerfile also runs the model load at build time so the bge-small-en-v1.5
    weights (~30MB) are baked into the image layer; this function just warms
    the in-process ONNX session against those already-cached weights, which is
    sub-second on HF Spaces.
    """
    if is_warm():
        return
    try:
        await asyncio.to_thread(_embed_sync, ["warmup"])
        logger.info("Embedding model warm; subsequent queries skip cold-start.",
                    extra={"component": "embedder"})
    except Exception as exc:  # pragma: no cover — best-effort
        logger.warning(
            "Embedding warmup failed (will retry on first query): %s",
            exc, extra={"component": "embedder"},
        )
