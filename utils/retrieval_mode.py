"""
Retrieval mode resolution (V3.8).

Replaces the boolean HYBRID_RETRIEVAL=0|1 flag with a three-state knob that
explicitly captures the user's *intent* and the system's *effective behavior*
when the requested mode isn't available.

Modes
-----
- ``auto``    : try hybrid, gracefully fall back to lexical if sqlite-vec
                cannot load on this Python build. **Default.**
- ``hybrid``  : require hybrid; raise on startup if capability probe fails.
                Use this in CI / eval runs where silent degradation would
                invalidate the experiment.
- ``lexical`` : disable embeddings entirely; force BM25 + FlashRank only.

Resolution order (highest precedence first)
-------------------------------------------
1. Per-request override (ContextVar set by orchestrator from RuntimeConfig).
2. ``RETRIEVAL_MODE`` env var.
3. Legacy ``HYBRID_RETRIEVAL`` env var (deprecated alias):
     ``HYBRID_RETRIEVAL=1`` → ``hybrid``
     ``HYBRID_RETRIEVAL=0`` → ``lexical``
4. Default: ``auto``.

After resolution, the *effective* mode is computed from the requested mode
plus the runtime capability probe. The effective mode is what actually
drives retrieval; the requested mode is what the caller asked for. We log
the (requested, effective, reason) triple at lifespan startup so an
evaluator never has to wonder which path their demo turn ran on.
"""
from __future__ import annotations

import contextvars
import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalMode(str, Enum):
    AUTO = "auto"
    HYBRID = "hybrid"
    LEXICAL = "lexical"


@dataclass(frozen=True)
class EffectiveRetrievalMode:
    requested: RetrievalMode
    effective: RetrievalMode  # always HYBRID or LEXICAL (never AUTO after resolution)
    reason: str | None        # populated when effective != requested (i.e. fell back)

    @property
    def is_hybrid(self) -> bool:
        return self.effective == RetrievalMode.HYBRID

    @property
    def is_fallback(self) -> bool:
        return self.reason is not None

    def to_metadata(self) -> dict:
        """Shape for `run_metadata.retrieval_mode` so evaluators see per-turn
        which path actually ran, not just the global default."""
        return {
            "requested": self.requested.value,
            "effective": self.effective.value,
            "reason": self.reason,
        }


_override: contextvars.ContextVar[RetrievalMode | None] = contextvars.ContextVar(
    "retrieval_mode_override", default=None,
)


def set_override(mode: RetrievalMode | None) -> contextvars.Token:
    """Bind a per-request mode override. Orchestrator calls this at the start
    of each turn from the resolved RuntimeConfig."""
    return _override.set(mode)


def reset_override(token: contextvars.Token) -> None:
    _override.reset(token)


def _requested_mode_from_env() -> RetrievalMode:
    """Read the env-level mode, honoring the legacy HYBRID_RETRIEVAL alias."""
    explicit = os.getenv("RETRIEVAL_MODE", "").strip().lower()
    if explicit:
        try:
            return RetrievalMode(explicit)
        except ValueError:
            logger.warning(
                "Unknown RETRIEVAL_MODE=%r; falling back to 'auto'. Valid: %s",
                explicit, [m.value for m in RetrievalMode],
            )
            return RetrievalMode.AUTO
    # Legacy alias
    legacy = os.getenv("HYBRID_RETRIEVAL", "").strip()
    if legacy == "1":
        logger.info(
            "HYBRID_RETRIEVAL=1 is deprecated; please use RETRIEVAL_MODE=hybrid. "
            "Honoring legacy value for now."
        )
        return RetrievalMode.HYBRID
    if legacy == "0":
        logger.info(
            "HYBRID_RETRIEVAL=0 is deprecated; please use RETRIEVAL_MODE=lexical. "
            "Honoring legacy value for now."
        )
        return RetrievalMode.LEXICAL
    return RetrievalMode.AUTO


def requested_mode() -> RetrievalMode:
    """Per-request override > env. Pure resolution; no capability probe."""
    ov = _override.get()
    if ov is not None:
        return ov
    return _requested_mode_from_env()


def resolve(requested: RetrievalMode, vec_available: bool) -> EffectiveRetrievalMode:
    """Combine the requested mode with the runtime capability probe."""
    if requested == RetrievalMode.LEXICAL:
        return EffectiveRetrievalMode(
            requested=requested, effective=RetrievalMode.LEXICAL, reason=None,
        )
    if requested == RetrievalMode.HYBRID:
        if not vec_available:
            # Fail-loud is the contract for this mode — caller wants reproducibility.
            raise RuntimeError(
                "RETRIEVAL_MODE=hybrid requested but sqlite-vec is unavailable on "
                "this Python build. Either switch to RETRIEVAL_MODE=auto (graceful "
                "fallback) or use a Python build with --enable-loadable-sqlite-extensions."
            )
        return EffectiveRetrievalMode(
            requested=requested, effective=RetrievalMode.HYBRID, reason=None,
        )
    # AUTO: try hybrid, fall back silently.
    if vec_available:
        return EffectiveRetrievalMode(
            requested=requested, effective=RetrievalMode.HYBRID, reason=None,
        )
    return EffectiveRetrievalMode(
        requested=requested,
        effective=RetrievalMode.LEXICAL,
        reason="sqlite_extensions_unavailable",
    )


def effective_mode_for_request(vec_available: bool) -> EffectiveRetrievalMode:
    """Convenience: full resolution for the current request context."""
    return resolve(requested_mode(), vec_available)
