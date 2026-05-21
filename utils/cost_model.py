"""Per-1K-token cost lookup. All entries currently $0 (free-tier providers).
Table exists so cost math is wired through the pipeline and ready when the
deployment moves off free tier."""
from __future__ import annotations

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gemini-2.5-flash": {"prompt": 0.0, "completion": 0.0},
    "groq-llama-3.3-70b": {"prompt": 0.0, "completion": 0.0},
    "gpt-4o-mini-github": {"prompt": 0.0, "completion": 0.0},
    "openrouter-deepseek-r1": {"prompt": 0.0, "completion": 0.0},
    # Sarvam Model API — Apache-2.0 base models; API-tier pricing not yet
    # published. Verify against https://docs.sarvam.ai/ before billing.
    "sarvam-m": {"prompt": 0.0, "completion": 0.0},
    "sarvam-30b": {"prompt": 0.0, "completion": 0.0},
    "sarvam-105b": {"prompt": 0.0, "completion": 0.0},
}

DEFAULT_MODEL = "gemini-2.5-flash"


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute USD cost for a given (model, prompt, completion) triple.

    Unknown models return 0.0 — we choose silent zero over raising because
    cost rollup is informational and must never break an eval run.
    """
    rates = COST_PER_1K_TOKENS.get(model)
    if rates is None:
        return 0.0
    return (
        (prompt_tokens / 1000.0) * rates["prompt"]
        + (completion_tokens / 1000.0) * rates["completion"]
    )
