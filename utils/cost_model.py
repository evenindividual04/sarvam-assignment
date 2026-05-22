"""Per-1K-token cost lookup (USD).

Rates derived from published per-1M pricing as of late 2025 / early 2026,
divided by 1000 to give per-1K. The deployment runs on free tiers, so the
actual bill is $0 — these rates exist so eval cost rollups reflect the
"realistic" rate a paying tenant would see.
"""
from __future__ import annotations

# All rates are USD per 1K tokens (published per-1M ÷ 1000).
COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    # Source: ai.google.dev/gemini-api/docs/pricing as of 2026-01-15
    # Paid tier: $0.075 / $0.30 per 1M tokens.
    "gemini-2.5-flash": {"prompt": 0.000075, "completion": 0.00030},
    # Source: groq.com/pricing as of 2026-01-15
    # Llama 3.3 70B Versatile: $0.59 / $0.79 per 1M tokens.
    "groq-llama-3.3-70b": {"prompt": 0.00059, "completion": 0.00079},
    # Source: openai.com/api/pricing as of 2026-01-15 (GitHub Models mirrors
    # the OpenAI rate card). GPT-4o-mini: $0.15 / $0.60 per 1M tokens.
    "gpt-4o-mini-github": {"prompt": 0.00015, "completion": 0.00060},
    # Source: openrouter.ai/deepseek/deepseek-r1 as of 2026-01-15
    # DeepSeek R1 (cache): $0.14 / $0.28 per 1M tokens.
    "openrouter-deepseek-r1": {"prompt": 0.00014, "completion": 0.00028},
    # Sarvam Model API — free during public preview as of 2026-01-15.
    # No published $/token rate yet; verify against https://docs.sarvam.ai/
    # before billing. Tracked at 0 intentionally.
    "sarvam-m": {"prompt": 0.0, "completion": 0.0},
    "sarvam-30b": {"prompt": 0.0, "completion": 0.0},
    "sarvam-105b": {"prompt": 0.0, "completion": 0.0},
    # Source: inference.cerebras.ai/pricing as of 2026-01-15
    # Llama tier: $0.10 / $0.10 per 1M tokens.
    "cerebras-llama": {"prompt": 0.00010, "completion": 0.00010},
    # Local Ollama: self-hosted, no API cost.
    "ollama-local": {"prompt": 0.0, "completion": 0.0},
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
