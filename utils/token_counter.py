"""
tiktoken-based token counting. cl100k_base encoding everywhere.
ContextBudget allocates the 16K token total across pipeline stages.

V3.4 note (Hindi): Devanagari script inflates ``cl100k_base`` token counts
roughly 2.5× relative to equivalent English text — each Devanagari character
typically encodes as multiple BPE tokens. The budget below stays the same for
Hindi turns, which means fewer Hindi chunks fit in the same ``web_context``
slice. This is an acceptable graceful-degradation behaviour: the context engine
will simply select fewer (but still ranked) Hindi snippets. A Hindi-specific
budget multiplier is a future optimization, not required for V3.4.
"""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate `text` so it encodes to at most `max_tokens` cl100k_base tokens.
    Returns `text` unchanged when already within budget. Useful as a precise
    last-resort fallback when char-based slicing would overshoot the budget
    (e.g. Devanagari, where 1 char ≈ 2.5 tokens)."""
    if max_tokens <= 0:
        return ""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


@dataclass
class ContextBudget:
    total_tokens: int = 16000
    system_pct: float = 0.15       # 2,400 — system instructions
    history_pct: float = 0.25      # 4,000 — rolling summary + recent turns
    web_context_pct: float = 0.40  # 6,400 — selected web snippets
    output_pct: float = 0.20       # 3,200 — reserved for generation
    # Explicit override for the web-context allocation in absolute tokens.
    # When set (non-None), bypasses the percentage-derived computation. Used
    # by the orchestrator to expand to 24K for ``difficulty == "hard"`` when
    # the synthesizer is Gemini (1M context window).
    web_context_override: int | None = None

    @property
    def web_context_budget(self) -> int:
        if self.web_context_override is not None:
            return int(self.web_context_override)
        return int(self.total_tokens * self.web_context_pct)  # 6,400

    @property
    def history_budget(self) -> int:
        return int(self.total_tokens * self.history_pct)  # 4,000

    def adapt_to_complexity(self, query_count: int) -> None:
        """Allocate more tokens to web context for complex queries (many hops)."""
        if query_count >= 3:
            # Complex: shift 15% from history to web context
            self.history_pct = 0.10
            self.web_context_pct = 0.55
