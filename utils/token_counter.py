"""
tiktoken-based token counting. cl100k_base encoding everywhere.
ContextBudget allocates the 16K token total across pipeline stages.
"""
from __future__ import annotations

from dataclasses import dataclass

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass
class ContextBudget:
    total_tokens: int = 16000
    system_pct: float = 0.15       # 2,400 — system instructions
    history_pct: float = 0.25      # 4,000 — rolling summary + recent turns
    web_context_pct: float = 0.40  # 6,400 — selected web snippets
    output_pct: float = 0.20       # 3,200 — reserved for generation

    @property
    def web_context_budget(self) -> int:
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
