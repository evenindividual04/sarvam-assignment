"""Streaming filter that strips reasoning / chain-of-thought tag blocks.

Several models we route through (Sarvam-m, Qwen 3 235B on Cerebras, DeepSeek R1)
emit their internal reasoning inside `<think>...</think>` or
`<thinking>...</thinking>` tags. When that streams straight through to the
SSE pipeline, the user sees the model's scratchpad above its real answer.

This filter is stateful: opening and closing tags can arrive in any chunk
boundary. It buffers ambiguous bytes (a lone `<` that might be the start of a
reasoning tag) until the tag is resolved, so we never emit a partial tag and
never drop legitimate angle brackets that turn out not to be reasoning tags.

Compliance constraint: assignment §"no hidden CoT" — the user must see the
final answer only; reasoning belongs in DB run_metadata, not the SSE stream.
"""
from __future__ import annotations

import re
from typing import Final

# Order matters: the longer tag must come first so a "<thinking>" prefix
# isn't accidentally matched as "<think>" + stray "ing>".
_OPEN_TAGS: Final[tuple[str, ...]] = ("<thinking>", "<think>")
_CLOSE_TAGS: Final[tuple[str, ...]] = ("</thinking>", "</think>")

# Maximum prefix we might be buffering while waiting to decide if a "<" begins
# a reasoning tag. len("<thinking>") == 10, plus one byte of headroom.
_MAX_AMBIGUOUS_PREFIX: Final[int] = max(len(t) for t in _OPEN_TAGS) + 1

# Pre-compiled fast path for when a whole chunk is clean (no relevant char).
_HAS_INTEREST = re.compile(r"[<]")


class CotStreamFilter:
    """Stateful filter for streamed text. Feed chunks in, get clean text out.

    Usage:
        f = CotStreamFilter()
        for chunk in stream:
            clean = f.feed(chunk)
            if clean:
                yield clean
        tail = f.flush()
        if tail:
            yield tail
    """

    __slots__ = ("_buf", "_in_think")

    def __init__(self) -> None:
        self._buf: str = ""
        self._in_think: bool = False

    def feed(self, chunk: str) -> str:
        """Consume one streamed chunk; return the safe-to-emit text."""
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            if self._in_think:
                # Looking for any closing tag; drop everything until we find one.
                idx, tag = _find_first(self._buf, _CLOSE_TAGS)
                if idx == -1:
                    # No close yet. Hold a tail just in case the close tag
                    # is straddling the next chunk boundary.
                    keep = max(len(t) for t in _CLOSE_TAGS) - 1
                    self._buf = self._buf[-keep:] if len(self._buf) > keep else self._buf
                    return "".join(out)
                # Drop CoT content + the close tag itself.
                self._buf = self._buf[idx + len(tag):]
                self._in_think = False
                continue

            # Not currently inside a think block — emit until we hit "<".
            lt = self._buf.find("<")
            if lt == -1:
                out.append(self._buf)
                self._buf = ""
                return "".join(out)

            # Flush everything before the "<" — that's clean answer text.
            if lt > 0:
                out.append(self._buf[:lt])
                self._buf = self._buf[lt:]

            # Decide what the "<" is: open tag, false alarm, or still ambiguous?
            tag = _starts_with_any(self._buf, _OPEN_TAGS)
            if tag:
                self._buf = self._buf[len(tag):]
                self._in_think = True
                continue

            # Could the buffer still grow into an open tag?
            if any(t.startswith(self._buf) for t in _OPEN_TAGS):
                # Wait for more input.
                return "".join(out)

            # The "<" is not the start of a reasoning tag — emit it verbatim
            # and resume scanning after it.
            out.append(self._buf[0])
            self._buf = self._buf[1:]

        return "".join(out)

    def flush(self) -> str:
        """Emit any buffered tail at end-of-stream.

        If we end mid-reasoning-block (model truncated before closing
        </think>), drop the unterminated buffer rather than leak CoT. If we
        end with a clean buffer, emit it.
        """
        if self._in_think:
            self._buf = ""
            self._in_think = False
            return ""
        tail = self._buf
        self._buf = ""
        return tail


def _find_first(haystack: str, needles: tuple[str, ...]) -> tuple[int, str]:
    """Find the earliest occurrence of any needle. Return (-1, '') if none."""
    best_idx = -1
    best_tag = ""
    for n in needles:
        i = haystack.find(n)
        if i == -1:
            continue
        if best_idx == -1 or i < best_idx:
            best_idx, best_tag = i, n
    return best_idx, best_tag


def _starts_with_any(s: str, needles: tuple[str, ...]) -> str | None:
    for n in needles:
        if s.startswith(n):
            return n
    return None
