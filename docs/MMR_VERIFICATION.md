# MMR Selector Verification (Phase 5c)

## Live selector

The active selection path is chosen at request time in
`agent/orchestrator.py:472-486` based on `config.selection_strategy`:

- `heuristic` (default) → `select_with_diversity` in
  `agent/context_engine.py:162` (via `rank_and_select` at line 278).
- `mmr` → `rank_and_select_mmr` in `agent/context_engine.py:379`.

**P1 finding:** The default selection strategy is `heuristic`, not MMR
(`agent/orchestrator.py:51`, `CONTEXT_SELECTION_STRATEGY` env defaults to
`"heuristic"`). MMR is implemented and reachable but only runs when an
operator (or per-request override) flips the knob. `rank_and_select_mmr` is
**not** dead code — the orchestrator dispatches to it when configured.

## `MMR_LAMBDA`

Controls the relevance-vs-novelty tradeoff inside MMR:
`score = λ·relevance − (1−λ)·max_similarity_to_already_selected`.

- λ = 1.0 → pure relevance (no diversity penalty).
- λ = 0.0 → pure novelty (ignores relevance once first chunk picked).
- **Default: 0.7** (per `MMR_LAMBDA` env fallback in
  `agent/context_engine.py:41` and `RuntimeConfig.from_overrides`
  in `agent/orchestrator.py`).

Values outside `[0.0, 1.0]` are clamped in `RuntimeConfig.from_overrides`.

## Per-query override

The `/chat` endpoint accepts `ChatRequest.overrides`, which is layered onto
env defaults by `RuntimeConfig.from_overrides` (`agent/orchestrator.py:67`).
Example body:

```json
{
  "query": "...",
  "session_id": "...",
  "overrides": {
    "CONTEXT_SELECTION_STRATEGY": "mmr",
    "MMR_LAMBDA": "0.5"
  }
}
```

The resolved knobs are also surfaced by `GET /settings/defaults`
(`main.py:213`), which now lists `MMR_LAMBDA` alongside the existing knobs.

## Sample debug log line

With `logging` set to `DEBUG`, the live selector emits one line per
selection invocation (`agent/context_engine.py`, inside each selector):

```
DEBUG agent.context_engine: Context selection path=mmr lambda=0.7
DEBUG agent.context_engine: Context selection path=legacy_diversity lambda=n/a
```

This makes it trivial to confirm in a production trace which selector
actually ran for a given turn, without needing to inspect run metadata.
