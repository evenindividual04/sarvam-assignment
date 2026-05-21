"""
Multi-provider LLM router.
  plan()            → Groq  Llama 3.3 70B  max_tokens=200
  call_groq()       → Groq  Llama 3.3 70B  (conflict detection, rolling summary)
  synthesize()      → Gemini 2.5 Flash      max_tokens=1500 (streams)
  judge()           → GitHub Models GPT-4o-mini max_tokens=400

Synthesis falls back to OpenRouter DeepSeek R1 on ResourceExhausted.
"""
from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from agent.models import PlannerOutput
from utils.prompt_registry import PROMPT_REGISTRY

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_GITHUB_MODEL = "gpt-4o-mini"
_OPENROUTER_MODEL = "deepseek/deepseek-r1"

# ── Synthesis system prompt (verbatim from spec Section 2.9) ──────────────
SYNTHESIS_SYSTEM_PROMPT = PROMPT_REGISTRY["synthesizer"]["system"]

# ── Planning prompt template (from spec Section 2.9) ─────────────────────
PLANNING_PROMPT_TEMPLATE = PROMPT_REGISTRY["planner"]["template"]


def parse_planner_output(raw: str, query: str) -> PlannerOutput:
    """Parse planner output with deterministic fallback."""
    fallback = PlannerOutput(strategy="Direct retrieval fallback", queries=[query])
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return fallback
    try:
        parsed = PlannerOutput.model_validate_json(raw[start:end])
    except Exception:
        return fallback
    queries = [q.strip() for q in parsed.queries if isinstance(q, str) and q.strip()]
    if not queries:
        queries = [query]
    strategy = (parsed.strategy or fallback.strategy).strip()
    if not strategy:
        strategy = fallback.strategy
    return PlannerOutput(strategy=strategy, queries=queries[:4])


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def plan(query: str, prior_summary: str = "No prior context.") -> PlannerOutput:
    """Generate strategy + search queries via Groq."""
    import groq as groq_sdk
    client = groq_sdk.AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    prompt = PLANNING_PROMPT_TEMPLATE.format(query=query, prior_summary=prior_summary)
    resp = await client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    parsed = parse_planner_output(raw, query)
    if parsed.strategy == "Direct retrieval fallback":
        logger.warning("Plan parse failed, using fallback. Raw output: %s", raw)
    return parsed


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def call_groq(prompt: str, max_tokens: int = 200) -> str:
    """Generic Groq call for conflict detection, rolling summary, etc."""
    import groq as groq_sdk
    client = groq_sdk.AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    resp = await client.chat.completions.create(
        model=_GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


async def synthesize(
    query: str,
    context_xml: str,
    doc_map: dict[str, tuple[str, str, str]],
    history_text: str = "",
    conflict_note: Optional[str] = None,
) -> AsyncIterator[tuple[str, int, int]]:
    """
    Streams synthesis from Gemini 2.5 Flash.
    Falls back to OpenRouter DeepSeek R1 on ResourceExhausted.
    Yields (text_chunk, prompt_tokens, completion_tokens) — token counts sent on final chunk.
    """
    # Build doc listing for the prompt
    doc_listing = "\n".join(
        f"  [{doc_id}]: {title} — {domain} ({url})"
        for doc_id, (title, url, domain) in doc_map.items()
    )
    conflict_instruction = ""
    if conflict_note:
        conflict_instruction = f"\n\nCONFLICT DETECTED: {conflict_note}\nYou MUST present both sides explicitly."

    user_prompt = f"""Available documents:
{doc_listing}

{context_xml}
{conflict_instruction}

Research question: {query}

{("Prior conversation context:\n" + history_text) if history_text else ""}

Answer the research question using only the documents above. Cite every factual claim with [doc_N]."""

    try:
        async for chunk in _synthesize_gemini(user_prompt):
            yield chunk
    except Exception as e:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not openrouter_key:
            raise
        if "ResourceExhausted" in str(e) or "429" in str(e) or "404" in str(e) or "not found" in str(e).lower():
            logger.warning("Gemini unavailable, falling back to OpenRouter: %s", e)
            async for chunk in _synthesize_openrouter(user_prompt):
                yield chunk
        else:
            logger.warning("Gemini failed, falling back to OpenRouter: %s", e)
            async for chunk in _synthesize_openrouter(user_prompt):
                yield chunk


async def _synthesize_gemini(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = types.GenerateContentConfig(
        system_instruction=SYNTHESIS_SYSTEM_PROMPT,
        max_output_tokens=1500,
        temperature=0.2,
    )

    prompt_tokens = 0
    completion_tokens = 0

    candidate_models = [
        _GEMINI_MODEL,
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    seen: set[str] = set()
    last_err: Exception | None = None
    for model_name in candidate_models:
        if model_name in seen:
            continue
        seen.add(model_name)
        try:
            async for response in await client.aio.models.generate_content_stream(
                model=model_name,
                contents=user_prompt,
                config=config,
            ):
                if response.text:
                    yield (response.text, 0, 0)
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = response.usage_metadata.prompt_token_count or 0
                    completion_tokens = response.usage_metadata.candidates_token_count or 0
            # Final sentinel with token counts
            yield ("", prompt_tokens, completion_tokens)
            return
        except Exception as e:
            last_err = e
            if "not found" in str(e).lower() or "404" in str(e):
                logger.warning("Gemini model unavailable (%s): %s", model_name, e)
                continue
            raise

    if last_err is not None:
        raise last_err
    raise RuntimeError("No Gemini model candidates available")


async def _synthesize_openrouter(user_prompt: str) -> AsyncIterator[tuple[str, int, int]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        timeout=90.0,
    )
    messages = [
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    resp = await client.chat.completions.create(
        model=_OPENROUTER_MODEL,
        messages=messages,
        max_tokens=1500,
        stream=True,
    )
    async for chunk in resp:
        text = chunk.choices[0].delta.content or ""
        if text:
            yield (text, 0, 0)
    yield ("", 0, 0)


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3), reraise=True)
async def judge(prompt: str) -> str:
    """GitHub Models GPT-4o-mini for eval judging. Different family from generator."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ["GITHUB_TOKEN"],
        base_url="https://models.inference.ai.azure.com",
        timeout=60.0,
    )
    resp = await client.chat.completions.create(
        model=_GITHUB_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


async def rolling_summary(turns_text: str) -> str:
    """Compress old turns into a rolling summary via Groq."""
    prompt = f"""Summarize the following research conversation into a concise paragraph.
Preserve all key facts, entities, and conclusions. Do not add any information not present.

{turns_text}

Summary:"""
    return await call_groq(prompt, max_tokens=300)
