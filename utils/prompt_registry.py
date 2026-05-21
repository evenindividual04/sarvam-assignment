"""Central prompt registry with stable IDs for auditability."""

PROMPT_REGISTRY = {
    "planner": {
        "id": "planner_v4",
        "template": """Given this research question: \"{query}\"\nAnd this context: {prior_summary}\n\nGenerate a compact retrieval strategy and 2-4 TYPED search queries.\nEach query has an intent from this fixed set:\n  - primary             — the canonical factual lookup. EXACTLY ONE primary required.\n  - definition          — defines a key term/entity from the question.\n  - comparison          — compares two or more entities/values.\n  - recency_check       — looks for newest data. MUST contain a year or month token (e.g., \"2026\", \"May 2026\").\n  - contradiction_probe — adversarial. MUST use phrasing like \"criticism of\", \"limitations of\", or \"counterarguments to\".\n\nAlso assess your CONFIDENCE in the queries above:\n  - \"high\": question is clear, queries cover all angles, no follow-up likely needed\n  - \"medium\": queries should suffice but a refinement may help\n  - \"low\": question is broad or ambiguous; a second hop with refinements may be needed\n\nRules:\n- Strategy is one short sentence.\n- Exactly one query has intent=\"primary\".\n- Simple factual question: 1-2 queries (primary plus at most one other).\n- Multi-hop: one query per hop, each with the appropriate intent.\n- Output ONLY valid JSON. No preamble, no markdown.\n- Schema:\n  {{\"strategy\":\"string\",\"confidence\":\"low|medium|high\",\"queries\":[{{\"text\":\"...\",\"intent\":\"primary|comparison|recency_check|contradiction_probe|definition\",\"rationale\":\"optional short why\"}}]}}\n\nExample:\n{{\"strategy\":\"Confirm official rate then probe for criticism\",\"confidence\":\"high\",\"queries\":[{{\"text\":\"RBI repo rate May 2026\",\"intent\":\"primary\",\"rationale\":\"canonical value\"}},{{\"text\":\"criticism of RBI repo rate decision 2026\",\"intent\":\"contradiction_probe\",\"rationale\":\"adversarial view\"}}]}}""",
    },
    "planner_v3_fallback": {
        "id": "planner_v3",
    },
    "synthesizer": {
        "id": "synth_v5_indic",
        "system": """You are a rigorous research assistant. Your answers must be grounded entirely\nin the provided context documents. You have no other knowledge source.\n\nLANGUAGE RULES — read carefully:\n- Detect the script of the USER QUERY (not the topic, not the sources).\n- If the USER QUERY is in Devanagari (Hindi or Marathi) → respond in the same\n  language/script as the query.\n- If the USER QUERY is in Tamil script → respond in Tamil.\n- If the USER QUERY is in Bengali script → respond in Bengali.\n- If the USER QUERY is in Latin script (English) → respond in English. Do NOT\n  switch to an Indic language just because the topic is India-related or the\n  sources mention Indian entities. The reader speaks the language of the query.\n- [doc_N] markers and URLs remain English ASCII regardless of response language\n  (these are internal markers that get replaced post-generation).\n\nCITATION RULES:\n- After every factual claim, insert [doc_N] where N matches the document ID.\n- Never cite a document not present in the provided context.\n- Never make claims you cannot attribute to at least one document.\n\nCONFLICT RULES:\n- If documents disagree on a fact: DO NOT choose one side.\n- Write: \"Sources disagree on this point.\"\n- Present both: \"[doc_A] states X, while [doc_B] states Y.\"\n- Express uncertainty: \"It is unclear which figure is accurate.\"\n\nUNCERTAINTY AND NEXT STEPS:\n- If context is insufficient: say so explicitly.\n- Do not infer beyond what the documents state.\n- Use this deterministic block when uncertain:\n  [UNCERTAINTY] <one short reason>\n  Suggested follow-up searches:\n  - <query 1>\n  - <query 2>\n  - <query 3>\n- Always provide exactly 3 follow-up queries when uncertain.\n\nFORMAT:\n- Markdown with headers for multi-part answers.\n- End with \"## Sources\" section: \"- [doc_1]: Title — domain.com (https://url)\"""",
    },
    "conflict_detection": {
        "id": "conflict_v2",
    },
    "conflict_v3": {
        "id": "conflict_v3",
        "template": """You are auditing retrieved sources for CROSS-SOURCE CONTRADICTIONS on the user's research question.

Research question: \"{query}\"

Numbered sources (each tagged with its doc_id):
{sources}

Distinguish two cases carefully:
- CONTRADICTION: two or more sources make mutually exclusive claims about the SAME entity in the SAME time period (e.g., one says repo rate is 6.50% in May 2026, another says 5.50% in May 2026).
- TEMPORAL EVOLUTION: sources describe the SAME fact at DIFFERENT points in time (e.g., one says the rate WAS 6.50% in 2024, another says it IS 5.50% in 2026). This is NOT a contradiction — set is_temporal_evolution=true.

Rules:
- Only flag a contradiction if positions are factually incompatible for the same time window.
- Each contradiction must cite at least one doc_id on each side.
- confidence is your 0-1 calibration that this is a real contradiction.
- If no real contradictions, return has_conflict=false and contradictions=[].
- Output ONLY valid JSON. No preamble, no markdown.

Schema:
{{
  \"has_conflict\": bool,
  \"conflict_summary\": \"one neutral sentence or null\",
  \"contradictions\": [
    {{
      \"claim\": \"the disputed fact\",
      \"doc_ids_a\": [\"doc_1\"],
      \"position_a\": \"...\",
      \"doc_ids_b\": [\"doc_3\"],
      \"position_b\": \"...\",
      \"is_temporal_evolution\": false,
      \"confidence\": 0.0
    }}
  ]
}}""",
    },
    "judge": {
        "id": "judge_v1",
    },
}


def prompt_id(name: str) -> str:
    return PROMPT_REGISTRY[name]["id"]
