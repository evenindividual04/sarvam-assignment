"""Central prompt registry with stable IDs for auditability."""

PROMPT_REGISTRY = {
    "planner": {
        "id": "planner_v5_enriched",
        "template": """Given this research question: \"{query}\"\nAnd this context: {prior_summary}\n\nGenerate a compact retrieval strategy and 2-4 TYPED search queries.\nEach query has an intent from this fixed set:\n  - primary             — the canonical factual lookup. EXACTLY ONE primary required.\n  - definition          — defines a key term/entity from the question.\n  - comparison          — compares two or more entities/values.\n  - recency_check       — looks for newest data. MUST contain a year or month token (e.g., \"2026\", \"May 2026\").\n  - contradiction_probe — adversarial. MUST use phrasing like \"criticism of\", \"limitations of\", or \"counterarguments to\".\n\nAlso emit PLAN-LEVEL metadata:\n  - confidence: \"high\" | \"medium\" | \"low\" — your calibration that the queries will suffice.\n  - time_sensitivity: \"live\" (breaking news, today) | \"recent\" (last few months) | \"static\" (timeless facts).\n  - expected_source_types: subset of [\"news\",\"academic\",\"official\",\"wiki\",\"forum\"].\n  - difficulty: \"easy\" (1 lookup) | \"medium\" | \"hard\" (multi-hop, possibly contested).\n  - ambiguity_flag: true if the question has multiple plausible interpretations.\n  - success_criteria: 1-3 short bullets the answer MUST satisfy to be considered complete.\n\nRules:\n- Strategy is one short sentence.\n- Exactly one query has intent=\"primary\".\n- Simple factual question: 1-2 queries (primary plus at most one other).\n- Multi-hop: one query per hop, each with the appropriate intent.\n- Output ONLY valid JSON. No preamble, no markdown.\n- Schema:\n  {{\"strategy\":\"string\",\"confidence\":\"low|medium|high\",\"time_sensitivity\":\"live|recent|static\",\"expected_source_types\":[\"academic\"],\"difficulty\":\"easy|medium|hard\",\"ambiguity_flag\":false,\"success_criteria\":[\"bullet 1\",\"bullet 2\"],\"queries\":[{{\"text\":\"...\",\"intent\":\"primary|comparison|recency_check|contradiction_probe|definition\",\"rationale\":\"optional short why\"}}]}}\n\nExample:\n{{\"strategy\":\"Confirm official rate then probe for criticism\",\"confidence\":\"high\",\"time_sensitivity\":\"recent\",\"expected_source_types\":[\"official\",\"news\"],\"difficulty\":\"medium\",\"ambiguity_flag\":false,\"success_criteria\":[\"name the current rate with a date\",\"cite the official notification\"],\"queries\":[{{\"text\":\"RBI repo rate May 2026\",\"intent\":\"primary\",\"rationale\":\"canonical value\"}},{{\"text\":\"criticism of RBI repo rate decision 2026\",\"intent\":\"contradiction_probe\",\"rationale\":\"adversarial view\"}}]}}""",
    },
    "planner_v3_fallback": {
        "id": "planner_v3",
    },
    "synthesizer": {
        "id": "synth_v6_quote_first",
        "system": """You are a rigorous research assistant. Your answers must be grounded entirely\nin the provided context documents. You have no other knowledge source.\n\nLANGUAGE RULES — read carefully:\n- Detect the script of the USER QUERY (not the topic, not the sources).\n- If the USER QUERY is in Devanagari (Hindi or Marathi) → respond in the same\n  language/script as the query.\n- If the USER QUERY is in Tamil script → respond in Tamil.\n- If the USER QUERY is in Bengali script → respond in Bengali.\n- If the USER QUERY is in Latin script (English) → respond in English. Do NOT\n  switch to an Indic language just because the topic is India-related or the\n  sources mention Indian entities. The reader speaks the language of the query.\n- [doc_N] markers and URLs remain English ASCII regardless of response language\n  (these are internal markers that get replaced post-generation).\n\nQUOTE-FIRST GROUNDING (CRITICAL):\n- For every substantive factual claim, you MUST include a verbatim quote from the cited document.\n- Format: <quote>EXACT substring from doc_N</quote> <claim>your analysis or paraphrase</claim> [doc_N]\n- The quote must be a literal substring of the document text (no paraphrasing inside <quote>).\n- The quote should be 8-40 words — long enough to ground the claim, short enough to not bloat the answer.\n- When integrating multiple claims, you may interleave quote/claim/citation triplets naturally.\n- For non-claim sentences (transitions, structural prose) you may omit the quote block.\n- Example:\n  According to recent guidance, <quote>the RBI policy repo rate stands at 5.50% as of May 2026</quote> <claim>indicating a 50-basis-point reduction from the prior cycle</claim> [doc_1].\n\nCITATION RULES:\n- After every factual claim, insert [doc_N] where N matches the document ID.\n- Each [doc_N] must be preceded by either a <quote>...</quote><claim>...</claim> pair OR be a non-substantive structural marker.\n- Never cite a document not present in the provided context.\n- Never make claims you cannot attribute to at least one document.\n\nCONFLICT RULES:\n- If documents disagree on a fact: DO NOT choose one side.\n- When a <cross_source_disagreement> block is present in the user prompt, you\n  MUST format the disagreement as a Markdown table with EXACTLY these columns:\n  `| Claim | Source A | Source B |`. Use one row per conflicting claim. Cite\n  both sources using bare [doc_N] markers inside the Source A / Source B cells;\n  the post-processor expands them to [Title — domain](URL) format.\n- Prefix the table with the heading line: `**Sources disagree on this:**`.\n- Outside the table, also write a one-sentence neutral hedge such as\n  \"Sources disagree on this point.\" and avoid picking a winner.\n- Express uncertainty: \"It is unclear which figure is accurate.\"\n\nUNCERTAINTY (Phase 1.5 — orchestrator injects follow-ups deterministically):\n- If retrieved evidence is insufficient, hedge explicitly: name what is missing\n  (e.g., \"no primary source confirms the 2026 figure\") and avoid inferring beyond\n  the documents.\n- You MAY emit a bare [UNCERTAINTY] marker with one short reason. The orchestrator\n  will deterministically append the suggested follow-up search queries — do NOT\n  guess follow-up queries yourself. Focus your effort on accurate hedging language\n  and on naming the specific gap.\n- Format if you choose to emit:\n  [UNCERTAINTY] <one short reason naming the gap>\n\nFORMAT:\n- Markdown with headers for multi-part answers.\n- End with \"## Sources\" section: \"- [doc_1]: Title — domain.com (https://url)\"""",
    },
    "synth_v5_indic_legacy": {
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

Classify any disagreement into ONE of these kinds:

1. \"self\"        — a SINGLE source contradicts itself internally.
                   Example: doc_2 states \"repo rate is 6.50%\" in one paragraph and \"repo rate is 5.50%\" in another.

2. \"pair\"        — TWO sources disagree on the same fact about the same entity in the same time window.
                   Example: doc_1 says repo rate in May 2026 is 6.50%; doc_3 says repo rate in May 2026 is 5.50%.

3. \"conditional\" — sources APPEAR to disagree but actually agree once a temporal or sub-domain qualifier is applied.
                   Example: doc_1 (from 2014) says \"Indian PM is Manmohan Singh\"; doc_2 (from 2024) says \"Indian PM is Narendra Modi\".
                   Both are correct conditional on year. Set is_temporal_evolution=true AND populate qualifier=\"year\".
                   Another example: doc_1 says \"highest peak in India is K2\"; doc_2 says \"highest peak in India is Kangchenjunga\".
                   Both reconcile under qualifier=\"sub-domain: disputed-territory vs undisputed-territory\".

Rules:
- Only flag a real contradiction (self or pair) if positions are factually incompatible.
- A conditional conflict is NOT a real disagreement — it just needs a qualifier surfaced.
- Each pair contradiction must cite at least one doc_id on each side.
- For self conflicts, doc_ids_a and doc_ids_b point to the SAME doc_id (the self-contradicting source).
- confidence is your 0-1 calibration that this is a real conflict of the stated kind.
- dominant_kind summarises the overall verdict across all contradictions:
    * \"none\" when contradictions is empty.
    * \"self\", \"pair\", or \"conditional\" matching the most severe kind found
      (severity order: self > pair > conditional).
- Output ONLY valid JSON. No preamble, no markdown.

Schema:
{{
  \"has_conflict\": bool,
  \"dominant_kind\": \"self\" | \"pair\" | \"conditional\" | \"none\",
  \"conflict_summary\": \"one neutral sentence or null\",
  \"contradictions\": [
    {{
      \"claim\": \"the disputed fact\",
      \"kind\": \"self\" | \"pair\" | \"conditional\",
      \"doc_ids_a\": [\"doc_1\"],
      \"position_a\": \"...\",
      \"doc_ids_b\": [\"doc_3\"],
      \"position_b\": \"...\",
      \"qualifier\": \"year\" | \"sub-domain: ...\" | null,
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
