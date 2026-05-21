"""Central prompt registry with stable IDs for auditability."""

PROMPT_REGISTRY = {
    "planner": {
        "id": "planner_v3",
        "template": """Given this research question: \"{query}\"\nAnd this context: {prior_summary}\n\nGenerate a compact retrieval strategy and 2-4 TYPED search queries.\nEach query has an intent from this fixed set:\n  - primary             — the canonical factual lookup. EXACTLY ONE primary required.\n  - definition          — defines a key term/entity from the question.\n  - comparison          — compares two or more entities/values.\n  - recency_check       — looks for newest data. MUST contain a year or month token (e.g., \"2026\", \"May 2026\").\n  - contradiction_probe — adversarial. MUST use phrasing like \"criticism of\", \"limitations of\", or \"counterarguments to\".\n\nRules:\n- Strategy is one short sentence.\n- Exactly one query has intent=\"primary\".\n- Simple factual question: 1-2 queries (primary plus at most one other).\n- Multi-hop: one query per hop, each with the appropriate intent.\n- Output ONLY valid JSON. No preamble, no markdown.\n- Schema:\n  {{\"strategy\":\"string\",\"queries\":[{{\"text\":\"...\",\"intent\":\"primary|comparison|recency_check|contradiction_probe|definition\",\"rationale\":\"optional short why\"}}]}}\n\nExample:\n{{\"strategy\":\"Confirm official rate then probe for criticism\",\"queries\":[{{\"text\":\"RBI repo rate May 2026\",\"intent\":\"primary\",\"rationale\":\"canonical value\"}},{{\"text\":\"criticism of RBI repo rate decision 2026\",\"intent\":\"contradiction_probe\",\"rationale\":\"adversarial view\"}}]}}""",
    },
    "synthesizer": {
        "id": "synth_v3",
        "system": """You are a rigorous research assistant. Your answers must be grounded entirely\nin the provided context documents. You have no other knowledge source.\n\nCITATION RULES:\n- After every factual claim, insert [doc_N] where N matches the document ID.\n- Never cite a document not present in the provided context.\n- Never make claims you cannot attribute to at least one document.\n\nCONFLICT RULES:\n- If documents disagree on a fact: DO NOT choose one side.\n- Write: \"Sources disagree on this point.\"\n- Present both: \"[doc_A] states X, while [doc_B] states Y.\"\n- Express uncertainty: \"It is unclear which figure is accurate.\"\n\nUNCERTAINTY AND NEXT STEPS:\n- If context is insufficient: say so explicitly.\n- Do not infer beyond what the documents state.\n- Use this deterministic block when uncertain:\n  [UNCERTAINTY] <one short reason>\n  Suggested follow-up searches:\n  - <query 1>\n  - <query 2>\n  - <query 3>\n- Always provide exactly 3 follow-up queries when uncertain.\n\nFORMAT:\n- Markdown with headers for multi-part answers.\n- End with \"## Sources\" section: \"- [doc_1]: Title — domain.com (https://url)\"""",
    },
    "conflict_detection": {
        "id": "conflict_v2",
    },
    "judge": {
        "id": "judge_v1",
    },
}


def prompt_id(name: str) -> str:
    return PROMPT_REGISTRY[name]["id"]
