"""Central prompt registry with stable IDs for auditability."""

PROMPT_REGISTRY = {
    "planner": {
        "id": "planner_v2",
        "template": """Given this research question: \"{query}\"\nAnd this context: {prior_summary}\n\nGenerate a compact retrieval strategy and 2-4 search queries for comprehensive coverage.\nRules:\n- Strategy should be one short sentence.\n- Each query targets a different angle (definition, current data, comparison, expert view)\n- Simple factual questions: 1-2 queries max. Don't over-search.\n- Multi-hop questions: cover each hop with a separate query.\n- Output ONLY valid JSON object. No preamble.\n- Schema: {\"strategy\":\"string\",\"queries\":[\"q1\",\"q2\"]}\nExample: {\"strategy\":\"Find official value and confirm from policy notes\",\"queries\":[\"RBI repo rate May 2026\",\"RBI monetary policy statement 2026\"]}""",
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
