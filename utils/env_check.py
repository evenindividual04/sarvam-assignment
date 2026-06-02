import os
from dotenv import load_dotenv

load_dotenv()

# Core required keys — accept both singular (legacy) and plural (multi-key
# rotator) forms. The provider_router reads GROQ_API_KEYS / GEMINI_API_KEYS;
# the singular form is kept for backward compat with simple deployments.
# validate() passes as long as at least one form is present per provider.
_GROQ_KEYS = ["GROQ_API_KEY", "GROQ_API_KEYS"]
_GEMINI_KEYS = ["GEMINI_API_KEY", "GEMINI_API_KEYS"]

# At least one search provider key must be present (Parallel, Tavily, or Serper)
SEARCH_KEYS = ["PARALLEL_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"]


def validate() -> None:
    """Validate environment variables.

    - Requires GITHUB_TOKEN.
    - Requires at least one of GROQ_API_KEY / GROQ_API_KEYS.
    - Requires at least one of GEMINI_API_KEY / GEMINI_API_KEYS.
    - Requires at least one search provider key (PARALLEL_API_KEY,
      TAVILY_API_KEY, or SERPER_API_KEY).
    """
    msgs: list[str] = []

    if not any(os.getenv(k) for k in _GROQ_KEYS):
        msgs.append(f"Missing Groq key — set one of: {_GROQ_KEYS}")
    if not any(os.getenv(k) for k in _GEMINI_KEYS):
        msgs.append(f"Missing Gemini key — set one of: {_GEMINI_KEYS}")
    if not os.getenv("GITHUB_TOKEN"):
        msgs.append("Missing required env var: GITHUB_TOKEN")
    if not any(os.getenv(k) for k in SEARCH_KEYS):
        msgs.append(
            f"At least one search provider key required: {SEARCH_KEYS}"
        )

    if msgs:
        msgs.append("Copy .env.example to .env and fill in the values.")
        raise EnvironmentError("\n".join(msgs))
