import os
from dotenv import load_dotenv

load_dotenv()

# Core required keys: LLMs and judge/token reporting
REQUIRED = ["GEMINI_API_KEY", "GROQ_API_KEY", "GITHUB_TOKEN"]

# At least one search provider key must be present (Parallel, Tavily, or Serper)
SEARCH_KEYS = ["PARALLEL_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY"]


def validate() -> None:
    """Validate environment variables.

    - Requires all keys in `REQUIRED`.
    - Requires at least one of the `SEARCH_KEYS` to allow the app to perform web search.
    This avoids hard failure when a single optional provider is not configured.
    """
    missing = [k for k in REQUIRED if not os.getenv(k)]
    has_search = any(os.getenv(k) for k in SEARCH_KEYS)
    if missing or not has_search:
        msgs = []
        if missing:
            msgs.append(f"Missing required env vars: {missing}")
        if not has_search:
            msgs.append(
                f"At least one search provider key required: {SEARCH_KEYS} (set PARALLEL_API_KEY or TAVILY_API_KEY or SERPER_API_KEY)"
            )
        msgs.append("Copy .env.example to .env and fill in the values.")
        raise EnvironmentError("\n".join(msgs))
