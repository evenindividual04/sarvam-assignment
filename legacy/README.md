# Legacy Streamlit UI

This directory holds the original Streamlit-based UI from V1 of the project. It is **no longer the production path** — the active frontend is the Next.js app at `frontend/`.

The code is preserved here for two reasons:

1. **Reproducibility** of the original assignment submission (the V1 demo video used this UI).
2. **Fallback** in case the React frontend is unavailable during a demo recording.

## What's here

| File | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit chat UI (was `app.py` in the repo root). |
| `streamlit_pages/1_Evaluation_Dashboard.py` | Streamlit eval dashboard. |
| `utils/async_bridge.py` | Thread+queue bridge that lets Streamlit consume the async orchestrator's `AsyncIterator`. FastAPI's `StreamingResponse` doesn't need this — it consumes async generators natively. |
| `requirements-legacy.txt` | Streamlit pin, kept out of the main `requirements.txt`. |

## How to run (if you really want to)

```bash
pip install -r legacy/requirements-legacy.txt
# Streamlit expects app.py at root; symlink or copy:
cp legacy/streamlit_app.py app.py
mkdir -p pages && cp legacy/streamlit_pages/*.py pages/
streamlit run app.py
```

The current FastAPI backend (`main.py`) still exposes every endpoint the legacy UI used (`/chat/stream` alias is preserved), so the legacy UI will work against the live backend.

## When to delete this directory

After the assignment is submitted and reviewed, this directory has no production purpose. Delete it any time the repo gets a cleanup pass.
