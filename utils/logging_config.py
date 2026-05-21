"""JSON structured logging. Call setup_logging() once at app startup."""
import json
import logging
import os


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "module": getattr(record, "component", getattr(record, "module", record.name)),
            "message": record.getMessage(),
            "session_id": getattr(record, "session_id", None),
            "turn_id": getattr(record, "turn_id", None),
            "duration_ms": getattr(record, "duration_ms", None),
        }
        return json.dumps({k: v for k, v in base.items() if v is not None})


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
