"""Async quota-exhaustion alerter. POSTs to configured webhook URL.

Env vars:
  QUOTA_WEBHOOK_URL=https://hooks.slack.com/...  (or any URL accepting JSON POST)
  QUOTA_ALERT_THRESHOLD=0.8  (fraction; 0.8 = alert at 80%)
  QUOTA_ALERT_COOLDOWN_S=3600  (don't re-alert same provider within 1hr)

Payload:
  {"text": "<provider> at <pct>% of daily quota (<used>/<limit>)",
   "provider": ..., "fraction": ..., "used": ..., "limit": ..., "unit": ...}

Never raises. Failure modes (no URL set, timeout, non-2xx) are logged at WARNING
and swallowed so the request path can never be blocked by alerting.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_THRESHOLD = 0.8
_DEFAULT_COOLDOWN_S = 3600.0
_HTTP_TIMEOUT_S = 5.0

# In-memory last-alert-time per (provider, unit). Resets on container restart;
# that's acceptable — worst case is one duplicate alert after a redeploy.
_last_alert_at: dict[tuple[str, str], float] = {}
_lock = threading.Lock()


def _threshold() -> float:
    raw = os.environ.get("QUOTA_ALERT_THRESHOLD")
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return _DEFAULT_THRESHOLD


def _cooldown_s() -> float:
    raw = os.environ.get("QUOTA_ALERT_COOLDOWN_S")
    if not raw:
        return _DEFAULT_COOLDOWN_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return _DEFAULT_COOLDOWN_S


def _should_alert(provider: str, unit: str) -> bool:
    """Check cooldown. Mutates last-alert time atomically when returning True."""
    now = time.time()
    cd = _cooldown_s()
    with _lock:
        last = _last_alert_at.get((provider, unit), 0.0)
        if now - last < cd:
            return False
        _last_alert_at[(provider, unit)] = now
        return True


def reset_cooldown_state() -> None:
    """Test helper. Clears the cooldown registry."""
    with _lock:
        _last_alert_at.clear()


async def notify_quota_threshold(
    provider: str,
    used: int,
    limit: int,
    unit: str = "tokens",
    *,
    webhook_url: Optional[str] = None,
) -> None:
    """Fire webhook if fraction > threshold AND cooldown elapsed. Never raises."""
    try:
        if limit <= 0:
            return
        fraction = used / limit
        if fraction < _threshold():
            return
        url = webhook_url if webhook_url is not None else os.environ.get("QUOTA_WEBHOOK_URL", "")
        url = url.strip()
        if not url:
            return
        if not _should_alert(provider, unit):
            return
        pct = round(fraction * 100, 1)
        payload = {
            "text": f"{provider} at {pct}% of daily quota ({used}/{limit} {unit})",
            "provider": provider,
            "fraction": fraction,
            "used": used,
            "limit": limit,
            "unit": unit,
        }
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    logger.warning(
                        "quota_webhook non-2xx for %s: %s",
                        provider,
                        resp.status_code,
                        extra={"component": "quota_webhook"},
                    )
        except Exception as exc:
            logger.warning(
                "quota_webhook POST failed for %s: %s",
                provider,
                exc,
                extra={"component": "quota_webhook"},
            )
    except Exception as exc:  # pragma: no cover — outermost safety net
        logger.warning(
            "quota_webhook unexpected error: %s",
            exc,
            extra={"component": "quota_webhook"},
        )
