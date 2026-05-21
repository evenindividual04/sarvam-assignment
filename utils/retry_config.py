"""Shared Tenacity retry decorators."""
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

retry_transient = retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError, Exception)),
    reraise=True,
)
