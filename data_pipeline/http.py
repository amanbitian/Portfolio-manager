"""Shared HTTP helpers: one pooled session, per-host rate limiting, retry/backoff."""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import PipelineConfig

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)


class RetryableHTTPError(RuntimeError):
    """Raised for 429/5xx responses so tenacity retries them."""


class _RateLimiter:
    """Simple monotonic-clock spacing limiter, one lock per host key."""

    def __init__(self) -> None:
        self._next_at: dict[str, float] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def wait(self, key: str, rps: float) -> None:
        if rps <= 0:
            return
        with self._guard:
            lock = self._locks.setdefault(key, threading.Lock())
        min_gap = 1.0 / rps
        with lock:
            now = time.monotonic()
            earliest = self._next_at.get(key, 0.0)
            if now < earliest:
                time.sleep(earliest - now)
            self._next_at[key] = max(now, earliest) + min_gap


class HttpClient:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._limiter = _RateLimiter()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _BROWSER_UA})

    def _rps_for(self, host_key: str) -> float:
        rl = self.cfg.rate_limits.get(host_key)
        return rl.rps if rl else 3.0

    def get(
        self,
        url: str,
        *,
        host_key: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        expect_json: bool = True,
        retries: int | None = None,
        timeout: float | None = None,
    ) -> Any:
        """GET with rate limiting + retry. Returns parsed JSON or the Response."""

        @retry(
            retry=retry_if_exception_type(
                (RetryableHTTPError, requests.exceptions.RequestException)
            ),
            stop=stop_after_attempt(retries or self.cfg.http_retries),
            wait=wait_exponential_jitter(initial=2, max=30),
            reraise=True,
        )
        def _do() -> Any:
            self._limiter.wait(host_key, self._rps_for(host_key))
            resp = self.session.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout or self.cfg.http_timeout,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise RetryableHTTPError(f"{resp.status_code} for {resp.url}")
            resp.raise_for_status()
            return resp.json() if expect_json else resp

        return _do()

    def raw_get(
        self,
        url: str,
        *,
        host_key: str,
        params: dict[str, Any] | None = None,
    ) -> requests.Response:
        """Rate-limited GET returning the raw Response; retries transport errors only."""
        last_exc: Exception | None = None
        for attempt in range(3):
            self._limiter.wait(host_key, self._rps_for(host_key))
            try:
                return self.session.get(
                    url, params=params, timeout=self.cfg.http_timeout
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                time.sleep(3 * (attempt + 1))
        raise last_exc  # type: ignore[misc]

    def prime_cookies(self, url: str, host_key: str) -> None:
        """Hit a homepage first so bot-protected APIs hand us a session cookie."""
        try:
            self._limiter.wait(host_key, self._rps_for(host_key))
            self.session.get(url, timeout=self.cfg.http_timeout)
        except requests.exceptions.RequestException:
            pass
