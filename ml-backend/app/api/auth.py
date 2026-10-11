"""Shared-key auth between the web app and this service.

The web app sends the key as ``x-api-key`` on its HTTP calls. The live socket
is opened by the browser, which cannot set headers, so it carries a short-lived
token the web app signs with the same key (``signLiveToken`` in
``lib/ml-backend.ts``): ``<unix expiry>.<hex hmac-sha256 of the expiry>``.

With no key configured both checks pass.
"""

from __future__ import annotations

import hashlib
import hmac
import time

from fastapi import Header, HTTPException

from app.config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Route dependency: 401 unless the request carries the shared key."""
    key = settings.api_key
    if key and not hmac.compare_digest((x_api_key or "").encode(), key.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def valid_live_token(token: str | None) -> bool:
    """Whether ``token`` is an unexpired live-socket token signed with the key."""
    key = settings.api_key
    if not key:
        return True
    expires, _, signature = (token or "").partition(".")
    expected = hmac.new(key.encode(), expires.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature.encode(), expected.encode()):
        return False
    # Only now is the expiry trusted: the web app signed it, so it is an integer.
    return int(expires) >= time.time()
