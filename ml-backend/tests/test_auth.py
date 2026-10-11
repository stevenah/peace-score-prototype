"""Shared-key auth on the analyze routes and the live socket (app/api/auth.py)."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.websockets import WebSocketDisconnect

from app.api import auth
from app.api import websocket as ws_mod
from app.config import settings

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "live_token.example.json"
KEY = "test-key"
WS = "/api/v1/ws/live"


def _token(expires: float, key: str = KEY) -> str:
    digits = str(int(expires))
    return f"{digits}.{hmac.new(key.encode(), digits.encode(), hashlib.sha256).hexdigest()}"


@pytest.fixture()
def keyed(monkeypatch):
    """Turn auth on (conftest pins it off for every other test)."""
    monkeypatch.setattr(settings, "api_key", KEY)


# -- HTTP routes -------------------------------------------------------------


@pytest.mark.parametrize("headers", [{}, {"x-api-key": "wrong"}, {"x-api-key": "clé"}])
def test_analyze_routes_refuse_a_missing_or_wrong_key(client, keyed, headers):
    encoded = {k: v.encode() for k, v in headers.items()}
    assert client.get("/api/v1/analyze/some-id", headers=encoded).status_code == 401
    assert client.post("/api/v1/analyze/s3", json={"s3_key": "uploads/x.mp4"}, headers=encoded).status_code == 401
    assert client.post("/api/v1/analyze/frame", headers=encoded).status_code == 401


def test_analyze_routes_accept_the_key(client, keyed, store):
    headers = {"x-api-key": KEY}
    with patch("app.api.routes.job_store", store):
        created = client.post("/api/v1/analyze/s3", json={"s3_key": "uploads/x.mp4"}, headers=headers)
        assert created.status_code == 200
        fetched = client.get(f"/api/v1/analyze/{created.json()['analysis_id']}", headers=headers)
    assert fetched.status_code == 200


def test_health_needs_no_key(client, keyed):
    """Fly's health check cannot send one."""
    assert client.get("/api/v1/health").status_code == 200


def test_the_key_stays_out_of_the_startup_log(keyed):
    summary = settings.public_summary()
    assert "api_key" not in summary
    assert KEY not in json.dumps(summary)
    assert summary["api_key_set"] is True


# -- Live socket -------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "",
        "?token=",
        "?token=not-a-token",
        # Built at collection time: an hour ahead so these two are still
        # unexpired when they run, and are refused for their signature alone.
        f"?token={int(time.time()) + 3600}.{'0' * 64}",
        f"?token={_token(time.time() + 3600, key='another-key')}",
        f"?token={_token(time.time() - 1)}",
    ],
    ids=["none", "empty", "malformed", "bad-signature", "wrong-key", "expired"],
)
def test_socket_refuses_a_bad_token(client, keyed, query):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(WS + query):
            pass
    # Refused before admission: it never held one of the live slots.
    assert ws_mod.live_socket_count() == 0


def test_socket_accepts_a_valid_token(client, keyed):
    with client.websocket_connect(f"{WS}?token={_token(time.time() + 60)}"):
        assert ws_mod.live_socket_count() == 1


# -- Contract with the web app -----------------------------------------------


@pytest.mark.skipif(not CONTRACT.exists(), reason="contracts/ lives at the monorepo root")
def test_accepts_the_token_the_web_app_signs(monkeypatch):
    contract = json.loads(CONTRACT.read_text())
    monkeypatch.setattr(settings, "api_key", contract["key"])

    monkeypatch.setattr(auth.time, "time", lambda: contract["expires"] - 1)
    assert auth.valid_live_token(contract["token"])

    monkeypatch.setattr(auth.time, "time", lambda: contract["expires"] + 1)
    assert not auth.valid_live_token(contract["token"])
