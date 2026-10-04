"""The Python station taxonomy must match contracts/stations.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ml.landmarks.stations import SCHEMA, STATIONS

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "stations.json"


@pytest.mark.skipif(not CONTRACT.exists(), reason="contracts/ lives at the monorepo root")
def test_stations_match_contract():
    data = json.loads(CONTRACT.read_text())
    assert data["schema"] == SCHEMA
    ours = [
        {"key": s.key, "esge": s.esge, "folder": s.folder, "region": s.region,
         "label": s.label, "short": s.short}
        for s in STATIONS
    ]
    assert data["stations"] == ours


def test_station_order_is_esge_numbering():
    assert [s.esge for s in STATIONS] == list(range(1, 11))
    assert {s.region for s in STATIONS} == {"esophagus", "stomach", "duodenum"}
