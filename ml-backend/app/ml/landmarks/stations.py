"""ESGE upper-GI photodocumentation stations (schema esge10.v1).

Mirrors ``contracts/stations.json`` at the repo root, which the frontend reads
too; ``tests/test_contract.py`` keeps them in sync. The order is the ESGE
proposed photo order and is the index order of every per-station array on the
wire (``probs``, ``stations``, ``auto_enabled``) and of the model's logits.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA = "esge10.v1"


@dataclass(frozen=True)
class Station:
    key: str
    esge: int
    folder: str  # class folder name in data/landmarks/
    region: str  # one of the PEACE regions: esophagus | stomach | duodenum
    label: str
    short: str


STATIONS: tuple[Station, ...] = (
    Station("esophagus_proximal", 1, "Proximal esophagus", "esophagus",
            "Proximal esophagus", "Prox. esophagus"),
    Station("esophagus_distal", 2, "Distal esophagus", "esophagus",
            "Distal esophagus", "Dist. esophagus"),
    Station("z_line", 3, "Z-line", "esophagus", "Z-line", "Z-line"),
    Station("duodenal_bulb", 4, "Duodenal bulb", "duodenum",
            "Duodenal bulb", "Bulb"),
    Station("duodenum_descending", 5, "Descending duodenum", "duodenum",
            "Descending duodenum", "D2"),
    Station("antrum", 6, "Antrum", "stomach", "Antrum", "Antrum"),
    Station("cardia_fundus_retroflex", 7, "Cardia, fundus inversion", "stomach",
            "Cardia & fundus (inversion)", "Cardia/fundus"),
    Station("lesser_curvature_retroflex", 8, "Lesser curvature", "stomach",
            "Lesser curvature (partial inversion)", "Lesser curve"),
    Station("incisura", 9, "Incisura angularis", "stomach",
            "Incisura angularis", "Incisura"),
    Station("corpus_greater_curvature", 10, "Gastric corpus", "stomach",
            "Gastric corpus (greater curvature)", "Corpus"),
)

NUM_STATIONS = len(STATIONS)
STATION_ORDER: tuple[str, ...] = tuple(s.key for s in STATIONS)
STATION_INDEX: dict[str, int] = {k: i for i, k in enumerate(STATION_ORDER)}
STATION_REGION: dict[str, str] = {s.key: s.region for s in STATIONS}
FOLDER_TO_STATION: dict[str, str] = {s.folder: s.key for s in STATIONS}
