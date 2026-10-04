"""Patient-ID and filename-suffix parsing for landmark clip names. Pure stdlib.

Clip names are hand-typed, e.g. ``HT123-eso1 (2).mp4``: a study pseudonym
``HT###``, a station suffix, an optional take number and optional copy
markers added by the file manager. The audit found every dirty variant
below, and each has a unit test (the numbers here and in the tests are
invented):

    HT2O4-inc          letter O typed for zero            -> HT204
    HT23-bd            two digits                         -> HT023 (padded)
    HT0123-a           four digits                        -> ambiguous: needs an override
    HT123eso2          no separator                       -> HT123, eso2
    HT123a-2           take after a hyphen                -> a, take 2
    HT-123-bd          separator before the digits        -> HT123
    HT123- bd          stray space                        -> HT123, bd
    HT123-a (2)        copy marker, also "(2)", "(2)(1)"  -> copies (2,), (2, 1)
    HT123-eso1(10      truncated copy marker              -> copy 10
    HT123-eso11        eso1 + take 1 (eso is always followed by 1|2)
    HT123-lc+inc       combo of two stations              -> is_combo
    HT-inc+a           no ID at all                       -> unparsed: needs an override

Four-digit and missing IDs are never guessed (a four-digit ID can be a
three-digit one with a doubled or appended zero). Such names, and names
whose OSD date contradicts their ID, are resolved by a reviewable override
table. The concrete table is dataset-derived, so it lives in the gitignored
``data/landmarks_private/overrides/patient_id_overrides.csv`` rather than in
this public module; ``load_id_overrides`` reads it.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.ml.landmarks.stations import STATION_ORDER

# Canonical suffix token -> station key. "x" marks an unknown/other view.
SUFFIX_TO_STATION: dict[str, str | None] = {
    "eso1": "esophagus_proximal",
    "eso2": "esophagus_distal",
    "zline": "z_line",
    "bd": "duodenal_bulb",
    "dd": "duodenum_descending",
    "a": "antrum",
    "cfi": "cardia_fundus_retroflex",
    "lc": "lesser_curvature_retroflex",
    "inc": "incisura",
    "gc": "corpus_greater_curvature",
    "x": None,
}
SUFFIX_ALIASES = {"z": "zline", "cf": "cfi", "b": "bd", "db": "bd", "d": "dd", "in": "inc", "l": "lc"}
assert all(v is None or v in STATION_ORDER for v in SUFFIX_TO_STATION.values())

# "HT", optional separators, then 1-4 digits (letter O accepted as zero, at
# least one real digit), not followed by another digit/O.
_ID_RE = re.compile(r"^HT[\s_-]*(?P<id>(?=[0-9O]*\d)[0-9O]{1,4})(?![0-9O])(?P<rest>.*)$", re.I)
_NO_ID_RE = re.compile(r"^HT[\s_-]*(?P<rest>.*)$", re.I)
# Trailing copy markers " (2)", "(1)", and the truncated "(10".
_COPY_RE = re.compile(r"\s*\((\d+)\)?\s*$")
_TOKEN_RE = re.compile(r"^(eso[12]|zline|cfi|inc|gc|lc|bd|db|dd|cf|in|a|b|d|l|x|z)(\d*)$")

UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ParsedName:
    file: str
    raw_id: str | None  # the ID token as typed ("O12", "0750"), None if absent
    suffix: str  # canonical tokens joined by "+", e.g. "a", "lc+inc"; "" if none
    suffix_parts: tuple[str, ...] = ()
    suffix_stations: tuple[str | None, ...] = ()  # per part; None = unknown token
    suffix_class: str = ""  # the station when the suffix is ONE known station
    take: int | None = None  # "a1" -> 1, "a-2" -> 2
    copies: tuple[int, ...] = ()  # "(2)(1)" -> (2, 1)
    unknown_tokens: tuple[str, ...] = field(default=())

    @property
    def is_combo(self) -> bool:
        return len(self.suffix_parts) > 1


def _split_copies(stem: str) -> tuple[str, tuple[int, ...]]:
    copies: list[int] = []
    while m := _COPY_RE.search(stem):
        if m.start() == 0:
            break
        copies.insert(0, int(m.group(1)))
        stem = stem[: m.start()]
    return stem, tuple(copies)


def parse_suffix(rest: str) -> tuple[tuple[str, ...], tuple[str, ...], int | None]:
    """'-lc+inc' -> (('lc', 'inc'), unknown, take). Take numbers attach to a token."""
    s = rest.strip().lower().strip(" _-").replace("z-line", "zline")
    parts: list[str] = []
    unknown: list[str] = []
    take: int | None = None
    for chunk in (c for c in re.split(r"\+", s) if c.strip(" _-")):
        for piece in (p for p in re.split(r"[\s_-]+", chunk) if p):
            if piece.isdigit():  # "a-2": take after a hyphen
                take = int(piece)
                continue
            m = _TOKEN_RE.match(piece)
            if not m:
                unknown.append(piece)
                continue
            tok = SUFFIX_ALIASES.get(m.group(1), m.group(1))
            if m.group(2):
                take = int(m.group(2))
            if tok not in parts:
                parts.append(tok)
    return tuple(parts), tuple(unknown), take


def parse_clip_name(file: str) -> ParsedName:
    """Split a clip file name into ID token, suffix tokens, take and copies."""
    stem = re.sub(r"\.mp4$", "", file.strip(), flags=re.I)
    stem, copies = _split_copies(stem)
    m = _ID_RE.match(stem)
    raw_id, rest = (m.group("id"), m.group("rest")) if m else (None, stem)
    if m is None and (m2 := _NO_ID_RE.match(stem)):
        rest = m2.group("rest")
    parts, unknown, take = parse_suffix(rest)
    stations = tuple(SUFFIX_TO_STATION.get(p) for p in parts)
    single = stations[0] if len(parts) == 1 and not unknown else None
    return ParsedName(
        file=file,
        raw_id=raw_id,
        suffix="+".join(parts + unknown),
        suffix_parts=parts + unknown,
        suffix_stations=stations + (None,) * len(unknown),
        suffix_class=single or "",
        take=take,
        copies=copies,
        unknown_tokens=unknown,
    )


def normalize_patient_id(raw_id: str | None) -> tuple[str | None, str]:
    """Generic rule only (no overrides): returns (patient_id, id_resolution).

    id_resolution: regex | regex_letter_o | regex_padded | ambiguous | unparsed.
    """
    if not raw_id:
        return None, "unparsed"
    digits = raw_id.upper().replace("O", "0")
    if not digits.isdigit():
        return None, "unparsed"
    if len(digits) == 3:
        return f"HT{digits}", "regex_letter_o" if "O" in raw_id.upper() else "regex"
    if len(digits) < 3:
        return f"HT{int(digits):03d}", "regex_padded"
    return None, "ambiguous"


@dataclass(frozen=True)
class IdOverride:
    match_type: str  # "file" (exact file name) | "raw_id" (ID token as typed)
    match: str
    patient_id: str  # "HT###" or "uncertain"
    reason: str = ""
    reviewer: str = ""


@dataclass
class IdOverrides:
    by_file: dict[str, IdOverride] = field(default_factory=dict)
    by_raw_id: dict[str, IdOverride] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows) -> "IdOverrides":
        out = cls()
        for r in rows:
            o = IdOverride(r["match_type"].strip(), r["match"].strip(), r["patient_id"].strip(),
                           r.get("reason", "").strip(), r.get("reviewer", "").strip())
            if o.patient_id != UNCERTAIN and not re.fullmatch(r"HT\d{3}", o.patient_id):
                raise ValueError(f"bad override patient_id {o.patient_id!r} for {o.match!r}")
            if o.match_type == "file":
                out.by_file[o.match] = o
            elif o.match_type == "raw_id":
                out.by_raw_id[o.match.upper()] = o
            else:
                raise ValueError(f"bad override match_type {o.match_type!r}")
        return out


OVERRIDE_FIELDS = ["match_type", "match", "patient_id", "reason", "reviewer"]


def load_id_overrides(path: Path) -> IdOverrides:
    if not path.exists():
        return IdOverrides()
    with open(path, newline="", encoding="utf-8") as fh:
        return IdOverrides.from_rows(csv.DictReader(fh))


def resolve_patient_id(p: ParsedName, overrides: IdOverrides | None = None) -> tuple[str | None, str]:
    """(patient_id or None, id_resolution). File overrides beat raw-ID overrides
    beat the generic rule; an override to "uncertain" yields (None, "uncertain")."""
    overrides = overrides or IdOverrides()
    o = overrides.by_file.get(p.file) or (overrides.by_raw_id.get(p.raw_id.upper()) if p.raw_id else None)
    if o is not None:
        return (None, UNCERTAIN) if o.patient_id == UNCERTAIN else (o.patient_id, "override")
    return normalize_patient_id(p.raw_id)
