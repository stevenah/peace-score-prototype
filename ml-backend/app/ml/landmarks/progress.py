"""Progress lines an eval subprocess prints for ``app.services.evals``.

One line per update on stdout: ``@@progress {"done": 3, "total": 10, "label": "clips"}``.
The runner reads the latest one and keeps these lines out of the log it shows.
"""

from __future__ import annotations

import json

PREFIX = "@@progress "


def report(done: int, total: int, label: str = "") -> None:
    print(PREFIX + json.dumps({"done": int(done), "total": int(total), "label": label}), flush=True)


def parse(line: str) -> dict | None:
    """The update a log line carries, or None for an ordinary line."""
    if not line.startswith(PREFIX):
        return None
    try:
        data = json.loads(line[len(PREFIX):])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
