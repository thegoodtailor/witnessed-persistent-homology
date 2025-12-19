"""witnessed_ph.io

JSON (de)serialisation helpers for diagrams.

The outputs of this library are plain dicts + lists. We still provide helpers to:
- convert numpy arrays to JSONable lists
- save diagrams to disk with provenance metadata intact
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json

from .utils import to_jsonable


def diagram_to_json(diagram: Dict[str, Any], indent: int = 2) -> str:
    """Convert a diagram dict to a JSON string."""
    return json.dumps(to_jsonable(diagram), indent=indent, ensure_ascii=False)


def save_diagram(diagram: Dict[str, Any], path: str | Path, indent: int = 2) -> Path:
    """Save a diagram to JSON on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(diagram_to_json(diagram, indent=indent), encoding="utf-8")
    return path
