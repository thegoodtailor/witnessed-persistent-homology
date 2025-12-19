"""witnessed_ph

Reproducible reference implementation for the book's Chapter 4 experiments.

The public API is intentionally small:

- default_config
- analyse_text_single_slice
- analyse_conversation
- analyse_conversation_slices / analyse_conversation_dynamics
- list_bars_by_persistence, print_diagram_summary
- diagram_to_json, save_diagram
"""

from .config import default_config
from .analysis import analyse_text_single_slice, analyse_conversation, list_bars_by_persistence, print_diagram_summary
from .temporal import analyse_conversation_slices, analyse_conversation_dynamics
from .io import diagram_to_json, save_diagram

__all__ = [
    "default_config",
    "analyse_text_single_slice",
    "analyse_conversation",
    "analyse_conversation_slices",
    "analyse_conversation_dynamics",
    "list_bars_by_persistence",
    "print_diagram_summary",
    "diagram_to_json",
    "save_diagram",
]
