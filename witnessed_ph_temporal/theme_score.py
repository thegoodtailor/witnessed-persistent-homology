"""
Witnessed Persistent Homology: Theme Score Visualization
=========================================================

"Where Themes Learn to Breathe"

Generates the musical-stave visualization showing bar evolution
across time slices with spawn/carry/drift/rupture/re-entry events.

The Theme Score displays:
- Each time slice as a vertical section
- Bars as boxes with their state ([spawned], [drifting], [carried], etc.)
- Ruptures as double-bordered boxes
- Re-entries marked with ★
- Event symbols in the margin: • spawn, → carry, ↝ drift, × rupture, ★ re-entry

References:
    Chapter 4, "Bars: How Themes Learn to Breathe"
"""

from typing import Dict, List, Optional, Set, Tuple
from schema import (
    BarDynamicsResult, BarJourney, BarEvent, DiagramSlice,
    BarRef, BarDynamicsStats
)


# =============================================================================
# Event Symbol Mapping
# =============================================================================

EVENT_SYMBOLS = {
    "spawn": "•",
    "carry_by_name": "→",
    "drift": "↝",
    "rupture_out": "×",
    "rupture_in": "•",      # Same as spawn visually
    "reentry": "★",
}


# =============================================================================
# Bar State at Each Slice
# =============================================================================

def _get_bar_states_at_slice(
    result: BarDynamicsResult,
    tau: int
) -> List[Dict]:
    """
    Get all bar states at a given time slice.
    
    Returns list of dicts with:
    - bar_id, name (derived from witnesses), persistence
    - state: "spawned", "carried", "drifting", "ruptured", "reentered"
    - witnesses: list of token surface forms
    - journey_id: which journey this bar belongs to
    - is_reentry: whether this specific appearance is a re-entry
    """
    bar_states = []
    
    # Find the diagram slice
    diagram_slice = None
    for s in result["slices"]:
        if s["time"] == tau:
            diagram_slice = s
            break
    
    if diagram_slice is None:
        return bar_states
    
    bars = diagram_slice["diagram"].get("bars", [])
    
    # Build a map from (tau, bar_id) to journey and event info
    bar_to_journey = {}  # (tau, bar_id) -> (journey, event_type, is_reentry)
    
    for journey in result["journeys"]:
        for event in journey["events"]:
            # Check if this event involves a bar at this tau
            if event["target"] and event["target"]["time"] == tau:
                bar_id = event["target"]["bar_id"]
                is_reentry = event["event_type"] == "reentry"
                bar_to_journey[(tau, bar_id)] = (journey, event["event_type"], is_reentry)
            
            if event["source"] and event["source"]["time"] == tau:
                bar_id = event["source"]["bar_id"]
                # Check if there's a rupture_out at this tau
                if event["event_type"] == "rupture_out":
                    # Mark the bar as about to rupture
                    if (tau, bar_id) not in bar_to_journey:
                        bar_to_journey[(tau, bar_id)] = (journey, "rupture_out", False)
    
    for bar in bars:
        bar_id = bar["id"]
        witnesses = bar.get("witness", {}).get("tokens", {}).get("surface", [])
        persistence = bar.get("persistence", 0.0)
        
        # Determine state
        key = (tau, bar_id)
        if key in bar_to_journey:
            journey, event_type, is_reentry = bar_to_journey[key]
            
            if event_type == "spawn" or event_type == "rupture_in":
                state = "spawned"
            elif event_type == "carry_by_name":
                state = "carried"
            elif event_type == "drift":
                state = "drifting"
            elif event_type == "reentry":
                state = "re-entered"
            elif event_type == "rupture_out":
                state = "ruptured"
            else:
                state = "unknown"
            
            journey_id = journey["root"]["bar_id"]
        else:
            state = "spawned"  # Default for first appearance
            journey_id = bar_id
            is_reentry = False
        
        # Create a name from top witnesses
        if witnesses:
            name = witnesses[0].upper() if len(witnesses[0]) > 2 else "THEME"
        else:
            name = "THEME"
        
        bar_states.append({
            "bar_id": bar_id,
            "name": name,
            "persistence": persistence,
            "state": state,
            "witnesses": witnesses,
            "journey_id": journey_id,
            "is_reentry": is_reentry,
        })
    
    return bar_states


def _get_events_at_slice(
    result: BarDynamicsResult,
    tau: int
) -> Tuple[Set[str], Set[str]]:
    """
    Get incoming and outgoing event types at a slice.
    
    Returns (incoming_events, outgoing_events) as sets of event symbols.
    """
    incoming = set()  # Events where this slice is the target
    outgoing = set()  # Events where this slice is the source (for ruptures)
    
    for journey in result["journeys"]:
        for event in journey["events"]:
            if event["target"] and event["target"]["time"] == tau:
                incoming.add(EVENT_SYMBOLS.get(event["event_type"], "?"))
            
            if event["source"] and event["source"]["time"] == tau:
                if event["event_type"] == "rupture_out":
                    outgoing.add(EVENT_SYMBOLS.get(event["event_type"], "?"))
    
    return incoming, outgoing


# =============================================================================
# Persistence Bar Rendering
# =============================================================================

def _render_persistence_bar(persistence: float, width: int = 20) -> str:
    """Render a persistence value as a filled bar."""
    import numpy as np
    
    if np.isinf(persistence):
        persistence = 1.0
    
    # Clamp to [0, 1]
    p = max(0.0, min(1.0, persistence))
    filled = int(p * width)
    
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} ({persistence:.3f})"


# =============================================================================
# Box Drawing
# =============================================================================

def _draw_bar_box(
    name: str,
    state: str,
    witnesses: List[str],
    persistence: float,
    is_ruptured: bool = False,
    is_reentry: bool = False,
    width: int = 45
) -> List[str]:
    """
    Draw a box for a single bar.
    
    Ruptured bars get double borders.
    Re-entries get a ★ marker.
    """
    lines = []
    
    # State label
    state_label = f"[{state}]"
    if is_reentry:
        name = f"{name} ★"
    
    # Witness preview
    witness_preview = " ".join(witnesses[:5])
    if len(witnesses) > 5:
        witness_preview += "..."
    
    # Persistence bar
    pers_line = f"persistence: {_render_persistence_bar(persistence, 12)}"
    
    # Content lines
    content = [
        f"{name} {state_label}",
        witness_preview[:width - 4],
        pers_line,
    ]
    
    # Determine max content width
    inner_width = max(len(line) for line in content)
    inner_width = max(inner_width, width - 4)
    
    if is_ruptured:
        # Double border for ruptures
        lines.append("╔" + "═" * (inner_width + 2) + "╗")
        for line in content:
            padded = line.ljust(inner_width)
            lines.append(f"║ {padded} ║")
        lines.append("╚" + "═" * (inner_width + 2) + "╝")
    else:
        # Single border for normal bars
        lines.append("┌" + "─" * (inner_width + 2) + "┐")
        for line in content:
            padded = line.ljust(inner_width)
            lines.append(f"│ {padded} │")
        lines.append("└" + "─" * (inner_width + 2) + "┘")
    
    return lines


def _draw_rupture_notice(
    name: str,
    witnesses: List[str],
    width: int = 45
) -> List[str]:
    """Draw a rupture notice box."""
    inner_width = width - 4
    
    title = f"{name} - RUPTURED"
    witness_str = f"witnesses dispersed: {', '.join(witnesses[:3])}"
    if len(witnesses) > 3:
        witness_str += "..."
    
    lines = [
        "╔" + "═" * (inner_width + 2) + "╗",
        f"║ {title.ljust(inner_width)} ║",
        f"║ {witness_str[:inner_width].ljust(inner_width)} ║",
        "╚" + "═" * (inner_width + 2) + "╝",
    ]
    
    return lines


# =============================================================================
# Slice Rendering
# =============================================================================

def _render_slice(
    result: BarDynamicsResult,
    tau: int,
    slice_label: str,
    max_bars: int = 6
) -> List[str]:
    """
    Render a single time slice as ASCII.
    
    Format:
        τ=0 | User
            | • ×
            |
            | [bar box]
            |
            | [bar box]
    """
    lines = []
    
    # Get bar states and events
    bar_states = _get_bar_states_at_slice(result, tau)
    incoming, outgoing = _get_events_at_slice(result, tau)
    
    # Header line
    header = f"τ={tau} | {slice_label}"
    lines.append(header)
    
    # Event symbols line
    symbols = " ".join(sorted(incoming | outgoing))
    if symbols:
        lines.append(f"    | {symbols}")
    else:
        lines.append("    |")
    
    # Render each bar
    bars_shown = 0
    ruptured_this_slice = []
    
    for bar_state in bar_states:
        if bars_shown >= max_bars:
            remaining = len(bar_states) - bars_shown
            lines.append(f"    | ... ({remaining} more bars)")
            break
        
        lines.append("    |")
        
        is_ruptured = bar_state["state"] == "ruptured"
        is_reentry = bar_state["is_reentry"] or bar_state["state"] == "re-entered"
        
        if is_ruptured:
            # Show rupture notice
            ruptured_this_slice.append(bar_state)
            box_lines = _draw_rupture_notice(
                bar_state["name"],
                bar_state["witnesses"]
            )
        else:
            box_lines = _draw_bar_box(
                bar_state["name"],
                bar_state["state"],
                bar_state["witnesses"],
                bar_state["persistence"],
                is_ruptured=False,
                is_reentry=is_reentry
            )
        
        for box_line in box_lines:
            lines.append(f"    | {box_line}")
        
        bars_shown += 1
    
    return lines


# =============================================================================
# Main Theme Score Renderer
# =============================================================================

def render_theme_score(
    result: BarDynamicsResult,
    title: str = "THEME SCORE",
    subtitle: str = "Where Themes Learn to Breathe",
    max_bars_per_slice: int = 6
) -> str:
    """
    Render the complete Theme Score visualization.
    
    Parameters
    ----------
    result : BarDynamicsResult
        Output from analyse_slices() or analyse_diagram_slices()
    title : str
        Title of the score
    subtitle : str
        Subtitle (the famous tagline)
    max_bars_per_slice : int
        Maximum bars to show per slice before truncating
    
    Returns
    -------
    ASCII string of the complete theme score.
    
    Example output:
    ```
                        THEME SCORE
                Where Themes Learn to Breathe
    
    τ=0 | User
        | • ×
        |
        | ┌───────────────────────────────────────┐
        | │ THINKING [spawned]                    │
        | │ User thinking climate change lot      │
        | │ persistence: ████████████ (0.367)     │
        | └───────────────────────────────────────┘
    
    τ=1 | Assistant
        | → ↝
        |
        | ┌───────────────────────────────────────┐
        | │ THINKING [carried]                    │
        | │ thinking climate challenges           │
        | │ persistence: ████████████ (0.401)     │
        | └───────────────────────────────────────┘
    ```
    """
    lines = []
    
    # Header
    lines.append("")
    lines.append(f"{title:^60}")
    lines.append(f"{subtitle:^60}")
    lines.append("")
    
    # Render each slice
    for i, slice_data in enumerate(result["slices"]):
        tau = slice_data["time"]
        label = slice_data.get("label", f"Slice {tau}")
        
        # Extract speaker from label if present
        if "(" in label and ")" in label:
            speaker = label[label.index("(") + 1 : label.index(")")]
        elif "User" in label or "Assistant" in label:
            speaker = "User" if "User" in label else "Assistant"
        else:
            speaker = label
        
        slice_lines = _render_slice(
            result, tau, speaker, max_bars_per_slice
        )
        lines.extend(slice_lines)
        lines.append("")
    
    # Legend
    lines.append("-" * 60)
    lines.append("Legend: • spawn  → carry  ↝ drift  × rupture  ★ re-entry")
    
    return "\n".join(lines)


def print_theme_score(
    result: BarDynamicsResult,
    **kwargs
) -> None:
    """Print the theme score to stdout."""
    print(render_theme_score(result, **kwargs))


# =============================================================================
# Journey Timeline (Alternative View)
# =============================================================================

def render_journey_timeline(
    journey: BarJourney,
    max_tau: int = 20
) -> str:
    """
    Render a single journey as a horizontal timeline.
    
    Example:
        CLIMATE [climate, change, carbon...]
        τ: 0   1   2   3   4   5   6   7   8   9
           S───C───C───D───R   .   .   E───C───→
    """
    lines = []
    
    # Header with witnesses
    witnesses = journey["root_witnesses"][:4]
    name = witnesses[0].upper() if witnesses else "THEME"
    witness_str = ", ".join(witnesses)
    if len(journey["root_witnesses"]) > 4:
        witness_str += "..."
    
    lines.append(f"{name} [{witness_str}]")
    
    # Build event map
    event_at_tau = {}  # tau -> symbol
    
    for event in journey["events"]:
        if event["target"]:
            tau = event["target"]["time"]
            etype = event["event_type"]
            
            if etype == "spawn" or etype == "rupture_in":
                event_at_tau[tau] = "S"
            elif etype == "carry_by_name":
                event_at_tau[tau] = "C"
            elif etype == "drift":
                event_at_tau[tau] = "D"
            elif etype == "reentry":
                event_at_tau[tau] = "E"
        
        if event["source"] and event["event_type"] == "rupture_out":
            tau = event["time_to"]
            event_at_tau[tau] = "R"
    
    # Determine tau range
    all_taus = list(event_at_tau.keys())
    if not all_taus:
        return "\n".join(lines)
    
    min_tau = min(all_taus)
    max_display = min(max(all_taus) + 3, max_tau)
    
    # Time axis
    tau_line = "τ: " + "   ".join(f"{t:2d}" for t in range(min_tau, max_display + 1))
    lines.append(tau_line)
    
    # Event line
    event_line = "   "
    prev_symbol = None
    
    for tau in range(min_tau, max_display + 1):
        if tau in event_at_tau:
            sym = event_at_tau[tau]
            if prev_symbol and prev_symbol not in ["R", "."]:
                event_line = event_line[:-1] + "─" + sym + "──"
            else:
                event_line += f" {sym}  "
            prev_symbol = sym
        else:
            if prev_symbol and prev_symbol not in ["R", "."]:
                event_line += "────"
                prev_symbol = "-"
            else:
                event_line += " .  "
                prev_symbol = "."
    
    # Add arrow if still alive
    if journey["state"] == "alive":
        event_line = event_line.rstrip() + "→"
    
    lines.append(event_line.rstrip())
    
    return "\n".join(lines)


def print_all_journey_timelines(
    result: BarDynamicsResult,
    top_n: int = 10
) -> None:
    """Print timeline view for top N journeys."""
    # Sort by event count
    journeys = sorted(
        result["journeys"],
        key=lambda j: len(j["events"]),
        reverse=True
    )[:top_n]
    
    print("\n" + "=" * 60)
    print("JOURNEY TIMELINES")
    print("=" * 60)
    
    for i, journey in enumerate(journeys):
        print(f"\n{i + 1}. ", end="")
        print(render_journey_timeline(journey))
        print(f"   State: {journey['state']}")
