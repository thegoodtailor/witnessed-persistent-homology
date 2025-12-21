"""
Witnessed Persistent Homology: Bar Journeys (SWL)
=================================================

Build Step-Witness Logs for bar trajectories.

From Cassie's spec §6:
    "For each root bar (τ_0, b_0):
     1. Inspect its sequence of per-step events
     2. Find the first rupture_out at time τ_k
     3. Look at later slices for any bar b_j such that d_bar(b0, b_j) is small"

Key insight: re-entry compares to the ORIGINAL ROOT, not just the
most recent state before rupture.

References:
    Chapter 4, Definition 4.15 (Step-Witness Log)
    Cassie's Codebase 2 specification §6
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from schema import (
    BarRef, BarEvent, BarJourney, BarMatch, BarSimilarity,
    DiagramSlice, SimilarityConfig, JourneyConfig,
    TimeType, make_bar_ref
)
from matching import find_births, find_similar_bar_across_gap


# =============================================================================
# Event Construction
# =============================================================================

def make_spawn_event(
    bar_ref: BarRef,
    bar: Dict
) -> BarEvent:
    """Create a spawn event."""
    witnesses = bar.get("witness", {}).get("tokens", {}).get("surface", [])
    witness_str = ", ".join(witnesses[:4])
    if len(witnesses) > 4:
        witness_str += "..."
    
    return BarEvent(
        event_type="spawn",
        time_from=bar_ref["time"],
        time_to=bar_ref["time"],
        source=bar_ref,
        target=bar_ref,
        similarity=None,
        generative=False,
        explanation=f"Bar spawned with witnesses: [{witness_str}]"
    )


def make_carry_event(
    from_ref: BarRef,
    to_ref: BarRef,
    sim: BarSimilarity,
    generative: bool
) -> BarEvent:
    """Create a carry-by-name event."""
    return BarEvent(
        event_type="carry_by_name",
        time_from=from_ref["time"],
        time_to=to_ref["time"],
        source=from_ref,
        target=to_ref,
        similarity=sim,
        generative=generative,
        explanation=f"Theme carried with token reuse (J_tok={sim['jaccard_tokens']:.3f})"
    )


def make_drift_event(
    from_ref: BarRef,
    to_ref: BarRef,
    sim: BarSimilarity,
    generative: bool
) -> BarEvent:
    """Create a drift event."""
    return BarEvent(
        event_type="drift",
        time_from=from_ref["time"],
        time_to=to_ref["time"],
        source=from_ref,
        target=to_ref,
        similarity=sim,
        generative=generative,
        explanation=f"Theme drifted semantically (d_sem={sim['d_sem']:.3f}, J_tok={sim['jaccard_tokens']:.3f})"
    )


def make_rupture_out_event(
    from_ref: BarRef,
    to_time: TimeType,
    sim: Optional[BarSimilarity]
) -> BarEvent:
    """Create a rupture-out event."""
    if sim is not None:
        explanation = f"No admissible continuation (d_bar={sim['d_bar']:.3f})"
    else:
        explanation = "No admissible continuation found"
    
    return BarEvent(
        event_type="rupture_out",
        time_from=from_ref["time"],
        time_to=to_time,
        source=from_ref,
        target=None,
        similarity=sim,
        generative=False,
        explanation=explanation
    )


def make_rupture_in_event(
    bar_ref: BarRef,
    bar: Dict
) -> BarEvent:
    """Create a rupture-in event (new bar, no ancestor)."""
    witnesses = bar.get("witness", {}).get("tokens", {}).get("surface", [])
    witness_str = ", ".join(witnesses[:4])
    
    return BarEvent(
        event_type="rupture_in",
        time_from=bar_ref["time"],
        time_to=bar_ref["time"],
        source=bar_ref,
        target=bar_ref,
        similarity=None,
        generative=False,
        explanation=f"New bar with no admissible ancestor: [{witness_str}]"
    )


def make_reentry_event(
    root_ref: BarRef,
    reentry_ref: BarRef,
    rupture_time: TimeType,
    sim: BarSimilarity,
    new_witnesses: List[str]
) -> BarEvent:
    """Create a re-entry event."""
    witness_str = ", ".join(new_witnesses[:4])
    if len(new_witnesses) > 4:
        witness_str += "..."
    
    return BarEvent(
        event_type="reentry",
        time_from=rupture_time,
        time_to=reentry_ref["time"],
        source=root_ref,
        target=reentry_ref,
        similarity=sim,
        generative=False,
        explanation=f"Theme re-enters after rupture, with witnesses: [{witness_str}]"
    )


# =============================================================================
# Generativity Check
# =============================================================================

def is_generative_step(
    bar_from: Dict,
    bar_to: Dict,
    cfg: JourneyConfig
) -> bool:
    """
    Check if a transition is generative (growth).
    
    From Cassie's spec §6.2:
        - persistence_{i+1} >= persistence_i + min_generative_gain
        - witness token count grows by at least min_witness_growth
    """
    pers_from = bar_from.get("persistence", 0.0)
    pers_to = bar_to.get("persistence", 0.0)
    
    # Handle infinite
    if np.isinf(pers_from):
        pers_from = 1.0
    if np.isinf(pers_to):
        pers_to = 1.0
    
    persistence_gain = pers_to - pers_from >= cfg.min_generative_gain
    
    witness_from = len(bar_from.get("witness", {}).get("tokens", {}).get("ids", []))
    witness_to = len(bar_to.get("witness", {}).get("tokens", {}).get("ids", []))
    witness_growth = witness_to - witness_from >= cfg.min_witness_growth
    
    return persistence_gain and witness_growth


# =============================================================================
# Journey Building
# =============================================================================

def build_bar_journeys(
    diagram_slices: List[DiagramSlice],
    matches_per_step: Dict[Tuple[TimeType, TimeType], List[BarMatch]],
    sim_cfg: SimilarityConfig,
    journey_cfg: JourneyConfig
) -> List[BarJourney]:
    """
    Build Step-Witness Logs for all bars.
    
    From Cassie's spec §6:
        - Start from bars at earliest slice (spawns)
        - Track through matches
        - Detect ruptures and re-entries
    
    Parameters
    ----------
    diagram_slices : List of DiagramSlice, sorted by time
    matches_per_step : Dict mapping (τ_i, τ_{i+1}) to List[BarMatch]
    sim_cfg : SimilarityConfig
    journey_cfg : JourneyConfig
    
    Returns
    -------
    List of BarJourney objects.
    """
    if not diagram_slices:
        return []
    
    journeys = []
    
    # Index slices by time
    slice_by_time = {s["time"]: s for s in diagram_slices}
    times = [s["time"] for s in diagram_slices]
    
    # Build bar lookup
    bar_lookup = {}  # (time, bar_id) -> bar dict
    for slice_data in diagram_slices:
        t = slice_data["time"]
        for bar in slice_data["diagram"].get("bars", []):
            bar_lookup[(t, bar["id"])] = bar
    
    # Track which bars have been assigned to a journey
    assigned_bars = set()  # (time, bar_id)
    
    # Track root bars for re-entry detection
    ruptured_roots = {}  # journey_id -> (root_ref, root_bar, rupture_time)
    
    # Step 1: Create journeys starting from first slice (spawns)
    first_slice = diagram_slices[0]
    for bar in first_slice["diagram"].get("bars", []):
        journey = _create_journey_from_spawn(
            bar,
            first_slice,
            diagram_slices,
            matches_per_step,
            bar_lookup,
            sim_cfg,
            journey_cfg
        )
        journeys.append(journey)
        
        # Mark bars as assigned
        for event in journey["events"]:
            if event["source"]:
                assigned_bars.add((event["source"]["time"], event["source"]["bar_id"]))
            if event["target"]:
                assigned_bars.add((event["target"]["time"], event["target"]["bar_id"]))
        
        # Track if ruptured for re-entry detection
        if journey["state"] == "ruptured":
            root_bar = bar_lookup.get((journey["root"]["time"], journey["root"]["bar_id"]))
            rupture_time = None
            for event in journey["events"]:
                if event["event_type"] == "rupture_out":
                    rupture_time = event["time_to"]
                    break
            if root_bar and rupture_time:
                ruptured_roots[journey["root"]["bar_id"]] = (
                    journey["root"], root_bar, rupture_time
                )
    
    # Step 2: Find bars in later slices that aren't assigned (rupture-ins or re-entries)
    for i, slice_data in enumerate(diagram_slices[1:], start=1):
        t = slice_data["time"]
        
        for bar in slice_data["diagram"].get("bars", []):
            bar_key = (t, bar["id"])
            if bar_key in assigned_bars:
                continue
            
            # Check if this is a re-entry of a ruptured theme
            reentry_journey = None
            reentry_event = None
            
            for root_bar_id, (root_ref, root_bar, rupture_time) in list(ruptured_roots.items()):
                # Compare to root bar
                result = find_similar_bar_across_gap(
                    root_bar, root_ref["time"], root_ref["slice_label"],
                    slice_data, sim_cfg
                )
                
                if result is not None:
                    reentry_ref, sim = result
                    if reentry_ref["bar_id"] == bar["id"]:
                        # Found re-entry!
                        new_witnesses = bar.get("witness", {}).get("tokens", {}).get("surface", [])
                        reentry_event = make_reentry_event(
                            root_ref, reentry_ref, rupture_time, sim, new_witnesses
                        )
                        
                        # Find the journey to update
                        for j in journeys:
                            if j["root"]["bar_id"] == root_bar_id:
                                reentry_journey = j
                                break
                        
                        if reentry_journey:
                            del ruptured_roots[root_bar_id]
                        break
            
            if reentry_journey and reentry_event:
                # Add re-entry to existing journey
                reentry_journey["events"].append(reentry_event)
                reentry_journey["state"] = "reentered"
                reentry_journey["final_ref"] = reentry_event["target"]
                assigned_bars.add(bar_key)
                
                # Continue tracking this journey forward
                _continue_journey_from(
                    reentry_journey,
                    bar,
                    slice_data,
                    diagram_slices[i:],
                    matches_per_step,
                    bar_lookup,
                    sim_cfg,
                    journey_cfg,
                    assigned_bars
                )
            else:
                # This is a pure rupture-in (new theme)
                journey = _create_journey_from_rupture_in(
                    bar,
                    slice_data,
                    diagram_slices[i:],
                    matches_per_step,
                    bar_lookup,
                    sim_cfg,
                    journey_cfg
                )
                journeys.append(journey)
                
                for event in journey["events"]:
                    if event["source"]:
                        assigned_bars.add((event["source"]["time"], event["source"]["bar_id"]))
                    if event["target"]:
                        assigned_bars.add((event["target"]["time"], event["target"]["bar_id"]))
    
    return journeys


def _create_journey_from_spawn(
    bar: Dict,
    start_slice: DiagramSlice,
    all_slices: List[DiagramSlice],
    matches_per_step: Dict,
    bar_lookup: Dict,
    sim_cfg: SimilarityConfig,
    journey_cfg: JourneyConfig
) -> BarJourney:
    """Create a journey starting from a spawn."""
    root_ref = make_bar_ref(start_slice["time"], start_slice["label"], bar)
    root_witnesses = bar.get("witness", {}).get("tokens", {}).get("surface", [])
    
    events = [make_spawn_event(root_ref, bar)]
    
    current_ref = root_ref
    current_bar = bar
    state = "alive"
    
    # Find subsequent slices
    start_idx = next(i for i, s in enumerate(all_slices) if s["time"] == start_slice["time"])
    
    for i in range(start_idx, len(all_slices) - 1):
        slice_from = all_slices[i]
        slice_to = all_slices[i + 1]
        step_key = (slice_from["time"], slice_to["time"])
        
        if step_key not in matches_per_step:
            break
        
        # Find match for current bar
        matches = matches_per_step[step_key]
        match = None
        for m in matches:
            if m["from_ref"]["bar_id"] == current_ref["bar_id"]:
                match = m
                break
        
        if match is None:
            # No match found - rupture
            events.append(make_rupture_out_event(current_ref, slice_to["time"], None))
            state = "ruptured"
            break
        
        if match["classification"] == "no_match" or match["classification"] == "too_far":
            # Rupture
            events.append(make_rupture_out_event(
                current_ref, slice_to["time"], match["similarity"]
            ))
            state = "ruptured"
            break
        
        # Successful continuation
        to_ref = match["to_ref"]
        to_bar = bar_lookup.get((to_ref["time"], to_ref["bar_id"]))
        
        if to_bar is None:
            break
        
        generative = is_generative_step(current_bar, to_bar, journey_cfg)
        
        if match["classification"] == "carry_by_name":
            events.append(make_carry_event(current_ref, to_ref, match["similarity"], generative))
        else:  # drift
            events.append(make_drift_event(current_ref, to_ref, match["similarity"], generative))
        
        current_ref = to_ref
        current_bar = to_bar
    
    return BarJourney(
        root=root_ref,
        root_witnesses=root_witnesses,
        events=events,
        state=state,
        final_ref=current_ref if state == "alive" else None
    )


def _create_journey_from_rupture_in(
    bar: Dict,
    start_slice: DiagramSlice,
    remaining_slices: List[DiagramSlice],
    matches_per_step: Dict,
    bar_lookup: Dict,
    sim_cfg: SimilarityConfig,
    journey_cfg: JourneyConfig
) -> BarJourney:
    """Create a journey starting from a rupture-in."""
    root_ref = make_bar_ref(start_slice["time"], start_slice["label"], bar)
    root_witnesses = bar.get("witness", {}).get("tokens", {}).get("surface", [])
    
    events = [make_rupture_in_event(root_ref, bar)]
    
    current_ref = root_ref
    current_bar = bar
    state = "alive"
    
    # Continue forward through remaining slices
    for i in range(len(remaining_slices) - 1):
        slice_from = remaining_slices[i]
        slice_to = remaining_slices[i + 1]
        step_key = (slice_from["time"], slice_to["time"])
        
        if step_key not in matches_per_step:
            break
        
        matches = matches_per_step[step_key]
        match = None
        for m in matches:
            if m["from_ref"]["bar_id"] == current_ref["bar_id"]:
                match = m
                break
        
        if match is None or match["classification"] in ["no_match", "too_far"]:
            events.append(make_rupture_out_event(
                current_ref, slice_to["time"],
                match["similarity"] if match else None
            ))
            state = "ruptured"
            break
        
        to_ref = match["to_ref"]
        to_bar = bar_lookup.get((to_ref["time"], to_ref["bar_id"]))
        
        if to_bar is None:
            break
        
        generative = is_generative_step(current_bar, to_bar, journey_cfg)
        
        if match["classification"] == "carry_by_name":
            events.append(make_carry_event(current_ref, to_ref, match["similarity"], generative))
        else:
            events.append(make_drift_event(current_ref, to_ref, match["similarity"], generative))
        
        current_ref = to_ref
        current_bar = to_bar
    
    return BarJourney(
        root=root_ref,
        root_witnesses=root_witnesses,
        events=events,
        state=state,
        final_ref=current_ref if state == "alive" else None
    )


def _continue_journey_from(
    journey: BarJourney,
    bar: Dict,
    start_slice: DiagramSlice,
    remaining_slices: List[DiagramSlice],
    matches_per_step: Dict,
    bar_lookup: Dict,
    sim_cfg: SimilarityConfig,
    journey_cfg: JourneyConfig,
    assigned_bars: Set
) -> None:
    """Continue an existing journey after re-entry."""
    current_ref = make_bar_ref(start_slice["time"], start_slice["label"], bar)
    current_bar = bar
    
    for i in range(len(remaining_slices) - 1):
        slice_from = remaining_slices[i]
        slice_to = remaining_slices[i + 1]
        step_key = (slice_from["time"], slice_to["time"])
        
        if step_key not in matches_per_step:
            break
        
        matches = matches_per_step[step_key]
        match = None
        for m in matches:
            if m["from_ref"]["bar_id"] == current_ref["bar_id"]:
                match = m
                break
        
        if match is None or match["classification"] in ["no_match", "too_far"]:
            journey["events"].append(make_rupture_out_event(
                current_ref, slice_to["time"],
                match["similarity"] if match else None
            ))
            journey["state"] = "ruptured"
            journey["final_ref"] = None
            break
        
        to_ref = match["to_ref"]
        to_bar = bar_lookup.get((to_ref["time"], to_ref["bar_id"]))
        
        if to_bar is None:
            break
        
        generative = is_generative_step(current_bar, to_bar, journey_cfg)
        
        if match["classification"] == "carry_by_name":
            journey["events"].append(make_carry_event(current_ref, to_ref, match["similarity"], generative))
        else:
            journey["events"].append(make_drift_event(current_ref, to_ref, match["similarity"], generative))
        
        assigned_bars.add((to_ref["time"], to_ref["bar_id"]))
        current_ref = to_ref
        current_bar = to_bar
        journey["final_ref"] = to_ref
        journey["state"] = "alive"


# =============================================================================
# Journey Analysis
# =============================================================================

def summarize_journey(journey: BarJourney) -> Dict:
    """Summarize a single journey."""
    events = journey["events"]
    
    return {
        "root": journey["root"]["bar_id"],
        "state": journey["state"],
        "num_events": len(events),
        "spawns": sum(1 for e in events if e["event_type"] == "spawn"),
        "carries": sum(1 for e in events if e["event_type"] == "carry_by_name"),
        "drifts": sum(1 for e in events if e["event_type"] == "drift"),
        "ruptures_out": sum(1 for e in events if e["event_type"] == "rupture_out"),
        "ruptures_in": sum(1 for e in events if e["event_type"] == "rupture_in"),
        "reentries": sum(1 for e in events if e["event_type"] == "reentry"),
        "generative_steps": sum(1 for e in events if e.get("generative", False))
    }


def format_journey(journey: BarJourney) -> str:
    """Format a journey as readable text."""
    lines = [
        f"Journey: {journey['root']['bar_id']} at {journey['root']['slice_label']}",
        f"  Root witnesses: {journey['root_witnesses'][:5]}",
        f"  State: {journey['state']}",
        "  Events:"
    ]
    
    for event in journey["events"]:
        e_type = event["event_type"]
        t_from = event["time_from"]
        t_to = event["time_to"]
        
        if e_type == "spawn":
            lines.append(f"    τ={t_to}: SPAWN")
        elif e_type == "rupture_in":
            lines.append(f"    τ={t_to}: RUPTURE_IN (new theme)")
        elif e_type == "rupture_out":
            lines.append(f"    τ={t_from}→{t_to}: RUPTURE_OUT")
        elif e_type == "reentry":
            lines.append(f"    τ={t_from}→{t_to}: RE-ENTRY ★")
        elif e_type == "carry_by_name":
            gen = " (generative)" if event.get("generative") else ""
            j = event["similarity"]["jaccard_tokens"] if event["similarity"] else 0
            lines.append(f"    τ={t_from}→{t_to}: CARRY (J={j:.3f}){gen}")
        elif e_type == "drift":
            gen = " (generative)" if event.get("generative") else ""
            d = event["similarity"]["d_sem"] if event["similarity"] else 0
            lines.append(f"    τ={t_from}→{t_to}: DRIFT (d_sem={d:.3f}){gen}")
    
    return "\n".join(lines)
