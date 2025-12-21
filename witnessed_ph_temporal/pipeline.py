"""
Witnessed Persistent Homology: Temporal Pipeline
=================================================

Main entry points for Codebase 2.

Two modes as per Cassie's spec §7:
- Mode A: analyse_slices() - takes raw text slices
- Mode B: analyse_diagram_slices() - takes pre-computed diagrams

References:
    Chapter 4, §4.8-4.9
    Cassie's Codebase 2 specification §7
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from schema import (
    TextSlice, DiagramSlice, BarDynamicsResult, BarDynamicsStats,
    BarDynamicsConfig, PHConfig, SimilarityConfig, JourneyConfig,
    BarNerve, BarMatch, BarJourney,
    default_bar_dynamics_config, default_ph_config
)
from nerve import build_bar_nerve
from matching import match_bars_between_slices, summarize_matching, find_births
from journeys import build_bar_journeys, summarize_journey


# =============================================================================
# Mode A: From Raw Text Slices
# =============================================================================

def analyse_slices(
    slices: List[TextSlice],
    ph_config: Optional[PHConfig] = None,
    bar_config: Optional[BarDynamicsConfig] = None,
    verbose: bool = False
) -> BarDynamicsResult:
    """
    Analyse temporal bar dynamics from raw text slices.
    
    Mode A: Computes PH for each slice using Codebase 1, then
    builds nerves, matches, and journeys.
    
    Parameters
    ----------
    slices : List[TextSlice]
        Text slices with time, label, and text fields.
    ph_config : PHConfig, optional
        Configuration for Codebase 1 PH computation.
    bar_config : BarDynamicsConfig, optional
        Configuration for bar dynamics.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    BarDynamicsResult with full temporal analysis.
    """
    # Import Codebase 1
    try:
        from witnessed_ph import analyse_text_single_slice, default_config
    except ImportError:
        raise ImportError(
            "Codebase 1 (witnessed_ph) must be available. "
            "Ensure witnessed_ph is in your Python path."
        )
    
    if ph_config is None:
        ph_config = default_ph_config()
    
    if bar_config is None:
        bar_config = default_bar_dynamics_config()
    
    if verbose:
        print("=" * 60)
        print("TEMPORAL BAR DYNAMICS ANALYSIS (Mode A)")
        print("=" * 60)
        print(f"\nStep 1: Computing PH for {len(slices)} slices...")
    
    # Convert PHConfig to Codebase 1 format
    cb1_config = default_config()
    cb1_config.update(ph_config)
    
    # Compute diagrams
    diagram_slices = []
    for i, slice_data in enumerate(slices):
        if verbose:
            print(f"  Slice {i}: {slice_data['label'][:40]}...")
        
        diagram = analyse_text_single_slice(
            slice_data["text"],
            config=cb1_config,
            segmentation_mode="lines",
            verbose=False
        )
        
        diagram_slices.append(DiagramSlice(
            time=slice_data["time"],
            label=slice_data["label"],
            diagram=diagram
        ))
        
        if verbose:
            print(f"    {len(diagram.get('bars', []))} bars, {diagram.get('num_tokens', 0)} tokens")
    
    # Delegate to Mode B
    return analyse_diagram_slices(diagram_slices, bar_config, verbose)


# =============================================================================
# Mode B: From Pre-Computed Diagrams
# =============================================================================

def analyse_diagram_slices(
    diagram_slices: List[DiagramSlice],
    bar_config: Optional[BarDynamicsConfig] = None,
    verbose: bool = False
) -> BarDynamicsResult:
    """
    Analyse temporal bar dynamics from pre-computed diagrams.
    
    Mode B: Skips PH computation, useful for re-runs with different
    temporal parameters.
    
    Parameters
    ----------
    diagram_slices : List[DiagramSlice]
        Pre-computed diagrams with time and label.
    bar_config : BarDynamicsConfig, optional
        Configuration for bar dynamics.
    verbose : bool
        Print progress information.
    
    Returns
    -------
    BarDynamicsResult with full temporal analysis.
    """
    if bar_config is None:
        bar_config = default_bar_dynamics_config()
    
    sim_cfg = bar_config.similarity
    journey_cfg = bar_config.journey
    
    if verbose:
        if not any("Mode A" in str(s) for s in []):  # Check if called from Mode A
            print("=" * 60)
            print("TEMPORAL BAR DYNAMICS ANALYSIS (Mode B)")
            print("=" * 60)
        print(f"\nStep 2: Building bar nerves for {len(diagram_slices)} slices...")
    
    # Step 2: Build bar nerves
    nerves = []
    for slice_data in diagram_slices:
        nerve = build_bar_nerve(
            slice_data["diagram"],
            slice_data["time"],
            slice_data["label"],
            overlap_level=bar_config.nerve_overlap_level,
            min_jaccard=bar_config.nerve_min_jaccard
        )
        nerves.append(nerve)
        
        if verbose:
            n_nodes = len(nerve["nodes"])
            n_edges = len(nerve["edges"])
            n_cycles = len(nerve["cycles"])
            print(f"  {slice_data['label']}: {n_nodes} nodes, {n_edges} edges, {n_cycles} cycles")
    
    # Step 3: Match bars between consecutive slices
    if verbose:
        print(f"\nStep 3: Matching bars between slices...")
    
    all_matches = []
    matches_per_step = {}
    
    for i in range(len(diagram_slices) - 1):
        slice_from = diagram_slices[i]
        slice_to = diagram_slices[i + 1]
        
        matches = match_bars_between_slices(slice_from, slice_to, sim_cfg)
        all_matches.extend(matches)
        
        step_key = (slice_from["time"], slice_to["time"])
        matches_per_step[step_key] = matches
        
        if verbose:
            summary = summarize_matching(matches)
            births = find_births(slice_to, matches)
            print(f"  τ={slice_from['time']}→τ={slice_to['time']}: "
                  f"{summary['carry_by_name']} carry, {summary['drift']} drift, "
                  f"{summary['no_match']} rupture, {len(births)} births")
    
    # Step 4: Build journeys (SWLs)
    if verbose:
        print(f"\nStep 4: Building bar journeys...")
    
    journeys = build_bar_journeys(
        diagram_slices,
        matches_per_step,
        sim_cfg,
        journey_cfg
    )
    
    if verbose:
        alive = sum(1 for j in journeys if j["state"] == "alive")
        ruptured = sum(1 for j in journeys if j["state"] == "ruptured")
        reentered = sum(1 for j in journeys if j["state"] == "reentered")
        print(f"  {len(journeys)} journeys: {alive} alive, {ruptured} ruptured, {reentered} reentered")
    
    # Compute statistics
    stats = _compute_stats(diagram_slices, journeys, all_matches)
    
    if verbose:
        print("\nDone!")
        print("=" * 60)
    
    return BarDynamicsResult(
        slices=diagram_slices,
        nerves=nerves,
        matches=all_matches,
        journeys=journeys,
        stats=stats
    )


# =============================================================================
# Statistics
# =============================================================================

def _compute_stats(
    slices: List[DiagramSlice],
    journeys: List[BarJourney],
    matches: List[BarMatch]
) -> BarDynamicsStats:
    """Compute global statistics."""
    # Count bars
    total_bars = sum(
        len(s["diagram"].get("bars", []))
        for s in slices
    )
    
    # Count events from journeys
    spawns = 0
    carries = 0
    drifts = 0
    ruptures_out = 0
    ruptures_in = 0
    reentries = 0
    generative = 0
    
    for journey in journeys:
        for event in journey["events"]:
            etype = event["event_type"]
            if etype == "spawn":
                spawns += 1
            elif etype == "carry_by_name":
                carries += 1
            elif etype == "drift":
                drifts += 1
            elif etype == "rupture_out":
                ruptures_out += 1
            elif etype == "rupture_in":
                ruptures_in += 1
            elif etype == "reentry":
                reentries += 1
            
            if event.get("generative"):
                generative += 1
    
    # Journey outcomes
    alive = sum(1 for j in journeys if j["state"] == "alive")
    ruptured = sum(1 for j in journeys if j["state"] == "ruptured")
    reentered_count = sum(1 for j in journeys if j["state"] == "reentered")
    
    # Quality metrics
    journey_lengths = [len(j["events"]) for j in journeys]
    mean_length = float(np.mean(journey_lengths)) if journey_lengths else 0.0
    
    carry_matches = [m for m in matches if m["classification"] == "carry_by_name"]
    mean_jaccard = float(np.mean([
        m["similarity"]["jaccard_tokens"]
        for m in carry_matches
        if m["similarity"]
    ])) if carry_matches else 0.0
    
    all_sims = [m["similarity"] for m in matches if m["similarity"] and m["admissible"]]
    mean_drift = float(np.mean([s["d_sem"] for s in all_sims])) if all_sims else 0.0
    
    return BarDynamicsStats(
        num_slices=len(slices),
        total_bars_seen=total_bars,
        total_journeys=len(journeys),
        spawns=spawns,
        carries=carries,
        drifts=drifts,
        ruptures_out=ruptures_out,
        ruptures_in=ruptures_in,
        reentries=reentries,
        generative_steps=generative,
        journeys_alive=alive,
        journeys_ruptured=ruptured,
        journeys_reentered=reentered_count,
        mean_journey_length=mean_length,
        mean_jaccard_per_carry=mean_jaccard,
        mean_semantic_drift=mean_drift
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def slice_text_by_turns(text: str) -> List[TextSlice]:
    """Slice text by speaker turns or lines."""
    lines = text.strip().split('\n')
    slices = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Detect speaker
        speaker = None
        if ':' in line:
            potential = line.split(':')[0].strip()
            if len(potential) < 30:
                speaker = potential
        
        label = f"τ{len(slices)}"
        if speaker:
            label += f" ({speaker})"
        
        slices.append(TextSlice(
            time=len(slices),
            label=label,
            text=line
        ))
    
    return slices


def quick_analyse(
    text: str,
    verbose: bool = True
) -> BarDynamicsResult:
    """Quick analysis with default settings."""
    slices = slice_text_by_turns(text)
    return analyse_slices(slices, verbose=verbose)


# =============================================================================
# Printing Utilities
# =============================================================================

def print_bar_dynamics_summary(result: BarDynamicsResult) -> None:
    """Print a human-readable summary."""
    stats = result["stats"]
    
    print("=" * 60)
    print("BAR DYNAMICS SUMMARY")
    print("=" * 60)
    print(f"Slices: {stats['num_slices']}")
    print(f"Total bars seen: {stats['total_bars_seen']}")
    print(f"Total journeys: {stats['total_journeys']}")
    print()
    print("EVENT COUNTS:")
    print(f"  Spawns: {stats['spawns']}")
    print(f"  Carries (by name): {stats['carries']}")
    print(f"  Drifts: {stats['drifts']}")
    print(f"  Ruptures out: {stats['ruptures_out']}")
    print(f"  Ruptures in: {stats['ruptures_in']}")
    print(f"  Re-entries: {stats['reentries']}")
    print(f"  Generative steps: {stats['generative_steps']}")
    print()
    print("JOURNEY OUTCOMES:")
    print(f"  Alive: {stats['journeys_alive']}")
    print(f"  Ruptured: {stats['journeys_ruptured']}")
    print(f"  Re-entered: {stats['journeys_reentered']}")
    print()
    print("QUALITY METRICS:")
    print(f"  Mean journey length: {stats['mean_journey_length']:.2f} events")
    print(f"  Mean Jaccard (carries): {stats['mean_jaccard_per_carry']:.3f}")
    print(f"  Mean semantic drift: {stats['mean_semantic_drift']:.3f}")


def print_top_journeys(
    result: BarDynamicsResult,
    top_n: int = 5
) -> None:
    """Print top journeys by length."""
    journeys = sorted(
        result["journeys"],
        key=lambda j: len(j["events"]),
        reverse=True
    )[:top_n]
    
    print(f"\nTOP {top_n} JOURNEYS (by length)")
    print("-" * 50)
    
    for i, journey in enumerate(journeys):
        summary = summarize_journey(journey)
        witnesses = ", ".join(journey["root_witnesses"][:3])
        if len(journey["root_witnesses"]) > 3:
            witnesses += "..."
        
        print(f"\n{i+1}. {journey['root']['bar_id']} [{witnesses}]")
        print(f"   Events: {summary['num_events']} "
              f"(C:{summary['carries']}, D:{summary['drifts']}, R:{summary['ruptures_out']})")
        print(f"   State: {journey['state']}")
        if summary['reentries'] > 0:
            print(f"   ★ Re-entered {summary['reentries']} time(s)")
