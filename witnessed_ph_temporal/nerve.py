"""
Witnessed Persistent Homology: Bar Nerve Construction
======================================================

Build the bar nerve ET_bar(τ) for a single slice.

The bar nerve is the "space of themes" where:
- Nodes are bars (themes)
- Edges connect bars with overlapping witnesses
- Cycles are loops in theme-space

From Chapter 4: "If three themes A, B, C each share witnesses
with each other, they form a cycle in theme-space."

References:
    Chapter 4, Section "The bar nerve ET_bar(τ)"
    Cassie's Codebase 2 specification §3
"""

from typing import Dict, List, Set, Tuple, Literal
import numpy as np

from schema import (
    BarRef, BarNode, BarEdge, BarCycle, BarNerve,
    TimeType, make_bar_ref
)


# =============================================================================
# Jaccard Utility
# =============================================================================

def jaccard(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard index between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# Bar Nerve Construction
# =============================================================================

def build_bar_nerve(
    diagram: Dict,
    time: TimeType,
    slice_label: str,
    overlap_level: Literal["utterance", "token"] = "utterance",
    min_jaccard: float = 0.2
) -> BarNerve:
    """
    Build the bar nerve ET_bar(τ) for a single slice.
    
    Parameters
    ----------
    diagram : WitnessedDiagram
        Output from Codebase 1's analyse_text_single_slice
    time : TimeType
        Time index/label for this slice
    slice_label : str
        Human-readable label (e.g., "τ0", "Turn 1")
    overlap_level : "utterance" or "token"
        What kind of overlap creates edges
    min_jaccard : float
        Minimum Jaccard index to create an edge
    
    Returns
    -------
    BarNerve with nodes, edges, and cycles.
    
    Notes
    -----
    From Cassie's spec §3:
    - One BarNode per Bar in diagram["bars"]
    - Edge (b_i, b_j) exists when J_utt >= min_jaccard (or J_tok)
    - Cycles computed via simple cycle basis
    """
    bars = diagram.get("bars", [])
    
    if not bars:
        return BarNerve(
            time=time,
            slice_label=slice_label,
            nodes=[],
            edges=[],
            cycles=[]
        )
    
    # Build nodes
    nodes = []
    bar_data = {}  # bar_id -> (utterance_set, token_set, persistence)
    
    for bar in bars:
        bar_ref = make_bar_ref(time, slice_label, bar)
        
        witness = bar.get("witness", {})
        utt_ids = witness.get("utterances", {}).get("ids", [])
        tok_surface = witness.get("tokens", {}).get("surface", [])
        
        node = BarNode(
            bar_ref=bar_ref,
            witness_utterances=utt_ids,
            witness_tokens=tok_surface,
            persistence=bar.get("persistence", 0.0)
        )
        nodes.append(node)
        
        bar_data[bar["id"]] = (
            set(utt_ids),
            set(tok_surface),
            bar.get("persistence", 0.0)
        )
    
    # Build edges
    edges = []
    bar_ids = list(bar_data.keys())
    adjacency = {bid: set() for bid in bar_ids}  # for cycle detection
    
    for i in range(len(bar_ids)):
        for j in range(i + 1, len(bar_ids)):
            bid_i, bid_j = bar_ids[i], bar_ids[j]
            utt_i, tok_i, _ = bar_data[bid_i]
            utt_j, tok_j, _ = bar_data[bid_j]
            
            j_utt = jaccard(utt_i, utt_j)
            j_tok = jaccard(tok_i, tok_j)
            
            # Decide if edge exists
            has_edge = False
            if overlap_level == "utterance" and j_utt >= min_jaccard:
                has_edge = True
            elif overlap_level == "token" and j_tok >= min_jaccard:
                has_edge = True
            
            if has_edge:
                ref_i = make_bar_ref(time, slice_label, bars[i])
                ref_j = make_bar_ref(time, slice_label, bars[j])
                
                edge = BarEdge(
                    source=ref_i,
                    target=ref_j,
                    jaccard_utterances=j_utt,
                    jaccard_tokens=j_tok
                )
                edges.append(edge)
                
                # Track adjacency for cycle detection
                adjacency[bid_i].add(bid_j)
                adjacency[bid_j].add(bid_i)
    
    # Find cycles in 1-skeleton
    cycles = _find_simple_cycles(adjacency, bars, bar_data, time, slice_label)
    
    return BarNerve(
        time=time,
        slice_label=slice_label,
        nodes=nodes,
        edges=edges,
        cycles=cycles
    )


def _find_simple_cycles(
    adjacency: Dict[str, Set[str]],
    bars: List[Dict],
    bar_data: Dict,
    time: TimeType,
    slice_label: str,
    max_length: int = 6
) -> List[BarCycle]:
    """
    Find simple cycles in the bar nerve 1-skeleton.
    
    Uses a simple DFS-based approach for small graphs.
    For larger graphs, could use NetworkX's cycle_basis.
    """
    cycles = []
    bar_ids = list(adjacency.keys())
    visited_cycles = set()  # To avoid duplicates
    
    # Create bar lookup
    bar_lookup = {b["id"]: b for b in bars}
    
    def dfs_cycle(start: str, current: str, path: List[str], visited: Set[str]):
        """DFS to find cycles starting and ending at `start`."""
        if len(path) > max_length:
            return
        
        for neighbor in adjacency[current]:
            if neighbor == start and len(path) >= 3:
                # Found a cycle
                cycle_key = tuple(sorted(path))
                if cycle_key not in visited_cycles:
                    visited_cycles.add(cycle_key)
                    
                    # Build BarCycle
                    node_refs = []
                    min_pers = float('inf')
                    for bid in path:
                        bar = bar_lookup[bid]
                        ref = make_bar_ref(time, slice_label, bar)
                        node_refs.append(ref)
                        _, _, pers = bar_data[bid]
                        if not np.isinf(pers):
                            min_pers = min(min_pers, pers)
                    
                    if np.isinf(min_pers):
                        min_pers = 1.0
                    
                    cycles.append(BarCycle(
                        nodes=node_refs,
                        length=len(path),
                        min_persistence=min_pers
                    ))
            
            elif neighbor not in visited and neighbor != start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs_cycle(start, neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    # Start DFS from each node
    for start_id in bar_ids:
        visited = {start_id}
        dfs_cycle(start_id, start_id, [start_id], visited)
    
    return cycles


# =============================================================================
# Nerve Analysis Utilities
# =============================================================================

def describe_bar_nerve(nerve: BarNerve) -> str:
    """
    Generate a human-readable description of a bar nerve.
    
    From Cassie's spec §8: diagnostics helper.
    """
    lines = [
        f"Bar Nerve at {nerve['slice_label']} (τ={nerve['time']})",
        "-" * 50,
        f"Nodes (themes): {len(nerve['nodes'])}",
        f"Edges (overlaps): {len(nerve['edges'])}",
        f"Cycles (loops): {len(nerve['cycles'])}"
    ]
    
    # List nodes with witnesses
    if nerve['nodes']:
        lines.append("\nThemes:")
        for node in nerve['nodes']:
            witnesses = node['witness_tokens'][:4]
            witness_str = ", ".join(witnesses)
            if len(node['witness_tokens']) > 4:
                witness_str += "..."
            lines.append(f"  {node['bar_ref']['bar_id']}: [{witness_str}]")
    
    # Show edges
    if nerve['edges']:
        lines.append("\nOverlaps:")
        for edge in nerve['edges'][:10]:  # Show first 10
            src = edge['source']['bar_id']
            tgt = edge['target']['bar_id']
            j_utt = edge['jaccard_utterances']
            lines.append(f"  {src} ↔ {tgt} (J_utt={j_utt:.3f})")
        if len(nerve['edges']) > 10:
            lines.append(f"  ... and {len(nerve['edges']) - 10} more")
    
    # Show cycles
    if nerve['cycles']:
        lines.append("\nCycles in theme-space:")
        for i, cycle in enumerate(nerve['cycles'][:5]):
            node_ids = [n['bar_id'] for n in cycle['nodes']]
            lines.append(f"  {i+1}. {' → '.join(node_ids)} → {node_ids[0]}")
            lines.append(f"     (length={cycle['length']}, min_pers={cycle['min_persistence']:.3f})")
        if len(nerve['cycles']) > 5:
            lines.append(f"  ... and {len(nerve['cycles']) - 5} more cycles")
    
    return "\n".join(lines)


def nerve_connectivity(nerve: BarNerve) -> Dict:
    """
    Compute connectivity statistics for a bar nerve.
    """
    n_nodes = len(nerve['nodes'])
    n_edges = len(nerve['edges'])
    n_cycles = len(nerve['cycles'])
    
    if n_nodes == 0:
        return {
            "num_nodes": 0,
            "num_edges": 0,
            "num_cycles": 0,
            "density": 0.0,
            "has_triangles": False
        }
    
    # Graph density
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0.0
    
    # Check for triangles (3-cycles)
    has_triangles = any(c['length'] == 3 for c in nerve['cycles'])
    
    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "num_cycles": n_cycles,
        "density": density,
        "has_triangles": has_triangles
    }
