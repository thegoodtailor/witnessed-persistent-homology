#!/usr/bin/env python3
"""
Witnessed Bars Analyser
=======================

Paste your text below between the triple quotes, then run:
    python analyse_text.py

Shows full H₀ and H₁ witnessed bars with γ (cycle) structure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# PASTE YOUR TEXT HERE
# =============================================================================

TEXT = """
    User: I've been thinking about climate change a lot lately.
    Assistant: Climate change is certainly one of the defining challenges of our time. What aspects concern you most?
    User: The economic impacts worry me. How do we balance growth with sustainability?
    Assistant: That tension between economic growth and environmental protection is real. Many economists now argue for sustainable development models.
    User: What about carbon pricing? Does it actually work?
    Assistant: Carbon pricing can be effective when well-designed. It creates market incentives to reduce emissions while allowing flexibility in how reductions are achieved.
    User: Let's change topics. Tell me about ancient Rome.
    Assistant: Ancient Rome was a remarkable civilization that lasted over a thousand years. From a small city-state to a vast empire that ruled over many regions, it then plummeted and simply wasn't competitive with other barbarian invaders.
    User: Actually, let's go back to climate. What about renewable energy?
    Assistant: Renewable energy has seen remarkable growth. Solar and wind costs have plummeted, making them competitive with fossil fuels in many regions.
    User: So there's hope for addressing climate change through technology?
    Assistant: Technology is certainly part of the solution. Renewables, storage, and efficiency improvements all contribute to reducing carbon emissions.
"""

# =============================================================================
# CONFIGURATION (adjust if needed)
# =============================================================================

CONFIG = {
    "min_persistence": 0.03,    # Lower = more bars (try 0.01 for everything)
    "min_witness_tokens": 2,    # Minimum tokens per witness
    "segmentation": "lines",    # "lines" or "turns" (for Speaker: format)
}

# =============================================================================
# ANALYSIS (no need to edit below)
# =============================================================================

def main():
    from witnessed_ph import analyse_text_single_slice, default_config
    
    print("=" * 70)
    print("WITNESSED BARS ANALYSIS")
    print("=" * 70)
    
    # Setup config
    config = default_config()
    config["min_persistence"] = CONFIG["min_persistence"]
    config["min_witness_tokens"] = CONFIG["min_witness_tokens"]
    
    print(f"\nConfig: min_persistence={config['min_persistence']}, "
          f"min_witness_tokens={config['min_witness_tokens']}")
    print()
    
    # Run analysis
    diagram = analyse_text_single_slice(
        TEXT, 
        config=config,
        segmentation_mode=CONFIG["segmentation"],
        verbose=True
    )
    
    # Separate by dimension
    h0_bars = [b for b in diagram['bars'] if b['dim'] == 0]
    h1_bars = [b for b in diagram['bars'] if b['dim'] == 1]
    
    print()
    print("=" * 70)
    print(f"RESULTS: {len(h0_bars)} H₀ bars, {len(h1_bars)} H₁ bars")
    print("=" * 70)
    
    # ==========================================================================
    # H₀ BARS (Connected Components / Themes)
    # ==========================================================================
    print()
    print("=" * 70)
    print("H₀ BARS (Themes / Connected Components)")
    print("=" * 70)
    
    for bar in h0_bars:
        print(f"\n{'─' * 60}")
        print(f"{bar['id']} [H₀]")
        print(f"{'─' * 60}")
        print(f"  Birth: {bar['birth']:.4f}")
        print(f"  Death: {bar['death']:.4f}" if bar['death'] != float('inf') else "  Death: ∞")
        print(f"  Persistence: {bar['persistence']:.4f}")
        
        # Witness tokens
        tokens = bar['witness']['tokens']
        print(f"\n  Witness tokens ({len(tokens['ids'])}):")
        for tid, surface, lemma in zip(tokens['ids'], tokens['surface'], tokens['lemmas']):
            print(f"    • {surface} ({lemma}) [{tid}]")
        
        # Utterances
        utts = bar['witness']['utterances']
        print(f"\n  Utterances ({len(utts['ids'])}): {utts['ids']}")
        for uid, text in zip(utts['ids'], utts['text_samples']):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"    [{uid}] {preview}")
        
        # Cycle structure (for H0, just the vertices)
        cycle = bar['witness']['cycle']
        print(f"\n  γ (component vertices): {len(cycle['simplices'])} vertices")
    
    # ==========================================================================
    # H₁ BARS (Loops / Cycles)
    # ==========================================================================
    if h1_bars:
        print()
        print("=" * 70)
        print("H₁ BARS (Loops / Semantic Cycles)")
        print("=" * 70)
        
        for bar in h1_bars:
            print(f"\n{'─' * 60}")
            print(f"{bar['id']} [H₁] ★ LOOP")
            print(f"{'─' * 60}")
            print(f"  Birth: {bar['birth']:.4f}")
            print(f"  Death: {bar['death']:.4f}")
            print(f"  Persistence: {bar['persistence']:.4f}")
            
            # γ cycle structure - THIS IS THE KEY H1 OUTPUT
            cycle = bar['witness']['cycle']
            print(f"\n  γ (cycle structure):")
            print(f"    Dimension: {cycle['dimension']}")
            print(f"    Edges forming the loop:")
            for edge in cycle['simplices']:
                # Get surface forms for the edge
                edge_tokens = []
                for tid in edge:
                    for i, t in enumerate(bar['witness']['tokens']['ids']):
                        if t == tid:
                            edge_tokens.append(bar['witness']['tokens']['surface'][i])
                            break
                print(f"      {edge[0]} ↔ {edge[1]}")
                print(f"        ({' ↔ '.join(edge_tokens)})")
            
            # Witness tokens
            tokens = bar['witness']['tokens']
            print(f"\n  Witness tokens ({len(tokens['ids'])}):")
            for tid, surface, lemma in zip(tokens['ids'], tokens['surface'], tokens['lemmas']):
                print(f"    • {surface} ({lemma}) [{tid}]")
            
            # Utterances
            utts = bar['witness']['utterances']
            print(f"\n  Utterances ({len(utts['ids'])}): {utts['ids']}")
            for uid, text in zip(utts['ids'], utts['text_samples']):
                preview = text[:80] + "..." if len(text) > 80 else text
                print(f"    [{uid}] {preview}")
    else:
        print()
        print("=" * 70)
        print("No H₁ bars found")
        print("=" * 70)
        print("(Try lowering min_persistence to 0.01 in CONFIG above)")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Tokens analysed: {diagram['num_tokens']}")
    print(f"  Utterances: {diagram['num_utterances']}")
    print(f"  H₀ bars (themes): {len(h0_bars)}")
    print(f"  H₁ bars (loops): {len(h1_bars)}")
    
    if h0_bars:
        avg_h0_size = sum(len(b['witness']['tokens']['ids']) for b in h0_bars) / len(h0_bars)
        print(f"  Avg H₀ witness size: {avg_h0_size:.1f} tokens")
    
    if h1_bars:
        avg_h1_pers = sum(b['persistence'] for b in h1_bars) / len(h1_bars)
        print(f"  Avg H₁ persistence: {avg_h1_pers:.4f}")
        
        # Show which utterances have H1 loops spanning them
        h1_utts = set()
        for bar in h1_bars:
            h1_utts.update(bar['witness']['utterances']['ids'])
        print(f"  Utterances with H₁ structure: {sorted(h1_utts)}")


if __name__ == "__main__":
    main()
