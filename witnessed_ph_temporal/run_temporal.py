#!/usr/bin/env python3
"""
Temporal Bar Dynamics: Example Runner
======================================

Codebase 2: "Where Themes Learn to Breathe"

Run with: python run_temporal.py
"""

import sys
import os

# Add the directory containing this script to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Now import from the local modules directly (not as a package)
from schema import (
    BarDynamicsConfig, SimilarityConfig, JourneyConfig,
    default_bar_dynamics_config, default_ph_config
)
from pipeline import (
    slice_text_by_turns,
    analyse_slices,
    print_bar_dynamics_summary,
    print_top_journeys
)
from theme_score import (
    render_theme_score,
    print_theme_score,
    print_all_journey_timelines
)


def main():
    # Example conversation with topic shifts
    conversation = """
User: I've been thinking about climate change a lot lately.
Assistant: Climate change is certainly one of the defining challenges of our time. What aspects concern you most?
User: The economic impacts worry me. How do we balance growth with sustainability?
Assistant: That tension between economic growth and environmental protection is real. Many economists argue for sustainable development models.
User: Let's change topics. Tell me about ancient Rome.
Assistant: Ancient Rome was a fascinating civilization that lasted over a thousand years. From a small city-state to a vast empire.
User: Actually, let's go back to climate. What about renewable energy?
Assistant: Renewable energy has seen remarkable growth. Solar and wind costs have plummeted, making them competitive with fossil fuels.
"""
    
    print("=" * 70)
    print("WITNESSED PERSISTENT HOMOLOGY - TEMPORAL DYNAMICS")
    print("Codebase 2: Where Themes Learn to Breathe")
    print("=" * 70)
    
    # Slice the conversation
    print("\n[Step 1] Slicing conversation into temporal windows...")
    slices = slice_text_by_turns(conversation)
    print(f"  Created {len(slices)} slices")
    
    for i, s in enumerate(slices):
        preview = s["text"][:50].replace("\n", " ")
        print(f"    τ{i}: {preview}...")
    
    # =========================================================================
    # CONFIGURE FOR DEBERTA EMBEDDINGS
    # =========================================================================
    ph_config = default_ph_config()
    ph_config["embedding_model"] = "microsoft/deberta-v3-base"  # <-- THE KEY LINE
    
    print(f"\n[Config] Using embedding model: {ph_config['embedding_model']}")
    
    # Run temporal analysis
    print("\n[Step 2] Running temporal bar dynamics analysis...")
    result = analyse_slices(slices, ph_config=ph_config, verbose=True)
    
    # Print summary statistics
    print("\n")
    print_bar_dynamics_summary(result)
    
    # Print the THEME SCORE visualization
    print("\n[Step 3] Generating Theme Score visualization...")
    print_theme_score(result, max_bars_per_slice=4)
    
    # Print journey timelines
    print("\n[Step 4] Generating journey timelines...")
    print_all_journey_timelines(result, top_n=5)
    
    # Save theme score to file
    output_path = os.path.join(script_dir, "theme_score_output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_theme_score(result))
    print(f"\n✓ Theme score saved to {output_path}")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    main()
