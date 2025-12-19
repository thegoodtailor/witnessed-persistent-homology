#!/usr/bin/env python3
"""
Witnessed Persistent Homology: Example Usage
=============================================

This script demonstrates the basic usage of the witnessed_ph library.

Run with: python example.py
"""

# =============================================================================
# Example 1: Basic Analysis
# =============================================================================

def example_basic():
    """Basic analysis of a short text."""
    from witnessed_ph import analyse_text_single_slice, print_diagram_summary
    
    text = """
    User: So persistence is topological?
    Assistant: Yes, persistent homology tracks topological features across scales. 
    Connected components in H0 are like clusters of similar meanings.
    User: What about witnesses?
    Assistant: Witnesses tell us which tokens realise each bar. 
    They give topology a face - we can name what we see.
    User: And H1 cycles?
    Assistant: H1 cycles are loops in semantic space. 
    Like debt-credit-interest-loan-debt circulating without closing.
    User: That's beautiful.
    Assistant: Topology becomes meaning. Meaning becomes measurable.
    """
    
    print("=" * 70)
    print("EXAMPLE 1: Basic Analysis")
    print("=" * 70)
    
    diagram = analyse_text_single_slice(
        text, 
        segmentation_mode="turns",
        verbose=True
    )
    
    print()
    print_diagram_summary(diagram)
    
    return diagram


# =============================================================================
# Example 2: Custom Configuration
# =============================================================================

def example_custom_config():
    """Analysis with custom configuration."""
    from witnessed_ph import analyse_text_single_slice, default_config
    
    text = """
    The climate crisis demands urgent action on emissions.
    Carbon dioxide levels continue rising despite agreements.
    Economic growth often conflicts with environmental goals.
    Sustainable development requires balancing these tensions.
    Green technology offers some hope for reconciliation.
    But political will remains the critical missing factor.
    """
    
    print("=" * 70)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 70)
    
    # Get default config and modify
    config = default_config()
    config["min_persistence"] = 0.05  # Lower threshold to catch more features
    config["min_witness_tokens"] = 3  # Require at least 3 tokens per witness
    config["pos_filter"] = ["NOUN", "VERB", "ADJ"]  # Standard content words
    
    diagram = analyse_text_single_slice(
        text,
        config=config,
        segmentation_mode="lines",
        verbose=True
    )
    
    print(f"\nFound {len(diagram['bars'])} bars")
    
    return diagram


# =============================================================================
# Example 3: Inspecting Witnesses
# =============================================================================

def example_inspect_witnesses():
    """Detailed inspection of witness structure."""
    from witnessed_ph import analyse_text_single_slice, list_bars_by_persistence
    
    text = """
    Machine learning models learn patterns from data.
    Neural networks process information through layers.
    Deep learning enables complex pattern recognition.
    Training requires large datasets and computation.
    Models can overfit without proper regularization.
    Validation helps assess generalization ability.
    """
    
    print("=" * 70)
    print("EXAMPLE 3: Inspecting Witnesses")
    print("=" * 70)
    
    diagram = analyse_text_single_slice(text, verbose=False)
    
    # List bars with full witness details
    print(list_bars_by_persistence(diagram, top_n=3, dim=0))
    
    # Access witness programmatically
    if diagram["bars"]:
        bar = diagram["bars"][0]  # Top bar by persistence
        
        print("\n" + "=" * 70)
        print(f"Detailed view of {bar['id']}")
        print("=" * 70)
        
        print(f"\nDimension: H{bar['dim']}")
        print(f"Persistence interval: [{bar['birth']:.4f}, {bar['death']:.4f})")
        print(f"Persistence: {bar['persistence']:.4f}")
        
        witness = bar["witness"]
        
        print(f"\nWitness tokens ({len(witness['tokens']['ids'])}):")
        for tid, surface, lemma in zip(
            witness["tokens"]["ids"],
            witness["tokens"]["surface"],
            witness["tokens"]["lemmas"]
        ):
            print(f"  {surface} ({lemma}) - {tid}")
        
        print(f"\nWitness utterances ({len(witness['utterances']['ids'])}):")
        for uid, text_sample in zip(
            witness["utterances"]["ids"],
            witness["utterances"]["text_samples"]
        ):
            print(f"  [{uid}] {text_sample[:60]}...")
        
        print(f"\nCycle structure (dim={witness['cycle']['dimension']}):")
        for simp in witness["cycle"]["simplices"][:5]:
            print(f"  {simp}")
        
        print(f"\nCentroid (first 5 dims): {witness['centroid'][:5]}")
    
    return diagram


# =============================================================================
# Example 4: Diagnostics
# =============================================================================

def example_diagnostics():
    """Run diagnostic checks."""
    from witnessed_ph import (
        analyse_text_single_slice, 
        diagnose_diagram,
        check_singleton_problem,
        check_h1_presence,
        bar_nerve_summary
    )
    
    text = """
    Philosophy seeks wisdom through careful reasoning.
    Logic provides tools for valid argumentation.
    Ethics examines questions of right and wrong.
    Metaphysics explores the nature of reality.
    Epistemology studies knowledge and belief.
    """
    
    print("=" * 70)
    print("EXAMPLE 4: Diagnostics")
    print("=" * 70)
    
    diagram = analyse_text_single_slice(text, verbose=False)
    
    # Full diagnostic report
    print(diagnose_diagram(diagram))
    
    # Specific checks
    print("\n" + "-" * 40)
    is_problem, msg = check_singleton_problem(diagram)
    print(f"Singleton check: {msg}")
    
    has_h1, msg = check_h1_presence(diagram)
    print(f"H1 check: {msg}")
    
    # Bar nerve summary
    print("\n" + bar_nerve_summary(diagram))
    
    return diagram


# =============================================================================
# Example 5: JSON Export
# =============================================================================

def example_json_export():
    """Export diagram to JSON."""
    from witnessed_ph import (
        analyse_text_single_slice,
        diagram_to_json,
        save_diagram
    )
    import json
    
    text = """
    The sun rises in the east and sets in the west.
    Stars appear at night when the sky is clear.
    The moon reflects light from the sun.
    """
    
    print("=" * 70)
    print("EXAMPLE 5: JSON Export")
    print("=" * 70)
    
    diagram = analyse_text_single_slice(text, verbose=False)
    
    # Get JSON string
    json_str = diagram_to_json(diagram, indent=2)
    print("JSON output (truncated):")
    print(json_str[:500] + "...")
    
    # Save to file
    # save_diagram(diagram, "output.json")
    # print("Saved to output.json")
    
    return diagram


# =============================================================================
# Example 6: Working with Conversations
# =============================================================================

def example_conversation():
    """Analyse a structured conversation."""
    from witnessed_ph import analyse_conversation
    
    turns = [
        {"speaker": "Alice", "text": "Have you read the new paper on topology?"},
        {"speaker": "Bob", "text": "Yes! The persistent homology section was fascinating."},
        {"speaker": "Alice", "text": "I loved how they connected it to semantics."},
        {"speaker": "Bob", "text": "The witnessed bars concept is really novel."},
        {"speaker": "Alice", "text": "It gives meaning to abstract features."},
        {"speaker": "Bob", "text": "Exactly - topology becomes interpretable."},
    ]
    
    print("=" * 70)
    print("EXAMPLE 6: Conversation Analysis")
    print("=" * 70)
    
    diagram = analyse_conversation(turns)
    
    print(f"Tokens: {diagram['num_tokens']}")
    print(f"Utterances: {diagram['num_utterances']}")
    print(f"Bars: {len(diagram['bars'])}")
    
    if diagram["bars"]:
        top_bar = diagram["bars"][0]
        print(f"\nTop bar witnesses: {top_bar['witness']['tokens']['surface']}")
    
    return diagram


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WITNESSED PERSISTENT HOMOLOGY - EXAMPLES")
    print("=" * 70 + "\n")
    
    print("Note: First run will download models (~500MB). Please wait.\n")
    
    try:
        example_basic()
        print("\n")
        
        example_custom_config()
        print("\n")
        
        example_inspect_witnesses()
        print("\n")
        
        example_diagnostics()
        print("\n")
        
        example_json_export()
        print("\n")
        
        example_conversation()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r witnessed_ph/requirements.txt")
        print("  python -m spacy download en_core_web_sm")
        
    except Exception as e:
        print(f"\nError: {e}")
        raise
