"""Chapter 4 – Worked example (single slice)

This mirrors the book's single-slice pipeline:
tokenise → contextual embeddings → witnessed persistence bars.

Run:
    python examples/chapter4_single_slice.py
"""

from witnessed_ph import default_config, analyse_text_single_slice, list_bars_by_persistence, print_diagram_summary


TEXT = r"""i've been thinking about climate change a lot lately
climate change is certainly one of the defining challenges of our time what aspects concern you most
the economic impacts worry me how do we balance growth with sustainability
that tension between economic growth and environmental protection is real many economists now argue for sustainable development models
what about carbon pricing does it actually work
carbon pricing can be effective when well designed it creates market incentives to reduce emissions while allowing flexibility in how reductions are achieved
let's change topics tell me about ancient rome
ancient rome was a remarkable civilization that lasted over a thousand years from a small city state to a vast empire spanning many regions it shaped law architecture and governance for centuries
actually let's go back to climate what about renewable energy
renewable energy has seen remarkable growth solar and wind costs have plummeted making them competitive with fossil fuels in many regions
so there's hope for addressing climate change through technology
technology is certainly part of the solution renewables storage and efficiency improvements all contribute to reducing carbon emissions"""

def main() -> None:
    cfg = default_config()
    cfg["min_persistence"] = 0.03
    cfg["min_witness_tokens"] = 2
    cfg["h0_mode"] = "theme"   # set "raw" for standard births=0

    diagram = analyse_text_single_slice(TEXT, config=cfg, segmentation_mode="lines", verbose=True, slice_id="chapter4")

    print("\n" + "=" * 72)
    print_diagram_summary(diagram)

    h0 = list_bars_by_persistence(diagram, dim=0, top_n=8)
    h1 = list_bars_by_persistence(diagram, dim=1, top_n=5)

    print("\nTop H0 theme bars (by persistence):")
    for b in h0:
        w = b.get("witness", {})
        toks = (w.get("tokens", {}) or {}).get("surface", [])
        print(f"  {b['id']}  [{b['birth']:.3f}, {b['death']:.3f})  pers={b['persistence']:.3f}  witness={toks}")

    if h1:
        print("\nTop H1 loop bars (by persistence):")
        for b in h1:
            w = b.get("witness", {})
            toks = (w.get("tokens", {}) or {}).get("surface", [])
            edges = (w.get("cycle", {}) or {}).get("edges", [])
            print(f"  {b['id']}  [{b['birth']:.3f}, {b['death']:.3f})  pers={b['persistence']:.3f}  witness={toks}  cycle_edges={edges}")


if __name__ == "__main__":
    main()
