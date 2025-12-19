"""Chapter 4 – Temporal dynamics (second experiment block)

This script treats each line/turn as a time slice τ_i, computes a witnessed diagram
per slice, then tracks H0 bars across time to label:

- carry
- drift
- rupture
- re-entry
- birth

Run:
    python examples/chapter4_temporal_dynamics.py
"""

from witnessed_ph import default_config
from witnessed_ph.temporal import analyse_conversation_dynamics
from witnessed_ph.pretty import print_transition_summary, print_theme_score


LINES = r"""i've been thinking about climate change a lot lately
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
technology is certainly part of the solution renewables storage and efficiency improvements all contribute to reducing carbon emissions""".splitlines()

def build_turns():
    turns = []
    for i, raw in enumerate([l.strip() for l in LINES if l.strip()]):
        speaker = "User" if i % 2 == 0 else "Assistant"
        turns.append({"speaker": speaker, "text": raw})
    return turns


def main() -> None:
    cfg = default_config()

    # Witness / bar filters
    cfg["min_persistence"] = 0.03
    cfg["min_witness_tokens"] = 2
    cfg["max_witness_tokens"] = 5
    cfg["h0_mode"] = "theme"

    # Dynamics thresholds (matches the book-style defaults)
    cfg["lambda_sem"] = 0.5
    cfg["epsilon_match"] = 0.8
    cfg["theta_carry"] = 0.4
    cfg["delta_sem_max"] = 0.6
    cfg["topo_endpoint_eps"] = 0.2

    turns = build_turns()
    result = analyse_conversation_dynamics(turns, config=cfg, verbose=False)

    dynamics = result["dynamics"]
    transitions = dynamics["transitions"]

    print("=" * 72)
    print("Temporal dynamics summary")
    print("=" * 72)
    total = {"carry": 0, "drift": 0, "rupture": 0, "reentry": 0, "birth": 0}
    for step in transitions:
        for k in total:
            total[k] += int(step.get(k, 0))

    print("\nPer-transition counts:")
    for step in transitions:
        print(
            f"  {step['from']} → {step['to']} : "
            f"carry={step['carry']}, drift={step['drift']}, rupture={step['rupture']}, "
            f"reentry={step['reentry']}, birth={step['birth']}"
        )

    print("\nTotals:")
    for k, v in total.items():
        print(f"  {k}: {v}")

    # Book-style CLI readout (Theme score / SWL rendering)
    print_transition_summary(transitions)
    print_theme_score(
        result["diagrams"],
        dynamics,
        max_tokens=int(cfg.get("max_witness_tokens", 5)),
        max_lines_per_slice=0,
        dim=0,
    )

    print("\nConfig used:")
    for k, v in dynamics.get("config", {}).items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
