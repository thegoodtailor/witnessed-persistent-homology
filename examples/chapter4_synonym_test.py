"""Chapter 4 – Lexical bridge / synonym robustness sketch

This is a template for the "synonym rewritten" robustness experiment.
Edit the SYNONYMS dict to match the exact rewrites used in the book.
"""

from witnessed_ph import default_config
from witnessed_ph.temporal import analyse_conversation_dynamics

ORIGINAL_TURNS = [
    {"speaker": "User", "text": "I've been thinking about climate change a lot lately."},
    {"speaker": "Assistant", "text": "Climate change is certainly one of the defining challenges of our time. What aspects concern you most?"},
    {"speaker": "User", "text": "The economic impacts worry me. How do we balance growth with sustainability?"},
    # ... add the rest of the dialogue here ...
]

# Edit this mapping to match your rewrite exactly.
SYNONYMS = {
    "climate change": "global warming",
    "economic": "financial",
    "growth": "expansion",
    "sustainability": "long-term viability",
    "renewable": "green",
    "carbon": "CO2",
}

def rewrite(text: str) -> str:
    out = text
    for k, v in SYNONYMS.items():
        out = out.replace(k, v).replace(k.title(), v)
    return out

def main() -> None:
    cfg = default_config()
    cfg["min_persistence"] = 0.03
    cfg["min_witness_tokens"] = 2
    cfg["h0_mode"] = "theme"

    original = analyse_conversation_dynamics(ORIGINAL_TURNS, config=cfg, verbose=False)
    rewritten_turns = [{"speaker": t["speaker"], "text": rewrite(t["text"])} for t in ORIGINAL_TURNS]
    rewritten = analyse_conversation_dynamics(rewritten_turns, config=cfg, verbose=False)

    def totals(res):
        total = {"carry": 0, "drift": 0, "rupture": 0, "reentry": 0, "birth": 0}
        for step in res["dynamics"]["transitions"]:
            for k in total:
                total[k] += int(step.get(k, 0))
        return total

    print("ORIGINAL totals:", totals(original))
    print("REWRITTEN totals:", totals(rewritten))

if __name__ == "__main__":
    main()
