"""
IVC Translator  (Phase 1: rule-based prototype)
------------------------------------------------
Maps detected symbol features and OCR text to preliminary
IVC energetic and functional interpretations.

You can expand the dictionaries and functions below to
represent the full IVC algorithm.
"""

import re
from typing import Dict, List

# ---------------------------------------------------------------------
# 1. Knowledge base: basic geometric → functional mappings
# ---------------------------------------------------------------------

GEOMETRIC_MEANINGS = {
    "spiral": {
        "energy": "Centripetal / centrifugal flow pattern",
        "function": "Field convergence or expansion; dynamic energy gate"
    },
    "lattice": {
        "energy": "Stable harmonic network",
        "function": "Field stabilization or resonance harmonizer"
    },
    "triangle": {
        "energy": "Triadic resonance balance",
        "function": "Three-phase energetic circuit; balance of polarity"
    },
    "square": {
        "energy": "Grounding / containment field",
        "function": "Material stabilization and spatial anchoring"
    },
    "arrow": {
        "energy": "Directed force vector",
        "function": "Intentional projection or energy transmission"
    },
    "circle": {
        "energy": "Closed resonance loop",
        "function": "Containment, cycling, and self-referential field"
    },
    "wave": {
        "energy": "Oscillatory motion",
        "function": "Frequency modulation or rhythmic communication"
    }
}


# ---------------------------------------------------------------------
# 2. Translator function
# ---------------------------------------------------------------------

def ivc_translate(symbol_data: Dict[str, List[str]], ocr_text: str = "") -> str:
    """
    Generate a rule-based translation from detected symbol features
    and OCR-extracted text.
    """

    interpretations = []

    shapes = symbol_data.get("shapes", [])
    patterns = symbol_data.get("patterns", [])
    frequencies = symbol_data.get("frequencies", [])

    # --- interpret geometric forms ---
    for shape in shapes:
        key = shape.lower()
        if key in GEOMETRIC_MEANINGS:
            data = GEOMETRIC_MEANINGS[key]
            interpretations.append(
                f"**{shape.title()}** → Energy: *{data['energy']}*; Function: *{data['function']}*."
            )

    # --- interpret structural patterns ---
    if "triad" in patterns:
        interpretations.append("Triadic relationship detected — suggests multi-dimensional harmonic balance.")
    if "lattice" in patterns:
        interpretations.append("Networked structure indicates collective resonance management.")
    if "spiral-arrow" in patterns:
        interpretations.append("Spiral-arrow composite — signifies directed vortex generation.")

    # --- frequency cues ---
    if frequencies:
        avg_freq = round(sum(frequencies) / len(frequencies), 2)
        interpretations.append(f"Average resonance frequency: **{avg_freq} Hz** (arbitrary units).")

    # --- OCR-based hints ---
    if ocr_text:
        if re.search(r"[0-9]", ocr_text):
            interpretations.append("Numeric inscriptions suggest quantitative calibration or measurement data.")
        elif re.search(r"[A-Za-z]", ocr_text):
            interpretations.append("Alphabetic characters found — possible linguistic overlay detected.")
        else:
            interpretations.append("Symbolic markings only; no phonetic overlay found.")
    else:
        interpretations.append("No text detected; analysis based on geometry only.")

    # --- fallback if nothing recognized ---
    if not interpretations:
        interpretations.append("Unclassified pattern — add to dataset for future training.")

    return "\n".join(interpretations)


# ---------------------------------------------------------------------
# 3. Simple demo (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_input = {
        "shapes": ["spiral", "arrow"],
        "patterns": ["spiral-arrow", "triad"],
        "frequencies": [12.3, 14.8, 13.2]
    }
    print(ivc_translate(demo_input, ocr_text="AB12"))
