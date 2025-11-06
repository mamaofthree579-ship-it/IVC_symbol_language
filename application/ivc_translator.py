import re
import pandas as pd
from datetime import datetime
from typing import Dict, List
import os

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

# CSV log location
LOG_FILE = "ivc_symbol_log.csv"


# ---------------------------------------------------------------------
# 2. Helper: log unrecognized symbols
# ---------------------------------------------------------------------

def log_unclassified(symbol_name: str, category: str, ocr_text: str):
    """Append unrecognized items to ivc_symbol_log.csv for future training."""
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "category": category,
        "symbol_name": symbol_name,
        "ocr_text": ocr_text.strip()[:100]
    }

    df = pd.DataFrame([record])

    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode="a", index=False, header=False)


# ---------------------------------------------------------------------
# 3. Translator function
# ---------------------------------------------------------------------

def ivc_translate(symbol_data: Dict[str, List[str]], ocr_text: str = "") -> str:
    """
    Generate a rule-based translation and log any unknown features.
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
        else:
            interpretations.append(f"🧩 Unknown shape detected: **{shape}** — logged for review.")
            log_unclassified(shape, "shape", ocr_text)

    # --- interpret structural patterns ---
    known_patterns = ["triad", "lattice", "spiral-arrow"]
    for pattern in patterns:
        if pattern in known_patterns:
            if pattern == "triad":
                interpretations.append("Triadic relationship detected — suggests multi-dimensional harmonic balance.")
            elif pattern == "lattice":
                interpretations.append("Networked structure indicates collective resonance management.")
            elif pattern == "spiral-arrow":
                interpretations.append("Spiral-arrow composite — signifies directed vortex generation.")
        else:
            interpretations.append(f"🧩 Unknown pattern detected: **{pattern}** — logged for review.")
            log_unclassified(pattern, "pattern", ocr_text)

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

    # --- fallback ---
    if not interpretations:
        interpretations.append("Unclassified pattern — add to dataset for future training.")

    return "\n".join(interpretations)


# ---------------------------------------------------------------------
# 4. Simple demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    demo_input = {
        "shapes": ["spiral", "unknown_shape"],
        "patterns": ["spiral-arrow", "mystery_pattern"],
        "frequencies": [12.3, 14.8, 13.2]
    }
    print(ivc_translate(demo_input, ocr_text="AB12"))
    print("\n→ Check ivc_symbol_log.csv for logged items.")
