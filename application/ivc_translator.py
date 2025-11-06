"""
IVC Translator (Phase 1.5 – rule-based + auto logging, stable version)
-----------------------------------------------------------------------
Safely maps IVC shapes/patterns to meanings and logs unrecognized items.
"""

import re
import os
import csv
from datetime import datetime
from typing import Dict, List

# --------------------------
# GEOMETRIC KNOWLEDGE BASE
# --------------------------
GEOMETRIC_MEANINGS = {
    "spiral": {"energy": "Centripetal / centrifugal flow pattern",
               "function": "Field convergence or expansion; dynamic energy gate"},
    "lattice": {"energy": "Stable harmonic network",
                "function": "Field stabilization or resonance harmonizer"},
    "triangle": {"energy": "Triadic resonance balance",
                 "function": "Three-phase energetic circuit; balance of polarity"},
    "square": {"energy": "Grounding / containment field",
               "function": "Material stabilization and spatial anchoring"},
    "arrow": {"energy": "Directed force vector",
              "function": "Intentional projection or energy transmission"},
    "circle": {"energy": "Closed resonance loop",
               "function": "Containment, cycling, and self-referential field"},
    "wave": {"energy": "Oscillatory motion",
             "function": "Frequency modulation or rhythmic communication"},
}

LOG_FILE = "ivc_symbol_log.csv"


# --------------------------
# LOGGING FUNCTION
# --------------------------
def log_unclassified(entry: Dict):
    """Safely appends an entry to ivc_symbol_log.csv."""
    try:
        header = ["timestamp", "shapes", "patterns", "ocr_text", "notes", "pending_label"]
        file_exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)
    except Exception as e:
        # Fail silently in Streamlit to prevent user crash
        print(f"[IVC Translator] Logging error: {e}")


# --------------------------
# MAIN TRANSLATOR
# --------------------------
def ivc_translate(symbol_data: Dict[str, List[str]], ocr_text: str = "") -> str:
    """Generate a rule-based translation and log unknown shapes/patterns."""
    interpretations = []
    unclassified_detected = False

    shapes = symbol_data.get("shapes", []) or []
    patterns = symbol_data.get("patterns", []) or []
    frequencies = symbol_data.get("frequencies", []) or []

    # --- SHAPES ---
    for shape in shapes:
        key = shape.lower().strip()
        if key in GEOMETRIC_MEANINGS:
            data = GEOMETRIC_MEANINGS[key]
            interpretations.append(
                f"**{shape.title()}** → Energy: *{data['energy']}*; Function: *{data['function']}*."
            )
        else:
            interpretations.append(f"🧩 Unknown geometric form: **{shape}**")
            unclassified_detected = True

    # --- PATTERNS ---
    known_patterns = ["triad", "lattice", "spiral-arrow"]
    for p in patterns:
        if p == "triad":
            interpretations.append("Triadic relationship — multi-dimensional harmonic balance.")
        elif p == "lattice":
            interpretations.append("Networked structure — collective resonance field.")
        elif p == "spiral-arrow":
            interpretations.append("Spiral-arrow composite — directed vortex generation.")
        elif p not in known_patterns:
            interpretations.append(f"🧩 Unknown pattern: **{p}**")
            unclassified_detected = True

    # --- FREQUENCIES ---
    if frequencies:
        try:
            avg_freq = round(sum(frequencies) / len(frequencies), 2)
            interpretations.append(f"Average resonance frequency: **{avg_freq} Hz** (a.u.)")
        except Exception:
            interpretations.append("Frequency data unreadable.")

    # --- OCR ANALYSIS ---
    if ocr_text:
        if re.search(r"[0-9]", ocr_text):
            interpretations.append("Numeric inscriptions suggest calibration or measurement data.")
        elif re.search(r"[A-Za-z]", ocr_text):
            interpretations.append("Alphabetic overlay detected — possible linguistic encoding.")
        else:
            interpretations.append("Symbolic markings only; no phonetic overlay found.")
    else:
        interpretations.append("No text detected; geometry-based interpretation only.")

    # --- LOG UNCLASSIFIED ---
    if unclassified_detected:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "shapes": ",".join(shapes),
            "patterns": ",".join(patterns),
            "ocr_text": ocr_text[:200],
            "notes": "Auto-logged from IVC Analyzer",
            "pending_label": ""
        }
        log_unclassified(entry)
        interpretations.append("🧾 Unclassified symbols logged for future dataset training.")

    # --- FINAL OUTPUT ---
    if not interpretations:
        interpretations.append("No recognizable structures detected.")

    return "\n".join(interpretations)


# --------------------------
# DEMO
# --------------------------
if __name__ == "__main__":
    demo = {
        "shapes": ["spiral", "hexagon"],
        "patterns": ["triad", "mystery_pattern"],
        "frequencies": [10.2, 11.4, 9.9]
    }
    print(ivc_translate(demo, ocr_text="AB12"))
