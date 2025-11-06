"""
IVC Translator (Phase 1 + Logging)
-----------------------------------
Adds automatic logging of unclassified symbols for dataset building.
"""

import re, os, csv
from datetime import datetime
from typing import Dict, List

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
             "function": "Frequency modulation or rhythmic communication"}
}

LOG_FILE = "ivc_symbol_log.csv"   # dataset grows here

def log_unclassified(entry: Dict):
    """Append new or unknown patterns to a CSV dataset."""
    header = ["timestamp", "shapes", "patterns", "ocr_text", "notes", "pending_label"]
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(entry)

def ivc_translate(symbol_data: Dict[str, List[str]], ocr_text: str = "") -> str:
    interpretations = []
    shapes = symbol_data.get("shapes", [])
    patterns = symbol_data.get("patterns", [])
    frequencies = symbol_data.get("frequencies", [])

    unclassified_detected = False

    # --- interpret geometric forms ---
    for shape in shapes:
        key = shape.lower()
        if key in GEOMETRIC_MEANINGS:
            data = GEOMETRIC_MEANINGS[key]
            interpretations.append(
                f"**{shape.title()}** → Energy: *{data['energy']}*; Function: *{data['function']}*."
            )
        else:
            interpretations.append(f"Unclassified geometric form detected: **{shape}**")
            unclassified_detected = True

    # --- interpret patterns ---
    if "triad" in patterns:
        interpretations.append("Triadic relationship detected — multi-dimensional harmonic balance.")
    if "lattice" in patterns:
        interpretations.append("Networked structure indicates collective resonance management.")
    if "spiral-arrow" in patterns:
        interpretations.append("Spiral-arrow composite — directed vortex generation.")
    if not any(p in ["triad", "lattice", "spiral-arrow"] for p in patterns):
        interpretations.append("Unclassified pattern geometry.")
        unclassified_detected = True

    # --- frequencies ---
    if frequencies:
        avg_freq = round(sum(frequencies) / len(frequencies), 2)
        interpretations.append(f"Average resonance frequency: **{avg_freq} Hz** (a.u.)")

    # --- OCR hints ---
    if ocr_text:
        if re.search(r"[0-9]", ocr_text):
            interpretations.append("Numeric inscriptions suggest calibration or measurement data.")
        elif re.search(r"[A-Za-z]", ocr_text):
            interpretations.append("Alphabetic overlay detected — possible linguistic encoding.")
        else:
            interpretations.append("Symbolic markings only; no phonetic overlay found.")
    else:
        interpretations.append("No text detected; geometry-based interpretation only.")

    # --- log new data if needed ---
    if unclassified_detected:
        log_unclassified({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "shapes": ",".join(shapes),
            "patterns": ",".join(patterns),
            "ocr_text": ocr_text[:200],
            "notes": "Auto-logged from IVC Analyzer",
            "pending_label": ""
        })
        interpretations.append("🧾 Unclassified elements logged for dataset growth.")

    return "\n".join(interpretations)
