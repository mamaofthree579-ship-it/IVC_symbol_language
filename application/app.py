# app.py
import streamlit as st
import numpy as np
import cv2
import pytesseract
import json
import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# -----------------------
# Config / files
# -----------------------
st.set_page_config(page_title="IVC Analyzer — Editor & Labeler", layout="wide", initial_sidebar_state="collapsed")
st.title("🌀 IVC Analyzer — Library Editor & Manual Labeler")

SYMBOL_LIB_FILE = "ivc_symbol_library.json"
LOG_FILE = "ivc_symbol_log.csv"
COMPARE_LOG_FILE = "ivc_compare_log.csv"
LABELED_CSV = "ivc_labeled_dataset.csv"

# Default library (if JSON missing)
DEFAULT_LIBRARY = {
    "spiral_arrow": {"core": "Energy flow initiation / Field rotation", "domain": "Rotational Vortex", "notes": ""},
    "nested_squares": {"core": "Dimensional layering", "domain": "Compression", "notes": ""},
    "lattice": {"core": "Stabilization / Field grid", "domain": "Resonant Grid", "notes": ""},
    "triangle_in_square": {"core": "Coupled tunneling", "domain": "Quantum Interface", "notes": ""},
    "fish": {"core": "Particle / Consciousness unit", "domain": "Quantum-Material Interface", "notes": ""}
}

# ensure symbol library exists
if not os.path.exists(SYMBOL_LIB_FILE):
    with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_LIBRARY, f, indent=2)

with open(SYMBOL_LIB_FILE, "r", encoding="utf-8") as f:
    SYMBOL_LIB = json.load(f)

# -----------------------
# Helpers: IO and logs
# -----------------------
def append_csv_row(filepath: str, row: Dict[str, Any]):
    new_file = not os.path.exists(filepath)
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        st.error(f"Failed to append to {filepath}: {e}")

def save_symbol_library(lib: Dict):
    try:
        with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
            json.dump(lib, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save library: {e}")
        return False

# -----------------------
# Image / detection helpers
# -----------------------
def read_image(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def detect_contour_candidates(img, min_area_ratio=0.0005):
    """Return list of candidate crops and detection metadata."""
    gray = preprocess_gray(img)
    edges = cv2.Canny(gray, 100, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    min_area = max(80, int(h * w * min_area_ratio))
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        pad = int(0.05 * max(ww, hh))
        xa, ya = max(0, x - pad), max(0, y - pad)
        xb, yb = min(w, x + ww + pad), min(h, y + hh + pad)
        crop = img[ya:yb, xa:xb]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        sides = len(approx)
        label_hint = "unknown"
        if sides == 3:
            label_hint = "triangle"
        elif sides == 4:
            label_hint = "quad"
        elif sides > 6:
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            label_hint = "circle" if circularity > 0.6 else "complex"
        candidates.append({"bbox": (xa, ya, xb, yb), "area": area, "crop": crop, "hint": label_hint})
    candidates = sorted(candidates, key=lambda x: -x["area"])
    return candidates, edges

# -----------------------
# Template matching
# -----------------------
def run_template_matching(img, templates: List[np.ndarray], threshold=0.65):
    gray = preprocess_gray(img)
    matches = []
    for i, templ in enumerate(templates):
        try:
            tgray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, tgray, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)
            if maxv >= threshold:
                th, tw = tgray.shape[:2]
                x, y = maxloc
                matches.append({"template_idx": i, "score": float(maxv), "bbox": (x, y, x + tw, y + th)})
        except Exception:
            continue
    return matches

# -----------------------
# OCR
# -----------------------
def run_ocr(img):
    try:
        gray = preprocess_gray(img)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text", []) if w.strip()]
        text = " ".join(words).strip()
        return text, data
    except Exception as e:
        return f"[OCR error: {e}]", None

# -----------------------
# Energy visuals
# -----------------------
def plot_energy_lines(edges):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(edges, cmap="inferno")
    ax.axis("off")
    fig.tight_layout()
    return fig

# -----------------------
# Symbol matching heuristics
# -----------------------
def match_symbols(detected: Dict, templates_matched: List[Dict], ocr_text: str, lib: Dict) -> List[str]:
    matches = []
    det_shapes = set([s.lower() for s in detected.get("shapes", [])])
    det_patterns = set([p.lower() for p in detected.get("patterns", [])])
    if "lattice" in det_patterns and "lattice" in lib:
        matches.append("lattice")
    if "nested_squares" in det_patterns and "nested_squares" in lib:
        matches.append("nested_squares")
    if "spiral_cluster" in det_patterns and "spiral_arrow" in lib:
        matches.append("spiral_arrow")
    for tm in templates_matched:
        if tm.get("label"):
            if tm["label"] in lib and tm["label"] not in matches:
                matches.append(tm["label"])
    if isinstance(ocr_text, str):
        low = ocr_text.lower()
        for k in lib.keys():
            if k.replace("_", " ") in low and k not in matches:
                matches.append(k)
    if "triangle" in det_shapes and "square" in det_shapes and "triangle_in_square" in lib and "triangle_in_square" not in matches:
        matches.append("triangle_in_square")
    out = []
    for m in matches:
        if m not in out:
            out.append(m)
    return out

# -----------------------
# UI Layout
# -----------------------
tabs = st.tabs(["🔍 Analyze", "✍️ Manual Label / Annotate", "📘 Library Editor", "📂 Logs"])

# =======================
# TAB: Manual Label / Annotate
# =======================
with tabs[1]:
    st.header("Manual Labeling & Annotation (create training data)")
    st.markdown("Upload an image, inspect detected candidate crops, and assign a label from the library to create a labeled dataset.")
    up = st.file_uploader("Upload image for annotation", type=["jpg", "jpeg", "png"], key="label_upload")
    if up:
        imgL = read_image(up)
        candidates, edges = detect_contour_candidates(imgL)
        st.image(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB), caption="Source image", use_column_width=True)
        st.markdown(f"Detected {len(candidates)} candidate regions (filtered by size).")
        if not candidates:
            st.info("No candidates found. Try a different image.")
        else:
            cols = st.columns(3)
            assigned = []
            for i, cand in enumerate(candidates):
                crop = cand["crop"]
                bbox = cand["bbox"]
                col = cols[i % 3]
                with col:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_column_width=True)
                    hint = cand.get("hint", "")
                    label = st.selectbox(
                        f"Assign label for region #{i} (hint: {hint})",
                        options=["(none)"] + list(SYMBOL_LIB.keys()),
                        index=0,
                        key=f"lbl_{i}"
                    )
                    notes = st.text_input(f"Notes #{i}", key=f"note_{i}")
                    assigned.append({
                        "index": i,
                        "bbox": bbox,
                        "label": label if label != "(none)" else "",
                        "notes": notes
                    })
            if st.button("Save labeled rows"):
                rows = []
                for a in assigned:
                    if not a["label"]:
                        continue
                    x1, y1, x2, y2 = a["bbox"]
                    row = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "image_file": getattr(up, "name", "uploaded"),
                        "bbox": f"{x1},{y1},{x2},{y2}",
                        "label": a["label"],
                        "notes": a["notes"]
                    }
                    rows.append(row)
                if rows:
                    for r in rows:
                        append_csv_row(LABELED_CSV, r)
                    st.success(f"Saved {len(rows)} labeled rows to {LABELED_CSV}")
                else:
                    st.info("No labels selected to save.")

# =======================
# TAB: Library Editor
# =======================
with tabs[2]:
    st.header("Symbol Library Editor")
    st.markdown("Edit the JSON for the symbol library below. Keys are symbol IDs (use underscores). Click Save to persist.")
    lib_text = json.dumps(SYMBOL_LIB, indent=2)
    edited = st.text_area("Edit library JSON", value=lib_text, height=400)
    if st.button("Save library"):
        try:
            newlib = json.loads(edited)
            if save_symbol_library(newlib):
                st.success("Library saved. Reload the app to use updated library.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
    st.json(SYMBOL_LIB)

# =======================
# TAB: Logs
# =======================
with tabs[3]:
    st.header("Logs & Datasets")
    if os.path.exists(LABELED_CSV):
        st.markdown("### Labeled Dataset")
        dfl = pd.read_csv(LABELED_CSV)
        st.dataframe(dfl, use_container_width=True)
        st.download_button("Download labeled dataset", data=dfl.to_csv(index=False).encode("utf-8"), file_name=LABELED_CSV)
    else:
        st.info("No labeled dataset yet. Use Manual Labeling tab to create one.")

st.markdown("---")
st.markdown("✅ App ready — you can now label symbols, edit your vector library, and export a dataset for model training.")
