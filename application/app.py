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
import shutil
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="IVC Analyzer", layout="wide", initial_sidebar_state="collapsed")
st.title("🌀 IVC Analyzer — Energy & Symbolic Field Interpreter")

SYMBOL_LIB_FILE = "ivc_symbol_library.json"
LOG_FILE = "ivc_symbol_log.csv"
COMPARE_LOG_FILE = "ivc_compare_log.csv"
LABELED_CSV = "ivc_labeled_dataset.csv"
BACKUP_DIR = "backups"

# -----------------------
# Ensure backup directory exists
# -----------------------
os.makedirs(BACKUP_DIR, exist_ok=True)

def backup_file(filepath):
    """Create a timestamped backup of the given file."""
    if os.path.exists(filepath):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.basename(filepath)
        dest = os.path.join(BACKUP_DIR, f"{ts}_{fname}")
        try:
            shutil.copy(filepath, dest)
            st.info(f"Backup created: {dest}")
        except Exception as e:
            st.error(f"Backup failed for {filepath}: {e}")

# -----------------------
# Default Symbol Library
# -----------------------
DEFAULT_LIBRARY = {
    "spiral_arrow": {"core": "Energy flow initiation / Field rotation", "domain": "Rotational Vortex"},
    "nested_squares": {"core": "Dimensional layering / Boundary harmonics", "domain": "Gravitational Compression"},
    "lattice": {"core": "Stabilization / Field grid / Linkage", "domain": "Resonant Grid"},
    "triangle_in_square": {"core": "Coupled tunneling / Dimensional harmony", "domain": "Quantum Interface"},
    "fish": {"core": "Particle / Consciousness unit / Self", "domain": "Quantum-Material Interface"}
}

# -----------------------
# Init symbol library
# -----------------------
if not os.path.exists(SYMBOL_LIB_FILE):
    with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_LIBRARY, f, indent=2)

with open(SYMBOL_LIB_FILE, "r", encoding="utf-8") as f:
    SYMBOL_LIB = json.load(f)

# -----------------------
# CSV Utilities
# -----------------------
def append_csv_row(filepath, row):
    """Append row and automatically back up CSV after update."""
    new_file = not os.path.exists(filepath)
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        backup_file(filepath)
    except Exception as e:
        st.error(f"Error writing to {filepath}: {e}")

def safe_load_csv(path: str) -> pd.DataFrame:
    """Safely load a CSV, skipping malformed rows."""
    import csv
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            for r in reader:
                if len(r) == len(header):
                    rows.append(r)
        return pd.DataFrame(rows, columns=header)
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return pd.DataFrame()

# -----------------------
# Image Utilities
# -----------------------
def read_image(uploaded):
    data = np.frombuffer(uploaded.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def preprocess_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def detect_edges(gray):
    return cv2.Canny(gray, 100, 200)

def plot_energy_lines(edges):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(edges, cmap="inferno")
    ax.set_title("Energy Lines Map (Inferno)")
    ax.axis("off")
    return fig

def plot_field_flow(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    mag = cv2.magnitude(grad_x, grad_y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mag, cmap="plasma")
    ax.set_title("Field Flow Overlay")
    ax.axis("off")
    return fig

# -----------------------
# OCR
# -----------------------
def run_ocr(img):
    try:
        gray = preprocess_gray(img)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        return f"[OCR Error: {e}]"

# -----------------------
# Main Tabs
# -----------------------
tabs = st.tabs(["🔍 Analyze", "✍️ Manual Label / Annotate", "📘 Library Editor", "📂 Logs"])

# =======================
# TAB 1: Analyze
# =======================
with tabs[0]:
    st.header("Analyze Image for IVC Symbolism and Energy Mapping")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = read_image(uploaded)
        gray = preprocess_gray(img)
        edges = detect_edges(gray)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        # Show Energy Lines and Field Flow stacked
        st.pyplot(plot_energy_lines(edges))
        st.pyplot(plot_field_flow(gray))

        # OCR extraction
        ocr_text = run_ocr(img)
        st.markdown("### OCR Extracted Text")
        st.text(ocr_text if ocr_text else "[No text detected]")

        # Simple matching
        matched = []
        for sym, info in SYMBOL_LIB.items():
            if sym.replace("_", " ").lower() in ocr_text.lower():
                matched.append(sym)

        if matched:
            st.success(f"Recognized symbols: {', '.join(matched)}")
            for m in matched:
                st.write(f"**{m}** → {SYMBOL_LIB[m]['core']} ({SYMBOL_LIB[m]['domain']})")
        else:
            st.warning("No direct textual symbol matches found.")

        # Log result
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "image_name": uploaded.name,
            "ocr_text": ocr_text,
            "recognized_symbols": ", ".join(matched)
        }
        append_csv_row(LOG_FILE, row)

# =======================
# TAB 2: Manual Label
# =======================
with tabs[1]:
    st.header("Manual Labeling & Annotation (Training Data)")
    up = st.file_uploader("Upload image for labeling", type=["jpg", "jpeg", "png"], key="label_upload")
    if up:
        imgL = read_image(up)
        grayL = preprocess_gray(imgL)
        edgesL = detect_edges(grayL)
        st.image(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB), caption="Source Image", use_column_width=True)

        st.pyplot(plot_energy_lines(edgesL))

        label = st.selectbox("Assign symbol label", ["(none)"] + list(SYMBOL_LIB.keys()))
        notes = st.text_input("Notes")
        if st.button("Save labeled entry"):
            if label != "(none)":
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "image_file": up.name,
                    "label": label,
                    "notes": notes
                }
                append_csv_row(LABELED_CSV, row)
                st.success(f"Labeled entry saved: {label}")
            else:
                st.warning("Please select a label before saving.")

# =======================
# TAB 3: Library Editor
# =======================
with tabs[2]:
    st.header("Edit Symbol Library")
    st.markdown("Modify the JSON below to add or update IVC symbols.")
    lib_text = json.dumps(SYMBOL_LIB, indent=2)
    edited = st.text_area("Edit JSON", value=lib_text, height=400)
    if st.button("Save Library"):
        try:
            new_lib = json.loads(edited)
            backup_file(SYMBOL_LIB_FILE)
            with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
                json.dump(new_lib, f, indent=2)
            st.success("Library saved successfully!")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
    st.json(SYMBOL_LIB)

# =======================
# TAB 4: Logs
# =======================
with tabs[3]:
    st.header("Logs & Datasets")

    if os.path.exists(LOG_FILE):
        st.markdown("### Analysis Log")
        df = safe_load_csv(LOG_FILE)
        st.dataframe(df)
        st.download_button("Download Analysis Log", data=df.to_csv(index=False).encode("utf-8"), file_name=LOG_FILE)
    else:
        st.info("No analysis log yet.")

    if os.path.exists(LABELED_CSV):
        st.markdown("### Labeled Dataset")
        dfl = safe_load_csv(LABELED_CSV)
        st.dataframe(dfl)
        st.download_button("Download Labeled Dataset", data=dfl.to_csv(index=False).encode("utf-8"), file_name=LABELED_CSV)
    else:
        st.info("No labeled dataset yet.")
