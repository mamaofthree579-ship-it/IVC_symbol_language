"""
IVC Analyzer (Enhanced Logging + Energy & OCR Visualization)
-------------------------------------------------------------
Uploads an image, runs symbol extraction, OCR, and IVC translation.
Now logs all detected symbols and shows energy + OCR overlays.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
from datetime import datetime
from ivc_translator import ivc_translate, log_unclassified, LOG_FILE, GEOMETRIC_MEANINGS

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="IVC Analyzer",
    page_icon="🌀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🌀 IVC Symbol Analyzer")
st.caption("Decode, visualize, and log ancient IVC field control symbols.")

# ------------------------------
# HELPERS
# ------------------------------
def preprocess_image(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return img, gray

def extract_symbols(gray):
    """Simple shape detection using contour approximation."""
    detected_shapes = []
    edges = cv2.Canny(gray, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        sides = len(approx)
        if sides == 3:
            detected_shapes.append("triangle")
        elif sides == 4:
            detected_shapes.append("square")
        elif sides > 5:
            detected_shapes.append("circle")
    patterns = ["lattice"] if len(detected_shapes) > 2 else []
    frequencies = [np.random.uniform(7, 13) for _ in range(3)]
    return {"shapes": detected_shapes, "patterns": patterns, "frequencies": frequencies}

def run_ocr(gray):
    """Run OCR and return text + bounding box data."""
    try:
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        text = " ".join([w for w in data["text"] if w.strip()])
        return text.strip(), data
    except Exception as e:
        return f"[OCR unavailable: {e}]", None

def draw_energy_overlay(image, symbol_data):
    """Draw color-coded energy zones based on detected shapes."""
    overlay = image.copy()
    color_map = {
        "spiral": (0, 255, 255),
        "triangle": (0, 128, 255),
        "square": (0, 255, 0),
        "circle": (255, 0, 0),
        "arrow": (255, 0, 255),
        "lattice": (255, 255, 0)
    }
    h, w, _ = image.shape
    cx, cy = w // 2, h // 2

    for shape in symbol_data["shapes"]:
        color = color_map.get(shape.lower(), (200, 200, 200))
        if shape == "triangle":
            pts = np.array([[cx, cy - 60], [cx - 50, cy + 60], [cx + 50, cy + 60]], np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        elif shape == "square":
            cv2.rectangle(overlay, (cx - 50, cy - 50), (cx + 50, cy + 50), color, 2)
        elif shape == "circle":
            cv2.circle(overlay, (cx, cy), 60, color, 2)
        elif shape == "spiral":
            for r in range(10, 70, 10):
                cv2.circle(overlay, (cx, cy), r, color, 1)
        elif shape == "arrow":
            cv2.arrowedLine(overlay, (cx - 50, cy), (cx + 50, cy), color, 2, tipLength=0.3)
    return overlay

def draw_ocr_overlay(image, ocr_data):
    """Draw rectangles over recognized OCR text regions."""
    overlay = image.copy()
    if ocr_data and "text" in ocr_data:
        n_boxes = len(ocr_data["text"])
        for i in range(n_boxes):
            if int(ocr_data["conf"][i]) > 30:
                (x, y, w, h) = (ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i])
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return overlay

def log_run(symbol_data, ocr_text):
    """Log every run — not just unclassified ones."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "shapes": ",".join(symbol_data.get("shapes", [])),
        "patterns": ",".join(symbol_data.get("patterns", [])),
        "ocr_text": ocr_text[:200],
        "notes": "Auto-logged run (all symbols)",
        "pending_label": ""
    }
    log_unclassified(entry)

# ------------------------------
# UI TABS
# ------------------------------
tabs = st.tabs(["🔍 Analyze", "📘 Symbol Log Viewer"])

# --- TAB 1: Analyze ---
with tabs[0]:
    uploaded_file = st.file_uploader("📤 Upload artifact image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Artifact", use_container_width=True)

        if st.button("▶️ Run IVC Analysis"):
            file_bytes = uploaded_file.read()
            img, gray = preprocess_image(file_bytes)

            with st.spinner("Running symbol detection + OCR..."):
                ocr_text, ocr_data = run_ocr(gray)
                symbol_data = extract_symbols(gray)
                translation_output = ivc_translate(symbol_data, ocr_text)
                log_run(symbol_data, ocr_text)

            st.success("✅ Analysis Complete")

            st.markdown("### 🧩 Symbol Data")
            st.json(symbol_data)

            st.markdown("### 🔠 OCR Text")
            st.code(ocr_text or "[No text detected]")

            # ENERGY MAPPING
            st.markdown("### ⚡ Energy Mapping Visualization")
            overlay_energy = draw_energy_overlay(img, symbol_data)
            st.image(cv2.cvtColor(overlay_energy, cv2.COLOR_BGR2RGB),
                     caption="Energy Flow Overlay", use_container_width=True)

            # OCR VISUALIZATION
            st.markdown("### 🔤 OCR Visualization")
            if ocr_data:
                overlay_ocr = draw_ocr_overlay(img, ocr_data)
                st.image(cv2.cvtColor(overlay_ocr, cv2.COLOR_BGR2RGB),
                         caption="Detected Text Regions", use_container_width=True)
            else:
                st.info("No OCR overlay available.")

            # TRANSLATION
            st.markdown("### 📜 IVC Translation Output")
            st.markdown(translation_output)

    else:
        st.info("📸 Please upload an artifact image to begin analysis.")

# --- TAB 2: Symbol Log Viewer ---
with tabs[1]:
    st.markdown("### 📘 Logged Symbol Data")
    try:
        df = pd.read_csv(LOG_FILE)
        search = st.text_input("🔍 Search symbols or text")
        if search:
            mask = df.apply(lambda r: search.lower() in r.astype(str).str.lower().to_string(), axis=1)
            df = df[mask]
        st.dataframe(df, use_container_width=True)
        st.caption(f"Total logged entries: {len(df)}")

        st.download_button(
            label="💾 Download Symbol Log CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ivc_symbol_log.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.info("🧾 No log file yet — symbols will appear here after the first run.")
