"""
IVC Analyzer — Streamlit App (mobile + dataset viewer)
-------------------------------------------------------
Uploads an ancient artifact image, runs OCR + symbol analysis + IVC translation,
and lets you view your auto-growing dataset (ivc_symbol_log.csv).
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
from io import BytesIO
from datetime import datetime
from ivc_translator import ivc_translate, LOG_FILE

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="IVC Analyzer",
    page_icon="🌀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🌀 IVC Symbol Analyzer")
st.caption("Upload an artifact image, run the IVC Algorithm, and explore your growing translation dataset.")

# ---------------------------------
# FILE UPLOAD
# ---------------------------------
uploaded_file = st.file_uploader("📤 Upload artifact image", type=["jpg", "jpeg", "png"])

# ---------------------------------
# HELPER FUNCTIONS
# ---------------------------------
def preprocess_image(file_bytes):
    """Convert upload to grayscale OpenCV image."""
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return gray

def extract_symbols(gray):
    """Simple mock: detect shapes using contour approximation."""
    detected_shapes = []
    edges = cv2.Canny(gray, 50, 150)
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
    # Just mock some pattern detection
    patterns = ["lattice"] if len(detected_shapes) > 2 else []
    frequencies = [np.random.uniform(7, 13) for _ in range(3)]
    return {"shapes": detected_shapes, "patterns": patterns, "frequencies": frequencies}

def run_ocr(gray):
    """Run OCR (Tesseract must be installed locally)."""
    try:
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        return f"[OCR unavailable: {e}]"

# ---------------------------------
# MAIN TABS
# ---------------------------------
tabs = st.tabs(["🔍 Analyze", "📘 Symbol Log Viewer"])

# ---------------------------------
# TAB 1: ANALYZE
# ---------------------------------
with tabs[0]:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Artifact", use_container_width=True)

        if st.button("▶️ Run IVC Analysis"):
            file_bytes = uploaded_file.read()
            gray = preprocess_image(file_bytes)

            with st.spinner("Running OCR and geometric decoding..."):
                ocr_text = run_ocr(gray)
                symbol_data = extract_symbols(gray)
                translation_output = ivc_translate(symbol_data, ocr_text)

            st.success("✅ Analysis complete")
            st.markdown("### 🧩 Symbol Data")
            st.json(symbol_data)

            st.markdown("### 📜 IVC Translation")
            st.markdown(translation_output)

            # Optional: save basic report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report = {
                "timestamp": timestamp,
                "symbols": symbol_data,
                "ocr_text": ocr_text,
                "translation": translation_output
            }
            df = pd.DataFrame([report])
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Download Session Report",
                data=csv,
                file_name=f"IVC_report_{timestamp}.csv",
                mime="text/csv"
            )

    else:
        st.info("📸 Please upload an image to begin analysis.")

# ---------------------------------
# TAB 2: SYMBOL LOG VIEWER
# ---------------------------------
with tabs[1]:
    st.markdown("### 📘 Logged Symbols & Unclassified Entries")
    try:
        df = pd.read_csv(LOG_FILE)
        search = st.text_input("🔍 Search shapes or patterns")
        if search:
            mask = df.apply(lambda r: search.lower() in r.astype(str).str.lower().to_string(), axis=1)
            df_filtered = df[mask]
        else:
            df_filtered = df
        st.dataframe(df_filtered, use_container_width=True)

        st.caption(f"Total logged entries: {len(df)}")
        st.download_button(
            label="💾 Download Symbol Log CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="ivc_symbol_log.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.info("🧾 No log file yet — new symbols will appear here after analysis.")
