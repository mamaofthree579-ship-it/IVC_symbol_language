import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict

# --- SETUP ---
st.set_page_config(page_title="IVC Symbol Decoder", layout="wide")

if "symbol_log" not in st.session_state:
    st.session_state["symbol_log"] = []

# --- UTILITIES ---

def save_symbol_log(entry: Dict):
    st.session_state["symbol_log"].append(entry)
    with open("symbol_log.json", "w") as f:
        json.dump(st.session_state["symbol_log"], f, indent=2)


def analyze_symbols(ocr_text: str) -> Dict:
    shapes = []
    text = ocr_text.lower()
    if "triangle" in text or "△" in text:
        shapes.append("triangle")
    if "square" in text or "□" in text:
        shapes.append("square")
    if "circle" in text or "○" in text:
        shapes.append("circle")
    if "spiral" in text or "↻" in text:
        shapes.append("spiral")
    if "arrow" in text or "→" in text:
        shapes.append("arrow")
    if "grid" in text or "lattice" in text:
        shapes.append("lattice")
    return {"shapes": shapes}


# --- ENERGY FIELD VISUALIZATION ---

def draw_energy_overlay(img: np.ndarray, symbol_data: Dict, mode: str = "Edge Flow") -> np.ndarray:
    """Visualize energy fields derived from the artifact image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- Edge Flow Mode ---
    edges = cv2.Canny(gray, 100, 200)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    mag_uint8 = np.uint8(mag_norm)
    hsv = np.zeros_like(img)
    hsv[..., 0] = np.uint8((angle / 2) % 180)
    hsv[..., 1] = 255
    hsv[..., 2] = mag_uint8
    flow_colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    edge_colored = cv2.bitwise_and(flow_colored, flow_colored, mask=edges)
    edge_colored = cv2.GaussianBlur(edge_colored, (3, 3), 0)
    edge_overlay = cv2.addWeighted(img, 0.7, edge_colored, 0.8, 0)

    # --- Gradient Field Mode ---
    y_indices, x_indices = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    freq = 0.02 * (len(symbol_data.get("shapes", [])) + 1)
    flow = (np.sin(x_indices * freq) + np.cos(y_indices * freq * 0.7)) * 127 + 128
    flow = np.uint8(flow)
    plasma = cv2.applyColorMap(flow, cv2.COLORMAP_TWILIGHT)
    gradient_overlay = cv2.addWeighted(img, 0.7, plasma, 0.5, 0)

    # --- Hybrid Mode ---
    if mode == "Hybrid":
        result = cv2.addWeighted(edge_overlay, 0.7, gradient_overlay, 0.7, 0)
    elif mode == "Gradient Field":
        result = gradient_overlay
    else:
        result = edge_overlay

    # --- Symbol Tint ---
    tint = np.zeros_like(img)
    color_map = {
        "spiral": (0, 255, 255),
        "triangle": (0, 128, 255),
        "square": (0, 255, 0),
        "circle": (255, 0, 0),
        "arrow": (255, 0, 255),
        "lattice": (255, 255, 0),
    }
    for shape in symbol_data.get("shapes", []):
        tint[:] = cv2.add(tint, np.full_like(tint, color_map.get(shape.lower(), (200, 200, 200))))
    final = cv2.addWeighted(result, 0.9, tint, 0.1, 0)

    return final


# --- IVC TRANSLATOR (placeholder logic) ---
def ivc_translate(symbol_data: Dict, text: str) -> str:
    if not symbol_data["shapes"]:
        return "No recognized symbols for translation."
    desc = ", ".join(symbol_data["shapes"])
    return f"Decoded energy interaction pattern: {desc}. This represents balance between form and field."


# --- MAIN APP INTERFACE ---

st.title("🌀 IVC Symbol Decoder & Energy Mapping")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "🔬 Compare Mode", "📜 Symbol Log Viewer"])

# --- TAB 1: ANALYZE SINGLE IMAGE ---
with tab1:
    st.header("Analyze Artifact Image")
    field_mode = st.sidebar.selectbox("Field Visualization Mode", ["Edge Flow", "Gradient Field", "Hybrid"])

    uploaded_file = st.file_uploader("Upload Artifact Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            ocr_text = pytesseract.image_to_string(gray)
        except:
            ocr_text = "Tesseract not available."

        symbol_data = analyze_symbols(ocr_text)
        energy_map = draw_energy_overlay(img, symbol_data, field_mode)
        translation_output = ivc_translate(symbol_data, ocr_text)

        st.image(cv2.cvtColor(energy_map, cv2.COLOR_BGR2RGB), caption="Energy Mapping", use_column_width=True)
        st.subheader("Translation Output")
        st.write(translation_output)
        st.text_area("Detected OCR Text", ocr_text, height=150)

        log_entry = {
            "timestamp": str(datetime.now()),
            "symbols": symbol_data["shapes"],
            "ocr": ocr_text,
            "translation": translation_output
        }
        save_symbol_log(log_entry)

# --- TAB 2: COMPARE MODE ---
with tab2:
    st.header("Compare Two Artifacts Side-by-Side")
    field_mode = st.sidebar.selectbox("Compare Field Mode", ["Edge Flow", "Gradient Field", "Hybrid"])

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload First Artifact", key="file1", type=["jpg", "png", "jpeg"])
    with col2:
        file2 = st.file_uploader("Upload Second Artifact", key="file2", type=["jpg", "png", "jpeg"])

    if file1 and file2:
        img1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        ocr1 = pytesseract.image_to_string(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        ocr2 = pytesseract.image_to_string(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

        symbols1 = analyze_symbols(ocr1)
        symbols2 = analyze_symbols(ocr2)

        map1 = draw_energy_overlay(img1, symbols1, field_mode)
        map2 = draw_energy_overlay(img2, symbols2, field_mode)

        sim_score = len(set(symbols1["shapes"]) & set(symbols2["shapes"])) / max(
            1, len(set(symbols1["shapes"]) | set(symbols2["shapes"]))
        )
        st.metric("Symbolic Similarity", f"{sim_score * 100:.1f}%")

        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(map1, cv2.COLOR_BGR2RGB), caption="Artifact 1 Energy Map", use_column_width=True)
        with c2:
            st.image(cv2.cvtColor(map2, cv2.COLOR_BGR2RGB), caption="Artifact 2 Energy Map", use_column_width=True)

# --- TAB 3: LOG VIEWER ---
with tab3:
    st.header("Symbol Log Viewer")
    if st.session_state["symbol_log"]:
        df = pd.DataFrame(st.session_state["symbol_log"])
        st.dataframe(df)
        st.download_button(
            "Download Symbol Log",
            data=json.dumps(st.session_state["symbol_log"], indent=2),
            file_name="symbol_log.json",
            mime="application/json"
        )
    else:
        st.info("No symbol analyses logged yet.")
