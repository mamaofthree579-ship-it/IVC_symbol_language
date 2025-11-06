"""
IVC Analyzer — Mobile + Log Viewer + Hybrid Compare Mode
Place this file next to ivc_translator.py (which must provide ivc_translate, log_unclassified, LOG_FILE, GEOMETRIC_MEANINGS).
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import pytesseract
import io
import os
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List

# Import translator utilities from your ivc_translator module
from ivc_translator import ivc_translate, log_unclassified, LOG_FILE, GEOMETRIC_MEANINGS

# New compare log file
COMPARE_LOG_FILE = "ivc_compare_log.csv"

# -----------------------
# Page config & styling
# -----------------------
st.set_page_config(page_title="IVC Analyzer", page_icon="🌀", layout="centered", initial_sidebar_state="collapsed")
st.title("🌀 IVC Analyzer — Analyze · Log · Compare")
st.caption("Upload artifacts, decode with IVC, review logged items, or compare two artifacts (hybrid mode).")

st.markdown("""
<style>
button, .stTextInput, .stSelectbox, .stDownloadButton { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# Helpers: image preprocessing, detection, OCR
# -----------------------
def read_image_bytes(uploaded_file) -> np.ndarray:
    file_bytes = uploaded_file.read()
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img, file_bytes

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return gray

def extract_shapes_and_patterns(gray: np.ndarray) -> Dict:
    """Contour-based simple detector to find triangles/squares/circles and mock patterns."""
    shapes = []
    edges = cv2.Canny(gray, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        sides = len(approx)
        if sides == 3:
            shapes.append("triangle")
        elif sides == 4:
            shapes.append("square")
        elif sides > 5:
            shapes.append("circle")
    # remove duplicates, preserve order
    shapes = list(dict.fromkeys(shapes))
    # patterns heuristic
    patterns = []
    if len(shapes) >= 3:
        patterns.append("lattice")
    if "circle" in shapes and "square" in shapes:
        patterns.append("spiral-arrow")  # heuristic composite
    # frequencies: mocked set of resonance estimates
    frequencies = [round(float(np.random.uniform(7, 14)), 2) for _ in range(3)]
    return {"shapes": shapes, "patterns": patterns, "frequencies": frequencies, "edges": edges}

def run_ocr_and_data(gray: np.ndarray):
    try:
        ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        # join words into a single string
        words = [w for w in ocr_data.get("text", []) if w.strip()]
        text = " ".join(words).strip()
        return text, ocr_data
    except Exception as e:
        return f"[OCR error: {e}]", None

def draw_energy_overlay(img: np.ndarray, symbol_data: Dict) -> np.ndarray:
    """Simple overlay: draw color-coded shapes at center (heuristic)."""
    overlay = img.copy()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    color_map = {
        "spiral": (0, 255, 255),
        "triangle": (0, 128, 255),
        "square": (0, 255, 0),
        "circle": (255, 0, 0),
        "arrow": (255, 0, 255),
        "lattice": (255, 255, 0)
    }
    for i, shape in enumerate(symbol_data.get("shapes", [])):
        color = color_map.get(shape.lower(), (200, 200, 200))
        # draw scaled shapes around center to visualize "energy zones"
        offset = 40 + i * 30
        if shape == "triangle":
            pts = np.array([[cx, cy - offset], [cx - offset, cy + offset], [cx + offset, cy + offset]], np.int32)
            cv2.polylines(overlay, [pts], True, color, 2)
        elif shape == "square":
            cv2.rectangle(overlay, (cx - offset, cy - offset), (cx + offset, cy + offset), color, 2)
        elif shape == "circle":
            cv2.circle(overlay, (cx, cy), offset, color, 2)
        elif shape == "spiral":
            for r in range(10, offset+10, 10):
                cv2.circle(overlay, (cx, cy), r, color, 1)
        elif shape == "arrow":
            cv2.arrowedLine(overlay, (cx - offset, cy), (cx + offset, cy), color, 2, tipLength=0.3)
    return overlay

def draw_ocr_boxes(img: np.ndarray, ocr_data):
    overlay = img.copy()
    if not ocr_data:
        return overlay
    n = len(ocr_data.get("text", []))
    for i in range(n):
        try:
            conf = float(ocr_data["conf"][i])
        except Exception:
            conf = -1.0
        if conf > 30:
            x, y, w, h = int(ocr_data["left"][i]), int(ocr_data["top"][i]), int(ocr_data["width"][i]), int(ocr_data["height"][i])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return overlay

# -----------------------
# Utility: similarity scoring
# -----------------------
def jaccard(a: List[str], b: List[str]) -> float:
    set_a, set_b = set([s.lower() for s in a]), set([s.lower() for s in b])
    if not set_a and not set_b:
        return 1.0
    inter = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return len(inter) / len(union) if union else 0.0

def text_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()

# -----------------------
# Logging comparison results
# -----------------------
def log_compare_result(entry: Dict):
    header = ["timestamp", "imgA", "imgB", "shapesA", "shapesB", "patternsA", "patternsB",
              "ocrA", "ocrB", "shape_jaccard", "pattern_jaccard", "ocr_similarity", "combined_score", "notes"]
    new_file = not os.path.exists(COMPARE_LOG_FILE)
    try:
        with open(COMPARE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            df = pd.DataFrame([entry])
            if new_file:
                df.to_csv(f, index=False)
            else:
                df.to_csv(f, index=False, header=False)
    except Exception as e:
        print(f"[Compare log] write error: {e}")

# -----------------------
# UI Tabs: Analyze · Log Viewer · Compare
# -----------------------
tabs = st.tabs(["🔍 Analyze", "📘 Symbol Log Viewer", "🔬 Compare Mode"])

# ---------- TAB: Analyze ----------
with tabs[0]:
    st.header("Analyze a Single Artifact")
    uploaded_file = st.file_uploader("Upload artifact image (jpg/png)", type=["jpg", "jpeg", "png"], key="analyze_upload")
    if uploaded_file:
        img, file_bytes = read_image_bytes(uploaded_file)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Artifact", use_container_width=True)
        if st.button("▶️ Run IVC Analysis", key="run_single"):
            gray = preprocess_gray(img)
            symbol_data = extract_shapes_and_patterns(gray)
            ocr_text, ocr_data = run_ocr_and_data(gray)
            translation_output = ivc_translate(symbol_data, ocr_text)
            # full-run log (every run)
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "shapes": ",".join(symbol_data.get("shapes", [])),
                "patterns": ",".join(symbol_data.get("patterns", [])),
                "ocr_text": ocr_text[:200],
                "notes": "Auto-logged run (Analyze tab)",
                "pending_label": ""
            }
            # use translator logging helper for consistent storage
            try:
                log_unclassified(entry)
            except Exception:
                # fallback if direct writer not available
                pass

            st.success("✅ Analysis complete")
            st.subheader("Symbol Data")
            st.json({k:v for k,v in symbol_data.items() if k != "edges"})

            st.subheader("IVC Translation")
            st.markdown(translation_output)

            st.subheader("Energy Map (Overlay)")
            energy_overlay = draw_energy_overlay(img, symbol_data)
            st.image(cv2.cvtColor(energy_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.subheader("OCR output and overlay")
            st.code(ocr_text or "[No OCR text detected]")
            ocr_overlay = draw_ocr_boxes(img, ocr_data)
            st.image(cv2.cvtColor(ocr_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

            # session report download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_report = {
                "timestamp": timestamp,
                "shapes": ",".join(symbol_data.get("shapes", [])),
                "patterns": ",".join(symbol_data.get("patterns", [])),
                "ocr_text": ocr_text,
                "translation": translation_output
            }
            csv_bytes = pd.DataFrame([session_report]).to_csv(index=False).encode("utf-8")
            st.download_button(label="💾 Download Analysis CSV", data=csv_bytes, file_name=f"ivc_session_{timestamp}.csv", mime="text/csv")

# ---------- TAB: Symbol Log Viewer ----------
with tabs[1]:
    st.header("Symbol Log Viewer")
    st.markdown("Browse & download your `ivc_symbol_log.csv` (auto-grown).")
    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE)
            search = st.text_input("🔎 Search log (shapes, patterns, text)", key="log_search")
            if search:
                mask = df_log.apply(lambda r: search.lower() in r.astype(str).str.lower().to_string(), axis=1)
                df_filtered = df_log[mask]
            else:
                df_filtered = df_log
            st.dataframe(df_filtered, use_container_width=True)
            st.caption(f"Total log entries: {len(df_log)}")
            st.download_button("💾 Download symbol log", data=df_log.to_csv(index=False).encode("utf-8"), file_name="ivc_symbol_log.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to load log: {e}")
    else:
        st.info("No symbol log yet — run analyses to build it.")

# ---------- TAB: Compare Mode ----------
with tabs[2]:
    st.header("🔬 Compare Mode — Hybrid visual + database + energy mapping")
    col1, col2 = st.columns(2)
    with col1:
        imgA_file = st.file_uploader("Upload Image A", type=["jpg","jpeg","png"], key="cmp_A")
    with col2:
        imgB_file = st.file_uploader("Upload Image B", type=["jpg","jpeg","png"], key="cmp_B")

    if imgA_file and imgB_file:
        # read both
        imgA, _ = read_image_bytes(imgA_file)
        imgB, _ = read_image_bytes(imgB_file)

        st.markdown("### Preview (A | B)")
        cols = st.columns(2)
        cols[0].image(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB), use_container_width=True)
        cols[1].image(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB), use_container_width=True)

        if st.button("▶️ Run Compare Analysis"):
            # preprocess and extract
            grayA = preprocess_gray(imgA)
            grayB = preprocess_gray(imgB)
            dataA = extract_shapes_and_patterns(grayA)
            dataB = extract_shapes_and_patterns(grayB)
            ocrA, ocr_dataA = run_ocr_and_data(grayA)
            ocrB, ocr_dataB = run_ocr_and_data(grayB)

            # translations
            transA = ivc_translate(dataA, ocrA)
            transB = ivc_translate(dataB, ocrB)

            # similarity metrics
            shape_score = jaccard(dataA.get("shapes", []), dataB.get("shapes", []))
            pattern_score = jaccard(dataA.get("patterns", []), dataB.get("patterns", []))
            ocr_score = text_similarity(ocrA or "", ocrB or "")
            # weighted combination
            combined_score = round((0.4 * shape_score + 0.3 * pattern_score + 0.3 * ocr_score), 3)

            # energy overlays and merged resonance map
            overlayA = draw_energy_overlay(imgA, dataA)
            overlayB = draw_energy_overlay(imgB, dataB)

            # merge overlays visually: simple alpha blend after resizing to same dims
            h = max(overlayA.shape[0], overlayB.shape[0])
            w = max(overlayA.shape[1], overlayB.shape[1])
            # create canvases and place images centered
            canvasA = np.zeros((h, w, 3), dtype=np.uint8)
            canvasB = np.zeros((h, w, 3), dtype=np.uint8)
            # place A
            canvasA[:overlayA.shape[0], :overlayA.shape[1]] = overlayA
            canvasB[:overlayB.shape[0], :overlayB.shape[1]] = overlayB
            merged = cv2.addWeighted(canvasA, 0.5, canvasB, 0.5, 0)

            # display results
            st.success("✅ Compare complete")
            st.subheader("A: Detected Shapes & Translation")
            st.json({k:v for k,v in dataA.items() if k!="edges"})
            st.markdown(transA)

            st.subheader("B: Detected Shapes & Translation")
            st.json({k:v for k,v in dataB.items() if k!="edges"})
            st.markdown(transB)

            st.subheader("Similarity Scores")
            st.write(f"- Shape (Jaccard): **{shape_score:.3f}**")
            st.write(f"- Pattern (Jaccard): **{pattern_score:.3f}**")
            st.write(f"- OCR text similarity: **{ocr_score:.3f}**")
            st.write(f"- **Combined similarity score**: **{combined_score:.3f}**")

            st.subheader("Energy Overlays (A | B | Merged)")
            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(canvasA, cv2.COLOR_BGR2RGB), caption="Energy A", use_container_width=True)
            c2.image(cv2.cvtColor(canvasB, cv2.COLOR_BGR2RGB), caption="Energy B", use_container_width=True)
            c3.image(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB), caption="Merged Resonance Map", use_container_width=True)

            # OCR overlays
            st.subheader("OCR Overlays (A | B)")
            ocrovA = draw_ocr_boxes(imgA, ocr_dataA)
            ocrovB = draw_ocr_boxes(imgB, ocr_dataB)
            o1, o2 = st.columns(2)
            o1.image(cv2.cvtColor(ocrovA, cv2.COLOR_BGR2RGB), caption="OCR A", use_container_width=True)
            o2.image(cv2.cvtColor(ocrovB, cv2.COLOR_BGR2RGB), caption="OCR B", use_container_width=True)

            # log comparison
            cmp_entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "imgA": imgA_file.name,
                "imgB": imgB_file.name,
                "shapesA": ",".join(dataA.get("shapes", [])),
                "shapesB": ",".join(dataB.get("shapes", [])),
                "patternsA": ",".join(dataA.get("patterns", [])),
                "patternsB": ",".join(dataB.get("patterns", [])),
                "ocrA": ocrA[:200],
                "ocrB": ocrB[:200],
                "shape_jaccard": round(shape_score, 3),
                "pattern_jaccard": round(pattern_score, 3),
                "ocr_similarity": round(ocr_score, 3),
                "combined_score": combined_score,
                "notes": "Auto-compare (hybrid)"
            }
            log_compare_result(cmp_entry)

            st.success("🔎 Comparison saved to ivc_compare_log.csv")
            # provide download of the comparison row
            st.download_button("💾 Download comparison row (CSV)", data=pd.DataFrame([cmp_entry]).to_csv(index=False).encode("utf-8"),
                               file_name=f"ivc_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    else:
        st.info("Upload both Image A and Image B to run a hybrid compare.")

# Footer: quick access to logs if present
st.markdown("---")
if os.path.exists(COMPARE_LOG_FILE):
    if st.button("Open comparison log (ivc_compare_log.csv)"):
        try:
            df_cmp = pd.read_csv(COMPARE_LOG_FILE)
            st.dataframe(df_cmp, use_container_width=True)
        except Exception as e:
            st.error(f"Cannot open compare log: {e}")
