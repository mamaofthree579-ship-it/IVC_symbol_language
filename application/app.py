# app.py
import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import os
import csv
from datetime import datetime
from typing import Dict, List, Optional

# --------------------
# Config
# --------------------
st.set_page_config(page_title="IVC Analyzer", layout="wide", initial_sidebar_state="collapsed")
st.title("🌀 IVC Analyzer — Image-driven Symbol Detection & Energy Mapping")

# Log files
LOG_FILE = "ivc_symbol_log.csv"
COMPARE_LOG_FILE = "ivc_compare_log.csv"

# --------------------
# Try to import ivc_translator (if present). Otherwise fallback.
# --------------------
try:
    from ivc_translator import ivc_translate as imported_ivc_translate, LOG_FILE as TRANSLATOR_LOG_FILE
    def ivc_translate(symbol_data: Dict, ocr_text: str = "") -> str:
        try:
            return imported_ivc_translate(symbol_data, ocr_text)
        except Exception as e:
            return f"[Translator error: {e}]"
except Exception:
    # fallback simple translator if module not present
    def ivc_translate(symbol_data: Dict, ocr_text: str = "") -> str:
        shapes = symbol_data.get("shapes", [])
        if not shapes:
            return "No recognized visual symbol shapes detected for translation."
        return f"IVC translation (rule-based stub): detected shapes = {', '.join(shapes)}."

# --------------------
# Utilities: IO / logging
# --------------------
def append_log_row(logfile: str, row: Dict):
    """Append a row (dict) to a CSV, creating header if needed."""
    header = list(row.keys())
    new_file = not os.path.exists(logfile)
    try:
        with open(logfile, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        st.error(f"Failed to write log {logfile}: {e}")

# --------------------
# Image helpers
# --------------------
def read_uploaded_image(uploaded_file) -> Optional[np.ndarray]:
    """Return BGR OpenCV image or None."""
    if uploaded_file is None:
        return None
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    return gray

# --------------------
# Shape extraction: image-driven (contour-based + heuristics)
# --------------------
def analyze_symbols_from_image(img: np.ndarray) -> Dict:
    """
    Analyze image to detect symbol-like shapes and simple patterns.
    Returns dict: {shapes: [...], patterns: [...], frequencies: [...], edges: edge_map}
    """
    gray = preprocess_gray(img)
    # Edge map
    edges = cv2.Canny(gray, 80, 180)

    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    # heuristics thresholds
    min_area = max(100, (img.shape[0] * img.shape[1]) // 10000)  # scale with image
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        sides = len(approx)
        if sides == 3:
            shapes.append("triangle")
        elif sides == 4:
            # check aspect ratio to separate squares vs rectangles
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h) if h != 0 else 0
            shapes.append("square" if 0.8 <= ar <= 1.25 else "rectangle")
        elif sides == 5:
            shapes.append("pentagon")
        elif sides > 5:
            # compute circularity
            circularity = (4 * np.pi * area) / (peri * peri) if peri != 0 else 0
            if circularity > 0.6:
                shapes.append("circle")
            else:
                shapes.append("complex")
        else:
            shapes.append("unknown")

    # dedupe preserve order
    seen = set()
    uniq_shapes = []
    for s in shapes:
        if s not in seen:
            uniq_shapes.append(s)
            seen.add(s)

    # patterns heuristics
    patterns = []
    if len(uniq_shapes) >= 3:
        patterns.append("lattice")
    if "circle" in uniq_shapes and "triangle" in uniq_shapes:
        patterns.append("spiral-arrow")  # heuristic
    # frequency placeholder: estimate from edge density
    edge_density = np.count_nonzero(edges) / (img.shape[0] * img.shape[1])
    # map density to a few pseudo-frequencies (arbitrary units)
    frequencies = [round(5 + 50 * edge_density + np.random.uniform(-2, 2), 2) for _ in range(3)]

    return {"shapes": uniq_shapes, "patterns": patterns, "frequencies": frequencies, "edges": edges}

# --------------------
# OCR helper
# --------------------
def run_ocr(img: np.ndarray) -> (str, Optional[dict]):
    """Run Tesseract OCR; returns text and tesseract data dict (or None on error)."""
    try:
        gray = preprocess_gray(img)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text", []) if w.strip()]
        text = " ".join(words).strip()
        return text, data
    except Exception as e:
        return f"[OCR error: {e}]", None

# --------------------
# Energy overlay: edge-driven flow lines + symbol tint
# --------------------
def draw_energy_overlay(img: np.ndarray, symbol_data: Dict, mode: str = "Edge Flow") -> np.ndarray:
    """
    Create an edge/gradient driven energy overlay:
    - uses Canny edges + Sobel gradients
    - colour-codes by gradient direction and intensity
    - masks by edges to draw fine energy lines that follow the image
    - adds a light tint based on detected shapes
    """
    # prepare
    out = img.copy()
    h, w = img.shape[:2]
    gray = preprocess_gray(img)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # direction -> hue, intensity -> value
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = np.uint8((angle / 2) % 180)
    hsv[..., 1] = 255
    hsv[..., 2] = mag_norm
    flow_coloured = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_coloured = cv2.GaussianBlur(flow_coloured, (5, 5), 0)

    # mask to edges to get "lines" only
    mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.bitwise_and(flow_coloured, flow_coloured, mask=mask)

    # gradient/plasma background if requested
    if mode == "Gradient Field":
        y_indices, x_indices = np.mgrid[0:h, 0:w]
        freq = 0.02 * (len(symbol_data.get("shapes", [])) + 1)
        flow = (np.sin(x_indices * freq) + np.cos(y_indices * freq * 0.7)) * 127 + 128
        flow = np.uint8(flow)
        plasma = cv2.applyColorMap(flow, cv2.COLORMAP_TWILIGHT)
        # combine lines over plasma
        combined = cv2.addWeighted(plasma, 0.6, lines, 0.9, 0)
    elif mode == "Hybrid":
        y_indices, x_indices = np.mgrid[0:h, 0:w]
        freq = 0.02 * (len(symbol_data.get("shapes", [])) + 1)
        flow = (np.sin(x_indices * freq) + np.cos(y_indices * freq * 0.7)) * 127 + 128
        flow = np.uint8(flow)
        plasma = cv2.applyColorMap(flow, cv2.COLORMAP_TWILIGHT)
        combined = cv2.addWeighted(plasma, 0.5, lines, 0.9, 0)
    else:
        combined = lines

    # smooth lines for glow effect
    combined = cv2.GaussianBlur(combined, (5, 5), 0)

    # tint layer based on shapes (very light)
    tint = np.zeros_like(img)
    color_map = {
        "spiral": (0, 255, 255),
        "triangle": (0, 128, 255),
        "square": (0, 255, 0),
        "circle": (255, 0, 0),
        "arrow": (255, 0, 255),
        "lattice": (255, 255, 0),
        "rectangle": (0, 180, 180),
        "complex": (200, 100, 50),
        "pentagon": (120, 200, 80),
        "unknown": (150,150,150)
    }
    for s in symbol_data.get("shapes", []):
        c = color_map.get(s.lower(), (180, 180, 180))
        # Add small area of color in center or across image (light)
        cv2.circle(tint, (w // 2, h // 2), int(min(w, h) * 0.35), c, -1)

    # blend: original + combined lines + tint
    field_overlay = cv2.addWeighted(img, 0.65, combined, 0.9, 0)
    final = cv2.addWeighted(field_overlay, 0.92, tint, 0.08, 0)
    # slight bilateral filter to preserve edges but smooth
    final = cv2.bilateralFilter(final, 7, 75, 75)

    return final

# --------------------
# OCR overlay draw (boxes)
# --------------------
def draw_ocr_overlay(img: np.ndarray, ocr_data) -> np.ndarray:
    out = img.copy()
    if ocr_data is None:
        return out
    n = len(ocr_data.get("text", []))
    for i in range(n):
        conf = ocr_data.get("conf", [None]*n)[i]
        try:
            c = float(conf)
        except Exception:
            c = -1.0
        if c > 30:
            x = int(ocr_data["left"][i])
            y = int(ocr_data["top"][i])
            w = int(ocr_data["width"][i])
            h = int(ocr_data["height"][i])
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out

# --------------------
# Similarity helpers
# --------------------
def jaccard(a: List[str], b: List[str]) -> float:
    sa = set([x.lower() for x in a])
    sb = set([x.lower() for x in b])
    if not sa and not sb:
        return 1.0
    union = sa.union(sb)
    inter = sa.intersection(sb)
    return len(inter) / len(union) if union else 0.0

# --------------------
# UI: Tabs
# --------------------
tabs = st.tabs(["🔍 Analyze", "📘 Symbol Log Viewer", "🔬 Compare Mode"])

# Sidebar: choose energy field style
field_mode = st.sidebar.selectbox("Field Visualization Mode", ["Edge Flow", "Gradient Field", "Hybrid"])

# ----- TAB 1: Analyze -----
with tabs[0]:
    st.header("Analyze — single artifact")
    uploaded = st.file_uploader("Upload artifact image (jpg/png)", type=["jpg", "jpeg", "png"], key="analyze_upload")
    if uploaded is not None:
        img = read_uploaded_image(uploaded)
        if img is None:
            st.error("Could not read image.")
        else:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Artifact", use_column_width=True)

            if st.button("▶️ Run Analysis", key="run_analysis"):
                # OCR
                ocr_text, ocr_data = run_ocr(img)
                # image-driven symbol detection
                symbol_data = analyze_symbols_from_image(img)
                # translation (uses imported translator if available)
                translation = ivc_translate(symbol_data, ocr_text)

                # draw overlays
                energy_map = draw_energy_overlay(img, symbol_data, mode=field_mode)
                ocr_overlay = draw_ocr_overlay(img, ocr_data)

                # display
                st.subheader("Energy Map (image-driven)")
                st.image(cv2.cvtColor(energy_map, cv2.COLOR_BGR2RGB), use_column_width=True)

                st.subheader("OCR (detected text)")
                st.code(ocr_text if ocr_text else "[No OCR text]")

                st.subheader("OCR Visual (boxes)")
                st.image(cv2.cvtColor(ocr_overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

                st.subheader("Detected Symbol Data")
                st.json({k:v for k,v in symbol_data.items() if k != "edges"})

                st.subheader("IVC Translation")
                st.markdown(translation)

                # log the run to CSV
                log_row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "file": uploaded.name,
                    "shapes": ",".join(symbol_data.get("shapes", [])),
                    "patterns": ",".join(symbol_data.get("patterns", [])),
                    "freqs": ",".join([str(f) for f in symbol_data.get("frequencies", [])]),
                    "ocr_text": ocr_text.replace("\n", " ")[:500],
                    "translation": translation
                }
                append_log_row(LOG_FILE, log_row)
                st.success("Logged analysis to ivc_symbol_log.csv")

# ----- TAB 2: Log Viewer -----
with tabs[1]:
    st.header("Symbol Log Viewer")
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            q = st.text_input("🔍 Filter log (matches any column)", key="log_search")
            if q:
                mask = df.apply(lambda r: q.lower() in r.astype(str).str.lower().to_string(), axis=1)
                df = df[mask]
            st.dataframe(df, use_container_width=True)
            st.download_button("💾 Download symbol log (CSV)", data=df.to_csv(index=False).encode("utf-8"), file_name="ivc_symbol_log.csv", mime="text/csv")
            st.caption(f"Total entries: {len(df)}")
        except Exception as e:
            st.error(f"Could not read log file: {e}")
    else:
        st.info("No symbol log found yet. Run analyses to populate ivc_symbol_log.csv")

# ----- TAB 3: Compare Mode -----
with tabs[2]:
    st.header("Compare Mode — hybrid visual + similarity + energy overlay")
    c1, c2 = st.columns(2)
    imgA_file = c1.file_uploader("Upload Image A", type=["jpg","jpeg","png"], key="cmpA")
    imgB_file = c2.file_uploader("Upload Image B", type=["jpg","jpeg","png"], key="cmpB")

    if imgA_file is not None and imgB_file is not None:
        imgA = read_uploaded_image(imgA_file)
        imgB = read_uploaded_image(imgB_file)
        st.markdown("### Previews")
        p1, p2 = st.columns(2)
        p1.image(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB), caption="Image A", use_column_width=True)
        p2.image(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB), caption="Image B", use_column_width=True)

        if st.button("▶️ Run Compare", key="run_compare"):
            # process both
            ocrA, ocr_dataA = run_ocr(imgA)
            ocrB, ocr_dataB = run_ocr(imgB)
            symA = analyze_symbols_from_image(imgA)
            symB = analyze_symbols_from_image(imgB)
            transA = ivc_translate(symA, ocrA)
            transB = ivc_translate(symB, ocrB)

            # overlays
            mapA = draw_energy_overlay(imgA, symA, mode=field_mode)
            mapB = draw_energy_overlay(imgB, symB, mode=field_mode)
            ocrA_overlay = draw_ocr_overlay(imgA, ocr_dataA)
            ocrB_overlay = draw_ocr_overlay(imgB, ocr_dataB)

            # similarity metrics
            shape_score = jaccard(symA.get("shapes", []), symB.get("shapes", []))
            pattern_score = jaccard(symA.get("patterns", []), symB.get("patterns", []))
            # simple text similarity (ratio of matching chars)
            ocr_sim = 0.0
            try:
                from difflib import SequenceMatcher
                ocr_sim = SequenceMatcher(None, ocrA or "", ocrB or "").ratio()
            except Exception:
                ocr_sim = 0.0
            combined = round(0.45 * shape_score + 0.25 * pattern_score + 0.3 * ocr_sim, 3)

            # merged visual (alpha blend canvases)
            h = max(mapA.shape[0], mapB.shape[0])
            w = max(mapA.shape[1], mapB.shape[1])
            canvasA = np.zeros((h, w, 3), dtype=np.uint8)
            canvasB = np.zeros((h, w, 3), dtype=np.uint8)
            canvasA[:mapA.shape[0], :mapA.shape[1]] = mapA
            canvasB[:mapB.shape[0], :mapB.shape[1]] = mapB
            merged = cv2.addWeighted(canvasA, 0.5, canvasB, 0.5, 0)

            st.subheader("Translations")
            st.markdown("**A:** " + transA)
            st.markdown("**B:** " + transB)

            st.subheader("Similarity Scores")
            st.write(f"- Shape Jaccard: {shape_score:.3f}")
            st.write(f"- Pattern Jaccard: {pattern_score:.3f}")
            st.write(f"- OCR similarity: {ocr_sim:.3f}")
            st.write(f"- **Combined**: {combined:.3f}")

            st.subheader("Energy Overlays")
            colA, colB, colM = st.columns(3)
            colA.image(cv2.cvtColor(mapA, cv2.COLOR_BGR2RGB), caption="Energy A", use_column_width=True)
            colB.image(cv2.cvtColor(mapB, cv2.COLOR_BGR2RGB), caption="Energy B", use_column_width=True)
            colM.image(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB), caption="Merged Resonance Map", use_column_width=True)

            st.subheader("OCR Overlays")
            o1, o2 = st.columns(2)
            o1.image(cv2.cvtColor(ocrA_overlay, cv2.COLOR_BGR2RGB), caption="OCR A", use_column_width=True)
            o2.image(cv2.cvtColor(ocrB_overlay, cv2.COLOR_BGR2RGB), caption="OCR B", use_column_width=True)

            # log comparison
            cmp_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "imgA": getattr(imgA_file, "name", "imgA"),
                "imgB": getattr(imgB_file, "name", "imgB"),
                "shapesA": ",".join(symA.get("shapes", [])),
                "shapesB": ",".join(symB.get("shapes", [])),
                "patternsA": ",".join(symA.get("patterns", [])),
                "patternsB": ",".join(symB.get("patterns", [])),
                "ocrA": (ocrA or "")[:400],
                "ocrB": (ocrB or "")[:400],
                "shape_jaccard": shape_score,
                "pattern_jaccard": pattern_score,
                "ocr_similarity": ocr_sim,
                "combined_score": combined,
                "notes": "auto-compare"
            }
            append_log_row(COMPARE_LOG_FILE, cmp_row)
            st.success("Comparison saved to ivc_compare_log.csv")
            st.download_button("💾 Download comparison row CSV", data=pd.DataFrame([cmp_row]).to_csv(index=False).encode("utf-8"),
                               file_name=f"ivc_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# Footer quick links to logs if present
st.markdown("---")
col_l, col_r = st.columns(2)
with col_l:
    if os.path.exists(LOG_FILE):
        if st.button("Open symbol log"):
            st.write(pd.read_csv(LOG_FILE))
with col_r:
    if os.path.exists(COMPARE_LOG_FILE):
        if st.button("Open compare log"):
            st.write(pd.read_csv(COMPARE_LOG_FILE))
