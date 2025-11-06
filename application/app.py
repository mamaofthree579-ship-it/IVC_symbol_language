# app.py
import streamlit as st
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
import json
import os
import csv
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from difflib import SequenceMatcher

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="IVC Analyzer (Vector Library)", layout="wide", initial_sidebar_state="collapsed")
st.title("🌀 IVC Analyzer — Vector Symbol Library + Energy Lines")

# Files
SYMBOL_LIB_FILE = "ivc_symbol_library.json"
LOG_FILE = "ivc_symbol_log.csv"
COMPARE_LOG_FILE = "ivc_compare_log.csv"

# -----------------------
# Default symbol library (created if JSON missing)
# -----------------------
DEFAULT_LIBRARY = {
    "fish": {
        "core": "Particle / Consciousness unit / Self",
        "domain": "Quantum-Material Interface",
        "notes": "Quantum “seed” or living particle; recurring in water symbolism."
    },
    "jar": {
        "core": "Containment / Material manifestation / Body",
        "domain": "Gravitational Field / Matter",
        "notes": "Often found beside fish — likely denoting consciousness within matter."
    },
    "fish+jar": {
        "core": "Embodied consciousness / Life form",
        "domain": "Bio-Quantum Structure",
        "notes": "Encoded as “spirit within form.”"
    },
    "double_wavy_line": {
        "core": "Vibration / Oscillation / Energy Flow",
        "domain": "Waveform / Frequency Dynamics",
        "notes": "Linked to resonance; parallels with water-energy glyphs."
    },
    "arrow": {
        "core": "Linear movement / Directionality",
        "domain": "Temporal Flow / Vector Dynamics",
        "notes": "Motion, progression, causal shift."
    },
    "arrow_dot": {
        "core": "Singularity compression / Time-point",
        "domain": "Gravity Node / Quantum Collapse",
        "notes": "Focused energy or gravitational compression."
    },
    "triangle_in_square": {
        "core": "Coupled quantum tunneling / Dimensional harmony",
        "domain": "Quantum Interface",
        "notes": "Dual-stability structure."
    },
    "cross_spirals": {
        "core": "Nodes of action / Black–White hole interface",
        "domain": "Scalar-Vortex Dynamics",
        "notes": "Movement between energy domains; dual polarity."
    },
    "spiral_arrow": {
        "core": "Energy flow initiation / Field rotation",
        "domain": "Rotational Vortex / Scalar Initiation",
        "notes": "Activation or turning on resonance."
    },
    "nested_squares": {
        "core": "Dimensional layering / Boundary harmonics",
        "domain": "Gravitational or Dimensional Compression",
        "notes": "Energy containment or shielding."
    },
    "lattice": {
        "core": "Stabilization / Field grid / Linkage",
        "domain": "Resonant Grid / Planetary Network",
        "notes": "Large-scale energy distribution mapping."
    },
    "step": {
        "core": "Frequency ascension / Harmonic hierarchy",
        "domain": "Consciousness / Dimensional Ascent",
        "notes": "Step function of frequency scaling."
    },
    "dual_circles": {
        "core": "Dual polarity / Energy coupling",
        "domain": "Electromagnetic Resonance",
        "notes": "Balancing positive and negative."
    },
    "dot_circle": {
        "core": "Focus point / Center of consciousness",
        "domain": "Scalar Resonance Center",
        "notes": "Point zero from which form emanates."
    },
    "knot_loop": {
        "core": "Entanglement / Resonance link / Memory bridge",
        "domain": "Quantum Coupling",
        "notes": "Information feedback or linked state."
    },
    "lined_grid": {
        "core": "Dimensional matrix / Space-time fabric",
        "domain": "Field Geometry",
        "notes": "Space partitioning, lattice of manifestation."
    },
    "circle_cross": {
        "core": "Unification of four forces / Space-time anchor",
        "domain": "Scalar Equilibrium Point",
        "notes": "Universal equilibrium cosmogram."
    },
    "spiral_wavy": {
        "core": "Coupled resonance / Light-wave harmonization",
        "domain": "Photon-Gravity Interaction",
        "notes": "Resonance-based levitation or cutting."
    },
    "diamond": {
        "core": "Crystalline formation / Structured energy",
        "domain": "Piezoelectric or Crystal Field",
        "notes": "Crystalline consciousness or lattice memory."
    },
    "step_pyramid": {
        "core": "Ascension structure / Field amplification",
        "domain": "Frequency Amplification",
        "notes": "Physical and energetic hierarchies."
    },
    "grid_central_node": {
        "core": "Network hub / Conscious focal point",
        "domain": "Consciousness-Matter Interface",
        "notes": "Distributed energy network controlled by intent."
    }
}

# If library file missing, create it
if not os.path.exists(SYMBOL_LIB_FILE):
    with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_LIBRARY, f, indent=2)

# Load library
with open(SYMBOL_LIB_FILE, "r", encoding="utf-8") as f:
    SYMBOL_LIB = json.load(f)

# -----------------------
# Helpers: IO / logging
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
        st.error(f"Failed to write log {filepath}: {e}")

# -----------------------
# Image utilities
# -----------------------
def read_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

# -----------------------
# Core detectors (heuristics)
# -----------------------
def detect_contour_shapes(img: np.ndarray) -> Dict[str, Any]:
    """
    Return shapes, patterns, edges.
    shapes: list of names (triangle, square, circle, pentagon, nested_square, complex, arrow-like, spiral-like, lattice-like)
    patterns: heuristics like lattice, nested_squares, spiral_arrow
    edges: binary edge map
    """
    gray = preprocess_gray(img)
    edges = cv2.Canny(gray, 100, 200)

    # contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    contour_info = []
    min_area = max(60, (img.shape[0]*img.shape[1])//15000)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        sides = len(approx)
        bbox = cv2.boundingRect(approx)
        x,y,w,h = bbox
        circularity = (4*np.pi*area)/(peri*peri) if peri>0 else 0
        # heuristics
        label = "unknown"
        if sides == 3:
            label = "triangle"
        elif sides == 4:
            ar = w/(h+1e-6)
            if 0.8 <= ar <= 1.25:
                label = "square"
            else:
                label = "rectangle"
        elif sides == 5:
            label = "pentagon"
        elif sides > 5:
            label = "circle" if circularity > 0.6 else "complex"
        # detect arrow-like by convexity defects and elongated shape
        box = cv2.minAreaRect(c)
        bw = min(box[1])  # small side
        bh = max(box[1])  # large side
        if bw>0 and bh/bw > 2.2 and sides>=3:
            # long thin contour might be an arrow shaft
            label = "arrow_like"
        # record
        contour_info.append({"label": label, "area": area, "bbox": bbox, "contour": approx})
        shapes.append(label)

    # dedupe keep order
    seen = set()
    unique_shapes = []
    for s in shapes:
        if s not in seen:
            unique_shapes.append(s)
            seen.add(s)

    # Try nested squares detection: check multiple square-like contours inside each other
    nested_detected = False
    squares = [ci for ci in contour_info if ci["label"]=="square"]
    if len(squares) >= 2:
        # check containment by bbox overlap
        for a in squares:
            for b in squares:
                if a==b: continue
                xa,ya,wa,ha = a["bbox"]
                xb,yb,wb,hb = b["bbox"]
                if xa<xb and ya<yb and xa+wa>xb+wb and ya+ha>yb+hb:
                    nested_detected = True
    # Spiral-like detection: many concentric contours (circles) or strong circularity clusters
    circle_like = [ci for ci in contour_info if ci["label"]=="circle"]
    spiral_like = False
    if len(circle_like) >= 3:
        spiral_like = True

    # lattice detection: many near-parallel lines via HoughLine detection
    lattice_like = False
    try:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)
        if lines is not None and len(lines) > 8:
            lattice_like = True
    except Exception:
        lattice_like = False

    patterns = []
    if nested_detected:
        patterns.append("nested_squares")
    if spiral_like:
        patterns.append("spiral_cluster")
    if lattice_like:
        patterns.append("lattice")

    # composite heuristics, spiral-arrow = spiral + arrow_like
    if spiral_like and any(ci["label"]=="arrow_like" for ci in contour_info):
        patterns.append("spiral_arrow")

    # estimate pseudo-frequencies (edge density mapping)
    edge_density = np.count_nonzero(edges) / (img.shape[0]*img.shape[1])
    frequencies = [round(6 + 40*edge_density + np.random.uniform(-1,1),2) for _ in range(3)]

    return {"shapes": unique_shapes, "patterns": patterns, "frequencies": frequencies, "edges": edges, "contour_info": contour_info}

# -----------------------
# OCR helpers
# -----------------------
def run_ocr(img: np.ndarray):
    try:
        gray = preprocess_gray(img)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text", []) if w.strip()]
        text = " ".join(words).strip()
        return text, data
    except Exception as e:
        return f"[OCR error: {e}]", None

# -----------------------
# Symbol matching to library (simple heuristics)
# -----------------------
def match_symbols(detected: Dict, lib: Dict) -> List[str]:
    """
    Return list of matched symbol keys from SYMBOL_LIB based on heuristics.
    We use pattern name matching (pattern->library key) and shape composition.
    """
    matches = []
    det_shapes = set([s.lower() for s in detected.get("shapes", [])])
    det_patterns = set([p.lower() for p in detected.get("patterns", [])])

    # direct pattern -> library key
    if "lattice" in det_patterns and "lattice" in lib:
        matches.append("lattice")
    if "nested_squares" in det_patterns and "nested_squares" in lib:
        matches.append("nested_squares")
    if "spiral_cluster" in det_patterns and "spiral_arrow" in lib:
        # spiral cluster alone could match spiral variants
        matches.append("spiral_arrow")

    # shapes composition heuristics
    # spiral-arrow: spiral-like (spiral_cluster) + arrow_like in contour labels
    if "spiral_cluster" in det_patterns and any("arrow_like" in ci.get("label","") for ci in detected.get("contour_info", [])):
        if "spiral_arrow" in lib and "spiral_arrow" not in matches:
            matches.append("spiral_arrow")

    # fish/jar etc cannot be reliably detected by contour heuristics; use OCR hints
    text_hint = ""
    if isinstance(detected.get("ocr_text",""), str):
        text_hint = detected.get("ocr_text","").lower()
    # map text hints
    for key in lib.keys():
        if key.replace("_"," ") in text_hint:
            matches.append(key)

    # some heuristic mapping by shapes
    if "square" in det_shapes and "circle" in det_shapes and "nested_squares" in det_patterns:
        if "nested_squares" in lib and "nested_squares" not in matches:
            matches.append("nested_squares")

    if "triangle" in det_shapes and "square" in det_shapes:
        if "triangle_in_square" in lib and "triangle_in_square" not in matches:
            matches.append("triangle_in_square")

    # lattice if many lines
    if "lattice" in det_patterns and "lattice" in lib and "lattice" not in matches:
        matches.append("lattice")

    # fallback: match by similar names if nothing else
    if not matches:
        # attempt fuzzy name matching between detected shapes lists and library keys
        for s in det_shapes:
            for k in lib.keys():
                # check if shape substring occurs in key
                if s in k and k not in matches:
                    matches.append(k)
    # dedupe
    out = []
    for m in matches:
        if m not in out:
            out.append(m)
    return out

# -----------------------
# Energy visualizers
# -----------------------
def plot_energy_lines(edges):
    """Return matplotlib figure plotting edges with inferno colormap (pure line aesthetic)."""
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(edges, cmap="inferno")
    ax.set_title("Detected Symbolic Pathways (Energy Lines)")
    ax.axis("off")
    fig.tight_layout()
    return fig

def field_flow_overlay(img, detected, mode="Edge Flow"):
    """Return an OpenCV BGR image with a colorful flow overlay derived from gradients and edges."""
    h,w = img.shape[:2]
    gray = preprocess_gray(img)
    edges = cv2.Canny(gray, 100, 200)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[...,0] = np.uint8((ang/2)%180)
    hsv[...,1] = 255
    hsv[...,2] = mag_norm
    flow_col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_col = cv2.GaussianBlur(flow_col, (5,5), 0)
    # mask with edges
    mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.bitwise_and(flow_col, flow_col, mask=mask)
    # optionally add a soft plasma background
    if mode in ("Gradient Field","Hybrid"):
        y,x = np.mgrid[0:h,0:w]
        freq = 0.02 * (len(detected.get("shapes",[]))+1)
        flow = (np.sin(x*freq) + np.cos(y*freq*0.7))*127+128
        flow = np.uint8(flow)
        plasma = cv2.applyColorMap(flow, cv2.COLORMAP_TWILIGHT)
        if mode == "Hybrid":
            background = cv2.addWeighted(plasma, 0.5, flow_col, 0.5, 0)
        else:
            background = plasma
        combined = cv2.addWeighted(background, 0.6, lines, 0.9, 0)
    else:
        combined = lines
    # tint lightly by detected shapes (centered radial fill)
    tint = np.zeros_like(combined)
    color_map = {
        "spiral_arrow": (0,255,255),
        "nested_squares": (0,255,0),
        "lattice": (255,255,0),
        "triangle_in_square": (0,128,255),
        "arrow": (255,0,0)
    }
    for s in detected.get("shapes",[]):
        c = color_map.get(s, None)
        if c:
            cv2.circle(tint, (w//2,h//2), int(min(w,h)*0.35), c, -1)
    # blend
    overlay = cv2.addWeighted(img, 0.65, combined, 0.9, 0)
    final = cv2.addWeighted(overlay, 0.92, tint, 0.08, 0)
    final = cv2.bilateralFilter(final, 7, 75, 75)
    return final

# -----------------------
# UI: Tabs and flows
# -----------------------
tabs = st.tabs(["🔍 Analyze", "🔬 Compare Mode", "📘 Symbol Log Viewer"])

# Sidebar: choose visualization mode
field_mode = st.sidebar.selectbox("Field View", ["Energy Lines (inferno)", "Field Flow", "Hybrid"])

# ---------- Analyze Tab ----------
with tabs[0]:
    st.header("Analyze artifact")
    uploaded = st.file_uploader("Upload artifact image (jpg/png)", type=["jpg","jpeg","png"], key="analyze")
    if uploaded is not None:
        img = read_image(uploaded)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        if st.button("▶️ Run Analysis", key="run1"):
            # OCR
            ocr_text, ocr_data = run_ocr(img)
            # detect
            detected = detect_contour_shapes(img)
            # attach OCR text onto detected dict for matching
            detected["ocr_text"] = ocr_text
            # match to library
            matches = match_symbols(detected, SYMBOL_LIB)
            # form translation details
            translation_lines = []
            if matches:
                for m in matches:
                    info = SYMBOL_LIB.get(m, {})
                    translation_lines.append(f"**{m}** — {info.get('core','')}; Domain: {info.get('domain','')}. Notes: {info.get('notes','')}")
            else:
                translation_lines.append("No direct vector-symbol match found (logged for review).")

            # display Energy Lines (inferno)
            st.subheader("Energy Lines (Canny / inferno)")
            fig = plot_energy_lines(detected["edges"])
            st.pyplot(fig)

            # Field Flow overlay
            st.subheader("Field Flow Overlay")
            if field_mode == "Energy Lines (inferno)":
                # show a colored edges map as BGR for continuity
                flow_img = field_flow_overlay(img, detected, mode="Edge Flow")
            elif field_mode == "Field Flow":
                flow_img = field_flow_overlay(img, detected, mode="Gradient Field")
            else:
                flow_img = field_flow_overlay(img, detected, mode="Hybrid")
            st.image(cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            # OCR & OCR overlay
            st.subheader("OCR Extracted Text")
            st.code(ocr_text if ocr_text else "[No OCR text detected]")
            if ocr_data:
                ocr_vis = img.copy()
                n = len(ocr_data.get("text",[]))
                for i in range(n):
                    conf = ocr_data.get("conf",[None]*n)[i]
                    try:
                        if float(conf) > 30:
                            x = int(ocr_data["left"][i]); y=int(ocr_data["top"][i]); w=int(ocr_data["width"][i]); h=int(ocr_data["height"][i])
                            cv2.rectangle(ocr_vis, (x,y), (x+w,y+h), (0,255,0), 2)
                    except Exception:
                        pass
                st.image(cv2.cvtColor(ocr_vis, cv2.COLOR_BGR2RGB), use_column_width=True)

            # show detected shapes/patterns
            st.subheader("Detected Visual Features")
            st.json({k:v for k,v in detected.items() if k in ("shapes","patterns","frequencies")})

            st.subheader("IVC Translation / Matches")
            for line in translation_lines:
                st.markdown(line)

            # logging
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "file": getattr(uploaded, "name", "uploaded"),
                "shapes": ",".join(detected.get("shapes",[])),
                "patterns": ",".join(detected.get("patterns",[])),
                "freqs": ",".join([str(x) for x in detected.get("frequencies",[])]),
                "ocr_text": ocr_text.replace("\n"," ")[:500],
                "matches": ",".join(matches) or "",
                "notes": ""
            }
            append_csv_row(LOG_FILE, row)
            st.success("Analysis complete — logged to ivc_symbol_log.csv")

# ---------- Compare Tab ----------
with tabs[1]:
    st.header("Hybrid Compare Mode")
    colA, colB = st.columns(2)
    with colA:
        fileA = st.file_uploader("Upload Image A", type=["jpg","jpeg","png"], key="cmpA")
    with colB:
        fileB = st.file_uploader("Upload Image B", type=["jpg","jpeg","png"], key="cmpB")
    if fileA and fileB:
        imgA = read_image(fileA)
        imgB = read_image(fileB)
        st.markdown("### Previews")
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB), use_column_width=True, caption="A")
        c2.image(cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB), use_column_width=True, caption="B")
        if st.button("▶️ Run Compare", key="compare_run"):
            ocrA, ocrA_data = run_ocr(imgA)
            ocrB, ocrB_data = run_ocr(imgB)
            detA = detect_contour_shapes(imgA); detA["ocr_text"]=ocrA
            detB = detect_contour_shapes(imgB); detB["ocr_text"]=ocrB
            matchesA = match_symbols(detA, SYMBOL_LIB)
            matchesB = match_symbols(detB, SYMBOL_LIB)
            transA = [f"{m}: {SYMBOL_LIB.get(m,{}).get('core','')}" for m in matchesA] or ["No matches"]
            transB = [f"{m}: {SYMBOL_LIB.get(m,{}).get('core','')}" for m in matchesB] or ["No matches"]

            # similarity metrics
            def jaccard(a,b):
                A=set([x.lower() for x in a]); B=set([x.lower() for x in b])
                if not A and not B: return 1.0
                if not A or not B: return 0.0
                return len(A&B)/len(A|B)
            shape_score = jaccard(detA.get("shapes",[]), detB.get("shapes",[]))
            pattern_score = jaccard(detA.get("patterns",[]), detB.get("patterns",[]))
            ocr_score = SequenceMatcher(None, ocrA or "", ocrB or "").ratio()

            combined = round(0.45*shape_score + 0.25*pattern_score + 0.3*ocr_score, 3)

            # overlays
            figA = plot_energy_lines(detA["edges"])
            figB = plot_energy_lines(detB["edges"])
            st.subheader("Energy Lines (A | B)")
            cA, cB = st.columns(2)
            cA.pyplot(figA); cB.pyplot(figB)

            st.subheader("Field Flow Overlays (A | B | Merged)")
            flowA = field_flow_overlay(imgA, detA, mode=("Hybrid" if field_mode=="Hybrid" else ("Gradient Field" if field_mode=="Field Flow" else "Edge Flow")))
            flowB = field_flow_overlay(imgB, detB, mode=("Hybrid" if field_mode=="Hybrid" else ("Gradient Field" if field_mode=="Field Flow" else "Edge Flow")))
            # merged
            H = max(flowA.shape[0], flowB.shape[0]); W = max(flowA.shape[1], flowB.shape[1])
            canvasA = np.zeros((H,W,3), dtype=np.uint8); canvasB = np.zeros((H,W,3), dtype=np.uint8)
            canvasA[:flowA.shape[0],:flowA.shape[1]] = flowA
            canvasB[:flowB.shape[0],:flowB.shape[1]] = flowB
            merged = cv2.addWeighted(canvasA, 0.5, canvasB, 0.5, 0)
            m1,m2,m3 = st.columns(3)
            m1.image(cv2.cvtColor(flowA, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Flow A")
            m2.image(cv2.cvtColor(flowB, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Flow B")
            m3.image(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Merged")

            # show similarity
            st.subheader("Similarity")
            st.write(f"- Shape Jaccard: {shape_score:.3f}")
            st.write(f"- Pattern Jaccard: {pattern_score:.3f}")
            st.write(f"- OCR similarity: {ocr_score:.3f}")
            st.write(f"- Combined score: {combined:.3f}")

            # translations
            st.subheader("Matches / Translations (A | B)")
            st.markdown("**A:**"); st.write(transA)
            st.markdown("**B:**"); st.write(transB)

            # log compare
            cmp_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "imgA": getattr(fileA,"name","A"),
                "imgB": getattr(fileB,"name","B"),
                "shapesA": ",".join(detA.get("shapes",[])),
                "shapesB": ",".join(detB.get("shapes",[])),
                "patternsA": ",".join(detA.get("patterns",[])),
                "patternsB": ",".join(detB.get("patterns",[])),
                "ocrA": ocrA.replace("\n"," ")[:400],
                "ocrB": ocrB.replace("\n"," ")[:400],
                "shape_jaccard": shape_score,
                "pattern_jaccard": pattern_score,
                "ocr_similarity": ocr_score,
                "combined_score": combined,
                "matchesA": ",".join(matchesA),
                "matchesB": ",".join(matchesB)
            }
            append_csv_row(COMPARE_LOG_FILE, cmp_row)
            st.success("Comparison logged to ivc_compare_log.csv")

# ---------- Log Viewer ----------
with tabs[2]:
    st.header("Symbol Log Viewer")
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            q = st.text_input("Filter log", key="filter_log")
            if q:
                mask = df.apply(lambda r: q.lower() in r.astype(str).str.lower().to_string(), axis=1)
                df = df[mask]
            st.dataframe(df, use_container_width=True)
            st.download_button("Download symbol log", data=df.to_csv(index=False).encode("utf-8"), file_name=LOG_FILE, mime="text/csv")
        except Exception as e:
            st.error(f"Could not load log: {e}")
    else:
        st.info("No analysis log yet. Run analyses to create ivc_symbol_log.csv")

# Footer: quick link to edit symbol library
st.markdown("---")
st.markdown("**Symbol library file:** `ivc_symbol_library.json` — edit this file to refine meanings or add new keys. The app loads it at startup.")
if st.button("Open symbol library (preview)"):
    st.json(SYMBOL_LIB)
