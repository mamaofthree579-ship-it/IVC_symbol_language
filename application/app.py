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
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

def detect_contour_candidates(img, min_area_ratio=0.0005):
    """Return list of candidate crops and detection metadata."""
    gray = preprocess_gray(img)
    edges = cv2.Canny(gray, 100, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h,w = img.shape[:2]
    min_area = max(80, int(h*w*min_area_ratio))
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,ww,hh = cv2.boundingRect(c)
        # expand a little
        pad = int(0.05 * max(ww,hh))
        xa,ya = max(0,x-pad), max(0,y-pad)
        xb,yb = min(w, x+ww+pad), min(h, y+hh+pad)
        crop = img[ya:yb, xa:xb]
        # approximate shape
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03*peri, True)
        sides = len(approx)
        label_hint = "unknown"
        if sides==3: label_hint="triangle"
        elif sides==4: label_hint="quad"
        elif sides>6:
            # circle-like vs complex
            circularity = (4*np.pi*area)/(peri*peri) if peri>0 else 0
            label_hint = "circle" if circularity>0.6 else "complex"
        candidates.append({"bbox":(xa,ya,xb,yb), "area":area, "crop":crop, "hint":label_hint})
    # sort by area desc (bigger first)
    candidates = sorted(candidates, key=lambda x: -x["area"])
    return candidates, edges

# -----------------------
# Template matching
# -----------------------
def run_template_matching(img, templates: List[np.ndarray], threshold=0.65):
    """Return list of matches: each is dict with template_idx, max_val, bbox"""
    gray = preprocess_gray(img)
    matches = []
    for i,templ in enumerate(templates):
        try:
            tgray = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(gray, tgray, cv2.TM_CCOEFF_NORMED)
            minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
            if maxv >= threshold:
                th, tw = tgray.shape[:2]
                x,y = maxloc
                matches.append({"template_idx":i, "score":float(maxv), "bbox":(x,y,x+tw,y+th)})
        except Exception as e:
            # mismatched template sizes or errors
            continue
    return matches

# -----------------------
# OCR
# -----------------------
def run_ocr(img):
    try:
        gray = preprocess_gray(img)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text",[]) if w.strip()]
        text = " ".join(words).strip()
        return text, data
    except Exception as e:
        return f"[OCR error: {e}]", None

# -----------------------
# Energy visuals (two modes)
# -----------------------
def plot_energy_lines(edges):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(edges, cmap="inferno")
    ax.axis("off")
    fig.tight_layout()
    return fig

def field_flow_overlay(img, detected, mode="Edge Flow"):
    # reuse earlier flow function: edges + gradient hue encoding masked on edges
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
    mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.bitwise_and(flow_col, flow_col, mask=mask)
    if mode=="Gradient Field" or mode=="Hybrid":
        y,x = np.mgrid[0:h,0:w]
        freq = 0.02 * (len(detected.get("shapes",[]))+1)
        flow = (np.sin(x*freq) + np.cos(y*freq*0.7))*127 + 128
        flow = np.uint8(flow)
        plasma = cv2.applyColorMap(flow, cv2.COLORMAP_TWILIGHT)
        if mode=="Hybrid":
            combined = cv2.addWeighted(plasma, 0.5, lines, 0.9, 0)
        else:
            combined = cv2.addWeighted(plasma, 0.6, lines, 0.6, 0)
    else:
        combined = lines
    # light tint by detected patterns
    tint = np.zeros_like(combined)
    for s in detected.get("patterns",[]):
        if s in ["nested_squares","spiral_cluster","lattice"]:
            c = (0,255,0) if s=="nested_squares" else (0,255,255) if s=="spiral_cluster" else (255,255,0)
            cv2.circle(tint, (w//2,h//2), int(min(w,h)*0.3), c, -1)
    overlay = cv2.addWeighted(img, 0.65, combined, 0.9, 0)
    final = cv2.addWeighted(overlay, 0.92, tint, 0.08, 0)
    final = cv2.bilateralFilter(final, 7, 75, 75)
    return final, edges

# -----------------------
# Symbol matching heuristics (combine detection + templates + OCR hints)
# -----------------------
def match_symbols(detected: Dict, templates_matched: List[Dict], ocr_text: str, lib: Dict) -> List[str]:
    matches = []
    det_shapes = set([s.lower() for s in detected.get("shapes",[])])
    det_patterns = set([p.lower() for p in detected.get("patterns",[])])
    # pattern to key mapping
    if "lattice" in det_patterns and "lattice" in lib: matches.append("lattice")
    if "nested_squares" in det_patterns and "nested_squares" in lib: matches.append("nested_squares")
    if "spiral_cluster" in det_patterns and "spiral_arrow" in lib: matches.append("spiral_arrow")
    # add template matches (template idx keys map to library keys uploaded by user)
    for tm in templates_matched:
        # tm may store 'label' if user provided; otherwise ignore
        if tm.get("label"):
            if tm["label"] in lib and tm["label"] not in matches:
                matches.append(tm["label"])
    # OCR hint mapping: check if any library key appears in text
    if isinstance(ocr_text,str):
        low = ocr_text.lower()
        for k in lib.keys():
            if k.replace("_"," ") in low and k not in matches:
                matches.append(k)
    # shape-based heuristics
    if "triangle" in det_shapes and "square" in det_shapes and "triangle_in_square" in lib and "triangle_in_square" not in matches:
        matches.append("triangle_in_square")
    # dedupe
    out=[]
    for m in matches:
        if m not in out:
            out.append(m)
    return out

# -----------------------
# Manual labeling UI helpers
# -----------------------
def save_labeled_rows(rows: List[Dict[str,Any]]):
    for r in rows:
        append_csv_row(LABELED_CSV, r)

# -----------------------
# UI Layout: tabs
# -----------------------
tabs = st.tabs(["🔍 Analyze", "🔬 Compare", "✍️ Manual Label / Annotate", "📘 Library Editor", "📂 Logs"])

# Sidebar
field_view = st.sidebar.selectbox("Field Visualization", ["Energy Lines (inferno)", "Field Flow", "Hybrid"])
template_threshold = st.sidebar.slider("Template match threshold", 0.5, 0.95, 0.72)

# =======================
# TAB: Analyze
# =======================
with tabs[0]:
    st.header("Analyze Artifact — detection + template matching + visuals")

    uploaded = st.file_uploader("Upload artifact image", type=["jpg","jpeg","png"], key="analyze_upload")
    templates = st.file_uploader("Optional: upload template images (zip not supported) — multiple allowed", type=["jpg","jpeg","png"], accept_multiple_files=True, key="templates")
    if uploaded:
        img = read_image(uploaded)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Artifact", use_column_width=True)

        if st.button("▶️ Run Analysis", key="run_analysis"):
            # detect candidates and edges
            candidates, edges = detect_contour_candidates(img)
            # detect patterns heuristics (nested squares, spiral clusters, lattice)
            # re-use detect_contour_shapes logic simplified: we treat candidates' hints
            shapes = [c["hint"] for c in candidates]
            # patterns heuristics
            patterns = []
            if len([s for s in shapes if s in ("circle","complex")])>=3:
                patterns.append("spiral_cluster")
            # run template matching
            template_imgs = []
            for t in templates:
                try:
                    template_imgs.append(cv2.imdecode(np.frombuffer(t.read(), np.uint8), cv2.IMREAD_COLOR))
                except Exception:
                    continue
            tmatches = run_template_matching(img, template_imgs, threshold=template_threshold)
            # map tmatches to labels if the user gave template filenames matching a library key
            for i,tm in enumerate(tmatches):
                # heuristic: if template filename contains a library key, attach it
                try:
                    templ_name = templates[tm["template_idx"]].name.lower()
                    # map first key that is substring
                    for key in SYMBOL_LIB.keys():
                        if key.replace("_"," ") in templ_name or key in templ_name:
                            tm["label"] = key
                            break
                except Exception:
                    pass

            # OCR
            ocr_text, ocr_data = run_ocr(img)
            # assemble detected dict
            detected = {"shapes": shapes, "patterns": patterns, "frequencies": [], "edges": edges, "contour_info": candidates, "ocr_text": ocr_text}
            # matching
            matches = match_symbols(detected, tmatches, ocr_text, SYMBOL_LIB)

            # show energy lines using inferno
            st.subheader("Energy Lines (inferno)")
            fig = plot_energy_lines(edges)
            st.pyplot(fig)

            # field flow overlay
            st.subheader("Field Flow / Hybrid")
            flow_img, _ = field_flow_overlay(img, detected, mode=("Hybrid" if field_view=="Hybrid" else ("Gradient Field" if field_view=="Field Flow" else "Edge Flow")))
            st.image(cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            # OCR text & overlay
            st.subheader("OCR Text")
            st.code(ocr_text if ocr_text else "[no OCR detected]")

            if ocr_data:
                vis = img.copy()
                n = len(ocr_data.get("text",[]))
                for i in range(n):
                    try:
                        if float(ocr_data["conf"][i])>30:
                            x=int(ocr_data["left"][i]); y=int(ocr_data["top"][i]); w=int(ocr_data["width"][i]); h=int(ocr_data["height"][i])
                            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
                    except Exception:
                        pass
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

            # show matches & translation info
            st.subheader("Detected Visual Features & Matches")
            st.write("Shapes heuristics (candidate hints):", shapes)
            st.write("Patterns heuristics:", patterns)
            st.write("Template matches:", [{**tm} for tm in tmatches])
            st.write("Library matches:", matches)
            # show translation details from library
            st.subheader("IVC Translation (library lookup)")
            if matches:
                for m in matches:
                    info = SYMBOL_LIB.get(m, {})
                    st.markdown(f"**{m}** — {info.get('core','')}  \nDomain: {info.get('domain','')}  \nNotes: {info.get('notes','')}")
            else:
                st.info("No library match found — consider manual labeling.")

            # log
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "file": getattr(uploaded, "name", "uploaded"),
                "shapes": ",".join(shapes),
                "patterns": ",".join(patterns),
                "ocr_text": ocr_text.replace("\n"," ")[:400],
                "matches": ",".join(matches)
            }
            append_csv_row(LOG_FILE, row)
            st.success("Analysis complete — logged to ivc_symbol_log.csv")

# =======================
# TAB: Compare
# =======================
with tabs[1]:
    st.header("Compare Mode — side-by-side + logs")
    colA, colB = st.columns(2)
    fileA = colA.file_uploader("Image A", type=["jpg","jpeg","png"], key="cmpA")
    fileB = colB.file_uploader("Image B", type=["jpg","jpeg","png"], key="cmpB")
    if fileA and fileB:
        imgA = read_image(fileA); imgB = read_image(fileB)
        st.image([cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB), cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)], caption=["A","B"], width=300)
        if st.button("▶️ Run Compare", key="run_cmp"):
            detA, edgesA = detect_contour_candidates(imgA)
            detB, edgesB = detect_contour_candidates(imgB)
            # simple shape lists
            shapesA = [c["hint"] for c in detA]
            shapesB = [c["hint"] for c in detB]
            ocrA, _ = run_ocr(imgA)
            ocrB, _ = run_ocr(imgB)
            matchesA = match_symbols({"shapes":shapesA,"patterns":[],"ocr_text":ocrA}, [], ocrA, SYMBOL_LIB)
            matchesB = match_symbols({"shapes":shapesB,"patterns":[],"ocr_text":ocrB}, [], ocrB, SYMBOL_LIB)
            # similarity
            def jaccard(a,b):
                A=set([x.lower() for x in a]); B=set([x.lower() for x in b])
                if not A and not B: return 1.0
                if not A or not B: return 0.0
                return len(A&B)/len(A|B)
            shape_score = jaccard(shapesA, shapesB)
            ocr_sim = SequenceMatcher(None, ocrA or "", ocrB or "").ratio()
            combined = round(0.5*shape_score + 0.5*ocr_sim,3)
            # show energy lines (both)
            st.subheader("Energy Lines (A | B)")
            figA = plot_energy_lines(edgesA); figB = plot_energy_lines(edgesB)
            c1,c2 = st.columns(2); c1.pyplot(figA); c2.pyplot(figB)
            st.write(f"Shape similarity: {shape_score:.3f}  | OCR similarity: {ocr_sim:.3f}  | Combined: {combined:.3f}")
            # log compare
            cmp_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "imgA": getattr(fileA,"name","A"),
                "imgB": getattr(fileB,"name","B"),
                "shapesA": ",".join(shapesA),
                "shapesB": ",".join(shapesB),
                "ocrA": ocrA[:200],
                "ocrB": ocrB[:200],
                "shape_jaccard": shape_score,
                "ocr_similarity": ocr_sim,
                "combined_score": combined
            }
            append_csv_row(COMPARE_LOG_FILE, cmp_row)
            st.success("Compare logged to ivc_compare_log.csv")

# =======================
# TAB: Manual Label / Annotate
# =======================
with tabs[2]:
    st.header("Manual Labeling & Annotation (create training data)")
    st.markdown("Upload an image, inspect detected candidate crops, and assign a label from the library to create a labeled dataset.")
    up = st.file_uploader("Upload image for annotation", type=["jpg","jpeg","png"], key="label_upload")
    if up:
        imgL = read_image(up)
        candidates, edges = detect_contour_candidates(imgL)
        st.image(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB), caption="Source image", use_column_width=True)
        st.markdown(f"Detected {len(candidates)} candidate regions (filtered by size).")
        if not candidates:
            st.info("No candidates found. Try lowering min_area_ratio in code or use a different image.")
        else:
            # show crops in a grid with dropdowns
            cols = st.columns(3)
    assigned = []
    for i, cand in enumerate(candidates):
         crop = cand["crop"]
         bbox = cand["bbox"]
         col = cols[i % 3]
        with col:
        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_column_width=True)
        hint = cand.get("hint", "")
        label = st.selectbox(f"Assign label for region #{i} (hint: {hint})",
                             options=["(none)"] + list(SYMBOL_LIB.keys()), index=0, key=f"lbl_{i}")
        notes = st.text_input(f"notes #{i}", key=f"note_{i}")
        assigned.append({
            "index": i,
            "bbox": bbox,
            "label": label if label != "(none)" else "",
            "notes": notes
        })
                    rows.append(row)
                if rows:
                    save_labeled_rows = True
                    save_labeled_rows = False
                    # actually append rows
                    for r in rows:
                        append_csv_row(LABELED_CSV, r)
                    st.success(f"Saved {len(rows)} labeled rows to {LABELED_CSV}")
                else:
                    st.info("No labels selected to save.")

# =======================
# TAB: Library Editor / Logs
# =======================
with tabs[3]:
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
    st.markdown("Preview (read-only):")
    st.json(SYMBOL_LIB)

with tabs[4]:
    st.header("Logs & Datasets")
    if os.path.exists(LOG_FILE):
        st.markdown("### Analysis Log")
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download analysis log", data=df.to_csv(index=False).encode("utf-8"), file_name=LOG_FILE)
    else:
        st.info("No analysis log yet.")
    if os.path.exists(COMPARE_LOG_FILE):
        st.markdown("### Compare Log")
        df2 = pd.read_csv(COMPARE_LOG_FILE)
        st.dataframe(df2, use_container_width=True)
        st.download_button("Download compare log", data=df2.to_csv(index=False).encode("utf-8"), file_name=COMPARE_LOG_FILE)
    else:
        st.info("No compare log yet.")
    if os.path.exists(LABELED_CSV):
        st.markdown("### Labeled Dataset")
        dfl = pd.read_csv(LABELED_CSV)
        st.dataframe(dfl, use_container_width=True)
        st.download_button("Download labeled dataset", data=dfl.to_csv(index=False).encode("utf-8"), file_name=LABELED_CSV)
    else:
        st.info("No labeled dataset yet. Use Manual Labeling tab to create one.")
