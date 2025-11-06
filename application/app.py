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
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher
import matplotlib.pyplot as plt

# -----------------------
# Config / files
# -----------------------
st.set_page_config(page_title="IVC Analyzer — ORB Feature Matching", layout="wide", initial_sidebar_state="collapsed")
st.title("🌀 IVC Analyzer — ORB Feature Matching Enabled")

SYMBOL_LIB_FILE = "ivc_symbol_library.json"
LOG_FILE = "ivc_symbol_log.csv"
COMPARE_LOG_FILE = "ivc_compare_log.csv"
LABELED_CSV = "ivc_labeled_dataset.csv"

# Default library
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
    SYMBOL_LIB: Dict[str, Dict[str, str]] = json.load(f)

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
# Image utilities
# -----------------------
def read_image(uploaded) -> np.ndarray:
    data = np.frombuffer(uploaded.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

# -----------------------
# Contour detection & heuristics
# -----------------------
def detect_contour_shapes(img: np.ndarray) -> Dict[str, Any]:
    gray = preprocess_gray(img)
    edges = cv2.Canny(gray, 100, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    contour_info = []
    h, w = img.shape[:2]
    min_area = max(60, (h * w) // 15000)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        sides = len(approx)
        x,y,ww,hh = cv2.boundingRect(approx)
        circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
        label = "unknown"
        if sides == 3:
            label = "triangle"
        elif sides == 4:
            ar = ww / (hh + 1e-6)
            label = "square" if 0.8 <= ar <= 1.25 else "rectangle"
        elif sides == 5:
            label = "pentagon"
        elif sides > 5:
            label = "circle" if circularity > 0.6 else "complex"
        minr, maxr = min(ww, hh), max(ww, hh)
        if minr > 0 and maxr / (minr + 1e-6) > 2.2 and sides >= 3:
            label = "arrow_like"
        contour_info.append({"label": label, "area": area, "bbox": (x,y,ww,hh), "contour": approx})
        shapes.append(label)

    seen = set()
    uniq_shapes = []
    for s in shapes:
        if s not in seen:
            uniq_shapes.append(s)
            seen.add(s)

    nested_detected = False
    squares = [ci for ci in contour_info if ci["label"] == "square"]
    if len(squares) >= 2:
        for a in squares:
            xa,ya,wa,ha = a["bbox"]
            for b in squares:
                if a is b: continue
                xb,yb,wb,hb = b["bbox"]
                if xa < xb and ya < yb and xa+wa > xb+wb and ya+ha > yb+hb:
                    nested_detected = True

    circle_like = [ci for ci in contour_info if ci["label"] == "circle"]
    spiral_like = len(circle_like) >= 3

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
    if spiral_like and any(ci["label"]=="arrow_like" for ci in contour_info):
        patterns.append("spiral_arrow")

    edge_density = np.count_nonzero(edges) / (h * w)
    frequencies = [round(6 + 40 * edge_density + np.random.uniform(-1,1), 2) for _ in range(3)]

    return {"shapes": uniq_shapes, "patterns": patterns, "frequencies": frequencies, "edges": edges, "contour_info": contour_info}

# -----------------------
# OCR
# -----------------------
def run_ocr(img: np.ndarray) -> Tuple[str, Any]:
    try:
        gray = preprocess_gray(img)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text", []) if w.strip()]
        text = " ".join(words).strip()
        return text, data
    except Exception as e:
        return f"[OCR error: {e}]", None

# -----------------------
# Energy visualizers
# -----------------------
def plot_energy_lines(edges: np.ndarray):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(edges, cmap="inferno")
    ax.set_title("Detected Symbolic Pathways (Energy Lines)")
    ax.axis("off")
    fig.tight_layout()
    return fig

def field_flow_overlay(img: np.ndarray, detected: Dict, mode: str = "Edge Flow") -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    gray = preprocess_gray(img)
    edges = cv2.Canny(gray, 100, 200)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = np.uint8((ang / 2) % 180)
    hsv[...,1] = 255
    hsv[...,2] = mag_norm
    flow_col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_col = cv2.GaussianBlur(flow_col, (5,5), 0)
    mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.bitwise_and(flow_col, flow_col, mask=mask)

    if mode == "Gradient Field" or mode == "Hybrid":
        y, x = np.mgrid[0:h, 0:w]
        freq = 0.02 * (len(detected.get("shapes", [])) + 1)
        flow = (np.sin(x * freq) + np.cos(y * freq * 0.7)) * 127 + 128
        flow = np.uint8(flow)
        plasma = cv2.applyColorMap(flow, cv2.COLORMAP_TWILIGHT)
        if mode == "Hybrid":
            combined = cv2.addWeighted(plasma, 0.5, lines, 0.9, 0)
        else:
            combined = cv2.addWeighted(plasma, 0.6, lines, 0.6, 0)
    else:
        combined = lines

    combined = cv2.GaussianBlur(combined, (5,5), 0)

    tint = np.zeros_like(combined)
    for s in detected.get("patterns", []):
        if s in ["nested_squares", "spiral_cluster", "lattice"]:
            c = (0,255,0) if s=="nested_squares" else (0,255,255) if s=="spiral_cluster" else (255,255,0)
            cv2.circle(tint, (w//2, h//2), int(min(w,h)*0.3), c, -1)

    overlay = cv2.addWeighted(img, 0.65, combined, 0.9, 0)
    final = cv2.addWeighted(overlay, 0.92, tint, 0.08, 0)
    final = cv2.bilateralFilter(final, 7, 75, 75)
    return final, edges

# -----------------------
# ORB feature matching (robust)
# -----------------------
def run_feature_matching(img: np.ndarray, templates: List[np.ndarray], threshold: float = 0.12):
    """
    ORB-based feature matching between img and template images.
    Returns list of matches: {template_idx, score (0..1), good_matches, total_template_kp, bbox_est (if homography)}
    Score = good_matches / max(1, total_template_kp)
    threshold is a score threshold e.g. 0.12 (low because ORB keypoints may be few)
    """
    # create ORB and matcher
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    gray = preprocess_gray(img)
    kp1, des1 = orb.detectAndCompute(gray, None)
    results = []

    for ti, templ in enumerate(templates):
        try:
            tgray = preprocess_gray(templ)
            kp2, des2 = orb.detectAndCompute(tgray, None)
            if des1 is None or des2 is None or len(kp2) == 0:
                results.append({"template_idx": ti, "score": 0.0, "good": 0, "template_kp": len(kp2) if kp2 is not None else 0})
                continue
            matches = bf.knnMatch(des2, des1, k=2)  # template->image
            # ratio test (Lowe)
            good = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            score = len(good) / float(max(1, len(kp2)))
            entry = {"template_idx": ti, "score": float(score), "good": len(good), "template_kp": len(kp2)}
            # try to estimate bbox via homography if enough matches
            if len(good) >= 4:
                src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h2, w2 = tgray.shape[:2]
                    pts = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, M)
                    # get integer bbox enclosing dst
                    xs = dst[:,0,0]; ys = dst[:,0,1]
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    entry["bbox_est"] = (x_min, y_min, x_max, y_max)
                else:
                    entry["bbox_est"] = None
            else:
                entry["bbox_est"] = None
            results.append(entry)
        except Exception as e:
            results.append({"template_idx": ti, "score": 0.0, "good": 0, "template_kp": 0})
    # filter by threshold and sort
    filtered = [r for r in results if r["score"] >= threshold]
    filtered_sorted = sorted(filtered, key=lambda x: -x["score"])
    return filtered_sorted, results

# -----------------------
# Symbol matching heuristics (library lookup)
# -----------------------
def match_symbols(detected: Dict, feature_matches: List[Dict], templates_names: List[str], ocr_text: str, lib: Dict) -> List[str]:
    matches = []
    det_shapes = set([s.lower() for s in detected.get("shapes", [])])
    det_patterns = set([p.lower() for p in detected.get("patterns", [])])
    if "lattice" in det_patterns and "lattice" in lib:
        matches.append("lattice")
    if "nested_squares" in det_patterns and "nested_squares" in lib:
        matches.append("nested_squares")
    if "spiral_cluster" in det_patterns and "spiral_arrow" in lib:
        matches.append("spiral_arrow")
    if "triangle" in det_shapes and "square" in det_shapes and "triangle_in_square" in lib:
        matches.append("triangle_in_square")

    # templates names mapping
    for fm in feature_matches:
        idx = fm.get("template_idx")
        if idx is None or idx >= len(templates_names):
            continue
        name = templates_names[idx].lower()
        # attempt to map filename substring to library key
        for k in lib.keys():
            if k.replace("_"," ") in name or k in name:
                if k not in matches:
                    matches.append(k)

    # OCR hints
    if isinstance(ocr_text, str):
        low = ocr_text.lower()
        for k in lib.keys():
            if k.replace("_"," ") in low and k not in matches:
                matches.append(k)

    if not matches:
        for s in det_shapes:
            for k in lib.keys():
                if s in k and k not in matches:
                    matches.append(k)

    # dedupe
    out = []
    for m in matches:
        if m not in out:
            out.append(m)
    return out

# -----------------------
# Manual labeling helpers
# -----------------------
def detect_contour_candidates(img: np.ndarray, min_area_ratio: float = 0.0005) -> Tuple[List[Dict], np.ndarray]:
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
        x,y,ww,hh = cv2.boundingRect(c)
        pad = int(0.05 * max(ww, hh))
        xa, ya = max(0, x - pad), max(0, y - pad)
        xb, yb = min(w, x + ww + pad), min(h, y + hh + pad)
        crop = img[ya:yb, xa:xb]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        sides = len(approx)
        hint = "unknown"
        if sides == 3:
            hint = "triangle"
        elif sides == 4:
            hint = "quad"
        elif sides > 6:
            circularity = (4 * np.pi * area) / (peri * peri) if peri > 0 else 0
            hint = "circle" if circularity > 0.6 else "complex"
        candidates.append({"bbox": (xa, ya, xb, yb), "area": area, "crop": crop, "hint": hint})
    candidates = sorted(candidates, key=lambda x: -x["area"])
    return candidates, edges

# -----------------------
# UI: Tabs
# -----------------------
tabs = st.tabs(["🔍 Analyze", "✍️ Manual Label / Annotate", "📘 Library Editor", "📂 Logs"])

# Sidebar controls
field_view = st.sidebar.selectbox("Field Visualization", ["Energy Lines (inferno)", "Field Flow", "Hybrid"])
template_threshold = st.sidebar.slider("Feature-match score threshold", 0.01, 0.5, 0.12, step=0.01)

# -----------------------
# TAB: Analyze
# -----------------------
with tabs[0]:
    st.header("Analyze — single artifact (ORB-based templates optional)")
    uploaded = st.file_uploader("Upload artifact image (jpg/png)", type=["jpg","jpeg","png"], key="analyze_upload")
    templates = st.file_uploader("Optional: upload template images (multiple)", type=["jpg","jpeg","png"], accept_multiple_files=True, key="templates_analyze")
    if uploaded is not None:
        img = read_image(uploaded)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Artifact", use_column_width=True)

        if st.button("▶️ Run Analysis", key="run_analysis"):
            ocr_text, ocr_data = run_ocr(img)
            detected = detect_contour_shapes(img)
            detected["ocr_text"] = ocr_text

            # load templates bytes into cv2 images & names list
            template_imgs = []
            template_names = []
            if templates:
                for t in templates:
                    try:
                        template_imgs.append(cv2.imdecode(np.frombuffer(t.read(), np.uint8), cv2.IMREAD_COLOR))
                        template_names.append(t.name)
                    except Exception:
                        continue

            # run ORB feature matching
            feature_matches, all_matches = run_feature_matching(img, template_imgs, threshold=template_threshold)

            # map features -> library matches
            matches = match_symbols(detected, feature_matches, template_names, ocr_text, SYMBOL_LIB)

            # show Energy Lines (inferno)
            st.subheader("1️⃣ Energy Lines (Canny / inferno)")
            fig = plot_energy_lines(detected["edges"])
            st.pyplot(fig)

            # show Field Flow overlay
            st.subheader("2️⃣ Field Flow Overlay")
            flow_img, edges = field_flow_overlay(img, detected, mode=("Gradient Field" if field_view=="Field Flow" else ("Hybrid" if field_view=="Hybrid" else "Edge Flow")))
            st.image(cv2.cvtColor(flow_img, cv2.COLOR_BGR2RGB), use_column_width=True)

            # OCR text + overlay
            st.subheader("3️⃣ OCR Extracted Text")
            st.code(ocr_text if ocr_text else "[No OCR text detected]")
            if ocr_data:
                vis = img.copy()
                n = len(ocr_data.get("text", []))
                for i in range(n):
                    try:
                        if float(ocr_data["conf"][i]) > 30:
                            x = int(ocr_data["left"][i]); y = int(ocr_data["top"][i]); w = int(ocr_data["width"][i]); h = int(ocr_data["height"][i])
                            cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
                    except Exception:
                        pass
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

            # detected features & matches
            st.subheader("4️⃣ Detected Visual Features")
            st.json({k:v for k,v in detected.items() if k in ("shapes","patterns","frequencies")})

            st.subheader("5️⃣ Template Feature Matches (ORB)")
            if feature_matches:
                for fm in feature_matches:
                    idx = fm["template_idx"]
                    name = template_names[idx] if idx < len(template_names) else f"template_{idx}"
                    st.write(f"- Template `{name}` — score: {fm['score']:.3f}, good matches: {fm['good']}")
                    if fm.get("bbox_est") is not None:
                        st.write(f"  Estimated bbox: {fm['bbox_est']}")
            else:
                st.info("No template feature matches above threshold.")

            st.subheader("6️⃣ IVC Translation / Library Matches")
            if matches:
                for m in matches:
                    info = SYMBOL_LIB.get(m, {})
                    st.markdown(f"**{m}** — {info.get('core','')}  \nDomain: {info.get('domain','')}  \nNotes: {info.get('notes','')}")
            else:
                st.info("No library matches found; consider manual labeling.")

            # logging
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "file": getattr(uploaded,"name","uploaded"),
                "shapes": ",".join(detected.get("shapes", [])),
                "patterns": ",".join(detected.get("patterns", [])),
                "freqs": ",".join([str(x) for x in detected.get("frequencies", [])]),
                "ocr_text": ocr_text.replace("\n"," ")[:500],
                "matches": ",".join(matches),
                "template_matches": ";".join([f"{template_names[fm['template_idx']]}:{fm['score']:.3f}" for fm in feature_matches]),
                "notes": ""
            }
            append_csv_row(LOG_FILE, row)
            st.success("Analysis complete — logged to ivc_symbol_log.csv")

# -----------------------
# TAB: Manual Label / Annotate
# -----------------------
with tabs[1]:
    st.header("Manual Labeling & Annotation (create training data)")
    st.markdown("Upload an image, inspect detected candidate crops, optionally upload templates to match, and assign a label from the library to create a labeled dataset.")
    up = st.file_uploader("Upload image for annotation", type=["jpg","jpeg","png"], key="label_upload")
    templates_lbl = st.file_uploader("Optional: template images to use for matching", type=["jpg","jpeg","png"], accept_multiple_files=True, key="templates_label")
    if up:
        imgL = read_image(up)
        candidates, edges = detect_contour_candidates(imgL)
        st.image(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB), caption="Source image", use_column_width=True)
        st.markdown(f"Detected {len(candidates)} candidate regions (filtered by size).")
        if not candidates:
            st.info("No candidates found. Try a different image or check image resolution.")
        else:
            # load templates for feature matching
            template_imgs = []
            template_names = []
            if templates_lbl:
                for t in templates_lbl:
                    try:
                        template_imgs.append(cv2.imdecode(np.frombuffer(t.read(), np.uint8), cv2.IMREAD_COLOR))
                        template_names.append(t.name)
                    except Exception:
                        pass
                # run feature matching to annotate candidate crops
                fm_all, fm_results = run_feature_matching(imgL, template_imgs, threshold=template_threshold)
            else:
                fm_all, fm_results = [], []

            cols = st.columns(3)
            assigned = []
            for i, cand in enumerate(candidates):
                crop = cand["crop"]
                bbox = cand["bbox"]
                col = cols[i % 3]
                with col:
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_column_width=True)
                    hint = cand.get("hint", "")
                    # try to find templates that overlap bbox_est (from fm_all)
                    suggested = []
                    for fm in fm_all:
                        be = fm.get("bbox_est")
                        if be:
                            x1,y1,x2,y2 = be
                            bx1,by1,bx2,by2 = bbox
                            # simple IoU-ish overlap test
                            inter_x1 = max(x1, bx1); inter_y1 = max(y1, by1)
                            inter_x2 = min(x2, bx2); inter_y2 = min(y2, by2)
                            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                                # matched template index
                                idx = fm["template_idx"]
                                if idx < len(template_names):
                                    suggested.append(template_names[idx])
                    options = ["(none)"] + list(SYMBOL_LIB.keys())
                    label = st.selectbox(f"Assign label for region #{i} (hint: {hint})", options=options, index=0, key=f"lbl_{i}")
                    notes = st.text_input(f"Notes #{i}", key=f"note_{i}")
                    assigned.append({"index": i, "bbox": bbox, "label": label if label != "(none)" else "", "notes": notes, "suggested": suggested})
                    if suggested:
                        st.caption("Suggested templates overlapping this region: " + ", ".join(suggested))

            if st.button("Save labeled rows"):
                rows = []
                for a in assigned:
                    if not a["label"]:
                        continue
                    x1,y1,x2,y2 = a["bbox"]
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

# -----------------------
# TAB: Library Editor
# -----------------------
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
    st.markdown("Preview:")
    st.json(SYMBOL_LIB)

# -----------------------
# TAB: Logs
# -----------------------
with tabs[3]:
    st.header("Logs & Datasets")
    if os.path.exists(LOG_FILE):
        st.markdown("### Analysis Log (ivc_symbol_log.csv)")
        df = pd.read_csv(LOG_FILE)
        q = st.text_input("Filter analysis log", key="filter_log")
        if q:
            mask = df.apply(lambda r: q.lower() in r.astype(str).str.lower().to_string(), axis=1)
            df = df[mask]
        st.dataframe(df)
        st.download_button("Download analysis log", data=df.to_csv(index=False).encode("utf-8"), file_name=LOG_FILE, mime="text/csv")
    else:
        st.info("No analysis log yet. Run analyses to populate ivc_symbol_log.csv")

    if os.path.exists(COMPARE_LOG_FILE):
        st.markdown("### Compare Log (ivc_compare_log.csv)")
        df2 = pd.read_csv(COMPARE_LOG_FILE)
        st.dataframe(df2)
        st.download_button("Download compare log", data=df2.to_csv(index=False).encode("utf-8"), file_name=COMPARE_LOG_FILE, mime="text/csv")
    else:
        st.info("No compare log yet.")

    if os.path.exists(LABELED_CSV):
        st.markdown("### Labeled Dataset (ivc_labeled_dataset.csv)")
        dfl = pd.read_csv(LABELED_CSV)
        st.dataframe(dfl)
        st.download_button("Download labeled dataset", data=dfl.to_csv(index=False).encode("utf-8"), file_name=LABELED_CSV, mime="text/csv")
    else:
        st.info("No labeled dataset yet. Use Manual Labeling tab to create one.")

# Footer
st.markdown("---")
st.markdown("Done — ORB-based feature matching is active. If you'd like, next I can provide a training script (transfer learning) that consumes ivc_labeled_dataset.csv and creates a classifier you can plug back into this app.")
