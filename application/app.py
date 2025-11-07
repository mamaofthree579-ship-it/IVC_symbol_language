# app.py
"""
IVC Symbol Recognizer — Enhanced with CFG Fallback
Includes:
- Symbol recognition against ivc_symbol_library.json
- Energy / field overlay visualization
- CFG induction with fallback heuristic when rules are weak
- Library editor and log tracking
"""

import os
import json
import io
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ivc_cfg dependency for CFG induction
try:
    from ivc_cfg import sequitur_infer, rules_to_text, rules_to_graph_edges, export_rules_json
    IVCCFG_AVAILABLE = True
except Exception:
    IVCCFG_AVAILABLE = False

# ---------------------------
# Config / paths
# ---------------------------
BASE_DIR = os.getcwd()
APP_DIR = os.path.join(BASE_DIR, "application")
os.makedirs(APP_DIR, exist_ok=True)

SYMBOL_LIB_FILE = os.path.join(APP_DIR, "ivc_symbol_library.json")
SYMBOLS_DIR = os.path.join(APP_DIR, "symbols")
os.makedirs(SYMBOLS_DIR, exist_ok=True)

LOG_FILE = os.path.join(APP_DIR, "ivc_symbol_log.csv")
LABELED_CSV = os.path.join(APP_DIR, "ivc_labeled_dataset.csv")
BACKUP_DIR = os.path.join(APP_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

st.set_page_config(page_title="IVC Symbol Recognizer", layout="wide")
st.title("IVC Symbol Recognizer — Reference Matching & CFG Fallback")

# ---------------------------
# Utilities
# ---------------------------
def backup_file(path: str):
    try:
        if os.path.exists(path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            import shutil
            shutil.copy(path, os.path.join(BACKUP_DIR, f"{ts}_{os.path.basename(path)}"))
    except Exception as e:
        st.warning(f"Backup failed for {path}: {e}")

def safe_load_json(path: str) -> Any:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Failed to read JSON {path}: {e}")
        return {}

def save_json(path: str, obj: Any):
    backup_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_csv_row(filepath: str, row: Dict[str, Any]):
    import csv
    new_file = not os.path.exists(filepath)
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        st.error(f"Failed to write {filepath}: {e}")
    backup_file(filepath)

def safe_load_csv(path: str) -> pd.DataFrame:
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
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"safe_load_csv: error reading {path}: {e}")
        return pd.DataFrame()

# ---------------------------
# Descriptor utilities
# ---------------------------
def hu_moments_descriptor(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thr)
    hu = cv2.HuMoments(M).flatten()
    for i in range(len(hu)):
        if hu[i] == 0:
            hu[i] = 0.0
        else:
            hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30)
    return np.nan_to_num(hu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def contour_signature_descriptor(contour: np.ndarray, length: int = 64) -> np.ndarray:
    if contour is None or len(contour) < 6:
        return np.zeros(length//2, dtype=np.float32)
    pts = contour.reshape(-1,2).astype(np.float32)
    diffs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cum = np.concatenate([[0], np.cumsum(diffs)])
    if cum[-1] == 0:
        res = np.tile(pts[0], (length,1))
    else:
        samp = np.linspace(0, cum[-1], length)
        res = np.zeros((length,2), dtype=np.float32)
        idx = 0
        for i,s in enumerate(samp):
            while idx < len(cum)-1 and cum[idx+1] < s:
                idx += 1
            if idx == len(cum)-1:
                res[i] = pts[-1]
            else:
                t = (s - cum[idx]) / (cum[idx+1] - cum[idx] + 1e-12)
                res[i] = pts[idx] * (1-t) + pts[idx+1] * t
    vecs = res[1:] - res[:-1]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vn = vecs / norms
    dots = (vn[:-1] * vn[1:]).sum(axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    bins = np.array_split(angles, length//2)
    feats = np.array([b.mean() if len(b)>0 else 0.0 for b in bins], dtype=np.float32)
    return feats

def descriptor_from_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim==3 else img_bgr
    hu = hu_moments_descriptor(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        sig = np.zeros(32, dtype=np.float32)
    else:
        main = max(cnts, key=cv2.contourArea)
        sig = contour_signature_descriptor(main, length=64)
    desc = np.concatenate([hu, sig])
    norm = np.linalg.norm(desc) + 1e-9
    return (desc / norm).astype(np.float32)

# ---------------------------
# Load / build reference library
# ---------------------------
def load_symbol_library(lib_path: str, symbols_dir: str) -> Dict[str, Dict[str, Any]]:
    lib = safe_load_json(lib_path)
    if not isinstance(lib, dict):
        lib = {}
    descriptors = {}
    for key, info in lib.items():
        ipath = info.get("image_path") or os.path.join(symbols_dir, f"{key}.png")
        if os.path.exists(ipath):
            try:
                img = cv2.imread(ipath)
                desc = descriptor_from_image(img)
                descriptors[key] = {"info": info, "desc": desc, "image_path": ipath}
            except Exception as e:
                st.warning(f"Descriptor build failed for {key}: {e}")
        else:
            descriptors[key] = {"info": info, "desc": None, "image_path": ipath}
    return descriptors

if not os.path.exists(SYMBOL_LIB_FILE):
    save_json(SYMBOL_LIB_FILE, {})

SYMBOL_LIBRARY = load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR)

# ---------------------------
# Matching helpers
# ---------------------------
def match_descriptor(desc: np.ndarray, library: Dict[str, Dict[str, Any]], top_k: int = 3) -> List[Tuple[str, float]]:
    entries = []
    for key, v in library.items():
        d = v.get("desc")
        if d is None:
            continue
        if np.linalg.norm(desc)==0 or np.linalg.norm(d)==0:
            dist = float("inf")
        else:
            cos = np.dot(desc, d) / (np.linalg.norm(desc) * np.linalg.norm(d) + 1e-12)
            cos = max(min(cos, 1.0), -1.0)
            dist = 1.0 - cos
        entries.append((key, float(dist)))
    entries.sort(key=lambda x: x[1])
    return entries[:top_k]

# ---------------------------
# Fallback CFG / heuristic
# ---------------------------
def build_fallback_grammar(symbol_data: List[str]):
    """Build adjacency graph if CFG inference fails"""
    G = nx.DiGraph()
    for i, sym in enumerate(symbol_data[:-1]):
        G.add_edge(sym, symbol_data[i + 1])
    return G

# ---------------------------
# Energy visualizer
# ---------------------------
def field_flow_overlay(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag_n = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h,w = gray.shape[:2]
    hsv = np.zeros((h,w,3), dtype=np.uint8)
    hsv[...,0] = np.uint8((ang/2) % 180)
    hsv[...,1] = 255
    hsv[...,2] = mag_n
    flow_col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(gray, 80, 200)
    mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.bitwise_and(flow_col, flow_col, mask=mask)
    overlay = cv2.addWeighted(img_bgr, 0.65, lines, 0.9, 0)
    overlay = cv2.bilateralFilter(overlay, 7, 75, 75)
    return overlay, edges

def plot_energy_inferno(edges: np.ndarray):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(edges, cmap="inferno")
    ax.axis("off")
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.sidebar.header("Recognition Controls")
match_threshold = st.sidebar.slider("Match distance threshold", 0.01, 0.6, 0.18, 0.01)
top_k = st.sidebar.number_input("Top-K matches shown", 1,5,3)
show_unrecognized_panel = st.sidebar.checkbox("Enable new reference creation panel", True)

tabs = st.tabs(["Upload & Recognize", "Library Editor", "CFG Induction", "Logs"])

# ------------------ TAB: Upload & Recognize ------------------
with tabs[0]:
    st.header("Upload script images & run recognition")
    uploaded = st.file_uploader("Upload images", type=["jpg","png","tif"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            st.subheader(f"File: {up.name}")
            file_bytes = up.read()
            arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image")
                continue
            overlay, edges = field_flow_overlay(img)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.pyplot(plot_energy_inferno(edges))

            # detect contours
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray,(3,3),0)
            cnts, _ = cv2.findContours(cv2.Canny(blur,60,180), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = []
            h,w = img.shape[:2]
            for c in cnts:
                x,y,ww,hh = cv2.boundingRect(c)
                if ww*hh < max(100,(h*w)//3000):
                    continue
                bboxes.append((x,y,ww,hh,c))
            bboxes = sorted(bboxes, key=lambda z:(z[1], z[0]))
            st.write(f"Detected {len(bboxes)} candidate regions")
            if not bboxes:
                st.info("No candidate contours found")
                continue

            # match to library
            cols = st.columns(4)
            detected_sequence = []
            fallback_mode = False
            for i,(x,y,ww,hh,c) in enumerate(bboxes):
                crop = img[y:y+hh, x:x+ww].copy()
                pad = max(ww,hh)
                canvas = np.ones((pad,pad,3), dtype=np.uint8)*255
                canvas[(pad-hh)//2:(pad-hh)//2+hh,(pad-ww)//2:(pad-ww)//2+ww] = crop
                desc = descriptor_from_image(canvas)
                matches = match_descriptor(desc, SYMBOL_LIBRARY, top_k)
                label = ""
                if matches:
                    best_key, best_dist = matches[0]
                    if best_dist <= match_threshold:
                        label = best_key
                with cols[i%4]:
                    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)
                    if label:
                        detected_sequence.append(label)
                        st.success(f"{label}")
                    else:
                        st.warning("Unrecognized")
                        if show_unrecognized_panel:
                            new_name = st.text_input(f"Name region #{i}", key=f"newname_{up.name}_{i}")
                            if st.button(f"Save region #{i}", key=f"save_new_{up.name}_{i}"):
                                if not new_name:
                                    st.error("Enter name")
                                else:
                                    sid = new_name.lower().replace(" ","_")
                                    dest_path = os.path.join(SYMBOLS_DIR,f"{sid}.png")
                                    cv2.imwrite(dest_path, canvas)
                                    lib = safe_load_json(SYMBOL_LIB_FILE)
                                    lib[sid] = {"display_name": new_name,"image_path": dest_path}
                                    save_json(SYMBOL_LIB_FILE, lib)
                                    SYMBOL_LIBRARY.update(load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR))
                                    st.success(f"Saved new symbol {sid}")

            st.subheader("Detected sequence")
            if detected_sequence:
                st.write(detected_sequence)
            else:
                placeholders = [f"S{i+1}" for i in range(len(bboxes))]
                st.write(placeholders)

            # --- CFG induction with fallback ---
            if IVCCFG_AVAILABLE and detected_sequence:
                res = sequitur_infer([detected_sequence], min_rule_occurrence=2)
                if not res["rules"]:
                    fallback_mode = True
                    st.warning("⚠️ No strong CFG rules found — using heuristic adjacency graph")
                    grammar_model = build_fallback_grammar(detected_sequence)
                else:
                    grammar_model = res
            else:
                fallback_mode = True
                grammar_model = build_fallback_grammar(detected_sequence)
                st.info("CFG not available or sequence empty — using fallback adjacency graph")

            # log
            log_row = {
                "timestamp": datetime.now().isoformat(),
                "file": up.name,
                "num_regions": len(bboxes),
                "recognized_sequence": ",".join(detected_sequence)[:500],
                "fallback_used": fallback_mode
            }
            append_csv_row(LOG_FILE, log_row)
