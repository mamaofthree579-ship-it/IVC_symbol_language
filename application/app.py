# app.py
"""
IVC Symbol Recognizer — Full Version with Robust JSON & Multi-Tab Support
Features:
- Symbol recognition with fallback CFG adjacency graph
- Energy / field overlay visualization
- Library editor and logging
- Multi-tab support
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import cv2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

# ---------------------------
# Paths & Config
# ---------------------------
BASE_DIR = os.getcwd()
APP_DIR = os.path.join(BASE_DIR, "application")
os.makedirs(APP_DIR, exist_ok=True)

SYMBOL_LIB_FILE = os.path.join(APP_DIR, "ivc_symbol_library.json")
SYMBOLS_DIR = os.path.join(APP_DIR, "symbols")
os.makedirs(SYMBOLS_DIR, exist_ok=True)

LOG_FILE = os.path.join(APP_DIR, "ivc_symbol_log.csv")
BACKUP_DIR = os.path.join(APP_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

st.set_page_config(page_title="IVC Symbol Recognizer", layout="wide")
st.title("IVC Symbol Recognizer — Multi-Tab Version")

# ---------------------------
# JSON Utilities
# ---------------------------
def backup_file(path: str):
    if os.path.exists(path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        import shutil
        shutil.copy(path, os.path.join(BACKUP_DIR, f"{ts}_{os.path.basename(path)}"))

def safe_load_json(path: str) -> dict:
    """Load JSON safely; returns empty dict if invalid."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.warning(f"JSON decode error in {path}, returning empty dict.")
        return {}
    except Exception as e:
        st.warning(f"Failed to read JSON {path}: {e}")
        return {}

def save_json(path: str, obj: Any):
    backup_file(path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save JSON {path}: {e}")

# Ensure library exists
if not os.path.exists(SYMBOL_LIB_FILE):
    save_json(SYMBOL_LIB_FILE, {})

# ---------------------------
# Symbol Descriptor Utilities
# ---------------------------
def hu_moments_descriptor(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thr)
    hu = cv2.HuMoments(M).flatten()
    hu = [-np.sign(h)*np.log10(abs(h)+1e-30) if h!=0 else 0.0 for h in hu]
    return np.nan_to_num(np.array(hu, dtype=np.float32))

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
                res[i] = pts[idx]*(1-t) + pts[idx+1]*t
    vecs = res[1:] - res[:-1]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)+1e-9
    vn = vecs/norms
    dots = (vn[:-1]*vn[1:]).sum(axis=1)
    dots = np.clip(dots,-1.0,1.0)
    angles = np.arccos(dots)
    bins = np.array_split(angles, length//2)
    feats = np.array([b.mean() if len(b)>0 else 0.0 for b in bins], dtype=np.float32)
    return feats

def descriptor_from_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim==3 else img_bgr
    hu = hu_moments_descriptor(gray)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur,50,200)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        sig = np.zeros(32, dtype=np.float32)
    else:
        main = max(cnts,key=cv2.contourArea)
        sig = contour_signature_descriptor(main,64)
    desc = np.concatenate([hu,sig])
    norm = np.linalg.norm(desc)+1e-9
    return (desc/norm).astype(np.float32)

# ---------------------------
# Load Symbol Library
# ---------------------------
def load_symbol_library(lib_path: str, symbols_dir: str) -> dict:
    lib = safe_load_json(lib_path)
    descriptors = {}
    for key, info in lib.items():
        ipath = info.get("image_path") or os.path.join(symbols_dir, f"{key}.png")
        if os.path.exists(ipath):
            try:
                img = cv2.imread(ipath)
                desc = descriptor_from_image(img)
                descriptors[key] = {"info":info,"desc":desc,"image_path":ipath}
            except Exception as e:
                st.warning(f"Descriptor build failed for {key}: {e}")
        else:
            descriptors[key] = {"info":info,"desc":None,"image_path":ipath}
    return descriptors

SYMBOL_LIBRARY = load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR)

# ---------------------------
# Matching Helper
# ---------------------------
def match_descriptor(desc: np.ndarray, library: dict, top_k:int=3) -> List[Tuple[str,float]]:
    entries = []
    for key, v in library.items():
        d = v.get("desc")
        if d is None:
            continue
        dist = 1.0 - np.clip(np.dot(desc,d)/(np.linalg.norm(desc)*np.linalg.norm(d)+1e-12),-1.0,1.0)
        entries.append((key,float(dist)))
    entries.sort(key=lambda x:x[1])
    return entries[:top_k]

# ---------------------------
# CFG fallback
# ---------------------------
def build_fallback_grammar(symbol_data: List[str]):
    """Adjacency graph fallback if CFG fails"""
    G = nx.DiGraph()
    for i,sym in enumerate(symbol_data[:-1]):
        G.add_edge(sym, symbol_data[i+1])
    return G

# ---------------------------
# Energy overlay
# ---------------------------
def field_flow_overlay(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F,1,0,ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F,0,1,ksize=5)
    mag, ang = cv2.cartToPolar(gx,gy,angleInDegrees=True)
    mag_n = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    h,w = gray.shape[:2]
    hsv = np.zeros((h,w,3),dtype=np.uint8)
    hsv[...,0] = np.uint8((ang/2)%180)
    hsv[...,1] = 255
    hsv[...,2] = mag_n
    flow_col = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    edges = cv2.Canny(gray,80,200)
    mask = cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=1)
    lines = cv2.bitwise_and(flow_col,flow_col,mask=mask)
    overlay = cv2.addWeighted(img_bgr,0.65,lines,0.9,0)
    overlay = cv2.bilateralFilter(overlay,7,75,75)
    return overlay, edges

def plot_energy_inferno(edges: np.ndarray):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(edges,cmap="inferno")
    ax.axis("off")
    return fig

# ---------------------------
# Tabs UI
# ---------------------------
tabs = st.tabs(["Upload & Recognize","Library Editor","Logs"])

# ---------- TAB 1 ----------
with tabs[0]:
    st.header("Upload images for recognition")
    uploaded = st.file_uploader("Upload images", type=["png","jpg","tif"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            st.subheader(f"File: {up.name}")
            arr = np.frombuffer(up.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image")
                continue
            overlay, edges = field_flow_overlay(img)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)
            st.pyplot(plot_energy_inferno(edges))

            # Contours
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cnts,_ = cv2.findContours(cv2.Canny(gray,60,180),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            detected_sequence = []
            fallback_mode = False
            for i,c in enumerate(cnts[:20]):
                x,y,w,h = cv2.boundingRect(c)
                if w*h < 100: continue
                crop = img[y:y+h, x:x+w].copy()
                desc = descriptor_from_image(crop)
                matches = match_descriptor(desc,SYMBOL_LIBRARY)
                label=""
                if matches:
                    best, dist = matches[0]
                    if dist<=0.18:
                        label=best
                        detected_sequence.append(label)
                st.image(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB), width=80)
                st.write(label if label else "Unrecognized")
            st.write("Detected sequence:", detected_sequence)
            if not detected_sequence:
                st.info("No symbols matched; proceeding with heuristic graph")
                fallback_mode=True
                grammar_model=build_fallback_grammar([f"S{i+1}" for i in range(len(cnts))])

            log_row = {
                "timestamp": datetime.now().isoformat(),
                "file": up.name,
                "num_regions": len(cnts),
                "recognized_sequence": ",".join(detected_sequence)[:500],
                "fallback_used": fallback_mode
            }
            # log CSV
            import csv
            new_file = not os.path.exists(LOG_FILE)
            with open(LOG_FILE,"a",newline="",encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=log_row.keys())
                if new_file: writer.writeheader()
                writer.writerow(log_row)

# ---------- TAB 2 ----------
with tabs[1]:
    st.header("Symbol Library Editor")
    SYMBOL_LIBRARY = load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR)
    st.write("Existing symbols in library:")
    for k,v in SYMBOL_LIBRARY.items():
        st.image(cv2.imread(v["image_path"]), width=50)
        st.write(k, v["info"].get("display_name",""))

# ---------- TAB 3 ----------
with tabs[2]:
    st.header("Logs")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.dataframe(df)
    else:
        st.info("No logs yet.")
