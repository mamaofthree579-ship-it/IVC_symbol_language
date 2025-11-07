# app.py
"""
IVC Symbol Recognizer — JSON-Free with Converter Tab
Features:
- Symbol detection & origin alignment
- Energy / field overlays
- Recursive pattern & adjacency detection
- Translation mapping (runtime)
- CSV logging
- Multi-tab Streamlit interface
- Converter: Export JSON/CSV of symbols with IVC meanings
"""

import os
from datetime import datetime
import json
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------
# Config
# ---------------------------
BASE_DIR = os.getcwd()
APP_DIR = os.path.join(BASE_DIR, "application")
os.makedirs(APP_DIR, exist_ok=True)

LOG_FILE = os.path.join(APP_DIR, "ivc_symbol_log.csv")
BACKUP_DIR = os.path.join(APP_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

st.set_page_config(page_title="IVC Symbol Recognizer", layout="wide")
st.title("IVC Symbol Recognizer — JSON-Free Mode with Converter")

# ---------------------------
# Session-State Initialization
# ---------------------------
if "session_symbols" not in st.session_state:
    st.session_state.session_symbols = []

# ---------------------------
# IVC Symbol Library (for translation)
# ---------------------------
IVC_SYMBOL_LIBRARY = {
    "circle": {"meaning": "Particle / Consciousness unit / Self", "functional_domain": "Quantum-Material Interface"},
    "jar": {"meaning": "Containment / Material manifestation / Body", "functional_domain": "Gravitational Field / Matter"},
    "fish": {"meaning": "Particle / Consciousness unit / Self", "functional_domain": "Quantum-Material Interface"},
    "double_wavy": {"meaning": "Vibration / Oscillation / Energy Flow", "functional_domain": "Waveform / Frequency Dynamics"},
    "arrow": {"meaning": "Linear movement through time / Directionality", "functional_domain": "Temporal Flow / Vector Dynamics"},
    "triangle_square": {"meaning": "Coupled quantum tunneling / Dimensional harmony", "functional_domain": "Quantum Interface"},
    "spiral_arrow": {"meaning": "Energy flow initiation / Field rotation", "functional_domain": "Rotational Vortex / Scalar Initiation"},
    "nested_squares": {"meaning": "Dimensional layering / Boundary harmonics", "functional_domain": "Gravitational or Dimensional Compression"},
    "lattice": {"meaning": "Stabilization / Field grid / Linkage", "functional_domain": "Resonant Grid / Planetary Network"},
    # Add more as needed
}

# ---------------------------
# CSV Utilities
# ---------------------------
def safe_load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

def append_csv_row(filepath: str, row: dict):
    import csv
    new_file = not os.path.exists(filepath)
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        st.error(f"Failed to write CSV log: {e}")

# ---------------------------
# Descriptor Utilities
# ---------------------------
def hu_moments_descriptor(gray):
    blur = cv2.GaussianBlur(gray, (5,5),0)
    _, thr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(thr)
    hu = cv2.HuMoments(M).flatten()
    hu = [-np.sign(h)*np.log10(abs(h)+1e-30) if h!=0 else 0.0 for h in hu]
    return np.nan_to_num(np.array(hu,dtype=np.float32))

def contour_signature_descriptor(contour, length=32):
    pts = contour.reshape(-1,2).astype(np.float32)
    if len(pts)<2: return np.zeros(length, dtype=np.float32)
    diffs = pts[1:]-pts[:-1]
    norms = np.linalg.norm(diffs,axis=1,keepdims=True)+1e-9
    vn = diffs/norms
    dots = (vn[:-1]*vn[1:]).sum(axis=1)
    dots = np.clip(dots,-1,1)
    angles = np.arccos(dots)
    bins = np.array_split(angles,length)
    feats = np.array([b.mean() if len(b)>0 else 0.0 for b in bins],dtype=np.float32)
    return feats

def descriptor_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hu = hu_moments_descriptor(gray)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,50,200)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        sig = np.zeros(32, dtype=np.float32)
    else:
        main = max(cnts,key=cv2.contourArea)
        sig = contour_signature_descriptor(main,32)
    desc = np.concatenate([hu,sig])
    norm = np.linalg.norm(desc)+1e-9
    return (desc/norm).astype(np.float32)

# ---------------------------
# Fallback CFG / adjacency
# ---------------------------
def build_fallback_graph(symbol_sequence):
    G = nx.DiGraph()
    for i,sym in enumerate(symbol_sequence[:-1]):
        G.add_edge(sym, symbol_sequence[i+1])
    return G

# ---------------------------
# Energy overlay
# ---------------------------
def field_flow_overlay(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    overlay = cv2.addWeighted(img,0.65,lines,0.9,0)
    overlay = cv2.bilateralFilter(overlay,7,75,75)
    return overlay, edges

def plot_energy_inferno(edges):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(edges,cmap="inferno")
    ax.axis("off")
    return fig

# ---------------------------
# Streamlit Tabs
# ---------------------------
tabs = st.tabs(["Upload & Recognize","Symbol Analysis","Logs","Converter"])

# ---------- TAB 1 ----------
with tabs[0]:
    st.header("Upload Images / Scripts")
    uploaded = st.file_uploader("Upload images", type=["png","jpg","tif"], accept_multiple_files=True)
    session_symbols = st.session_state.session_symbols
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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cnts,_ = cv2.findContours(cv2.Canny(gray,60,180),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            detected_sequence = []
            for i,c in enumerate(cnts[:20]):
                x,y,w,h = cv2.boundingRect(c)
                if w*h < 100: continue
                crop = img[y:y+h, x:x+w].copy()
                desc = descriptor_from_image(crop)
                label = f"S{i+1}"
                detected_sequence.append(label)
                # Try to guess meaning based on shape if possible
                probable_meaning = ""
                functional_domain = ""
                # Placeholder: advanced shape recognition can fill this in
                session_symbols.append({
                    "label": label,
                    "descriptor": desc,
                    "crop": crop,
                    "meaning": probable_meaning,
                    "functional_domain": functional_domain
                })
                st.image(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB), width=80)
                st.write(label)

            st.write("Detected sequence:", detected_sequence)
            fallback_mode = False
            if not detected_sequence:
                fallback_mode=True
                fallback_graph = build_fallback_graph([f"S{i+1}" for i in range(len(cnts))])
                st.write("Fallback adjacency graph built.")

            # Log CSV
            log_row = {
                "timestamp": datetime.now().isoformat(),
                "file": up.name,
                "num_regions": len(cnts),
                "recognized_sequence": ",".join(detected_sequence)[:500],
                "fallback_used": fallback_mode
            }
            append_csv_row(LOG_FILE, log_row)

# ---------- TAB 2 ----------
with tabs[1]:
    st.header("Symbol Analysis & Translation")
    session_symbols = st.session_state.session_symbols
    if not session_symbols:
        st.info("No symbols detected yet. Upload images in Tab 1.")
    else:
        st.subheader("Detected Symbols")
        for sym in session_symbols:
            st.image(cv2.cvtColor(sym["crop"],cv2.COLOR_BGR2RGB), width=60)
            st.write(sym["label"])
        # Adjacency graph
        seq_labels = [s["label"] for s in session_symbols]
        G = build_fallback_graph(seq_labels)
        st.subheader("Adjacency Graph (Fallback CFG)")
        pos = nx.spring_layout(G)
        plt.figure(figsize=(5,5))
        nx.draw(G,pos,with_labels=True,node_color='lightblue',node_size=700,edge_color='gray')
        st.pyplot(plt.gcf())
        plt.clf()

# ---------- TAB 3 ----------
with tabs[2]:
    st.header("Logs")
    df = safe_load_csv(LOG_FILE)
    if df.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(df)

# ---------- TAB 4: Converter ----------
with tabs[3]:
    st.header("Converter — Export JSON / CSV")
    uploaded_conv = st.file_uploader("Upload single image for conversion", type=["png","jpg","tif"])
    if uploaded_conv:
        arr = np.frombuffer(uploaded_conv.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Failed to decode image")
        else:
            overlay, edges = field_flow_overlay(img)
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Energy Overlay", use_column_width=True)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cnts,_ = cv2.findContours(cv2.Canny(gray,60,180),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            symbols_data = []
            for i,c in enumerate(cnts):
                x,y,w,h = cv2.boundingRect(c)
                if w*h<50: continue
                crop = img[y:y+h, x:x+w].copy()
                desc = descriptor_from_image(crop)
                label = f"S{i+1}"
                # Try to match with IVC library (placeholder, can be extended)
                meaning = ""
                functional_domain = ""
                symbols_data.append({
                    "id": label,
                    "bbox": [int(x),int(y),int(w),int(h)],
                    "hu_moments": desc[:7].tolist(),
                    "signature": desc[7:].tolist(),
                    "meaning": meaning,
                    "functional_domain": functional_domain
                })
            st.subheader(f"Detected {len(symbols_data)} symbols")
            df_symbols = pd.DataFrame(symbols_data)
            st.dataframe(df_symbols)

            # JSON Download
            export_json = {
                "filename": uploaded_conv.name,
                "symbols_detected": len(symbols_data),
                "symbols": symbols_data
            }
            st.download_button("Download JSON", json.dumps(export_json, indent=2), file_name=f"{uploaded_conv.name}_converted.json")

            # CSV Download
            st.download_button("Download CSV", df_symbols.to_csv(index=False), file_name=f"{uploaded_conv.name}_converted.csv")
