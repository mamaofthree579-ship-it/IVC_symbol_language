# app.py
"""
IVC Symbol Recognizer — JSON-Free Version
Features:
- Symbol recognition from uploaded images
- Energy / field overlay visualization
- CSV logging
- Fallback CFG adjacency graph
- Multi-tab Streamlit interface
"""

import os
from datetime import datetime
import numpy as np
import cv2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------
# Paths & Config
# ---------------------------
BASE_DIR = os.getcwd()
APP_DIR = os.path.join(BASE_DIR, "application")
os.makedirs(APP_DIR, exist_ok=True)

LOG_FILE = os.path.join(APP_DIR, "ivc_symbol_log.csv")
BACKUP_DIR = os.path.join(APP_DIR, "backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

st.set_page_config(page_title="IVC Symbol Recognizer", layout="wide")
st.title("IVC Symbol Recognizer — JSON-Free Mode")

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

def descriptor_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hu = hu_moments_descriptor(gray)
    blur = cv2.GaussianBlur(gray, (5,5),0)
    edges = cv2.Canny(blur,50,200)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        sig = np.zeros(32, dtype=np.float32)
    else:
        main = max(cnts,key=cv2.contourArea)
        pts = main.reshape(-1,2).astype(np.float32)
        vecs = pts[1:]-pts[:-1]
        norms = np.linalg.norm(vecs,axis=1,keepdims=True)+1e-9
        vn = vecs/norms
        dots = (vn[:-1]*vn[1:]).sum(axis=1)
        dots = np.clip(dots,-1,1)
        angles = np.arccos(dots)
        bins = np.array_split(angles,16)
        sig = np.array([b.mean() if len(b)>0 else 0.0 for b in bins],dtype=np.float32)
    desc = np.concatenate([hu,sig])
    norm = np.linalg.norm(desc)+1e-9
    return (desc/norm).astype(np.float32)

# ---------------------------
# Fallback CFG / adjacency
# ---------------------------
def build_fallback_grammar(symbol_sequence):
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
# Tabs UI
# ---------------------------
tabs = st.tabs(["Upload & Recognize","Logs"])

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

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cnts,_ = cv2.findContours(cv2.Canny(gray,60,180),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            detected_sequence = []
            fallback_mode = False
            for i,c in enumerate(cnts[:20]):
                x,y,w,h = cv2.boundingRect(c)
                if w*h < 100: continue
                crop = img[y:y+h, x:x+w].copy()
                desc = descriptor_from_image(crop)
                label = f"S{i+1}"  # runtime label
                detected_sequence.append(label)
                st.image(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB), width=80)
                st.write(label)

            st.write("Detected sequence:", detected_sequence)
            if not detected_sequence:
                fallback_mode=True
                grammar_model = build_fallback_grammar([f"S{i+1}" for i in range(len(cnts))])

            # Log
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
    st.header("Logs")
    df = safe_load_csv(LOG_FILE)
    if df.empty:
        st.info("No logs yet or CSV is empty/malformed.")
    else:
        st.dataframe(df)
