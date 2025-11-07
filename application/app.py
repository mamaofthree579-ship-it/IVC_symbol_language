# app.py
"""
IVC Symbol Research Studio — Symbol Recognition Enhancement
Reads/writes symbol reference at application/ivc_symbol_library.json and images at application/symbols/
Detects contours in uploaded script images, computes descriptors, and matches to library.
Provides quick workflow to create new reference symbols from detected crops.
Also integrates CFG induction (ivc_cfg.py expected in same folder).
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

# ivc_cfg dependency for CFG induction (sequitur)
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
st.title("IVC Symbol Recognizer — Reference Matching & Cataloging")

# ---------------------------
# Utilities
# ---------------------------
def backup_file(path: str):
    try:
        if os.path.exists(path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(BACKUP_DIR, f"{ts}_{os.path.basename(path)}")
            import shutil
            shutil.copy(path, dest)
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
    """Compute log-transformed Hu moments (7 dims) as descriptor."""
    # ensure binary contour presence
    try:
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        M = cv2.moments(thr)
        hu = cv2.HuMoments(M).flatten()
        # log transform: sign * log10(abs)
        for i in range(len(hu)):
            if hu[i] == 0:
                hu[i] = 0.0
            else:
                hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30)
        hu = np.nan_to_num(hu, nan=0.0, posinf=0.0, neginf=0.0)
        return hu.astype(np.float32)
    except Exception:
        return np.zeros(7, dtype=np.float32)

def contour_signature_descriptor(contour: np.ndarray, length: int = 64) -> np.ndarray:
    """
    Resample contour to fixed length and compute normalized curvature-based signature.
    Returns vector of length `length//2` (aggregated).
    """
    if contour is None or len(contour) < 6:
        return np.zeros(length//2, dtype=np.float32)
    pts = contour.reshape(-1,2).astype(np.float32)
    # compute cumulative length and resample
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
    # aggregate into length//2 bins
    bins = np.array_split(angles, length//2)
    feats = np.array([b.mean() if len(b)>0 else 0.0 for b in bins], dtype=np.float32)
    return feats

def descriptor_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """Combine Hu moments + resampled contour signature into a single descriptor vector."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim==3 else img_bgr
    hu = hu_moments_descriptor(gray)
    # find main contour
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 200)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        sig = np.zeros(32, dtype=np.float32)
    else:
        main = max(cnts, key=cv2.contourArea)
        sig = contour_signature_descriptor(main, length=64)
    # normalize and concatenate
    desc = np.concatenate([hu, sig])
    # normalize scale
    norm = np.linalg.norm(desc) + 1e-9
    return (desc / norm).astype(np.float32)

# ---------------------------
# Load / build reference library descriptors
# ---------------------------
def load_symbol_library(lib_path: str, symbols_dir: str) -> Dict[str, Dict[str, Any]]:
    lib = safe_load_json(lib_path)
    # ensure structure is dict
    if not isinstance(lib, dict):
        lib = {}
    descriptors = {}
    # for each key, if image_path exists, compute descriptor; otherwise check symbols dir for file
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
            # skip for now
            descriptors[key] = {"info": info, "desc": None, "image_path": ipath}
    return descriptors

# initialize library
if not os.path.exists(SYMBOL_LIB_FILE):
    # create empty file
    save_json(SYMBOL_LIB_FILE, {})  # create file with empty dict

SYMBOL_LIBRARY = load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR)

# ---------------------------
# Matching helpers
# ---------------------------
def match_descriptor(desc: np.ndarray, library: Dict[str, Dict[str, Any]], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    Match a descriptor against the library.
    Returns list of (symbol_key, distance) sorted ascending (smaller = more similar).
    Only considers library entries with computed descriptors.
    """
    entries = []
    for key, v in library.items():
        d = v.get("desc")
        if d is None:
            continue
        # use cosine distance (1 - cosine similarity)
        if np.linalg.norm(desc)==0 or np.linalg.norm(d)==0:
            dist = float("inf")
        else:
            cos = np.dot(desc, d) / (np.linalg.norm(desc) * np.linalg.norm(d) + 1e-12)
            # clamp
            cos = max(min(cos, 1.0), -1.0)
            dist = 1.0 - cos
        entries.append((key, float(dist)))
    entries.sort(key=lambda x: x[1])
    return entries[:top_k]

# ---------------------------
# Energy visualizer (edge + gradient overlay)
# ---------------------------
def plot_energy_inferno(edges: np.ndarray):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(edges, cmap="inferno")
    ax.axis("off")
    return fig

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

# ---------------------------
# Streamlit UI
# ---------------------------

st.sidebar.header("Recognition Controls")
match_threshold = st.sidebar.slider("Match distance threshold (lower = stricter)", 0.01, 0.6, 0.18, step=0.01)
top_k = st.sidebar.number_input("Top-K matches shown", min_value=1, max_value=5, value=3)
show_unrecognized_panel = st.sidebar.checkbox("Show panel to create new reference for unrecognized regions", True)

tabs = st.tabs(["Upload & Recognize", "Library Editor", "CFG Induction", "Logs"])

# ------------------ TAB: Upload & Recognize ------------------
with tabs[0]:
    st.header("Upload script images and run recognition")
    uploaded = st.file_uploader("Upload one or more artifact images (jpg/png/tif)", type=["jpg","jpeg","png","tif"], accept_multiple_files=True)
    if uploaded:
        for up in uploaded:
            st.subheader(f"File: {up.name}")
            file_bytes = up.read()
            arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image")
                continue

            # energy & flow visualization
            overlay, edges = field_flow_overlay(img)
            st.markdown("**Energy / Field Flow Overlay**")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.markdown("**Edge (inferno) view**")
            fig = plot_energy_inferno(edges)
            st.pyplot(fig)

            # detect contours to find candidate glyphs
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            edges_small = cv2.Canny(blur, 60, 180)
            cnts, _ = cv2.findContours(edges_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # sort left-to-right then top-to-bottom
            bboxes = []
            h,w = img.shape[:2]
            for c in cnts:
                x,y,ww,hh = cv2.boundingRect(c)
                area = ww*hh
                if area < max(100, (h*w)//3000):
                    continue
                bboxes.append((x,y,ww,hh,c))
            bboxes = sorted(bboxes, key=lambda z: (z[1], z[0]))

            st.write(f"Detected {len(bboxes)} candidate regions")
            if not bboxes:
                st.info("No candidate contours found — try adjusting image contrast or upload different image.")
                continue

            # display crops in grid and match to library
            cols = st.columns(4)
            detected_sequence = []
            for i, (x,y,ww,hh,c) in enumerate(bboxes):
                crop = img[y:y+hh, x:x+ww].copy()
                # pad to square
                pad = max(ww,hh)
                canvas = np.ones((pad,pad,3), dtype=np.uint8)*255
                cx = (pad - ww)//2
                cy = (pad - hh)//2
                canvas[cy:cy+hh, cx:cx+ww] = crop
                # descriptor
                desc = descriptor_from_image(canvas)
                matches = match_descriptor(desc, SYMBOL_LIBRARY, top_k)
                label = ""
                confidence = None
                if matches:
                    best_key, best_dist = matches[0]
                    confidence = 1.0 - best_dist
                    if best_dist <= match_threshold:
                        label = best_key
                    else:
                        label = ""  # treated as unrecognized due to threshold
                with cols[i % 4]:
                    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_column_width=True)
                    if label:
                        info = SYMBOL_LIBRARY.get(label, {}).get("info", {})
                        disp = info.get("display_name", label)
                        st.success(f"{disp}  ({label}) — conf {confidence:.2f}")
                        detected_sequence.append(label)
                    else:
                        st.warning("Unrecognized")
                        # show top k suggestions regardless of threshold
                        if matches:
                            st.caption("Top suggestions:")
                            for mk, md in matches:
                                nm = SYMBOL_LIBRARY.get(mk, {}).get("info", {}).get("display_name", mk)
                                st.write(f"- {nm} ({mk}) — distance {md:.3f}")
                        # quick UI to create new reference
                        if show_unrecognized_panel:
                            new_name = st.text_input(f"Name this region #{i}", key=f"newname_{up.name}_{i}")
                            if st.button(f"Save region #{i} as new symbol", key=f"save_new_{up.name}_{i}"):
                                if not new_name:
                                    st.error("Enter a display name before saving")
                                else:
                                    sid = new_name.strip().lower().replace(" ", "_")
                                    # ensure unique id
                                    if sid in SYMBOL_LIBRARY:
                                        sid = f"{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    # save image file
                                    dest_path = os.path.join(SYMBOLS_DIR, f"{sid}.png")
                                    cv2.imwrite(dest_path, canvas)
                                    # update library JSON
                                    lib = safe_load_json(SYMBOL_LIB_FILE)
                                    lib[sid] = {"display_name": new_name, "core": "", "domain": "", "notes": "", "image_path": dest_path}
                                    save_json(SYMBOL_LIB_FILE, lib)
                                    # update in-memory library descriptors
                                    SYMBOL_LIBRARY.update(load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR))
                                    st.success(f"Saved new symbol {sid}")

            # show detected sequence (labels if available, else placeholders)
            st.subheader("Detected sequence (labels or placeholders)")
            if detected_sequence:
                st.write(detected_sequence)
            else:
                # if no recognized labels, show placeholders for sequence (S1..)
                placeholders = [f"S{i+1}" for i in range(len(bboxes))]
                st.write(placeholders)

            # log upload
            log_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "file": up.name,
                "num_regions": len(bboxes),
                "recognized_sequence": ",".join(detected_sequence)[:500]
            }
            append_csv_row(LOG_FILE, log_row)

# ------------------ TAB: Library Editor ------------------
with tabs[1]:
    st.header("Symbol Library Editor")
    st.markdown(f"Library file: `{SYMBOL_LIB_FILE}` — symbols dir: `{SYMBOLS_DIR}`")
    lib = safe_load_json(SYMBOL_LIB_FILE)
    if not lib:
        st.info("Library appears empty — create new symbols from the Upload tab or add JSON entries here.")
    edited = st.text_area("Edit library JSON", value=json.dumps(lib, indent=2), height=400)
    if st.button("Save library JSON"):
        try:
            newlib = json.loads(edited)
            save_json(SYMBOL_LIB_FILE, newlib)
            # refresh in-memory descriptors
            SYMBOL_LIBRARY = load_symbol_library(SYMBOL_LIB_FILE, SYMBOLS_DIR)
            st.success("Library saved and reloaded.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.markdown("---")
    st.subheader("Reference Gallery")
    # show thumbnails (grid)
    items = []
    # prefer index CSV if exists
    for sid, info in lib.items():
        path = info.get("image_path") or os.path.join(SYMBOLS_DIR, f"{sid}.png")
        if os.path.exists(path):
            items.append((sid, info.get("display_name", sid), path, info.get("domain","")))
    if not items:
        st.info("No reference images available.")
    else:
        cols = st.columns(4)
        for i,(sid, dname, path, domain) in enumerate(items):
            with cols[i%4]:
                try:
                    im = Image.open(path)
                    st.image(im, caption=f"{dname}\n({sid})", use_column_width=True)
                    if st.button(f"Delete {sid}", key=f"del_{sid}"):
                        # delete reference
                        confirm = st.checkbox(f"Confirm delete {sid}", key=f"confirm_del_{sid}")
                        if confirm:
                            # remove entry
                            lib = safe_load_json(SYMBOL_LIB_FILE)
                            if sid in lib:
                                lib.pop(sid, None)
                                save_json(SYMBOL_LIB_FILE, lib)
                                SYMBOL_LIBRARY.pop(sid, None)
                                st.success(f"Deleted {sid}; refresh page.")
                except Exception:
                    st.write(dname)

# ------------------ TAB: CFG Induction ------------------
with tabs[2]:
    st.header("CFG Induction (Sequitur-style)")
    if not IVCCFG_AVAILABLE:
        st.warning("ivc_cfg.py not found or failed to import — CFG induction disabled.")
    else:
        # allow sequences upload or use recent log recognized sequences
        seq_source = st.radio("Select sequences source", ["Upload sequences JSON", "Use recent recognized sequences"])
        sequences = []
        if seq_source.startswith("Upload"):
            seq_file = st.file_uploader("Upload sequences JSON", type=["json"])
            if seq_file:
                try:
                    sequences = json.load(seq_file)
                    st.success(f"Loaded {len(sequences)} sequences from uploaded file")
                except Exception as e:
                    st.error(f"Failed to read JSON: {e}")
        else:
            # derive sequences from log file recognized_sequence column
            df_log = safe_load_csv(LOG_FILE)
            if not df_log.empty and "recognized_sequence" in df_log.columns:
                sequences = []
                for _, r in df_log.iterrows():
                    seq = str(r.get("recognized_sequence","")).strip()
                    if seq:
                        items = [s for s in seq.split(",") if s]
                        if items:
                            sequences.append(items)
                st.info(f"Using {len(sequences)} sequences from log")
            else:
                st.info("No recognized sequences found in logs. Upload sequences JSON or run recognition first.")

        if sequences:
            min_occ = st.slider("Minimum rule occurrence", 2, 20, 2)
            if st.button("Infer CFG rules"):
                res = sequitur_infer(sequences, min_rule_occurrence=min_occ)
                st.subheader("Rules")
                if res["rules"]:
                    st.code(rules_to_text(res["rules"]))
                    tmpf = os.path.join(tempfile.gettempdir(), "ivc_cfg_rules.json")
                    export_rules_json(tmpf, res)
                    with open(tmpf, "r", encoding="utf-8") as fh:
                        st.download_button("Download CFG JSON", data=fh.read(), file_name="ivc_cfg_rules.json", mime="application/json")
                    # show graph
                    edges = rules_to_graph_edges(res["rules"])
                    if edges:
                        G = nx.DiGraph()
                        G.add_edges_from(edges)
                        fig, ax = plt.subplots(figsize=(7,5))
                        pos = nx.spring_layout(G, k=0.6, iterations=40)
                        nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightgray", ax=ax)
                        st.pyplot(fig)
                else:
                    st.info("No strong rules found.")
        else:
            st.info("No sequences available for CFG induction.")

# ------------------ TAB: Logs ------------------
with tabs[3]:
    st.header("Logs & Datasets")
    st.subheader("Recognition Log")
    if os.path.exists(LOG_FILE):
        dfl = safe_load_csv(LOG_FILE)
        st.dataframe(dfl)
        st.download_button("Download log CSV", data=dfl.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(LOG_FILE))
    else:
        st.info("No logs yet.")
    st.subheader("Labeled dataset")
    if os.path.exists(LABELED_CSV):
        dfl2 = safe_load_csv(LABELED_CSV)
        st.dataframe(dfl2)
        st.download_button("Download labeled CSV", data=dfl2.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(LABELED_CSV))
    else:
        st.info("No labeled dataset yet.")

st.markdown("---")
st.caption("Done — symbol recognition enhancement added. New workflow: upload script images → the app detects contours → attempts to match to your reference library at application/ivc_symbol_library.json → if unrecognized you can save region to library. Use CFG tab to infer hierarchical rules from recognized sequences.")
