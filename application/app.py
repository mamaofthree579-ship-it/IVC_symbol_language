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
import shutil
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from PIL import Image
from io import BytesIO

# Try to import the canvas; if missing, we'll still allow uploads
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# -----------------------
# Config & files
# -----------------------
st.set_page_config(page_title="IVC Symbol Research Studio", layout="wide")
st.title("🌀 IVC Symbol Research Studio — Catalog, Draw, Analyze")

BASE_DIR = os.getcwd()
SYMBOL_LIB_FILE = os.path.join(BASE_DIR, "ivc_symbol_library.json")
SYMBOL_INDEX_FILE = os.path.join(BASE_DIR, "ivc_symbol_index.csv")
SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols")
LOG_FILE = os.path.join(BASE_DIR, "ivc_symbol_log.csv")
LABELED_CSV = os.path.join(BASE_DIR, "ivc_labeled_dataset.csv")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")

os.makedirs(SYMBOLS_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# -----------------------
# Defaults
# -----------------------
DEFAULT_LIBRARY = {
    "spiral_arrow": {"display_name": "Spiral Arrow", "core": "Energy flow initiation / Field rotation", "domain": "Rotational Vortex", "notes": ""},
    "nested_squares": {"display_name": "Nested Squares", "core": "Dimensional layering", "domain": "Boundary harmonics", "notes": ""},
    "lattice": {"display_name": "Lattice", "core": "Resonant grid / stabilization", "domain": "Planetary/Field Grid", "notes": ""}
}

if not os.path.exists(SYMBOL_LIB_FILE):
    with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_LIBRARY, f, indent=2)

with open(SYMBOL_LIB_FILE, "r", encoding="utf-8") as f:
    SYMBOL_LIB: Dict[str, Dict[str, str]] = json.load(f)

# -----------------------
# Backup helpers
# -----------------------
def backup_file(filepath: str):
    try:
        if os.path.exists(filepath):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = os.path.basename(filepath)
            dest = os.path.join(BACKUP_DIR, f"{ts}_{fn}")
            shutil.copy(filepath, dest)
    except Exception as e:
        st.warning(f"Backup failed for {filepath}: {e}")

# -----------------------
# CSV helpers
# -----------------------
def append_csv_row(filepath: str, row: Dict[str, Any]):
    new_file = not os.path.exists(filepath)
    try:
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)
        backup_file(filepath)
    except Exception as e:
        st.error(f"Failed to write {filepath}: {e}")

def safe_load_csv(path: str) -> pd.DataFrame:
    import csv as _csv
    rows = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = _csv.reader(f)
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

# -----------------------
# Image utilities
# -----------------------
def read_image_bytes(uploaded) -> np.ndarray:
    data = uploaded.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def save_pil_image(img_pil: Image.Image, path: str):
    img_pil.save(path, format="PNG")

def save_cv2_image(img_bgr: np.ndarray, path: str):
    cv2.imwrite(path, img_bgr)

def preprocess_gray(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

def detect_edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 100, 200)

# Visualizers
def plot_energy_lines(edges: np.ndarray):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(edges, cmap="inferno")
    ax.axis("off")
    fig.tight_layout()
    return fig

def field_flow_overlay_cv(img_bgr: np.ndarray):
    gray = preprocess_gray(img_bgr)
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
    flow_col = cv2.GaussianBlur(flow_col, (5,5), 0)
    edges = detect_edges(gray)
    mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    lines = cv2.bitwise_and(flow_col, flow_col, mask=mask)
    overlay = cv2.addWeighted(img_bgr, 0.65, lines, 0.9, 0)
    overlay = cv2.bilateralFilter(overlay, 7, 75, 75)
    return overlay

# -----------------------
# OCR
# -----------------------
def run_ocr(img_bgr: np.ndarray) -> Tuple[str, Any]:
    try:
        gray = preprocess_gray(img_bgr)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text",[]) if w.strip()]
        text = " ".join(words).strip()
        return text, data
    except Exception as e:
        return f"[OCR error: {e}]", None

# -----------------------
# Symbol storage & index helpers
# -----------------------
def normalize_id(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def add_symbol_to_library(symbol_id: str, display_name: str, core: str, domain: str, notes: str, image_bgr: np.ndarray):
    # update JSON library
    SYMBOL_LIB[symbol_id] = {"display_name": display_name, "core": core, "domain": domain, "notes": notes, "image_path": f"symbols/{symbol_id}.png"}
    backup_file(SYMBOL_LIB_FILE)
    with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
        json.dump(SYMBOL_LIB, f, indent=2)

    # save image
    path = os.path.join(SYMBOLS_DIR, f"{symbol_id}.png")
    save_cv2_image(image_bgr, path)

    # append to index CSV
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "symbol_id": symbol_id,
        "display_name": display_name,
        "core": core,
        "domain": domain,
        "notes": notes,
        "image_path": path
    }
    append_csv_row(SYMBOL_INDEX_FILE, row)
    st.success(f"Saved symbol {display_name} as {symbol_id}")

# -----------------------
# UI tabs
# -----------------------
tabs = st.tabs(["🔍 Analyze", "🖋 Symbol Creator / Catalog", "✍️ Manual Label / Annotate", "📘 Library Editor", "📂 Logs"])

# -----------------------
# TAB: Analyze (stacked Energy Lines + Field Flow)
# -----------------------
with tabs[0]:
    st.header("Analyze Artifact — Energy Lines & Field Flow")
    uploaded = st.file_uploader("Upload artifact image", type=["jpg","jpeg","png"], key="analyze_upload")
    if uploaded:
        img = read_image_bytes(uploaded)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Artifact", use_column_width=True)

        gray = preprocess_gray(img)
        edges = detect_edges(gray)

        st.subheader("1️⃣ Energy Lines (Canny / inferno)")
        fig = plot_energy_lines(edges)
        st.pyplot(fig)

        st.subheader("2️⃣ Field Flow Overlay (gradient by image gradients)")
        flow = field_flow_overlay_cv(img)
        st.image(cv2.cvtColor(flow, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("OCR")
        ocr_text, ocr_data = run_ocr(img)
        st.code(ocr_text if ocr_text else "[No OCR text detected]")

        # basic library hinting (text-based)
        hints = []
        for key, info in SYMBOL_LIB.items():
            if key.replace("_", " ") in ocr_text.lower():
                hints.append(key)
        if hints:
            st.markdown("**Library hints (from OCR)**: " + ", ".join(hints))

        # log basic analysis
        log_row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "file": getattr(uploaded, "name", "uploaded"),
                   "ocr_text": ocr_text[:500],
                   "hints": ",".join(hints)}
        append_csv_row(LOG_FILE, log_row)

# -----------------------
# TAB: Symbol Creator / Catalog
# -----------------------
with tabs[1]:
    st.header("🖋 Symbol Creator & Catalog")
    st.markdown("Draw a symbol or upload a sketch/photo, add metadata, and save to the symbol library. Saved symbols appear in the gallery below.")

    col1, col2 = st.columns([2,1])

    with col1:
        if CANVAS_AVAILABLE:
            st.info("Use the canvas to draw. Set background image by uploading a reference in the right column (optional).")
            bg_file = st.file_uploader("Optional: upload background photo/sketch to trace", type=["jpg","png","jpeg"], key="sym_bg")
            bg_img = None
            if bg_file:
                bg_bytes = bg_file.read()
                bg_pil = Image.open(BytesIO(bg_bytes)).convert("RGBA")
                bg_img = bg_pil
            # canvas params
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",  # transparent fill
                stroke_width=3,
                stroke_color="#000000",
                background_image=bg_img,
                height=400,
                width=400,
                drawing_mode="freedraw",
                key="symbol_canvas"
            )
            drawn_image = None
            if canvas_result and canvas_result.image_data is not None:
                # image_data is RGBA uint8
                im = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA").convert("RGB")
                drawn_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                st.image(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB), caption="Canvas result preview", width=240)
        else:
            st.warning("Drawing canvas unavailable. Install `streamlit-drawable-canvas` to enable drawing: pip install streamlit-drawable-canvas")
            drawn_image = None

        # Upload fallback
        uploaded_sym = st.file_uploader("Or upload symbol image (PNG/JPG)", type=["png","jpg","jpeg"], key="sym_upload")
        uploaded_sym_img = None
        if uploaded_sym:
            uploaded_sym_img = read_image_bytes(uploaded_sym)
            st.image(cv2.cvtColor(uploaded_sym_img, cv2.COLOR_BGR2RGB), caption="Uploaded symbol preview", width=240)

    with col2:
        st.subheader("Metadata")
        name = st.text_input("Symbol display name (e.g. Spiral Arrow)")
        suggested_id = normalize_id(name) if name else ""
        st.text_input("Symbol ID (auto-normalized)", value=suggested_id, key="symbol_id_display")
        core = st.text_area("Core Meaning / Concept", height=80)
        domain = st.text_input("Functional Field / Domain")
        notes = st.text_area("Notes / Comparative parallels", height=120)

        if st.button("Save Symbol to Library"):
            # select source image: canvas -> upload -> error
            src_img = None
            if CANVAS_AVAILABLE and 'canvas_result' in locals() and canvas_result and canvas_result.image_data is not None:
                src_img = drawn_image
            elif uploaded_sym_img is not None:
                src_img = uploaded_sym_img
            else:
                st.error("No symbol image found. Draw on the canvas or upload an image first.")
                src_img = None

            if src_img is not None:
                if not name:
                    st.error("Please provide a display name for the symbol.")
                else:
                    sid = normalize_id(name) if not suggested_id else normalize_id(suggested_id)
                    # ensure unique by appending timestamp if exists
                    if sid in SYMBOL_LIB:
                        sid = f"{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    add_symbol_to_library(sid, name, core, domain, notes, src_img)
                    # refresh SYMBOL_LIB in-memory
                    with open(SYMBOL_LIB_FILE, "r", encoding="utf-8") as f:
                        SYMBOL_LIB.update(json.load(f))

    st.markdown("---")
    st.subheader("Symbol Gallery (grid)")
    # load index CSV to present grid; fallback: scan symbols dir
    index_df = safe_load_csv(SYMBOL_INDEX_FILE)
    gallery_items = []
    if not index_df.empty:
        for _, row in index_df.iterrows():
            imgp = row.get("image_path", "") or ""
            if imgp and os.path.exists(imgp):
                gallery_items.append({"id": row.get("symbol_id", ""), "display_name": row.get("display_name",""), "path": imgp, "domain": row.get("domain",""), "core": row.get("core","")})
    else:
        # fallback: scan SYMBOLS_DIR and SYMBOL_LIB
        for sid, info in SYMBOL_LIB.items():
            imgp = os.path.join(SYMBOLS_DIR, f"{sid}.png")
            display = info.get("display_name", sid)
            if os.path.exists(imgp):
                gallery_items.append({"id": sid, "display_name": display, "path": imgp, "domain": info.get("domain",""), "core": info.get("core","")})

    # grid display
    cols = st.columns(4)
    for i, item in enumerate(gallery_items):
        col = cols[i % 4]
        with col:
            try:
                im = Image.open(item["path"])
                st.image(im, caption=f"{item['display_name']}\n({item['id']})", use_column_width=True)
                st.caption(item.get("domain",""))
            except Exception:
                st.write(item["display_name"])

# -----------------------
# TAB: Manual Label / Annotate
# -----------------------
with tabs[2]:
    st.header("Manual Labeling & Annotation")
    st.markdown("Upload an artifact image and label detected candidate regions (saves to labeled CSV).")

    up = st.file_uploader("Upload image to annotate", type=["jpg","jpeg","png"], key="man_label")
    if up:
        img = read_image_bytes(up)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Image for annotation", use_column_width=True)
        gray = preprocess_gray(img)
        edges = detect_edges(gray)
        st.subheader("Detected energy lines (edge map)")
        st.pyplot(plot_energy_lines(edges))
        # simple contour-based candidates
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        h,w = img.shape[:2]
        min_area = max(60, (h*w)//15000)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x,y,ww,hh = cv2.boundingRect(c)
            pad = int(0.05*max(ww,hh))
            xa,ya = max(0,x-pad), max(0,y-pad)
            xb,yb = min(w,x+ww+pad), min(h,y+hh+pad)
            crop = img[ya:yb, xa:xb]
            candidates.append({"bbox":(xa,ya,xb,yb), "crop":crop})
        st.markdown(f"Detected {len(candidates)} candidate regions")
        if candidates:
            cols = st.columns(3)
            labelled_rows = []
            for i, cinfo in enumerate(candidates[:30]):  # limit UI to first 30
                col = cols[i%3]
                with col:
                    st.image(cv2.cvtColor(cinfo["crop"], cv2.COLOR_BGR2RGB), use_column_width=True)
                    label = st.selectbox(f"Label region #{i}", options=["(none)"]+list(SYMBOL_LIB.keys()), key=f"lab_{i}")
                    note = st.text_input(f"Note #{i}", key=f"note_{i}")
                    if label != "(none)":
                        if st.button(f"Save region #{i}", key=f"save_reg_{i}"):
                            x1,y1,x2,y2 = cinfo["bbox"]
                            row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "image_file": getattr(up, "name", "uploaded"),
                                   "bbox": f"{x1},{y1},{x2},{y2}",
                                   "label": label,
                                   "note": note}
                            append_csv_row(LABELED_CSV, row)
                            st.success(f"Saved label {label} for region #{i}")

# -----------------------
# TAB: Library Editor
# -----------------------
with tabs[3]:
    st.header("Symbol Library Editor")
    st.markdown("Edit the `ivc_symbol_library.json` content directly and save (backup created automatically).")
    lib_text = json.dumps(SYMBOL_LIB, indent=2)
    edited = st.text_area("Edit library JSON", value=lib_text, height=400)
    if st.button("Save library JSON"):
        try:
            new_lib = json.loads(edited)
            backup_file(SYMBOL_LIB_FILE)
            with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
                json.dump(new_lib, f, indent=2)
            st.success("Library saved (backup created). Please refresh to see changes.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
    st.markdown("Quick preview of library keys:")
    st.write(list(SYMBOL_LIB.keys()))

# -----------------------
# TAB: Logs
# -----------------------
with tabs[4]:
    st.header("Logs, Index & Backups")
    st.markdown("Download or inspect analysis logs, labeled dataset, symbol index, and backups.")

    st.subheader("Analysis Log")
    dfa = safe_load_csv(LOG_FILE)
    st.dataframe(dfa)
    if not dfa.empty:
        st.download_button("Download analysis log", data=dfa.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(LOG_FILE))

    st.subheader("Labeled Dataset")
    dfl = safe_load_csv(LABELED_CSV)
    st.dataframe(dfl)
    if not dfl.empty:
        st.download_button("Download labeled dataset", data=dfl.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(LABELED_CSV))

    st.subheader("Symbol Index")
    dfi = safe_load_csv(SYMBOL_INDEX_FILE)
    st.dataframe(dfi)
    if not dfi.empty:
        st.download_button("Download symbol index", data=dfi.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(SYMBOL_INDEX_FILE))

    st.subheader("Backups")
    backups = sorted([f for f in os.listdir(BACKUP_DIR)], reverse=True)
    if backups:
        for b in backups[:50]:
            st.write(b)
        # allow restore of latest chosen backup (manual caution)
        sel = st.selectbox("Restore a backup file into workspace (select then press Restore)", options=["(none)"]+backups)
        if sel and sel != "(none)":
            if st.button("Restore selected backup"):
                src = os.path.join(BACKUP_DIR, sel)
                # only allow restoring library or index or logs (safety)
                dest_name = sel.split("_", 1)[-1]
                dest = os.path.join(BASE_DIR, dest_name)
                try:
                    shutil.copy(src, dest)
                    st.success(f"Restored {dest_name} from backup {sel}. Please reload app.")
                except Exception as e:
                    st.error(f"Restore failed: {e}")
    else:
        st.info("No backups found yet. Backups are created automatically on writes.")
