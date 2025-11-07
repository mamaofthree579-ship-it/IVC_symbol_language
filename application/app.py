# app.py
# IVC Symbol Research Studio — Full-featured with Classifier Builder, Polygon Annotator, Template Augmentation
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

# try optional imports
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    from tensorflow.keras import layers, models
except Exception:
    TF_AVAILABLE = False

import streamlit as st
from ivc_framework import (
    run_full_pipeline_on_folder,
    plot_fractal_boxcount,
    plot_radial_signature,
    plot_adjacency_graph,
)
import os
import numpy as np
import tempfile
import json

# ───────────────────────────────────────────────
# 🧠 Research Framework Tab
# ───────────────────────────────────────────────
def research_framework_tab():
    st.title("📚 Indus Script Research Framework")

    st.markdown("""
    ### Overview
    This module implements Phases I–VIII of the IVC Symbol Analysis Framework.
    Upload or select a folder of glyph images, then run the full analysis pipeline.
    """)

    # --- Folder input
    st.markdown("#### Step 1: Select or upload glyph folder")
    glyph_folder = st.text_input(
        "Enter path to glyph image folder:",
        value="data/glyphs",
        help="Folder containing .png, .jpg, or .tif images of symbols",
    )

    uploaded_files = st.file_uploader(
        "Or upload one or more glyph images", type=["png", "jpg", "jpeg", "tif"], accept_multiple_files=True
    )

    # Save uploaded images to temp dir
    tmp_dir = None
    if uploaded_files:
        tmp_dir = tempfile.mkdtemp()
        for file in uploaded_files:
            with open(os.path.join(tmp_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
        glyph_folder = tmp_dir
        st.success(f"Uploaded {len(uploaded_files)} files → using temp folder {tmp_dir}")

    # --- Run analysis
    if st.button("🚀 Run Full Research Pipeline"):
        if not os.path.exists(glyph_folder):
            st.error(f"Folder not found: {glyph_folder}")
            return

        with st.spinner("Running multi-phase analysis... please wait"):
            res = run_full_pipeline_on_folder(glyph_folder, n_clusters=6)

        st.success("✅ Analysis complete!")

        # ───── Display Phase I outputs
        st.subheader("**Phase I – Vectorization & Normalization**")
        st.write(f"Processed {len(res['vector_results']['vectors'])} glyphs.")
        st.json(
            {k: {"centroid": v.get("centroid"), "signature_len": len(v["signature_vector"])}
             for k, v in list(res["vector_results"]["vectors"].items())[:5]}
        )

        # ───── Display Phase II outputs
        st.subheader("**Phase II – Fractal / Recursive Structure**")
        for i, (p, info) in enumerate(list(res["fractal_dims"].items())[:3]):
            st.write(f"Glyph: `{os.path.basename(p)}` → Fractal D = {info['D']:.3f}")
            if st.checkbox(f"Show box-count plot for {os.path.basename(p)}", key=f"frc{i}"):
                sizes = np.array(info["sizes"])
                counts = np.array(info["counts"])
                plot_fractal_boxcount(sizes, counts)

        # ───── Display Phase III outputs
        st.subheader("**Phase III – Clustering & Semantic Fields**")
        clust = res.get("clusters", {})
        if "labels" in clust and len(clust["labels"]) > 0:
            unique_labels = np.unique(clust["labels"])
            st.write(f"Found {len(unique_labels)} clusters.")
            st.bar_chart(np.bincount(clust["labels"]))
        else:
            st.warning("No clustering results available.")

        # ───── Display Phase IV (placeholder)
        st.subheader("**Phase IV – Harmonic Resonance Mapping**")
        st.markdown("You can run individual glyph FFT harmonic analysis below:")
        files = list(res["vector_results"]["vectors"].keys())
        if files:
            selected = st.selectbox("Choose glyph for harmonic analysis", files)
            from ivc_framework import radial_signature_fft
            img = res["vector_results"]["vectors"][selected]["edges"]
            fft_res = radial_signature_fft(img)
            if len(fft_res["fft_freqs"]) > 0:
                plot_radial_signature(
                    fft_res["angles"], fft_res["radii"], fft_res["fft_freqs"], fft_res["fft_mag"]
                )
                st.json(fft_res["dominant"])

        # ───── Save summary file
        out_summary = {
            "timestamp": str(os.path.getmtime(glyph_folder)),
            "num_glyphs": len(res["vector_results"]["vectors"]),
            "num_clusters": int(len(res.get("clusters", {}).get("labels", []))),
            "avg_fractal_D": float(np.mean([v["D"] for v in res["fractal_dims"].values()] or [0])),
        }
        tmp_json = os.path.join(tempfile.gettempdir(), "ivc_summary.json")
        with open(tmp_json, "w") as f:
            json.dump(out_summary, f, indent=2)

        st.download_button(
            "📥 Download summary JSON",
            data=json.dumps(out_summary, indent=2),
            file_name="ivc_summary.json",
            mime="application/json",
        )

    else:
        st.info("Select or upload images, then click **Run Full Research Pipeline** to begin.")


# ───────────────────────────────────────────────
# Integrate into your main tab layout
# ───────────────────────────────────────────────
def main():
    tabs = ["Catalog", "Drawing", "Research Framework"]
    choice = st.sidebar.radio("Navigation", tabs)

    if choice == "Research Framework":
        research_framework_tab()
    elif choice == "Catalog":
        st.write("🗂 Catalog view placeholder")
    elif choice == "Drawing":
        st.write("✏️ Drawing tools placeholder")
    else:
        st.write("Welcome to the Indus Research Application!")


if __name__ == "__main__":
    main()

# -----------------------
# Config & files
# -----------------------
st.set_page_config(page_title="IVC Symbol Research Studio — Full", layout="wide")
st.title("🌀 IVC Symbol Research Studio — Classifier Builder, Polygon Annotator, Augmentation")

BASE_DIR = os.getcwd()
SYMBOL_LIB_FILE = os.path.join(BASE_DIR, "ivc_symbol_library.json")
SYMBOL_INDEX_FILE = os.path.join(BASE_DIR, "ivc_symbol_index.csv")
SYMBOLS_DIR = os.path.join(BASE_DIR, "symbols")
LOG_FILE = os.path.join(BASE_DIR, "ivc_symbol_log.csv")
LABELED_CSV = os.path.join(BASE_DIR, "ivc_labeled_dataset.csv")
BACKUP_DIR = os.path.join(BASE_DIR, "backups")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(SYMBOLS_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Defaults & load library
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
# ORB matching + augmentation helper
# -----------------------
def augment_template(img_bgr: np.ndarray, rotations: List[int]=[0, 90, 180, 270], scales: List[float]=[0.8,1.0,1.2]) -> List[np.ndarray]:
    """Return augmented templates by rotation and scale."""
    out = []
    h,w = img_bgr.shape[:2]
    for s in scales:
        sw, sh = int(w * s), int(h * s)
        resized = cv2.resize(img_bgr, (max(1,sw), max(1,sh)), interpolation=cv2.INTER_AREA)
        for r in rotations:
            M = cv2.getRotationMatrix2D((resized.shape[1]//2, resized.shape[0]//2), r, 1.0)
            rot = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            out.append(rot)
    return out

def run_feature_matching(img: np.ndarray, templates: List[np.ndarray], threshold: float = 0.12):
    # ORB matcher
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
            matches = bf.knnMatch(des2, des1, k=2)
            good = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            score = len(good) / float(max(1, len(kp2)))
            entry = {"template_idx": ti, "score": float(score), "good": len(good), "template_kp": len(kp2)}
            if len(good) >= 4:
                src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h2, w2 = tgray.shape[:2]
                    pts = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
                    dst = cv2.perspectiveTransform(pts, M)
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
    filtered = [r for r in results if r["score"] >= threshold]
    filtered_sorted = sorted(filtered, key=lambda x: -x["score"])
    return filtered_sorted, results

# -----------------------
# Classifier training helper (TensorFlow transfer learning)
# -----------------------
def prepare_training_data_from_labeled_csv(labeled_csv_path: str, output_dir: str, symbols_dir: str, crop_limit_per_label: int = 100):
    """
    Reads labeled CSV with columns [image_file,bbox,label,...] and crops referenced images into subfolders under output_dir/label/.
    bbox format: x1,y1,x2,y2 or x,y,w,h (we'll detect two variants).
    Returns (num_classes, class_names)
    """
    if not os.path.exists(labeled_csv_path):
        raise FileNotFoundError("Labeled CSV not found.")
    df = safe_load_csv(labeled_csv_path)
    if df.empty:
        raise ValueError("No labeled rows found in CSV.")
    os.makedirs(output_dir, exist_ok=True)
    class_counts = {}
    for _, row in df.iterrows():
        image_file = row.get("image_file") or row.get("file") or ""
        bbox_str = row.get("bbox", "")
        label = row.get("label", "")
        if not image_file or not bbox_str or not label:
            continue
        # attempt to find image path either absolute or in current dir
        if os.path.exists(image_file):
            img_path = image_file
        else:
            # try in BASE_DIR
            img_path = os.path.join(BASE_DIR, image_file)
            if not os.path.exists(img_path):
                # try file name in symbols dir
                possible = [os.path.join(BASE_DIR, p) for p in os.listdir(BASE_DIR) if os.path.basename(p) == image_file]
                img_path = possible[0] if possible else None
            if not img_path or not os.path.exists(img_path):
                continue
        try:
            xys = [int(float(x)) for x in bbox_str.split(",")]
        except Exception:
            continue
        # support two formats
        if len(xys) == 4:
            x1,y1,x2,y2 = xys
        elif len(xys) == 4:
            x1,y1,x2,y2 = xys
        else:
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h,w = img.shape[:2]
        x1 = max(0, min(w-1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = img[y1:y2, x1:x2]
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        curr_count = len(os.listdir(label_dir))
        if curr_count >= crop_limit_per_label:
            continue
        # save crop resized to 224x224
        crop_resized = cv2.resize(crop, (224,224), interpolation=cv2.INTER_AREA)
        fname = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        cv2.imwrite(os.path.join(label_dir, fname), crop_resized)
        class_counts[label] = class_counts.get(label, 0) + 1
    class_names = list(class_counts.keys())
    return len(class_names), class_names

def build_and_train_model(train_dir: str, epochs: int = 10, batch_size: int = 16, out_model_path: str = None):
    """
    Uses tf.keras with MobileNetV2 transfer learning.
    train_dir expected to have subfolders per class name.
    """
    if not TF_AVAILABLE:
        raise EnvironmentError("TensorFlow not available. Install tensorflow to train.")
    if out_model_path is None:
        out_model_path = os.path.join(MODEL_DIR, "ivc_classifier.h5")
    img_size = (160,160)  # MobileNetV2 default-ish
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, image_size=img_size, batch_size=batch_size, label_mode='categorical', shuffle=True)
    class_names = train_ds.class_names
    num_classes = len(class_names)
    # simple transfer learning
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False
    inputs = tf.keras.Input(shape=img_size + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_ds, epochs=epochs)
    model.save(out_model_path)
    return out_model_path, class_names, history.history

def load_trained_model(path: str):
    if not TF_AVAILABLE:
        raise EnvironmentError("TensorFlow not available.")
    if not os.path.exists(path):
        raise FileNotFoundError("Model file not found.")
    model = tf.keras.models.load_model(path)
    return model

def predict_on_crop(model, crop_bgr: np.ndarray, class_names: List[str]):
    # expects 160x160
    img = cv2.resize(crop_bgr, (160,160))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    arr = np.expand_dims(img_rgb / 255.0, axis=0)
    pred = model.predict(arr)[0]
    idx = int(np.argmax(pred))
    return class_names[idx], float(pred[idx])

# -----------------------
# UI: Tabs
# -----------------------
tabs = st.tabs(["🔍 Analyze", "🖋 Symbol Creator / Catalog", "✍️ Manual Label / Annotate", "🧭 Polygon Annotator", "🧪 Classifier Builder", "📘 Library Editor", "📂 Logs & Backups"])

# -----------------------
# TAB: Analyze
# -----------------------
with tabs[0]:
    st.header("Analyze — Energy Lines & Field Flow")
    uploaded = st.file_uploader("Upload artifact image", type=["jpg","jpeg","png"], key="analyze_upload")
    model_path = os.path.join(MODEL_DIR, "ivc_classifier.h5")
    model_loaded = None
    class_names = []
    if TF_AVAILABLE and os.path.exists(model_path):
        try:
            model_loaded = load_trained_model(model_path)
            # class names saved as sidecar
            cn_path = os.path.join(MODEL_DIR, "class_names.json")
            if os.path.exists(cn_path):
                with open(cn_path, "r", encoding="utf-8") as f:
                    class_names = json.load(f)
        except Exception as e:
            st.warning(f"Could not load model: {e}")
    if uploaded:
        img = read_image_bytes(uploaded)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Artifact", use_column_width=True)
        gray = preprocess_gray(img)
        edges = detect_edges(gray)

        st.subheader("1️⃣ Energy Lines (Canny / inferno)")
        fig = plot_energy_lines(edges)
        st.pyplot(fig)

        st.subheader("2️⃣ Field Flow Overlay")
        flow = field_flow_overlay_cv(img)
        st.image(cv2.cvtColor(flow, cv2.COLOR_BGR2RGB), use_column_width=True)

        st.subheader("3️⃣ OCR")
        ocr_text, ocr_data = run_ocr(img)
        st.code(ocr_text if ocr_text else "[No OCR text detected]")

        # Optionally run classifier on detected candidate regions
        if model_loaded is not None:
            st.subheader("4️⃣ Classifier suggestions (top detected contours)")
            # simple contour candidates
            cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            h,w = img.shape[:2]
            min_area = max(60, (h*w)//15000)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x,y,ww,hh = cv2.boundingRect(c)
                if ww < 8 or hh < 8:
                    continue
                crop = img[y:y+hh, x:x+ww]
                label, prob = predict_on_crop(model_loaded, crop, class_names)
                st.write(f"- Region at ({x},{y},{ww},{hh}) => {label} ({prob:.2f})")
        else:
            st.info("No trained classifier found. Use Classifier Builder tab to train one.")

        # basic OCR-based library hints
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
    st.markdown("Draw a symbol (canvas) or upload an image, add metadata, and save to the library. Gallery shows saved symbols in a grid.")
    col1, col2 = st.columns([2,1])
    with col1:
        drawn_image = None
        if CANVAS_AVAILABLE:
            st.info("Drawable canvas available.")
            bg_file = st.file_uploader("Optional: upload background to trace", type=["jpg","png","jpeg"], key="sym_bg")
            bg_img = None
            if bg_file:
                bg_pil = Image.open(BytesIO(bg_file.read())).convert("RGBA")
                bg_img = bg_pil
            canvas_result = st_canvas(
                fill_color="rgba(0,0,0,0)",
                stroke_width=3,
                stroke_color="#000000",
                background_image=bg_img,
                height=400,
                width=400,
                drawing_mode="freedraw",
                key="symbol_canvas"
            )
            if canvas_result and canvas_result.image_data is not None:
                im = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA").convert("RGB")
                drawn_image = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                st.image(cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB), caption="Canvas preview", width=240)
        else:
            st.warning("Canvas not available. Install streamlit-drawable-canvas.")

        uploaded_sym = st.file_uploader("Or upload symbol image", type=["png","jpg","jpeg"], key="sym_upload")
        uploaded_sym_img = None
        if uploaded_sym:
            uploaded_sym_img = read_image_bytes(uploaded_sym)
            st.image(cv2.cvtColor(uploaded_sym_img, cv2.COLOR_BGR2RGB), caption="Uploaded symbol preview", width=240)

    with col2:
        name = st.text_input("Display name")
        suggested_id = name.strip().lower().replace(" ", "_") if name else ""
        st.text_input("Symbol ID (auto-normalized)", value=suggested_id, key="symbol_id_display")
        core = st.text_area("Core meaning", height=80)
        domain = st.text_input("Functional domain")
        notes = st.text_area("Notes / parallels", height=120)
        if st.button("Save Symbol"):
            src_img = None
            if drawn_image is not None:
                src_img = drawn_image
            elif uploaded_sym_img is not None:
                src_img = uploaded_sym_img
            if src_img is None:
                st.error("Please draw or upload a symbol image first.")
            else:
                if not name:
                    st.error("Please enter a display name.")
                else:
                    sid = suggested_id or normalize_id(name)
                    if sid in SYMBOL_LIB:
                        sid = f"{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    # update library and save image
                    SYMBOL_LIB[sid] = {"display_name": name, "core": core, "domain": domain, "notes": notes, "image_path": os.path.join("symbols", f"{sid}.png")}
                    backup_file(SYMBOL_LIB_FILE)
                    with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
                        json.dump(SYMBOL_LIB, f, indent=2)
                    path = os.path.join(SYMBOLS_DIR, f"{sid}.png")
                    save_cv2_image(src_img, path)
                    row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                           "symbol_id": sid, "display_name": name, "core": core, "domain": domain, "notes": notes, "image_path": path}
                    append_csv_row(SYMBOL_INDEX_FILE, row)
                    st.success(f"Saved symbol {name} as {sid}")

    st.markdown("---")
    st.subheader("Gallery (grid)")
    idx_df = safe_load_csv(SYMBOL_INDEX_FILE)
    gallery_items = []
    if not idx_df.empty:
        for _, r in idx_df.iterrows():
            p = r.get("image_path","")
            if p and os.path.exists(p):
                gallery_items.append({"id": r.get("symbol_id",""), "name": r.get("display_name",""), "path": p, "domain": r.get("domain",""), "core": r.get("core","")})
    else:
        for sid, info in SYMBOL_LIB.items():
            p = os.path.join(SYMBOLS_DIR, f"{sid}.png")
            if os.path.exists(p):
                gallery_items.append({"id": sid, "name": info.get("display_name", sid), "path": p, "domain": info.get("domain",""), "core": info.get("core","")})
    cols = st.columns(4)
    for i, item in enumerate(gallery_items):
        with cols[i % 4]:
            try:
                im = Image.open(item["path"])
                st.image(im, caption=f"{item['name']}\n({item['id']})", use_column_width=True)
                st.caption(item.get("domain",""))
            except Exception:
                st.write(item["name"])

# -----------------------
# TAB: Manual Label / Annotate (existing)
# -----------------------
with tabs[2]:
    st.header("Manual Labeling & Annotation")
    up = st.file_uploader("Upload image to annotate", type=["jpg","jpeg","png"], key="man_label")
    if up:
        img = read_image_bytes(up)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Image for annotation", use_column_width=True)
        gray = preprocess_gray(img)
        edges = detect_edges(gray)
        st.subheader("Detected energy lines")
        st.pyplot(plot_energy_lines(edges))
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
            for i, cinfo in enumerate(candidates[:30]):
                with cols[i%3]:
                    st.image(cv2.cvtColor(cinfo["crop"], cv2.COLOR_BGR2RGB), use_column_width=True)
                    label = st.selectbox(f"Label region #{i}", options=["(none)"]+list(SYMBOL_LIB.keys()), key=f"lab_{i}")
                    note = st.text_input(f"Note #{i}", key=f"note_{i}")
                    if st.button(f"Save region #{i}", key=f"save_reg_{i}"):
                        if label == "(none)":
                            st.warning("Choose a label before saving")
                        else:
                            x1,y1,x2,y2 = cinfo["bbox"]
                            row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "image_file": getattr(up, "name", "uploaded"),
                                   "bbox": f"{x1},{y1},{x2},{y2}",
                                   "label": label,
                                   "note": note}
                            append_csv_row(LABELED_CSV, row)
                            st.success(f"Saved {label} for region #{i}")

# -----------------------
# TAB: Polygon Annotator (pixel-accurate masks)
# -----------------------
with tabs[3]:
    st.header("🎯 Polygon Annotator (create pixel masks)")
    st.markdown("Use the canvas polygon tool to create a mask for a symbol region. Save mask PNG and optionally create a labeled CSV entry.")
    upm = st.file_uploader("Upload image to annotate mask", type=["jpg","jpeg","png"], key="poly_upload")
    if upm:
        base_img = Image.open(BytesIO(upm.read())).convert("RGBA")
        st.image(base_img, caption="Base image (for reference)", use_column_width=True)
        if not CANVAS_AVAILABLE:
            st.warning("Install streamlit-drawable-canvas to use polygon drawing: pip install streamlit-drawable-canvas")
        else:
            canvas_result = st_canvas(
                fill_color="rgba(255,0,0,0.5)",
                stroke_width=2,
                stroke_color="#ff0000",
                background_image=base_img,
                height=400,
                width=600,
                drawing_mode="polygon",
                key="poly_canvas"
            )
            if canvas_result and canvas_result.image_data is not None:
                im = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA").convert("RGB")
                st.image(im, caption="Canvas result (mask + image)", use_column_width=True)
                if st.button("Save mask as PNG"):
                    # convert alpha/drawn areas to mask
                    arr = np.array(canvas_result.image_data)  # RGBA
                    alpha = arr[...,3]
                    mask = (alpha > 10).astype(np.uint8) * 255
                    mask_img = Image.fromarray(mask)
                    mask_name = f"mask_{os.path.splitext(upm.name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    mask_path = os.path.join(SYMBOLS_DIR, mask_name)
                    mask_img.save(mask_path)
                    st.success(f"Saved mask to {mask_path}")
                    # optional: offer to save a labeled CSV row referencing this mask
                    label = st.selectbox("Label this masked region", options=["(none)"]+list(SYMBOL_LIB.keys()))
                    if st.button("Save labeled mask entry"):
                        if label == "(none)":
                            st.warning("Choose a label first")
                        else:
                            row = {"timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "image_file": upm.name,
                                   "mask_path": mask_path,
                                   "label": label}
                            append_csv_row(LABELED_CSV, row)
                            st.success("Saved labeled mask entry")

# -----------------------
# TAB: Classifier Builder (train / export)
# -----------------------
with tabs[4]:
    st.header("🧪 Classifier Builder (from labeled dataset)")
    st.markdown("This builds a small transfer-learning classifier from `ivc_labeled_dataset.csv`. It crops labeled regions into a training folder and trains MobileNetV2.")
    st.markdown("**Warning:** Training on CPU can be slow. For best results use a GPU environment.")

    st.subheader("1) Prepare dataset from labeled CSV")
    train_prep_dir = st.text_input("Training output folder", value=os.path.join(BASE_DIR, "train_data"))
    limit_per_label = st.number_input("Max crops per label", value=200, min_value=10, max_value=5000)
    if st.button("Prepare training data"):
        try:
            n_classes, class_names = prepare_training_data_from_labeled_csv(LABELED_CSV, train_prep_dir, SYMBOLS_DIR, crop_limit_per_label=int(limit_per_label))
            st.success(f"Prepared training data for {n_classes} classes: {class_names}")
        except Exception as e:
            st.error(f"Failed to prepare training data: {e}")

    st.subheader("2) Train model (MobileNetV2 transfer learning)")
    epochs = st.number_input("Epochs", value=8, min_value=1, max_value=200)
    batch = st.number_input("Batch size", value=16, min_value=4, max_value=128)
    model_name = st.text_input("Model filename (saved to models/)", value="ivc_classifier.h5")
    if st.button("Train model"):
        if not TF_AVAILABLE:
            st.error("TensorFlow not available. Install tensorflow to train.")
        else:
            try:
                model_out = os.path.join(MODEL_DIR, model_name)
                out_model_path, class_names, history = build_and_train_model(train_prep_dir, epochs=int(epochs), batch_size=int(batch), out_model_path=model_out)
                # save class names alongside model
                with open(os.path.join(MODEL_DIR, "class_names.json"), "w", encoding="utf-8") as f:
                    json.dump(class_names, f)
                st.success(f"Trained model saved to {out_model_path}. Classes: {class_names}")
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.subheader("3) Quick inference test (use a crop)")
    model_choice = st.file_uploader("Upload model .h5 for quick test (optional)", type=["h5"])
    test_img = st.file_uploader("Upload crop to test", type=["jpg","jpeg","png"])
    if st.button("Run quick test"):
        try:
            model_to_load = None
            if model_choice:
                # save temp and load
                tmp_path = os.path.join(MODEL_DIR, model_choice.name)
                with open(tmp_path, "wb") as f:
                    f.write(model_choice.read())
                model_to_load = load_trained_model(tmp_path)
                if os.path.exists(os.path.join(MODEL_DIR, "class_names.json")):
                    with open(os.path.join(MODEL_DIR, "class_names.json"), "r", encoding="utf-8") as f:
                        cn = json.load(f)
                else:
                    cn = []
            elif os.path.exists(os.path.join(MODEL_DIR, "ivc_classifier.h5")):
                model_to_load = load_trained_model(os.path.join(MODEL_DIR, "ivc_classifier.h5"))
                if os.path.exists(os.path.join(MODEL_DIR, "class_names.json")):
                    with open(os.path.join(MODEL_DIR, "class_names.json"), "r", encoding="utf-8") as f:
                        cn = json.load(f)
                else:
                    cn = []
            else:
                st.error("No model available")
                model_to_load = None
                cn = []
            if model_to_load and test_img:
                timg = read_image_bytes(test_img)
                lab, p = predict_on_crop(model_to_load, timg, cn)
                st.write(f"Predicted: {lab} ({p:.3f})")
        except Exception as e:
            st.error(f"Quick test failed: {e}")

# -----------------------
# TAB: Library Editor & Logs
# -----------------------
with tabs[5]:
    st.header("Symbol Library Editor")
    st.markdown("Edit the JSON for the symbol library. Backup is created automatically.")
    lib_text = json.dumps(SYMBOL_LIB, indent=2)
    edited = st.text_area("Edit library JSON", value=lib_text, height=400)
    if st.button("Save library JSON"):
        try:
            new_lib = json.loads(edited)
            backup_file(SYMBOL_LIB_FILE)
            with open(SYMBOL_LIB_FILE, "w", encoding="utf-8") as f:
                json.dump(new_lib, f, indent=2)
            st.success("Library saved (backup created).")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

with tabs[6]:
    st.header("Logs & Backups")
    st.subheader("Analysis Log")
    dfa = safe_load_csv(LOG_FILE)
    st.dataframe(dfa)
    if not dfa.empty:
        st.download_button("Download analysis log", data=dfa.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(LOG_FILE))
    st.subheader("Labeled dataset")
    dfl = safe_load_csv(LABELED_CSV)
    st.dataframe(dfl)
    if not dfl.empty:
        st.download_button("Download labeled dataset", data=dfl.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(LABELED_CSV))
    st.subheader("Symbol index")
    dfi = safe_load_csv(SYMBOL_INDEX_FILE)
    st.dataframe(dfi)
    if not dfi.empty:
        st.download_button("Download symbol index", data=dfi.to_csv(index=False).encode("utf-8"), file_name=os.path.basename(SYMBOL_INDEX_FILE))
    st.subheader("Backups folder")
    backups = sorted([f for f in os.listdir(BACKUP_DIR)], reverse=True)
    if backups:
        st.write("\n".join(backups[:100]))
        sel = st.selectbox("Select backup to restore", options=["(none)"] + backups)
        if sel and sel != "(none)":
            if st.button("Restore selected backup"):
                src = os.path.join(BACKUP_DIR, sel)
                dest_name = sel.split("_",1)[-1]
                dest = os.path.join(BASE_DIR, dest_name)
                try:
                    shutil.copy(src, dest)
                    st.success(f"Restored {dest_name}. Reload app to pick up changes.")
                except Exception as e:
                    st.error(f"Restore failed: {e}")
    else:
        st.info("No backups yet. Writes will create backups automatically.")

# -----------------------
# Utilities
# -----------------------
def normalize_id(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

# Expose helper functions used by Classifier Builder
def prepare_training_data_from_labeled_csv(labeled_csv_path: str, output_dir: str, symbols_dir: str, crop_limit_per_label: int = 100):
    # included earlier in file; reuse by calling into this module's function (duplicate safe)
    # For simplicity, call local wrapper above (same name)
    return prepare_training_data_from_labeled_csv.__wrapped__(labeled_csv_path, output_dir, symbols_dir, crop_limit_per_label) if hasattr(prepare_training_data_from_labeled_csv, "__wrapped__") else (_prepare_training_data_from_labeled_csv(labeled_csv_path, output_dir, symbols_dir, crop_limit_per_label))

# If TensorFlow is not available, inform user which features are disabled
if not TF_AVAILABLE:
    st.sidebar.warning("TensorFlow not installed. Classifier training/inference disabled. Install tensorflow to enable.")
if not CANVAS_AVAILABLE:
    st.sidebar.info("streamlit-drawable-canvas not installed. Drawing & polygon tools disabled.")

# end of file
