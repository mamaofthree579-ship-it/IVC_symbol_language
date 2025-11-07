# app.py
# --------------------------------------------------
# Indus Symbol Language Research Framework
# Phase I–VIII with image upload + CFG induction
# --------------------------------------------------

import os
import json
import cv2
import numpy as np
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from ivc_cfg import sequitur_infer, rules_to_text, rules_to_graph_edges, export_rules_json

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------
st.set_page_config(page_title="IVC Symbol Research Framework", layout="wide")
st.title("🕉️ Indus Symbol Language – Quantum–Holographic Research Framework")

st.markdown("""
Upload ancient script images or corpus files, automatically extract sequences of symbols,
and run the **Recursive–CFG Induction** process to discover hierarchical symbol rules.
""")

# --------------------------------------------------
# Section 1 – Upload Corpus or Image
# --------------------------------------------------
st.sidebar.header("Input Options")
input_mode = st.sidebar.radio("Select input mode:", ["Upload Script Images", "Upload Sequence JSON"])

uploaded_images = None
sequences = []

if input_mode == "Upload Script Images":
    uploaded_images = st.file_uploader(
        "Upload one or more images of ancient scripts",
        type=["png", "jpg", "jpeg", "tif"],
        accept_multiple_files=True
    )
else:
    seq_file = st.file_uploader("Upload a JSON file of symbol sequences", type=["json"])
    if seq_file:
        try:
            sequences = json.load(seq_file)
            st.success(f"Loaded {len(sequences)} sequences from uploaded JSON")
        except Exception as e:
            st.error(f"Error reading JSON: {e}")

# --------------------------------------------------
# Section 2 – Image Processing → Symbol Extraction
# --------------------------------------------------
symbol_sequences = []
preview_images = []

if uploaded_images:
    st.header("🖼️ Step 1 – Script Preprocessing & Symbol Detection")

    for img_file in uploaded_images:
        bytes_data = img_file.read()
        np_img = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 80, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])  # left-to-right order
        sequence = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < 100:  # skip noise
                continue
            roi = edges[y:y + h, x:x + w]
            # Very simple "symbol label" from aspect ratio and area
            ar = round(w / h, 2)
            label = f"S{len(sequence) + 1}_AR{ar}"
            sequence.append(label)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        if len(sequence) > 0:
            symbol_sequences.append(sequence)
        preview_images.append((img_file.name, img, sequence))

    st.success(f"Extracted {len(symbol_sequences)} symbol sequences from uploaded images.")

    # Preview the results
    cols = st.columns(len(preview_images))
    for i, (name, proc_img, seq) in enumerate(preview_images):
        with cols[i]:
            st.image(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB), caption=f"{name}\nSequence: {seq}")

# --------------------------------------------------
# Section 3 – Prepare data for CFG
# --------------------------------------------------
if len(symbol_sequences) > 0:
    sequences = symbol_sequences

if not sequences:
    st.info("No sequences detected or uploaded yet.")
else:
    st.header("🧩 Step 2 – Context-Free Grammar (CFG) Induction")
    min_rule_occurrence = st.slider("Minimum rule occurrence threshold", 2, 10, 2)
    if st.button("🔎 Run CFG Induction"):
        seq_res = sequitur_infer(sequences, min_rule_occurrence=min_rule_occurrence)
        if seq_res["rules"]:
            st.subheader("Inferred Grammar Rules")
            st.code(rules_to_text(seq_res["rules"]), language="text")

            tmp_json = os.path.join(tempfile.gettempdir(), "ivc_cfg_rules.json")
            export_rules_json(tmp_json, seq_res)
            with open(tmp_json, "r", encoding="utf-8") as f:
                st.download_button("📥 Download CFG JSON",
                                   data=f.read(),
                                   file_name="ivc_cfg_rules.json",
                                   mime="application/json")

            # Visualization
            edges = rules_to_graph_edges(seq_res["rules"])
            if edges:
                G = nx.DiGraph()
                G.add_edges_from(edges)
                fig, ax = plt.subplots(figsize=(7, 5))
                pos = nx.spring_layout(G, k=0.6, iterations=40)
                nx.draw(G, pos, with_labels=True, node_size=800,
                        node_color="lightgray", font_size=9, arrows=True, ax=ax)
                ax.set_title("CFG Rule Hierarchy Graph")
                st.pyplot(fig)
        else:
            st.warning("No significant grammar rules detected. Try lowering the threshold.")

# --------------------------------------------------
# Section 4 – Next Phase Summaries
# --------------------------------------------------
st.header("📘 Phase III–VIII Overview")
st.markdown("""
#### Phase III – Symbolic–Functional Correlation
Cross-cultural mapping and semantic clustering (e.g., **Fish → Consciousness**, **Jar → Matter**).

#### Phase IV – Resonance & Cognitive Mapping
EEG / harmonic overlays connecting symbol geometry to frequency response.

#### Phase V – Algorithmic Decoding
Weighted integration of structure, energy, and meaning layers.

#### Phase VI – Quantum–Holographic Interface
Symbolic functions as operational instructions within a unified field code.

#### Phase VII – Cross-System Validation
Applying IVC decoding to **Rongorongo**, **Proto-Elamite**, **Olmec** for shared principles.

#### Phase VIII – Synthesis
Generates unified **Symbol Dictionary**, **Resonance Maps**, and **Field Interaction Diagrams**.
""")
