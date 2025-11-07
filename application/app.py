import streamlit as st
from src.data_processing import normalize_symbol_image
from src.semantic_mapping import predict_meanings
from src.visualization import plot_adjacency_graph, show_interface_diagram
import json, os
st.set_page_config(page_title="Indus Decoder (Prototype)", layout="wide")

st.title("Indus Decoder — Prototype Streamlit App")
st.markdown("Upload an Indus symbol image or choose an example to see decoded suggestions, adjacency graphs, and interface visuals. This is an open-source starter — extend the models in `src/`.")

# Load dictionary
DATA_DIR = os.path.join(os.path.dirname(__file__),"data")
with open(os.path.join(DATA_DIR,"dictionary.json"),"r",encoding="utf-8") as f:
    dictionary = json.load(f)

col1, col2 = st.columns([1,2])

with col1:
    st.header("Input")
    uploaded = st.file_uploader("Upload a symbol image (PNG/SVG/JPEG)", type=["png","jpg","jpeg"])
    example = st.selectbox("Or select an example symbol", ["None"] + list(dictionary.keys()))
    if st.button("Decode"):
        if uploaded is None and example=="None":
            st.warning("Upload an image or select an example symbol.")
        else:
            if uploaded:
                img_bytes = uploaded.read()
                feat = normalize_symbol_image(img_bytes)
                label, scores = predict_meanings(feat)
                st.success(f"Top suggestion: **{label}**")
                st.write("Top candidates (score):")
                st.write(scores)
            else:
                # example selected
                st.info(f"Showing dictionary entry for **{example}**")
                st.write(dictionary[example])

with col2:
    st.header("Visualizations")
    st.subheader("Adjacency Graph (prototype)")
    plot_adjacency_graph()
    st.subheader("Quantum-Holographic Interface (prototype)")
    show_interface_diagram()

st.markdown("---")
st.caption("Prototype app generated from the Indus decoding project. Extend `src/` modules to add real models and datasets.")
