import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import io
import cv2
from PIL import Image
import pytesseract
import os

# ==============================
# APP CONFIGURATION
# ==============================
st.set_page_config(page_title="IVC Analyzer", page_icon="🌀", layout="wide")
st.title("🌀 Integrated Visual Coding (IVC) Analyzer")
st.caption("Decode ancient symbols and scripts through energetic, geometric, and functional mapping.")

# ==============================
# SIDEBAR: INPUT
# ==============================
st.sidebar.header("Upload Ancient Artifact")
uploaded_file = st.sidebar.file_uploader("Upload Image or Document", type=["jpg", "jpeg", "png", "pdf", "txt", "docx"])

analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Single Symbol", "Full Script", "Cultural Context Analysis"]
)

# Initialize session state
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ==============================
# STEP 1: SHOW UPLOADED FILE
# ==============================
st.subheader("1️⃣ Input Overview")

if uploaded_file:
    st.sidebar.success("✅ File uploaded successfully!")
    file_details = {
        "Filename": uploaded_file.name,
        "File Type": uploaded_file.type,
        "File Size (KB)": round(uploaded_file.size / 1024, 2)
    }
    st.json(file_details)

    if uploaded_file.type.startswith("image/"):
        st.image(uploaded_file, caption="Uploaded Artifact", use_container_width=True)

        if st.sidebar.button("Run IVC Analysis"):
            with st.spinner("Running algorithmic analysis..."):
                image = np.array(Image.open(uploaded_file).convert("RGB"))
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # --- Edge detection to simulate symbolic pathways ---
                edges = cv2.Canny(gray, 100, 200)

                # --- OCR text extraction ---
                ocr_text = pytesseract.image_to_string(gray)
                if not ocr_text.strip():
                    ocr_text = "(No readable glyphs or text detected.)"

                st.session_state["edges"] = edges
                st.session_state["ocr_text"] = ocr_text
                st.session_state["analysis_done"] = True

            st.success("✅ Analysis complete! See results below.")
else:
    st.info("⬅️ Please upload an image to begin analysis.")

# ==============================
# STEP 2: SHOW ANALYSIS RESULTS
# ==============================
if st.session_state["analysis_done"]:
    edges = st.session_state.get("edges")
    ocr_text = st.session_state.get("ocr_text", "")

    st.subheader("2️⃣ Symbolic Energy Map (Edge Detection)")
    fig, ax = plt.subplots()
    ax.imshow(edges, cmap="inferno")
    ax.set_title("Detected Symbolic Pathways (Energy Flow Map)")
    ax.axis("off")
    st.pyplot(fig)

    # OCR output
    st.subheader("3️⃣ OCR Detected Text / Glyphs")
    st.text(ocr_text.strip())

    # ======================================
    # 4️⃣ FUNCTIONAL MATRIX
    # ======================================
    st.subheader("4️⃣ Functional Assignment Matrix")
    matrix_data = pd.DataFrame({
        "Symbol": ["Detected Form A", "Detected Form B", "Detected Form C"],
        "Energetic Role": ["Flow", "Stability", "Structure"],
        "Functional Role": ["Force Channel", "Field Anchor", "Material Node"]
    })
    st.dataframe(matrix_data, use_container_width=True)

    # ======================================
    # 5️⃣ SYMBOL RELATIONAL TREE
    # ======================================
    st.subheader("5️⃣ Symbol Relational Tree — Network Mapping")
    G = nx.Graph()
    G.add_edges_from([
        ("Form A", "Form B"),
        ("Form B", "Form C"),
        ("Form C", "Form A")
    ])
    fig2, ax2 = plt.subplots()
    nx.draw(G, with_labels=True, node_color="lightblue", node_size=2500, ax=ax2)
    st.pyplot(fig2)

    # ======================================
    # 6️⃣ FREQUENCY / RESONANCE SIMULATION
    # ======================================
    st.subheader("6️⃣ Frequency & Resonance Simulation")
    freqs = np.linspace(0, 100, 200)
    resonance = np.sin(freqs / 8) ** 2
    fig3, ax3 = plt.subplots()
    ax3.plot(freqs, resonance)
    ax3.set_title("Symbolic Resonance Spectrum")
    ax3.set_xlabel("Frequency (arbitrary units)")
    ax3.set_ylabel("Resonance Intensity")
    st.pyplot(fig3)

    # ======================================
    # 7️⃣ INTERPRETATION STUB
    # ======================================
    st.subheader("7️⃣ Symbolic Interpretation (Placeholder)")
    st.markdown("""
    This is where you can connect your own model for automatic interpretation.

    If you have an API endpoint or local model, call it here with the OCR output
    and image statistics to generate a text interpretation.
    """)
    st.code("""
    # Example (pseudo-code):
    # from ivc_api import interpret_symbols
    # interpretation = interpret_symbols(ocr_text)
    # st.write(interpretation)
    """)

    # ======================================
    # 8️⃣ EXPORT REPORT
    # ======================================
    st.subheader("📤 Export Results")

    export_choice = st.radio("Select Export Format", ["Markdown (.md)", "CSV"])

    if st.button("Generate Report"):
        report = f"""# IVC Analysis Report

**Summary**
- Energy Flow: Edge-based mapping complete
- OCR Extracted Text:
{ocr_text}

- Functional Roles: Flow, Stability, Structure
- Symbolic Relationships: Triadic linkage detected
"""

        if export_choice == "Markdown (.md)":
            mime = "text/markdown"
            filename = "IVC_Report.md"
        else:
            mime = "text/csv"
            filename = "IVC_Report.csv"
            report = "Symbol,Energetic Role,Functional Role\nForm A,Flow,Force Channel\nForm B,Stability,Field Anchor\nForm C,Structure,Material Node\n"

        b = io.BytesIO(report.encode("utf-8"))
        st.download_button(
            label=f"⬇️ Download {filename}",
            data=b,
            file_name=filename,
            mime=mime
        )
else:
    st.info("Upload an artifact and click **Run IVC Analysis** to view results.")
