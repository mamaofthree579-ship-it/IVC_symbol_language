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

# Import the IVC translator module
from ivc_translator import ivc_translate

# ====================================
# STREAMLIT PAGE CONFIG (MOBILE FRIENDLY)
# ====================================
st.set_page_config(
    page_title="IVC Analyzer",
    page_icon="🌀",
    layout="centered",  # phone-optimized
    initial_sidebar_state="collapsed"
)

# Add mobile-friendly CSS
st.markdown("""
<style>
button, .stTextInput, .stSelectbox, .stDownloadButton {
    font-size: 18px !important;
}
.stImage { text-align: center; }
</style>
""", unsafe_allow_html=True)

# ====================================
# APP HEADER
# ====================================
st.title("🌀 Integrated Visual Coding (IVC) Analyzer")
st.caption("Decode ancient symbols and scripts through energetic, geometric, and functional mapping — optimized for mobile.")

# ====================================
# UPLOAD SECTION
# ====================================
st.markdown("### 📤 Upload Artifact")
uploaded_file = st.file_uploader("Upload an image of an ancient artifact", type=["jpg", "jpeg", "png"])

analysis_type = st.selectbox(
    "Select Analysis Type",
    ["Single Symbol", "Full Script", "Cultural Context Analysis"]
)

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ====================================
# FILE UPLOAD AND ANALYSIS
# ====================================
if uploaded_file:
    st.success("✅ File uploaded successfully!")
    file_details = {
        "Filename": uploaded_file.name,
        "File Type": uploaded_file.type,
        "File Size (KB)": round(uploaded_file.size / 1024, 2)
    }
    st.json(file_details)

    st.image(uploaded_file, caption="Uploaded Artifact", use_container_width=True)

    if st.button("▶️ Run IVC Analysis", use_container_width=True):
        with st.spinner("Running algorithmic analysis..."):
            image = np.array(Image.open(uploaded_file).convert("RGB"))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # --- Edge detection ---
            edges = cv2.Canny(gray, 100, 200)

            # --- OCR extraction ---
            try:
                ocr_text = pytesseract.image_to_string(gray)
            except pytesseract.TesseractNotFoundError:
                ocr_text = "(⚠️ Tesseract not found on this device. Please install it to enable OCR.)"

            if not ocr_text.strip():
                ocr_text = "(No readable glyphs or text detected.)"

            # Dummy symbolic detections (you can later connect to a detector)
            detected_shapes = ["spiral", "arrow"]
            detected_patterns = ["spiral-arrow", "triad"]
            detected_freqs = [12.3, 14.8, 13.2]

            symbol_data = {
                "shapes": detected_shapes,
                "patterns": detected_patterns,
                "frequencies": detected_freqs
            }

            # --- Run IVC translation ---
            translation_output = ivc_translate(symbol_data, ocr_text)

            # Save to session
            st.session_state["edges"] = edges
            st.session_state["ocr_text"] = ocr_text
            st.session_state["symbol_data"] = symbol_data
            st.session_state["translation_output"] = translation_output
            st.session_state["analysis_done"] = True

        st.success("✅ Analysis complete! See results below.")
else:
    st.info("📸 Upload an image to begin your IVC analysis.")

# ====================================
# RESULTS DISPLAY
# ====================================
if st.session_state["analysis_done"]:
    edges = st.session_state.get("edges")
    ocr_text = st.session_state.get("ocr_text", "")
    translation_output = st.session_state.get("translation_output", "")

    # TABS FOR NAVIGATION
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Energy Map", "Functional Matrix", "Resonance", "Translation", "Report"])

    # ---------------------------
    # TAB 1: ENERGY MAP
    # ---------------------------
    with tab1:
        st.markdown("### 🌀 Symbolic Energy Map")
        fig, ax = plt.subplots()
        ax.imshow(edges, cmap="inferno")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

        st.markdown("### 🔠 OCR Detected Text")
        st.text(ocr_text.strip())

    # ---------------------------
    # TAB 2: FUNCTIONAL MATRIX
    # ---------------------------
    with tab2:
        st.markdown("### ⚙️ Functional Assignment Matrix")

        matrix_data = pd.DataFrame({
            "Symbol": ["Detected Form A", "Detected Form B", "Detected Form C"],
            "Energetic Role": ["Flow", "Stability", "Structure"],
            "Functional Role": ["Force Channel", "Field Anchor", "Material Node"]
        })

        search = st.text_input("🔍 Search symbols")
        if search:
            filtered = matrix_data[matrix_data["Symbol"].str.contains(search, case=False)]
            st.dataframe(filtered, use_container_width=True)
        else:
            st.dataframe(matrix_data, use_container_width=True)

        with st.expander("🕸️ Show Symbolic Relational Tree"):
            G = nx.Graph()
            G.add_edges_from([
                ("Form A", "Form B"),
                ("Form B", "Form C"),
                ("Form C", "Form A")
            ])
            fig2, ax2 = plt.subplots()
            nx.draw(G, with_labels=True, node_color="lightblue", node_size=2500, ax=ax2)
            st.pyplot(fig2, use_container_width=True)

    # ---------------------------
    # TAB 3: RESONANCE SIMULATION
    # ---------------------------
    with tab3:
        st.markdown("### 🌐 Frequency & Resonance Spectrum")
        freqs = np.linspace(0, 100, 200)
        resonance = np.sin(freqs / 8) ** 2
        fig3, ax3 = plt.subplots()
        ax3.plot(freqs, resonance)
        ax3.set_xlabel("Frequency (a.u.)")
        ax3.set_ylabel("Resonance Intensity")
        st.pyplot(fig3, use_container_width=True)

    # ---------------------------
    # TAB 4: TRANSLATION (NEW)
    # ---------------------------
    with tab4:
        st.markdown("### 🧬 IVC Translation Output")
        st.markdown(translation_output)

    # ---------------------------
    # TAB 5: REPORT EXPORT
    # ---------------------------
    with tab5:
        st.markdown("### 📤 Export Analysis Report")
        export_choice = st.radio("Select Export Format", ["Markdown (.md)", "CSV"])

        if st.button("Generate Report", use_container_width=True):
            report = f"""# IVC Analysis Report

**Summary**
- Energy Flow Mapping Complete
- OCR Extracted Text:
{ocr_text}

**IVC Translation:**
{translation_output}

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
                mime=mime,
                use_container_width=True
            )
else:
    st.info("⬆️ Upload an artifact and tap **Run IVC Analysis** to view results.")
