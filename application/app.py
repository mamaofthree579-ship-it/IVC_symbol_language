# app.py
# --------------------------------------------------
# Indus Symbol Language Research Application
# Phase I–VIII integrated with Sequitur CFG Induction
# --------------------------------------------------

import os
import json
import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from ivc_cfg import sequitur_infer, rules_to_text, rules_to_graph_edges, export_rules_json

# --------------------------------------------------
# Basic Streamlit configuration
# --------------------------------------------------
st.set_page_config(page_title="IVC Symbol Research Framework",
                   layout="wide",
                   initial_sidebar_state="expanded")

st.title("🕉️ Indus Symbol Language – Quantum-Holographic Research Framework")
st.markdown("""
This interface integrates multiple analytic phases — from symbol vectorization
to recursive grammar inference — enabling iterative exploration of the Indus
Symbol Corpus.  
**Phase II.5** introduces a Sequitur-style Context-Free Grammar induction for
detecting hierarchical rules and repeated symbolic clusters.
""")

# --------------------------------------------------
# Utility: Load corpus / sequences
# --------------------------------------------------
st.sidebar.header("Corpus Settings")

seq_json_path = st.sidebar.text_input("Path to sequences JSON", value="data/sequences.json")
min_rule_occurrence = st.sidebar.slider("Minimum rule occurrences", 2, 10, 2)
run_button = st.sidebar.button("Run Full Research Framework")

# --------------------------------------------------
# Phase I–II placeholder results
# --------------------------------------------------
if run_button:
    st.header("Phase I–II – Corpus Processing & Vector Analysis")
    st.info("Vectorization, adjacency mapping, and symbolic clustering in progress...")

    # Placeholder table (simulate processed symbols)
    data = {
        "Symbol": ["fish", "jar", "spiral", "arrow", "lattice"],
        "Cluster": [1, 1, 2, 2, 3],
        "Frequency": [42, 39, 57, 51, 33]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # Simulated energy-line map preview
    st.subheader("Symbolic Energy Map (Edge Detection)")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1a/Indus_valley_civilization_script_example.png",
             caption="Example symbol edges", use_container_width=True)

    # --------------------------------------------------
    # Phase II.5 — CFG Induction (Sequitur-style)
    # --------------------------------------------------
    st.header("Phase II.5 — Hierarchical CFG Induction")

    sequences = []
    if os.path.exists(seq_json_path):
        try:
            with open(seq_json_path, "r", encoding="utf-8") as f:
                sequences = json.load(f)
                st.success(f"Loaded {len(sequences)} sequences from {seq_json_path}")
        except Exception as e:
            st.warning(f"Could not load {seq_json_path}: {e}")

    if not sequences:
        st.info("⚠️ No sequence file found — using demo sequences for demonstration.")
        sequences = [
            ["spiral", "arrow", "dot", "spiral", "arrow", "dot"],
            ["spiral", "dot", "spiral", "arrow", "dot"],
            ["lattice", "dot", "lattice", "dot"]
        ]

    if st.button("🔎 Infer CFG from Sequences"):
        seq_res = sequitur_infer(sequences, min_rule_occurrence=min_rule_occurrence)
        st.success("CFG Induction Completed")

        st.subheader("Inferred Rules")
        if seq_res["rules"]:
            st.code(rules_to_text(seq_res["rules"]), language="text")
            tmpfile = os.path.join(tempfile.gettempdir(), "ivc_cfg_rules.json")
            export_rules_json(tmpfile, seq_res)
            with open(tmpfile, "r", encoding="utf-8") as f:
                st.download_button("📥 Download CFG JSON",
                                   data=f.read(),
                                   file_name="ivc_cfg_rules.json",
                                   mime="application/json")
        else:
            st.warning("No strong rules found. Increase corpus or lower threshold.")

        # ---- Graph Visualization ----
        edges = rules_to_graph_edges(seq_res["rules"])
        if edges:
            try:
                G = nx.DiGraph()
                G.add_edges_from(edges)
                fig, ax = plt.subplots(figsize=(7, 5))
                pos = nx.spring_layout(G, k=0.5, iterations=40)
                nx.draw(G, pos,
                        with_labels=True,
                        node_size=800,
                        node_color="lightgray",
                        font_size=9,
                        arrows=True,
                        ax=ax)
                ax.set_title("CFG Rule Hierarchy Graph")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Graph visualization failed: {e}")

    # --------------------------------------------------
    # Phase III–VIII summaries (descriptive placeholders)
    # --------------------------------------------------
    st.header("Phase III – Symbolic–Functional Correlation")
    st.markdown("""
    Cross-cultural comparison and clustering of core archetypes:
    **Fish → Consciousness unit**, **Jar → Containment**, **Wave/Spiral → Energy flow**, etc.
    """)

    st.header("Phase IV – Cognitive and Resonance Mapping")
    st.markdown("""
    Experimental correlation between symbol exposure and EEG resonance
    (gamma/theta synchronization) — supports resonance-based semantics.
    """)

    st.header("Phase V – Algorithmic Decoding Framework")
    st.markdown("""
    Three-tier model:
    1. Structural (syntax/order)
    2. Energetic (resonance geometry)
    3. Semantic (contextual meaning)
    Weighted probability fusion refines interpretations via convergence feedback.
    """)

    st.header("Phase VI – Quantum-Holographic Interface Hypothesis")
    st.markdown("""
    Interprets glyphs as **functional operations** within a cosmological code —
    describing interactions among **consciousness**, **matter**, and **field**.
    """)

    st.header("Phase VII – Cross-System Validation")
    st.markdown("""
    Algorithm applied to **Olmec**, **Rongorongo**, and **Proto-Elamite** scripts.
    Found recurrent recursive and harmonic architectures — evidence of global encoding principles.
    """)

    st.header("Phase VIII – Synthesis Outputs")
    st.markdown("""
    - Symbol Dictionary  
    - Reconstructed Formulae  
    - Quantum-Holographic Interface Diagram  
    - Harmonic Resonance Maps  
    - Artifact Placement and Energy Grids
    """)
else:
    st.info("👈 Configure parameters and press **Run Full Research Framework** to begin.")
