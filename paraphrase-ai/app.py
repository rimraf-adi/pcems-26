import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analyzer import ParaphraseAnalyzer, get_device

# Page Configuration
st.set_page_config(
    page_title="Manuscript Paraphrase & Quality Evaluator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clear cached resource if analyzer model signature updated
st.cache_resource.clear()

# Custom CSS for Premium Glassmorphism & Neon Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Background Gradient Glow */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 45%, #090d16 100%) !important;
        background-attachment: fixed !important;
    }

    /* Main Title Styling */
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        background: linear-gradient(135deg, #C084FC 0%, #6366F1 40%, #38BDF8 80%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.8rem;
    }
    
    /* Glassmorphism Cards */
    .stCard {
        background: rgba(22, 30, 46, 0.65);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.4rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 1.2rem;
    }
    
    .stCard:hover {
        border-color: rgba(139, 92, 246, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 25px 50px -12px rgba(139, 92, 246, 0.15);
    }

    /* Metric Containers */
    .metric-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem 1rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #8B5CF6, #3B82F6);
    }

    .metric-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1;
    }

    .metric-lbl {
        font-size: 0.82rem;
        color: #CBD5E1;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 8px;
    }

    .metric-sub {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 4px;
    }

    .metric-expl {
        font-size: 0.73rem;
        color: #94A3B8;
        margin-top: 6px;
        line-height: 1.3;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        padding-top: 6px;
    }

    /* Status Badges */
    .badge-engine {
        background: linear-gradient(135deg, #8B5CF6 0%, #6366F1 100%);
        color: white;
        padding: 0.35rem 0.9rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        display: inline-block;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.4);
        margin-bottom: 0.5rem;
    }

    /* Risk Tags */
    .tag-verbatim {
        background: rgba(244, 63, 94, 0.2);
        color: #FF859B;
        border: 1px solid rgba(244, 63, 94, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(244, 63, 94, 0.25);
    }

    .tag-high {
        background: rgba(245, 158, 11, 0.2);
        color: #FDE047;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
    }

    .tag-low {
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }

    .advice-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-left: 5px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 12px;
        margin-top: 0.5rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.15);
    }

    /* Style Text Areas */
    .stTextArea textarea {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        color: #F1F5F9 !important;
        font-family: 'Inter', monospace !important;
        font-size: 0.92rem !important;
        line-height: 1.6 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

def clean_latex(text: str) -> str:
    """Strip LaTeX formatting tags to focus pure prose text similarity."""
    text = re.sub(r'%.*', '', text)
    text = re.sub(r'\\(?:section|subsection|subsubsection|caption|label|cite|ref)\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\{|\}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Path resolution for project files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_CONTENT_PATH = os.path.join(BASE_DIR, "ai-content", "content.tex")
PAPER_CONTENT_PATH = os.path.join(BASE_DIR, "paper", "content.tex")

def load_manuscripts():
    """Read latest content.tex files directly from disk."""
    ai_tex = ""
    paper_tex = ""
    if os.path.exists(AI_CONTENT_PATH):
        with open(AI_CONTENT_PATH, "r", encoding="utf-8") as f:
            ai_tex = f.read()
    if os.path.exists(PAPER_CONTENT_PATH):
        with open(PAPER_CONTENT_PATH, "r", encoding="utf-8") as f:
            paper_tex = f.read()
    return ai_tex, paper_tex

def extract_section(full_text: str, max_lines: int = 120) -> str:
    """Extract initial section for sub-second responsiveness."""
    lines = full_text.splitlines()
    return "\n".join(lines[:max_lines])

def load_analyzer_instance(model_key: str):
    model_name = ParaphraseAnalyzer.AVAILABLE_MODELS[model_key]
    return ParaphraseAnalyzer(model_name=model_name)

# Helper to sync disk files into session state
def sync_disk_to_state(full_mode: bool):
    ai_raw, paper_raw = load_manuscripts()
    st.session_state["orig_input_area"] = ai_raw if full_mode else extract_section(ai_raw, 120)
    st.session_state["para_input_area"] = paper_raw if full_mode else extract_section(paper_raw, 120)

# App Header
st.markdown('<div class="main-title">Manuscript Paraphrase & Quality Evaluator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Multi-dimensional semantic fidelity, vocabulary substitution, Perplexity & Embedding Space inspector.</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Evaluation Controls")
    
    selected_model_key = st.selectbox(
        "NLP Evaluation Engine",
        list(ParaphraseAnalyzer.AVAILABLE_MODELS.keys()),
        index=0
    )
    
    device_type = get_device()
    st.markdown(f'<div class="badge-engine">⚡ {device_type.upper()}</div>', unsafe_allow_html=True)

    full_manuscript_mode = st.toggle(
        "📄 Full Manuscript (All 780 Lines)",
        value=st.session_state.get("prev_full_mode", False),
        help="Toggle OFF for lightning-fast Abstract/Intro evaluation, toggle ON for full paper."
    )

    if "prev_full_mode" not in st.session_state or st.session_state["prev_full_mode"] != full_manuscript_mode:
        st.session_state["prev_full_mode"] = full_manuscript_mode
        sync_disk_to_state(full_manuscript_mode)

    strip_latex_toggle = st.toggle("Clean LaTeX Command Tags", value=True, help="Clean out \\cite, \\section, \\ref tags to compare pure prose text.")

    st.markdown("---")
    st.subheader("🔄 Live Sync Controls")
    if st.button("🔄 Reload Latest content.tex from Disk", type="primary", use_container_width=True):
        sync_disk_to_state(full_manuscript_mode)
        st.toast("✅ Re-synced latest content.tex files from disk!", icon="📜")
        st.rerun()

    st.markdown("---")
    st.subheader("🎯 Risk Thresholds")
    high_threshold = st.slider(
        "High Risk Similarity Cutoff",
        min_value=0.05,
        max_value=0.90,
        value=0.85,
        step=0.05,
        help="Sentences scoring above this value will be flagged as requiring rephrasing."
    )

# Initial State Sync if empty
if "orig_input_area" not in st.session_state or "para_input_area" not in st.session_state:
    sync_disk_to_state(full_manuscript_mode)

# Quick Action Bar above text areas
action_col1, action_col2 = st.columns([3, 1])
with action_col1:
    st.caption("Editing manuscript prose below triggers real-time multi-metric evaluation.")
with action_col2:
    if st.button("🔄 Sync Disk Files", use_container_width=True):
        sync_disk_to_state(full_manuscript_mode)
        st.rerun()

# Text Input Section pre-populated with repo content.tex
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Baseline content.tex (ai-content/)")
    orig_input = st.text_area(
        "Baseline content.tex",
        height=240,
        key="orig_input_area"
    )
    st.caption(f"Words: {len(orig_input.split())} | Characters: {len(orig_input)}")

with col2:
    st.subheader("✍️ Current content.tex (paper/)")
    para_input = st.text_area(
        "Current content.tex",
        height=240,
        key="para_input_area"
    )
    st.caption(f"Words: {len(para_input.split())} | Characters: {len(para_input)}")

# Process Text Inputs
proc_orig = clean_latex(orig_input) if strip_latex_toggle else orig_input
proc_para = clean_latex(para_input) if strip_latex_toggle else para_input

# Real-Time Computation
if proc_orig.strip() and proc_para.strip():
    analyzer = load_analyzer_instance(selected_model_key)
    results = analyzer.analyze(proc_orig, proc_para, high_similarity_threshold=high_threshold)

    if "error" in results:
        st.error(results["error"])
    else:
        doc_data = results.get("document", {})
        para_data = results.get("paragraph", {})
        sent_data = results.get("sentence", {})
        sent_metrics = sent_data.get("metrics", {})
        pca_data = results.get("embeddings_pca", {})

        # Safe dictionary getters
        sem_sim = doc_data.get("semantic_similarity", doc_data.get("similarity", 0.0))
        jaccard_sim = doc_data.get("jaccard_similarity", 0.0)
        lex_repl_rate = doc_data.get("lexical_replacement_rate", 1.0 - jaccard_sim)
        status_badge = doc_data.get("status_badge", "📊 Analyzed")
        status_desc = doc_data.get("status_desc", doc_data.get("status", "Analysis completed."))
        
        vocab_rich_orig = doc_data.get("orig_vocab_richness", 0.5)
        vocab_rich_para = doc_data.get("para_vocab_richness", 0.5)
        
        read_base = doc_data.get("readability_baseline", {"avg_sentence_length": 15.0})
        read_curr = doc_data.get("readability_current", {"avg_sentence_length": 15.0})

        bp_base = doc_data.get("burstiness_perplexity_baseline", {})
        bp_curr = doc_data.get("burstiness_perplexity_current", {})

        # Navigation Tabs
        tab_overview, tab_embed, tab_bp, tab_paragraph, tab_sentence, tab_export = st.tabs([
            "📊 Executive Dashboard",
            "🌌 Embedding Vector Graph",
            "🎲 Perplexity & Burstiness",
            "📑 Paragraph Alignment",
            "🔍 Sentence Risk Inspector",
            "📄 Report & CSV Export"
        ])

        # ----------------------------------------------------
        # TAB 1: EXECUTIVE DASHBOARD
        # ----------------------------------------------------
        with tab_overview:
            # Collapsible Metric Guide
            with st.expander("ℹ️ How to Read These Metric Cards (Click to expand guide)", expanded=False):
                st.markdown(r"""
                - 🧠 **Semantic Fidelity (Meaning Preservation)**: Vector-space cosine similarity measuring how accurately core technical facts are preserved. **50%–80%** is optimal for paraphrasing.
                - 🔤 **Vocabulary Re-Wording (Lexical Substitution)**: Percentage of unique words changed. **>60%** indicates strong vocabulary re-wording.
                - 🚨 **Verbatim Risk Sentences**: Count of sentences with near-identical wording ($\ge 95\%$ similarity). **Goal is 0** before submission.
                - 📚 **Vocabulary Richness (Type-Token Ratio)**: Lexical diversity ratio ($\text{TTR}_{\text{current}} / \text{TTR}_{\text{baseline}}$). **>1.0x** means richer, less repetitive prose.
                """)

            st.markdown('<div class="stCard">', unsafe_allow_html=True)
            
            # Top Multi-Dimensional Metric Cards with Explicit Explanations
            m1, m2, m3, m4 = st.columns(4)
            
            with m1:
                sem_pct = int(sem_sim * 100)
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: #38BDF8;">{sem_pct}%</div>
                    <div class="metric-lbl">Semantic Fidelity</div>
                    <div class="metric-sub">Meaning Preservation</div>
                    <div class="metric-expl">Preserves <b>{sem_pct}%</b> of original scientific intent & facts. (Target: 50%–80%)</div>
                </div>
                ''', unsafe_allow_html=True)

            with m2:
                lex_pct = int(lex_repl_rate * 100)
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: #34D399;">{lex_pct}%</div>
                    <div class="metric-lbl">Vocabulary Re-Wording</div>
                    <div class="metric-sub">Lexical Substitution Rate</div>
                    <div class="metric-expl"><b>{lex_pct}%</b> of words replaced with new terms. (Target: >60%)</div>
                </div>
                ''', unsafe_allow_html=True)

            with m3:
                verb_cnt = sent_metrics.get("verbatim_copy_count", sent_metrics.get("high_risk_count", 0))
                color_v = "#FB7185" if verb_cnt > 0 else "#34D399"
                expl_v = f"<b>{verb_cnt}</b> near-identical sentence requires re-wording." if verb_cnt > 0 else "0 verbatim sentence copies detected."
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: {color_v};">{verb_cnt}</div>
                    <div class="metric-lbl">Verbatim Risk Sentences</div>
                    <div class="metric-sub">Near-Identical Wording</div>
                    <div class="metric-expl">{expl_v} (Target: 0)</div>
                </div>
                ''', unsafe_allow_html=True)

            with m4:
                ttr_ratio = round(vocab_rich_para / max(0.01, vocab_rich_orig), 2)
                expl_r = f"<b>{(ttr_ratio-1)*100:.0f}%</b> richer vocabulary diversity." if ttr_ratio >= 1.0 else "Slightly less lexical variety."
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: #FBBF24;">{ttr_ratio}x</div>
                    <div class="metric-lbl">Vocabulary Richness</div>
                    <div class="metric-sub">Type-Token Ratio Ratio</div>
                    <div class="metric-expl">{expl_r} (Target: >1.0x)</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Gauge & Executive Summary Callout
            g_col, c_col = st.columns([1, 1])

            with g_col:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sem_sim * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Semantic Overlap Index", 'font': {'size': 18, 'color': '#F8FAFC'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': "#64748B"},
                        'bar': {'color': "#8B5CF6"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(52, 211, 153, 0.2)"},
                            {'range': [50, 75], 'color': "rgba(251, 191, 36, 0.2)"},
                            {'range': [75, 100], 'color': "rgba(244, 63, 94, 0.25)"}
                        ],
                        'threshold': {
                            'line': {'color': "#F43F5E", 'width': 4},
                            'thickness': 0.75,
                            'value': high_threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(height=270, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC"))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c_col:
                st.subheader("💡 Executive Manuscript Diagnosis")
                st.markdown(f"**Status**: `{status_badge}`")
                st.write(status_desc)

                st.markdown(f"""
                <div class="advice-card">
                    <b>🎯 Academic Editing Insights & Benchmarks:</b><br>
                    • <b>Lexical Overlap</b>: Exact word token overlap is <b>{int(jaccard_sim*100)}%</b> (meaning <b>{int(lex_repl_rate*100)}%</b> of words were re-worded).<br>
                    • <b>Burstiness Style</b>: Current sentence length variation score is <b>{bp_curr.get('burstiness_score', 0.4)}</b> (<i>{bp_curr.get('burstiness_status', '')}</i>).<br>
                    • <b>Readability Evolution</b>: Average sentence length shifted from <b>{read_base.get('avg_sentence_length', 15.0)} words</b> (baseline) to <b>{read_curr.get('avg_sentence_length', 15.0)} words</b> (current).
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------
        # TAB 2: VISUAL EMBEDDING SPACE GRAPH
        # ----------------------------------------------------
        with tab_embed:
            st.subheader("🌌 Visual Embedding Vector Space Graph & Trajectories")
            st.caption("High-dimensional SentenceTransformer embeddings projected onto 2D and 3D Principal Components (PCA). Dashed neon tracks illustrate how individual sentence pairs moved across semantic vector space.")

            nodes_df = pca_data.get("nodes", pd.DataFrame())
            traj_df = pca_data.get("trajectories", pd.DataFrame())
            pca_2d_var = pca_data.get("pca_2d_explained_var", [0, 0])
            pca_3d_var = pca_data.get("pca_3d_explained_var", [0, 0, 0])

            if not nodes_df.empty:
                embed_subtab1, embed_subtab2, embed_subtab3 = st.tabs([
                    "2D Vector Trajectory Graph",
                    "3D Interactive Vector Space",
                    "🔥 Pairwise Cosine Heatmap"
                ])

                with embed_subtab1:
                    # 2D PCA Plot with Trajectory Lines
                    fig_2d = go.Figure()

                    # 1. Add Trajectory Lines connecting matching sentences
                    if not traj_df.empty:
                        for _, row in traj_df.iterrows():
                            # Line color based on similarity: Cyan for strong, Red for verbatim
                            line_color = "rgba(244, 63, 94, 0.4)" if row["similarity"] >= 0.95 else "rgba(139, 92, 246, 0.3)"
                            fig_2d.add_trace(go.Scatter(
                                x=[row["orig_x2d"], row["match_x2d"]],
                                y=[row["orig_y2d"], row["match_y2d"]],
                                mode="lines",
                                line=dict(color=line_color, width=1.5, dash="dash"),
                                hoverinfo="text",
                                text=f"Shift Track: Sent #{int(row['orig_id'])} ➔ #{int(row['match_id'])}<br>Semantic Sim: {row['similarity']*100:.1f}%<br>PCA Vector Distance: {row['pca_distance']:.3f}",
                                showlegend=False
                            ))

                    # 2. Add Baseline Scatter Points
                    orig_nodes = nodes_df[nodes_df["doc"] == "Baseline content.tex"]
                    fig_2d.add_trace(go.Scatter(
                        x=orig_nodes["x2d"],
                        y=orig_nodes["y2d"],
                        mode="markers+text",
                        marker=dict(size=12, color="#38BDF8", symbol="circle", line=dict(width=2, color="#0284C7")),
                        text=[f"#{row['local_id']}" for _, row in orig_nodes.iterrows()],
                        textposition="top center",
                        name="Baseline content.tex",
                        hoverinfo="text",
                        hovertext=[f"<b>[Baseline Sent #{row['local_id']}]</b><br>{row['text']}" for _, row in orig_nodes.iterrows()]
                    ))

                    # 3. Add Current Scatter Points
                    para_nodes = nodes_df[nodes_df["doc"] == "Current content.tex"]
                    fig_2d.add_trace(go.Scatter(
                        x=para_nodes["x2d"],
                        y=para_nodes["y2d"],
                        mode="markers+text",
                        marker=dict(size=12, color="#34D399", symbol="diamond", line=dict(width=2, color="#059669")),
                        text=[f"#{row['local_id']}" for _, row in para_nodes.iterrows()],
                        textposition="bottom center",
                        name="Current content.tex",
                        hoverinfo="text",
                        hovertext=[f"<b>[Current Sent #{row['local_id']}]</b><br>{row['text']}" for _, row in para_nodes.iterrows()]
                    ))

                    var_sum_2d = sum(pca_2d_var[:2])
                    fig_2d.update_layout(
                        title=f"2D PCA Sentence Vector Projection (Explained Variance: {var_sum_2d:.1f}%)",
                        xaxis_title=f"PCA Component 1 ({pca_2d_var[0]}% var)",
                        yaxis_title=f"PCA Component 2 ({pca_2d_var[1]}% var)" if len(pca_2d_var)>1 else "PCA Component 2",
                        template="plotly_dark",
                        height=560,
                        margin=dict(l=20, r=20, t=50, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(15, 23, 42, 0.6)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)

                with embed_subtab2:
                    # 3D PCA Scatter Plot
                    fig_3d = go.Figure()

                    # Add 3D Trajectory Lines
                    if not traj_df.empty:
                        for _, row in traj_df.iterrows():
                            fig_3d.add_trace(go.Scatter3d(
                                x=[row["orig_x3d"], row["match_x3d"]],
                                y=[row["orig_y3d"], row["match_y3d"]],
                                z=[row["orig_z3d"], row["match_z3d"]],
                                mode="lines",
                                line=dict(color="rgba(139, 92, 246, 0.4)", width=2),
                                hoverinfo="none",
                                showlegend=False
                            ))

                    # Baseline 3D Nodes
                    fig_3d.add_trace(go.Scatter3d(
                        x=orig_nodes["x3d"],
                        y=orig_nodes["y3d"],
                        z=orig_nodes["z3d"],
                        mode="markers",
                        marker=dict(size=6, color="#38BDF8", opacity=0.9),
                        name="Baseline content.tex",
                        hoverinfo="text",
                        hovertext=[f"<b>[Baseline #{row['local_id']}]</b><br>{row['text']}" for _, row in orig_nodes.iterrows()]
                    ))

                    # Current 3D Nodes
                    fig_3d.add_trace(go.Scatter3d(
                        x=para_nodes["x3d"],
                        y=para_nodes["y3d"],
                        z=para_nodes["z3d"],
                        mode="markers",
                        marker=dict(size=6, color="#34D399", opacity=0.9),
                        name="Current content.tex",
                        hoverinfo="text",
                        hovertext=[f"<b>[Current #{row['local_id']}]</b><br>{row['text']}" for _, row in para_nodes.iterrows()]
                    ))

                    var_sum_3d = sum(pca_3d_var[:3])
                    fig_3d.update_layout(
                        title=f"3D Vector Embedding Space (Explained Variance: {var_sum_3d:.1f}%)",
                        template="plotly_dark",
                        height=580,
                        margin=dict(l=0, r=0, t=40, b=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                        scene=dict(
                            xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                            bgcolor="rgba(15, 23, 42, 0.4)"
                        )
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                with embed_subtab3:
                    # Pairwise Cosine Similarity Heatmap
                    sent_matrix = sent_data.get("matrix", np.zeros((0,0)))
                    if sent_matrix.shape[0] > 0 and sent_matrix.shape[1] > 0:
                        fig_heat = go.Figure(data=go.Heatmap(
                            z=sent_matrix,
                            x=[f"Curr #{j+1}" for j in range(sent_matrix.shape[1])],
                            y=[f"Base #{i+1}" for i in range(sent_matrix.shape[0])],
                            colorscale="Plasma",
                            colorbar=dict(title="Cosine Sim")
                        ))
                        fig_heat.update_layout(
                            title="Full Sentence-by-Sentence Pairwise Similarity Heatmap",
                            xaxis_title="Current content.tex Sentences",
                            yaxis_title="Baseline content.tex Sentences",
                            template="plotly_dark",
                            height=550,
                            margin=dict(l=20, r=20, t=50, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(15, 23, 42, 0.6)"
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)

        # ----------------------------------------------------
        # TAB 3: PERPLEXITY & BURSTINESS
        # ----------------------------------------------------
        with tab_bp:
            st.subheader("🎲 Perplexity & Burstiness AI Signature Analysis")
            st.caption("AI text generators exhibit low burstiness (monotonous sentence length) and low perplexity (predictable word choices).")

            bp_c1, bp_c2, bp_c3 = st.columns(3)

            with bp_c1:
                b_score = bp_curr.get("burstiness_score", 0.0)
                b_status = bp_curr.get("burstiness_status", "Normal")
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: #A7F3D0;">{b_score}</div>
                    <div class="metric-lbl">Burstiness Index (CV)</div>
                    <div class="metric-sub">{b_status}</div>
                    <div class="metric-expl">Sentence length variation. Higher score = natural human rhythm.</div>
                </div>
                ''', unsafe_allow_html=True)

            with bp_c2:
                p_entropy = bp_curr.get("perplexity_entropy", 0.0)
                p_status = bp_curr.get("perplexity_status", "Normal")
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: #38BDF8;">{p_entropy}</div>
                    <div class="metric-lbl">Perplexity Proxy (Entropy)</div>
                    <div class="metric-sub">{p_status}</div>
                    <div class="metric-expl">Word choice randomness. Higher score = unpredictable human style.</div>
                </div>
                ''', unsafe_allow_html=True)

            with bp_c3:
                trans_cnt = bp_curr.get("ai_transition_count", 0)
                color_t = "#FCA5A5" if trans_cnt > 5 else "#6EE7B7"
                st.markdown(f'''
                <div class="metric-box">
                    <div class="metric-val" style="color: {color_t};">{trans_cnt}</div>
                    <div class="metric-lbl">Formulaic AI Connectors</div>
                    <div class="metric-sub">Overused Transition Words</div>
                    <div class="metric-expl">Frequency of connectors like 'furthermore', 'however', 'in conclusion'.</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Side-by-Side Burstiness & Perplexity Comparison")

            b_df = pd.DataFrame([
                {
                    "Metric": "Burstiness Score (CV)",
                    "Baseline content.tex": str(bp_base.get("burstiness_score", 0.0)),
                    "Current content.tex": str(bp_curr.get("burstiness_score", 0.0)),
                    "Interpretation": "Measures variation in sentence length"
                },
                {
                    "Metric": "Perplexity Entropy (Bits)",
                    "Baseline content.tex": str(bp_base.get("perplexity_entropy", 0.0)),
                    "Current content.tex": str(bp_curr.get("perplexity_entropy", 0.0)),
                    "Interpretation": "Measures vocabulary unpredictability"
                },
                {
                    "Metric": "AI Transition Words Count",
                    "Baseline content.tex": str(bp_base.get("ai_transition_count", 0)),
                    "Current content.tex": str(bp_curr.get("ai_transition_count", 0)),
                    "Interpretation": "Overused logical connectors"
                },
                {
                    "Metric": "Average Sentence Length",
                    "Baseline content.tex": f"{bp_base.get('sentence_len_mean', 0.0)} words",
                    "Current content.tex": f"{bp_curr.get('sentence_len_mean', 0.0)} words",
                    "Interpretation": "Mean words per sentence"
                },
                {
                    "Metric": "Sentence Length Std Dev",
                    "Baseline content.tex": f"± {bp_base.get('sentence_len_std', 0.0)} words",
                    "Current content.tex": f"± {bp_curr.get('sentence_len_std', 0.0)} words",
                    "Interpretation": "Spread/rhythm variation"
                }
            ])

            st.dataframe(b_df, use_container_width=True, hide_index=True)

        # ----------------------------------------------------
        # TAB 4: PARAGRAPH BREAKDOWN
        # ----------------------------------------------------
        with tab_paragraph:
            st.subheader("📑 Paragraph Alignment & Lexical Overlap")
            
            p_matches = para_data.get("matches", [])
            if not p_matches:
                st.info("No distinct paragraphs detected.")
            else:
                for p_match in p_matches:
                    sem_v = p_match.get("semantic_similarity", p_match.get("similarity", 0.0))
                    lex_v = p_match.get("lexical_overlap", 0.0)

                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    p_col1, p_col2 = st.columns([1, 1])
                    
                    with p_col1:
                        st.markdown(f"**Baseline Paragraph #{p_match.get('orig_index', 1)}** ({p_match.get('word_count_orig', 0)} words)")
                        st.write(p_match.get('orig_paragraph', ''))
                        
                    with p_col2:
                        st.markdown(f"**Current Paragraph #{p_match.get('best_match_index', 1)}** ({p_match.get('word_count_para', 0)} words)")
                        st.write(p_match.get('best_match_paragraph', ''))

                    p_bar1, p_bar2 = st.columns(2)
                    with p_bar1:
                        st.progress(min(1.0, max(0.0, sem_v)), text=f"Semantic Meaning Retention: {sem_v * 100:.1f}%")
                    with p_bar2:
                        st.progress(min(1.0, max(0.0, lex_v)), text=f"Direct Lexical Word Overlap: {lex_v * 100:.1f}%")

                    st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------
        # TAB 5: SENTENCE RISK INSPECTOR
        # ----------------------------------------------------
        with tab_sentence:
            st.subheader("🔍 Sentence Risk & Alignment Inspector")

            s_matches = sent_data.get("matches", [])
            if not s_matches:
                st.info("No sentences detected.")
            else:
                filter_risk = st.radio(
                    "Filter Sentences by Risk Category",
                    ["All Sentences", "High Risk / Verbatim Only", "Moderate Only", "Distinct Only"],
                    horizontal=True
                )

                for s_match in s_matches:
                    sim_v = s_match.get("similarity", 0.0)
                    lex_v = s_match.get("lexical_overlap", 0.0)
                    risk_t = s_match.get("risk_level", "Analyzed")
                    badge_str = s_match.get("risk_badge", "📊 Analyzed")

                    if filter_risk == "High Risk / Verbatim Only" and not ("Direct" in risk_t or "High" in risk_t):
                        continue
                    elif filter_risk == "Moderate Only" and "Moderate" not in risk_t:
                        continue
                    elif filter_risk == "Distinct Only" and "Distinct" not in risk_t:
                        continue

                    if "Verbatim" in badge_str or "High" in badge_str:
                        badge_html = f'<span class="tag-verbatim">{badge_str} (Semantic: {sim_v * 100:.1f}%, Lexical: {lex_v * 100:.1f}%)</span>'
                    elif "Moderate" in badge_str:
                        badge_html = f'<span class="tag-high">{badge_str} (Semantic: {sim_v * 100:.1f}%)</span>'
                    else:
                        badge_html = f'<span class="tag-low">{badge_str} (Semantic: {sim_v * 100:.1f}%)</span>'

                    st.markdown('<div class="stCard">', unsafe_allow_html=True)
                    st.markdown(f"**Sentence #{s_match.get('orig_id', 1)}** &nbsp;&nbsp; {badge_html}", unsafe_allow_html=True)
                    
                    s_c1, s_c2 = st.columns([1, 1])
                    with s_c1:
                        st.markdown("**Baseline Sentence:**")
                        st.info(s_match.get("orig_sentence", ""))
                    with s_c2:
                        st.markdown(f"**Current Sentence (Match #{s_match.get('best_match_id', 1)}):**")
                        st.success(s_match.get("best_match_sentence", ""))

                    st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------------------------------
        # TAB 6: EXPORT & REPORT
        # ----------------------------------------------------
        with tab_export:
            st.subheader("📄 Export Detailed Alignment Report")
            
            s_matches = sent_data.get("matches", [])
            df_export = pd.DataFrame(s_matches)
            if not df_export.empty:
                cols_to_show = [c for c in ["orig_id", "orig_sentence", "best_match_id", "best_match_sentence", "similarity", "lexical_overlap", "risk_level"] if c in df_export.columns]
                st.dataframe(
                    df_export[cols_to_show],
                    column_config={
                        "orig_id": "Baseline #",
                        "orig_sentence": "Baseline Sentence",
                        "best_match_id": "Current Match #",
                        "best_match_sentence": "Current Sentence",
                        "similarity": st.column_config.NumberColumn("Semantic Sim", format="%.4f"),
                        "lexical_overlap": st.column_config.NumberColumn("Lexical Overlap", format="%.4f"),
                        "risk_level": "Risk Category"
                    },
                    use_container_width=True
                )

                csv_data = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Alignment CSV Report",
                    data=csv_data,
                    file_name="manuscript_paraphrase_evaluation_report.csv",
                    mime="text/csv",
                    type="primary"
                )
else:
    st.info("👈 Please load content.tex into both text editors above to trigger semantic analysis.")
