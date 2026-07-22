"""
ProteinForge: Dynamic AI Structure Prediction Dashboard
100% DYNAMIC - No hard-coded sequences or structures
All data fetched from APIs or user input at runtime
"""
import streamlit as st
import py3Dmol
from stmol import showmol
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
# Add the project root to sys.path so we can import the backend_integration package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_integration.utils import (
    validate_sequence, parse_fasta, fetch_uniprot_sequence,
    predict_structure_esmfold, fetch_alphafold_structure,
    extract_plddt_from_pdb, analyze_sequence, 
    format_sequence_display, get_confidence_category, 
    parse_pdb_coordinates, calculate_rmsd,
    get_example_protein_ids, fetch_uniprot_metadata,
    get_plddt_color_scale,
    # ── Hyperparameter-tunable analysis ───────────────────────────────────────
    calculate_sequence_quality, calculate_hydrophobicity_profile,
    predict_disorder_profile, calculate_plddt_smoothed,
    trim_low_confidence_termini, assess_prediction_reliability,
)
import time


# Page configuration
st.set_page_config(
    page_title="ProteinForge - AI Structure Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS with Modern Design
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sub Header */
    .sub-header {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Dynamic Badge */
    .dynamic-badge {
        background: linear-gradient(120deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        margin-left: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 1px solid rgba(255,255,255,0.18);
        backdrop-filter: blur(10px);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 45px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        border: none;
        padding: 0.6rem 1.5rem;
        letter-spacing: 0.02em;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    
    /* Primary Button */
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander Styling */
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: white !important;
        font-weight: 700;
        font-size: 1.8rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: white;
        border-right: 2px solid #f0f0f0;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: #667eea;
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        color: #333;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] .stRadio > div {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    
    /* Info/Success/Warning Boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    
    /* Text Input */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 0.7rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Text Area */
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        padding: 0.7rem 1rem;
        font-size: 0.95rem;
        font-family: 'Courier New', monospace;
        transition: all 0.3s;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* Download Button Customization */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s;
    }
    
    .stDownloadButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        background: rgba(102, 126, 234, 0.05);
        transition: all 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: #764ba2;
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox>div>div {
        border-radius: 10px;
        border: 2px solid #e5e7eb;
    }
    
    /* Code Block */
    .stCodeBlock {
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }
    
    /* Responsive Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    p, span, div {
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'predicted_structure' not in st.session_state:
    st.session_state.predicted_structure = None
if 'current_sequence' not in st.session_state:
    st.session_state.current_sequence = None
if 'sequence_analysis' not in st.session_state:
    st.session_state.sequence_analysis = None
if 'plddt_scores' not in st.session_state:
    st.session_state.plddt_scores = None
if 'protein_name' not in st.session_state:
    st.session_state.protein_name = "Predicted Structure"
if 'prediction_source' not in st.session_state:
    st.session_state.prediction_source = None

# ── Hyperparameter defaults (bias-variance trade-off controls) ───────────────
if 'hp_plddt_threshold' not in st.session_state:
    st.session_state.hp_plddt_threshold = 70
if 'hp_window_size' not in st.session_state:
    st.session_state.hp_window_size = 9
if 'hp_disorder_threshold' not in st.session_state:
    st.session_state.hp_disorder_threshold = 0.5
if 'hp_auto_trim' not in st.session_state:
    st.session_state.hp_auto_trim = True
if 'hp_smooth_sigma' not in st.session_state:
    st.session_state.hp_smooth_sigma = 1.5
if 'hp_ensemble_mode' not in st.session_state:
    st.session_state.hp_ensemble_mode = True


def main():
    """Main application entry point."""
    
    # Header
    st.markdown(
        '<h1 class="main-header">🧬 ProteinForge</h1>'
        '<p class="sub-header">AI-Powered Protein Structure Prediction'
        '<span class="dynamic-badge">100% DYNAMIC DATA</span></p>', 
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Navigation
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🔬 Predict", "📊 Batch", "🔄 Compare", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Model selection
        st.subheader("🤖 Prediction Engine")
        model_choice = st.radio(
            "Choose model:",
            ["ESMFold API", "AlphaFold DB"],
            help="ESMFold: Fast predictions from any sequence.\nAlphaFold DB: Pre-computed structures (UniProt ID required)."
        )
        st.session_state.model_choice = model_choice
        
        st.divider()
        
        # Data source info
        st.subheader("📡 Data Sources")
        st.success("**All data fetched live:**")
        st.caption("✓ UniProt REST API")
        st.caption("✓ AlphaFold Database")
        st.caption("✓ ESMFold API")
        st.caption("✓ Zero hard-coded data")
        
        st.divider()
        
        # ── Hyperparameter Tuning Controls ───────────────────────────────────────
        st.subheader("🎛️ Tuning Controls")
        with st.expander("Adjust Hyperparameters", expanded=False):
            st.caption("📐 **Bias ←→ Variance Controls:**")
            
            plddt_thr = st.slider(
                "pLDDT Confidence Threshold",
                min_value=50, max_value=90, step=5,
                value=int(st.session_state.hp_plddt_threshold),
                help="Higher = stricter: more bias, less variance. Filters unreliable residues."
            )
            st.session_state.hp_plddt_threshold = plddt_thr
            
            win = st.select_slider(
                "Hydrophobicity Window",
                options=[5, 7, 9, 11, 13, 15, 17, 19, 21],
                value=int(st.session_state.hp_window_size),
                help="Larger window = smoother profile (↓ variance, ↑ bias)."
            )
            st.session_state.hp_window_size = win
            
            dis_thr = st.slider(
                "Disorder Threshold",
                min_value=0.1, max_value=0.9, step=0.1,
                value=float(st.session_state.hp_disorder_threshold),
                help="Fraction above which a window is flagged as disordered."
            )
            st.session_state.hp_disorder_threshold = dis_thr
            
            sigma = st.slider(
                "pLDDT Smoothing σ",
                min_value=0.5, max_value=5.0, step=0.5,
                value=float(st.session_state.hp_smooth_sigma),
                help="Gaussian smoothing. Higher σ = smoother curve (↓ variance)."
            )
            st.session_state.hp_smooth_sigma = sigma
            
            auto_trim = st.checkbox(
                "Auto-trim Disordered Termini",
                value=bool(st.session_state.hp_auto_trim),
                help="Remove low-pLDDT N/C terminal residues to reduce variance."
            )
            st.session_state.hp_auto_trim = auto_trim
            
            ensemble = st.checkbox(
                "Ensemble Mode",
                value=bool(st.session_state.hp_ensemble_mode),
                help="Compare ESMFold + AlphaFold when UniProt ID is known."
            )
            st.session_state.hp_ensemble_mode = ensemble
            
            # Live bias-variance dial
            bv = (plddt_thr - 50) / 40.0
            st.progress(bv, text=f"← Low Bias | Threshold: {plddt_thr} | Low Variance →")
        
        st.divider()
        
        # Quick stats if structure is loaded
        if st.session_state.predicted_structure:
            st.subheader("📈 Current Structure")
            st.info(f"**{st.session_state.protein_name}**")
            if st.session_state.plddt_scores:
                _, plddt = st.session_state.plddt_scores
                if len(plddt) > 0:
                    avg_plddt = sum(plddt) / len(plddt)
                    st.metric("Avg Confidence", f"{avg_plddt:.1f}")
    
    # Route to pages
    if page == "🏠 Home":
        show_home()
    elif page == "🔬 Predict":
        show_prediction()
    elif page == "📊 Batch":
        show_batch_analysis()
    elif page == "🔄 Compare":
        show_comparison()
    elif page == "ℹ️ About":
        show_about()


def show_home():
    """Home page with overview."""
    
    # Feature cards with modern gradient design
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">⚡</div>
                <h2 style="color: white; margin-bottom: 0.5rem;">Lightning Fast</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1rem;">Get predictions in 30-60 seconds using state-of-the-art AI</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</div>
                <h2 style="color: white; margin-bottom: 0.5rem;">High Accuracy</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1rem;">Powered by ESMFold & AlphaFold models</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div style="text-align: center; color: white;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌐</div>
                <h2 style="color: white; margin-bottom: 0.5rem;">100% Dynamic</h2>
                <p style="color: rgba(255,255,255,0.9); font-size: 1rem;">All data fetched live from APIs</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide with modern cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.12);">
        <h2 style="text-align: center; margin-bottom: 2rem; color: #667eea;">🚀 Quick Start Guide</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">1️⃣ Input Your Protein</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li><strong>Paste</strong> amino acid sequence directly</li>
                <li><strong>Upload</strong> FASTA file format</li>
                <li><strong>Fetch</strong> from UniProt database</li>
                <li><strong>Try</strong> example proteins below</li>
            </ul>
            <br>
            <h3 style="color: #667eea; margin-bottom: 1rem;">2️⃣ Choose Prediction Engine</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li><strong>ESMFold API</strong>: Fast, works with any sequence</li>
                <li><strong>AlphaFold DB</strong>: Pre-computed, requires UniProt ID</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">3️⃣ Visualize Results</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li>Interactive <strong>3D structure viewer</strong></li>
                <li>Confidence metrics (<strong>pLDDT scores</strong>)</li>
                <li>Detailed <strong>sequence analysis</strong></li>
                <li>Amino acid <strong>composition charts</strong></li>
            </ul>
            <br>
            <h3 style="color: #667eea; margin-bottom: 1rem;">4️⃣ Export & Analyze</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li>Download <strong>PDB files</strong></li>
                <li>Compare with <strong>reference structures</strong></li>
                <li>Batch process <strong>multiple sequences</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Example proteins section with modern design
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.12);">
        <h2 style="text-align: center; margin-bottom: 1rem; color: #667eea;">🧪 Try Example Proteins</h2>
        <p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 1.5rem;">
            Click any protein below to fetch it dynamically from UniProt API
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    examples = get_example_protein_ids()
    cols = st.columns(5)
    
    for idx, (name, info) in enumerate(examples.items()):
        with cols[idx]:
            # Create card design for each example
            st.markdown(f"""
            <div class="feature-card" style="text-align: center; padding: 1.5rem 1rem; min-height: 180px;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🧬</div>
                <h4 style="color: #667eea; margin-bottom: 0.5rem; font-size: 0.95rem;">{name.split('(')[0].strip()}</h4>
                <p style="color: #888; font-size: 0.8rem; margin-bottom: 1rem;">{info['uniprot']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Load", key=f"ex_{idx}", use_container_width=True):
                with st.spinner(f"Fetching {name}..."):
                    seq, prot_name = fetch_uniprot_sequence(info['uniprot'])
                    if seq:
                        st.session_state.temp_sequence = seq
                        st.session_state.temp_name = prot_name
                        st.session_state.temp_uniprot = info['uniprot']
                        st.success(f"✅ Loaded! Go to Predict page")
                        st.balloons()
                    else:
                        st.error("Failed to fetch")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Hugging Face Dataset section with modern design
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.12);">
        <h2 style="text-align: center; margin-bottom: 1rem; color: #667eea;">🗄️ Browse Protein Dataset</h2>
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
            Explore 500K+ protein sequences from Hugging Face dataset (loaded dynamically)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    from utils import get_random_sequences_from_dataset, get_dataset_stats
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎲 Load Random Sequences from HF Dataset", use_container_width=True):
            with st.spinner("Loading from Hugging Face..."):
                random_seqs = get_random_sequences_from_dataset(5)
                if random_seqs:
                    st.session_state.random_dataset_sequences = random_seqs
                    st.success(f"✅ Loaded {len(random_seqs)} sequences")
                else:
                    st.error("Failed to load dataset")
    
    with col2:
        if st.button("📊 Dataset Stats", use_container_width=True):
            with st.spinner("Loading statistics..."):
                stats = get_dataset_stats()
                if stats:
                    st.session_state.dataset_stats = stats
    
    # Display dataset stats if available
    if 'dataset_stats' in st.session_state and st.session_state.dataset_stats:
        stats = st.session_state.dataset_stats
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sequences", f"{stats['total_sequences']:,}")
        with col2:
            st.metric("Avg Length", f"{stats['avg_length']:.0f} aa")
        with col3:
            st.metric("Min Length", f"{stats['min_length']} aa")
        with col4:
            st.metric("Max Length", f"{stats['max_length']} aa")
    
    # Display random sequences if loaded
    if 'random_dataset_sequences' in st.session_state:
        st.subheader("🧬 Random Sequences from Dataset")
        for seq_info in st.session_state.random_dataset_sequences:
            with st.expander(f"🔬 {seq_info['id']} - {seq_info['length']} residues ({seq_info['method']}, {seq_info['resolution']}Å)"):
                st.code(seq_info['sequence'])
                if st.button(f"Use this sequence", key=f"use_{seq_info['id']}", use_container_width=True):
                    st.session_state.temp_sequence = seq_info['sequence']
                    st.session_state.temp_name = seq_info['id']
                    st.success("✅ Sequence loaded! Go to Predict page")


def show_prediction():
    """Main prediction page - fully dynamic."""
    
    # Modern header design
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); margin-bottom: 2rem;">
        <h1 style="text-align: center; color: #667eea; margin-bottom: 0.5rem;">🔬 Protein Structure Prediction</h1>
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
            Predict 3D structure from amino acid sequence using AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("🌐 **All data fetched dynamically** - No hard-coded sequences")
    
    # Check if sequence was loaded from home page
    sequence = None
    protein_name = "Predicted Protein"
    uniprot_id = None
    
    if 'temp_sequence' in st.session_state and st.session_state.temp_sequence:
        sequence = st.session_state.temp_sequence
        protein_name = st.session_state.get('temp_name', 'Loaded Sequence')
        uniprot_id = st.session_state.get('temp_uniprot', None)
        # Copy to current_input for persistence
        st.session_state.current_input_sequence = sequence
        st.session_state.current_input_name = protein_name
        st.session_state.current_input_uniprot = uniprot_id
        # Clear temp to avoid showing message repeatedly
        st.session_state.temp_sequence = None
        st.success(f"📥 Pre-loaded: {protein_name} ({len(sequence)} residues)")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["📝 Paste Sequence", "📄 Upload FASTA", "🔗 UniProt ID", "🎲 Random from Dataset"],
        horizontal=True
    )
    
    if input_method == "📝 Paste Sequence":
        # Get existing sequence from session state if available
        default_value = ""
        if 'current_input_sequence' in st.session_state and input_method == "📝 Paste Sequence":
            default_value = st.session_state.current_input_sequence
        
        sequence_input = st.text_area(
            "Enter amino acid sequence:",
            value=default_value,
            height=150,
            placeholder="Example: MSKGEELFTGVVPILVELD... (paste your amino acid sequence here)",
            help="Paste raw sequence using standard amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
        )
        
        if sequence_input:
            sequence = sequence_input.replace(" ", "").replace("\n", "").upper()
            protein_name = "User Sequence"
            # Store in session state
            st.session_state.current_input_sequence = sequence
            st.session_state.current_input_name = protein_name
            st.session_state.current_input_uniprot = None
    
    elif input_method == "📄 Upload FASTA":
        uploaded_file = st.file_uploader(
            "Upload FASTA file:",
            type=['fasta', 'fa', 'txt', 'faa'],
            help="Upload FASTA format file"
        )
        
        if uploaded_file:
            fasta_content = uploaded_file.read().decode('utf-8')
            sequences = parse_fasta(fasta_content)
            
            if sequences:
                if len(sequences) > 1:
                    selected = st.selectbox(
                        "Multiple sequences found. Select one:",
                        range(len(sequences)),
                        format_func=lambda i: f"{sequences[i][0]} ({len(sequences[i][1])} residues)"
                    )
                    protein_name, sequence = sequences[selected]
                else:
                    protein_name, sequence = sequences[0]
                
                # Store in session state
                st.session_state.current_input_sequence = sequence
                st.session_state.current_input_name = protein_name
                st.session_state.current_input_uniprot = None
                st.success(f"✅ Loaded: {protein_name} ({len(sequence)} residues)")
    
    elif input_method == "🔗 UniProt ID":
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uniprot_input = st.text_input(
                "Enter UniProt Accession:",
                value=uniprot_id if uniprot_id else "",
                placeholder="e.g., P69905, P01308, P42212"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_btn = st.button("🔍 Fetch", use_container_width=True)
        
        if fetch_btn and uniprot_input:
            uniprot_id = uniprot_input.strip()
            with st.spinner(f"Fetching from UniProt API..."):
                sequence, protein_name = fetch_uniprot_sequence(uniprot_id)
                
                if sequence:
                    # Store in session state
                    st.session_state.current_input_sequence = sequence
                    st.session_state.current_input_name = protein_name
                    st.session_state.current_input_uniprot = uniprot_id
                    
                    st.success(f"✅ Fetched: {protein_name} ({len(sequence)} residues)")
                    
                    # Fetch metadata
                    metadata = fetch_uniprot_metadata(uniprot_id)
                    if metadata:
                        with st.expander("📋 Protein Metadata"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Name:** {metadata['name']}")
                                st.write(f"**Organism:** {metadata['organism']}")
                            with col2:
                                st.write(f"**Gene:** {metadata['gene']}")
                                st.write(f"**Length:** {metadata['length']} aa")
                else:
                    st.error(f"❌ Could not fetch {uniprot_id}")
    
    elif input_method == "🎲 Random from Dataset":
        from utils import get_random_sequences_from_dataset
        
        if st.button("🎲 Load Random Sequence", use_container_width=True):
            with st.spinner("Loading from dataset..."):
                random_seqs = get_random_sequences_from_dataset(1)
                if random_seqs:
                    seq_info = random_seqs[0]
                    # Store in session state so it persists across reruns
                    st.session_state.temp_sequence = seq_info['sequence']
                    st.session_state.temp_name = seq_info['id']
                    st.session_state.temp_uniprot = None  # No UniProt ID from dataset
                    st.success(f"✅ Loaded: {seq_info['id']} ({seq_info['length']} residues)")
                    st.rerun()  # Force rerun to update the display
        
        # Load from session state if available
        if 'temp_sequence' in st.session_state and st.session_state.temp_sequence:
            sequence = st.session_state.temp_sequence
            protein_name = st.session_state.get('temp_name', 'Dataset Sequence')
            st.success(f"📥 Loaded: {protein_name} ({len(sequence)} residues)")
    
    st.markdown("---")
    
    # Get sequence from session state (persists across button clicks)
    if 'current_input_sequence' in st.session_state:
        sequence = st.session_state.current_input_sequence
        protein_name = st.session_state.get('current_input_name', 'Predicted Protein')
        uniprot_id = st.session_state.get('current_input_uniprot', None)
    
    # Debug: Show what we have
    if sequence:
        st.caption(f"✅ Sequence ready for prediction ({len(sequence)} residues)")
    
    # Prediction section
    if sequence:
        # Validate sequence
        is_valid, message = validate_sequence(sequence)
        
        if not is_valid:
            st.error(f"❌ Invalid sequence: {message}")
            return
        
        # Use validated sequence
        sequence = message  # validate_sequence returns cleaned sequence in message
        
        # Show sequence info
        with st.expander("📋 Sequence Information", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.text(f"Name: {protein_name}")
                st.text(f"Length: {len(sequence)} residues")
                if uniprot_id:
                    st.text(f"UniProt: {uniprot_id}")
            with col2:
                st.metric("Sequence Length", f"{len(sequence)} aa")
            
            st.text_area("Sequence:", format_sequence_display(sequence), height=100)
        
        # ── Sequence Quality Assessment ──────────────────────────────────────────
        quality = calculate_sequence_quality(sequence)
        q_score = quality['quality_score']
        with st.expander("🔍 Sequence Quality Assessment", expanded=True):
            q_icon = "🟢" if q_score >= 70 else ("🟡" if q_score >= 40 else "🔴")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality Score", f"{q_icon} {q_score}/100")
            with col2:
                st.metric("Complexity", f"{quality['complexity']:.1f}%")
            with col3:
                st.metric("Disorder Risk", f"{quality['disorder_score']:.1f}%")
            st.progress(q_score / 100)
            for warn in quality['warnings']:
                st.warning(f"⚠️ {warn}")
            if not quality['warnings']:
                st.success("✅ Sequence looks good for structure prediction!")
        
        st.markdown("---")
        
        # Prediction button
        model_choice = st.session_state.get('model_choice', 'ESMFold API')
        
        if model_choice == "ESMFold API":
            predict_btn = st.button("🚀 Predict with ESMFold", use_container_width=True, type="primary")
            
            if predict_btn:
                st.info("🧬 Starting ESMFold prediction... This will take 30-60 seconds.")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress bar in background while making API call
                status_text.text("Submitting sequence to ESMFold API...")
                progress_bar.progress(10)
                
                # Make the actual API call
                pdb_string, error = predict_structure_esmfold(sequence)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                if pdb_string:
                    st.session_state.predicted_structure = pdb_string
                    st.session_state.current_sequence = sequence
                    st.session_state.protein_name = protein_name
                    st.session_state.sequence_analysis = analyze_sequence(sequence)
                    st.session_state.prediction_source = "ESMFold API"
                    
                    # Extract pLDDT
                    residues, plddt = extract_plddt_from_pdb(pdb_string)
                    st.session_state.plddt_scores = (residues, plddt)
                    
                    st.success("✅ Structure prediction completed successfully!")
                    st.balloons()
                    st.rerun()  # Force page refresh to show results
                else:
                    st.error(f"❌ Prediction failed: {error}")
        
        else:  # AlphaFold DB
            if not uniprot_id:
                st.warning("⚠️ AlphaFold DB requires UniProt ID. Use UniProt ID input method.")
            else:
                predict_btn = st.button("📥 Fetch from AlphaFold DB", use_container_width=True, type="primary")
                
                if predict_btn:
                    with st.spinner("📥 Fetching from AlphaFold Database..."):
                        pdb_string, error = fetch_alphafold_structure(uniprot_id)
                        
                        if pdb_string:
                            st.session_state.predicted_structure = pdb_string
                            st.session_state.current_sequence = sequence
                            st.session_state.protein_name = protein_name
                            st.session_state.sequence_analysis = analyze_sequence(sequence)
                            st.session_state.prediction_source = "AlphaFold DB"
                            
                            residues, plddt = extract_plddt_from_pdb(pdb_string)
                            st.session_state.plddt_scores = (residues, plddt)
                            
                            st.success("✅ Structure fetched successfully!")
                            st.rerun()  # Force page refresh
                        else:
                            st.error(f"❌ {error}")
    
    # Display results
    if st.session_state.predicted_structure:
        st.markdown("---")
        st.info(f"📡 **Source:** {st.session_state.prediction_source}")
        show_results()


def show_results():
    """Display results with 3D viewer and analysis."""
    st.header(f"📊 Results: {st.session_state.protein_name}")
    
    tabs = st.tabs(["🎨 3D Viewer", "📈 Confidence", "🔬 Analysis", "💾 Download", "⚗️ Reliability"])
    
    # Tab 1: 3D Visualization
    with tabs[0]:
        st.subheader("Interactive 3D Structure")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.write("**Controls:**")
            style = st.selectbox("Style:", ["cartoon", "sphere", "stick", "line"], index=0)
            color_scheme = st.selectbox("Color:", ["pLDDT", "Spectrum", "Secondary Structure"], index=0)
            spin = st.checkbox("Auto-rotate", value=False)
            bg_color = st.color_picker("Background:", "#FFFFFF")
        
        with col1:
            view = py3Dmol.view(width=800, height=600)
            view.addModel(st.session_state.predicted_structure, 'pdb')
            
            if color_scheme == "pLDDT":
                view.setStyle({style: {'colorscheme': {'prop': 'b', 'gradient': 'roygb', 'min': 50, 'max': 90}}})
            elif color_scheme == "Secondary Structure":
                view.setStyle({style: {'colorscheme': 'ssJmol'}})
            else:
                view.setStyle({style: {'color': 'spectrum'}})
            
            view.setBackgroundColor(bg_color)
            view.zoomTo()
            if spin:
                view.spin(True)
            
            showmol(view, height=600, width=800)
        
        if color_scheme == "pLDDT":
            st.markdown("---")
            st.write("**pLDDT Confidence Scale:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("🔵 **Very High** (>90)")
            with col2:
                st.markdown("🟢 **Confident** (70-90)")
            with col3:
                st.markdown("🟡 **Low** (50-70)")
            with col4:
                st.markdown("🟠 **Very Low** (<50)")
    
    # Tab 2: Confidence
    with tabs[1]:
        st.subheader("📈 Confidence Metrics (pLDDT)")
        
        if st.session_state.plddt_scores:
            residues, plddt = st.session_state.plddt_scores
            
            if len(plddt) > 0:
                avg_plddt = sum(plddt) / len(plddt)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average", f"{avg_plddt:.2f}")
                with col2:
                    st.metric("Maximum", f"{max(plddt):.2f}")
                with col3:
                    st.metric("Minimum", f"{min(plddt):.2f}")
                with col4:
                    st.metric("Quality", get_confidence_category(avg_plddt))
                
                st.markdown("---")
                
                # Per-residue plot
                colors = ['#FF7D45' if p < 50 else '#FFDB13' if p < 70 else '#65CBF3' if p < 90 else '#0053D6' for p in plddt]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=residues, y=plddt,
                    mode='lines+markers',
                    marker=dict(color=colors, size=4),
                    line=dict(color='lightgray', width=1),
                    name='Raw pLDDT'
                ))
                
                # Smoothed confidence line (Gaussian, sigma = hyperparameter)
                _sigma = st.session_state.get('hp_smooth_sigma', 1.5)
                smoothed_plddt = calculate_plddt_smoothed(residues, plddt, sigma=float(_sigma))
                fig.add_trace(go.Scatter(
                    x=residues, y=smoothed_plddt,
                    mode='lines',
                    line=dict(color='#667eea', width=2.5),
                    name=f'Smoothed (σ={_sigma})'
                ))
                
                fig.add_hline(y=90, line_dash="dash", line_color="green", annotation_text="Very High")
                fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Confident")
                fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Low")
                
                fig.update_layout(
                    title="Per-Residue Confidence",
                    xaxis_title="Residue Number",
                    yaxis_title="pLDDT Score",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution
                st.markdown("---")
                bins = [0, 50, 70, 90, 100]
                labels = ['Very Low', 'Low', 'Confident', 'Very High']
                hist_data = pd.cut(plddt, bins=bins, labels=labels)
                counts = hist_data.value_counts()
                
                fig2 = go.Figure(data=[go.Bar(
                    x=labels,
                    y=[counts.get(label, 0) for label in labels],
                    marker_color=['#FF7D45', '#FFDB13', '#65CBF3', '#0053D6']
                )])
                
                fig2.update_layout(title="Confidence Distribution", height=400)
                st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 3: Analysis
    with tabs[2]:
        st.subheader("🔬 Sequence Analysis")
        
        if st.session_state.sequence_analysis:
            analysis = st.session_state.sequence_analysis
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", f"{analysis['length']} aa")
                st.metric("MW", f"{analysis['molecular_weight']:.0f} Da")
            with col2:
                st.metric("Aromaticity", f"{analysis['aromaticity']:.3f}")
                st.metric("Instability", f"{analysis['instability_index']:.2f}")
            with col3:
                st.metric("pI", f"{analysis['isoelectric_point']:.2f}")
                stability = "Stable" if analysis['instability_index'] < 40 else "Unstable"
                st.metric("Stability", stability)
            
            st.markdown("---")
            
            # AA composition
            aa_comp = analysis['aa_composition']
            aa_df = pd.DataFrame({
                'AA': list(aa_comp.keys()),
                'Percentage': [v * 100 for v in aa_comp.values()]
            }).sort_values('Percentage', ascending=False)
            
            fig = px.bar(aa_df, x='AA', y='Percentage',
                        title='Amino Acid Composition',
                        color='Percentage',
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Secondary structure
            sec_struct = analysis['secondary_structure']
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("α-Helix", f"{sec_struct['helix']*100:.1f}%")
            with col2:
                st.metric("β-Turn", f"{sec_struct['turn']*100:.1f}%")
            with col3:
                st.metric("β-Sheet", f"{sec_struct['sheet']*100:.1f}%")
    
    # Tab 4: Download
    with tabs[3]:
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 Download PDB File",
                data=st.session_state.predicted_structure,
                file_name=f"{st.session_state.protein_name.replace(' ', '_')}.pdb",
                mime="chemical/x-pdb",
                use_container_width=True
            )
        
        with col2:
            if st.session_state.sequence_analysis and st.session_state.plddt_scores:
                analysis = st.session_state.sequence_analysis
                _, plddt = st.session_state.plddt_scores
                avg_plddt = sum(plddt) / len(plddt) if len(plddt) > 0 else 0
                
                report = f"""Protein Structure Analysis Report
=====================================
Protein: {st.session_state.protein_name}
Source: {st.session_state.prediction_source}

Sequence Properties:
-------------------
Length: {analysis['length']} aa
Molecular Weight: {analysis['molecular_weight']:.2f} Da
pI: {analysis['isoelectric_point']:.2f}
Instability Index: {analysis['instability_index']:.2f}

Confidence:
-----------
Average pLDDT: {avg_plddt:.2f}
Max pLDDT: {max(plddt):.2f}
Min pLDDT: {min(plddt):.2f}
"""
                
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"{st.session_state.protein_name.replace(' ', '_')}_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with st.expander("👁️ Preview PDB"):
            st.code(st.session_state.predicted_structure[:2000] + "\n...", language="text")
    
    # Tab 5: Reliability Analysis
    with tabs[4]:
        st.subheader("⚗️ Prediction Reliability Analysis")
        
        # Read all hyperparameters
        plddt_threshold = int(st.session_state.get('hp_plddt_threshold', 70))
        window_size     = int(st.session_state.get('hp_window_size', 9))
        disorder_thr    = float(st.session_state.get('hp_disorder_threshold', 0.5))
        auto_trim       = bool(st.session_state.get('hp_auto_trim', True))
        smooth_sigma    = float(st.session_state.get('hp_smooth_sigma', 1.5))
        
        st.info(
            f"🎛️ Active hyperparameters — "
            f"pLDDT threshold: **{plddt_threshold}** | "
            f"Window: **{window_size}** | "
            f"Disorder threshold: **{disorder_thr:.1f}** | "
            f"σ: **{smooth_sigma}** | "
            f"Auto-trim: **{'on' if auto_trim else 'off'}**"
        )
        
        if st.session_state.plddt_scores:
            residues_r, plddt_r = st.session_state.plddt_scores
            
            # ── Reliability Grade Card ────────────────────────────────────────
            reliability = assess_prediction_reliability(plddt_r, threshold=plddt_threshold)
            gc = reliability['grade_color']
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {gc}22, {gc}44);
                        border: 3px solid {gc}; border-radius: 16px;
                        padding: 2rem; text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 5rem; font-weight: 900; color: {gc}; line-height: 1;">
                    {reliability['grade']}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #333; margin-top: 0.3rem;">
                    {reliability['grade_label']} Prediction</div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 0.4rem;">
                    Evaluated at pLDDT &ge; {plddt_threshold}</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg pLDDT", f"{reliability['avg_plddt']:.1f}")
            with col2:
                st.metric("Reliable Residues",
                          f"{reliability['n_reliable']}/{reliability['n_total']}")
            with col3:
                st.metric("% Reliable", f"{reliability['percent_reliable']:.1f}%")
            with col4:
                st.metric("High-Conf Segments",
                          len(reliability['high_confidence_segments']))
            
            if reliability['high_confidence_segments']:
                segs = reliability['high_confidence_segments']
                seg_text = ", ".join([f"Res {s}–{e}" for s, e in segs[:8]])
                if len(segs) > 8:
                    seg_text += f" (+{len(segs) - 8} more)"
                st.success(f"✅ High-Confidence Regions: {seg_text}")
            
            st.markdown("---")
            
            # ── Hydrophobicity Profile ────────────────────────────────────────
            if st.session_state.current_sequence:
                seq = st.session_state.current_sequence
                
                st.subheader(f"🌊 Hydrophobicity Profile  (window = {window_size})")
                st.caption("Kyte-Doolittle scale | Larger window → smoother (↓ variance, ↑ bias)")
                h_pos, h_vals = calculate_hydrophobicity_profile(seq, window_size=window_size)
                
                if h_pos:
                    fig_h = go.Figure()
                    h_colors = ['#dc2626' if v > 0 else '#2563eb' for v in h_vals]
                    fig_h.add_trace(go.Bar(
                        x=h_pos, y=h_vals,
                        marker_color=h_colors,
                        opacity=0.75,
                        name='Hydrophobicity',
                    ))
                    fig_h.add_hline(y=0, line_color='#333', line_width=1)
                    fig_h.update_layout(
                        title=f"Kyte-Doolittle Hydrophobicity (window = {window_size})",
                        xaxis_title="Residue Position",
                        yaxis_title="Hydrophobicity",
                        height=340,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                    col_l, col_r = st.columns(2)
                    col_l.caption("🔴 Red = Hydrophobic (>0)")
                    col_r.caption("🔵 Blue = Hydrophilic (<0)")
                
                st.markdown("---")
                
                # ── Disorder Profile ────────────────────────────────────────
                st.subheader(
                    f"🌀 Disorder Profile  (window={window_size}, threshold={disorder_thr:.1f})"
                )
                st.caption("Higher threshold → stricter (↑ bias, ↓ variance)")
                d_pos, d_scores, d_flags = predict_disorder_profile(
                    seq, window_size=window_size, threshold=disorder_thr
                )
                
                if d_pos:
                    fig_d = go.Figure()
                    d_colors = ['#ef4444' if f else '#10b981' for f in d_flags]
                    fig_d.add_trace(go.Bar(
                        x=d_pos, y=d_scores,
                        marker_color=d_colors,
                        opacity=0.75,
                        name='Disorder Score',
                    ))
                    fig_d.add_hline(
                        y=disorder_thr,
                        line_dash='dash', line_color='#f59e0b',
                        annotation_text=f"Threshold ({disorder_thr:.1f})",
                    )
                    fig_d.update_layout(
                        title=f"Disorder Propensity (threshold = {disorder_thr:.1f})",
                        xaxis_title="Residue Position",
                        yaxis_title="Disorder Score (0-1)",
                        yaxis=dict(range=[0, 1.05]),
                        height=340,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        showlegend=False,
                    )
                    st.plotly_chart(fig_d, use_container_width=True)
                    
                    n_dis = sum(d_flags)
                    pct_d = n_dis / len(d_flags) * 100 if d_flags else 0
                    if n_dis > 0:
                        st.warning(
                            f"⚠️ {n_dis} window positions ({pct_d:.1f}%) classified as "
                            f"disordered at threshold = {disorder_thr:.1f}."
                        )
                    else:
                        st.success("✅ No significant disordered regions detected.")
            
            st.markdown("---")
            
            # ── Terminal Trimming Analysis ────────────────────────────────────
            if auto_trim and st.session_state.predicted_structure and st.session_state.current_sequence:
                st.subheader("✂️ Terminal Trimming  (Variance Reduction)")
                st.caption(
                    f"Removing N/C-terminal residues with pLDDT < {plddt_threshold} "
                    "reduces prediction variance at the cost of small coverage bias."
                )
                t_seq, t_pdb, n_tr, c_tr = trim_low_confidence_termini(
                    st.session_state.current_sequence,
                    st.session_state.predicted_structure,
                    min_plddt=float(plddt_threshold),
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("N-term Trimmed", f"{n_tr} residues")
                with col2:
                    st.metric("C-term Trimmed", f"{c_tr} residues")
                with col3:
                    retained = len(t_seq)
                    total    = len(st.session_state.current_sequence)
                    st.metric("Retained", f"{retained}/{total} aa")
                
                if n_tr > 0 or c_tr > 0:
                    st.info(
                        f"ℹ️ Removing {n_tr + c_tr} low-confidence terminal residues "
                        f"reduces pLDDT variance in the retained core structure."
                    )
                    st.download_button(
                        label="📥 Download Trimmed PDB",
                        data=t_pdb,
                        file_name=(
                            f"{st.session_state.protein_name.replace(' ', '_')}_trimmed.pdb"
                        ),
                        mime="chemical/x-pdb",
                        use_container_width=True,
                    )
                else:
                    st.success(
                        f"✅ No terminal trimming needed — all termini exceed "
                        f"the pLDDT threshold ({plddt_threshold})."
                    )
        else:
            st.info("ℹ️ Run a structure prediction first to see reliability analysis.")


def show_batch_analysis():
    """Batch analysis page."""
    st.header("📊 Batch Sequence Analysis")
    
    st.info("Upload a multi-FASTA file to analyze multiple sequences")
    
    uploaded_file = st.file_uploader("Upload multi-FASTA:", type=['fasta', 'fa', 'txt', 'faa'])
    
    if uploaded_file:
        fasta_content = uploaded_file.read().decode('utf-8')
        sequences = parse_fasta(fasta_content)
        
        if sequences:
            st.success(f"✅ Parsed {len(sequences)} sequences")
            
            with st.expander("📋 Preview"):
                for i, (name, seq) in enumerate(sequences[:10]):
                    st.text(f"{i+1}. {name} ({len(seq)} aa)")
                if len(sequences) > 10:
                    st.text(f"... and {len(sequences) - 10} more")
            
            st.markdown("---")
            
            num_to_process = st.slider("Number to analyze:", 1, min(len(sequences), 50), min(5, len(sequences)))
            
            if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, seq) in enumerate(sequences[:num_to_process]):
                    status_text.text(f"Analyzing {i+1}/{num_to_process}: {name}")
                    
                    is_valid, message = validate_sequence(seq)
                    if is_valid:
                        analysis = analyze_sequence(message)
                        if analysis:
                            results.append({
                                'Name': name,
                                'Length': len(message),
                                'MW (Da)': f"{analysis['molecular_weight']:.0f}",
                                'pI': f"{analysis['isoelectric_point']:.2f}",
                                'Instability': f"{analysis['instability_index']:.2f}",
                                'Stability': 'Stable' if analysis['instability_index'] < 40 else 'Unstable'
                            })
                    
                    progress_bar.progress((i + 1) / num_to_process)
                
                status_text.text("✅ Complete!")
                
                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name="batch_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


def show_comparison():
    """Structure comparison page."""
    st.header("🔄 Structure Comparison (RMSD)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Structure")
        if st.session_state.predicted_structure:
            st.success(f"✅ {st.session_state.protein_name}")
        else:
            st.warning("⚠️ No predicted structure")
    
    with col2:
        st.subheader("Reference Structure")
        ref_file = st.file_uploader("Upload reference PDB:", type=['pdb'])
    
    if st.session_state.predicted_structure and ref_file:
        ref_pdb = ref_file.read().decode('utf-8')
        
        with st.spinner("Calculating RMSD..."):
            pred_coords = parse_pdb_coordinates(st.session_state.predicted_structure)
            ref_coords = parse_pdb_coordinates(ref_pdb)
            
            if len(pred_coords) > 0 and len(ref_coords) > 0 and len(pred_coords) == len(ref_coords):
                rmsd = calculate_rmsd(pred_coords, ref_coords)
                
                if rmsd is not None:
                    st.success("✅ RMSD Calculated")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RMSD (Å)", f"{rmsd:.3f}")
                    with col2:
                        st.metric("Aligned Atoms", len(pred_coords))
                    with col3:
                        quality = "Excellent" if rmsd < 2.0 else "Good" if rmsd < 4.0 else "Fair" if rmsd < 6.0 else "Poor"
                        st.metric("Quality", quality)
            else:
                st.error("❌ Length mismatch or invalid coordinates")


def show_about():
    """About page with professional design."""
    
    # Header
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.12); margin-bottom: 2rem;">
        <h1 style="text-align: center; color: #667eea; margin-bottom: 0.5rem;">ℹ️ About ProteinForge</h1>
        <p style="text-align: center; color: #666; font-size: 1.1rem;">
            Dynamic AI-Powered Protein Structure Prediction Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview Section
    st.markdown("""
    <div class="feature-card">
        <h2 style="color: #667eea; margin-bottom: 1rem;">🧬 Overview</h2>
        <p style="font-size: 1.1rem; line-height: 1.8; color: #555;">
            <strong>ProteinForge</strong> is a cutting-edge protein structure prediction dashboard powered by 
            state-of-the-art AI models. Built for researchers, students, and bioinformatics professionals, 
            it provides instant access to ESMFold and AlphaFold predictions with zero hard-coded data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Features
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">✨ Key Features</h3>
            <ul style="line-height: 2.2; font-size: 1rem; color: #555;">
                <li>✅ <strong>100% Dynamic Data</strong> - Zero hard-coded sequences</li>
                <li>🤖 <strong>Multiple AI Models</strong> - ESMFold & AlphaFold DB</li>
                <li>🎨 <strong>Interactive 3D Viewer</strong> - Rotate, zoom, style</li>
                <li>📊 <strong>Comprehensive Analysis</strong> - Full sequence metrics</li>
                <li>🗄️ <strong>Dataset Integration</strong> - 500K+ sequences</li>
                <li>⚡ <strong>Batch Processing</strong> - Multiple sequences at once</li>
                <li>🔄 <strong>Structure Comparison</strong> - RMSD calculations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">📊 Data Sources</h3>
            <ul style="line-height: 2.2; font-size: 1rem; color: #555;">
                <li>🔗 <strong>UniProt REST API</strong> - Protein sequences & metadata</li>
                <li>📦 <strong>AlphaFold Database</strong> - Pre-computed structures</li>
                <li>⚡ <strong>ESMFold API</strong> - On-demand predictions</li>
                <li>🤗 <strong>Hugging Face</strong> - Curated protein datasets</li>
            </ul>
            <br>
            <h3 style="color: #667eea; margin-bottom: 1rem;">🎯 Quality Metrics</h3>
            <p style="line-height: 2; font-size: 1rem; color: #555;">
                <strong>pLDDT Confidence Scale:</strong><br>
                🔵 <strong>>90</strong>: Very High (~95% accuracy)<br>
                🟢 <strong>70-90</strong>: Confident (reliable)<br>
                🟡 <strong>50-70</strong>: Low (use with caution)<br>
                🟠 <strong><50</strong>: Very Low (likely disordered)
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction Engines
    st.markdown("""
    <div class="feature-card">
        <h2 style="color: #667eea; margin-bottom: 1.5rem; text-align: center;">🔬 Prediction Engines</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);">
            <h3 style="color: #667eea; margin-bottom: 1rem;">⚡ ESMFold API</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li>Fast predictions (30-60 seconds)</li>
                <li>Works with any amino acid sequence</li>
                <li>Based on ESM-2 language model</li>
                <li>State-of-the-art accuracy</li>
                <li>No UniProt ID required</li>
            </ul>
            <br>
            <p style="font-size: 0.9rem; color: #888; font-style: italic;">
                Reference: Lin et al. (2023). Evolutionary-scale prediction of atomic-level 
                protein structure with a language model. Science, 379(6637), 1123-1130.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);">
            <h3 style="color: #10b981; margin-bottom: 1rem;">🏆 AlphaFold Database</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li>Pre-computed high-quality structures</li>
                <li>Instant retrieval (no waiting)</li>
                <li>Requires UniProt accession ID</li>
                <li>200M+ protein structures</li>
                <li>Highest confidence scores</li>
            </ul>
            <br>
            <p style="font-size: 0.9rem; color: #888; font-style: italic;">
                Reference: Jumper et al. (2021). Highly accurate protein structure prediction 
                with AlphaFold. Nature, 596(7873), 583-589.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Limitations & Best Practices
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card" style="border-left: 4px solid #f59e0b;">
            <h3 style="color: #f59e0b; margin-bottom: 1rem;">⚠️ Limitations</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li>Sequence length limited to ~2000 residues</li>
                <li>Monomeric structures only (no complexes)</li>
                <li>Low confidence may indicate disorder</li>
                <li>Post-translational modifications not modeled</li>
                <li>Always validate with experimental data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card" style="border-left: 4px solid #10b981;">
            <h3 style="color: #10b981; margin-bottom: 1rem;">✅ Best Practices</h3>
            <ul style="line-height: 2; font-size: 1rem; color: #555;">
                <li>Check pLDDT scores for reliability</li>
                <li>Compare with AlphaFold DB when available</li>
                <li>Use batch mode for multiple sequences</li>
                <li>Download PDB files for further analysis</li>
                <li>Validate predictions experimentally</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Links
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">🔗 Useful Links</h3>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <a href="https://predictioncenter.org/" target="_blank" style="text-decoration: none;">
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 2px solid #e5e7eb;">
                    <strong style="color: #667eea;">CASP</strong><br>
                    <span style="color: #888; font-size: 0.9rem;">Critical Assessment of Structure Prediction</span>
                </div>
            </a>
            <a href="https://www.uniprot.org/" target="_blank" style="text-decoration: none;">
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 2px solid #e5e7eb;">
                    <strong style="color: #667eea;">UniProt</strong><br>
                    <span style="color: #888; font-size: 0.9rem;">Universal Protein Resource</span>
                </div>
            </a>
            <a href="https://alphafold.ebi.ac.uk/" target="_blank" style="text-decoration: none;">
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 2px solid #e5e7eb;">
                    <strong style="color: #667eea;">AlphaFold DB</strong><br>
                    <span style="color: #888; font-size: 0.9rem;">200M+ protein structures</span>
                </div>
            </a>
            <a href="https://esmatlas.com/" target="_blank" style="text-decoration: none;">
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 2px solid #e5e7eb;">
                    <strong style="color: #667eea;">ESM Atlas</strong><br>
                    <span style="color: #888; font-size: 0.9rem;">ESMFold Metagenomic Atlas</span>
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 16px; text-align: center; color: white; margin-top: 3rem;">
        <h3 style="color: white; margin-bottom: 1rem;">Built with ❤️ for the Bioinformatics Community</h3>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-bottom: 0.5rem;">
            Powered by Streamlit · py3Dmol · Plotly · Hugging Face · Biopython
        </p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">
            © 2026 ProteinForge | 100% Open Source | Zero Hard-Coded Data
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
