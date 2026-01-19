# --- 1. COMPATIBILITY PATCH (FOR PYTHON 3.12+) ---
import sys
try:
    import distutils.version
except ImportError:
    import looseversion
    import types
    distutils = types.ModuleType("distutils")
    distutils.version = types.ModuleType("version")
    distutils.version.LooseVersion = looseversion.LooseVersion
    sys.modules["distutils"] = distutils
    sys.modules["distutils.version"] = distutils.version

# --- 2. IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import hdbscan
from sklearn.metrics import silhouette_score
import time

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroCluster | Stress Analysis", layout="wide", page_icon="🧠")

# Professional Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em; 
        background-image: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white; border: none; font-weight: bold; transition: 0.4s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 10px 20px rgba(0,0,0,0.4); }
    .upload-box { border: 2px dashed #444; padding: 30px; border-radius: 15px; text-align: center; }
    .metric-card { background: #1e2130; padding: 20px; border-radius: 15px; border-left: 5px solid #2575fc; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. LANDING & UPLOAD ---
st.title("🧠 NeuroCluster AI")
st.subheader("Clustering Analisis Tingkat Stress Manusia")

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

# Step 1: File Upload
uploaded_file = st.file_uploader("📁 Step 1: Upload CSV Data to Start Analysis", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Successfully loaded {uploaded_file.name} ({len(df)} rows)")
    
    # Show Data Preview
    with st.expander("👀 View Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)

    # Step 2: Configuration columns
    st.divider()
    st.markdown("### ⚙️ Step 2: Configure Parameters")
    
    col_cfg1, col_cfg2 = st.columns([1, 1])
    
    with col_cfg1:
        st.markdown("#### Feature Mapping")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 3:
            st.error("Error: Dataset must have at least 3 numerical columns.")
            st.stop()
            
        feat_x = st.selectbox("X-Axis (e.g., Stress Triggers)", num_cols, index=0)
        feat_y = st.selectbox("Y-Axis (e.g., Resilience)", num_cols, index=1)
        feat_z = st.selectbox("Z-Axis (e.g., Current Load)", num_cols, index=2)
        
    with col_cfg2:
        st.markdown("#### Algorithm Selection")
        selected_algo = st.selectbox("Choose Clustering Logic", [
            "K-Means (Centroid-based)",
            "K-Medoids (Exemplar-based)",
            "DBSCAN (Density-based)",
            "OPTICS (Multi-density)",
            "HDBSCAN (Hierarchical)",
            "Gaussian Mixture (Probabilistic)",
            "BIRCH (Grid-based)"
        ])
        
        # Dynamic hyperparams
        algo_params = {}
        if "K-Means" in selected_algo or "K-Medoids" in selected_algo:
            algo_params['k'] = st.slider("Number of Clusters (k)", 2, 10, 3)
        elif "DBSCAN" in selected_algo:
            algo_params['eps'] = st.slider("Epsilon (Radius)", 0.1, 5.0, 0.5)
            algo_params['min_samples'] = st.number_input("Min Samples", 2, 20, 5)

    # Step 3: Trigger Analysis
    if st.button("🚀 START CLUSTERING ANALYSIS"):
        st.session_state.analyzed = True
        with st.spinner("Initializing NeuroCluster Engines..."):
            time.sleep(1.5) # For dramatic effect

    # --- 5. EXECUTION & VISUALIZATION ---
    if st.session_state.analyzed:
        st.divider()
        
        # Prepare Data
        X = df[[feat_x, feat_y, feat_z]].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Modeling
        try:
            if "K-Means" in selected_algo:
                model = KMeans(n_clusters=algo_params['k'], random_state=42)
            elif "K-Medoids" in selected_algo:
                model = KMedoids(n_clusters=algo_params['k'], random_state=42)
            elif "DBSCAN" in selected_algo:
                model = DBSCAN(eps=algo_params['eps'], min_samples=algo_params['min_samples'])
            elif "OPTICS" in selected_algo:
                model = OPTICS(min_samples=5)
            elif "HDBSCAN" in selected_algo:
                model = hdbscan.HDBSCAN(min_cluster_size=5)
            elif "Gaussian Mixture" in selected_algo:
                model = GaussianMixture(n_components=3)
            elif "BIRCH" in selected_algo:
                model = Birch(n_clusters=3)
            
            labels = model.fit_predict(X_scaled)
            df['Cluster'] = labels.astype(str)
            
            # --- Results Dashboard ---
            res1, res2, res3 = st.columns(3)
            with res1:
                st.markdown(f'<div class="metric-card"><h4>Algorithm</h4><h2>{selected_algo.split()[0]}</h2></div>', unsafe_allow_html=True)
            with res2:
                n_clusters = len(np.unique(labels[labels != -1]))
                st.markdown(f'<div class="metric-card"><h4>Clusters Found</h4><h2>{n_clusters}</h2></div>', unsafe_allow_html=True)
            with res3:
                if n_clusters > 1:
                    score = silhouette_score(X_scaled, labels)
                    st.markdown(f'<div class="metric-card"><h4>Silhouette Score</h4><h2>{score:.2f}</h2></div>', unsafe_allow_html=True)
            
            # --- Extraordinary Animations ---
            st.markdown("### 🌐 Dynamic 3D Stress Mapping")
            
            fig = px.scatter_3d(
                df, x=feat_x, y=feat_y, z=feat_z,
                color='Cluster',
                hover_name=df.columns[0],
                symbol='Cluster',
                opacity=0.8,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Alphabet,
                animation_frame='Cluster' if "K-Means" in selected_algo else None # Animate group appearance
            )
            
            fig.update_layout(
                scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
                margin=dict(l=0, r=0, b=0, t=0),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison View
            st.markdown("### 📊 Regional Cluster Distribution")
            fig_bar = px.histogram(df, x="Cluster", color="Cluster", 
                                   animation_frame="Cluster",
                                   template="plotly_dark", barmode="group")
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

else:
    # Initial State Instruction
    st.info("👋 Welcome! Please upload a CSV file to begin the human stress clustering process.")
    st.markdown("""
    **Required Data Format:**
    - A CSV file containing at least 3 numerical columns (e.g., Stress Score, Heart Rate, Hours of Sleep).
    - Optional: A text column for names/identifiers (e.g., Country or Participant ID).
    """)
