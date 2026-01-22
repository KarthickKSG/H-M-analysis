import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="NeuroCluster Premium", layout="wide", page_icon="💎")

# --- SMOOTH BLUE GLASSY CSS ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #051937, #000c1d);
        color: #e2e8f0;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 25px;
        margin-bottom: 25px;
    }
    .main-title {
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3rem;
        text-align: center;
    }
    /* Style Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.markdown("## ⚙️ Logic Control")
    uploaded_file = st.file_uploader("Upload CSV Data", type="csv")
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        
        st.divider()
        nums = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        f_x = st.selectbox("X-Axis (Spatial)", nums, index=0)
        f_y = st.selectbox("Y-Axis (Temporal)", nums, index=min(1, len(nums)-1))
        f_z = st.selectbox("Z-Axis (Mortality)", nums, index=min(2, len(nums)-1))
        
        st.divider()
        algo = st.selectbox("Algorithm", ["K-Means", "K-Medoids", "DBSCAN"])
        k = st.slider("Target Clusters", 2, 8, 4)
        
        # Color Palette Selection
        palette = st.selectbox("Cluster Color Theme", ["Prism", "Vivid", "Bold", "Safe"])
        palette_map = {"Prism": px.colors.qualitative.Prism, "Vivid": px.colors.qualitative.Vivid, 
                       "Bold": px.colors.qualitative.Bold, "Safe": px.colors.qualitative.Safe}

# --- MAIN INTERFACE ---
st.markdown('<h1 class="main-title">NeuroCluster Analysis</h1>', unsafe_allow_html=True)

if uploaded_file is not None:
    # 1. Processing Logic
    X = df_raw[[f_x, f_y, f_z]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo == "K-Medoids":
        model = KMedoids(n_clusters=k, random_state=42)
    else:
        model = DBSCAN(eps=0.5, min_samples=3)
        
    labels = model.fit_predict(X_scaled)
    df_raw['Cluster'] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]
    
    # --- TOP ROW METRICS ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="glass-card"><h4>Engine</h4><h2>{algo}</h2></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="glass-card"><h4>Groups</h4><h2>{len(set(labels))}</h2></div>', unsafe_allow_html=True)
    with m3:
        try: score = silhouette_score(X_scaled, labels)
        except: score = 0
        st.markdown(f'<div class="glass-card"><h4>Reliability</h4><h2>{score:.2f}</h2></div>', unsafe_allow_html=True)

    # --- MAIN 3D CHART (SHAPE: CIRCLE, COLORS: DIFFERENT) ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌐 3D Spatial Pattern Mapping")
    
    fig_3d = px.scatter_3d(
        df_raw, x=f_x, y=f_y, z=f_z,
        color='Cluster',
        hover_name=df_raw.columns[0],
        # Forces the marker to be a circle (sphere in 3D)
        symbol_sequence=['circle'], 
        # Uses the selected different-colored palette
        color_discrete_sequence=palette_map[palette],
        opacity=0.85, height=700, template="plotly_dark"
    )
    
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            zaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- MULTI-VERACITY CHART SECTION ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📉 Temporal Spline", "☀️ Cluster Composition", "🎻 Spread Density"])
    
    with tab1:
        st.markdown("### Smooth Trend Analysis")
        # Creating a unique timeline for sorting
        if 'Year' in df_raw.columns and 'Week number' in df_raw.columns:
            df_raw['Timeline'] = df_raw['Year'] + (df_raw['Week number'] / 53)
            sort_col = 'Timeline'
        else:
            sort_col = f_x
            
        fig_line = px.line(
            df_raw.sort_values(sort_col), 
            x=sort_col, y=f_z, color='Cluster',
            line_shape='spline', # Makes the line smooth
            color_discrete_sequence=palette_map[palette],
            template="plotly_dark"
        )
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.markdown("### Population Hierarchy")
        # Sunburst chart showing how data is nested
        path_cols = ['Cluster']
        if 'Country' in df_raw.columns: path_cols.append('Country')
        if 'Age' in df_raw.columns: path_cols.append('Age')
        
        fig_sun = px.sunburst(
            df_raw, path=path_cols, values=f_z,
            color='Cluster',
            color_discrete_sequence=palette_map[palette],
            template="plotly_dark"
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    with tab3:
        st.markdown("### Cluster Value Distribution")
        fig_violin = px.violin(
            df_raw, y=f_z, x='Cluster', color='Cluster',
            box=True, points="all",
            color_discrete_sequence=palette_map[palette],
            template="plotly_dark"
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👋 Upload a CSV file in the sidebar to generate the glassy analysis dashboard.")
