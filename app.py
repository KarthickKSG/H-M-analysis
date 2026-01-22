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
st.set_page_config(page_title="NeuroCluster Pro", layout="wide", page_icon="💎")

# --- SMOOTH BLUE GLASSY CSS ---
st.markdown("""
    <style>
    /* Base Background */
    .stApp {
        background: radial-gradient(circle at top right, #0a192f, #020617);
        color: #e2e8f0;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(30, 64, 175, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Titles */
    .main-title {
        background: linear-gradient(90deg, #60a5fa, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Smooth Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        border: none;
        border-radius: 12px;
        color: white;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.markdown("## ⚙️ Logic Engine")
    uploaded_file = st.file_uploader("Upload Mortality CSV", type="csv")
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        
        st.divider()
        st.markdown("### 🎯 Data Slicing")
        genders = df_raw['Gender'].unique().tolist()
        sel_gender = st.selectbox("Focus Gender", genders, index=0)
        
        ages = df_raw['Age'].unique().tolist()
        sel_age = st.multiselect("Focus Age Groups", ages, default=ages)
        
        df_filtered = df_raw[(df_raw['Gender'] == sel_gender) & (df_raw['Age'].isin(sel_age))].copy()
        
        st.divider()
        st.markdown("### 🤖 Algorithm Tuning")
        algo = st.selectbox("Logic", ["K-Means", "K-Medoids", "DBSCAN"])
        k = st.slider("Clusters (K)", 2, 8, 4)
        
        # Mapping numerical columns
        nums = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        f_x = st.selectbox("X-Axis", nums, index=nums.index('Week number'))
        f_y = st.selectbox("Y-Axis", nums, index=nums.index('Year'))
        f_z = st.selectbox("Z-Axis", nums, index=nums.index('Value'))

# --- MAIN INTERFACE ---
st.markdown('<h1 class="main-title">NeuroCluster Analysis</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>Deep Neural Insights into Human Mortality & Excess Stress Patterns</p>", unsafe_allow_html=True)

if uploaded_file is not None:
    # 1. Processing
    X = df_filtered[[f_x, f_y, f_z]].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Clustering logic
    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo == "K-Medoids":
        model = KMedoids(n_clusters=k, random_state=42)
    else:
        model = DBSCAN(eps=0.5, min_samples=3)
        
    labels = model.fit_predict(X_scaled)
    df_filtered['Cluster'] = [f"Pattern {l}" if l != -1 else "Noise" for l in labels]
    
    # --- DASHBOARD LAYOUT ---
    
    # Row 1: Key Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="glass-card"><h6>Algorithm</h6><h3>{algo}</h3></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="glass-card"><h6>Samples</h6><h3>{len(df_filtered)}</h3></div>', unsafe_allow_html=True)
    with m3:
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        st.markdown(f'<div class="glass-card"><h6>Groups</h6><h3>{n_c}</h3></div>', unsafe_allow_html=True)
    with m4:
        try: score = silhouette_score(X_scaled, labels)
        except: score = 0
        st.markdown(f'<div class="glass-card"><h6>Reliability</h6><h3>{score:.2f}</h3></div>', unsafe_allow_html=True)

    # Row 2: Main 3D Chart
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌐 3D Spatial Mortality Mapping")
    fig_3d = px.scatter_3d(
        df_filtered, x=f_x, y=f_y, z=f_z,
        color='Cluster', hover_name='Country',
        symbol='Cluster',
        color_discrete_sequence=px.colors.sequential.Blues_r,
        opacity=0.8, height=600, template="plotly_dark"
    )
    fig_3d.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Multi-Veracity Analysis Tabs
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Temporal Trends", "🎻 Distribution", "☀️ Composition", "🕸️ Radar Profile"])
    
    with tab1:
        st.markdown("#### Mortality Evolution Over Time")
        # Creating a timeline for smooth plotting
        df_filtered['Timeline'] = df_filtered['Year'] + (df_filtered['Week number'] / 53)
        fig_line = px.line(df_filtered.sort_values('Timeline'), 
                           x='Timeline', y='Value', color='Cluster',
                           line_shape='spline', render_mode='svg',
                           template="plotly_dark", color_discrete_sequence=px.colors.sequential.Cyan_r)
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.markdown("#### Statistical Cluster Density")
        fig_violin = px.violin(df_filtered, y='Value', x='Cluster', color='Cluster',
                               box=True, points="all", template="plotly_dark",
                               color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(fig_violin, use_container_width=True)

    with tab3:
        st.markdown("#### Hierarchical Population Breakout")
        fig_sun = px.sunburst(df_filtered, path=['Cluster', 'Country', 'Age'], values='Value',
                              color='Value', color_continuous_scale='Blues',
                              template="plotly_dark")
    st.plotly_chart(fig_sun, use_container_width=True)

    with tab4:
        st.markdown("#### Comparative Cluster DNA")
        # Aggregating for radar
        radar_df = df_filtered.groupby('Cluster')[[f_x, f_y, f_z]].mean().reset_index()
        fig_radar = go.Figure()
        for i, row in radar_df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[f_x], row[f_y], row[f_z], row[f_x]],
                theta=[f_x, f_y, f_z, f_x],
                fill='toself',
                name=row['Cluster']
            ))
        fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=True, gridcolor="#334155")))
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("💎 Welcome to the Premium NeuroCluster Analysis. Please upload your CSV to initiate the glass-morphic dashboard.")
