import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import hdbscan
from sklearn.metrics import silhouette_score
import time

# --- CONFIG ---
st.set_page_config(page_title="NeuroCluster | Excess Mortality", layout="wide", page_icon="📊")

# --- CYBER STYLING ---
st.markdown("""
    <style>
    .stApp { background: #0e1117; color: white; }
    section[data-testid="stSidebar"] { background-color: rgba(20, 25, 35, 0.8); }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4facfe;
        margin-bottom: 10px;
    }
    h1 { color: #4facfe; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload Excess Deaths CSV", type="csv")
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        
        st.divider()
        st.markdown("### 🔍 Filter Population")
        # Allow user to narrow down the data (e.g., just 'Total' gender or specific 'Age')
        unique_genders = df_raw['Gender'].unique().tolist()
        sel_gender = st.multiselect("Select Gender", unique_genders, default=unique_genders[0])
        
        unique_ages = df_raw['Age'].unique().tolist()
        sel_age = st.multiselect("Select Age Group", unique_ages, default=unique_ages)
        
        # Apply Filters
        df_filtered = df_raw[(df_raw['Gender'].isin(sel_gender)) & (df_raw['Age'].isin(sel_age))]
        
        st.divider()
        st.markdown("### 🧬 Axis Mapping")
        num_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        
        # Default mapping for your dataset
        feat_x = st.selectbox("X-Axis (Temporal)", num_cols, index=num_cols.index('Week number') if 'Week number' in num_cols else 0)
        feat_y = st.selectbox("Y-Axis (Yearly)", num_cols, index=num_cols.index('Year') if 'Year' in num_cols else 0)
        feat_z = st.selectbox("Z-Axis (Impact/Value)", num_cols, index=num_cols.index('Value') if 'Value' in num_cols else 0)
        
        st.divider()
        selected_algo = st.selectbox("Clustering Engine", ["K-Means", "K-Medoids", "DBSCAN", "Gaussian Mixture"])
        k_val = st.slider("Clusters (K)", 2, 10, 4) if selected_algo != "DBSCAN" else None
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5) if selected_algo == "DBSCAN" else None

# --- MAIN CONTENT ---
st.title("🧠 NeuroCluster: Excess Mortality Analysis")

if uploaded_file is None:
    st.info("👈 Please upload your Excess Deaths CSV to begin.")
    st.markdown("""
    **Dataset detected structure:**
    - **Countries:** Groups like CZE, NLD, SWE.
    - **Time:** Week and Year tracking.
    - **Impact:** Excess deaths ('Value' column).
    """)
else:
    # Prepare Data
    X = df_filtered[[feat_x, feat_y, feat_z]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if st.sidebar.button("🚀 RUN CLUSTER ANALYSIS"):
        try:
            if selected_algo == "K-Means":
                model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            elif selected_algo == "K-Medoids":
                model = KMedoids(n_clusters=k_val, random_state=42)
            elif selected_algo == "DBSCAN":
                model = DBSCAN(eps=eps, min_samples=3)
            elif selected_algo == "Gaussian Mixture":
                model = GaussianMixture(n_components=k_val)
            
            labels = model.fit_predict(X_scaled)
            df_filtered['Cluster'] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]
            
            # --- RESULTS ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card">Target: <b>{selected_algo}</b></div>', unsafe_allow_html=True)
            with col2:
                n_c = len(set(labels)) - (1 if -1 in labels else 0)
                st.markdown(f'<div class="metric-card">Groups Found: <b>{n_c}</b></div>', unsafe_allow_html=True)
            with col3:
                score = silhouette_score(X_scaled, labels) if n_c > 1 else 0
                st.markdown(f'<div class="metric-card">Reliability: <b>{score:.2f}</b></div>', unsafe_allow_html=True)

            # --- 3D VISUALIZATION ---
            st.markdown("### 🌐 3D Mortality Clusters")
            fig = px.scatter_3d(
                df_filtered, x=feat_x, y=feat_y, z=feat_z,
                color='Cluster',
                hover_name='Country',
                hover_data=['Gender', 'Age', 'Value'],
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Safe,
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- ANALYSIS ---
            t1, t2 = st.tabs(["📊 Statistics", "📋 Data View"])
            with t1:
                st.markdown("#### Average Impact per Cluster")
                avg_impact = df_filtered.groupby('Cluster')[['Value']].mean()
                st.bar_chart(avg_impact)
            with t2:
                st.dataframe(df_filtered)
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
