# --- 1. COMPATIBILITY PATCH (MUST BE AT THE VERY TOP) ---
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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import hdbscan
from sklearn.metrics import silhouette_score
import io

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="NeuroCluster Pro | Custom Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #ff4b4b; color: white; border-radius: 10px; height: 3em; width: 100%;
        transition: all 0.3s ease-in-out;
    }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4259; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. DATA HANDLING LOGIC ---
@st.cache_data
def load_default_data():
    csv_data = """Country/Region,Confirmed,Deaths,Recovered,Active,WHO Region
    Afghanistan,36263,1269,25198,9796,Eastern Mediterranean
    Albania,4880,144,2745,1991,Europe
    Algeria,27973,1163,18837,7973,Africa
    Argentina,167416,3059,72575,91782,Americas
    Australia,15303,167,9311,5825,Western Pacific
    Brazil,2442375,87618,1846641,508116,Americas
    Egypt,92482,4652,34838,52992,Eastern Mediterranean
    India,1480073,33408,951166,495499,South-East Asia
    US,4290259,148011,1325804,2816444,Americas"""
    return pd.read_csv(io.StringIO(csv_data))

# Sidebar: File Upload
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3663/3663335.png", width=80)
st.sidebar.title("Data Center")
uploaded_file = st.sidebar.file_uploader("Upload your Stress Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom Dataset Loaded!")
else:
    df = load_default_data()
    st.sidebar.info("Using Default Example Dataset")

# Sidebar: Feature Mapping (Dynamic)
st.sidebar.subheader("Feature Mapping")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 3:
    st.error("Dataset needs at least 3 numerical columns for 3D Clustering.")
    st.stop()

# Let user choose which columns represent which stress metric
feat_x = st.sidebar.selectbox("Stress Triggers (X-Axis)", numeric_cols, index=0)
feat_y = st.sidebar.selectbox("Resilience Score (Y-Axis)", numeric_cols, index=min(1, len(numeric_cols)-1))
feat_z = st.sidebar.selectbox("Stress Load (Z-Axis)", numeric_cols, index=min(2, len(numeric_cols)-1))
feat_size = st.sidebar.selectbox("Breakdown Rate (Bubble Size)", numeric_cols, index=min(3, len(numeric_cols)-1))

selected_features = [feat_x, feat_y, feat_z, feat_size]

# Sidebar: Algorithm Selection
st.sidebar.divider()
st.sidebar.subheader("Algorithm Settings")
selected_algo = st.sidebar.selectbox("Select Model", [
    "K-Means (Centroid)", "K-Medoids (Exemplar)", "DBSCAN (Density)", 
    "OPTICS (Ordering)", "HDBSCAN (Hierarchical)", "Gaussian Mixture", "BIRCH"
])

# --- 5. CLUSTERING ENGINE ---
X = df[selected_features].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clusters = None
if selected_algo == "K-Means (Centroid)":
    k = st.sidebar.slider("Clusters", 2, 8, 3)
    clusters = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
elif selected_algo == "K-Medoids (Exemplar)":
    k = st.sidebar.slider("Clusters", 2, 8, 3)
    clusters = KMedoids(n_clusters=k, random_state=42).fit_predict(X_scaled)
elif selected_algo == "DBSCAN (Density)":
    eps = st.sidebar.slider("Epsilon", 0.1, 2.0, 0.5)
    clusters = DBSCAN(eps=eps, min_samples=2).fit_predict(X_scaled)
elif selected_algo == "OPTICS (Ordering)":
    clusters = OPTICS(min_samples=2).fit_predict(X_scaled)
elif selected_algo == "HDBSCAN (Hierarchical)":
    clusters = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(X_scaled)
elif selected_algo == "Gaussian Mixture":
    n = st.sidebar.slider("Components", 2, 8, 3)
    clusters = GaussianMixture(n_components=n).fit_predict(X_scaled)
elif selected_algo == "BIRCH":
    clusters = Birch(n_clusters=3).fit_predict(X_scaled)

df['Cluster_Label'] = clusters.astype(str)

# --- 6. MAIN DASHBOARD ---
st.title("🧠 NeuroCluster: Compare & Analyze")
st.markdown(f"Currently comparing clusters using **{selected_algo}** across selected features.")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Total Samples", len(df))
with m2:
    st.metric("Detected Clusters", len(np.unique(clusters)))
with m3:
    if len(np.unique(clusters)) > 1:
        score = silhouette_score(X_scaled, clusters)
        st.metric("Clustering Accuracy (Silh.)", f"{score:.2f}")

st.divider()

# EXTRAORDINARY 3D ANIMATION
st.subheader("🌐 3D Dynamic Stress Map")
fig_3d = px.scatter_3d(
    df, x=feat_x, y=feat_y, z=feat_z,
    color='Cluster_Label', size=feat_size,
    hover_name=df.columns[0], # Usually the name/ID column
    animation_frame='Cluster_Label', # Smooth transition between clusters
    template="plotly_dark",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    height=700
)
fig_3d.update_layout(scene=dict(bgcolor='black'))
st.plotly_chart(fig_3d, use_container_width=True)

# TABULAR COMPARISON
st.subheader("📋 Feature Distribution by Cluster")
col_a, col_b = st.columns([2, 1])

with col_a:
    # Animated Bar Chart
    fig_bar = px.bar(
        df, x='Cluster_Label', y=feat_z, 
        color='Cluster_Label',
        animation_frame='Cluster_Label',
        title=f"Distribution of {feat_z}",
        template="plotly_dark"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.write("**Cluster Summary Data**")
    summary = df.groupby('Cluster_Label')[selected_features].mean()
    st.dataframe(summary.style.background_gradient(cmap='Blues'), use_container_width=True)

# DOWNLOAD SECTION
st.divider()
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Clustered Data", data=csv, file_name="clustered_stress_data.csv", mime="text/csv")
