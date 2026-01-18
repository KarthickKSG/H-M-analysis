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
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import hdbscan
from sklearn.metrics import silhouette_score
import io

# --- 3. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="NeuroCluster | Human Stress Analysis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div.stButton > button:first-child {
        background-color: #ff4b4b; color: white; border-radius: 10px; height: 3em; width: 100%;
        transition: all 0.3s ease-in-out;
    }
    div.stButton > button:hover { transform: scale(1.02); border: 2px solid white; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .reportview-container { animation: fadeIn 1.5s; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. DATA LOADING & MAPPING ---
@st.cache_data
def load_data():
    # Using the provided dataset structure
    csv_data = """Country/Region,Confirmed,Deaths,Recovered,Active,New cases,New deaths,New recovered,Deaths / 100 Cases,Recovered / 100 Cases,Deaths / 100 Recovered,Confirmed last week,1 week change,1 week % increase,WHO Region
Afghanistan,36263,1269,25198,9796,106,10,18,3.5,69.49,5.04,35526,737,2.07,Eastern Mediterranean
Albania,4880,144,2745,1991,117,6,63,2.95,56.25,5.25,4171,709,17.0,Europe
Algeria,27973,1163,18837,7973,616,8,749,4.16,67.34,6.17,23691,4282,18.07,Africa
Argentina,167416,3059,72575,91782,4890,120,2057,1.83,43.35,4.21,130774,36642,28.02,Americas
Australia,15303,167,9311,5825,368,6,137,1.09,60.84,1.79,12428,2875,23.13,Western Pacific
Brazil,2442375,87618,1846641,508116,23284,614,33728,3.59,75.61,4.74,2118646,323729,15.28,Americas
Egypt,92482,4652,34838,52992,420,46,1007,5.03,37.67,13.35,88402,4080,4.62,Eastern Mediterranean
India,1480073,33408,951166,495499,44457,637,33598,2.26,64.26,3.51,1155338,324735,28.11,South-East Asia
US,4290259,148011,1325804,2816444,56336,1076,27941,3.45,30.9,11.16,3834677,455582,11.88,Americas"""
    
    # In a real app, use pd.read_csv("country_wise_latest.csv")
    df = pd.read_csv(io.StringIO(csv_data))
    
    # Mapping to Stress Metrics for the human stress context
    df = df.rename(columns={
        "Confirmed": "Stress_Trigger_Frequency",
        "Deaths": "Breakdown_Rate",
        "Recovered": "Psychological_Resilience",
        "Active": "Current_Stress_Load"
    })
    return df

df = load_data()

# --- 5. SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3663/3663335.png", width=100)
st.sidebar.title("NeuroControl")
selected_algo = st.sidebar.selectbox("Choose Algorithm", [
    "K-Means (Centroid)",
    "K-Medoids (Exemplar)",
    "DBSCAN (Density)",
    "OPTICS (Ordering Points)",
    "HDBSCAN (Hierarchical Density)",
    "Gaussian Mixture (Distribution)",
    "BIRCH (Grid-Based)"
])

# Feature Selection for Clustering
features = ["Stress_Trigger_Frequency", "Breakdown_Rate", "Psychological_Resilience", "Current_Stress_Load"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. CLUSTERING LOGIC ---
clusters = None
info_text = ""

if selected_algo == "K-Means (Centroid)":
    k = st.sidebar.slider("Clusters (k)", 2, 5, 3)
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_scaled)
    info_text = "K-Means partitions data into k clusters where each point belongs to the cluster with the nearest mean."

elif selected_algo == "K-Medoids (Exemplar)":
    k = st.sidebar.slider("Clusters (k)", 2, 5, 3)
    model = KMedoids(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_scaled)
    info_text = "K-Medoids uses actual data points as centers, making it more robust to extreme outliers in stress levels."

elif selected_algo == "DBSCAN (Density)":
    eps = st.sidebar.slider("Epsilon (Radius)", 0.1, 2.0, 0.8)
    model = DBSCAN(eps=eps, min_samples=2)
    clusters = model.fit_predict(X_scaled)
    info_text = "DBSCAN finds core stress patterns and marks isolated cases as noise (-1)."

elif selected_algo == "OPTICS (Ordering Points)":
    model = OPTICS(min_samples=2)
    clusters = model.fit_predict(X_scaled)
    info_text = "OPTICS identifies clusters of varying density, ideal for complex human behavioral data."

elif selected_algo == "HDBSCAN (Hierarchical Density)":
    model = hdbscan.HDBSCAN(min_cluster_size=2)
    clusters = model.fit_predict(X_scaled)
    info_text = "HDBSCAN creates a hierarchy of clusters, effectively separating subtle stress levels from intense ones."

elif selected_algo == "Gaussian Mixture (Distribution)":
    n = st.sidebar.slider("Components", 2, 5, 3)
    model = GaussianMixture(n_components=n)
    clusters = model.fit_predict(X_scaled)
    info_text = "GMM models data as a mixture of multiple Gaussian distributions (Probabilistic clustering)."

elif selected_algo == "BIRCH (Grid-Based)":
    model = Birch(n_clusters=3)
    clusters = model.fit_predict(X_scaled)
    info_text = "BIRCH uses a Clustering Feature Tree to quickly process data in memory-constrained environments."

df['Cluster'] = clusters.astype(str)

# --- 7. MAIN UI & EXTRAORDINARY ANIMATIONS ---
st.title("🧠 Human Stress Level Clustering Analysis")
st.markdown("### Perbandingan K-Means, K-Medoids, dan Algoritma Lanjut")

col1, col2 = st.columns([3, 1])

with col1:
    # 3D Animated Scatter Plot
    fig = px.scatter_3d(
        df, 
        x='Stress_Trigger_Frequency', 
        y='Psychological_Resilience', 
        z='Current_Stress_Load',
        color='Cluster',
        size='Breakdown_Rate',
        hover_name='Country/Region',
        animation_frame='Cluster', # Animation by cluster reveal
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(
        scene=dict(xaxis_title='Triggers', yaxis_title='Resilience', zaxis_title='Load'),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.info(f"**Current Algorithm:** {selected_algo}")
    st.write(info_text)
    
    if len(set(clusters)) > 1:
        score = silhouette_score(X_scaled, clusters)
        st.metric("Model Quality (Silhouette)", f"{score:.2f}")
    
    st.write("---")
    st.write("**Quick Data View:**")
    st.dataframe(df[['Country/Region', 'Cluster']].head(10))

# --- 8. TREND ANIMATION ---
st.subheader("📊 Regional Stress Intensity Trend")
fig_bar = px.bar(df, x="Country/Region", y="Current_Stress_Load", color="Cluster",
             animation_frame="WHO Region", hover_data=features,
             template="plotly_dark", barmode="group")
st.plotly_chart(fig_bar, use_container_width=True)
