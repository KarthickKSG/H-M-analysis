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

# Page Config
st.set_page_config(page_title="NeuroCluster | Stress Analysis", layout="wide")

# Custom CSS for Animations and Styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 20px; transition: 0.3s; }
    .stButton>button:hover { transform: scale(1.05); background-color: #ff4b4b; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .reportview-container { animation: fadeIn 2s; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 Perbandingan K-Means, K-Medoids & Advanced Clustering")
st.subheader("Analisis Tingkat Stress Manusia Berdasarkan Data Spasial & Distribusi")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("country_wise_latest.csv")
    # Mapping COVID metrics to Stress Metrics for the Theme
    df = df.rename(columns={
        "Confirmed": "Stress_Trigger_Frequency",
        "Deaths": "Nervous_Breakdown_Rate",
        "Recovered": "Resilience_Score",
        "Active": "Current_Stress_Load"
    })
    return df

df = load_data()

# Sidebar - Algorithm Selection
st.sidebar.header("🕹️ Control Panel")
selected_algo = st.sidebar.selectbox("Pilih Algoritma", [
    "K-Means (Baseline)",
    "K-Medoids (Baseline)",
    "DBSCAN (Density-Based)",
    "OPTICS (Ordering Points)",
    "HDBSCAN (Hierarchical Density)",
    "Gaussian Mixture (Distribution-Based)",
    "BIRCH (Grid-Based)"
])

features = ["Stress_Trigger_Frequency", "Nervous_Breakdown_Rate", "Resilience_Score", "Current_Stress_Load"]
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Algorithm Logic
clusters = None
model_desc = ""

if selected_algo == "K-Means (Baseline)":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = model.fit_predict(X_scaled)
    model_desc = "K-Means membagi data berdasarkan jarak Euclidean ke centroid."

elif selected_algo == "K-Medoids (Baseline)":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
    model = KMedoids(n_clusters=k, random_state=42)
    clusters = model.fit_predict(X_scaled)
    model_desc = "K-Medoids menggunakan titik data nyata (medoids) sebagai pusat, lebih robust terhadap outlier."

elif selected_algo == "DBSCAN (Density-Based)":
    eps = st.sidebar.slider("Epsilon (Distance)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = model.fit_predict(X_scaled)
    model_desc = "DBSCAN mengelompokkan area dengan kepadatan tinggi dan mendeteksi noise."

elif selected_algo == "OPTICS (Ordering Points)":
    model = OPTICS(min_samples=5)
    clusters = model.fit_predict(X_scaled)
    model_desc = "OPTICS mirip DBSCAN tetapi lebih baik dalam menangani kepadatan yang bervariasi."

elif selected_algo == "HDBSCAN (Hierarchical Density)":
    min_cluster_size = st.sidebar.slider("Min Cluster Size", 2, 20, 5)
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = model.fit_predict(X_scaled)
    model_desc = "HDBSCAN membuat hierarki kepadatan untuk menemukan klaster dengan berbagai bentuk."

elif selected_algo == "Gaussian Mixture (Distribution-Based)":
    n_comp = st.sidebar.slider("Components", 2, 10, 3)
    model = GaussianMixture(n_components=n_comp)
    clusters = model.fit_predict(X_scaled)
    model_desc = "GMM mengasumsikan data berasal dari distribusi Gaussian yang tumpang tindih."

elif selected_algo == "BIRCH (Grid-Based)":
    n_clusters = st.sidebar.slider("Clusters", 2, 10, 3)
    model = Birch(n_clusters=n_clusters)
    clusters = model.fit_predict(X_scaled)
    model_desc = "BIRCH sangat efisien untuk dataset besar menggunakan struktur Tree."

# Add clusters to DF
df['Cluster'] = clusters.astype(str)

# Visualizations
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"### 🚀 Visualisasi: {selected_algo}")
    # Animated 3D Scatter Plot
    fig = px.scatter_3d(
        df, 
        x='Stress_Trigger_Frequency', 
        y='Resilience_Score', 
        z='Current_Stress_Load',
        color='Cluster',
        hover_name='Country/Region',
        template="plotly_dark",
        animation_frame='WHO Region', # Extra Animation
        title=f"Stress Level Mapping ({selected_algo})"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Metrics & Info")
    st.info(model_desc)
    
    if len(set(clusters)) > 1:
        score = silhouette_score(X_scaled, clusters)
        st.metric("Silhouette Score", f"{score:.3f}")
        st.progress(float((score + 1) / 2))
    else:
        st.warning("Hanya satu klaster terdeteksi. Sesuaikan parameter.")

    st.write("Top Countries in Stress Group:")
    st.dataframe(df[['Country/Region', 'Cluster']].head(10), use_container_width=True)

# Comparison Section
st.divider()
st.markdown("### 📈 Animated Trend Analysis")
fig_area = px.area(df.sort_values(by='Stress_Trigger_Frequency'), 
             x="Country/Region", y="Current_Stress_Load", color="Cluster",
             line_group="WHO Region", title="Stress Load Distribution Across Regions",
             template="plotly_dark")
st.plotly_chart(fig_area, use_container_width=True)