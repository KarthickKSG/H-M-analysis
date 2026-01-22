import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="NeuroCluster Ultra-Fast", layout="wide", page_icon="⚡")

# --- CUSTOM STYLING (Blue Glassy) ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #001f3f, #000814); color: #f8fafc; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px; padding: 20px; margin-bottom: 20px;
    }
    .title-text {
        background: linear-gradient(90deg, #38bdf8, #2563eb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ⚡ PERFORMANCE ENGINES (CACHING) ---

@st.cache_data(show_spinner=False)
def load_and_preprocess(file):
    """Cache data loading to avoid re-reading CSV on every interaction."""
    df = pd.read_csv(file)
    # Downsample if data is too massive (>50k rows) to keep Plotly fast
    if len(df) > 50000:
        df = df.sample(50000)
    return df

@st.cache_data(show_spinner=False)
def run_clustering(data, features, algo_type, k_clusters):
    """Cache the ML logic. Only reruns if parameters or data change."""
    X = data[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algo_type == "K-Means":
        model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    else:
        model = KMedoids(n_clusters=k_clusters, random_state=42)
        
    labels = model.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels) if k_clusters > 1 else 0
    return labels, sil_score, X_scaled

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚡ Fast Controls")
    uploaded_file = st.file_uploader("Upload CSV", type=\"csv\")
    
    if uploaded_file:
        df_raw = load_and_preprocess(uploaded_file)
        nums = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        fx = st.selectbox("X-Axis", nums, index=0)
        fy = st.selectbox("Y-Axis", nums, index=min(1, len(nums)-1))
        fz = st.selectbox("Z-Axis", nums, index=min(2, len(nums)-1))
        
        st.divider()
        algo = st.radio("Clustering Method", ["K-Means", "K-Medoids"], horizontal=True)
        k = st.select_slider("Cluster Count", options=range(2, 9), value=4)
        
        palette = px.colors.qualitative.G10

# --- MAIN UI ---
st.markdown('<h1 class="title-text">NeuroCluster Ultra-Fast</h1>', unsafe_allow_html=True)

if uploaded_file:
    # ⚡ Run Cached Analysis
    features = [fx, fy, fz]
    labels, sil_score, X_scaled = run_clustering(df_raw, features, algo, k)
    df_raw['Cluster'] = [f"Pattern {l}" for l in labels]

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Processing Time", "Instant (Cached)")
    col2.metric("Total Samples", len(df_raw))
    col3.metric("Silhouette Score", f"{sil_score:.2f}")

    # --- 3D CHART ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_3d = px.scatter_3d(df_raw, x=fx, y=fy, z=fz, color='Cluster',
                           symbol_sequence=['circle'], color_discrete_sequence=palette,
                           template="plotly_dark", height=600)
    # Optimized Plotly Rendering
    fig_3d.update_traces(marker=dict(size=4)) 
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📉 Trends", "🧩 Composition", "📊 Distribution"])
    
    with tab1:
        st.markdown("### Smooth Trend Analysis")
        # Optimization: Timeline calculation is also cached inside the logic if needed
        if 'Year' in df_raw.columns and 'Week number' in df_raw.columns:
            df_raw['Timeline'] = df_raw['Year'] + (df_raw['Week number'] / 53)
            sort_col = 'Timeline'
        else: sort_col = fx
        
        fig_line = px.line(df_raw.sort_values(sort_col), x=sort_col, y=fz, color='Cluster',
                           line_shape='spline', color_discrete_sequence=palette, template="plotly_dark")
        st.plotly_chart(fig_line, use_container_width=True)

    with tab2:
        st.markdown("### Population Hierarchy")
        fig_sun = px.sunburst(df_raw, path=['Cluster', 'Country'] if 'Country' in df_raw.columns else ['Cluster'], 
                              values=fz, color='Cluster', color_discrete_sequence=palette, template="plotly_dark")
        st.plotly_chart(fig_sun, use_container_width=True)

    with tab3:
        st.markdown("### Distribution Density")
        fig_violin = px.violin(df_raw, y=fz, x='Cluster', color='Cluster', box=True, 
                               color_discrete_sequence=palette, template="plotly_dark")
        st.plotly_chart(fig_violin, use_container_width=True)

else:
    st.info("⚡ Upload data to see the cached high-speed performance.")
