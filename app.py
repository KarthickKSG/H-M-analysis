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
st.set_page_config(page_title="NeuroCluster Ultra", layout="wide", page_icon="⚡")

# --- PREMIUM GLASSY CSS ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #001f3f, #000814); color: #f8fafc; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px; padding: 25px; margin-bottom: 25px;
    }
    .title-text {
        background: linear-gradient(90deg, #38bdf8, #2563eb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ⚡ SPEED OPTIMIZATION (CACHING) ---
@st.cache_data(show_spinner=False)
def get_data(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner=False)
def process_clustering(df, feats, algo, k):
    X = df[feats].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if algo == "K-Means":
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    else:
        model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if k > 1 else 0
    return labels, score

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🎛️ Engine Controls")
    uploaded_file = st.file_uploader("Source CSV", type="csv")
    if uploaded_file:
        df_raw = get_data(uploaded_file)
        nums = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        f_x = st.selectbox("X-Axis (Spatial)", nums, index=0)
        f_y = st.selectbox("Y-Axis (Temporal)", nums, index=min(1, len(nums)-1))
        f_z = st.selectbox("Z-Axis (Intensity)", nums, index=min(2, len(nums)-1))
        
        st.divider()
        algo = st.radio("Method", ["K-Means", "K-Medoids"], horizontal=True)
        k = st.select_slider("Clusters", options=range(2, 9), value=4)
        palette = px.colors.qualitative.G10

# --- MAIN DASHBOARD ---
st.markdown('<h1 class="title-text">NeuroCluster Pro</h1>', unsafe_allow_html=True)

if uploaded_file:
    # ⚡ Instant Execution
    labels, sil_score = process_clustering(df_raw, [f_x, f_y, f_z], algo, k)
    df_raw['Cluster'] = [f"Group {l}" for l in labels]

    # 3D Spatial Map
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig_3d = px.scatter_3d(df_raw, x=f_x, y=f_y, z=f_z, color='Cluster', 
                           symbol_sequence=['circle'], color_discrete_sequence=palette,
                           template="plotly_dark", height=600)
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Multi-Veracity Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Comparison", "📈 Trend", "🧩 Composition", "📊 Distribution"])

    with tab1:
        st.markdown("### Comparison (Among Items & Targets)")
        c1, c2 = st.columns(2)
        with c1:
            avg_df = df_raw.groupby('Cluster')[f_z].mean().reset_index()
            st.plotly_chart(px.bar(avg_df, x='Cluster', y=f_z, color='Cluster', template="plotly_dark"), use_container_width=True)
        with c2:
            st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=df_raw[f_z].mean(), 
                title={'text': "Avg vs Target"}, gauge={'axis': {'range': [None, df_raw[f_z].max()]}, 'bar': {'color': "#38bdf8"}})).update_layout(template="plotly_dark", height=300), use_container_width=True)

    with tab2:
        st.markdown("### Trend (Over Time)")
        t_col = 'Year' if 'Year' in df_raw.columns else f_y
        fig_trend = px.line(df_raw.sort_values(t_col), x=t_col, y=f_z, color='Cluster', line_shape='spline', template="plotly_dark")
        st.plotly_chart(fig_trend, use_container_width=True)
        st.plotly_chart(px.area(df_raw.sort_values(t_col), x=t_col, y=f_z, color='Cluster', template="plotly_dark"), use_container_width=True)

    with tab3:
        st.markdown("### Composition (Simple & Complex)")
        c3a, c3b = st.columns(2)
        with c3a: st.plotly_chart(px.pie(df_raw, names='Cluster', hole=0.5, template="plotly_dark"), use_container_width=True)
        with c3b: st.plotly_chart(px.treemap(df_raw, path=['Cluster', df_raw.columns[0]], values=f_z, template="plotly_dark"), use_container_width=True)
        st.plotly_chart(px.bar(df_raw, x='Cluster', y=f_z, color='Cluster', barmode='stack', template="plotly_dark"), use_container_width=True)

    with tab4:
        st.markdown("### Distribution (Spread & Density)")
        st.plotly_chart(px.histogram(df_raw, x=f_z, color='Cluster', marginal="box", template="plotly_dark"), use_container_width=True)
        st.plotly_chart(px.violin(df_raw, y=f_z, x='Cluster', color='Cluster', box=True, points="all", template="plotly_dark"), use_container_width=True)

else:
    st.info("⚡ System Ready. Upload CSV to initiate cached neural analysis.")
