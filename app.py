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
st.set_page_config(page_title="NeuroCluster Pro | Optimized", layout="wide", page_icon="💎")

# --- PREMIUM BLUE GLASSY CSS ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #001f3f, #000814);
        color: #f8fafc;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .title-text {
        background: linear-gradient(90deg, #38bdf8, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 3rem; text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px; padding: 10px 20px; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important; color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ⚡ SPEED OPTIMIZATION: CACHED FUNCTIONS ---
@st.cache_data(show_spinner=False)
def load_data(file):
    return pd.read_csv(file)

@st.cache_data(show_spinner=False)
def apply_clustering(df, features, algo_type, k_val):
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if algo_type == "K-Means":
        model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    else:
        model = KMedoids(n_clusters=k_val, random_state=42)
        
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if k_val > 1 else 0
    return labels, score

# --- SIDEBAR CONTROL ---
with st.sidebar:
    st.markdown("### 🎛️ Analytics Core")
    uploaded_file = st.file_uploader("Source CSV Data", type="csv")
    
    if uploaded_file:
        df_raw = load_data(uploaded_file)
        nums = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        
        st.divider()
        f_x = st.selectbox("X-Axis (Spatial)", nums, index=0)
        f_y = st.selectbox("Y-Axis (Temporal)", nums, index=min(1, len(nums)-1))
        f_z = st.selectbox("Z-Axis (Intensity)", nums, index=min(2, len(nums)-1))
        
        st.divider()
        algo = st.radio("Algorithm", ["K-Means", "K-Medoids"], horizontal=True)
        k = st.select_slider("Clusters", options=range(2, 9), value=4)
        palette = px.colors.qualitative.Prism

# --- MAIN DASHBOARD ---
st.markdown('<h1 class="title-text">NeuroCluster Analysis</h1>', unsafe_allow_html=True)

if uploaded_file:
    # 1. Cached Execution
    features = [f_x, f_y, f_z]
    labels, sil_score = apply_clustering(df_raw, features, algo, k)
    df_raw['Cluster'] = [f"Group {l}" for l in labels]

    # 2. Main 3D Pattern Map
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌐 3D Spatial Pattern Mapping")
    fig_3d = px.scatter_3d(df_raw, x=f_x, y=f_y, z=f_z, color='Cluster', 
                           symbol_sequence=['circle'], color_discrete_sequence=palette,
                           template="plotly_dark", height=600)
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Multi-Veracity Analytics Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Comparison", "📈 Trend", "🧩 Composition", "📊 Distribution"])

    # --- TAB 1: COMPARISON ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Item Average (Column)")
            avg_df = df_raw.groupby('Cluster')[f_z].mean().reset_index()
            st.plotly_chart(px.bar(avg_df, x='Cluster', y=f_z, color='Cluster', 
                                   color_discrete_sequence=palette, template="plotly_dark"), use_container_width=True)
        with c2:
            st.markdown("#### Performance vs Target (Bullet)")
            fig_bullet = go.Figure(go.Indicator(
                mode = "number+gauge+delta", value = df_raw[f_z].mean(),
                delta = {'reference': df_raw[f_z].median()},
                gauge = {'shape': "bullet", 'bar': {'color': "#38bdf8"}},
                title = {'text': "Avg Intensity"}
            ))
            fig_bullet.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig_bullet, use_container_width=True)

    # --- TAB 2: TREND ---
    with tab2:
        st.markdown("#### Temporal Trend Evolution")
        t_col = 'Year' if 'Year' in df_raw.columns else f_y
        df_sorted = df_raw.sort_values(t_col)
        fig_line = px.line(df_sorted, x=t_col, y=f_z, color='Cluster', line_shape='spline', 
                           color_discrete_sequence=palette, template="plotly_dark")
        st.plotly_chart(fig_line, use_container_width=True)
        st.plotly_chart(px.area(df_sorted, x=t_col, y=f_z, color='Cluster', 
                                color_discrete_sequence=palette, template="plotly_dark"), use_container_width=True)

    # --- TAB 3: COMPOSITION ---
    with tab3:
        c3a, c3b = st.columns(2)
        with c3a: 
            st.markdown("#### Simple Composition (Donut)")
            st.plotly_chart(px.pie(df_raw, names='Cluster', hole=0.5, 
                                   color_discrete_sequence=palette, template="plotly_dark"), use_container_width=True)
        with c3b: 
            st.markdown("#### Complex Hierarchy (Treemap)")
            st.plotly_chart(px.treemap(df_raw, path=['Cluster', df_raw.columns[0]], values=f_z, 
                                       color='Cluster', color_discrete_sequence=palette, template="plotly_dark"), use_container_width=True)
        st.markdown("#### Segment Stacked Breakdown")
        st.plotly_chart(px.bar(df_raw, x='Cluster', y=f_z, color='Cluster', barmode='stack', 
                               template="plotly_dark"), use_container_width=True)

    # --- TAB 4: DISTRIBUTION ---
    with tab4:
        st.markdown("#### Statistical Spread (Violin & Box)")
        st.plotly_chart(px.violin(df_raw, y=f_z, x='Cluster', color='Cluster', box=True, points="all", 
                                  color_discrete_sequence=palette, template="plotly_dark"), use_container_width=True)
        st.markdown("#### Population Density (Histogram)")
        st.plotly_chart(px.histogram(df_raw, x=f_z, color='Cluster', marginal="box", 
                                     template="plotly_dark"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("💎 Welcome to NeuroCluster Pro. Upload your dataset to initiate high-speed cached analysis.")
