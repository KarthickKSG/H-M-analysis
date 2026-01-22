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
st.set_page_config(page_title="NeuroCluster AI | Analytics", layout="wide", page_icon="💎")

# --- CUSTOM CSS: PREMIUM BLUE GLASSY THEME ---
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #001f3f, #000814);
        color: #f8fafc;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
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
        font-weight: 800; font-size: 2.8rem; text-align: center;
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

# --- SIDEBAR ENGINE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.markdown("### 🎛️ Analytics Core")
    uploaded_file = st.file_uploader("Data Source (CSV)", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        
        f_x = st.selectbox("X-Axis (Frequency/Time)", nums, index=0)
        f_y = st.selectbox("Y-Axis (Intensity/Year)", nums, index=min(1, len(nums)-1))
        f_z = st.selectbox("Z-Axis (Value/Load)", nums, index=min(2, len(nums)-1))
        
        st.divider()
        algo_name = st.selectbox("Clustering Logic", ["K-Means", "K-Medoids"])
        k = st.slider("Target Segments (k)", 2, 8, 4)
        
        # Color Theme
        c_seq = px.colors.qualitative.G10 # Professional palette

# --- MAIN DASHBOARD ---
st.markdown('<h1 class="title-text">NeuroCluster Analytics</h1>', unsafe_allow_html=True)

if uploaded_file:
    # 1. Cluster Execution
    X = df[[f_x, f_y, f_z]].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=k, random_state=42) if algo_name == "K-Means" else KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    df['Cluster'] = [f"Pattern {l}" for l in labels]

    # 2. Hero Visual (3D Cluster)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🌐 3D Spatial Neural Mapping")
    fig_3d = px.scatter_3d(df, x=f_x, y=f_y, z=f_z, color='Cluster', 
                           symbol_sequence=['circle'], color_discrete_sequence=c_seq,
                           template="plotly_dark", height=600)
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Multiple Veracities of Analysis Tabs
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["⚖️ Comparison", "📈 Trend", "🧩 Composition", "📊 Distribution"])

    # --- TAB 1: COMPARISON (Bar, Bullet) ---
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Item Comparison (Column)")
            avg_val = df.groupby('Cluster')[f_z].mean().reset_index()
            fig_bar = px.bar(avg_val, x='Cluster', y=f_z, color='Cluster', color_discrete_sequence=c_seq, template="plotly_dark")
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            st.markdown("#### Performance vs Target (Bullet)")
            curr_val = df[f_z].mean()
            target_val = df[f_z].max() * 0.8 # Simulated target
            fig_bullet = go.Figure(go.Indicator(
                mode = "number+gauge+delta", value = curr_val,
                delta = {'reference': target_val},
                gauge = {'shape': "bullet", 'bar': {'color': "#38bdf8"}},
                title = {'text': "Avg Load"}
            ))
            fig_bullet.update_layout(template="plotly_dark", height=250)
            st.plotly_chart(fig_bullet, use_container_width=True)

    # --- TAB 2: TREND (Line, Area) ---
    with tab2:
        st.markdown("#### Temporal Trend Evolution")
        # Ensure a timeline if columns exist
        timeline_col = 'Year' if 'Year' in df.columns else f_x
        df_sort = df.sort_values(timeline_col)
        
        fig_area = px.area(df_sort, x=timeline_col, y=f_z, color='Cluster', 
                           line_shape='spline', color_discrete_sequence=c_seq, template="plotly_dark")
        st.plotly_chart(fig_area, use_container_width=True)

    # --- TAB 3: COMPOSITION (Donut, Treemap, Stacked Bar) ---
    with tab3:
        st.markdown("#### Hierarchical Composition")
        c3a, c3b = st.columns([1, 2])
        with c3a:
            fig_donut = px.pie(df, names='Cluster', hole=0.5, color_discrete_sequence=c_seq, template="plotly_dark")
            fig_donut.update_layout(showlegend=False)
            st.plotly_chart(fig_donut, use_container_width=True)
        with c3b:
            path_cols = ['Cluster']
            if 'Country' in df.columns: path_cols.append('Country')
            fig_tree = px.treemap(df, path=path_cols, values=f_z, color='Cluster', 
                                  color_discrete_sequence=c_seq, template="plotly_dark")
            st.plotly_chart(fig_tree, use_container_width=True)
        
        st.markdown("#### Segment Stacked Breakdown")
        fig_stacked = px.bar(df, x='Cluster', y=f_z, color='Cluster', barmode='stack', template="plotly_dark")
        st.plotly_chart(fig_stacked, use_container_width=True)

    # --- TAB 4: DISTRIBUTION (Histogram, Box Plot) ---
    with tab4:
        c4a, c4b = st.columns(2)
        with c4a:
            st.markdown("#### Population Density (Histogram)")
            fig_hist = px.histogram(df, x=f_z, color='Cluster', marginal="rug", template="plotly_dark")
            st.plotly_chart(fig_hist, use_container_width=True)
        with c4b:
            st.markdown("#### Statistical Spread (Box Plot)")
            fig_box = px.box(df, y=f_z, x='Cluster', color='Cluster', template="plotly_dark")
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("💎 Welcome to NeuroCluster Analytics. Please upload your CSV to begin.")
