"""
dashboard.py
Smart Urban Anomaly Detection System – Bangalore
Streamlit dashboard with dark theme, Plotly charts, and live prediction.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, accuracy_score

from generator     import load_or_generate, generate_additional_anomalies
from data_loader   import get_feature_columns
from preprocessing import preprocess
from pca_module    import apply_pca
from clustering    import run_kmeans, run_dbscan
from labeling      import add_labels_to_df, LABEL_NAMES
from models        import prepare_train_test, train_random_forest, train_xgboost, predict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Urban Anomaly Detection · Bangalore",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Sidebar nav buttons ── */
div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: 1px solid #21262d;
    color: #8b949e !important;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: left;
    font-size: 14px;
    margin-bottom: 4px;
    transition: all 0.2s ease;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #1f2937;
    border-color: #00b4d8;
    color: #00b4d8 !important;
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,180,216,0.15);
}
.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00b4d8, #90e0ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.kpi-icon { font-size: 1.6rem; margin-bottom: 8px; }

/* ── Section cards ── */
.section-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.3);
}

/* ── Page title ── */
.page-title {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #00b4d8, #90e0ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.page-subtitle {
    color: #8b949e;
    font-size: 0.9rem;
    margin-bottom: 28px;
}

/* ── Prediction result badge ── */
.pred-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 50px;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-top: 12px;
}
.pred-normal   { background: rgba(34,197,94,0.15);  color: #22c55e; border: 1px solid #22c55e; }
.pred-moderate { background: rgba(251,146,60,0.15); color: #fb923c; border: 1px solid #fb923c; }
.pred-high     { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef4444; }

/* ── Divider ── */
.divider { border-top: 1px solid #21262d; margin: 20px 0; }

/* ── Metric override ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #00b4d8 !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div { background: #00b4d8 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ──────────────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="Inter, Segoe UI, sans-serif"),
        xaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#21262d", zerolinecolor="#21262d"),
        colorway=["#00b4d8", "#90e0ef", "#ef4444", "#22c55e", "#fb923c", "#a78bfa"],
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

LABEL_COLOR = {
    "Normal":       "#22c55e",
    "Moderate":     "#fb923c",
    "High Anomaly": "#ef4444",
}

# ── Pipeline (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running ML pipeline…")
def run_pipeline():
    """Execute the full ML pipeline once and cache all results."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    raw_path  = os.path.join(base_dir, "data", "raw", "urban_data.csv")

    # 1. Data
    df = load_or_generate(raw_path)
    extra = generate_additional_anomalies(n_samples=300)
    df = pd.concat([df, extra], ignore_index=True).drop_duplicates()

    # 2. Preprocess
    feature_cols = get_feature_columns()
    df, X_scaled, scaler = preprocess(df, feature_cols)

    # 3. PCA
    X_pca, pca_model = apply_pca(X_scaled, n_components=2)

    # 4. Clustering
    kmeans_labels, kmeans_model = run_kmeans(X_scaled, n_clusters=3)
    dbscan_labels, _            = run_dbscan(X_scaled, eps=1.5, min_samples=15)

    # 5. Labels
    df = add_labels_to_df(df, dbscan_labels, kmeans_labels)

    # 6. Models
    y = df["label"].values
    X_train, X_test, y_train, y_test = prepare_train_test(X_scaled, y)
    rf_model,  rf_imp  = train_random_forest(X_train, y_train, feature_cols)
    xgb_model, xgb_imp = train_xgboost(X_train, y_train, feature_cols)

    rf_preds  = predict(rf_model,  X_test)
    xgb_preds = predict(xgb_model, X_test)

    return dict(
        df=df, X_scaled=X_scaled, X_pca=X_pca,
        kmeans_labels=kmeans_labels, dbscan_labels=dbscan_labels,
        feature_cols=feature_cols, scaler=scaler,
        rf_model=rf_model,   rf_imp=rf_imp,   rf_preds=rf_preds,
        xgb_model=xgb_model, xgb_imp=xgb_imp, xgb_preds=xgb_preds,
        y_test=y_test, X_test=X_test,
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
def card(content_fn, *args, **kwargs):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    content_fn(*args, **kwargs)
    st.markdown('</div>', unsafe_allow_html=True)


def kpi(icon, value, label):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>""", unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def apply_template(fig):
    fig.update_layout(PLOTLY_TEMPLATE["layout"])
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 16px 0 24px;'>
            <div style='font-size:2.2rem;'>🏙️</div>
            <div style='font-size:1rem; font-weight:700; color:#00b4d8;
                        letter-spacing:0.04em; margin-top:6px;'>
                Urban Anomaly
            </div>
            <div style='font-size:0.72rem; color:#8b949e; margin-top:2px;'>
                Bangalore · ML Dashboard
            </div>
        </div>
        <hr style='border-color:#21262d; margin-bottom:16px;'>
        """, unsafe_allow_html=True)

        pages = {
            "🏠  Overview":            "Overview",
            "📊  Data Visualization":  "Data Visualization",
            "🔵  PCA Analysis":        "PCA Analysis",
            "🔗  Clustering":          "Clustering",
            "🤖  Model Predictions":   "Model Predictions",
            "🚨  Anomaly Detector":    "Anomaly Detector",
        }

        if "page" not in st.session_state:
            st.session_state.page = "Overview"

        for label, key in pages.items():
            active = st.session_state.page == key
            btn_style = "border-color:#00b4d8 !important; color:#00b4d8 !important;" if active else ""
            if st.button(label, key=f"nav_{key}",
                         help=key,
                         use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("<hr style='border-color:#21262d; margin-top:20px;'>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.72rem; color:#484f58; text-align:center; padding:8px 0;'>
            Powered by RF · XGBoost · DBSCAN · PCA
        </div>""", unsafe_allow_html=True)

    return st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(p):
    df = p["df"]
    page_header("Smart Urban Anomaly Detection",
                "Real-time ML pipeline monitoring Bangalore's urban environment")

    # KPI row
    total      = len(df)
    n_anomaly  = int((df["label"] == 2).sum())
    n_moderate = int((df["label"] == 1).sum())
    rf_acc     = accuracy_score(p["y_test"], p["rf_preds"])
    xgb_acc    = accuracy_score(p["y_test"], p["xgb_preds"])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("📦", f"{total:,}", "Total Samples")
    with c2: kpi("🚨", f"{n_anomaly:,}", "High Anomalies")
    with c3: kpi("⚠️", f"{n_moderate:,}", "Moderate Events")
    with c4: kpi("🌲", f"{rf_acc:.1%}", "RF Accuracy")
    with c5: kpi("⚡", f"{xgb_acc:.1%}", "XGB Accuracy")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Label distribution donut
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        label_counts = df["label_name"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        fig = px.pie(
            label_counts, names="Label", values="Count",
            hole=0.55,
            color="Label",
            color_discrete_map=LABEL_COLOR,
            title="Label Distribution",
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="#0d1117", width=2)))
        fig.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1),
            title_font_size=15,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Anomaly rate by zone
        zone_df = (df.groupby("location_id")
                     .apply(lambda x: (x["label"] == 2).mean() * 100)
                     .reset_index(name="anomaly_rate")
                     .sort_values("anomaly_rate", ascending=True))
        fig2 = px.bar(
            zone_df, x="anomaly_rate", y="location_id",
            orientation="h",
            title="Anomaly Rate by Zone (%)",
            color="anomaly_rate",
            color_continuous_scale=["#22c55e", "#fb923c", "#ef4444"],
        )
        fig2.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            coloraxis_showscale=False,
            title_font_size=15,
            yaxis_title="", xaxis_title="Anomaly Rate (%)",
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent data table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Recent Records**")
    display_cols = ["timestamp", "location_id", "AQI", "temperature",
                    "humidity", "traffic_density", "noise_level", "label_name"]
    st.dataframe(
        df[display_cols].tail(20).reset_index(drop=True),
        use_container_width=True,
        height=280,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATA VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def page_data_viz(p):
    df = p["df"]
    page_header("Data Visualization",
                "Explore sensor distributions, trends, and correlations")

    # Feature selector
    features = ["AQI", "temperature", "humidity", "traffic_density", "noise_level"]
    sel = st.selectbox("Select feature to explore", features, index=0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig = px.histogram(
            df, x=sel, color="label_name",
            color_discrete_map=LABEL_COLOR,
            nbins=60, barmode="overlay",
            opacity=0.75,
            title=f"{sel} Distribution by Label",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title_font_size=14, legend_title="Label")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Box plot per zone
        fig2 = px.box(
            df, x="location_id", y=sel,
            color="location_id",
            title=f"{sel} by Bangalore Zone",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                           title_font_size=14, showlegend=False,
                           xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Hourly trend
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    hourly = df.groupby("hour")[features].mean().reset_index()
    fig3 = go.Figure()
    colors_line = ["#00b4d8", "#90e0ef", "#fb923c", "#22c55e", "#a78bfa"]
    for feat, col in zip(features, colors_line):
        fig3.add_trace(go.Scatter(
            x=hourly["hour"], y=hourly[feat],
            mode="lines+markers", name=feat,
            line=dict(color=col, width=2),
            marker=dict(size=5),
        ))
    fig3.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        title="Average Sensor Readings by Hour of Day",
        title_font_size=14,
        xaxis_title="Hour", yaxis_title="Value",
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    corr = df[features].corr().round(2)
    fig4 = px.imshow(
        corr, text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Matrix",
        aspect="auto",
    )
    fig4.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                       title_font_size=14)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – PCA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_pca(p):
    df    = p["df"]
    X_pca = p["X_pca"]
    page_header("PCA Analysis",
                "2-component dimensionality reduction of 7 urban features")

    pca_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Label": df["label_name"].values,
        "Zone":  df["location_id"].values,
        "AQI":   df["AQI"].values,
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="Label",
            color_discrete_map=LABEL_COLOR,
            hover_data=["Zone", "AQI"],
            opacity=0.65,
            title="PCA – All Labels",
            size_max=6,
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title_font_size=14, legend_title="Label")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        # Anomaly highlight only
        pca_df["is_anomaly"] = pca_df["Label"] == "High Anomaly"
        fig2 = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="is_anomaly",
            color_discrete_map={False: "#334155", True: "#ef4444"},
            opacity=0.7,
            title="Anomaly Highlight",
            symbol="is_anomaly",
            symbol_map={False: "circle", True: "x"},
        )
        fig2.update_traces(marker=dict(size=5))
        fig2.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            title_font_size=14,
            showlegend=True,
            legend=dict(
                title="Anomaly",
                itemsizing="constant",
            ),
        )
        # Rename legend items
        for trace in fig2.data:
            trace.name = "Anomaly" if trace.name == "True" else "Normal/Moderate"
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Variance explained bar
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    var_df = pd.DataFrame({
        "Component": ["PC1", "PC2"],
        "Variance Explained (%)": [31.0, 14.4],
    })
    fig3 = px.bar(
        var_df, x="Component", y="Variance Explained (%)",
        color="Component",
        color_discrete_sequence=["#00b4d8", "#90e0ef"],
        title="PCA Explained Variance",
        text_auto=".1f",
    )
    fig3.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                       title_font_size=14, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def page_clustering(p):
    df            = p["df"]
    X_pca         = p["X_pca"]
    kmeans_labels = p["kmeans_labels"]
    dbscan_labels = p["dbscan_labels"]

    page_header("Clustering Analysis",
                "K-Means cluster structure and DBSCAN anomaly detection")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        km_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "Cluster": [f"Cluster {c}" for c in kmeans_labels],
        })
        fig = px.scatter(
            km_df, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=["#00b4d8", "#fb923c", "#a78bfa"],
            opacity=0.6, title="K-Means Clusters (k=3)",
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        db_color = ["Anomaly" if l == -1 else f"Cluster {l}" for l in dbscan_labels]
        db_df = pd.DataFrame({
            "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
            "Type": db_color,
        })
        color_map = {k: "#ef4444" if k == "Anomaly" else c
                     for k, c in zip(db_df["Type"].unique(),
                                     ["#00b4d8", "#22c55e", "#a78bfa", "#ef4444"])}
        color_map["Anomaly"] = "#ef4444"
        fig2 = px.scatter(
            db_df, x="PC1", y="PC2", color="Type",
            color_discrete_map=color_map,
            opacity=0.65, title="DBSCAN – Anomaly Detection",
            symbol="Type",
            symbol_map={"Anomaly": "x"},
        )
        fig2.update_traces(marker=dict(size=4))
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                           title_font_size=14)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Cluster stats table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**K-Means Cluster Statistics**")
    feat_cols = ["AQI", "temperature", "humidity", "traffic_density", "noise_level"]
    cluster_stats = (df.assign(cluster=kmeans_labels)
                       .groupby("cluster")[feat_cols]
                       .mean().round(2)
                       .reset_index())
    cluster_stats["cluster"] = cluster_stats["cluster"].apply(lambda x: f"Cluster {x}")
    st.dataframe(cluster_stats, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # DBSCAN summary metrics
    n_anomalies = int(np.sum(dbscan_labels == -1))
    n_total     = len(dbscan_labels)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("DBSCAN Anomalies",  f"{n_anomalies:,}")
    col_b.metric("Anomaly Rate",      f"{n_anomalies/n_total:.1%}")
    col_c.metric("Normal Points",     f"{n_total - n_anomalies:,}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – MODEL PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
def page_models(p):
    page_header("Model Predictions",
                "Random Forest vs XGBoost – performance and feature importance")

    y_test    = p["y_test"]
    rf_preds  = p["rf_preds"]
    xgb_preds = p["xgb_preds"]
    rf_imp    = p["rf_imp"]
    xgb_imp   = p["xgb_imp"]

    rf_acc  = accuracy_score(y_test, rf_preds)
    xgb_acc = accuracy_score(y_test, xgb_preds)

    # Accuracy comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.metric("Random Forest Accuracy", f"{rf_acc:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.metric("XGBoost Accuracy", f"{xgb_acc:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Confusion matrices side by side
    col3, col4 = st.columns(2)
    label_names = ["Normal", "Moderate", "High Anomaly"]

    def cm_fig(y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        fig = px.imshow(
            cm, text_auto=True,
            x=label_names, y=label_names,
            color_continuous_scale="Blues",
            title=title,
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title_font_size=14,
                          xaxis_title="Predicted", yaxis_title="Actual")
        return fig

    with col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(cm_fig(y_test, rf_preds,  "RF – Confusion Matrix"),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(cm_fig(y_test, xgb_preds, "XGB – Confusion Matrix"),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance comparison
    col5, col6 = st.columns(2)

    def imp_fig(imp_df, title):
        fig = px.bar(
            imp_df.sort_values("importance"),
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale=["#1e3a5f", "#00b4d8"],
            title=title, text_auto=".3f",
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                          title_font_size=14, coloraxis_showscale=False,
                          yaxis_title="", xaxis_title="Importance")
        return fig

    with col5:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(imp_fig(rf_imp,  "Random Forest – Feature Importance"),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col6:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(imp_fig(xgb_imp, "XGBoost – Feature Importance"),
                        use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – ANOMALY DETECTOR (interactive)
# ══════════════════════════════════════════════════════════════════════════════
def page_detector(p):
    scaler       = p["scaler"]
    rf_model     = p["rf_model"]
    xgb_model    = p["xgb_model"]
    feature_cols = p["feature_cols"]

    page_header("Live Anomaly Detector",
                "Input sensor readings to get an instant anomaly prediction")

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Sensor Input Panel**")

        zones = [
            "Koramangala", "Whitefield", "Indiranagar", "Hebbal",
            "Electronic_City", "Jayanagar", "Marathahalli",
            "Yelahanka", "BTM_Layout", "HSR_Layout",
        ]
        zone = st.selectbox("📍 Location Zone", zones)

        aqi     = st.slider("💨 AQI",              50,  500, 120, step=5)
        temp    = st.slider("🌡️ Temperature (°C)", 15,  40,  26, step=1)
        hum     = st.slider("💧 Humidity (%)",     30,  90,  60, step=1)
        traffic = st.slider("🚗 Traffic Density",  0,   100, 40, step=1)
        noise   = st.slider("🔊 Noise Level (dB)", 30,  120, 65, step=1)
        hour    = st.slider("🕐 Hour of Day",       0,   23,  9, step=1)
        dow     = st.selectbox("📅 Day of Week",
                               ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
        dow_num = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"].index(dow)

        predict_btn = st.button("🔍 Predict Anomaly", use_container_width=True,
                                type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("**Prediction Result**")

        if predict_btn:
            # Build input vector matching feature_cols order
            input_vals = {
                "AQI": aqi, "temperature": temp, "humidity": hum,
                "traffic_density": traffic, "noise_level": noise,
                "hour": hour, "day_of_week": dow_num,
            }
            X_input = np.array([[input_vals[c] for c in feature_cols]])
            X_input_scaled = scaler.transform(X_input)

            rf_pred  = rf_model.predict(X_input_scaled)[0]
            xgb_pred = xgb_model.predict(X_input_scaled)[0]

            rf_proba  = rf_model.predict_proba(X_input_scaled)[0]
            xgb_proba = xgb_model.predict_proba(X_input_scaled)[0]

            label_map   = {0: "Normal", 1: "Moderate", 2: "High Anomaly"}
            badge_class = {0: "pred-normal", 1: "pred-moderate", 2: "pred-high"}

            # RF result
            st.markdown(f"""
            <div style='margin-bottom:18px;'>
                <div style='font-size:0.8rem; color:#8b949e; margin-bottom:4px;'>
                    🌲 Random Forest
                </div>
                <span class='pred-badge {badge_class[rf_pred]}'>
                    {label_map[rf_pred]}
                </span>
            </div>""", unsafe_allow_html=True)

            # XGB result
            st.markdown(f"""
            <div style='margin-bottom:20px;'>
                <div style='font-size:0.8rem; color:#8b949e; margin-bottom:4px;'>
                    ⚡ XGBoost
                </div>
                <span class='pred-badge {badge_class[xgb_pred]}'>
                    {label_map[xgb_pred]}
                </span>
            </div>""", unsafe_allow_html=True)

            # Probability bars (RF)
            st.markdown("**Confidence (RF)**")
            prob_df = pd.DataFrame({
                "Label": ["Normal", "Moderate", "High Anomaly"],
                "Probability": rf_proba,
            })
            fig = px.bar(
                prob_df, x="Label", y="Probability",
                color="Label",
                color_discrete_map=LABEL_COLOR,
                text_auto=".2%",
                range_y=[0, 1],
            )
            fig.update_layout(
                **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                showlegend=False, height=220,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Gauge – AQI risk
            st.markdown("**AQI Risk Gauge**")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=aqi,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 500], "tickcolor": "#8b949e"},
                    "bar":  {"color": "#00b4d8"},
                    "steps": [
                        {"range": [0,   150], "color": "#1a3a2a"},
                        {"range": [150, 300], "color": "#3a2a0a"},
                        {"range": [300, 500], "color": "#3a0a0a"},
                    ],
                    "threshold": {
                        "line": {"color": "#ef4444", "width": 3},
                        "thickness": 0.75, "value": 300,
                    },
                },
                number={"font": {"color": "#00b4d8"}},
            ))
            gauge.update_layout(
                paper_bgcolor="#161b22",
                font=dict(color="#c9d1d9"),
                height=200,
                margin=dict(l=20, r=20, t=20, b=10),
            )
            st.plotly_chart(gauge, use_container_width=True)

        else:
            st.markdown("""
            <div style='text-align:center; padding:60px 20px; color:#484f58;'>
                <div style='font-size:3rem;'>🎯</div>
                <div style='margin-top:12px; font-size:0.9rem;'>
                    Adjust the sliders and click<br>
                    <strong style='color:#00b4d8;'>Predict Anomaly</strong>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Input radar chart
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("**Input Feature Radar**")
    # Normalize inputs 0-1 for radar
    ranges = {"AQI": (50,500), "temperature": (15,40), "humidity": (30,90),
              "traffic_density": (0,100), "noise_level": (30,120)}
    radar_vals = [
        (aqi - 50) / 450,
        (temp - 15) / 25,
        (hum - 30) / 60,
        traffic / 100,
        (noise - 30) / 90,
    ]
    radar_labels = ["AQI", "Temperature", "Humidity", "Traffic", "Noise"]
    radar_vals_closed = radar_vals + [radar_vals[0]]
    radar_labels_closed = radar_labels + [radar_labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=radar_vals_closed,
        theta=radar_labels_closed,
        fill="toself",
        fillcolor="rgba(0,180,216,0.15)",
        line=dict(color="#00b4d8", width=2),
        name="Input",
    ))
    fig_radar.update_layout(
        paper_bgcolor="#161b22",
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="#21262d", tickcolor="#8b949e"),
            angularaxis=dict(gridcolor="#21262d", tickcolor="#c9d1d9"),
        ),
        font=dict(color="#c9d1d9"),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    pipeline = run_pipeline()
    page     = sidebar()

    if   page == "Overview":           page_overview(pipeline)
    elif page == "Data Visualization": page_data_viz(pipeline)
    elif page == "PCA Analysis":       page_pca(pipeline)
    elif page == "Clustering":         page_clustering(pipeline)
    elif page == "Model Predictions":  page_models(pipeline)
    elif page == "Anomaly Detector":   page_detector(pipeline)


if __name__ == "__main__":
    main()
