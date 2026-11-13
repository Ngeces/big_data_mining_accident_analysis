"""
app.py
Dashboard Streamlit: Overview K3, peta tambang, penyimpanan Parquet, simulasi keparahan cedera,
benchmark Scikit-Learn vs Spark MLlib, dan graph analytics.
Dashboard hanya membaca artefak di folder artifacts/ (tidak ada training saat aplikasi berjalan).
"""

import os
import sys
import json
import html
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from config import (  # noqa: E402
    CAT_COLS, NUM_COLS, GRAPH_START_YR, GRAPH_END_YR, TRAIN_YEARS, VAL_YEAR, TEST_YEAR, RANDOM_SEED,
)

ARTIFACTS = os.path.join(ROOT, "artifacts")

# ==============================================================================
# 1. Konfigurasi halaman, warna, dan gaya
# ==============================================================================
st.set_page_config(
    page_title="MSHA Mine Safety Analytics",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#0b57d0",
    "severe": "#d93025",
    "minor": "#1e8e3e",
    "other": "#9aa0a6",
    "coal": "#3c4043",
    "metal": "#e37400",
    "mine": "#1a73e8",
    "standard": "#d93025",
    "accident_type": "#188038",
    "muted": "#5f6b7a",
}
MINE_TYPE_RGB = {"Surface": [227, 116, 0], "Underground": [26, 115, 232], "Facility": [24, 128, 56]}
COMMUNITY_PALETTE = px.colors.qualitative.Bold
FRAMEWORK_COLORS = {"Scikit-Learn": "#1a73e8", "Apache Spark MLlib": "#e37400", "Baseline": "#9aa0a6"}
NODE_TYPE_LABEL = {"mine": "Tambang", "standard": "Standar 30 CFR", "accident_type": "Tipe Kecelakaan"}
EDGE_TYPE_LABEL = {"CITED_FOR": "Tambang → Standar dilanggar", "REPORTED_ACCIDENT": "Tambang → Tipe kecelakaan"}

DISCLAIMER = (
    # "Dashboard ini merupakan prototipe akademik untuk analisis data historis MSHA "
    "(COMP8035041 Big Data Analytics). Estimasi skenario didasarkan pada data laporan retrospektif "
    "dan tidak boleh digunakan sebagai satu-satunya dasar pengambilan keputusan keselamatan operasional tambang."
)

st.markdown(
    """
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .page-title {font-size: 1.75rem; font-weight: 700; margin-bottom: 0.1rem;}
    .page-sub {color: #5f6b7a; margin-bottom: 1rem; font-size: 0.95rem;}
    .kpi {background: #ffffff; border: 1px solid #e3e7ee; border-radius: 12px;
          padding: 14px 16px; height: 100%; border-top: 4px solid var(--accent, #0b57d0);}
    .kpi .label {font-size: 0.75rem; color: #5f6b7a; text-transform: uppercase; letter-spacing: .05em;}
    .kpi .value {font-size: 1.55rem; font-weight: 700; color: #1f2328; line-height: 1.3;}
    .kpi .sub {font-size: 0.8rem; color: #5f6b7a;}
    .chip {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem;
           margin: 0 6px 6px 0; color: #ffffff;}
    .disclaimer {background: #fff8e1; border-left: 4px solid #f9a825; padding: 8px 12px;
                 border-radius: 6px; font-size: 0.8rem; color: #5d4037; margin-bottom: 1rem;}
    .verdict {border-radius: 10px; padding: 14px 16px; font-size: 0.95rem;}
    .verdict.pos {background: #fdecea; border-left: 5px solid #d93025;}
    .verdict.neg {background: #e6f4ea; border-left: 5px solid #1e8e3e;}
    div[data-testid="stSidebarNav"] {display: none;}

    /* =========================================================================
       MODERN UI: HILANGKAN RADIO CIRCLE (BULAT-BULAT) & GANTI DENGAN MODERN PILLS/MENU
       ========================================================================= */
    /* 1. Sembunyikan lingkaran bulat radio button */
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
        padding: 0 !important;
    }

    /* 2. Navigasi Sidebar Modern (Vertical Menu Cards) */
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
        padding-top: 4px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label {
        padding: 10px 14px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        border: 1px solid transparent !important;
        width: 100% !important;
        margin: 0 !important;
        color: #475569 !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #f1f5f9 !important;
        color: #0b57d0 !important;
        transform: translateX(2px);
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked),
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: #e8f0fe !important;
        color: #0b57d0 !important;
        font-weight: 600 !important;
        border: 1px solid #c2e7ff !important;
        box-shadow: 0 1px 3px rgba(11, 87, 208, 0.08) !important;
    }
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p,
    [data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p {
        color: #0b57d0 !important;
        font-weight: 600 !important;
    }

    /* 3. Segmented Control / Pill Buttons untuk Radio Horizontal (Warna, Ukuran, Tampilan) */
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] {
        background: #f1f5f9 !important;
        padding: 4px !important;
        border-radius: 10px !important;
        gap: 4px !important;
        border: 1px solid #e2e8f0 !important;
        display: inline-flex !important;
        align-items: center !important;
        width: auto !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label {
        padding: 6px 14px !important;
        border-radius: 7px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        font-size: 0.88rem !important;
        color: #64748b !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:hover {
        color: #0f172a !important;
        background: rgba(255, 255, 255, 0.6) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:has(input:checked),
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label[data-checked="true"] {
        background: #ffffff !important;
        color: #0b57d0 !important;
        font-weight: 600 !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06) !important;
    }
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label:has(input:checked) p,
    div[data-testid="stRadio"] div[role="radiogroup"][aria-orientation="horizontal"] > label[data-checked="true"] p {
        color: #0b57d0 !important;
        font-weight: 600 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Pemuatan artefak
# ==============================================================================
def artifact(name):
    return os.path.join(ARTIFACTS, name)


@st.cache_data(show_spinner=False)
def load_json(name):
    path = artifact(name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_csv(name, **kwargs):
    path = artifact(name)
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    model_path, preproc_path = artifact("best_model.joblib"), artifact("preprocessor.joblib")
    if os.path.exists(model_path) and os.path.exists(preproc_path):
        return joblib.load(model_path), joblib.load(preproc_path)
    return None, None


def model_categories(preproc):
    """Kategori yang dipelajari OneHotEncoder saat training."""
    try:
        ohe = preproc.named_transformers_["cat"]
        cols = preproc.transformers_[0][2]
        return {col: [str(c) for c in cats] for col, cats in zip(cols, ohe.categories_)}
    except (AttributeError, KeyError, IndexError):
        return {}


best_model, preprocessor = load_model()
benchmarks = load_json("benchmark_results.json")
storage = load_json("hadoop_storage_summary.json")
prep_meta = load_json("preprocessing_metadata.json")
model_meta = load_json("model_metadata.json")
closeness = load_json("mine_accident_closeness.json")
graph_summary = load_json("graph_top_edges.json")
df_nodes = load_csv("graph_nodes_summary.csv", dtype={"node_id": str, "label": str})
df_geo = load_csv("mine_geo_accidents.csv", dtype={"MINE_ID": str})
df_trend = load_csv("accident_annual_trend.csv")
df_explorer_edges = load_csv("graph_explorer_edges.csv")

MINE_NAMES = dict(zip(df_geo["MINE_ID"], df_geo["CURRENT_MINE_NAME"])) if not df_geo.empty else {}


# ==============================================================================
# 3. Komponen UI bersama
# ==============================================================================
def page_header(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">⚠️ <b>Academic Disclaimer:</b> {DISCLAIMER}</div>', unsafe_allow_html=True)


def kpi(col, label, value, sub="", accent=COLORS["primary"]):
    col.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def chips(items):
    """items: list of (label, hex_color)"""
    st.markdown(
        "".join(f'<span class="chip" style="background:{c}">{html.escape(str(t))}</span>' for t, c in items),
        unsafe_allow_html=True,
    )


def style_fig(fig, height=380, legend=True):
    has_title = bool(fig.layout.title and fig.layout.title.text)
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=40 if has_title else 15, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        hoverlabel=dict(font_size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    if has_title:
        fig.update_layout(title=dict(font=dict(size=14)))
    else:
        fig.update_layout(title_text=None)
    return fig


def fmt_int(x):
    return f"{int(x):,}" if x is not None and not pd.isna(x) else "n/a"


def mine_label(mine_id):
    name = MINE_NAMES.get(str(mine_id))
    return f"{mine_id} · {name}" if name else str(mine_id)


def hex_to_rgb(hex_color, alpha=210):
    h = hex_color.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alpha]


def community_hex(cid):
    return "#9aa0a6" if cid < 0 else COMMUNITY_PALETTE[int(cid) % len(COMMUNITY_PALETTE)]


def missing_artifact(name, script):
    st.info(f"Artefak `artifacts/{name}` belum tersedia. Jalankan `python src/{script}` terlebih dahulu.")


# ==============================================================================
# 4. Sidebar
# ==============================================================================
PAGES = {
    "overview": "📊 Overview & Tren",
    "map": "🗺️ Peta Tambang",
    "storage": "🗄️ Penyimpanan HDFS",
    "predictor": "🎯 Simulasi Keparahan",
    "benchmark": "⚡ Benchmark Model",
    "graph": "🕸️ Graph Analytics",
}

with st.sidebar:
    st.markdown("## ⛏️ MSHA Safety Analytics")
    st.caption("COMP8035041 Big Data Analytics · Magister Teknik Informatika BINUS University")
    page = st.radio("Navigasi", list(PAGES), format_func=PAGES.get, label_visibility="collapsed")

    st.divider()
    with st.expander("📦 Status artefak", expanded=False):
        expected = [
            "preprocessing_metadata.json", "model_metadata.json", "best_model.joblib",
            "benchmark_results.json", "hadoop_storage_summary.json", "accident_annual_trend.csv",
            "mine_geo_accidents.csv", "graph_nodes_summary.csv", "graph_explorer_edges.csv",
            "mine_accident_closeness.json",
        ]
        for name in expected:
            st.markdown(f"{'✅' if os.path.exists(artifact(name)) else '⚪'} `{name}`")
        st.caption("Artefak dibuat oleh `python src/run_all.py`.")

    st.caption(
        f"Partisi: train {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]} · val {VAL_YEAR} · test {TEST_YEAR}  \n"
        f"Jendela graf: {GRAPH_START_YR}–{GRAPH_END_YR}"
    )


# ==============================================================================
# 5. Halaman: Overview
# ==============================================================================
def page_overview():
    page_header(
        "Ringkasan Keselamatan Kerja Tambang MSHA",
        "Statistik deskriptif laporan kecelakaan tambang (MSHA Form 7000-1) dan dataset yang dipakai untuk pemodelan.",
    )

    split = prep_meta.get("split", {})
    sizes = {k: split.get(f"{k}_rows") for k in ("train", "val", "test")}
    ratios = split.get("class1_ratio", {})
    n_total = sum(v for v in sizes.values() if v) or None
    class1_pct = (
        100 * sum(ratios.get(k, 0) * (n or 0) for k, n in sizes.items()) / n_total if n_total and ratios else None
    )

    c = st.columns(4)
    kpi(c[0], f"Insiden berlabel {TRAIN_YEARS[0]}–{TEST_YEAR}", fmt_int(n_total), "Setelah eksklusi kode 07/08/09")
    kpi(c[1], "Rasio Lost-Time / Severe", f"{class1_pct:.1f}%" if class1_pct is not None else "n/a",
        "Class 1 (kode 01–05)", COLORS["severe"])
    if not df_trend.empty:
        last = df_trend.sort_values("CAL_YR").iloc[-1]
        kpi(c[2], f"Total laporan {int(last['CAL_YR'])}", fmt_int(last["total_accidents"]),
            f"{last['severe_pct']:.1f}% Lost-Time / Severe", COLORS["metal"])
    else:
        kpi(c[2], "Total laporan", "n/a")
    kpi(c[3], "Simpul graf heterogen", fmt_int(len(df_nodes)) if not df_nodes.empty else "n/a",
        "Tambang · standar · tipe kecelakaan", COLORS["minor"])
    if not prep_meta:
        missing_artifact("preprocessing_metadata.json", "data_pipeline.py")

    st.write("")
    left, right = st.columns([2.1, 1])

    with left, st.container(border=True):
        st.markdown("#### 📈 Tren tahunan laporan kecelakaan")
        if df_trend.empty:
            missing_artifact("accident_annual_trend.csv", "geo_and_viz_pipeline.py")
        else:
            ctrl1, ctrl2 = st.columns([1.4, 1])
            yr_min, yr_max = int(df_trend["CAL_YR"].min()), int(df_trend["CAL_YR"].max())
            years = ctrl1.slider("Rentang tahun", yr_min, yr_max, (yr_min, yr_max))
            view = ctrl2.radio("Tampilan", ["Keparahan", "Komoditas", "Proporsi"], horizontal=True)
            trend = df_trend[df_trend["CAL_YR"].between(*years)]

            fig = go.Figure()
            if view == "Keparahan":
                series = [
                    ("severe_accidents", "Lost-Time / Severe (01–05)", COLORS["severe"]),
                    ("minor_accidents", "No Lost-Time (00, 06, 10)", COLORS["minor"]),
                ]
                if "other_accidents" in trend:
                    series.append(("other_accidents", "Dieksklusi (07–09/kosong)", COLORS["other"]))
                for col, name, color in series:
                    fig.add_trace(go.Bar(x=trend["CAL_YR"], y=trend[col], name=name, marker_color=color))
                fig.add_trace(go.Scatter(
                    x=trend["CAL_YR"], y=trend["total_accidents"], name="Total", mode="lines+markers",
                    line=dict(color=COLORS["primary"], width=3),
                ))
                fig.update_layout(barmode="stack", yaxis_title="Jumlah laporan")
            elif view == "Komoditas":
                fig.add_trace(go.Scatter(x=trend["CAL_YR"], y=trend["coal_accidents"], name="Coal",
                                         mode="lines+markers", line=dict(color=COLORS["coal"], width=3)))
                fig.add_trace(go.Scatter(x=trend["CAL_YR"], y=trend["metal_accidents"], name="Metal / Non-Metal",
                                         mode="lines+markers", line=dict(color=COLORS["metal"], width=3)))
                fig.update_layout(yaxis_title="Jumlah laporan")
            else:
                fig.add_trace(go.Bar(x=trend["CAL_YR"], y=trend["severe_pct"], name="% Lost-Time / Severe",
                                     marker_color=COLORS["severe"], text=trend["severe_pct"].round(1),
                                     textposition="outside"))
                fig.update_layout(yaxis_title="% dari total laporan", yaxis_range=[0, 100])
            fig.update_layout(hovermode="x unified", xaxis=dict(dtick=1))
            st.plotly_chart(style_fig(fig, 400), width='stretch')

            total_period = int(trend["total_accidents"].sum())
            severe_period = int(trend["severe_accidents"].sum())
            avg_per_day = total_period / max(len(trend), 1) / 365.25
            m = st.columns(3)
            m[0].metric("Total laporan pada rentang", f"{total_period:,}")
            m[1].metric("Rata-rata per hari", f"{avg_per_day:.1f}")
            m[2].metric("Lost-Time / Severe", f"{severe_period:,}", f"{severe_period / max(total_period, 1) * 100:.1f}%",
                        delta_color="off")

            with st.expander("💡 Catatan analis"):
                st.markdown(
                    f"Data membuktikan bahwa **jumlah kecelakaan yang terjadi sangat banyak**—mencapai **{total_period:,} insiden** "
                    f"dalam periode {years[0]}–{years[1]} (rata-rata **{avg_per_day:.1f} kecelakaan per hari**). "
                    f"Meskipun terdapat tren penurunan bertahap sejak 2011 akibat regulasi keselamatan yang semakin ketat, "
                    f"angka kecelakaan tahunan tetap stabil di angka **5.000 hingga 6.000+ insiden per tahun**, di mana lebih dari **55% di antaranya** "
                    f"tergolong cedera parah (*Lost-Time / Severe Injury*). Hal ini menjadi justifikasi empiris yang sangat kuat mengapa "
                    f"solusi pemantauan **Big Data dan Machine Learning** mutlak dibutuhkan di industri pertambangan."
                )

    with right:
        with st.container(border=True):
            st.markdown("#### 🏭 Komoditas")
            commodity = prep_meta.get("commodity_counts", {})
            if commodity:
                names = {"C": "Coal", "M": "Metal / Non-Metal"}
                dfp = pd.DataFrame({"Komoditas": [names.get(k, k) for k in commodity], "Jumlah": list(commodity.values())})
                fig = px.pie(dfp, names="Komoditas", values="Jumlah", hole=0.55, color="Komoditas",
                             color_discrete_map={"Coal": COLORS["coal"], "Metal / Non-Metal": COLORS["metal"]})
                fig.update_traces(textinfo="percent+label", textposition="outside")
                st.plotly_chart(style_fig(fig, 260, legend=False), width='stretch')
            else:
                st.caption("Belum ada data komoditas.")

        with st.container(border=True):
            st.markdown("#### 🧪 Partisi temporal")
            if n_total:
                rows = []
                for key, label in (("train", f"Train {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]}"),
                                   ("val", f"Validation {VAL_YEAR}"), ("test", f"Test {TEST_YEAR}")):
                    n, r = sizes.get(key) or 0, ratios.get(key, 0)
                    rows += [{"Partisi": label, "Kelas": "Class 1", "Jumlah": round(n * r)},
                             {"Partisi": label, "Kelas": "Class 0", "Jumlah": n - round(n * r)}]
                fig = px.bar(pd.DataFrame(rows), y="Partisi", x="Jumlah", color="Kelas", orientation="h",
                             color_discrete_map={"Class 1": COLORS["severe"], "Class 0": COLORS["minor"]})
                fig.update_yaxes(categoryorder="array", categoryarray=[r["Partisi"] for r in rows[::-2]])
                st.plotly_chart(style_fig(fig, 220), width='stretch')
                st.caption("Train: fit preprocessing & model · Validation: pilih model & threshold · Test: evaluasi akhir.")
            else:
                st.caption("Belum ada metadata partisi.")


# ==============================================================================
# 6. Halaman: Peta tambang
# ==============================================================================
def page_map():
    page_header(
        "Peta Sebaran Tambang",
        f"Koordinat dari <code>Mines.txt</code> digabung dengan statistik kecelakaan dan metrik graf ({GRAPH_START_YR}–{GRAPH_END_YR}).",
    )
    if df_geo.empty:
        missing_artifact("mine_geo_accidents.csv", "geo_and_viz_pipeline.py")
        return

    with st.container(border=True):
        f = st.columns([1.4, 1.2, 1, 1])
        states = f[0].multiselect("Negara bagian", sorted(df_geo["STATE"].dropna().unique()), placeholder="Semua")
        types = f[1].multiselect("Tipe tambang", sorted(df_geo["CURRENT_MINE_TYPE"].dropna().unique()), placeholder="Semua")
        comm_ids = sorted(int(x) for x in df_geo["community_id"].unique() if x >= 0)
        comm = f[2].selectbox("Komunitas Louvain", ["Semua"] + [f"Komunitas {i}" for i in comm_ids])
        max_acc = int(min(df_geo["total_accidents"].max(), 100)) if len(df_geo) else 1
        min_acc = f[3].slider("Minimal kecelakaan", 0, max(max_acc, 1), 1)

        g = st.columns([1.4, 1.2, 2])
        color_by = g[0].radio("Warna", ["Tipe tambang", "Komunitas", "% severe"], horizontal=True)
        size_by = g[1].radio("Ukuran", ["Kecelakaan", "Weighted degree"], horizontal=True)

    data = df_geo.copy()
    if states:
        data = data[data["STATE"].isin(states)]
    if types:
        data = data[data["CURRENT_MINE_TYPE"].isin(types)]
    if comm != "Semua":
        data = data[data["community_id"] == int(comm.split()[-1])]
    data = data[data["total_accidents"] >= min_acc]

    total_acc = int(data["total_accidents"].sum())
    severe_acc = int(data["severe_accidents"].sum())
    k = st.columns(4)
    kpi(k[0], "Tambang ditampilkan", f"{len(data):,}", f"dari {len(df_geo):,} tambang")
    kpi(k[1], "Total kecelakaan", f"{total_acc:,}", f"{GRAPH_START_YR}–{GRAPH_END_YR}", COLORS["metal"])
    kpi(k[2], "Lost-Time / Severe", f"{severe_acc / max(total_acc, 1) * 100:.1f}%", f"{severe_acc:,} laporan",
        COLORS["severe"])
    kpi(k[3], "Negara bagian", f"{data['STATE'].nunique():,}", "pada filter aktif", COLORS["minor"])
    st.write("")

    if data.empty:
        st.info("Tidak ada tambang yang memenuhi kombinasi filter.")
        return

    left, right = st.columns([2.2, 1])
    with left, st.container(border=True):
        plot = data.sort_values("total_accidents", ascending=False).head(3000).copy()

        size_col = "total_accidents" if size_by == "Kecelakaan" else "weighted_degree"
        vals = plot[size_col].clip(lower=0).astype(float)
        plot["radius"] = 1500 + np.sqrt(vals / max(vals.max(), 1)) * 30000

        if color_by == "Tipe tambang":
            plot["color"] = plot["CURRENT_MINE_TYPE"].map(lambda t: MINE_TYPE_RGB.get(t, [128, 128, 128]) + [200])
            legend = [(t, "#%02x%02x%02x" % tuple(rgb)) for t, rgb in MINE_TYPE_RGB.items()]
        elif color_by == "Komunitas":
            plot["color"] = plot["community_id"].map(lambda c: hex_to_rgb(community_hex(c)))
            legend = [("Tanpa komunitas", "#9aa0a6")] + [(f"Komunitas {i}", community_hex(i)) for i in comm_ids]
        else:
            pct = (plot["severe_accidents"] / plot["total_accidents"].replace(0, np.nan)).fillna(0)
            plot["color"] = [[int(30 + 187 * p), int(142 - 94 * p), int(62 - 25 * p), 210] for p in pct]
            legend = [("0% severe", "#1e8e3e"), ("50%", "#7a5f31"), ("100% severe", "#d93025")]

        # Nilai teks berasal dari data publik: di-escape sebelum masuk tooltip HTML pydeck.
        for col in ("CURRENT_MINE_NAME", "CURRENT_OPERATOR_NAME", "top_hazard", "STATE", "CURRENT_MINE_TYPE", "MINE_ID"):
            plot[col] = plot[col].astype(str).map(html.escape)

        layer = pdk.Layer(
            "ScatterplotLayer", data=plot, get_position="[lon, lat]", get_fill_color="color",
            get_radius="radius", pickable=True, opacity=0.75, stroked=True, filled=True,
            radius_min_pixels=2, radius_max_pixels=28, line_width_min_pixels=0.5,
            get_line_color=[255, 255, 255, 160],
        )
        if states or comm != "Semua":
            view = pdk.ViewState(latitude=float(plot["lat"].mean()), longitude=float(plot["lon"].mean()), zoom=4.5)
        else:
            view = pdk.ViewState(latitude=38.5, longitude=-96.0, zoom=3.4)

        st.pydeck_chart(pdk.Deck(
            layers=[layer], initial_view_state=view, map_provider="carto", map_style="light",
            tooltip={
                "html": "<b>{CURRENT_MINE_NAME}</b><br/>MINE_ID: {MINE_ID} · {STATE}<br/>"
                        "Tipe: {CURRENT_MINE_TYPE}<br/>Operator: {CURRENT_OPERATOR_NAME}<br/>"
                        "Kecelakaan: {total_accidents} (severe: {severe_accidents})<br/>"
                        "Bahaya dominan: {top_hazard}<br/>Komunitas: {community_id}",
                "style": {"backgroundColor": "#1f2328", "color": "white", "fontSize": "12px"},
            },
        ))
        chips(legend)
        if len(data) > len(plot):
            st.caption(f"Peta menampilkan {len(plot):,} tambang dengan kecelakaan terbanyak dari {len(data):,} hasil filter.")

    with right:
        with st.container(border=True):
            st.markdown("#### Negara bagian teratas")
            by_state = data.groupby("STATE")[["total_accidents", "severe_accidents"]].sum().nlargest(10, "total_accidents")
            fig = px.bar(by_state.reset_index(), x="total_accidents", y="STATE", orientation="h",
                         labels={"total_accidents": "Kecelakaan", "STATE": ""}, color_discrete_sequence=[COLORS["metal"]])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig, 260, legend=False), width='stretch')
        with st.container(border=True):
            st.markdown("#### Bahaya dominan")
            hz = data[data["total_accidents"] > 0]["top_hazard"].value_counts().head(8).reset_index()
            hz.columns = ["Tipe", "Tambang"]
            fig = px.bar(hz, x="Tambang", y="Tipe", orientation="h", color_discrete_sequence=[COLORS["accident_type"]])
            fig.update_yaxes(autorange="reversed", title="")
            st.plotly_chart(style_fig(fig, 260, legend=False), width='stretch')

    with st.expander(f"📋 Tabel tambang ({len(data):,} baris)"):
        cols = ["MINE_ID", "CURRENT_MINE_NAME", "STATE", "CURRENT_MINE_TYPE", "CURRENT_OPERATOR_NAME",
                "total_accidents", "severe_accidents", "top_hazard", "weighted_degree", "community_id"]
        table = data[cols].sort_values("total_accidents", ascending=False)
        st.dataframe(
            table, hide_index=True, width='stretch', height=360,
            column_config={
                "CURRENT_MINE_NAME": "Nama tambang", "CURRENT_MINE_TYPE": "Tipe",
                "CURRENT_OPERATOR_NAME": "Operator", "top_hazard": "Bahaya dominan",
                "total_accidents": st.column_config.ProgressColumn(
                    "Kecelakaan", format="%d", min_value=0, max_value=int(max(table["total_accidents"].max(), 1))),
                "severe_accidents": st.column_config.NumberColumn("Severe", format="%d"),
                "weighted_degree": st.column_config.NumberColumn("Weighted degree", format="%d"),
                "community_id": st.column_config.NumberColumn("Komunitas", format="%d"),
            },
        )
        st.download_button("⬇️ Unduh CSV", table.to_csv(index=False).encode("utf-8"),
                           "tambang_terfilter.csv", "text/csv")


# ==============================================================================
# 7. Halaman: Penyimpanan Parquet
# ==============================================================================
def page_storage():
    page_header(
        "Penyimpanan Big Data: HDFS + Parquet",
        "Berkas mentah MSHA diunggah ke raw zone HDFS, lalu Spark menulis curated zone dalam Parquet (Snappy) "
        "dengan partisi Hive-style <code>CAL_YR=YYYY</code>.",
    )
    if not storage:
        missing_artifact("hadoop_storage_summary.json", "hadoop_storage.py")
        return

    fs_uri = storage.get("filesystem_uri", "file:///")
    on_hdfs = str(fs_uri).startswith("hdfs://")
    if on_hdfs:
        st.success(f"Filesystem: **HDFS** `{fs_uri}` · block size {storage.get('block_size_mb', 'n/a')} MB · "
                   f"replication {storage.get('replication_factor', 'n/a')}")
    else:
        st.warning("Ringkasan ini dibuat tanpa HDFS (filesystem lokal `file:///`). Jalankan ulang "
                   "`src/hadoop_storage.py` dengan `MSHA_HDFS_URI=hdfs://localhost:9000` agar hasilnya dari HDFS.")

    k = st.columns(4)
    kpi(k[0], "Raw TXT", f"{storage.get('raw_csv_size_mb', 'n/a')} MB", "pipe-delimited", COLORS["other"])
    kpi(k[1], "Parquet (Snappy)", f"{storage.get('hadoop_parquet_size_mb', 'n/a')} MB", "columnar", COLORS["primary"])
    kpi(k[2], "Penghematan disk", f"{storage.get('storage_savings_pct', 'n/a')}%", "vs file mentah", COLORS["minor"])
    kpi(k[3], "Kueri Spark SQL", f"{storage.get('olap_query_time_seconds', 'n/a')} s",
        f"{storage.get('num_partitions', 'n/a')} partisi tahun", COLORS["metal"])
    st.write("")

    left, right = st.columns([1.3, 1])
    with left, st.container(border=True):
        st.markdown("#### Ukuran penyimpanan")
        dfs = pd.DataFrame({
            "Format": ["Raw TXT", "Parquet (Snappy)"],
            "MB": [storage.get("raw_csv_size_mb"), storage.get("hadoop_parquet_size_mb")],
        })
        fig = px.bar(dfs, x="MB", y="Format", orientation="h", text="MB", color="Format",
                     color_discrete_map={"Raw TXT": COLORS["other"], "Parquet (Snappy)": COLORS["primary"]})
        fig.update_traces(texttemplate="%{text:.1f} MB", textposition="outside")
        st.plotly_chart(style_fig(fig, 230, legend=False), width='stretch')
        if "rows_written" in storage:
            st.caption(f"Baris ditulis: {storage['rows_written']:,} · tanpa CAL_YR valid: {storage.get('rows_dropped_null_year', 0):,}")

    with right, st.container(border=True):
        st.markdown("#### Struktur partisi")
        years = sorted(r["Tahun"] for r in storage.get("sample_query_results", [])) or ["YYYY"]
        raw_dir = storage.get("raw_zone_path", "<base>/raw")
        curated = storage.get("curated_zone_path", "<base>/warehouse/msha_accidents")
        tree = f"{raw_dir}/\n"
        tree += "".join(f"├── {name}\n" for name in storage.get("raw_files", {})) or "├── Accidents.txt\n"
        tree += f"\n{curated}/\n├── ...\n"
        tree += "".join(f"├── CAL_YR={y}/\n│   └── part-*.snappy.parquet\n" for y in years[-3:])
        tree += "└── _SUCCESS"
        st.code(tree, language="text")
        st.caption("Partition pruning: kueri dengan filter `CAL_YR` hanya membaca direktori tahun yang relevan.")

    with st.container(border=True):
        st.markdown("#### Kueri analitik Spark SQL")
        q1, q2 = st.columns([1, 1.2])
        q1.code(
            "SELECT CAL_YR AS Tahun, COUNT(*) AS Total_Insiden,\n"
            "  SUM(CASE WHEN DEGREE_INJURY_CD IN ('01','02','03','04','05')\n"
            "      THEN 1 ELSE 0 END) AS Insiden_Lost_Time_Severe,\n"
            "  ROUND(AVG(TOT_EXPER), 2) AS Rata_Rata_Pengalaman_Tahun\n"
            "FROM msha_accidents\n"
            "WHERE CAL_YR BETWEEN 2018 AND 2023\n"
            "GROUP BY CAL_YR ORDER BY CAL_YR DESC;",
            language="sql",
        )
        res = pd.DataFrame(storage.get("sample_query_results", []))
        if not res.empty:
            res = res.sort_values("Tahun")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=res["Tahun"], y=res["Total_Insiden"], name="Total", marker_color=COLORS["primary"]))
            fig.add_trace(go.Bar(x=res["Tahun"], y=res["Insiden_Lost_Time_Severe"], name="Lost-Time / Severe",
                                 marker_color=COLORS["severe"]))
            fig.add_trace(go.Scatter(x=res["Tahun"], y=res["Rata_Rata_Pengalaman_Tahun"], name="Rata-rata pengalaman (thn)",
                                     yaxis="y2", mode="lines+markers", line=dict(color=COLORS["coal"], width=2)))
            fig.update_layout(barmode="group", xaxis=dict(dtick=1),
                              yaxis2=dict(overlaying="y", side="right", showgrid=False, title="Tahun"))
            q2.plotly_chart(style_fig(fig, 300), width='stretch')
            with st.expander("Lihat hasil kueri"):
                st.dataframe(res, hide_index=True, width='stretch')


# ==============================================================================
# 8. Halaman: Simulasi keparahan
# ==============================================================================
def score(frame):
    return best_model.predict_proba(preprocessor.transform(frame[CAT_COLS + NUM_COLS]))[:, 1]


def gauge(value, threshold):
    color = COLORS["severe"] if value >= threshold else COLORS["minor"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100,
        number={"suffix": "%", "valueformat": ".1f", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [{"range": [0, threshold * 100], "color": "#e6f4ea"},
                      {"range": [threshold * 100, 100], "color": "#fdecea"}],
            "threshold": {"line": {"color": "#1f2328", "width": 3}, "thickness": 0.85, "value": threshold * 100},
        },
    ))
    return style_fig(fig, 250, legend=False)


def page_predictor():
    page_header(
        "Simulasi Skenario Keparahan Cedera",
        "Masukkan karakteristik insiden yang <b>sudah dilaporkan</b>. Model memberi skor apakah insiden termasuk "
        "<b>Lost-Time / Severe</b> (kode 01–05). Ini bukan prediksi apakah kecelakaan akan terjadi.",
    )
    if best_model is None or preprocessor is None:
        missing_artifact("best_model.joblib", "models_baseline.py")
        return

    cats = model_categories(preprocessor) or model_meta.get("categories", {})
    if not cats:
        st.error("Kategori model tidak dapat dibaca. Jalankan ulang `python src/models_baseline.py`.")
        return
    threshold = float(model_meta.get("threshold", 0.5))
    base_rate = prep_meta.get("split", {}).get("class1_ratio", {}).get("train")
    if not model_meta:
        st.warning("`model_metadata.json` belum ada; memakai ambang 0.5. Jalankan ulang `python src/models_baseline.py`.")

    medians = prep_meta.get("numeric_medians_train", {})
    labels = {
        "SUBUNIT": "Lokasi kerja (subunit)", "COAL_METAL_IND": "Komoditas (C = coal, M = metal/non-metal)",
        "SHIFT_PERIOD": "Shift", "ACCIDENT_TYPE": "Tipe insiden", "MINING_EQUIP": "Peralatan terlibat",
        "ACTIVITY": "Aktivitas pekerja",
    }
    num_labels = {"TOT_EXPER": "Total pengalaman tambang (thn)", "MINE_EXPER": "Pengalaman di tambang ini (thn)",
                  "JOB_EXPER": "Pengalaman di posisi ini (thn)"}

    left, right = st.columns([1.25, 1])
    with left, st.container(border=True):
        st.markdown("#### 📝 Skenario insiden")
        with st.form("scenario_form"):
            values = {}
            fc = st.columns(2)
            for i, col in enumerate(CAT_COLS):
                options = cats.get(col, [])
                prev = st.session_state.get("scenario_data", {}).get(col)
                idx = options.index(prev) if prev in options else (options.index("DAY") if "DAY" in options else 0)
                values[col] = fc[i % 2].selectbox(labels.get(col, col), options, index=idx)
            nc = st.columns(3)
            for i, col in enumerate(NUM_COLS):
                median = medians.get(col)
                median = 5.0 if median is None or pd.isna(median) else round(float(median), 1)
                default = float(np.clip(st.session_state.get("scenario_data", {}).get(col, median), 0.0, 60.0))
                values[col] = nc[i].number_input(num_labels[col], 0.0, 60.0, default, 0.5)
            submitted = st.form_submit_button("🔍 Hitung skor", type="primary", width='stretch')

    if submitted:
        st.session_state["scenario_data"] = values

        s = float(score(pd.DataFrame([values]))[0])
        st.session_state["score"] = s

        st.session_state.setdefault("history", []).insert(0, {
            "Waktu": datetime.now().strftime("%H:%M:%S"),
            **values,
            "Skor": round(s, 3),
            "Klasifikasi": (
                "Lost-Time / Severe"
                if s >= threshold
                else "No Lost-Time"
            ),
        })

        st.session_state["history"] = st.session_state["history"][:15]
    with right, st.container(border=True):
        st.markdown("#### 📍 Hasil")
        st.caption(f"Model: **{model_meta.get('model_name', 'n/a')}** · ambang (dipilih di validation): **{threshold:.2f}**")
        if "score" not in st.session_state:
            st.info("Isi skenario lalu klik **Hitung skor**.")
        else:
            s = st.session_state["score"]
            st.plotly_chart(gauge(s, threshold), width='stretch')
            positive = s >= threshold
            st.markdown(
                f'<div class="verdict {"pos" if positive else "neg"}"><b>'
                f'{"🚨 Lost-Time / Severe" if positive else "✅ No Lost-Time"}</b><br/>'
                f'Skor {s * 100:.1f}% {"≥" if positive else "<"} ambang {threshold * 100:.0f}%.'
                + (f" Proporsi Class 1 pada data train: {base_rate * 100:.1f}%." if base_rate else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption("Skor belum dikalibrasi; gunakan sebagai peringkat relatif, bukan peluang pasti.")

    if "scenario_data" not in st.session_state:
        return

    st.write("")
    with st.container(border=True):
        st.markdown("#### 🔬 Analisis what-if")
        st.caption("Ubah satu variabel, variabel lain tetap seperti skenario di atas.")
        base = st.session_state["scenario_data"]
        w1, w2 = st.columns([1, 1.4])
        with w1:
            num_var = st.radio("Variabel numerik", NUM_COLS, format_func=num_labels.get)
            grid = np.arange(0, 40.5, 0.5)
            frame = pd.DataFrame([base] * len(grid))
            frame[num_var] = grid
            fig = go.Figure(go.Scatter(x=grid, y=score(frame) * 100, mode="lines",
                                       line=dict(color=COLORS["primary"], width=3), name="Skor"))
            fig.add_hline(y=threshold * 100, line_dash="dash", line_color=COLORS["severe"],
                          annotation_text="ambang", annotation_position="top left")
            fig.add_vline(x=float(base[num_var]), line_dash="dot", line_color=COLORS["muted"])
            fig.update_layout(xaxis_title=num_labels[num_var], yaxis_title="Skor (%)", yaxis_range=[0, 100])
            st.plotly_chart(style_fig(fig, 330, legend=False), width='stretch')
        with w2:
            cat_var = st.selectbox("Variabel kategorikal", CAT_COLS, index=CAT_COLS.index("ACCIDENT_TYPE"),
                                   format_func=lambda c: labels.get(c, c))
            options = cats.get(cat_var, [])
            frame = pd.DataFrame([base] * len(options))
            frame[cat_var] = options
            res = pd.DataFrame({"Kategori": options, "Skor": score(frame) * 100}).sort_values("Skor")
            res["Status"] = np.where(res["Kategori"] == base[cat_var], "Skenario saat ini", "Alternatif")
            fig = px.bar(res, x="Skor", y="Kategori", orientation="h", color="Status",
                         color_discrete_map={"Skenario saat ini": COLORS["metal"], "Alternatif": "#aecbfa"})
            fig.add_vline(x=threshold * 100, line_dash="dash", line_color=COLORS["severe"])
            fig.update_layout(xaxis_title="Skor (%)", yaxis_title="", xaxis_range=[0, 100])
            fig.update_yaxes(tickfont=dict(size=10))
            st.plotly_chart(style_fig(fig, max(330, 18 * len(options))), width='stretch')

    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"🕘 Riwayat skenario ({len(history)})"):
            st.dataframe(
                pd.DataFrame(history), hide_index=True, width='stretch',
                column_config={"Skor": st.column_config.ProgressColumn("Skor", format="%.3f", min_value=0, max_value=1)},
            )
            if st.button("Hapus riwayat"):
                st.session_state["history"] = []
                st.rerun()


# ==============================================================================
# 9. Halaman: Benchmark model
# ==============================================================================
METRICS = {
    "Accuracy": "Accuracy", "Precision_Class1": "Precision (C1)", "Recall_Class1": "Recall (C1)",
    "F1_Class1": "F1 (C1)", "F1_Macro": "F1 Macro", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC",
}


def page_benchmark():
    page_header(
        "Benchmark: Scikit-Learn vs Apache Spark MLlib",
        f"Evaluasi pada test set {TEST_YEAR}. Semua model memakai partisi, fitur, class weight, dan fungsi metrik yang sama; "
        "threshold setiap model dipilih pada validation set.",
    )
    if not benchmarks:
        missing_artifact("benchmark_results.json", "models_spark.py")
        return

    df = pd.DataFrame([
        {"Model": m.get("Model Name", name), "Framework": m.get("Framework", ""), **{v: m.get(k) for k, v in METRICS.items()},
         "Val F1 (C1)": m.get("Val_F1_Class1"), "Threshold": m.get("Optimal_Threshold"),
         "Fit (s)": m.get("Fit_Time_Seconds"), "Inferensi (s)": m.get("Inference_Time_Seconds"),
         "CM": m.get("Confusion_Matrix")}
        for name, m in benchmarks.items()
    ])
    has_baseline = (df["Framework"] == "Baseline").any()
    show_base = st.toggle("Tampilkan baseline kelas mayoritas", value=has_baseline, disabled=not has_baseline)
    view = df if show_base else df[df["Framework"] != "Baseline"]
    models = view[view["Framework"] != "Baseline"]
    color_map = {m: FRAMEWORK_COLORS.get(f, "#888") for m, f in zip(df["Model"], df["Framework"])}
    palette = ["#1a73e8", "#8ab4f8", "#e37400", "#fcc934", "#9aa0a6"]
    model_colors = {m: palette[i % len(palette)] for i, m in enumerate(df["Model"])}

    k = st.columns(4)
    kpi(k[0], "Model dashboard", model_meta.get("model_name", "n/a"),
        f"dipilih dari Val F1 = {model_meta.get('val_f1_class1', 'n/a')}")
    if not models.empty:
        best_auc = models.loc[models["ROC-AUC"].idxmax()]
        kpi(k[1], "ROC-AUC tertinggi", f"{best_auc['ROC-AUC']:.4f}", best_auc["Model"], COLORS["minor"])
        best_f1 = models.loc[models["F1 (C1)"].idxmax()]
        kpi(k[2], "F1 (C1) test tertinggi", f"{best_f1['F1 (C1)']:.4f}", best_f1["Model"], COLORS["severe"])
        fastest = models.loc[models["Fit (s)"].idxmin()]
        kpi(k[3], "Fit tercepat", f"{fastest['Fit (s)']:.2f} s", fastest["Model"], COLORS["metal"])
    st.write("")

    t1, t2, t3, t4 = st.tabs(["📊 Metrik", "🧮 Confusion matrix", "⏱️ Waktu komputasi", "📋 Tabel"])

    with t1:
        chosen = st.multiselect("Metrik", list(METRICS.values()),
                                default=["Precision (C1)", "Recall (C1)", "F1 (C1)", "ROC-AUC", "PR-AUC"])
        if chosen:
            c1, c2 = st.columns([1.5, 1])
            long = view.melt(id_vars=["Model"], value_vars=chosen, var_name="Metrik", value_name="Nilai")
            fig = px.bar(long, x="Metrik", y="Nilai", color="Model", barmode="group", color_discrete_map=model_colors,
                         text_auto=".3f")
            fig.update_traces(textfont_size=9, textangle=0)
            fig.update_layout(yaxis_range=[0, 1.05])
            c1.plotly_chart(style_fig(fig, 420), width='stretch')

            radar = go.Figure()
            for _, row in view.iterrows():
                vals = [row[m] for m in chosen]
                radar.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=chosen + chosen[:1], name=row["Model"],
                                                fill="toself", opacity=0.45, line=dict(color=model_colors[row["Model"]])))
            radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickfont_size=9)))
            c2.plotly_chart(style_fig(radar, 420), width='stretch')

    with t2:
        c1, c2 = st.columns([1, 2])
        model = c1.selectbox("Model", view["Model"].tolist())
        normalize = c1.toggle("Normalisasi per kelas aktual", value=False)
        cm = np.array(view.loc[view["Model"] == model, "CM"].iloc[0])
        if normalize:
            z, fmt = cm / cm.sum(axis=1, keepdims=True), ".2%"
        else:
            z, fmt = cm, ","
        fig = px.imshow(z, x=["Prediksi 0", "Prediksi 1"], y=["Aktual 0", "Aktual 1"],
                        color_continuous_scale="Blues", text_auto=fmt, aspect="auto")
        fig.update_traces(textfont_size=18)
        c2.plotly_chart(style_fig(fig, 340, legend=False), width='stretch')
        tn, fp, fn, tp = cm.ravel()
        c1.markdown(f"- **TP** (severe terdeteksi): {tp:,}\n- **FN** (severe terlewat): {fn:,}\n"
                    f"- **FP** (alarm palsu): {fp:,}\n- **TN**: {tn:,}")

    with t3:
        timing = models.melt(id_vars=["Model", "Framework"], value_vars=["Fit (s)", "Inferensi (s)"],
                             var_name="Tahap", value_name="Detik")
        log = st.toggle("Skala logaritmik", value=True)
        fig = px.bar(timing, x="Model", y="Detik", color="Framework", facet_col="Tahap", text_auto=".2f",
                     color_discrete_map=FRAMEWORK_COLORS)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_yaxes(matches=None, type="log" if log else "linear")
        fig.update_xaxes(title="", tickangle=-20)
        st.plotly_chart(style_fig(fig, 420), width='stretch')
        st.caption("Fit Spark tidak termasuk waktu start SparkSession. Pada satu mesin dan data berukuran sedang, "
                   "overhead penjadwalan Spark membuat waktunya lebih lama; keunggulan Spark muncul saat data dibagi ke banyak node.")

    with t4:
        table = view.drop(columns=["CM"])
        st.dataframe(
            table.style.highlight_max(subset=[c for c in ["F1 (C1)", "ROC-AUC", "PR-AUC"] if c in table], color="#c8e6c9")
                       .format(precision=4, na_rep="—"),
            hide_index=True, width='stretch',
        )
        st.download_button("⬇️ Unduh CSV", table.to_csv(index=False).encode("utf-8"), "benchmark.csv", "text/csv")


# ==============================================================================
# 10. Halaman: Graph analytics
# ==============================================================================
@st.cache_data(show_spinner=False)
def spring_positions(edges):
    G = nx.Graph()
    G.add_edges_from((u, v) for u, v, _ in edges)
    k = 2.0 / np.sqrt(max(G.number_of_nodes(), 1))
    pos = nx.spring_layout(G, k=k, iterations=100, seed=RANDOM_SEED)
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}


@st.cache_data(show_spinner=False)
def node_index(nodes):
    return nodes.set_index("node_id") if not nodes.empty else nodes


def explorer_edges():
    if not df_explorer_edges.empty:
        return df_explorer_edges, True
    edges = pd.DataFrame(graph_summary.get("top_edges", []))
    if not edges.empty:  # artefak lama: pastikan source = tambang
        swap = ~edges["source"].str.startswith("mine::")
        edges.loc[swap, ["source", "target"]] = edges.loc[swap, ["target", "source"]].values
    return edges, False


def network_tab(nodes_idx):
    edges, balanced = explorer_edges()
    if edges.empty:
        missing_artifact("graph_explorer_edges.csv", "graph_analytics.py")
        return
    if not balanced:
        st.info("Memakai `graph_top_edges.json` (hampir semua edge bertipe CITED_FOR). Jalankan ulang "
                "`python src/graph_analytics.py` untuk membuat `graph_explorer_edges.csv` yang memuat kedua tipe relasi.")

    c = st.columns([1.6, 1.2, 1])
    types = c[0].multiselect("Relasi", list(EDGE_TYPE_LABEL), default=list(EDGE_TYPE_LABEL), format_func=EDGE_TYPE_LABEL.get)
    top_n = c[1].slider("Edge terberat per relasi", 10, 150, 60, 10)
    show_labels = c[2].toggle("Label standar & tipe", value=True)

    sub = (edges[edges["edge_type"].isin(types)].sort_values("weight", ascending=False)
           .groupby("edge_type", group_keys=False).head(top_n))
    if sub.empty:
        st.info("Pilih minimal satu relasi.")
        return

    pos = spring_positions(tuple(sub[["source", "target", "weight"]].itertuples(index=False, name=None)))
    fig = go.Figure()
    edge_colors = {"CITED_FOR": "rgba(217,48,37,0.35)", "REPORTED_ACCIDENT": "rgba(24,128,56,0.45)"}
    for etype, grp in sub.groupby("edge_type"):
        xs, ys = [], []
        for u, v in zip(grp["source"], grp["target"]):
            xs += [pos[u][0], pos[v][0], None]
            ys += [pos[u][1], pos[v][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip", name=EDGE_TYPE_LABEL[etype],
                                 line=dict(width=1, color=edge_colors.get(etype, "#bbb"))))

    local_w = pd.concat([sub[["source", "weight"]].rename(columns={"source": "n"}),
                         sub[["target", "weight"]].rename(columns={"target": "n"})]).groupby("n")["weight"].sum()
    symbols = {"mine": "circle", "standard": "diamond", "accident_type": "square"}
    for ntype in ("mine", "standard", "accident_type"):
        ids = [n for n in local_w.index if n.startswith(f"{ntype}::")]
        if not ids:
            continue
        hover, text = [], []
        for n in ids:
            raw = n.split("::", 1)[1]
            info = nodes_idx.loc[n] if n in nodes_idx.index else None
            name = mine_label(raw) if ntype == "mine" else raw
            line = f"<b>{html.escape(name)}</b><br>{NODE_TYPE_LABEL[ntype]}<br>Bobot di tampilan: {int(local_w[n]):,}"
            if info is not None:
                line += f"<br>Weighted degree global: {int(info['weighted_degree']):,}<br>PageRank: {info['pagerank']:.5f}"
                if ntype == "mine" and info["community_id"] >= 0:
                    line += f"<br>Komunitas: {int(info['community_id'])}"
            hover.append(line)
            text.append(raw[:28] if (show_labels and ntype != "mine") else "")
        sizes = 9 + 26 * np.sqrt(local_w[ids].values / local_w.max())
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ids], y=[pos[n][1] for n in ids], mode="markers+text",
            name=NODE_TYPE_LABEL[ntype], text=text, textposition="top center", textfont=dict(size=9),
            hovertext=hover, hoverinfo="text",
            marker=dict(size=sizes, color=COLORS[ntype], symbol=symbols[ntype], line=dict(width=1, color="white")),
        ))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), hovermode="closest",
                      plot_bgcolor="#fbfcfe")
    st.plotly_chart(style_fig(fig, 640), width='stretch')
    st.caption("Klik item legenda untuk menyembunyikan/menampilkan tipe simpul atau relasi. "
               "Ukuran simpul = jumlah bobot edge yang tampil. Bobot = frekuensi laporan, bukan risiko ternormalisasi.")


def inspector_tab():
    affinity = closeness.get("mine_affinity", {})
    peers = closeness.get("peer_proximity", {})
    if not affinity:
        missing_artifact("mine_accident_closeness.json", "graph_analytics.py")
        return

    ids = sorted(affinity, key=lambda m: -affinity[m].get("total_accidents", 0))
    c1, c2 = st.columns([2, 1])
    mine = c1.selectbox(f"Tambang ({len(ids):,} tambang dengan ≥3 kecelakaan, urut terbanyak)", ids, format_func=mine_label)
    info = affinity[mine]
    geo = df_geo[df_geo["MINE_ID"] == mine]
    with c2:
        kpi(st, f"Kecelakaan {GRAPH_START_YR}–{GRAPH_END_YR}", f"{info.get('total_accidents', 0):,}",
            (f"{geo.iloc[0]['STATE']} · {geo.iloc[0]['CURRENT_MINE_TYPE']}" if not geo.empty else ""), COLORS["metal"])

    hz = pd.DataFrame(info.get("top_hazards", []))
    left, right = st.columns([1.4, 1])
    with left, st.container(border=True):
        st.markdown("#### Profil tipe kecelakaan vs rata-rata")
        if not hz.empty:
            long = hz.melt(id_vars=["accident_type"], value_vars=["mine_affinity_pct", "baseline_pct"],
                           var_name="Seri", value_name="Persen")
            long["Seri"] = long["Seri"].map({"mine_affinity_pct": "Tambang ini", "baseline_pct": "Rata-rata tambang aktif"})
            fig = px.bar(long, x="Persen", y="accident_type", color="Seri", barmode="group", orientation="h",
                         color_discrete_map={"Tambang ini": COLORS["metal"], "Rata-rata tambang aktif": "#c4c7c5"})
            fig.update_yaxes(autorange="reversed", title="", tickfont=dict(size=10))
            fig.update_xaxes(title="% dari kecelakaan")
            st.plotly_chart(style_fig(fig, 340), width='stretch')
            lift = hz[["accident_type", "count", "risk_lift_ratio"]].rename(
                columns={"accident_type": "Tipe kecelakaan", "count": "Kasus", "risk_lift_ratio": "Lift"})
            st.dataframe(lift, hide_index=True, width='stretch',
                         column_config={"Lift": st.column_config.NumberColumn(format="%.2f×")})
            st.caption("Lift = proporsi di tambang ini ÷ proporsi rata-rata. Lift > 1 berarti tipe ini lebih sering "
                       "muncul di tambang ini. Pada tambang dengan sedikit kasus, lift mudah bernilai ekstrem.")

    with right, st.container(border=True):
        st.markdown("#### Tambang dengan profil paling mirip")
        p = pd.DataFrame(peers.get(mine, []))
        if not p.empty:
            p["Tambang"] = p["peer_mine_id"].map(mine_label)
            fig = px.bar(p, x="similarity_pct", y="Tambang", orientation="h", text="similarity_pct",
                         color_discrete_sequence=[COLORS["primary"]])
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
            fig.update_yaxes(autorange="reversed", title="", tickfont=dict(size=10))
            fig.update_xaxes(title="Cosine similarity (%)", range=[0, 100])
            st.plotly_chart(style_fig(fig, 340, legend=False), width='stretch')
            st.caption("Cosine similarity antar vektor jumlah tipe kecelakaan per tambang.")


def centrality_tab(nodes_idx):
    c = st.columns([1.4, 1, 1])
    ntype = c[0].radio("Tipe simpul", list(NODE_TYPE_LABEL), format_func=NODE_TYPE_LABEL.get, horizontal=True)
    metric = c[1].radio("Metrik", ["weighted_degree", "pagerank"], horizontal=True,
                        format_func=lambda m: "Weighted degree" if m == "weighted_degree" else "PageRank")
    top_n = c[2].slider("Top-N", 5, 30, 15)

    sel = df_nodes[df_nodes["node_type"] == ntype].nlargest(top_n, metric).copy()
    sel["Nama"] = sel["label"].map(mine_label) if ntype == "mine" else sel["label"]
    fig = px.bar(sel, x=metric, y="Nama", orientation="h", color_discrete_sequence=[COLORS[ntype]],
                 hover_data={"weighted_degree": ":,", "pagerank": ":.5f", "Nama": False})
    fig.update_yaxes(autorange="reversed", title="", tickfont=dict(size=10))
    fig.update_xaxes(title="Weighted degree" if metric == "weighted_degree" else "PageRank")
    st.plotly_chart(style_fig(fig, max(360, 24 * top_n), legend=False), width='stretch')
    st.caption("Weighted degree tinggi = banyak laporan/pelanggaran (high-frequency hub). Tanpa normalisasi jam kerja, "
               "tambang besar cenderung berada di atas.")


def community_tab():
    mines = df_nodes[(df_nodes["node_type"] == "mine") & (df_nodes["community_id"] >= 0)]
    if mines.empty:
        st.info("Belum ada hasil komunitas.")
        return
    sizes = mines.groupby("community_id").size().reset_index(name="Tambang")
    left, right = st.columns([1, 2])
    with left:
        fig = px.bar(sizes, x="community_id", y="Tambang", text="Tambang",
                     color=sizes["community_id"].astype(str),
                     color_discrete_map={str(i): community_hex(i) for i in sizes["community_id"]})
        fig.update_xaxes(title="Komunitas", type="category")
        st.plotly_chart(style_fig(fig, 280, legend=False), width='stretch')
        cid = st.selectbox("Lihat komunitas", sizes["community_id"].tolist(), format_func=lambda i: f"Komunitas {i}")
    members = df_geo[df_geo["community_id"] == cid] if not df_geo.empty else pd.DataFrame()
    with right:
        k = st.columns(3)
        kpi(k[0], "Anggota", f"{int(sizes.loc[sizes['community_id'] == cid, 'Tambang'].iloc[0]):,}", "tambang",
            community_hex(cid))
        if not members.empty:
            kpi(k[1], "Kecelakaan", f"{int(members['total_accidents'].sum()):,}", "total anggota", COLORS["metal"])
            kpi(k[2], "Negara bagian", f"{members['STATE'].nunique()}", ", ".join(members["STATE"].value_counts().index[:3]),
                COLORS["minor"])
            st.write("")
            m1, m2 = st.columns(2)
            st_counts = members["STATE"].value_counts().head(8).reset_index()
            st_counts.columns = ["Negara bagian", "Tambang"]
            fig = px.bar(st_counts, x="Tambang", y="Negara bagian", orientation="h",
                         color_discrete_sequence=[community_hex(cid)])
            fig.update_yaxes(autorange="reversed")
            m1.plotly_chart(style_fig(fig, 260, legend=False), width='stretch')
            mt = members["CURRENT_MINE_TYPE"].value_counts().reset_index()
            mt.columns = ["Tipe", "Tambang"]
            fig = px.pie(mt, names="Tipe", values="Tambang", hole=0.5, color="Tipe",
                         color_discrete_map={t: "#%02x%02x%02x" % tuple(rgb) for t, rgb in MINE_TYPE_RGB.items()})
            m2.plotly_chart(style_fig(fig, 260), width='stretch')

    if not members.empty:
        with st.expander(f"📋 Anggota komunitas {cid}"):
            st.dataframe(
                members[["MINE_ID", "CURRENT_MINE_NAME", "STATE", "CURRENT_MINE_TYPE", "total_accidents",
                         "weighted_degree", "top_hazard"]].sort_values("weighted_degree", ascending=False),
                hide_index=True, width='stretch',
            )
    st.caption("Komunitas Louvain pada proyeksi mine-to-mine (Jaccard tetangga ≥ 0.20) untuk 300 tambang "
               "dengan weighted degree tertinggi; tambang lain tidak memiliki komunitas.")


def page_graph():
    page_header(
        "Graph Analytics: Tambang · Standar Pelanggaran · Tipe Kecelakaan",
        f"Graf heterogen dari Accidents dan Violations {GRAPH_START_YR}–{GRAPH_END_YR}. "
        "Edge menunjukkan koeksistensi pada tambang yang sama, bukan hubungan sebab-akibat.",
    )
    if df_nodes.empty:
        missing_artifact("graph_nodes_summary.csv", "graph_analytics.py")
        return

    counts = df_nodes["node_type"].value_counts()
    k = st.columns(4)
    kpi(k[0], "Simpul", fmt_int(graph_summary.get("total_nodes", len(df_nodes))),
        f"{counts.get('mine', 0):,} tambang · {counts.get('standard', 0):,} standar · {counts.get('accident_type', 0)} tipe")
    kpi(k[1], "Edge", fmt_int(graph_summary.get("total_edges")), "berbobot frekuensi", COLORS["metal"])
    kpi(k[2], "Komunitas Louvain", fmt_int(graph_summary.get("num_communities")), "proyeksi mine-to-mine", COLORS["minor"])
    kpi(k[3], "Tambang di inspector", fmt_int(len(closeness.get("mine_affinity", {}))), "≥3 kecelakaan",
        COLORS["severe"])
    st.write("")

    nodes_idx = node_index(df_nodes)
    t1, t2, t3, t4 = st.tabs(["🕸️ Jaringan", "🎯 Inspector tambang", "📊 Sentralitas", "🧩 Komunitas"])
    with t1:
        network_tab(nodes_idx)
    with t2:
        inspector_tab()
    with t3:
        centrality_tab(nodes_idx)
    with t4:
        community_tab()


# ==============================================================================
# 11. Routing
# ==============================================================================
{
    "overview": page_overview,
    "map": page_map,
    "storage": page_storage,
    "predictor": page_predictor,
    "benchmark": page_benchmark,
    "graph": page_graph,
}[page]()
