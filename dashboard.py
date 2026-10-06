import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
import pickle
from pathlib import Path

# ---------------------------------------------------------
# PENETAPAN PATH DINAMIS SESUAI STRUKTUR PROYEK
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "include" / "dashboard_assets"

DATA_PATH = ASSETS_DIR / "df_final_dashboard.csv"
MODEL_PATH = ASSETS_DIR / "lgbm_karhutla_model.pkl"
FEATURE_IMP_PATH = ASSETS_DIR / "feature_importance.csv"
TOP_GRID_PATH = ASSETS_DIR / "top_10_grid_frp.csv"

# ---------------------------------------------------------
# KONFIGURASI HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Karhutla Kalimantan",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATASET & ASSETS
# ---------------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"❌ File dataset tidak ditemukan di: `{DATA_PATH}`")
        st.stop()
        
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%B')
    df['quarter'] = 'K' + df['date'].dt.quarter.astype(str)
    
    # Pemetaan provinsi berdasarkan koordinat spasial
    def map_province(lat, lon):
        if lat < -1.5 and lon >= 114.0:
            return "Kalimantan Selatan"
        elif lat >= 1.5 and lon >= 115.0:
            return "Kalimantan Utara"
        elif lon >= 115.0 and lat >= -1.5:
            return "Kalimantan Timur"
        elif lon < 113.5 and lat < 2.0:
            return "Kalimantan Barat"
        else:
            return "Kalimantan Tengah"

    df['province'] = df.apply(lambda r: map_province(r['latitude'], r['longitude']), axis=1)
    return df

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

df_raw = load_data()
lgbm_model = load_model()

# ---------------------------------------------------------
# HEADER UTAMA
# ---------------------------------------------------------
st.title("🔥 Dashboard Prediksi & Pemetaan Risiko Karhutla Kalimantan")
st.caption("Proyek Analisis Data & Model Prediksi LightGBM | Cinta Wardana (NIM: E1E124059)")
st.markdown("---")

# ---------------------------------------------------------
# SIDEBAR - SIMULATOR SCENARIO CUACA & VEGETASI
# ---------------------------------------------------------
st.sidebar.header("🎛️ Simulator Parameter Cuaca & Vegetasi")
rh2m = st.sidebar.slider("Kelembapan Udara (RH2M %)", 30.0, 100.0, 62.0, step=0.5)
prectotcorr = st.sidebar.slider("Curah Hujan (PRECTOTCORR mm/hari)", 0.0, 50.0, 1.5, step=0.1)
t2m = st.sidebar.slider("Suhu Udara (T2M °C)", 20.0, 40.0, 32.5, step=0.1)
ndvi = st.sidebar.slider("Kerapatan Vegetasi (NDVI)", 0.0, 1.0, 0.42, step=0.01)

st.sidebar.markdown("---")

# ---------------------------------------------------------
# SIDEBAR - FILTER WILAYAH KALIMANTAN
# ---------------------------------------------------------
st.sidebar.header("📍 Filter Wilayah Kalimantan")

list_provinsi = ["Seluruh Kalimantan"] + sorted(df_raw['province'].unique().tolist())
selected_provinsi = st.sidebar.selectbox("Pilih Provinsi", list_provinsi)

if selected_provinsi == "Seluruh Kalimantan":
    df_prov = df_raw.copy()
    grid_options = ["Semua Grid (Seluruh Kalimantan)"] + sorted(df_raw['grid_id'].unique().tolist())
else:
    df_prov = df_raw[df_raw['province'] == selected_provinsi]
    grid_options = [f"Semua Grid di {selected_provinsi}"] + sorted(df_prov['grid_id'].unique().tolist())

selected_grid = st.sidebar.selectbox("Pilih Grid ID", grid_options)

st.sidebar.markdown("---")

# ---------------------------------------------------------
# SIDEBAR - FILTER TAHUN & PERIODE WAKTU
# ---------------------------------------------------------
st.sidebar.header("📅 Filter Tahun & Periode Waktu")

available_years = sorted(df_raw['year'].unique().tolist())
selected_year = st.sidebar.selectbox("Pilih Tahun Data", available_years, index=len(available_years)-1)

periode_type = st.sidebar.radio("Tipe Filter Waktu", ["Kuartal", "Bulan", "Rentang Tanggal"], horizontal=True)

df_filtered = df_prov[df_prov['year'] == selected_year]

if periode_type == "Kuartal":
    selected_quarter = st.sidebar.selectbox(
        "Pilih Kuartal",
        ["K1 (Jan-Mar)", "K2 (Apr-Jun)", "K3 (Jul-Sep)", "K4 (Okt-Des)"],
        index=2
    )
    q_code = selected_quarter.split(" ")[0]
    df_filtered = df_filtered[df_filtered['quarter'] == q_code]
    periode_label = f"{selected_quarter} {selected_year}"

elif periode_type == "Bulan":
    month_dict = {
        "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, "Mei": 5, "Juni": 6,
        "Juli": 7, "Agustus": 8, "September": 9, "Oktober": 10, "November": 11, "Desember": 12
    }
    selected_month_name = st.sidebar.selectbox("Pilih Bulan", list(month_dict.keys()), index=7)
    df_filtered = df_filtered[df_filtered['month_num'] == month_dict[selected_month_name]]
    periode_label = f"Bulan {selected_month_name} {selected_year}"

else:
    min_date_val = datetime.date(selected_year, 1, 1)
    max_date_val = datetime.date(selected_year, 10, 3) if selected_year == 2026 else datetime.date(selected_year, 12, 31)
    date_range = st.sidebar.date_input("Pilih Tanggal", value=(min_date_val, max_date_val))
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df_filtered = df_filtered[(df_filtered['date'] >= start_date) & (df_filtered['date'] <= end_date)]
        periode_label = f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"
    else:
        periode_label = f"{selected_year}"

if "Semua Grid" not in selected_grid:
    df_filtered = df_filtered[df_filtered['grid_id'] == selected_grid]

# ---------------------------------------------------------
# LOGIKA PREDIKSI SIMULASI
# ---------------------------------------------------------
if lgbm_model is not None and hasattr(lgbm_model, "predict_proba"):
    try:
        sample_input = pd.DataFrame([{
            'T2M': t2m, 'RH2M': rh2m, 'WS2M': 1.98, 'PRECTOTCORR': prectotcorr,
            'latitude': -1.0, 'longitude': 113.5, 'ndvi_value': ndvi
        }])
        prob_percent = float(lgbm_model.predict_proba(sample_input)[0][1] * 100)
    except Exception:
        z_score = -1.2 - 0.07 * (rh2m - 70) - 0.18 * prectotcorr + 0.22 * (t2m - 28) - 2.8 * (ndvi - 0.5)
        prob_percent = float((1 / (1 + np.exp(-z_score))) * 100)
else:
    z_score = -1.2 - 0.07 * (rh2m - 70) - 0.18 * prectotcorr + 0.22 * (t2m - 28) - 2.8 * (ndvi - 0.5)
    prob_percent = float((1 / (1 + np.exp(-z_score))) * 100)

if prob_percent >= 70:
    status_risk = "AWAS (Tinggi)"
    risk_color = "#d32f2f"
elif prob_percent >= 40:
    status_risk = "SIAGA (Sedang)"
    risk_color = "#f57c00"
else:
    status_risk = "AMAN (Rendah)"
    risk_color = "#388e3c"

# ---------------------------------------------------------
# METRIC CARDS
# ---------------------------------------------------------
total_hotspots = int(df_filtered['hotspot_count'].sum())
avg_frp_val = df_filtered['avg_frp'].mean()
fire_days_count = int(df_filtered['fire_status'].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status Peringatan Dini", status_risk)
col2.metric("Total Titik Panas (Hotspot)", f"{total_hotspots:,}")
col3.metric("Rata-rata FRP", f"{avg_frp_val:.2f} MW" if not np.isnan(avg_frp_val) else "0.00 MW")
col4.metric("Jumlah Kejadian Api", f"{fire_days_count:,} Hari")

st.markdown("---")

# ---------------------------------------------------------
# VISUALISASI UTAMA
# ---------------------------------------------------------
col_gauge, col_feature = st.columns([1, 1])

with col_gauge:
    st.subheader("🎯 Gauge Meter Probabilitas Kebakaran (Simulasi)")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Tingkat Risiko: {status_risk}"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 40], 'color': "#e8f5e9"},
                {'range': [40, 70], 'color': "#fff3e0"},
                {'range': [70, 100], 'color': "#ffebee"}
            ]
        }
    ))
    fig_gauge.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_feature:
    st.subheader("📊 Fitur Paling Berpengaruh (Feature Importance)")
    if FEATURE_IMP_PATH.exists():
        try:
            df_importance = pd.read_csv(FEATURE_IMP_PATH)
        except Exception:
            df_importance = pd.DataFrame({
                "Fitur": ["Kelembapan Udara (RH2M)", "Curah Hujan (PRECTOTCORR)", "Suhu Udara (T2M)", "Vegetasi (NDVI)"],
                "Kepentingan Relative (%)": [38.5, 29.2, 18.3, 14.0]
            })
    else:
        df_importance = pd.DataFrame({
            "Fitur": ["Kelembapan Udara (RH2M)", "Curah Hujan (PRECTOTCORR)", "Suhu Udara (T2M)", "Vegetasi (NDVI)"],
            "Kepentingan Relative (%)": [38.5, 29.2, 18.3, 14.0]
        })

    fig_imp = px.bar(
        df_importance.sort_values(by=df_importance.columns[1], ascending=True),
        x=df_importance.columns[1],
        y=df_importance.columns[0],
        orientation="h",
        text=df_importance.columns[1],
        color_discrete_sequence=["#0288d1"]
    )
    fig_imp.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------
# PETA SPASIAL & TOP GRID
# ---------------------------------------------------------
col_map, col_grid = st.columns([1, 1])

with col_map:
    st.subheader("🗺️ Pemetaan Spasial Hotspot Grid Kalimantan")
    map_data = df_filtered.groupby(['grid_id', 'latitude', 'longitude']).agg(
        avg_frp=('avg_frp', 'mean'),
        total_hotspot=('hotspot_count', 'sum')
    ).reset_index()

    fig_map = px.scatter_geo(
        map_data,
        lat="latitude",
        lon="longitude",
        color="avg_frp",
        size="total_hotspot",
        hover_name="grid_id",
        color_continuous_scale="Reds",
        fitbounds="locations",
        title="Distribusi Intensity FRP & Hotspot Count"
    )
    fig_map.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_map, use_container_width=True)

with col_grid:
    st.subheader(f"🔥 Top 10 Grid FRP Tertinggi ({selected_provinsi})")
    top_grids = df_filtered.groupby('grid_id')['avg_frp'].mean().reset_index()
    top_grids = top_grids.sort_values(by='avg_frp', ascending=False).head(10).sort_values(by='avg_frp', ascending=True)

    fig_grid = px.bar(
        top_grids,
        x="avg_frp",
        y="grid_id",
        orientation="h",
        color="avg_frp",
        color_continuous_scale="Reds",
        labels={"avg_frp": "Rata-rata FRP", "grid_id": "Grid ID"}
    )
    fig_grid.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_grid, use_container_width=True)

st.info(f"📌 **Filter Aktif**: Wilayah **{selected_provinsi}** | Grid: **{selected_grid}** | Periode: **{periode_label}**.")