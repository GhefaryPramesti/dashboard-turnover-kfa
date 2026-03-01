import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import os

# ─────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dasbor Turnover Karyawan | PT Kimia Farma Apotek",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --primary: #0EA5E9;
    --primary-dark: #0284C7;
    --secondary: #10B981;
    --danger: #EF4444;
    --warning: #F59E0B;
    --bg: #0F172A;
    --bg-card: #1E293B;
    --bg-card2: #263347;
    --border: #334155;
    --text: #F1F5F9;
    --text-muted: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.main { background-color: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem !important; }

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1a2744 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* HEADER */
.dashboard-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 50%, #0F172A 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dashboard-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(14,165,233,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.header-title {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #0EA5E9, #38BDF8, #7DD3FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-subtitle {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 4px;
    font-weight: 400;
}

/* METRIC CARDS */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-2px); border-color: var(--primary); }
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 8px 0 4px;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-blue { color: #38BDF8; }
.metric-red { color: #F87171; }
.metric-green { color: #34D399; }
.metric-yellow { color: #FBBF24; }

/* SECTION CARD */
.section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* RISK BADGES */
.badge-tinggi {
    background: rgba(239,68,68,0.15);
    color: #F87171;
    border: 1px solid rgba(239,68,68,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-sedang {
    background: rgba(245,158,11,0.15);
    color: #FBBF24;
    border: 1px solid rgba(245,158,11,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-rendah {
    background: rgba(16,185,129,0.15);
    color: #34D399;
    border: 1px solid rgba(16,185,129,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* NOTIF CARDS */
.notif-card {
    background: var(--bg-card2);
    border-left: 4px solid;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.notif-critical { border-color: #EF4444; }
.notif-high { border-color: #F59E0B; }
.notif-medium { border-color: #0EA5E9; }
.notif-low { border-color: #10B981; }

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* FILE UPLOADER */
.stFileUploader {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* DATAFRAME */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* DIVIDER */
hr { border-color: var(--border) !important; }

/* SELECTBOX & INPUT */
.stSelectbox > div, .stMultiSelect > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
}

/* ALERT */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load('turnover_model.pkl')
        features = joblib.load('model_features.pkl')
        threshold = joblib.load('model_threshold.pkl')
        return model, features, threshold
    except:
        return None, None, 0.4

model, model_features, THRESHOLD = load_model()


# ─────────────────────────────────────────────
# FUNGSI UTILITAS
# ─────────────────────────────────────────────
def get_risk_level(prob):
    if prob >= 0.6:
        return 'Tinggi'
    elif prob >= 0.4:
        return 'Sedang'
    else:
        return 'Rendah'

def get_generation(usia):
    if usia >= 59:
        return 'Baby Boomer'
    elif usia >= 43:
        return 'Gen X'
    elif usia >= 27:
        return 'Milenial'
    else:
        return 'Gen Z'

def preprocess_uploaded(df):
    """Preprocessing data yang diupload untuk prediksi"""
    df = df.copy()
    
    # Standardisasi nama kolom
    df.columns = df.columns.str.strip().str.upper()
    
    required_cols = ['USIA_CLEAN', 'RINCIAN MASA KERJA NEW']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        # Coba nama alternatif
        col_map = {
            'USIA': 'USIA_CLEAN',
            'UMUR': 'USIA_CLEAN',
            'MASA KERJA': 'RINCIAN MASA KERJA NEW',
            'LAMA KERJA': 'RINCIAN MASA KERJA NEW',
        }
        for old, new in col_map.items():
            if old in df.columns and new not in df.columns:
                df[new] = df[old]
    
    return df

def predict_data(df, model, features, threshold):
    """Jalankan prediksi pada dataframe"""
    try:
        # Feature engineering
        if 'RINCIAN MASA KERJA NEW' in df.columns and 'USIA_CLEAN' in df.columns:
            df['Masa_Kerja_Tahun'] = df['RINCIAN MASA KERJA NEW'] / 12
            df['Usia_Masuk'] = df['USIA_CLEAN'] - df['Masa_Kerja_Tahun']
        
        # One-hot encoding untuk kolom kategorikal
        df_encoded = pd.get_dummies(df)
        
        # Align dengan features model
        X = pd.DataFrame(columns=features)
        for col in features:
            if col in df_encoded.columns:
                X[col] = df_encoded[col].values
            else:
                X[col] = 0
        
        X = X.fillna(0)
        
        # Prediksi
        probs = model.predict_proba(X)[:, 1]
        
        df['PROBABILITY_KELUAR'] = probs
        df['PREDICTION'] = ['Akan Keluar' if p >= threshold else 'Akan Bertahan' for p in probs]
        df['RISK_LEVEL'] = [get_risk_level(p) for p in probs]
        
        return df, None
    except Exception as e:
        return df, str(e)


def make_plotly_dark(fig):
    """Apply dark theme ke semua plotly figures"""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.5)',
        font=dict(family='Plus Jakarta Sans', color='#94A3B8'),
        legend=dict(
            bgcolor='rgba(30,41,59,0.8)',
            bordercolor='#334155',
            borderwidth=1
        ),
        margin=dict(t=40, b=20, l=10, r=10)
    )
    fig.update_xaxes(gridcolor='#1E293B', zerolinecolor='#334155')
    fig.update_yaxes(gridcolor='#1E293B', zerolinecolor='#334155')
    return fig


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0;'>
        <div style='font-size:2.5rem'>💊</div>
        <div style='font-weight:800; font-size:1rem; color:#38BDF8;'>PT Kimia Farma Apotek</div>
        <div style='font-size:0.75rem; color:#64748B; margin-top:2px;'>Sistem Prediksi Turnover</div>
    </div>
    <hr style='border-color:#1E293B; margin:8px 0 16px;'>
    """, unsafe_allow_html=True)
    
    st.markdown("**⚙️ Pengaturan Model**")
    threshold_display = st.slider(
        "Threshold Prediksi",
        min_value=0.1, max_value=0.9,
        value=float(THRESHOLD), step=0.05,
        help="Probabilitas minimum untuk diklasifikasikan sebagai 'Akan Keluar'"
    )
    
    st.markdown(f"""
    <div style='background:#1E293B; border:1px solid #334155; border-radius:8px; padding:12px; margin-top:8px; font-size:0.8rem;'>
        <div style='color:#94A3B8;'>Threshold aktif:</div>
        <div style='color:#38BDF8; font-weight:700; font-size:1.1rem;'>{threshold_display}</div>
        <div style='color:#64748B; margin-top:4px;'>Karyawan dengan probabilitas ≥ {threshold_display} diprediksi akan keluar</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color:#1E293B; margin:16px 0;'>", unsafe_allow_html=True)
    
    st.markdown("**📋 Status Model**")
    if model is not None:
        st.success("✅ Model berhasil dimuat")
    else:
        st.error("❌ Model tidak ditemukan")
        st.caption("Pastikan file turnover_model.pkl, model_features.pkl, dan model_threshold.pkl ada di direktori yang sama.")
    
    st.markdown("<hr style='border-color:#1E293B; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.72rem; color:#475569; line-height:1.6;'>
        <b style='color:#64748B;'>Dibuat untuk:</b><br>
        Skripsi S1 — Sistem Informasi<br>
        PT Kimia Farma Apotek<br>
        <br>
        <b style='color:#64748B;'>Algoritma:</b> Random Forest<br>
        <b style='color:#64748B;'>Framework:</b> CRISP-DM
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <p class="header-title">🏥 Dasbor Prediksi Turnover Karyawan</p>
    <p class="header-subtitle">PT Kimia Farma Apotek — Analisis & Prediksi Risiko Turnover Berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊  Analisis Tren Turnover",
    "🤖  Prediksi & Segmentasi Risiko",
    "🔔  Notifikasi Kontrak PKWT"
])


# ══════════════════════════════════════════════
# TAB 1: ANALISIS TREN TURNOVER
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 📂 Unggah Data Historis")
    
    uploaded_hist = st.file_uploader(
        "Upload file data karyawan historis (.csv atau .xlsx)",
        type=['csv', 'xlsx'],
        key="hist_upload",
        help="Data historis digunakan untuk analisis tren. Kolom yang dibutuhkan: USIA_CLEAN, STATUS, TGL_MASUK, UNIT BISNIS, JABATAN, dll."
    )
    
    # Template download
    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        template_hist = pd.DataFrame({
            'NAMA': ['KFA-10001', 'KFA-10002'],
            'USIA_CLEAN': [28, 35],
            'JENIS_KELAMIN': ['L', 'P'],
            'STATUS_PEGAWAI': ['PKWTT', 'PKWT'],
            'UNIT_BISNIS': ['JAYA 1', 'SURABAYA'],
            'JABATAN': ['Apoteker', 'Asisten Apoteker'],
            'RINCIAN_MASA_KERJA_NEW': [24, 12],
            'TGL_MASUK': ['2022-01-01', '2023-06-01'],
            'STATUS': ['Aktif', 'Resign'],
            'TGL_KELUAR': ['', '2024-06-01']
        })
        csv_template = template_hist.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Template CSV", csv_template, "template_data_historis.csv", "text/csv")
    
    if uploaded_hist is not None:
        try:
            if uploaded_hist.name.endswith('.xlsx'):
                df_hist = pd.read_excel(uploaded_hist)
            else:
                df_hist = pd.read_csv(uploaded_hist)
            
            df_hist.columns = df_hist.columns.str.strip().str.upper().str.replace(' ', '_')
            
            st.success(f"✅ Data berhasil dimuat: **{len(df_hist):,} baris**, **{df_hist.shape[1]} kolom**")
            
            # Deteksi kolom resign
            status_col = None
            for c in df_hist.columns:
                if 'STATUS' in c:
                    status_col = c
                    break
            
            if status_col:
                df_hist['IS_RESIGN'] = df_hist[status_col].astype(str).str.upper().str.contains('RESIGN|KELUAR|TERMINASI')
                
                # ── METRIK UTAMA ──
                st.markdown("---")
                st.markdown("#### 📌 Ringkasan Data Historis")
                
                total = len(df_hist)
                total_resign = df_hist['IS_RESIGN'].sum()
                total_aktif = total - total_resign
                turnover_rate = (total_resign / total * 100) if total > 0 else 0
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Karyawan</div>
                        <div class="metric-value metric-blue">{total:,}</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Karyawan Aktif</div>
                        <div class="metric-value metric-green">{total_aktif:,}</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Resign</div>
                        <div class="metric-value metric-red">{total_resign:,}</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Turnover Rate</div>
                        <div class="metric-value metric-yellow">{turnover_rate:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # ── ANALISIS PER UNIT BISNIS ──
                unit_col = next((c for c in df_hist.columns if 'UNIT' in c or 'BISNIS' in c), None)
                jabatan_col = next((c for c in df_hist.columns if 'JABATAN' in c or 'POSISI' in c), None)
                usia_col = next((c for c in df_hist.columns if 'USIA' in c or 'UMUR' in c), None)
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    if unit_col:
                        df_unit = df_hist.groupby(unit_col).agg(
                            Total=('IS_RESIGN', 'count'),
                            Resign=('IS_RESIGN', 'sum')
                        ).reset_index()
                        df_unit['Rate'] = (df_unit['Resign'] / df_unit['Total'] * 100).round(1)
                        df_unit = df_unit.sort_values('Resign', ascending=True).tail(15)
                        
                        fig_unit = px.bar(
                            df_unit, x='Resign', y=unit_col,
                            orientation='h',
                            title='🏢 Top 15 Unit Bisnis — Jumlah Resign',
                            color='Rate',
                            color_continuous_scale=['#1E40AF', '#0EA5E9', '#F59E0B', '#EF4444'],
                            labels={'Resign': 'Jumlah Resign', unit_col: 'Unit Bisnis', 'Rate': 'Rate (%)'}
                        )
                        fig_unit = make_plotly_dark(fig_unit)
                        fig_unit.update_layout(height=450, coloraxis_showscale=False)
                        st.plotly_chart(fig_unit, use_container_width=True)
                
                with col_right:
                    if jabatan_col:
                        df_jab = df_hist.groupby(jabatan_col).agg(
                            Total=('IS_RESIGN', 'count'),
                            Resign=('IS_RESIGN', 'sum')
                        ).reset_index()
                        df_jab['Rate'] = (df_jab['Resign'] / df_jab['Total'] * 100).round(1)
                        df_jab = df_jab.sort_values('Resign', ascending=True).tail(15)
                        
                        fig_jab = px.bar(
                            df_jab, x='Resign', y=jabatan_col,
                            orientation='h',
                            title='💼 Top 15 Jabatan — Jumlah Resign',
                            color='Rate',
                            color_continuous_scale=['#1E40AF', '#0EA5E9', '#10B981', '#EF4444'],
                            labels={'Resign': 'Jumlah Resign', jabatan_col: 'Jabatan', 'Rate': 'Rate (%)'}
                        )
                        fig_jab = make_plotly_dark(fig_jab)
                        fig_jab.update_layout(height=450, coloraxis_showscale=False)
                        st.plotly_chart(fig_jab, use_container_width=True)
                
                # ── ANALISIS GENERASI ──
                if usia_col:
                    df_hist['Generasi'] = pd.to_numeric(df_hist[usia_col], errors='coerce').apply(
                        lambda x: get_generation(x) if pd.notna(x) else 'Unknown'
                    )
                    df_gen = df_hist.groupby('Generasi').agg(
                        Total=('IS_RESIGN', 'count'),
                        Resign=('IS_RESIGN', 'sum')
                    ).reset_index()
                    df_gen['Rate'] = (df_gen['Resign'] / df_gen['Total'] * 100).round(1)
                    
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_gen = px.bar(
                            df_gen, x='Generasi', y='Resign',
                            title='👥 Resign per Generasi',
                            color='Generasi',
                            color_discrete_sequence=['#0EA5E9', '#10B981', '#F59E0B', '#EF4444'],
                            text='Resign'
                        )
                        fig_gen = make_plotly_dark(fig_gen)
                        fig_gen.update_traces(textposition='outside')
                        st.plotly_chart(fig_gen, use_container_width=True)
                    
                    with col_g2:
                        fig_rate = px.bar(
                            df_gen, x='Generasi', y='Rate',
                            title='📈 Turnover Rate per Generasi (%)',
                            color='Rate',
                            color_continuous_scale=['#10B981', '#F59E0B', '#EF4444'],
                            text='Rate'
                        )
                        fig_rate = make_plotly_dark(fig_rate)
                        fig_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                        fig_rate.update_layout(coloraxis_showscale=False)
                        st.plotly_chart(fig_rate, use_container_width=True)
                
                # ── TREN WAKTU ──
                tgl_col = next((c for c in df_hist.columns if 'TGL_KELUAR' in c or 'TGL_RESIGN' in c), None)
                if tgl_col:
                    df_resign_only = df_hist[df_hist['IS_RESIGN']].copy()
                    df_resign_only[tgl_col] = pd.to_datetime(df_resign_only[tgl_col], errors='coerce')
                    df_resign_only = df_resign_only.dropna(subset=[tgl_col])
                    
                    if len(df_resign_only) > 0:
                        df_resign_only['Bulan'] = df_resign_only[tgl_col].dt.to_period('M').astype(str)
                        df_tren = df_resign_only.groupby('Bulan').size().reset_index(name='Jumlah Resign')
                        
                        fig_tren = px.line(
                            df_tren, x='Bulan', y='Jumlah Resign',
                            title='📅 Tren Resign per Bulan',
                            markers=True,
                            line_shape='spline'
                        )
                        fig_tren.update_traces(
                            line_color='#0EA5E9',
                            marker=dict(color='#38BDF8', size=8),
                            fill='tozeroy',
                            fillcolor='rgba(14,165,233,0.1)'
                        )
                        fig_tren = make_plotly_dark(fig_tren)
                        fig_tren.update_layout(height=300)
                        st.plotly_chart(fig_tren, use_container_width=True)
            else:
                st.warning("⚠️ Kolom STATUS tidak ditemukan. Pastikan data mengandung kolom status karyawan.")
        
        except Exception as e:
            st.error(f"❌ Gagal memuat file: {e}")
    
    else:
        st.info("👆 Upload data historis karyawan untuk melihat analisis tren turnover.")


# ══════════════════════════════════════════════
# TAB 2: PREDIKSI & SEGMENTASI RISIKO
# ══════════════════════════════════════════════
with tab2:
    st.markdown("### 🤖 Prediksi Turnover Karyawan")
    
    if model is None:
        st.error("❌ Model tidak tersedia. Pastikan file **turnover_model.pkl**, **model_features.pkl**, dan **model_threshold.pkl** ada di direktori yang sama dengan app.py.")
        st.stop()
    
    uploaded_pred = st.file_uploader(
        "Upload data karyawan untuk diprediksi (.csv atau .xlsx)",
        type=['csv', 'xlsx'],
        key="pred_upload",
        help="Upload data karyawan aktif untuk mendapatkan prediksi risiko turnover."
    )
    
    # Template prediksi
    col_tdl, _ = st.columns([1, 4])
    with col_tdl:
        template_pred = pd.DataFrame({
            'NAMA': ['KFA-10001', 'KFA-10002', 'KFA-10003'],
            'USIA_CLEAN': [25, 32, 45],
            'RINCIAN MASA KERJA NEW': [6, 36, 120],
            'STATUS PEGAWAI': ['PKWTT', 'PKWTT', 'PKWTT'],
            'UNIT BISNIS': ['JAYA 1', 'SURABAYA', 'BEKASI'],
            'LOKASI KERJA GROUP': ['KF.0001', 'KF.0050', 'KF.0070'],
            'TINGKATAN JABATAN NEW': [3, 5, 7],
        })
        csv_pred = template_pred.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Template Prediksi", csv_pred, "template_prediksi.csv", "text/csv")
    
    if uploaded_pred is not None:
        try:
            if uploaded_pred.name.endswith('.xlsx'):
                df_pred = pd.read_excel(uploaded_pred)
            else:
                df_pred = pd.read_csv(uploaded_pred)
            
            st.success(f"✅ Data dimuat: **{len(df_pred):,} karyawan**")
            
            with st.spinner("⏳ Memproses prediksi..."):
                df_result, error = predict_data(df_pred.copy(), model, model_features, threshold_display)
            
            if error:
                st.warning(f"⚠️ Prediksi selesai dengan peringatan: {error}")
            
            # ── RINGKASAN HASIL ──
            st.markdown("---")
            st.markdown("#### 📌 Ringkasan Hasil Prediksi")
            
            total_pred = len(df_result)
            pred_keluar = (df_result['PREDICTION'] == 'Akan Keluar').sum()
            pred_bertahan = total_pred - pred_keluar
            risk_tinggi = (df_result['RISK_LEVEL'] == 'Tinggi').sum()
            risk_sedang = (df_result['RISK_LEVEL'] == 'Sedang').sum()
            pct_keluar = pred_keluar / total_pred * 100 if total_pred > 0 else 0
            
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Total Diproses</div><div class="metric-value metric-blue">{total_pred:,}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Diprediksi Keluar</div><div class="metric-value metric-red">{pred_keluar:,}</div><div class="metric-label">{pct_keluar:.1f}%</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Diprediksi Bertahan</div><div class="metric-value metric-green">{pred_bertahan:,}</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Risiko Tinggi</div><div class="metric-value metric-red">{risk_tinggi:,}</div></div>', unsafe_allow_html=True)
            with c5:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Risiko Sedang</div><div class="metric-value metric-yellow">{risk_sedang:,}</div></div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # ── VISUALISASI ──
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                dist_data = df_result['PREDICTION'].value_counts()
                fig_pie1 = go.Figure(go.Pie(
                    labels=dist_data.index,
                    values=dist_data.values,
                    hole=0.5,
                    marker_colors=['#EF4444', '#10B981'],
                    textinfo='label+percent',
                    textfont_size=13
                ))
                fig_pie1.update_layout(
                    title='Distribusi Prediksi Turnover',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8'),
                    showlegend=True,
                    legend=dict(bgcolor='rgba(30,41,59,0.8)', bordercolor='#334155', borderwidth=1),
                    height=320,
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig_pie1, use_container_width=True)
            
            with col_v2:
                risk_data = df_result['RISK_LEVEL'].value_counts()
                color_map = {'Tinggi': '#EF4444', 'Sedang': '#F59E0B', 'Rendah': '#10B981'}
                colors = [color_map.get(l, '#94A3B8') for l in risk_data.index]
                
                fig_pie2 = go.Figure(go.Pie(
                    labels=risk_data.index,
                    values=risk_data.values,
                    hole=0.5,
                    marker_colors=colors,
                    textinfo='label+percent',
                    textfont_size=13
                ))
                fig_pie2.update_layout(
                    title='Distribusi Tingkat Risiko',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8'),
                    showlegend=True,
                    legend=dict(bgcolor='rgba(30,41,59,0.8)', bordercolor='#334155', borderwidth=1),
                    height=320,
                    margin=dict(t=50, b=20)
                )
                st.plotly_chart(fig_pie2, use_container_width=True)
            
            # ── FEATURE IMPORTANCE ──
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### 🔍 Faktor Pendorong Turnover (Feature Importance)")
                
                fi = pd.DataFrame({
                    'Fitur': model_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True).tail(15)
                
                fig_fi = px.bar(
                    fi, x='Importance', y='Fitur',
                    orientation='h',
                    title='Top 15 Fitur Paling Berpengaruh',
                    color='Importance',
                    color_continuous_scale=['#1E3A5F', '#0EA5E9', '#38BDF8', '#FBBF24']
                )
                fig_fi = make_plotly_dark(fig_fi)
                fig_fi.update_layout(height=450, coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)
            
            # ── TABEL HASIL ──
            st.markdown("#### 👥 Daftar Karyawan & Hasil Prediksi")
            
            # Filter
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_pred = st.selectbox("Filter Prediksi", ["Semua", "Akan Keluar", "Akan Bertahan"])
            with col_f2:
                filter_risk = st.selectbox("Filter Risiko", ["Semua", "Tinggi", "Sedang", "Rendah"])
            
            df_display = df_result.copy()
            if filter_pred != "Semua":
                df_display = df_display[df_display['PREDICTION'] == filter_pred]
            if filter_risk != "Semua":
                df_display = df_display[df_display['RISK_LEVEL'] == filter_risk]
            
            # Kolom yang ditampilkan
            show_cols = ['PREDICTION', 'RISK_LEVEL', 'PROBABILITY_KELUAR']
            nama_col = next((c for c in df_result.columns if 'NAMA' in c.upper() or 'ID' in c.upper()), None)
            usia_col2 = next((c for c in df_result.columns if 'USIA' in c.upper() or 'UMUR' in c.upper()), None)
            
            if nama_col: show_cols = [nama_col] + show_cols
            if usia_col2 and usia_col2 not in show_cols: show_cols.insert(1, usia_col2)
            
            show_cols = [c for c in show_cols if c in df_display.columns]
            df_display_final = df_display[show_cols].copy()
            
            if 'PROBABILITY_KELUAR' in df_display_final.columns:
                df_display_final['PROBABILITY_KELUAR'] = df_display_final['PROBABILITY_KELUAR'].round(4)
            
            st.dataframe(df_display_final, use_container_width=True, height=400)
            
            # Download
            csv_result = df_display_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil Prediksi (CSV)",
                csv_result,
                "hasil_prediksi_turnover.csv",
                "text/csv"
            )
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
            st.exception(e)
    
    else:
        st.info("👆 Upload data karyawan untuk mendapatkan prediksi risiko turnover.")


# ══════════════════════════════════════════════
# TAB 3: NOTIFIKASI KONTRAK PKWT
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🔔 Notifikasi Kontrak PKWT Akan Berakhir")
    
    uploaded_pkwt = st.file_uploader(
        "Upload data karyawan PKWT (.csv atau .xlsx)",
        type=['csv', 'xlsx'],
        key="pkwt_upload",
        help="Upload data karyawan PKWT untuk melihat notifikasi kontrak yang akan berakhir."
    )
    
    # Template PKWT
    col_pt, _ = st.columns([1, 4])
    with col_pt:
        template_pkwt = pd.DataFrame({
            'NAMA': ['KFA-10001', 'KFA-10002', 'KFA-10003'],
            'UNIT_BISNIS': ['JAYA 1', 'SURABAYA', 'BEKASI'],
            'JABATAN': ['Apoteker', 'Asisten Apoteker', 'Kasir'],
            'LOKASI_KERJA': ['Jakarta', 'Surabaya', 'Bekasi'],
            'TGL_AKHIR_KONTRAK': [
                (datetime.today() + timedelta(days=3)).strftime('%Y-%m-%d'),
                (datetime.today() + timedelta(days=15)).strftime('%Y-%m-%d'),
                (datetime.today() + timedelta(days=25)).strftime('%Y-%m-%d'),
            ]
        })
        csv_pkwt = template_pkwt.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Unduh Template PKWT", csv_pkwt, "template_pkwt.csv", "text/csv")
    
    if uploaded_pkwt is not None:
        try:
            if uploaded_pkwt.name.endswith('.xlsx'):
                df_pkwt = pd.read_excel(uploaded_pkwt)
            else:
                df_pkwt = pd.read_csv(uploaded_pkwt)
            
            df_pkwt.columns = df_pkwt.columns.str.strip().str.upper().str.replace(' ', '_')
            
            # Cari kolom tanggal akhir kontrak
            tgl_akhir_col = next(
                (c for c in df_pkwt.columns if 'AKHIR' in c or 'KONTRAK' in c or 'BERAKHIR' in c or 'TGL_KELUAR' in c),
                None
            )
            
            if tgl_akhir_col is None:
                st.error("❌ Kolom tanggal akhir kontrak tidak ditemukan. Pastikan ada kolom seperti: TGL_AKHIR_KONTRAK, TGL_KELUAR, atau sejenisnya.")
            else:
                df_pkwt[tgl_akhir_col] = pd.to_datetime(df_pkwt[tgl_akhir_col], errors='coerce')
                df_pkwt = df_pkwt.dropna(subset=[tgl_akhir_col])
                
                today = pd.Timestamp.today().normalize()
                df_pkwt['SISA_HARI'] = (df_pkwt[tgl_akhir_col] - today).dt.days
                
                # Hanya yang akan berakhir dalam 30 hari ke depan
                df_notif = df_pkwt[(df_pkwt['SISA_HARI'] >= 0) & (df_pkwt['SISA_HARI'] <= 30)].copy()
                df_notif = df_notif.sort_values('SISA_HARI')
                
                # Kategorisasi urgensi
                def get_urgency(days):
                    if days <= 3:
                        return ('🔴 KRITIS', 'notif-critical', '≤ 3 hari')
                    elif days <= 7:
                        return ('🟠 TINGGI', 'notif-high', '≤ 1 minggu')
                    elif days <= 14:
                        return ('🔵 SEDANG', 'notif-medium', '≤ 2 minggu')
                    else:
                        return ('🟢 RENDAH', 'notif-low', '≤ 1 bulan')
                
                df_notif['URGENSI'] = df_notif['SISA_HARI'].apply(lambda x: get_urgency(x)[0])
                df_notif['CSS_CLASS'] = df_notif['SISA_HARI'].apply(lambda x: get_urgency(x)[1])
                
                # ── METRIK NOTIFIKASI ──
                c1, c2, c3, c4 = st.columns(4)
                kritis = (df_notif['SISA_HARI'] <= 3).sum()
                tinggi = ((df_notif['SISA_HARI'] > 3) & (df_notif['SISA_HARI'] <= 7)).sum()
                sedang = ((df_notif['SISA_HARI'] > 7) & (df_notif['SISA_HARI'] <= 14)).sum()
                rendah = (df_notif['SISA_HARI'] > 14).sum()
                
                with c1:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">🔴 Kritis (≤3 hari)</div><div class="metric-value metric-red">{kritis}</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">🟠 Tinggi (≤1 minggu)</div><div class="metric-value metric-yellow">{tinggi}</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">🔵 Sedang (≤2 minggu)</div><div class="metric-value metric-blue">{sedang}</div></div>', unsafe_allow_html=True)
                with c4:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">🟢 Rendah (≤1 bulan)</div><div class="metric-value metric-green">{rendah}</div></div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if len(df_notif) == 0:
                    st.success("✅ Tidak ada kontrak PKWT yang akan berakhir dalam 30 hari ke depan.")
                else:
                    st.markdown(f"#### ⚠️ {len(df_notif)} Karyawan PKWT Perlu Perhatian")
                    
                    # Filter urgensi
                    filter_urgensi = st.multiselect(
                        "Filter Urgensi",
                        options=['🔴 KRITIS', '🟠 TINGGI', '🔵 SEDANG', '🟢 RENDAH'],
                        default=['🔴 KRITIS', '🟠 TINGGI', '🔵 SEDANG', '🟢 RENDAH']
                    )
                    df_notif_filtered = df_notif[df_notif['URGENSI'].isin(filter_urgensi)]
                    
                    # Tampilkan notifikasi cards
                    nama_col_p = next((c for c in df_notif.columns if 'NAMA' in c or 'ID' in c), None)
                    unit_col_p = next((c for c in df_notif.columns if 'UNIT' in c or 'BISNIS' in c), None)
                    jabatan_col_p = next((c for c in df_notif.columns if 'JABATAN' in c), None)
                    lokasi_col_p = next((c for c in df_notif.columns if 'LOKASI' in c), None)
                    
                    for _, row in df_notif_filtered.iterrows():
                        nama = row.get(nama_col_p, 'N/A') if nama_col_p else 'N/A'
                        unit = row.get(unit_col_p, 'N/A') if unit_col_p else 'N/A'
                        jabatan = row.get(jabatan_col_p, 'N/A') if jabatan_col_p else 'N/A'
                        lokasi = row.get(lokasi_col_p, 'N/A') if lokasi_col_p else 'N/A'
                        sisa = int(row['SISA_HARI'])
                        tgl_akhir = row[tgl_akhir_col].strftime('%d %B %Y')
                        css = row['CSS_CLASS']
                        urgensi = row['URGENSI']
                        
                        sisa_text = f"{sisa} hari lagi" if sisa > 0 else "Hari ini!"
                        
                        st.markdown(f"""
                        <div class="notif-card {css}">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-weight:700; color:#F1F5F9; font-size:0.95rem;">{nama}</span>
                                    <span style="color:#64748B; margin:0 8px;">·</span>
                                    <span style="color:#94A3B8; font-size:0.85rem;">{jabatan}</span>
                                </div>
                                <span style="font-size:0.8rem; font-weight:600;">{urgensi}</span>
                            </div>
                            <div style="margin-top:6px; font-size:0.82rem; color:#94A3B8;">
                                🏢 {unit} &nbsp;·&nbsp; 📍 {lokasi} &nbsp;·&nbsp; 📅 Berakhir: <b style="color:#CBD5E1;">{tgl_akhir}</b> &nbsp;·&nbsp; ⏳ <b style="color:#FBBF24;">{sisa_text}</b>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Download notifikasi
                    st.markdown("<br>", unsafe_allow_html=True)
                    cols_show = [c for c in [nama_col_p, unit_col_p, jabatan_col_p, lokasi_col_p, tgl_akhir_col, 'SISA_HARI', 'URGENSI'] if c]
                    csv_notif = df_notif_filtered[cols_show].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Daftar Notifikasi (CSV)",
                        csv_notif,
                        "notifikasi_kontrak_pkwt.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {e}")
            st.exception(e)
    
    else:
        st.info("👆 Upload data karyawan PKWT untuk melihat notifikasi kontrak yang akan berakhir.")
        
        # Preview tampilan notifikasi
        st.markdown("#### 📋 Preview Tampilan Notifikasi")
        today_preview = datetime.today()
        
        contoh_data = [
            ("KFA-10001", "Apoteker", "Unit Jaya 1", "Jakarta", 2, "notif-critical", "🔴 KRITIS"),
            ("KFA-10002", "Asisten Apoteker", "Unit Surabaya", "Surabaya", 5, "notif-high", "🟠 TINGGI"),
            ("KFA-10003", "Kasir", "Unit Bekasi", "Bekasi", 12, "notif-medium", "🔵 SEDANG"),
            ("KFA-10004", "Staff Admin", "Unit Bandung", "Bandung", 25, "notif-low", "🟢 RENDAH"),
        ]
        
        for nama, jabatan, unit, lokasi, sisa, css, urgensi in contoh_data:
            tgl = (today_preview + timedelta(days=sisa)).strftime('%d %B %Y')
            st.markdown(f"""
            <div class="notif-card {css}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-weight:700; color:#F1F5F9; font-size:0.95rem;">{nama}</span>
                        <span style="color:#64748B; margin:0 8px;">·</span>
                        <span style="color:#94A3B8; font-size:0.85rem;">{jabatan}</span>
                    </div>
                    <span style="font-size:0.8rem; font-weight:600;">{urgensi}</span>
                </div>
                <div style="margin-top:6px; font-size:0.82rem; color:#94A3B8;">
                    🏢 {unit} &nbsp;·&nbsp; 📍 {lokasi} &nbsp;·&nbsp; 📅 Berakhir: <b style="color:#CBD5E1;">{tgl}</b> &nbsp;·&nbsp; ⏳ <b style="color:#FBBF24;">{sisa} hari lagi</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
