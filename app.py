import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import datetime
from datetime import timedelta

#page configuration
st.set_page_config(
    page_title="Dashboard Turnover Karyawan | PT Kimia Farma Apotek",
    layout="wide"
)

#dashboard title
st.title("📊Dashboard Turnover Karyawan PT Kimia Farma Apotek")
st.markdown("Analisis Prediksi Turnover dan Notifikasi Kontrak Karyawan PKWT")

#load model
@st.cache_resource
def load_model_resources():
    model = joblib.load('turnover_model.pkl')
    features = joblib.load('model_features.pkl')
    top_50_lokasi = joblib.load('top_50_lokasi.pkl')
    return model, features, top_50_lokasi

model, model_features, top_50_lokasi_ref = load_model_resources()

#load data
@st.cache_data
def load_data():
  df = pd.read_csv('data_cleaned.csv')

  if 'Unit Bisnis' in df.columns:
    df['Unit Bisnis'] = df['Unit Bisnis'].astype(str).str.upper().str.strip()

  date_cols = ['Tgl Mulai Bekerja(Dd/Mm/Yyyy)', 'Tgl Mulai Bekerja\n(Dd/Mm/Yyyy)', 'Tanggal Berakhir Kontrak', 'Tanggal Terminasi(Dd/Mm/Yyyy)', 'Tanggal Terminasi (Dd/Mm/Yyyy)']
  for col in date_cols:
    if col in df.columns:
      df[col] = pd.to_datetime(df[col], errors='coerce')
  return df

df = load_data()

def preprocess_uploaded_data(df):
  #1. membersihkan spasi di nama kolom
  df.columns = df.columns.str.strip()

  #2. buang kolom yg tdk perlu
  cols_to_drop = ['DAY', 'MONTH', 'YEAR', 'Tanggal Lahir\n(Dd/Mm/Yyyy)', 'Rincian Usia',
                    'Kategori Usia', 'Unit', 'COST CENTER INDUK', 'DAY.1', 'MONTH.1', 'YEAR.1',
                    'Masa Kerja New', 'Kategori Masa Kerja New', 'Masa Kerja (Y)', '>5 & >2',
                    'DAY.2', 'MONTH.2', 'YEAR.2', 'Date', 'Bulan', 'Tahun']
  df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

  #3. standarisasi unit bisnis
  if 'Unit Bisnis' in df.columns:
    df['Unit Bisnis'] = df['Unit Bisnis'].astype(str).str.upper().str.strip()

  #4. standarisasi status pegawai
  if 'Status Pegawai New' in df.columns:
    df['Status Pegawai New'] = df['Status Pegawai New'].astype(str).str.upper().str.strip()

  #5. standarisasi tingkatan jabatan
  if 'Tingkatan Jabatan' in df.columns:
    df['Tingkatan Jabatan'] = df['Tingkatan Jabatan'].astype(str).str.upper().str.strip()

  #6. buat kolom is_resign dari status terminasi
  if 'Status Terminasi' in df.columns:
    df['Status Terminasi Clean'] = df['Status Terminasi'].astype(str).str.upper().str.strip()
    keywords_resign = ['ATAS PERMINATAAN SENDIRI', 'ATAS PERMINTAAN SENDIRI', 'MENGUNDURKAN DIRI', 'NON AKTIF', 'PENGUNDURAN DIRI','PERMINTAAN SENDIRI',
                       'RESIGN', 'RESIGN / APOTEK TUTUP', 'RESIGN / MENUNGGU SK PHK KFA', 'RESIGN ATAS PERMINTAAN SENDIRI', 'TIDAK AKTIF']
    keywords_hapus = ['CUTI DILUAR TANGGUNGAN PERUSAHAAN', 'HABIS KONTRAK / APOTEK TUTUP', 'KHL', 'MENINGGAL DUNIA', 'MENINGGAL DUNIA',
                      'MUTASI ANTAR ENTITAS', 'PELANGGARAN', 'PENSIUN', 'PENSIUN DEFINITIF', 'PHK', 'RESTRUKTURISASI', 'WAFAT']
    def label_status(status):
      if any(k in status for k in keywords_hapus):
        return 'DELETE'
      if any(k in status for k in keywords_resign):
        return 1
      if 'AKTIF' in status:
        return 0
      if 'HABIS KONTRAK' in status:
        return 0
      return 'DELETE'
    df['Is_Resign'] = df['Status Terminasi Clean'].apply(label_status)
    df = df[df['Is_Resign'] != 'DELETE'].copy()
    df['Is_Resign'] = df['Is_Resign'].astype(int)

    #7. buat kolom usia_clean dari kolom usia
    if 'USIA' in df.columns:
      def clean_age(x):
        if isinstance(x, str):
           try:
            return int(x.split(' ')[0])
           except:
            return np.nan
        elif isinstance(x, (int, float)):
           return x
        return np.nan
      df['USIA_CLEAN'] = df['USIA'].apply(clean_age)
      df['USIA_CLEAN'] = pd.to_numeric(df['USIA_CLEAN'], errors='coerce')

    #8. konversi kolom rincian masa kerja new ke numerik
    if 'Rincian Masa Kerja New' in df.columns:
      df['Rincian Masa Kerja New'] = df['Rincian Masa Kerja New'].astype(str).str.upper().str.strip()

    #9. buat kolom usia_masuk
    if 'USIA_CLEAN' in df.columns and 'Rincian Masa Kerja New' in df.columns:
      df['USIA_CLEAN'] = df['USIA_CLEAN'].astype(float)
      df['Rincian Masa Kerja New'] = df['Rincian Masa Kerja New'].astype(float)
      df['Usia_Masuk'] = df['USIA_CLEAN'] - (df['Rincian Masa Kerja New'] / 12)

    #10. buat tingkatan_jabatan_new
    if 'Tingkatan Jabatan' in df.columns:
      jabatan_map={'DIREKSI': 1, 'MANAGER': 2, 'ASISTEN MANAGER': 3, 'SUPERVISOR': 4, 'PELAKSANA': 5}
      df['Tingkatan_Jabatan_New'] = df['Tingkatan Jabatan'].map(jabatan_map)

    #11. buat lokasi kerja group pake top 50 dari data training
    if 'Lokasi Kerja' in df.columns:
      df['Lokasi Kerja Group'] = df['Lokasi Kerja'].apply(
          lambda x: x if x in top_50_lokasi_ref else 'OTHER LOCATION'
      )

    #12. deduplikasi
    kolom_lahir = [c for c in df.columns if 'Lahir' in c]
    kolom_masuk = [c for c in df.columns if 'Mulai Bekerja' in c]
    if kolom_lahir and kolom_masuk:
      for col in [kolom_lahir[0], kolom_masuk[0]]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
      df = df.drop_duplicates(subset=[kolom_lahir[0], kolom_masuk[0]], keep='last')

    #13. konversi kolom tanggal
    date_cols = ['Tgl Mulai Bekerja\n(Dd/Mm/Yyyy) New', 'Tanggal Berakhir Kontrak',
                 'Tanggal Terminasi (Dd/Mm/Yyyy)']
    for col in date_cols:
      if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    #14. drop baris yang fitur utama nya kosong
    fitur_utama = ['USIA_CLEAN', 'Usia_Masuk', 'Rincian Masa Kerja New']
    fitur_ada = [f for f in fitur_utama if f in df.columns]
    df = df.dropna(subset=fitur_ada)

    return df


#sidebar
st.sidebar.header("Filter Dashboard")

#filter status pegawai
status_filter = st.sidebar.radio(
    "Pilih Status Pegawai:",
    options=["PKWT (Karyawan Kontrak)", "PKWTT (Karyawan Tetap)"],
    index=0

)

st.sidebar.markdown("---")

#filter unit bisnis
st.sidebar.caption("Filter ini hanya berlaku untuk Tab 1 - Analisis Tren Historis")
semua_unit = sorted(df['Unit Bisnis'].dropna().unique().tolist())
pilih_semua_unit = st.sidebar.checkbox("Semua Unit Bisnis", value=True)
if pilih_semua_unit:
  unit_filter = semua_unit
  st.sidebar.multiselect("Pilih Unit Bisnis:", options=semua_unit, disabled=True)
else:
  unit_filter = st.sidebar.multiselect("Pilih Unit Bisnis:", options=semua_unit, default=[])

st.markdown("---")
uploaded_main = st.file_uploader(
    "Upload data karyawan terbaru (.csv atau .xlsx)",
    type=['csv', 'xlsx'],
    key="main_upload"
)

if uploaded_main is not None:
  if uploaded_main.name.endswith('.xlsx'):
    df = pd.read_excel(uploaded_main)
  else:
    df = pd.read_csv(uploaded_main)

  df = preprocess_uploaded_data(df)
  st.success(f"File Berhasil Diupload!: **{len(df):,} baris**")
else:
  st.info("Menampilkan data default dari data_cleaned.csv")

#update df_filtered setelah upload + filter status pegawai
df_filtered = df[df['Unit Bisnis'].isin(unit_filter)] if unit_filter else df.copy()
if 'Status Pegawai New' in df_filtered.columns:
  if "PKWTT" in status_filter:
    df_filtered = df_filtered[df_filtered['Status Pegawai New'].astype(str).str.upper().str.strip() == 'PKWTT']
  elif "PKWT" in status_filter:
    df_filtered = df_filtered[df_filtered['Status Pegawai New'].astype(str).str.upper().str.strip() == 'PKWT']

#tabs berdasarkan status
if "PKWTT" in status_filter:
  tab1, tab2 = st.tabs(["Analisis Tren Historis", "Prediksi dan Segmentasi Risiko"])
  tab3 = None
else:
  tab1, tab2, tab3 = st.tabs(["Analisis Tren Historis", "Prediksi dan Segmentasi Risiko", "Notifikasi Kontrak PKWT"])

with tab1:
  st.subheader("Analisis Tren Historis Resign 2023 - 2025")
  df_resign = df_filtered[df_filtered['Is_Resign'] == 1].copy()

  if df_resign.empty:
    st.warning("Tidak ada data resign yang ditampilkan")
  else:
    total_karyawan = len(df_filtered)
    total_resign = len(df_resign)
    total_aktif = len(df_filtered[df_filtered['Is_Resign'] == 0])
    turnover_rate = (total_resign / total_karyawan * 100, 1) if total_karyawan > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Karyawan", f"{total_karyawan:,}")
    col2.metric("Karyawan Aktif", f"{total_aktif:,}")
    col3.metric("Total Resign", f"{total_resign:,}")
    st.markdown("---")

    tgl_col = None
    for c in ['Tanggal Terminasi (Dd/Mm/Yyyy)', 'Tanggal Terminasi(Dd/Mm/Yyyy)']:
      if c in df_resign.columns:
        tgl_col = c
        break
    col_trend, col_gen = st.columns(2)

    #grafik tren resign jan 2023 - agus 2025
    with col_trend:
      if tgl_col:
        tahun = df_resign[tgl_col].dt.year
        df_resign_filtered = df_resign[(tahun >= 2021) & (tahun <= 2026)]

        if not df_resign_filtered.empty:
          df_resign_filtered['Bulan_Resign'] = df_resign_filtered[tgl_col].dt.to_period('M').astype(str)
          trend_resign = df_resign_filtered.groupby('Bulan_Resign').size().reset_index(name='Jumlah Karyawan')
          trend_resign = trend_resign.sort_values('Bulan_Resign')

          fig_trend = px.line(trend_resign, x='Bulan_Resign', y='Jumlah Karyawan', markers=True, title="Tren Karyawan Resign per Bulan", line_shape='spline')
          fig_trend.update_traces(line_color='#0EA5E9', marker=dict(size=7, color='#38BDF8'))
          st.plotly_chart(fig_trend, use_container_width=True, key="grafik_tren_resign")

    #grafik generasi
    with col_gen:
      if 'Generasi' in df_resign.columns:
        fig_gen = px.pie(df_resign, names='Generasi', title="Distribusi Resign per Generasi", color_discrete_sequence=px.colors.sequential.Blues_r)
        fig_gen.update_traces(textinfo='label+value')
        st.plotly_chart(fig_gen, use_container_width=True, key="grafik_gen")

    #grafik ub & jabatan turnover tertinggi
    col3, col4 = st.columns([0.5, 0.5])
    with col3:
      top_unit = df_resign['Unit Bisnis'].value_counts().nlargest(5).reset_index()
      top_unit.columns = ['Unit Bisnis', 'Jumlah Resign']
      fig_unit = px.bar(top_unit, x='Jumlah Resign', y='Unit Bisnis', orientation='h',
                        title="Top 5 Unit Bisnis - Jumlah Resign", color='Jumlah Resign', color_continuous_scale=['#9CD5FF', '#0AC4E0', '#0992C2', '#0B2D72'])
      fig_unit.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
      st.plotly_chart(fig_unit, use_container_width=True, key="grafik_unit")

    with col4:
      jab_col = 'Tingkatan Jabatan' if 'Tingkatan Jabatan' in df_resign.columns else None
      if jab_col:
        top_jab = df_resign[jab_col].value_counts().reset_index()
        top_jab.columns = ['Jabatan', 'Jumlah Resign']
        fig_jab = px.bar(top_jab, x='Jumlah Resign', y='Jabatan', orientation='h',
                         title="Turnover Per Jabatan", color='Jumlah Resign', color_continuous_scale=['#9CD5FF', '#0AC4E0', '#0992C2', '#0B2D72'])
        fig_jab.update_layout(yaxis={'categoryorder': 'total ascending'}, coloraxis_showscale=False)
        st.plotly_chart(fig_jab, use_container_width=True, key="grafik_jabatan")

    if 'Rincian Masa Kerja New' in df_filtered.columns:
      st.subheader("Distribusi Masa Kerja Saat Resign")
      df_resign_mk = df_resign.dropna(subset=['Rincian Masa Kerja New']).copy()
      df_resign_mk['Masa Kerja (Tahun)'] = (df_resign_mk['Rincian Masa Kerja New'] / 12).round(1)
      fig_mk = px.histogram(df_resign_mk, x='Masa Kerja (Tahun)', nbins=20, title="Histogram Masa Kerja Karyawan Resign", color_discrete_sequence=['#0EA5E9'])
      st.plotly_chart(fig_mk, use_container_width=True, key="grafik_mk")

#TAB2
with tab2:
  st.subheader("Prediksi Risiko Turnover Karyawan")
  st.write("Status filter:", status_filter)
  if "PKWTT" in status_filter:
    df_active = df[
        (df['Is_Resign'] == 0) &
        (df['Status Pegawai New'].astype(str).str.upper().str.strip() == 'PKWTT')
    ].copy()
  elif "PKWT" in status_filter:
    df_active = df[
        (df['Is_Resign'] == 0) &
        (df['Status Pegawai New'].astype(str).str.upper().str.strip() == 'PKWT')
    ].copy()
  else:
    df_active = df[df['Is_Resign'] == 0].copy()

  if df_active.empty:
    st.warning("Tidak ada data karyawan aktif yang ditampilkan")
    if "PKWT" in status_filter:
      st.info("ℹ️ Karyawan PKWT yang aktif tidak ditemukan. Gunakan Tab 3 untuk melihat notifikasi kontrak PKWT.")
  else:
    try:
      #onehot encoding unit bisnis
      df_dummy = pd.get_dummies(df_active, columns=['Unit Bisnis', 'Lokasi Kerja Group'], drop_first=True)

      #tambah kolom yang kurang
      for col in model_features:
        if col not in df_dummy.columns:
          df_dummy[col] = 0

      #threshold & prediksi
      threshold = 0.4

      def tentukan_risiko(prob):
        if prob >= 0.6:
          return 'Tinggi'
        elif prob >= threshold:
          return 'Sedang'
        else:
          return 'Rendah'

      X_pred = df_dummy[model_features].fillna(0)
      df_active['Probabilitas_Resign'] = model.predict_proba(X_pred)[:, 1]
      df_active['Prediksi'] = df_active['Probabilitas_Resign'].apply(
          lambda x: 'Akan Keluar' if x >= threshold else 'Akan Bertahan'
      )
      df_active['Tingkat Risiko'] = df_active['Probabilitas_Resign'].apply(tentukan_risiko)


      #metrikk
      total_pred = len(df_active)
      pred_keluar = (df_active['Prediksi'] == 'Akan Keluar').sum()
      pred_bertahan = total_pred - pred_keluar
      risk_tinggi= (df_active['Tingkat Risiko'] == 'Tinggi').sum()
      risk_rendah = (df_active['Tingkat Risiko'] == 'Rendah').sum()
      pct_keluar = round(pred_keluar / total_pred * 100, 1) if total_pred > 0 else 0

      col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
      col_m1.metric("Total Diproses", f"{total_pred:,}")
      col_m2.metric("Diprediksi Keluar", f"{pred_keluar:,}")
      col_m3.metric("Diprediksi Bertahan", f"{pred_bertahan:,}")
      col_m4.metric("Risiko Tinggi", f"{risk_tinggi:,}", delta_color="inverse")
      col_m5.metric("Risiko Rendah", f"{risk_rendah:,}", delta_color="inverse")
      st.markdown("---")

      #visualisasi
      col_v1, col_v2 = st.columns(2)
      with col_v1:
        dist = df_active['Prediksi'].value_counts().reset_index()
        dist.columns = ['Prediksi', 'Jumlah']
        fig_pie1 = px.pie(dist, names='Prediksi', values='Jumlah', title="Distribusi Prediksi Turnover",
                          color_discrete_map={'Akan Keluar': '#EF4444', 'Akan Bertahan': '#10B981'}, hole=0.4)
        st.plotly_chart(fig_pie1, use_container_width=True, key="pie_prediksi")

      with col_v2:
        risk_dist = df_active['Tingkat Risiko'].value_counts().reset_index()
        risk_dist.columns = ['Tingkat Risiko', 'count']
        fig_pie2 = px.pie(risk_dist, names='Tingkat Risiko', values='count', title="Distribusi Tingkat Risiko", color='Tingkat Risiko',
                          color_discrete_map={'Tinggi': '#EF4444', 'Sedang': '#FFD93D', 'Rendah': '#6BCB77'}, hole=0.4)
        st.plotly_chart(fig_pie2, use_container_width=True, key="pie_risiko")

      #high risk perunitbisnis dan jabatan
      df_high_risk = df_active[df_active['Tingkat Risiko'] == 'Tinggi']
      if not df_high_risk.empty:
        st.subheader("Karyawan Risiko Tinggi")
        col_hr1, col_hr2 = st.columns(2)
        with col_hr1:
          if 'Unit Bisnis' in df_high_risk.columns:
            hr_unit = df_high_risk['Unit Bisnis'].value_counts().nlargest(10).reset_index()
            hr_unit.columns = ['Unit Bisnis', 'Jumlah']
            fig_hr_unit = px.bar(hr_unit, x='Jumlah', y='Unit Bisnis', orientation='h', title="High Risk per Unit Bisnis", color_discrete_sequence=['#EF4444'])
            fig_hr_unit.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_hr_unit, use_container_width=True, key="hr_unit")

        with col_hr2:
          jab_col2 = 'Tingkatan Jabatan' if 'Tingkatan Jabatan' in df_high_risk.columns else None
          if jab_col2:
            hr_jab = df_high_risk[jab_col2].value_counts().reset_index()
            hr_jab.columns = ['Jabatan', 'Jumlah']
            fig_hr_jab = px.bar(hr_jab, x='Jumlah', y='Jabatan', orientation='h', title="High Risk per Jabatan", color_discrete_sequence=['#EF4444'])
            fig_hr_jab.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_hr_jab, use_container_width=True, key="hr_jab")

      #feature importances
      if hasattr(model, 'feature_importances_'):
        st.subheader("Faktor Pendorong Turnover")
        fi = pd.DataFrame({
            'Fitur': model_features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True).tail(4)

        fig_fi = px.bar(fi, x='Importance', y='Fitur', orientation='h',
                        title="Top Fitur Paling Berpengaruh",
                        color='Importance',
                        color_continuous_scale=['#9CD5FF', '#0AC4E0', '#0992C2', '#0B2D72'])
        fig_fi.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance")


      #tabel detail
      st.subheader("Detail Prediksi Karyawan")
      col_f1, col_f2 = st.columns(2)
      with col_f1:
        filter_risiko = st.selectbox("Filter Risiko", ["Semua", "Tinggi", "Sedang", "Rendah"])
      with col_f2:
        filter_prediksi = st.selectbox("Filter Prediksi", ["Semua", "Akan Keluar", "Akan Bertahan"])
      df_tampil = df_active.copy()

      if filter_risiko != "Semua":
        df_tampil = df_tampil[df_tampil['Tingkat Risiko'] == filter_risiko]
      if filter_prediksi != "Semua":
        df_tampil = df_tampil[df_tampil['Prediksi'] == filter_prediksi]

      cols_to_show = ['ID_Karyawan', 'Jabatan', 'Unit Bisnis', 'Lokasi Kerja', 'Prediksi', 'Tingkat Risiko', 'Probabilitas_Resign']
      cols_to_show = [c for c in cols_to_show if c in df_tampil.columns]
      df_tampil_final = df_tampil.sort_values('Probabilitas_Resign', ascending=False)[cols_to_show].copy()
      df_tampil_final['Probabilitas_Resign'] = (df_tampil_final['Probabilitas_Resign'] * 100).round(1).astype(str) + '%'
      st.dataframe(df_tampil_final, use_container_width=True, height=400)

    except Exception as e:
      st.error(f"Terjadi Kesalahan: {e}")
      st.exception(e)

#TAB3
if tab3 is not None:
  with tab3:
    st.subheader("Notifikasi Kontrak PKWT Akan Berakhir")
    df_pkwt = df[df['Status Pegawai New'].astype(str).str.upper().str.strip() == 'PKWT'].copy()

    if df_pkwt.empty:
      st.warning("Tidak ada data karyawan PKWT yang ditampilkan")
    else:
      #mencari kolom tanggal berakhir kontrak
      tgl_kontrak_col = None
      for c in df_pkwt.columns:
        if 'BERAKHIR' in c.upper() or 'KONTRAK' in c.upper():
          tgl_kontrak_col = c
          break
          if tgl_kontrak_col is None:
            st.error("Kolom Tanggal Berakhir Tidak Ditemukan")
          else:
            df_pkwt[tgl_kontrak_col] = pd.to_datetime(df_pkwt[tgl_kontrak_col], errors='coerce')
            df_pkwt = df_pkwt.dropna(subset=[tgl_kontrak_col])

      #hitung sisa hari
      today_ts = pd.Timestamp.today().normalize()
      df_pkwt['Sisa_Hari'] = (df_pkwt[tgl_kontrak_col] - today_ts).dt.days

      #filter yang akan berakhir 30 hari ke depan
      df_notif = df_pkwt[(df_pkwt['Sisa_Hari'] >= 0) & (df_pkwt['Sisa_Hari'] <= 30)].copy()
      df_notif = df_notif.sort_values('Sisa_Hari')

      #kategori urgensi
      def get_urgensi(days):
        if days <= 3:
          return 'Kritis'
        elif days <= 7:
          return 'Tinggi'
        elif days <= 14:
          return 'Sedang'
        else:
          return 'Rendah'
      df_notif['Urgensi'] = df_notif['Sisa_Hari'].apply(get_urgensi)

      #metrik
      kritis = (df_notif['Sisa_Hari'] <= 3).sum()
      tinggi = ((df_notif['Sisa_Hari'] > 3) & (df_notif['Sisa_Hari'] <= 7)).sum()
      sedang_n = ((df_notif['Sisa_Hari'] > 7) & (df_notif['Sisa_Hari'] <= 14)).sum()
      rendah_n  = (df_notif['Sisa_Hari'] > 14).sum()

      col_n1, col_n2, col_n3, col_n4 = st.columns(4)
      col_n1.metric("Kritis (<3 hari)", kritis)
      col_n2.metric("Tinggi (<1 minggu)", tinggi)
      col_n3.metric("Sedang (<2 minggu)", sedang_n)
      col_n4.metric("Rendah (<1 bulan)", rendah_n)

      st.markdown("---")

      if df_notif.empty:
        st.success("Tidak ada kontrak PKWT yang akan berakhir dalam 30 hari ke depan")
      else:
        st.subheader(f"{len(df_notif)} Karyawan PKWT Perlu Perhatian")

        #filter urgensi
        filter_urgensi = st.multiselect(
            "Filter Urgensi:",
            options=['Kritis', 'Tinggi', 'Sedang', 'Rendah'],
            default=['Kritis', 'Tinggi', 'Sedang', 'Rendah']
        )
        df_notif_show = df_notif[df_notif['Urgensi'].isin(filter_urgensi)]

        #kolom yang ingin ditampilkan
        id_col = next((c for c in df_notif.columns if 'ID' in c.upper()), None)
        unit_col_n = next((c for c in df_notif.columns if 'UNIT' in c.upper()), None)
        jab_col_n = 'Jabatan' if 'Jabatan' in df_notif.columns else None
        lok_col_n = next((c for c in df_notif.columns if 'LOKASI' in c.upper()), None)

        shows_cols = [c for c in [id_col, unit_col_n, jab_col_n, lok_col_n, tgl_kontrak_col, 'Sisa_Hari', 'Urgensi'] if c]
        st.dataframe(
            df_notif_show[shows_cols].rename(columns={tgl_kontrak_col: 'Tgl Berakhir'}),
            use_container_width=True,
            height=400
        )

        #grafik distribusi urgensi
        urgensi_dist = df_notif_show['Urgensi'].value_counts().reset_index()
        urgensi_dist.columns = ['Urgensi', 'Jumlah']
        fig_urgensi = px.bar(urgensi_dist, x='Urgensi', y='Jumlah', title="Distribusi Kontrak Berakhir per Kategori Urgensi",
                             color='Urgensi', color_discrete_map={'Kritis': '#EF4444', 'Tinggi': '#F59E0B', 'Sedang': '#0EA5E9', 'Rendah': '#10B981'})
        st.plotly_chart(fig_urgensi, use_container_width=True, key="grafik_urgensi")
