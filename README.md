# 🔥 Karhutla Kalimantan Analysis Dashboard

Dashboard interaktif berbasis Streamlit untuk pemetaan spasial, analisis historis, dan simulasi prediksi risiko Kebakaran Hutan dan Lahan (Karhutla) di Kalimantan menggunakan model Machine Learning **LightGBM**.

---

## 🚀 Cara Menjalankan Aplikasi Secara Lokal

### 1. Clone Repositori

```bash
git clone https://github.com/USERNAME_KAMU/NAMA_REPOSITORY.git
cd NAMA_REPOSITORY
```

### 2. Setup Virtual Environment

**Pilihan A: Menggunakan Anaconda**

```bash
conda create --name karhutla-dashboard python=3.9 -y
conda activate karhutla-dashboard
pip install -r requirements.txt
```

**Pilihan B: Menggunakan Shell / Terminal (`venv`)**

```bash
python -m venv env_karhutla

# Mengaktifkan environment di Windows:
env_karhutla\Scripts\activate

# Mengaktifkan environment di macOS/Linux:
source env_karhutla/bin/activate

pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Streamlit

```bash
streamlit run dashboard.py
```

---

## 📂 Struktur Direktori Proyek

```text
.
├── include/
│   └── dashboard_assets/
│       ├── df_final_dashboard.csv
│       ├── feature_importance.csv
│       ├── lgbm_karhutla_model.pkl
│       └── top_10_grid_frp.csv
├── dashboard.py
├── requirements.txt
└── README.md
```

---

## 📊 Fitur Utama Dashboard

* **Simulator Parameter Cuaca & Vegetasi**: Simulasi interaktif probabilitas kebakaran berdasarkan kelembapan udara (`RH2M`), curah hujan (`PRECTOTCORR`), suhu (`T2M`), dan kerapatan vegetasi (`NDVI`).
* **Filter Wilayah Spasial**: Mencakup 5 provinsi di Kalimantan (*Kalimantan Barat, Kalimantan Tengah, Kalimantan Selatan, Kalimantan Timur, dan Kalimantan Utara*) hingga ke tingkat ID Grid spesifik.
* **Filter Periode Waktu Fleksibel**: Analisis data historis dan prediksi tahun **2025** & **2026** berdasarkan Kuartal, Bulan, atau Rentang Tanggal spesifik.
* **Pemetaan Spasial & Top Grid**: Visualisasi geografis sebaran titik panas (*hotspot*) dan intensitas *Fire Radiative Power* (FRP).

---

## 👤 Identitas Pengembang

* **Nama**: Cinta Wardana
* **NIM**: E1E124059
