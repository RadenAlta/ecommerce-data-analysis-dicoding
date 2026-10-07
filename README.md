# E-Commerce Dashboard ✨

Proyek Analisis Data - **E-Commerce Public Dataset** (Brasil), periode Januari 2017 - Agustus 2018.

Dashboard interaktif (Streamlit) menampilkan:
- **Tren Penjualan** - jumlah pesanan dan revenue bulanan
- **Kategori Produk** - kategori teratas berdasarkan revenue dan jumlah terjual
- **Geospatial & Pengiriman** - peta sebaran pelanggan per state, lama pengiriman, dan skor review
- **Segmentasi RFM** - segmentasi pelanggan berdasarkan Recency, Frequency, Monetary

## Struktur Direktori
```
submission
├───dashboard
│   ├───main_data.csv
│   ├───peta_pelanggan.html
│   └───dashboard.py
├───data
│   └───(9 berkas CSV dataset E-Commerce)
├───notebook.ipynb
├───README.md
├───requirements.txt
└───url.txt
```

## Setup Environment - Anaconda
```
conda create --name main-ds python=3.12
conda activate main-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```
> Library pada `requirements.txt` membutuhkan **Python 3.11 atau lebih baru**.

## Run Streamlit App
```
cd dashboard
streamlit run dashboard.py
```
Dashboard akan terbuka di `http://localhost:8501`.

## Menjalankan Notebook
Buka `notebook.ipynb` dari direktori utama proyek (agar folder `data/` terbaca), lalu jalankan semua sel (*Run All*).
Sel terakhir bagian *Cleaning Data* akan menghasilkan ulang `dashboard/main_data.csv`.
