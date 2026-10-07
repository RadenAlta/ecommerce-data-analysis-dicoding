import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import streamlit as st
from babel.numbers import format_currency

from pathlib import Path
st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")
sns.set_theme(style="white")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
PRIMARY, GRAY, ACCENT = "#2A6F97", "#CBD2D9", "#E07A5F"


# ------------------------------------------------------------------ data
BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_data():
    data_path = BASE_DIR / "main_data.csv"
    df = pd.read_csv(data_path, parse_dates=["order_purchase_timestamp"])
    df["revenue"] = df["price"]
    return df.sort_values("order_purchase_timestamp").reset_index(drop=True)


def format_brl(x):
    return format_currency(x, "BRL", locale="pt_BR")


# ------------------------------------------------------------------ helper dataframe
def create_monthly_df(df):
    out = (df.groupby("order_month")
           .agg(order_count=("order_id", "nunique"), revenue=("revenue", "sum")).reset_index())
    out["revenue_mom_pct"] = out["revenue"].pct_change() * 100
    return out


def create_category_df(df):
    return (df.groupby("category")
            .agg(total_terjual=("order_item_id", "count"), revenue=("revenue", "sum"),
                 skor_review=("review_score", "mean")).reset_index())


def create_state_df(df):
    # Pelanggan & revenue: level item. Lama kirim, keterlambatan, review: level order (satu baris per order_id)
    item_part = (df.groupby("customer_state")
                 .agg(customer_count=("customer_unique_id", "nunique"), revenue=("revenue", "sum"),
                      lat=("customer_lat", "mean"), lng=("customer_lng", "mean")))
    order_part = (df.drop_duplicates("order_id").groupby("customer_state")
                  .agg(order_count=("order_id", "count"), avg_delivery_days=("delivery_days", "mean"),
                       late_pct=("is_late", "mean"), avg_review=("review_score", "mean")))
    out = item_part.join(order_part).reset_index()
    out["late_pct"] *= 100
    return out.sort_values("customer_count", ascending=False)


def create_bin_df(df):
    order_level = df.drop_duplicates("order_id").copy()
    labels = ["0-7 hari", "8-14 hari", "15-21 hari", "22-30 hari", ">30 hari"]
    order_level["delivery_bin"] = pd.cut(order_level["delivery_days"], [0, 7, 14, 21, 30, np.inf], labels=labels)
    out = (order_level.groupby("delivery_bin", observed=True)
           .agg(jumlah_pesanan=("order_id", "count"), skor_review=("review_score", "mean")).reset_index())
    out["persen_pesanan"] = out["jumlah_pesanan"] / out["jumlah_pesanan"].sum() * 100
    return out


def create_rfm_df(df):
    snapshot = df["order_purchase_timestamp"].max().normalize() + pd.Timedelta(days=1)
    rfm = (df.groupby("customer_unique_id")
           .agg(last_purchase=("order_purchase_timestamp", "max"), frequency=("order_id", "nunique"),
                monetary=("revenue", "sum")).reset_index())
    rfm["recency"] = (snapshot - rfm["last_purchase"].dt.normalize()).dt.days
    rfm["r_score"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["f_score"] = pd.cut(rfm["frequency"], [0, 1, 2, np.inf], labels=[1, 2, 3]).astype(int)
    kondisi = [
        rfm["f_score"] >= 2,
        (rfm["r_score"] >= 4) & (rfm["m_score"] >= 4),
        rfm["r_score"] >= 4,
        (rfm["r_score"] <= 2) & (rfm["m_score"] >= 4),
        rfm["r_score"] <= 2,
    ]
    nama = ["Repeat Customers", "Recent Big Spenders", "Recent Buyers", "At-Risk Big Spenders", "Hibernating"]
    rfm["segment"] = np.select(kondisi, nama, default="Need Attention")
    seg = (rfm.groupby("segment")
           .agg(jumlah_pelanggan=("customer_unique_id", "count"), recency_rata2=("recency", "mean"),
                monetary_rata2=("monetary", "mean"), total_monetary=("monetary", "sum")).reset_index())
    seg["persen_pelanggan"] = seg["jumlah_pelanggan"] / seg["jumlah_pelanggan"].sum() * 100
    seg["persen_revenue"] = seg["total_monetary"] / seg["total_monetary"].sum() * 100
    return rfm, seg.sort_values("persen_revenue", ascending=False)


# ------------------------------------------------------------------ load & filter
all_df = load_data()
min_date, max_date = all_df["order_purchase_timestamp"].min().date(), all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.title("🛒 E-Commerce Dashboard")
    st.caption("Proyek Analisis Data - E-Commerce Public Dataset (Brasil), Jan 2017 - Agu 2018")
    rentang = st.date_input("Rentang waktu", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    semua_state = sorted(all_df["customer_state"].dropna().unique())
    pilih_state = st.multiselect("State pelanggan", semua_state, default=[],
                                 help="Kosongkan untuk menampilkan semua state")

if isinstance(rentang, (tuple, list)):
    start_date, end_date = (rentang[0], rentang[-1]) if len(rentang) > 0 else (min_date, max_date)
else:
    start_date = end_date = rentang

main_df = all_df[(all_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) &
                 (all_df["order_purchase_timestamp"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))]
if pilih_state:
    main_df = main_df[main_df["customer_state"].isin(pilih_state)]

st.header("Dashboard Analisis E-Commerce")
if main_df.empty:
    st.warning("Tidak ada data untuk filter yang dipilih. Ubah rentang waktu atau state.")
    st.stop()

# ------------------------------------------------------------------ KPI
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total pesanan", f"{main_df['order_id'].nunique():,}")
k2.metric("Total revenue", format_brl(main_df["revenue"].sum()))
k3.metric("Pelanggan unik", f"{main_df['customer_unique_id'].nunique():,}")
order_df = main_df.drop_duplicates("order_id")   # review & lama kirim dihitung per pesanan
k4.metric("Rata-rata skor review", f"{order_df['review_score'].mean():.2f} / 5")
k5.metric("Rata-rata lama kirim", f"{order_df['delivery_days'].mean():.1f} hari")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Tren Penjualan", "🏷️ Kategori Produk", "🗺️ Geospatial & Pengiriman", "👥 Segmentasi RFM"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Tren pesanan dan revenue bulanan")
    monthly_df = create_monthly_df(main_df)
    puncak_order = monthly_df["order_count"].idxmax()
    puncak_revenue = monthly_df["revenue"].idxmax()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    warna = [ACCENT if i == puncak_order else PRIMARY for i in range(len(monthly_df))]
    ax[0].bar(monthly_df["order_month"], monthly_df["order_count"], color=warna)
    ax[0].set_title("Jumlah pesanan per bulan", loc="left"); ax[0].set_ylabel("Pesanan")
    ax[0].yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
    ax[1].plot(monthly_df["order_month"], monthly_df["revenue"] / 1e3, color=PRIMARY, marker="o", linewidth=2.2)
    ax[1].scatter(monthly_df.loc[puncak_revenue, "order_month"], monthly_df.loc[puncak_revenue, "revenue"] / 1e3, color=ACCENT, s=90, zorder=3)
    ax[1].set_title("Revenue per bulan (ribu BRL)", loc="left"); ax[1].set_ylabel("Ribu BRL"); ax[1].set_ylim(bottom=0)
    ax[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
    p1, p2 = st.columns(2)
    p1.info(f"Puncak **jumlah pesanan**: **{monthly_df.loc[puncak_order, 'order_month']}** "
            f"({monthly_df.loc[puncak_order, 'order_count']:,} pesanan)")
    p2.info(f"Puncak **revenue**: **{monthly_df.loc[puncak_revenue, 'order_month']}** "
            f"({format_brl(monthly_df.loc[puncak_revenue, 'revenue'])})")
    with st.expander("Lihat tabel data bulanan"):
        st.dataframe(monthly_df.rename(columns={"order_month": "Bulan", "order_count": "Pesanan", "revenue": "Revenue",
                                                "revenue_mom_pct": "Revenue MoM (%)"}).round(2), width="stretch")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Kategori produk teratas")
    n_top = st.slider("Jumlah kategori yang ditampilkan", 3, 10, 5)
    cat_df = create_category_df(main_df)
    total_rev = cat_df["revenue"].sum()
    top_rev = cat_df.nlargest(n_top, "revenue").sort_values("revenue")
    top_qty = cat_df.nlargest(n_top, "total_terjual").sort_values("total_terjual")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, data, kol, judul in [(ax[0], top_rev, "revenue", f"{n_top} kategori dengan revenue tertinggi"),
                                (ax[1], top_qty, "total_terjual", f"{n_top} kategori dengan item terjual terbanyak")]:
        warna = [GRAY] * (len(data) - 1) + [PRIMARY]
        bars = a.barh(data["category"], data[kol], color=warna)
        a.set_title(judul, loc="left"); a.set_ylabel("")
        a.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        for b in bars:
            a.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():,.0f}", va="center", fontsize=9)
        a.set_xlim(0, data[kol].max() * 1.18)
    ax[0].set_xlabel("Revenue (BRL)"); ax[1].set_xlabel("Jumlah item terjual")
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)
    kontribusi = cat_df.nlargest(n_top, "revenue")["revenue"].sum() / total_rev * 100
    st.info(f"{n_top} kategori teratas menyumbang **{kontribusi:.1f}%** dari total revenue pada filter ini.")
    with st.expander("Lihat tabel kategori"):
        st.dataframe(cat_df.sort_values("revenue", ascending=False).round(2), width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Sebaran pelanggan dan kualitas pengiriman per state")
    state_df = create_state_df(main_df)
    metrik = st.radio("Warna lingkaran berdasarkan", ["Rata-rata lama kirim (hari)", "Rata-rata skor review", "Keterlambatan (%)"],
                      horizontal=True)
    kolom_warna = {"Rata-rata lama kirim (hari)": "avg_delivery_days", "Rata-rata skor review": "avg_review",
                   "Keterlambatan (%)": "late_pct"}[metrik]
    skala = "YlOrRd_r" if kolom_warna == "avg_review" else "YlOrRd"
    geo = state_df.dropna(subset=["lat", "lng"])
    fig_map = px.scatter_geo(
        geo, lat="lat", lon="lng", size="customer_count", color=kolom_warna, color_continuous_scale=skala,
        hover_name="customer_state", scope="south america", size_max=50,
        hover_data={"customer_count": ":,", "revenue": ":,.0f", "avg_delivery_days": ":.1f",
                    "avg_review": ":.2f", "late_pct": ":.1f", "lat": False, "lng": False},
        labels={"customer_count": "Pelanggan", "revenue": "Revenue", "avg_delivery_days": "Lama kirim (hari)",
                "avg_review": "Skor review", "late_pct": "Terlambat (%)"})
    fig_map.update_geos(fitbounds="locations", showcountries=True, countrycolor="#999", showland=True, landcolor="#F4F4F2",
                        showocean=True, oceancolor="#EAF2F8")
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520)
    st.plotly_chart(fig_map, width="stretch")
    st.caption("Ukuran lingkaran = jumlah pelanggan unik. Arahkan kursor ke lingkaran untuk detail tiap state.")

    c1, c2 = st.columns(2)
    with c1:
        top10 = state_df.head(10).sort_values("customer_count")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        warna = [GRAY] * max(len(top10) - 3, 0) + [PRIMARY] * min(3, len(top10))
        bars = ax.barh(top10["customer_state"], top10["customer_count"], color=warna)
        for b in bars:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():,.0f}", va="center", fontsize=9)
        ax.set_title("10 state dengan pelanggan terbanyak", loc="left"); ax.set_xlabel("Jumlah pelanggan unik")
        ax.set_xlim(0, top10["customer_count"].max() * 1.15)
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.0f}"))
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        bin_df = create_bin_df(main_df)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        warna = [PRIMARY if s >= 4 else (GRAY if s >= 3.5 else ACCENT) for s in bin_df["skor_review"]]
        bars = ax.bar(bin_df["delivery_bin"].astype(str), bin_df["skor_review"], color=warna)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(1, 5); ax.set_ylabel("Skor review (1-5)"); ax.set_xlabel("Lama pengiriman")
        ax.set_title("Skor review menurut lama pengiriman", loc="left")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    with st.expander("Lihat tabel per state"):
        st.dataframe(state_df.drop(columns=["lat", "lng"]).round(2), width="stretch")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("Segmentasi pelanggan berdasarkan RFM")
    st.caption("Recency = hari sejak pembelian terakhir, Frequency = jumlah pesanan, Monetary = total belanja. "
               "Dihitung ulang sesuai filter pada sidebar.")
    if main_df["customer_unique_id"].nunique() < 20:
        st.warning("Jumlah pelanggan terlalu sedikit untuk segmentasi RFM. Perluas rentang waktu atau state.")
    else:
        rfm_df, seg_df = create_rfm_df(main_df)
        m1, m2, m3 = st.columns(3)
        m1.metric("Rata-rata recency", f"{rfm_df['recency'].mean():.0f} hari")
        m2.metric("Rata-rata frequency", f"{rfm_df['frequency'].mean():.2f} pesanan")
        m3.metric("Rata-rata monetary", format_brl(rfm_df["monetary"].mean()))

        urut = seg_df.sort_values("persen_revenue")
        warna = [ACCENT if s == "At-Risk Big Spenders" else PRIMARY for s in urut["segment"]]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for a, kol, judul in [(ax[0], "persen_pelanggan", "% dari jumlah pelanggan"), (ax[1], "persen_revenue", "% dari total revenue")]:
            bars = a.barh(urut["segment"], urut[kol], color=warna)
            a.set_title(judul, loc="left"); a.set_xlabel("Persentase (%)"); a.set_ylabel("")
            a.set_xlim(0, urut[kol].max() * 1.2)
            for b in bars:
                a.text(b.get_width(), b.get_y() + b.get_height() / 2, f" {b.get_width():.1f}%", va="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with st.expander("Lihat ringkasan tiap segmen"):
            st.dataframe(seg_df.round(2), width="stretch")

st.caption("Dibuat dengan Streamlit | Sumber data: E-Commerce Public Dataset (Olist), periode Januari 2017 - Agustus 2018")
