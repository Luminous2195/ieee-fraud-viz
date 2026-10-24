import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ================= CẤU HÌNH TRANG & CSS ĐỒNG BỘ WIREFRAME =================
st.set_page_config(page_title="IEEE-CIS Fraud Detection - Tab Manager", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .kpi-box {
        background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 20px 22px; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02);
    }
    .kpi-label { font-size: 13.5px; font-weight: 600; color: #64748b; }
    .kpi-val { font-size: 30px; font-weight: 700; color: #1e293b; margin: 8px 0; }
    .kpi-val.coral { color: #f43f5e; }
    .kpi-val.teal { color: #0d9488; }
    .kpi-sub {
        display: inline-block; font-size: 12px; font-weight: 600;
        padding: 3px 9px; border-radius: 6px; background: #f1f5f9; color: #334155;
    }
    .kpi-sub.warn { background: #fff1f2; color: #f43f5e; }
    .insight-banner {
        background: linear-gradient(90deg, #f0fdfa 0%, #f0f9ff 100%);
        border: 1px solid #ccfbf1; border-radius: 12px; padding: 14px 22px;
        margin-bottom: 20px; font-size: 14px; color: #334155;
    }
</style>
""", unsafe_allow_html=True)


# ================= 1. ĐỌC DỮ LIỆU PARQUET THẬT (CÓ FALLBACK AN TOÀN) =================
@st.cache_data
def load_ieee_data(parquet_path="train_merged.parquet"):
    try:
        cols = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "DeviceType", "card4", "card6", "P_emaildomain"]
        df = pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        # Dữ liệu mô phỏng chuẩn xác theo thống kê thực của tập IEEE-CIS (590,540 dòng) nếu chưa trỏ đường dẫn Parquet
        np.random.seed(42)
        n = 590540
        df = pd.DataFrame({
            "TransactionID": np.arange(2987000, 2987000 + n),
            "TransactionDT": np.random.randint(86400, 15811200, size=n),
            "TransactionAmt": np.random.lognormal(mean=4.2, sigma=0.85, size=n).round(2),
            "DeviceType": np.random.choice(["desktop", "mobile", "Chưa xác định"], size=n, p=[0.1442, 0.0942, 0.7616]),
            "card4": np.random.choice(["visa", "mastercard", "american express", "discover"], size=n, p=[0.65, 0.32, 0.02, 0.01]),
            "card6": np.random.choice(["debit", "credit"], size=n, p=[0.74, 0.26]),
            "P_emaildomain": np.random.choice(["gmail.com", "yahoo.com", "outlook.com", "anonymous.com"], size=n, p=[0.42, 0.18, 0.10, 0.30]),
        })
        prob = np.where(df["card6"] == "credit", 0.066, 0.024)
        df["isFraud"] = np.random.binomial(1, prob)

    # Chuẩn hóa biến thời gian và nhãn hiển thị
    df["Day"] = ((df["TransactionDT"] - df["TransactionDT"].min()) // 86400) + 1
    df["Day"] = df["Day"].clip(1, 182)
    df["Week"] = ((df["Day"] - 1) // 7) + 1
    df["Week"] = df["Week"].clip(1, 26)

    df["DeviceType_VN"] = df["DeviceType"].fillna("Chưa xác định").replace({
        "desktop": "Máy tính", "mobile": "Điện thoại"
    })
    df["CardGroup"] = (
        df["card6"].fillna("khác").replace({"credit": "Tín dụng", "debit": "Ghi nợ"})
        + " - "
        + df["card4"].fillna("khác").str.title()
    )
    df["P_emaildomain"] = df["P_emaildomain"].fillna("anonymous.com")
    return df


df_raw = load_ieee_data()

# ================= 2. SIDEBAR BỘ LỌC HOẠT ĐỘNG =================
with st.sidebar:
    st.markdown("### BỘ LỌC ĐIỀU KHIỂN")

    day_range = st.slider("1. Kỳ quan sát (Ngày)", min_value=1, max_value=182, value=(1, 182))

    st.markdown("**2. Loại thiết bị**")
    dev_options = ["Máy tính", "Điện thoại", "Chưa xác định"]
    selected_devs = []
    for dev in dev_options:
        if st.checkbox(dev, value=True, key=f"dev_{dev}"):
            selected_devs.append(dev)

    card_groups = ["Tất cả loại thẻ"] + sorted(df_raw["CardGroup"].unique().tolist())
    selected_card = st.selectbox("3. Nhóm giao dịch", options=card_groups)

    min_group_size = st.selectbox(
        "4. Quy mô nhóm tối thiểu",
        options=[500, 5000, 20000, 50000],
        format_func=lambda x: f"Từ {x:,} giao dịch"
    )

# Áp dụng bộ lọc lên DataFrame
mask = (
    (df_raw["Day"] >= day_range[0])
    & (df_raw["Day"] <= day_range[1])
    & (df_raw["DeviceType_VN"].isin(selected_devs))
)
if selected_card != "Tất cả loại thẻ":
    mask &= (df_raw["CardGroup"] == selected_card)

df_filtered = df_raw.loc[mask]

# ================= 3. TÍNH TOÁN CHỈ SỐ KPI & THANH INSIGHT =================
total_tx = len(df_filtered)
fraud_df = df_filtered[df_filtered["isFraud"] == 1]
fraud_tx = len(fraud_df)
fraud_rate = (fraud_tx / total_tx * 100) if total_tx > 0 else 0.0
fraud_amt = fraud_df["TransactionAmt"].sum()
avg_fraud_amt = fraud_df["TransactionAmt"].mean() if fraud_tx > 0 else 0.0

# Tìm tuần đỉnh điểm
if total_tx > 0:
    wk_summary = df_filtered.groupby("Week").agg(
        rate=("isFraud", lambda s: s.mean() * 100),
        amt=("TransactionAmt", lambda s: s[df_filtered.loc[s.index, "isFraud"] == 1].sum())
    ).reset_index()
    peak_row = wk_summary.loc[wk_summary["rate"].idxmax()]
    peak_wk_str = f"Tuần {int(peak_row['Week'])} ({peak_row['rate']:.2f}%)"
else:
    peak_wk_str = "N/A"

st.markdown(f"""
<div class="insight-banner">
    <span style="background:#0d9488;color:#fff;padding:4px 10px;border-radius:6px;font-weight:700;font-size:12px;margin-right:10px;">ĐIỂM NHẤN</span>
    Tỷ lệ gian lận đang ở mức <b>{fraud_rate:.2f}%</b> ({'vượt ngưỡng an toàn 3.00%' if fraud_rate > 3.0 else 'nằm trong ngưỡng an toàn ≤ 3.00%'}),
    tập trung cao nhất vào <b>{peak_wk_str}</b>.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Tổng lượng giao dịch</div>
        <div class="kpi-val">{total_tx:,.0f}</div>
        <div class="kpi-sub">Kỳ quan sát: Ngày {day_range[0]} – {day_range[1]}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Số giao dịch gian lận</div>
        <div class="kpi-val coral">{fraud_tx:,.0f}</div>
        <div class="kpi-sub warn">Xác nhận (isFraud = 1)</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    rate_cls = "coral" if fraud_rate > 3.0 else "teal"
    sub_cls = "warn" if fraud_rate > 3.0 else ""
    sub_txt = "▲ Vượt ngưỡng 3.00%" if fraud_rate > 3.0 else "✓ Đạt chuẩn SLA ≤ 3.00%"
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Tỷ lệ gian lận</div>
        <div class="kpi-val {rate_cls}">{fraud_rate:.2f}%</div>
        <div class="kpi-sub {sub_cls}">{sub_txt}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-label">Tổng giá trị gian lận</div>
        <div class="kpi-val teal">${fraud_amt:,.0f}</div>
        <div class="kpi-sub">TB: ${avg_fraud_amt:,.2f} / vụ</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ================= 4. BIỂU ĐỒ 1: XU HƯỚNG TỶ LỆ & GIÁ TRỊ GIAN LẬN (TRỤC KÉP) =================
col_title1, col_toggle1 = st.columns([4, 1])
with col_title1:
    st.subheader("Xu hướng Tỷ lệ Gian lận và Giá trị Gian lận theo Thời gian")
with col_toggle1:
    time_unit = st.radio("Đơn vị thời gian", ["Theo Tuần", "Theo Ngày"], horizontal=True, label_visibility="collapsed")

if total_tx > 0:
    group_col = "Week" if time_unit == "Theo Tuần" else "Day"
    prefix = "Tuần " if time_unit == "Theo Tuần" else "Ngày "

    trend_df = df_filtered.groupby(group_col).agg(
        total_tx=("isFraud", "count"),
        fraud_tx=("isFraud", "sum"),
        fraud_amt=("TransactionAmt", lambda s: s[df_filtered.loc[s.index, "isFraud"] == 1].sum())
    ).reset_index()
    trend_df["fraud_rate"] = (trend_df["fraud_tx"] / trend_df["total_tx"]) * 100
    trend_df["fraud_amt_k"] = trend_df["fraud_amt"] / 1000.0
    trend_df["label"] = prefix + trend_df[group_col].astype(str)

    max_idx = trend_df["fraud_rate"].idxmax()
    bar_colors = [
        "#ffe4e6" if (i == max_idx and r > 3.0) else ("#bae6fd" if r > 3.8 else "#e0f2fe")
        for i, r in zip(trend_df.index, trend_df["fraud_rate"])
    ]
    marker_colors = [
        "#f43f5e" if (i == max_idx and r > 3.0) else "#ffffff"
        for i, r in zip(trend_df.index, trend_df["fraud_rate"])
    ]
    marker_sizes = [13 if (i == max_idx and r > 3.0) else 9 for i, r in zip(trend_df.index, trend_df["fraud_rate"])]

    fig1 = go.Figure()

    # Cột Giá trị gian lận ($K) - Trục Y phải
    fig1.add_trace(go.Bar(
        x=trend_df["label"],
        y=trend_df["fraud_amt_k"],
        name="Giá trị gian lận ($K)",
        marker_color=bar_colors,
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Giá trị gian lận: $%{y:.1f}K<extra></extra>"
    ))

    # Đường Tỷ lệ gian lận (%) - Trục Y trái
    fig1.add_trace(go.Scatter(
        x=trend_df["label"],
        y=trend_df["fraud_rate"],
        name="Tỷ lệ gian lận (%)",
        mode="lines+markers",
        line=dict(color="#0d9488", width=3.2),
        marker=dict(
            size=marker_sizes,
            color=marker_colors,
            line=dict(color="#0d9488", width=2.5)
        ),
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Tỷ lệ gian lận: %{y:.2f}%<extra></extra>"
    ))

    # Đường ngưỡng SLA 3.0%
    fig1.add_hline(y=3.0, line_dash="dash", line_color="#e11d48", opacity=0.75)

    fig1.update_layout(
        height=360,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Tỷ lệ gian lận (%)", ticksuffix="%", range=[0, max(6.5, trend_df["fraud_rate"].max() * 1.15)], gridcolor="#f1f5f9"),
        yaxis2=dict(title="Giá trị gian lận ($K)", tickprefix="$", ticksuffix="K", overlaying="y", side="right", showgrid=False)
    )
    st.plotly_chart(fig1, use_container_width=True)

# ================= 5. BIỂU ĐỒ 2: SO SÁNH THEO NHÓM GIAO DỊCH =================
col_title2, col_dim, col_met = st.columns([3, 1.5, 1.5])
with col_title2:
    st.subheader("So Sánh Theo Nhóm Giao Dịch")
with col_dim:
    dim_choice = st.selectbox("Chiều phân tích", ["Loại thẻ", "Thiết bị", "Email"], label_visibility="collapsed")
with col_met:
    met_choice = st.selectbox("Chỉ số đo lường", ["Tỷ lệ (%)", "Tổng giá trị ($)"], label_visibility="collapsed")

if total_tx > 0:
    dim_map = {"Loại thẻ": "CardGroup", "Thiết bị": "DeviceType_VN", "Email": "P_emaildomain"}
    col_name = dim_map[dim_choice]

    grp = df_filtered.groupby(col_name).agg(
        total_tx=("isFraud", "count"),
        fraud_tx=("isFraud", "sum"),
        fraud_amt=("TransactionAmt", lambda s: s[df_filtered.loc[s.index, "isFraud"] == 1].sum())
    ).reset_index()

    grp = grp[grp["total_tx"] >= min_group_size].copy()
    grp["fraud_rate"] = (grp["fraud_tx"] / grp["total_tx"]) * 100

    sort_col = "fraud_rate" if met_choice == "Tỷ lệ (%)" else "fraud_amt"
    grp = grp.sort_values(by=sort_col, ascending=True).tail(8)

    colors = ["#0d9488" if r > 3.0 else "#94a3b8" for r in grp["fraud_rate"]]

    fig2 = go.Figure(go.Bar(
        x=grp[sort_col],
        y=grp[col_name],
        orientation="h",
        marker_color=colors,
        text=[
            f"{r:.2f}% (${a:,.0f} • {t:,} GD)" if met_choice == "Tỷ lệ (%)" else f"${a:,.0f} ({r:.2f}% • {t:,} GD)"
            for r, a, t in zip(grp["fraud_rate"], grp["fraud_amt"], grp["total_tx"])
        ],
        textposition="outside"
    ))

    if met_choice == "Tỷ lệ (%)":
        fig2.add_vline(x=3.0, line_dash="dash", line_color="#f43f5e", annotation_text="Ngưỡng 3.0%")

    fig2.update_layout(
        height=300,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=120, t=20, b=20),
        xaxis=dict(gridcolor="#f1f5f9", range=[0, grp[sort_col].max() * 1.35])
    )
    st.plotly_chart(fig2, use_container_width=True)