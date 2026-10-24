import pandas as pd
import json

# 1. Đọc file Parquet thật của nhóm
df = pd.read_parquet("train_merged.parquet") # Đổi tên đúng với file Parquet của nhóm bạn

# Chuẩn hóa cột Ngày & Tuần từ TransactionDT
min_dt = df["TransactionDT"].min()
df["Day"] = ((df["TransactionDT"] - min_dt) // 86400) + 1
df["Day"] = df["Day"].clip(1, 182)
df["Week"] = ((df["Day"] - 1) // 7) + 1
df["Week"] = df["Week"].clip(1, 26)

# Chuẩn hóa Thiết bị và Nhóm thẻ
df["Device"] = df["DeviceType"].fillna("unknown").str.lower()
df["CardGroup"] = (
    df["card6"].fillna("other").str.lower() + "_" + df["card4"].fillna("other").str.lower()
)
df["Email"] = df["P_emaildomain"].fillna("anonymous.com").str.lower()

# 2. Tổng hợp dữ liệu theo tổ hợp Bộ lọc (Tuần x Thiết bị x Loại thẻ) để bộ lọc trên Web tính đúng 100%
grouped = df.groupby(["Week", "Device", "CardGroup"]).agg(
    total_tx=("isFraud", "count"),
    fraud_tx=("isFraud", "sum"),
    fraud_amt=("TransactionAmt", lambda s: float(s[df.loc[s.index, "isFraud"] == 1].sum()))
).reset_index()

# 3. Xuất ra file data.json để nhúng vào Web
records = grouped.to_dict(orient="records")
with open("real_data.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False)

print("Đã xuất thành công real_data.json từ dữ liệu Parquet thật!")