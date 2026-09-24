"""
คลีนข้อมูล RFID (RFID_-_2569.csv) + เติมเวลาเข้างานด้วย KMeans

ขั้นตอน
1) ลบแถวหัวตารางที่ซ้ำทุกวัน (แถวที่ UID == 'UID')
2) แปลงปี พ.ศ. -> ค.ศ. และเติมศูนย์นำหน้าเวลา (6:54:18 -> 06:54:18)
3) แยกเวลาสแกนหลายค่าในเซลล์เดียวเป็นรายการ -> หาเวลาแรก/เวลาสุดท้ายของแต่ละวัน
   (สมมติฐาน: เวลาแรก = เข้างาน, เวลาสุดท้าย = เลิกงาน)
4) วันที่สแกนครั้งเดียว:
     - ใช้ KMeans บน (เวลาเข้า, เวลาออก) ของ "วันที่สแกนครบ" ของคนนั้น ๆ เพื่อหากะงานปกติ
       (ถ้าคนนั้นมีข้อมูลไม่พอ ใช้ KMeans รวมทุกคนแทน)
     - ถ้าเวลาที่สแกนใกล้ "เวลาเลิกงาน" ของกะใดมากที่สุด -> เติมเวลาเข้างานด้วยศูนย์กลางกะนั้น
     - ถ้าใกล้ "เวลาเข้างาน" -> ถือเป็นเวลาเข้า (ขาดเวลาออก ไม่เติม)
     - ถ้าไกลจากทุกกะเกิน MAX_GAP ชม. -> ไม่เติม (scan_type = single_unclear)
วิธีใช้:  python clean_rfid_kmeans.py RFID_-_2569.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SRC = sys.argv[1] if len(sys.argv) > 1 else "RFID_-_2569.csv"
OUT = "rfid_clean_kmeans.csv"          # 1 แถว = 1 คน 1 วัน
OUT_SCANS = "rfid_scans_clean.csv"     # 1 แถว = 1 การสแกน
MIN_SPAN = 3.0        # ชม. ขั้นต่ำระหว่างสแกนแรก-สุดท้าย เพื่อนับว่าเป็น "วันที่สแกนครบ"
MIN_DAYS_PERSON = 8   # จำนวนวันครบขั้นต่ำที่จะใช้ KMeans รายบุคคล
MAX_K_PERSON = 3      # จำนวนกะสูงสุดต่อคน
MAX_K_GLOBAL = 4
MAX_GAP = 1.5         # ชม. ระยะห่างสูงสุดจากศูนย์กลางกะที่ยอมรับ


def to_hour(t: str) -> float:
    h, m, s = (int(x) for x in t.strip().split(":"))
    return h + m / 60 + s / 3600


def fmt(h: float) -> str:
    if pd.isna(h):
        return ""
    total = int(round(h * 60))
    return f"{total // 60:02d}:{total % 60:02d}"


def fit_shifts(X: np.ndarray, max_k: int):
    """เลือก k (1..max_k) ด้วย silhouette แล้วคืนศูนย์กลางกะ (start, end)"""
    best_k, best_s = 1, -1.0
    for k in range(2, max_k + 1):
        if len(X) <= k * 3:
            break
        s = silhouette_score(X, KMeans(k, n_init=10, random_state=0).fit_predict(X))
        if s > best_s:
            best_k, best_s = k, s
    if best_k == 1:
        return np.median(X, axis=0, keepdims=True), 1, float("nan")
    km = KMeans(best_k, n_init=20, random_state=0).fit(X)
    return km.cluster_centers_, best_k, best_s


# ---------- 1-3) อ่านและทำความสะอาดพื้นฐาน ----------
raw = pd.read_csv(SRC, header=None, names=["date", "uid", "name", "times"], dtype=str)
df = raw[raw["uid"] != "UID"].copy()                                               # ตัดหัวตารางซ้ำ
df["date"] = pd.to_datetime(df["date"].str.replace(r"^2569", "2026", regex=True))  # พ.ศ. -> ค.ศ.
df["uid"] = df["uid"].str.strip()
df["name"] = df["name"].str.strip()
df["hours"] = df["times"].apply(lambda s: sorted(to_hour(t) for t in s.split(",") if t.strip()))
df["n_scans"] = df["hours"].str.len()
df = df[df["n_scans"] > 0].drop_duplicates(["date", "uid"]).reset_index(drop=True)
df["first"] = df["hours"].str[0]
df["last"] = df["hours"].str[-1]

scans = df[["date", "uid", "name", "hours"]].explode("hours")      # 1 แถว = 1 สแกน (ใช้ทำ Apriori/EDA ต่อ)
scans["time"] = scans["hours"].apply(fmt)
scans.drop(columns="hours").to_csv(OUT_SCANS, index=False, encoding="utf-8-sig")

# ---------- 4) KMeans หากะงานจากวันที่สแกนครบ ----------
full = df[(df["n_scans"] >= 2) & ((df["last"] - df["first"]) >= MIN_SPAN)]
g_centers, g_k, g_s = fit_shifts(full[["first", "last"]].to_numpy(), MAX_K_GLOBAL)
print(f"[รวมทุกคน] k={g_k}, silhouette={g_s:.3f}")
print(pd.DataFrame(g_centers, columns=["start", "end"]).round(2), "\n")

person_centers = {}
for uid, g in full.groupby("uid"):
    if len(g) >= MIN_DAYS_PERSON:
        person_centers[uid] = fit_shifts(g[["first", "last"]].to_numpy(), MAX_K_PERSON)[0]


def classify_single(uid: str, h: float):
    """คืน (ชนิดสแกน, เวลาเข้างานที่ควรเป็น, ระดับที่ใช้)"""
    c, level = (person_centers[uid], "person") if uid in person_centers else (g_centers, "global")
    d_start, d_end = np.abs(c[:, 0] - h), np.abs(c[:, 1] - h)
    if min(d_start.min(), d_end.min()) > MAX_GAP:
        return "single_unclear", np.nan, level
    if d_end.min() <= d_start.min():
        return "single_checkout", c[d_end.argmin(), 0], level
    return "single_checkin", np.nan, level


df["scan_type"] = "multi"
df["start"] = df["first"]
df["end"] = df["last"]
df["start_imputed"] = False
df["kmeans_level"] = ""

for i in df.index[df["n_scans"] == 1]:
    kind, start, level = classify_single(df.at[i, "uid"], df.at[i, "first"])
    df.at[i, "scan_type"] = kind
    df.at[i, "kmeans_level"] = level
    if kind == "single_checkout":
        df.at[i, "start"], df.at[i, "end"] = start, df.at[i, "first"]
        df.at[i, "start_imputed"] = True
    elif kind == "single_checkin":
        df.at[i, "end"] = np.nan
    else:
        df.at[i, "start"] = df.at[i, "end"] = np.nan

df["work_hours"] = (df["end"] - df["start"]).round(2)
df["start_time"] = df["start"].apply(fmt)
df["end_time"] = df["end"].apply(fmt)
df["weekday"] = df["date"].dt.day_name()

cols = ["date", "weekday", "uid", "name", "n_scans", "scan_type", "start_time", "end_time",
        "start_imputed", "kmeans_level", "work_hours"]
df[cols].to_csv(OUT, index=False, encoding="utf-8-sig")

print(df["scan_type"].value_counts(), "\n")
print("เติมเวลาเข้างานด้วย KMeans:", int(df["start_imputed"].sum()), "วัน")
print(f"บันทึก: {OUT}, {OUT_SCANS}")