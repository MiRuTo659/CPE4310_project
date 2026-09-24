"""
คลีนข้อมูล RFID_-_2569.csv
ใช้งาน: python clean_rfid.py RFID_-_2569.csv
ผลลัพธ์:
  rfid_daily.csv  -> 1 แถว = 1 คน 1 วัน (พร้อมฟีเจอร์เบื้องต้น)
  rfid_scans.csv  -> 1 แถว = 1 การสแกน
"""
import re
import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "RFID_2569.csv"

# 1) อ่านไฟล์: ไฟล์ไม่มีหัวตารางจริง (แถวแรกของแต่ละวันคือหัวตารางซ้ำ)
raw = pd.read_csv(
    path,
    header=None,
    names=["date", "uid", "name", "times"],
    dtype=str,
    encoding="utf-8-sig",
    skipinitialspace=True,
)
print(f"อ่านได้ {len(raw)} แถว")

# 2) ตัดแถวหัวตารางซ้ำ (uid == 'UID') และแถวว่าง
df = raw[raw["uid"].str.strip() != "UID"].copy()
df = df.dropna(subset=["date", "uid", "name"])
print(f"หลังตัดหัวตารางซ้ำ {len(df)} แถว")

# 3) ทำความสะอาดข้อความ
df["uid"] = df["uid"].str.strip().str.upper()
df["name"] = df["name"].str.strip()
df["times"] = df["times"].fillna("").str.strip()

# 4) แปลงวันที่ พ.ศ. -> ค.ศ. (2569-06-01 -> 2026-06-01)
def be_to_ce(s: str) -> str:
    y, m, d = s.strip().split("-")
    y = int(y)
    return f"{y - 543 if y > 2400 else y:04d}-{m}-{d}"

df["date"] = pd.to_datetime(df["date"].map(be_to_ce), errors="coerce")
bad_dates = df["date"].isna().sum()
if bad_dates:
    print(f"เตือน: วันที่อ่านไม่ได้ {bad_dates} แถว (ถูกตัดออก)")
df = df.dropna(subset=["date"])

# 5) แยกเวลาสแกน: เติมศูนย์นำหน้า (6:54:18 -> 06:54:18), ตรวจรูปแบบ, เรียง, ตัดซ้ำ
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")

def parse_times(cell: str) -> list[str]:
    out = []
    for t in cell.split(","):
        m = TIME_RE.match(t.strip())
        if m and int(m[1]) < 24 and int(m[2]) < 60 and int(m[3]) < 60:
            out.append(f"{int(m[1]):02d}:{m[2]}:{m[3]}")
    return sorted(set(out))

df["scan_list"] = df["times"].map(parse_times)
df = df[df["scan_list"].str.len() > 0].copy()

# 6) ตัดแถวซ้ำ (คนเดียวกัน วันเดียวกัน) โดยรวมเวลาสแกนเข้าด้วยกัน
df = (
    df.groupby(["date", "uid", "name"], as_index=False)["scan_list"]
    .agg(lambda lists: sorted({t for l in lists for t in l}))
)

# 7) ตรวจ UID <-> ชื่อ ว่าตรงกันแบบ 1:1
assert df.groupby("uid")["name"].nunique().max() == 1, "พบ UID เดียวใช้หลายชื่อ"
assert df.groupby("name")["uid"].nunique().max() == 1, "พบชื่อเดียวมีหลาย UID"

# 8) ตารางรายคนรายวัน + ฟีเจอร์เบื้องต้น
to_hour = lambda t: int(t[:2]) + int(t[3:5]) / 60 + int(t[6:]) / 3600

df["n_scans"] = df["scan_list"].str.len()
df["first_scan"] = df["scan_list"].str[0]
df["last_scan"] = df["scan_list"].str[-1]
df["first_hour"] = df["first_scan"].map(to_hour).round(3)
df["last_hour"] = df["last_scan"].map(to_hour).round(3)
df["span_hours"] = (df["last_hour"] - df["first_hour"]).round(3)
df["dow"] = df["date"].dt.dayofweek          # 0 = จันทร์
df["is_late_shift"] = (df["first_hour"] >= 14).astype(int)
df["overtime"] = (df["last_hour"] >= 19).astype(int)
df["scan_times"] = df["scan_list"].str.join(" ")

daily = df.drop(columns="scan_list").sort_values(["name", "date"])

# 9) ตารางรายการสแกน (long format)
scans = (
    df[["date", "uid", "name", "scan_list"]]
    .explode("scan_list")
    .rename(columns={"scan_list": "scan_time"})
)
scans["hour"] = scans["scan_time"].map(to_hour).round(3)
scans["scan_no"] = scans.groupby(["date", "uid"]).cumcount() + 1
scans = scans.sort_values(["name", "date", "scan_time"])

# 10) รายงานสรุปคุณภาพข้อมูล
all_days = pd.date_range(daily["date"].min(), daily["date"].max())
missing = [d.strftime("%Y-%m-%d") for d in all_days if d not in set(daily["date"])]
print(f"พนักงาน {daily['uid'].nunique()} คน | {daily['date'].nunique()} วัน "
      f"({daily['date'].min().date()} ถึง {daily['date'].max().date()})")
print(f"รายการรายวัน {len(daily)} แถว | สแกนทั้งหมด {len(scans)} ครั้ง")
print("วันที่ไม่มีข้อมูลเลย:", missing or "ไม่มี")

daily.to_csv("rfid_daily.csv", index=False, encoding="utf-8-sig")
scans.to_csv("rfid_scans.csv", index=False, encoding="utf-8-sig")
print("บันทึก rfid_daily.csv และ rfid_scans.csv แล้ว")