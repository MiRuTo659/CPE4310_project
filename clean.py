import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "dataset/FAOSTAT_data_en_11-1-2024.csv"
OUTPUT_FILE = "Thailand_Temperature_Clean.csv"


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE,
    encoding="latin1"
)

print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print("Shape:", df.shape)
print(df.columns.tolist())


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.replace("\ufeff", "", regex=False)
    .str.strip()
)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. SELECT THAILAND
# ============================================================

df = df[
    df["Area"].astype(str).str.strip() == "Thailand"
].copy()

print("\nThailand rows:", len(df))


# ============================================================
# 4. SELECT TEMPERATURE CHANGE
# ============================================================

df = df[
    df["Element"].astype(str).str.strip()
    == "Temperature change"
].copy()


# ============================================================
# 5. SELECT ANNUAL DATA
# ============================================================

df = df[
    df["Months"].astype(str).str.strip()
    == "Meteorological year"
].copy()


# ============================================================
# 6. SELECT ONLY REQUIRED COLUMNS
# ============================================================

df = df[
    [
        "Area",
        "Year",
        "Unit",
        "Value",
        "Flag"
    ]
].copy()


# ============================================================
# 7. CONVERT DATA TYPE
# ============================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)

df["Value"] = pd.to_numeric(
    df["Value"],
    errors="coerce"
)


# ============================================================
# 8. REMOVE MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna(
    subset=["Year", "Value"]
)


# ============================================================
# 9. REMOVE DUPLICATES
# ============================================================

duplicates = df.duplicated().sum()

print("\nDuplicate rows:", duplicates)

df = df.drop_duplicates()


# ============================================================
# 10. SORT BY YEAR
# ============================================================

df = df.sort_values(
    "Year"
).reset_index(drop=True)


# ============================================================
# 11. CHECK DUPLICATE YEARS
# ============================================================

duplicate_years = (
    df["Year"].duplicated().sum()
)

print(
    "Duplicate years:",
    duplicate_years
)


# ============================================================
# 12. CHECK MISSING YEARS
# ============================================================

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

expected_years = set(
    range(min_year, max_year + 1)
)

actual_years = set(
    df["Year"].astype(int)
)

missing_years = sorted(
    expected_years - actual_years
)

print("\nYear range:")
print(min_year, "-", max_year)

print(
    "Missing years:",
    missing_years
)


# ============================================================
# 13. IF MISSING YEAR → INTERPOLATE
# ============================================================

if len(missing_years) > 0:

    full_years = pd.DataFrame({
        "Year": range(
            min_year,
            max_year + 1
        )
    })

    df = full_years.merge(
        df,
        on="Year",
        how="left"
    )

    df["Value"] = (
        df["Value"]
        .interpolate(
            method="linear"
        )
    )


# ============================================================
# 14. FINAL DATA CLEAN
# ============================================================

df = df[
    ["Year", "Value"]
].copy()

df = df.sort_values(
    "Year"
).reset_index(drop=True)


# ============================================================
# 15. CHECK FINAL DATA
# ============================================================

print("\n" + "=" * 70)
print("CLEAN DATA")
print("=" * 70)

print(df.to_string(index=False))


print("\nStatistics:")
print(df["Value"].describe())


# ============================================================
# 16. FIND LATEST YEAR
# ============================================================

latest_year = int(
    df["Year"].max()
)

print("\nLatest year:", latest_year)


# ============================================================
# 17. FORECAST YEARS
# ============================================================

future_years = np.arange(
    latest_year + 1,
    latest_year + 6
)

print("\nForecast years:")

for year in future_years:
    print(year)


# ============================================================
# 18. SAVE CLEAN DATA
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", OUTPUT_FILE)


# ============================================================
# 19. PLOT CLEAN DATA
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    df["Year"],
    df["Value"],
    marker="o"
)

plt.title(
    "Thailand Temperature Change (1961-2023)"
)

plt.xlabel("Year")

plt.ylabel(
    "Temperature Change (°C)"
)

plt.grid(True)

plt.tight_layout()

plt.show()