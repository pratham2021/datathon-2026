"""
clean_property_data.py
======================
Cleans PropertyAssessmentData.csv and writes two output files:
  - PropertyAssessmentData_cleaned.csv   (human-readable, cleaned)
  - PropertyAssessmentData_model.csv     (ML-ready: one-hot encoded + normalised)

Cleaning steps
--------------
1.  Drop structurally redundant columns (exact duplicates)
2.  Standardise boolean flags (Y/N → True/False)
3.  Decode numeric-coded categoricals (GarageTypeParking, PoolType)
4.  Decode AirConditioning codes
5.  Impute missing numeric fields with sensible defaults
6.  Parse ZipCode / ZipCodePlus4 as zero-padded strings
7.  Strip whitespace from all string columns
8.  Cast columns to their correct final dtypes
9.  Drop exact duplicate rows
10. Drop low-variance columns (any column where one value covers >= 95% of rows)
11. ML encoding: one-hot encode nominals, min-max normalise continuous features
12. Report before/after stats
"""

import pandas as pd
import numpy as np

# ── 0. Load ───────────────────────────────────────────────────────────────────

INPUT        = "PropertyAssessmentData.csv"
OUTPUT       = "PropertyAssessmentData_cleaned.csv"
OUTPUT_MODEL = "PropertyAssessmentData_model.csv"

df = pd.read_csv(INPUT, dtype_backend="numpy_nullable")
print(f"Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 1. Drop structurally redundant columns ────────────────────────────────────
#
#   CensusKeyDecennial is an exact duplicate of CensusKey. Drop it explicitly.
#   All zero/low-variance columns are handled generically in step 10.

if "CensusKeyDecennial" in df.columns:
    df.drop(columns=["CensusKeyDecennial"], inplace=True)
    print("Dropped redundant column: CensusKeyDecennial")

# ── 2. Standardise boolean flags → True / False ───────────────────────────────

BOOL_COLS = ["HasFireplace", "HasDeck", "HasGuestHouse",
             "HasSecurityAlarm", "HasSprinklers"]

for col in BOOL_COLS:
    if col in df.columns:
        df[col] = df[col].map({"Y": True, "N": False}).astype("boolean")

print(f"Boolean columns converted: {[c for c in BOOL_COLS if c in df.columns]}")

# ── 3. Decode numeric-coded categoricals ──────────────────────────────────────
#
#   Source: GarageTypeParkingValueReference.txt, PoolTypeValueReference.txt

if "GarageTypeParking" in df.columns:
    garage_map = {
        0:   "No Garage",
        4:   "Pole Building Garage",
        11:  "Garage, Attached",
        12:  "Garage, Detached",
        14:  "Garage, Finished",
        19:  "Detached, Finished",
        20:  "Detached, Unfinished",
        21:  "Attached, Finished",
        22:  "Attached, Unfinished",
        30:  "Carport (Unspecified)",
        40:  "Garage, Basement",
        52:  "Garage, Tuckunder",
        53:  "Garage, Built-in",
        999: "Type Not Specified",
    }
    df["GarageTypeParking"] = (
        df["GarageTypeParking"]
        .map(garage_map)
        .fillna("Unknown")
        .astype("string")
    )

if "PoolType" in df.columns:
    pool_map = {
        101: "Public/Municipal/Commercial",
        110: "Above-ground Pool",
        121: "In-ground Vinyl Pool",
        130: "Indoor Pool",
        401: "Pool (Unspecified), Heated",
        402: "Pool, Solar Heated",
        502: "Pool w/Hot Tub/Spa",
        610: "Pool (Unspecified), Enclosed",
        904: "Spa/Hot Tub (only)",
        999: "Type Unspecified",
    }
    df["PoolType"] = (
        df["PoolType"]
        .map(pool_map)
        .fillna("No Pool")
        .astype("string")
    )

print("Decoded: GarageTypeParking, PoolType")

# ── 4. Decode AirConditioning codes ───────────────────────────────────────────
#
#   Source: AirConditioningValueReference.txt

if "AirConditioning" in df.columns:
    ac_map = {
        "C": "Central",
        "E": "Evaporative Cooler",
        "F": "Office Only",
        "I": "Geo-Thermal",
        "K": "Packaged Unit",
        "L": "Window/Unit",
        "N": "None",
        "O": "Other",
        "P": "Partial",
        "Q": "Chilled Water",
        "R": "Refrigeration",
        "V": "Ventilation",
        "W": "Wall",
        "Y": "Yes",
    }
    df["AirConditioning"] = (
        df["AirConditioning"]
        .map(ac_map)
        .fillna("Unknown")
        .astype("string")
    )
    print("Decoded: AirConditioning")

# ── 5. Impute missing numerics ────────────────────────────────────────────────
#
#   Suite            : free-form / mostly absent → fill "" (string)
#   LotSizeAreaUnit  : all present rows are 'SF'; fill missing with 'SF'
#   YearBuilt        : 19 nulls → fill with median
#   NumberOfStories  : ~98% null → fill with 1.0 (most common value)
#   DeckArea         : 28 nulls → 0 (no deck)
#   GuestHouseArea   : 28 nulls → 0
#   PoolArea         : 28 nulls → 0
#   ZipCodePlus4     : 1 null  → 0 (unknown extension)

if "Suite" in df.columns:
    df["Suite"] = df["Suite"].fillna("").astype("string")

if "LotSizeAreaUnit" in df.columns:
    df["LotSizeAreaUnit"] = df["LotSizeAreaUnit"].fillna("SF").astype("string")

if "YearBuilt" in df.columns:
    median_year = int(df["YearBuilt"].median())
    n_filled = df["YearBuilt"].isna().sum()
    df["YearBuilt"] = df["YearBuilt"].fillna(median_year).astype("Int64")
    print(f"YearBuilt: filled {n_filled} nulls with median ({median_year})")

if "NumberOfStories" in df.columns:
    n_filled = df["NumberOfStories"].isna().sum()
    df["NumberOfStories"] = df["NumberOfStories"].fillna(1.0).astype("Float64")
    print(f"NumberOfStories: filled {n_filled} nulls with 1.0")

for col in ["DeckArea", "GuestHouseArea", "PoolArea"]:
    if col in df.columns:
        df[col] = df[col].fillna(0).astype("Int64")

if "ZipCodePlus4" in df.columns:
    df["ZipCodePlus4"] = df["ZipCodePlus4"].fillna(0).astype("Int64")

print("Numeric nulls imputed.")

# ── 6. Format ZIP codes as zero-padded strings ────────────────────────────────

if "ZipCode" in df.columns:
    df["ZipCode"] = df["ZipCode"].astype(str).str.zfill(5)

if "ZipCodePlus4" in df.columns:
    df["ZipCodePlus4"] = df["ZipCodePlus4"].astype(str).str.zfill(4)

print("ZipCode / ZipCodePlus4 padded to strings.")

# ── 7. Strip whitespace from all string columns ───────────────────────────────

str_cols = df.select_dtypes(include=["string", "object"]).columns
for col in str_cols:
    df[col] = df[col].str.strip()

print(f"Whitespace stripped from {len(str_cols)} string column(s).")

# ── 8. Cast remaining columns to correct dtypes ───────────────────────────────

int_cols = [
    "RecordId", "MAK", "FIPSCode", "CensusKey",
    "TotalAssessedValue", "SalesPriceFromAssessment",
    "TotalNumberOfRooms", "NumberOfBedrooms",
    "NumberOfPartialBaths", "NumberOfBuildings",
    "NumberOfUnits", "GarageParkingNumberOfCars",
    "FireplaceCount",
]
for col in int_cols:
    if col in df.columns:
        df[col] = df[col].astype("Int64")

float_cols = ["Latitude", "Longitude", "LotSizeOrArea", "NumberOfBaths"]
for col in float_cols:
    if col in df.columns:
        df[col] = df[col].astype("Float64")

str_cols_explicit = ["PropertyAddress", "City", "APN", "LotSizeAreaUnit"]
for col in str_cols_explicit:
    if col in df.columns:
        df[col] = df[col].astype("string")

print("Dtypes finalised.")

# ── 9. Drop duplicate rows ────────────────────────────────────────────────────

before = len(df)
df.drop_duplicates(inplace=True)
dupes_removed = before - len(df)
print(f"Duplicate rows removed: {dupes_removed}")

# ── 10. Drop low-variance columns ─────────────────────────────────────────────
#
#   Drop any column where a single value accounts for >= 95% of rows.
#   This catches near-constant columns (e.g. TotalNumberOfRooms is 0 for 96%
#   of rows) that carry no useful signal for KNN / cosine similarity.
#   Applied to both df (cleaned) and df_model (after one-hot encoding).

LOW_VARIANCE_THRESHOLD = 0.95

def drop_low_variance(frame, threshold, label=""):
    to_drop = []
    for col in frame.columns:
        top_freq = frame[col].value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq >= threshold:
            to_drop.append((col, top_freq))
    if to_drop:
        frame.drop(columns=[c for c, _ in to_drop], inplace=True)
        for col, freq in to_drop:
            print(f"  {label}'{col}' — dominant value covers {freq:.1%} of rows")
    else:
        print(f"  None found.")
    return [c for c, _ in to_drop]

print(f"Low-variance columns dropped (threshold >= {LOW_VARIANCE_THRESHOLD:.0%}):")
drop_low_variance(df, LOW_VARIANCE_THRESHOLD)

# ── 11. ML encoding: one-hot + min-max normalisation ─────────────────────────
#
#   Start from the cleaned df so the human-readable CSV is saved separately.
#
#   One-hot encode nominal categoricals (only those that survived step 10):
#       GarageTypeParking, PoolType, AirConditioning, CountyLandUseDescription
#
#   Drop non-feature identifier / string columns that carry no signal for KNN:
#       RecordId, MAK, PropertyAddress, Suite, APN,
#       ZipCode, ZipCodePlus4, City, FIPSCode, CensusKey, LotSizeAreaUnit
#
#   Cast boolean cols → int (0/1) so the matrix is fully numeric.
#
#   Min-max normalise all continuous numeric columns so no single feature
#   dominates cosine distance calculations.

NOMINAL_COLS = [
    "GarageTypeParking",
    "PoolType",
    "AirConditioning",
    "CountyLandUseDescription",
]

ID_COLS = [
    "RecordId", "MAK", "PropertyAddress", "Suite", "APN",
    "ZipCode", "ZipCodePlus4", "City", "FIPSCode", "CensusKey",
    "LotSizeAreaUnit",
]

df_model = df.copy()

# One-hot encode nominals that survived the low-variance drop
nominal_cols_present = [c for c in NOMINAL_COLS if c in df_model.columns]
if nominal_cols_present:
    df_model = pd.get_dummies(df_model, columns=nominal_cols_present, dtype=int)
    print(f"One-hot encoded: {nominal_cols_present}")

# Drop identifier columns
drop_id = [c for c in ID_COLS if c in df_model.columns]
df_model.drop(columns=drop_id, inplace=True)
print(f"Dropped identifier columns: {drop_id}")

# Drop low-variance columns from model frame (catches near-constant one-hot cols)
print(f"Low-variance columns dropped from model frame (threshold >= {LOW_VARIANCE_THRESHOLD:.0%}):")
drop_low_variance(df_model, LOW_VARIANCE_THRESHOLD)

# Bool -> int
bool_cols_present = df_model.select_dtypes(include="boolean").columns.tolist()
for col in bool_cols_present:
    df_model[col] = df_model[col].astype(int)

# Min-max normalise all continuous numeric columns
#   Skip binary / one-hot columns (only 0 and 1) to preserve their scale.
numeric_cols = df_model.select_dtypes(include=[np.number]).columns.tolist()
to_normalise = [c for c in numeric_cols if df_model[c].nunique() > 2]

for col in to_normalise:
    col_min = df_model[col].min()
    col_max = df_model[col].max()
    if col_max > col_min:
        df_model[col] = (df_model[col] - col_min) / (col_max - col_min)
    else:
        df_model[col] = 0.0   # constant column -> zero-fill

print(f"Min-max normalised {len(to_normalise)} continuous columns: {to_normalise}")

# ── 12. Save & report ─────────────────────────────────────────────────────────

df.to_csv(OUTPUT, index=False)
df_model.to_csv(OUTPUT_MODEL, index=False)

print()
print("=" * 55)
print(f"Cleaned : {df.shape[0]:,} rows x {df.shape[1]} columns  ->  {OUTPUT}")
print(f"ML-ready: {df_model.shape[0]:,} rows x {df_model.shape[1]} columns  ->  {OUTPUT_MODEL}")
print()
print("Remaining nulls (cleaned):")
remaining = df.isnull().sum()
remaining = remaining[remaining > 0]
print("  None" if remaining.empty else remaining.to_string())
print()
print("ML feature columns:")
print(df_model.columns.tolist())
