import pandas as pd

CSV_PATH = "./dataset/raw_dataset.csv"
OUT_PATH = "./dataset/clean_fire.csv"

# 1) The seven inputs the model is allowed to see.
# Every one of them is a physical measurement. This is the lean set that
# won the repeated cross-validation ablation: accuracy 0.5780 and
# ROC-AUC 0.5955, against 0.5706 / 0.5802 for all 22 columns.
FEATURES = [
    "X",
    "Y",
    "DMC",
    "DC",
    "temp",
    "RH",
    "wind",
]

# 2) Columns that exist in the raw file but are not used as inputs.
# Each reason below is a measurement, not a preference.
NOT_USED = [
    (
        "month",
        "11 one-hot columns, 8 of them at least 96% zero",
    ),
    (
        "day",
        "became is_weekend; removing it moved CV accuracy 0.5706 -> 0.5700",
    ),
    (
        "rain",
        "505 of 513 rows are 0.0 (98.4%), so the flag is near-constant",
    ),
    (
        "FFMC",
        "permutation importance 0.0007, the weakest feature in the model",
    ),
    (
        "ISI",
        "permutation importance 0.006; its 0-56.1 slider was 64% empty",
    ),
    (
        "area",
        "the raw target - keeping it would leak the answer",
    ),
]

# 3) Load dataset
df = pd.read_csv(CSV_PATH)

print("\n=== INITIAL HEAD ===")
print(df.head())

print("\n=== INITIAL INFO ===")
df.info()

print("\n=== INITIAL MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== INITIAL TARGET (area) ===")
print(df["area"].describe())

# 4) Remove duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"\nDuplicates removed: {before - after}")
print(f"Rows remaining    : {after}")

# 5) Outlier check
# The dataset notes that the number of outliers is unclear. Both kinds
# are reported here and kept: extreme fire weather is the signal, not
# a recording error. Removing 1.5 x IQR rows dropped ROC-AUC from
# 0.579 to 0.546, so removal would delete the very pattern we model.
print("\n=== TARGET OUTLIERS (area) ===")
print(f"Fires over 100 ha : {(df['area'] > 100).sum()}")
print(f"Largest fire      : {df['area'].max()} ha")
print(f"Median fire       : {df['area'].median()} ha")

print("\n=== FEATURE OUTLIER CHECK (1.5 x IQR) ===")
outlier_rows = pd.Series(False, index=df.index)

for col in FEATURES:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    outside = (df[col] < low) | (df[col] > high)
    outlier_rows = outlier_rows | outside

    print(
        f"{col:>5}: {outside.sum():>3} rows "
        f"outside [{low:.2f}, {high:.2f}]"
    )

print(
    f"Rows with at least one outlier: {outlier_rows.sum()} "
    f"({outlier_rows.mean():.1%})"
)
print("Kept on purpose: these are real fires, not bad measurements.")

# 6) Build the binary target
# area = 0 means the fire did not spread beyond a trivial burn.
df["fire_occurred"] = (df["area"] > 0).astype(int)

print("\n=== FIRE OCCURRED DISTRIBUTION ===")
print(df["fire_occurred"].value_counts())
print(f"Share that spread: {df['fire_occurred'].mean():.2%}")

# 7) Keep only the seven features plus the target
# This is the column removal step: it drops month, day, rain, area,
# FFMC and ISI in one place instead of scattering the decision.
print("\n=== COLUMNS NOT USED AS INPUTS ===")
for name, reason in NOT_USED:
    print(f"{name:>5}: {reason}")

dropped = [c for c in df.columns if c not in FEATURES + ["fire_occurred"]]
print(f"\nDropped {len(dropped)} columns: {', '.join(dropped)}")

df = df[FEATURES + ["fire_occurred"]]

# 8) Final snapshot
print("\n=== FINAL HEAD ===")
print(df.head())

print("\n=== FINAL COLUMNS ===")
print(df.columns.tolist())

print("\n=== FINAL MISSING VALUES ===")
print(df.isnull().sum().sum())

print("\n=== FINAL TARGET DISTRIBUTION ===")
print(df["fire_occurred"].value_counts())

# 9) Save clean dataset
df.to_csv(OUT_PATH, index=False)

print(f"\nSaved clean dataset to: {OUT_PATH}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
