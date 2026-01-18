import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # 🔹 Example engineered features
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity_ratio"] = df["temperature"] / (df["humidity"] + 1e-5)

    if "precipitation" in df.columns:
        # Count dry streaks per month
        df["dry_streak"] = (
            (df["precipitation"] == 0)
            .astype(int)
            .groupby(df["date"].dt.to_period("M"))
            .cumsum()
        )

    if "temperature" in df.columns:
        df["temp_anomaly"] = (
            df["temperature"] - df["temperature"].rolling(window=7, min_periods=1).mean()
        )

    # --- Handle NaN values ---
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Fill numeric columns with 0
    df[num_cols] = df[num_cols].fillna(0)

    # Fill categorical columns with "Unknown"
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    # --- Encode categorical variables (one-hot encoding) ---
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df
