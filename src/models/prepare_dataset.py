
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)

INPUT_FILE = Path("data/processed/features/all_stocks_features.csv")
OUTPUT_DIR = Path("data/processed/modeling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5


def prepare_dataset():
    """Prepare leakage-safe dataset for 5-day stock direction prediction."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    # Future closing price within each stock
    df["Future_Close_5D"] = (
        df.groupby("Ticker")["Close"].shift(-HORIZON)
    )

    # Future return: used to generate the target
    df["Forward_Return_5D"] = (
        df["Future_Close_5D"] / df["Close"] - 1
    )

    # Classification target
    df["Target_5D"] = (
        df["Forward_Return_5D"] > 0
    ).astype("int")

    # Last 5 rows of each ticker have no future target
    df = df.dropna(
        subset=["Future_Close_5D", "Forward_Return_5D"]
    )

    # Metadata and future information must not be model features
    excluded_columns = [
        "Date",
        "Ticker",
        "Future_Close_5D",
        "Forward_Return_5D",
        "Target_5D"
    ]

    feature_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col not in excluded_columns
    ]

    # Remove rows with missing feature values
    df = df.dropna(subset=feature_columns).copy()

    # Save modeling dataset
    output_file = OUTPUT_DIR / "stock_prediction_dataset.csv"

    df.to_csv(output_file, index=False)

    print("\n========== DATASET PREPARATION COMPLETE ==========")
    print("Dataset shape:", df.shape)
    print("Stocks:", df["Ticker"].nunique())
    print("Features:", len(feature_columns))
    print("Target distribution:")
    print(df["Target_5D"].value_counts())
    print("\nTarget percentage:")
    print(df["Target_5D"].value_counts(normalize=True) * 100)

    print("\nFeature columns:")
    print(feature_columns)

    print("\nSaved:", output_file)

    return df, feature_columns


if __name__ == "__main__":
    prepare_dataset()
