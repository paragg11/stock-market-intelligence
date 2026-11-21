
import json
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/processed/modeling/stock_prediction_dataset.csv")
OUTPUT_DIR = Path("data/processed/modeling")

TRAIN_RATIO = 0.80


def chronological_split(df, train_ratio=0.80):
    """Split each stock chronologically into train and test datasets."""

    train_parts = []
    test_parts = []

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Ticker", "Date"])

    for ticker, stock_df in df.groupby("Ticker"):

        stock_df = stock_df.sort_values("Date").reset_index(drop=True)

        split_index = int(len(stock_df) * train_ratio)

        train_stock = stock_df.iloc[:split_index].copy()
        test_stock = stock_df.iloc[split_index:].copy()

        train_parts.append(train_stock)
        test_parts.append(test_stock)

        print(f"\n{ticker}")
        print(f"Total observations: {len(stock_df)}")
        print(f"Training observations: {len(train_stock)}")
        print(f"Testing observations: {len(test_stock)}")
        print(f"Training period: {train_stock['Date'].min()} to {train_stock['Date'].max()}")
        print(f"Testing period: {test_stock['Date'].min()} to {test_stock['Date'].max()}")

        # Leakage validation
        assert train_stock["Date"].max() < test_stock["Date"].min(), (
            f"Temporal leakage detected for {ticker}"
        )

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    train_df, test_df = chronological_split(
        df,
        train_ratio=TRAIN_RATIO
    )

    # Save datasets
    train_file = OUTPUT_DIR / "train_dataset.csv"
    test_file = OUTPUT_DIR / "test_dataset.csv"

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    # Save metadata
    metadata = {
        "split_type": "chronological",
        "train_ratio": TRAIN_RATIO,
        "test_ratio": 1 - TRAIN_RATIO,
        "stocks": sorted(df["Ticker"].unique().tolist()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_start": str(train_df["Date"].min()),
        "train_end": str(train_df["Date"].max()),
        "test_start": str(test_df["Date"].min()),
        "test_end": str(test_df["Date"].max()),
    }

    metadata_file = OUTPUT_DIR / "split_metadata.json"

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n========== TIME-BASED SPLIT COMPLETE ==========")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Train ratio:", TRAIN_RATIO)
    print("Test ratio:", 1 - TRAIN_RATIO)
    print("\nSaved:")
    print(train_file)
    print(test_file)
    print(metadata_file)


if __name__ == "__main__":
    main()
