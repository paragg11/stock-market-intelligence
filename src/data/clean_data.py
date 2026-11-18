
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TRADING_DAYS_PER_YEAR,
)

# Create output directories
Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def clean_stock_data(file_path):
    """
    Clean and validate one stock's historical data.
    Calculate daily returns and rolling volatility.
    """

    df = pd.read_csv(file_path)

    ticker = Path(file_path).stem

    logging.info("Cleaning %s", ticker)

    original_rows = len(df)

    df["Date"] = pd.to_datetime(df["Date"])

    # 1. Sort data chronologically
    df = df.sort_values("Date").reset_index(drop=True)

    # 2. Detect duplicate dates
    duplicate_dates = int(
        df["Date"].duplicated().sum()
    )

    df = df.drop_duplicates(
        subset=["Date"],
        keep="last"
    )

    # 3. Detect missing values before cleaning
    missing_values = int(df.isnull().sum().sum())

    # 4. Identify invalid price records
    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
    ]

    invalid_prices = int(
        (df[price_columns] <= 0).any(axis=1).sum()
    )

    # 5. Identify invalid volume
    invalid_volume = int(
        (df["Volume"] < 0).sum()
    )

    # 6. Validate OHLC consistency
    invalid_ohlc = int(
        (
            (df["High"] < df["Low"])
            |
            (df["High"] < df[["Open", "Close"]].max(axis=1))
            |
            (df["Low"] > df[["Open", "Close"]].min(axis=1))
        ).sum()
    )

    # 7. Remove missing and invalid observations
    df = df.dropna(
        subset=price_columns + ["Volume"]
    )

    valid_prices = (
        (df[price_columns] > 0).all(axis=1)
    )

    valid_volume = df["Volume"] >= 0

    valid_ohlc = (
        (df["High"] >= df[["Open", "Close", "Low"]].max(axis=1))
        &
        (df["Low"] <= df[["Open", "Close"]].min(axis=1))
    )

    df = df[
        valid_prices & valid_volume & valid_ohlc
    ].copy()

    df = df.sort_values("Date").reset_index(drop=True)

    # 8. Calculate simple daily returns
    df["Daily_Return"] = (
        df["Adj Close"].pct_change()
    )

    # 9. Calculate logarithmic returns
    df["Log_Return"] = np.log(
        df["Adj Close"] / df["Adj Close"].shift(1)
    )

    # 10. Calculate annualized 20-day rolling volatility
    df["Rolling_Volatility_20"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
        * np.sqrt(TRADING_DAYS_PER_YEAR)
    )

    # 11. Save cleaned stock dataset
    output_file = (
        Path(PROCESSED_DATA_DIR)
        / f"{ticker}_clean.csv"
    )

    df.to_csv(output_file, index=False)

    # 12. Generate quality report
    report = {
        "Ticker": ticker,
        "Original_Rows": original_rows,
        "Final_Rows": len(df),
        "Removed_Rows": original_rows - len(df),
        "Duplicate_Dates": duplicate_dates,
        "Missing_Values_Before_Cleaning": missing_values,
        "Invalid_Prices": invalid_prices,
        "Invalid_Volume": invalid_volume,
        "Invalid_OHLC": invalid_ohlc,
        "First_Date": str(df["Date"].min().date()),
        "Last_Date": str(df["Date"].max().date()),
    }

    return df, report


def main():

    all_stocks = []
    quality_reports = []

    raw_directory = Path(RAW_DATA_DIR)

    # Process individual stock CSV files only
    stock_files = sorted(
        raw_directory.glob("*.NS.csv")
    )

    if not stock_files:
        raise FileNotFoundError(
            "No NSE stock CSV files found in data/raw/"
        )

    for file_path in stock_files:

        clean_df, report = clean_stock_data(file_path)

        all_stocks.append(clean_df)
        quality_reports.append(report)

        logging.info(
            "Processed %s: %d rows",
            report["Ticker"],
            report["Final_Rows"],
        )

    # Combine cleaned datasets
    combined_df = pd.concat(
        all_stocks,
        ignore_index=True
    )

    combined_df = combined_df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    combined_file = (
        Path(PROCESSED_DATA_DIR)
        / "combined_stocks.csv"
    )

    combined_df.to_csv(
        combined_file,
        index=False
    )

    # Save quality report
    report_df = pd.DataFrame(quality_reports)

    report_file = (
        Path(REPORTS_DIR)
        / "data_quality_report.csv"
    )

    report_df.to_csv(
        report_file,
        index=False
    )

    print("\n========== DATA CLEANING SUMMARY ==========")

    print(report_df.to_string(index=False))

    print("\nCombined dataset shape:", combined_df.shape)

    print("\nCleaned dataset:", combined_file)
    print("Quality report:", report_file)


if __name__ == "__main__":
    main()
