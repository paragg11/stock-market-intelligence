
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from config.settings import (
    TICKERS,
    START_DATE,
    END_DATE,
    RAW_DATA_DIR,
)

Path(RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def download_stock_data(ticker: str):

    logging.info("Downloading %s", ticker)

    try:
        df = yf.download(
            ticker,
            start=START_DATE.strftime("%Y-%m-%d"),
            end=END_DATE.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )

        if df.empty:
            logging.warning("No data returned for %s", ticker)
            return None

        # Handle Yahoo Finance MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Standardize dates
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df["Date"] = df["Date"].dt.tz_localize(None)

        df["Ticker"] = ticker

        required_columns = [
            "Date", "Open", "High", "Low",
            "Close", "Adj Close", "Volume", "Ticker"
        ]

        missing = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df[required_columns]
        df = df.sort_values("Date").reset_index(drop=True)

        output_file = Path(RAW_DATA_DIR) / f"{ticker}.csv"
        df.to_csv(output_file, index=False)

        logging.info(
            "Downloaded %s successfully: %d rows",
            ticker,
            len(df)
        )

        return df

    except Exception:
        logging.exception("Failed downloading %s", ticker)
        return None


def main():

    all_stocks = []

    for ticker in TICKERS:

        stock_df = download_stock_data(ticker)

        if stock_df is not None:
            all_stocks.append(stock_df)

    if not all_stocks:
        raise RuntimeError("No stock data downloaded.")

    combined_df = pd.concat(
        all_stocks,
        ignore_index=True
    )

    combined_df = combined_df.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    output_file = Path(RAW_DATA_DIR) / "all_stocks_raw.csv"

    combined_df.to_csv(output_file, index=False)

    print("\n========== DOWNLOAD SUMMARY ==========")

    print(
        combined_df.groupby("Ticker").agg(
            observations=("Date", "count"),
            first_date=("Date", "min"),
            last_date=("Date", "max")
        )
    )

    print("\nTotal observations:", len(combined_df))
    print("Saved:", output_file)


if __name__ == "__main__":
    main()
