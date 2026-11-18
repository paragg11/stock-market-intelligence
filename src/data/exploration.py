
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from config.settings import PROCESSED_DATA_DIR, REPORTS_DIR

CHART_DIR = Path(REPORTS_DIR) / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)


def visualize_stock(ticker):

    file_path = Path(PROCESSED_DATA_DIR) / f"{ticker}_clean.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])

    fig, axes = plt.subplots(
        4, 1,
        figsize=(15, 14),
        sharex=True
    )

    # Adjusted closing price
    axes[0].plot(df["Date"], df["Adj Close"])
    axes[0].set_title(f"{ticker} - Adjusted Closing Price")
    axes[0].set_ylabel("Price (INR)")
    axes[0].grid(alpha=0.3)

    # Daily returns
    axes[1].plot(df["Date"], df["Daily_Return"])
    axes[1].set_title("Daily Returns")
    axes[1].set_ylabel("Return")
    axes[1].grid(alpha=0.3)

    # Rolling volatility
    axes[2].plot(df["Date"], df["Rolling_Volatility_20"])
    axes[2].set_title("20-Day Annualized Rolling Volatility")
    axes[2].set_ylabel("Volatility")
    axes[2].grid(alpha=0.3)

    # Trading volume
    axes[3].plot(df["Date"], df["Volume"])
    axes[3].set_title("Daily Trading Volume")
    axes[3].set_ylabel("Shares")
    axes[3].set_xlabel("Date")
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"Stock Market Analysis: {ticker}", fontsize=16)

    plt.tight_layout()

    output_file = CHART_DIR / f"{ticker}_analysis.png"

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logging.info("Chart saved: %s", output_file)


def main():

    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS"
    ]

    for ticker in tickers:
        visualize_stock(ticker)

    print("\nEDA COMPLETED SUCCESSFULLY")
    print("Charts saved in:", CHART_DIR)


if __name__ == "__main__":
    main()
