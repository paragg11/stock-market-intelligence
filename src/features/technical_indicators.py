import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)

INPUT_DIR = Path("data/processed")
OUTPUT_DIR = INPUT_DIR / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)

    return rsi


def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    return macd, signal, macd - signal


def calculate_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where(
        (up_move > down_move) & (up_move > 0), 0
    )

    minus_dm = down_move.where(
        (down_move > up_move) & (down_move > 0), 0
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = 100 * (
        plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean() / atr
    )

    minus_di = 100 * (
        minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean() / atr
    )

    denominator = (plus_di + minus_di).replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / denominator

    adx = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return adx, plus_di, minus_di


def calculate_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)

    return (direction * volume).cumsum()


def calculate_mfi(df, period=14):
    typical_price = (
        df["High"] + df["Low"] + df["Close"]
    ) / 3

    money_flow = typical_price * df["Volume"]

    direction = typical_price.diff()

    positive_flow = money_flow.where(direction > 0, 0)
    negative_flow = money_flow.where(direction < 0, 0)

    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()

    ratio = positive_sum / negative_sum.replace(0, np.nan)

    mfi = 100 - (100 / (1 + ratio))

    mfi = mfi.mask(
        (negative_sum == 0) & (positive_sum > 0), 100
    )

    mfi = mfi.mask(
        (positive_sum == 0) & (negative_sum > 0), 0
    )

    return mfi


def engineer_indicators(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # -----------------------------
    # TREND INDICATORS
    # -----------------------------
    for period in [10, 20, 50, 100, 200]:
        df[f"SMA_{period}"] = close.rolling(period).mean()

    for period in [12, 26]:
        df[f"EMA_{period}"] = close.ewm(
            span=period, adjust=False
        ).mean()

    df["MACD"], df["MACD_Signal"], df["MACD_Histogram"] = (
        calculate_macd(close)
    )

    df["ADX_14"], df["PLUS_DI_14"], df["MINUS_DI_14"] = (
        calculate_adx(df)
    )

    # -----------------------------
    # MOMENTUM INDICATORS
    # -----------------------------
    df["RSI_14"] = calculate_rsi(close, 14)

    df["ROC_12"] = close.pct_change(12) * 100

    lowest_low = low.rolling(14).min()
    highest_high = high.rolling(14).max()

    price_range = (highest_high - lowest_low).replace(0, np.nan)

    df["STOCH_K"] = (
        100 * (close - lowest_low) / price_range
    )

    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    df["WILLIAMS_R"] = (
        -100 * (highest_high - close) / price_range
    )

    # -----------------------------
    # VOLATILITY INDICATORS
    # -----------------------------
    df["BB_MIDDLE"] = close.rolling(20).mean()

    bb_std = close.rolling(20).std()

    df["BB_UPPER"] = df["BB_MIDDLE"] + 2 * bb_std
    df["BB_LOWER"] = df["BB_MIDDLE"] - 2 * bb_std

    df["BB_WIDTH"] = (
        (df["BB_UPPER"] - df["BB_LOWER"])
        / df["BB_MIDDLE"]
    )

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR_14"] = true_range.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    df["Rolling_Std_20"] = df["Daily_Return"].rolling(20).std()

    # -----------------------------
    # VOLUME INDICATORS
    # -----------------------------
    df["OBV"] = calculate_obv(close, volume)

    df["Volume_MA_20"] = volume.rolling(20).mean()

    df["Volume_Change"] = volume.pct_change()

    df["MFI_14"] = calculate_mfi(df)

    # -----------------------------
    # RETURN & RISK FEATURES
    # -----------------------------
    for lag in [1, 5, 20]:
        df[f"Lag_Return_{lag}"] = df["Daily_Return"].shift(lag)

    df["Rolling_Max_20"] = close.rolling(20).max()

    df["Rolling_Drawdown_20"] = (
        close / df["Rolling_Max_20"] - 1
    )

    df["Gap_Return"] = (
        df["Open"] / previous_close - 1
    )

    df["Open_Close_Return"] = (
        df["Close"] / df["Open"] - 1
    )

    df["Close_SMA20_Ratio"] = (
        close / df["SMA_20"] - 1
    )

    df["Rolling_Volatility_20"] = (
        df["Daily_Return"].rolling(20).std()
        * np.sqrt(252)
    )

    return df


def main():
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS"
    ]

    all_features = []

    for ticker in tickers:
        input_file = INPUT_DIR / f"{ticker}_clean.csv"

        if not input_file.exists():
            raise FileNotFoundError(
                f"Cleaned data not found: {input_file}"
            )

        df = pd.read_csv(input_file)

        feature_df = engineer_indicators(df)

        feature_df["Ticker"] = ticker

        output_file = OUTPUT_DIR / f"{ticker}_features.csv"

        feature_df.to_csv(output_file, index=False)

        all_features.append(feature_df)

        logging.info(
            "%s | Rows=%s | Columns=%s",
            ticker,
            len(feature_df),
            len(feature_df.columns)
        )

    combined = pd.concat(all_features, ignore_index=True)

    combined = combined.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    combined_file = OUTPUT_DIR / "all_stocks_features.csv"

    combined.to_csv(combined_file, index=False)

    print("\nFEATURE ENGINEERING COMPLETED")
    print("Combined dataset shape:", combined.shape)
    print("Total columns:", len(combined.columns))
    print("Output:", combined_file)


if __name__ == "__main__":
    main()