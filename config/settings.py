
from datetime import date
import pandas as pd

# NSE stock universe
TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
]

# Historical data window: six calendar years
END_DATE = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
START_DATE = END_DATE - pd.DateOffset(years=6)

# Data directories
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
REPORTS_DIR = "data/reports"

# Financial settings
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.06

# Prediction settings
PREDICTION_HORIZON = 1
