import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "SMA_10",
    "RSI_14",
    "lag_1",
    "lag_5",
    "SMA_20",
    "BB_upper",
    "BB_lower",
    "MACD",
    "MACD_Signal",
]

PRICE_COLUMN_MAP = {
    "price_open": "open",
    "price_high": "high",
    "price_low": "low",
    "price_close": "close",
    "total_volume": "volume",
}


def _normalise_ohlcv_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df = df.rename(columns={old: new for old, new in PRICE_COLUMN_MAP.items() if old in df.columns})

    missing = [column for column in ["open", "high", "low", "close", "volume"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns for feature engineering: {missing}")

    if "created_at" in df.columns:
        df = df.sort_values("created_at")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.dropna(subset=numeric_columns).reset_index(drop=True)


def _calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)
    return rsi.fillna(50)


def generate_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Generate the exact feature set expected by best_timeseries_model.h5.
    Accepts database column names or already-normalised OHLCV names.
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("Cannot generate features from an empty dataframe")

    df = _normalise_ohlcv_columns(dataframe)
    if df.empty:
        raise ValueError("No valid OHLCV rows after cleaning")

    close = df["close"]

    df["SMA_10"] = close.rolling(window=10, min_periods=10).mean()
    df["RSI_14"] = _calculate_rsi(close, period=14)
    df["lag_1"] = close.shift(1)
    df["lag_5"] = close.shift(5)
    df["SMA_20"] = close.rolling(window=20, min_periods=20).mean()

    rolling_std_20 = close.rolling(window=20, min_periods=20).std()
    df["BB_upper"] = df["SMA_20"] + (rolling_std_20 * 2)
    df["BB_lower"] = df["SMA_20"] - (rolling_std_20 * 2)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df = df.dropna(subset=REQUIRED_FEATURES).reset_index(drop=True)
    logger.debug("Generated %s feature rows", len(df))
    return df


def validate_feature_rows(dataframe: pd.DataFrame, required_rows: int = 10) -> None:
    if dataframe is None or dataframe.empty:
        raise ValueError("Feature dataframe is empty")

    missing = [column for column in REQUIRED_FEATURES if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")

    if len(dataframe) < required_rows:
        raise ValueError(f"Need at least {required_rows} feature rows, got {len(dataframe)}")
