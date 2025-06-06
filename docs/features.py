import pandas as pd
import numpy as np

# Ensure datetime is datetime type and sorted
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)

# 1. Price-Based Features
df['hourly_return'] = (df['close'] - df['open']) / df['open']
df['log_return'] = np.log(df['close'] / df['open'])
df['prev_close'] = df['close'].shift(1)
df['prev_return'] = (df['close'] - df['prev_close']) / df['prev_close']
df['high_low'] = df['high'] - df['low']
df['close_open'] = df['close'] - df['open']
df['high_close'] = df['high'] - df['close']
df['low_close'] = df['low'] - df['close']

# Volatility (rolling std of returns)
for window in [3, 6, 12, 24]:
    df[f'volatility_{window}h'] = df['hourly_return'].rolling(window).std()

# Moving Averages
for window in [3, 6, 12, 24]:
    df[f'sma_{window}h'] = df['close'].rolling(window).mean()
    df[f'ema_{window}h'] = df['close'].ewm(span=window, adjust=False).mean()

# RSI (14-period)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = compute_rsi(df['close'], 14)

# Momentum
for lag in [3, 6, 12, 24]:
    df[f'momentum_{lag}h'] = df['close'] - df['close'].shift(lag)

# 2. Volume-Based Features (if available)
if 'volume' in df.columns:
    for window in [3, 6, 12, 24]:
        df[f'vol_sma_{window}h'] = df['volume'].rolling(window).mean()
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

# 3. Technical Indicators
# Bollinger Bands (20-period)
df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

# MACD
ema_12 = df['close'].ewm(span=12, adjust=False).mean()
ema_26 = df['close'].ewm(span=26, adjust=False).mean()
df['macd'] = ema_12 - ema_26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# ATR (14-period)
df['prior_close'] = df['close'].shift(1)
df['tr'] = np.maximum(df['high'] - df['low'],
                      np.maximum(abs(df['high'] - df['prior_close']),
                                 abs(df['low'] - df['prior_close'])))
df['atr_14'] = df['tr'].rolling(14).mean()

# 4. Time-Based Features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_opening_hour'] = df['hour'] == df['hour'].min()
df['is_closing_hour'] = df['hour'] == df['hour'].max()

# 5. Gap Features
df['date'] = df['datetime'].dt.date
df['day_open'] = df.groupby('date')['open'].transform('first')
df['prev_day_close'] = df.groupby('date')['close'].transform('last').shift(1)
df['gap'] = df['day_open'] - df['prev_day_close']
df['gap_pct'] = df['gap'] / df['prev_day_close'] * 100
df['gap_direction'] = np.where(df['gap'] > 0, 1, np.where(df['gap'] < 0, -1, 0))

# Gap fill indicator (if intraday price touches prev close)
def gap_fill(row):
    if pd.isna(row['prev_day_close']):
        return np.nan
    day_df = df[df['date'] == row['date']]
    filled = ((day_df['high'] >= row['prev_day_close']) & (day_df['low'] <= row['prev_day_close'])).any()
    return int(filled)

df['gap_filled'] = df.apply(gap_fill, axis=1)

# 6. Candlestick Patterns (simple example: doji)
df['doji'] = np.abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1

# 7. Rolling Max/Min
for window in [3, 6, 12, 24]:
    df[f'roll_max_{window}h'] = df['high'].rolling(window).max()
    df[f'roll_min_{window}h'] = df['low'].rolling(window).min()

# Drop helper columns if not needed
df = df.drop(['prior_close', 'tr', 'bb_std', 'date'], axis=1)

# Final DataFrame now has engineered features
