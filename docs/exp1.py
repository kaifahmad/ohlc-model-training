# Code Generated by Sidekick is for learning and experimentation purposes only.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# Assume df has columns: 'Open', 'High', 'Low', 'Close', 'Volume'
df['return'] = df['Close'].pct_change().shift(-1)  # Next period return as target

# Example features
df['ma_5'] = df['Close'].rolling(5).mean()
df['volatility_5'] = df['Close'].rolling(5).std()
df['rsi_14'] = ta.rsi(df['Close'], length=14)  # Using pandas-ta

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'ma_5', 'volatility_5', 'rsi_14']
df = df.dropna()

X = df[features]
y = df['return']

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(mean_squared_error(y_test, preds))
