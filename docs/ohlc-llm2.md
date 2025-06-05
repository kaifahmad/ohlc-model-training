# Machine Learning for Option Sellers: Predicting "Quiet" Bars for Short Volatility Strategies

Machine-learning models can be repurposed for an option-seller’s objective: identifying 60-minute bars after which the underlying will remain “quiet” for the next two bars (≈120 minutes), allowing you to collect theta (time-decay) on short straddles/strangles or iron-condors.

The task is best framed as a probabilistic, volatility-oriented classification/regression problem rather than a pure price-direction forecast.

---

## 1. Define the Trading Question in Data Science Terms

**Trading Objective:**  
Earn premium when the underlying stays inside a “small” range for 2 hours.

**Data-Science Formulation:**  
Predict: Will the realised range (or absolute return) over the next 2×60-min bars stay below a threshold T?

### 1.1 Target Variable (“y”)

**Compute forward window statistics:**
- `fwd_high = High.shift(-1).rolling(2).max()`
- `fwd_low  = Low.shift(-1).rolling(2).min()`
- `fwd_return = (Close.shift(-2) / Close) - 1`

**Choose a threshold T** that reflects how far the underlying can move before the sold strikes are breached. Typical choices:
- Fixed % of spot (e.g., ±0.4%)
- A multiple of recent ATR/σ (e.g., 0.7 × ATR20)

**Create labels:**
- **Range-bound (1):** `(fwd_high - Close)/Close < T` AND `(Close - fwd_low)/Close < T`
- **Breakout (0):** otherwise

> The model outputs P(range-bound | current info).

*Alternative: regress realised range and later transform into a probability.*

---

## 2. Feature Engineering (X)

The model must sense current compression, absence of momentum, liquidity, and external shocks. Below is a MECE-style list; use 30–50 well-curated features to start.

### 2.1 Price / Volatility
- ATR(5,10,20)
- Rolling σ of returns (5–20 bars)
- Parkinson or Garman-Klass volatility (High/Low based)
- Bollinger-band width, Keltner-channel width (% of price)
- Range ratio = (High–Low)/(ATR20) of current bar
- ADX, CCI, RSI (low ADX & neutral RSI suggest consolidation)

### 2.2 Momentum & Mean Reversion
- MACD histogram
- Z-score of price vs SMA(10,20)
- Lagged returns (1,2,3 bars)

### 2.3 Order–flow / Volume
- OBV slope, VWAP deviation
- Volume % of 20-bar average
- Bid-ask spread if available

### 2.4 Calendar / Microstructure
- Hour-of-day, Day-of-week (mid-session often quieter)
- Pre/post macro-announcement flags

### 2.5 Cross-Asset / Macro (optional)
- Spot VIX, IV percentile of the option chain
- Major index futures returns (e.g., ES, DXY, Crude)
- News or Twitter sentiment scores

*Assumption: implied-vol data are available; if not, rely on realised vol & VIX proxies.*

---

## 3. Algorithm Choice

Start with interpretable, robust models, then experiment with deep learning once a baseline edge is proven.

- **Gradient-Boosted Trees (XGBoost / LightGBM):** Handle non-linearities, interactions, missing data
- **Logistic / Elastic-Net:** For baseline and feature importance
- **Temporal CNN or LSTM:** For longer sequences (feed 10–20 bars of raw OHLCV)
- **Ensemble / Stacking:** (e.g., logistic meta-model on top of tree + LSTM outputs)

---

## 4. Training & Validation Pipeline

- **Data Split:** Walk-forward (expanding) or rolling window
- **Cross-Sectional Balance:** Range-bound may dominate; use `scale_pos_weight`, focal loss, or SMOTE-Tomek
- **Hyper-parameter Search:** Optuna / Bayesian optimisation with time-series CV
- **Metrics:**
  - AUC-ROC, F1 (classification)
  - Economic: expected P&L, hit-ratio vs. premium sold, drawdown
- **Feature Scaling & Leakage:** Standardise only with past data; shift targets by +2 bars

---

## 5. Accounting for Volatility & Regime Shifts

- **Target Normalisation:** Predict (future_range / current_ATR) → generalises across regimes
- **Regime Features:**
  - Hidden-Markov model state (high vs. low vol)
  - Rolling ADX quantile
- **Online Re-training:** Re-fit weekly/daily; decay old samples via time-based weighting
- **Ensemble across Horizons:** Separate models for intraday, post-lunch, pre-close

---

## 6. Example Skeleton (Tree Model)
https://github.com/kaifahmad/ohlc-model-training/blob/main/docs/exp1.py
