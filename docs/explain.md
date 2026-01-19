# ProTrader AI - Complete Technical Documentation

## Overview

ProTrader AI is a professional-grade stock analytics platform for Indian equities (NSE). It combines **ML prediction**, **sentiment analysis**, **technical indicators**, and **institutional flow data** to generate trading signals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  Sidebar Inputs → Data Fetching → Model Training → Visualization│
└─────────────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
    ┌─────────┐        ┌───────────┐        ┌─────────────┐
    │  Data   │        │   Models  │        │     UI      │
    │ Sources │        │           │        │ Components  │
    └─────────┘        └───────────┘        └─────────────┘
```

---

## Input Flow

### 1. User Inputs (Sidebar)

| Input | Default | Purpose |
|-------|---------|---------|
| Stock | RELIANCE | NSE stock symbol |
| Start Date | 2020-01-01 | Historical data start (5 years) |
| End Date | Today | Historical data end |
| Forecast Days | 10 | Days to predict ahead |
| Dynamic Fusion | ✓ | Enable adaptive model weighting |
| AutoML | ✗ | Hyperparameter tuning (slow) |

### 2. Loading Sequence

When user clicks **"Launch Analysis"**:

```
1. Progress bar created (0%)
2. Fetch stock OHLCV data from yfinance (10%)
3. Fetch fundamentals (P/E, Market Cap, etc.) (20%)
4. Fetch news articles (30%)
5. Fetch FII/DII institutional data (40%)
6. Fetch India VIX data (50%)
7. Run multi-source sentiment analysis (60%)
8. Detect chart patterns (70%)
9. Train hybrid model (80%)
10. Generate predictions (90%)
11. Complete (100%)
```

---

## Data Sources

### Stock Price Data
- **Source**: `yfinance` library
- **Ticker Format**: `{SYMBOL}.NS` (e.g., `RELIANCE.NS`)
- **Data Retrieved**: Open, High, Low, Close, Volume (OHLCV)
- **File**: `data/stock_data.py`

### Fundamentals
- **Source**: `yfinance` `.info` property
- **Fields**: Market Cap, P/E, Forward P/E, PEG Ratio, Dividend Yield
- **File**: `data/stock_data.py`

### FII/DII Data (Institutional Flows)

**Fallback Chain:**
```
1. NSE API (https://www.nseindia.com/api/fiidiiTradeReact)
   ↓ (if fails)
2. nselib library (pip install nselib)
   ↓ (if fails)
3. MoneyControl API
   ↓ (if fails)
4. Trendlyne API
   ↓ (if fails)
5. Manual User Input (paste JSON from browser)
   ↓ (if parse fails)
6. Gemini AI Parser (extracts data from raw text)
```

**File**: `data/fii_dii.py`

**Output Columns**:
- `FII_Buy_Value`, `FII_Sell_Value`, `FII_Net`
- `DII_Buy_Value`, `DII_Sell_Value`, `DII_Net`
- `FII_Cumulative`, `DII_Cumulative`

### India VIX (Volatility Index)
- **Source**: `yfinance` with ticker `^INDIAVIX`
- **Purpose**: Market fear indicator
- **File**: `data/vix_data.py`

### News & Sentiment

**Multi-Source Sentiment System** (`data/multi_sentiment.py`):

| Source | Weight | Method |
|--------|--------|--------|
| RSS Feeds (MoneyControl, ET, LiveMint) | 30% | feedparser |
| NewsAPI | 25% | REST API |
| Reddit (Indian market subs) | 25% | PRAW |
| Google Trends | 20% | pytrends |

**Sentiment Model**: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`

**Output**: Combined sentiment score (-1 to +1), confidence (0-1)

---

## Feature Engineering

**File**: `models/hybrid_model.py` → `create_advanced_features()`

### 14 Features Used in Model:

| Category | Feature | Formula |
|----------|---------|---------|
| **Price** | `Log_Ret` | log(Close_t / Close_t-1) |
| | `Volatility_5D` | 5-day rolling std of returns |
| | `RSI_Norm` | RSI / 100 |
| | `Vol_Ratio` | Volume / 20-day avg volume |
| | `MA_Div` | (Close - MA20) / Close |
| **Sentiment** | `Sentiment` | Basic news sentiment |
| | `Multi_Sentiment` | Combined multi-source sentiment |
| | `Sentiment_Confidence` | Confidence score |
| **Institutional** | `FII_Net_Norm` | Normalized FII net flow |
| | `DII_Net_Norm` | Normalized DII net flow |
| | `FII_5D_Avg` | 5-day rolling avg FII |
| | `DII_5D_Avg` | 5-day rolling avg DII |
| **VIX** | `VIX_Norm` | (VIX - 15) / 25 |
| | `VIX_Change` | VIX percentage change |

---

## ML Models

### 1. XGBoost (Primary)
```python
xgb.XGBRegressor(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8
)
```
**Weight**: ~50%

### 2. LSTM + GRU Parallel Network
```
Input (1, 14 features)
    ├── LSTM Branch: 64 → 32 units
    └── GRU Branch: 64 → 32 units
         ↓
    Concatenate → Dense(32) → Dense(16) → Output(1)
```
**Weight**: ~30%

### 3. ARIMA (Statistical)
```python
ARIMA(order=(2, 0, 2))
```
**Purpose**: Short-term pattern capture

### 4. Prophet (Statistical)
```python
Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True
)
```
**Purpose**: Trend + seasonality

**Combined Weight** (ARIMA + Prophet): ~20%

---

## Ensemble Logic

```
hybrid_pred = (xgb_weight × XGB_pred) + 
              (rnn_weight × LSTM_GRU_pred) + 
              (stat_weight × avg(ARIMA_pred, Prophet_pred))
```

Weights are **dynamically adjusted** based on individual model RMSE on test data.

### Variance Scaling
Predictions are scaled to match actual return variance:
```python
scale_factor = actual_std / pred_std
hybrid_pred = hybrid_pred * scale_factor
```

---

## Pattern Detection

**File**: `models/visual_analyst.py`

### Patterns Detected:
- Double Top / Double Bottom
- Head & Shoulders / Inverse H&S
- Support & Resistance Levels
- Trend Direction

**Algorithm**:
1. Find peaks/troughs using `scipy.signal.find_peaks`
2. Validate with strict criteria (price tolerance, time gaps)
3. Assign confidence scores (>70% to include)

---

## Technical Indicators

**File**: `utils/technical_indicators.py`

| Indicator | Parameters |
|-----------|------------|
| SMA | 20, 50 periods |
| EMA | 12, 26 periods |
| RSI | 14 periods |
| MACD | 12, 26, 9 |
| Bollinger Bands | 20, 2σ |
| ATR | 14 periods |
| Stochastic | 14, 3, 3 |
| OBV | - |

---

## AI Analysis (Gemini)

**File**: `ui/ai_analysis.py`

### Flow:
```
1. Collect all data (price, predictions, sentiment, patterns, FII/DII)
2. Format into structured prompt
3. Send to Gemini 2.0 Flash
4. Parse markdown response
5. Display with st.markdown()
```

### Output Format:
- 🎯 **Verdict**: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
- 📊 **Key Points**: AI signal, technical setup, institutional flow
- ⚠️ **Risks**: Top 2 risks
- 🎯 **Trade Setup**: Entry, Target, Stop Loss

---

## Visualization

**File**: `ui/charts.py`

### Charts:
1. **Candlestick/Line Chart** - Main price action
2. **Model Accuracy Chart** - Actual vs Predicted prices
3. **Predicted vs Actual Returns** - Raw model output comparison
4. **FII/DII Flow Chart** - Institutional activity
5. **Technical Indicator Overlays** - MA, Bollinger, etc.

### Prediction Visualization Fix:
Predictions are **anchored** to actual prices:
```python
predicted_price = prev_ACTUAL_price × exp(predicted_return)
```
This prevents cumulative divergence in the chart.

---

## Risk Management

**File**: `utils/risk_manager.py`

### Stop Loss Calculation:
```
SL = Current Price - (ATR × Multiplier)
```
- High confidence: Multiplier = 2.0
- Low confidence: Multiplier = 1.5

### Position Sizing:
Based on volatility and account risk tolerance.

---

## Backtesting

**File**: `models/backtester.py`

### Vectorized Backtester:
- Takes predicted returns + actual prices
- Applies transaction costs
- Calculates: Total Return, Sharpe Ratio, Max Drawdown, Win Rate

---

## Configuration

**File**: `config/settings.py`

```python
class ModelConfig:
    LOOKBACK_PERIOD = 30
    TRAIN_TEST_SPLIT = 0.8
    XGB_N_ESTIMATORS = 100
    GRU_UNITS = 128, 64, 32
    DEFAULT_EPOCHS = 50

class DataConfig:
    FII_DII_CACHE_TTL = 3600  # 1 hour cache
    NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"
```

---

## Environment Variables (.env)

```
GEMINI_API_KEY=xxx       # For AI analysis
NEWS_API_KEY=xxx         # For NewsAPI
REDDIT_CLIENT_ID=xxx     # For Reddit sentiment
REDDIT_CLIENT_SECRET=xxx
```

---

## File Structure

```
finance/
├── app.py                    # Main Streamlit app
├── config/
│   └── settings.py           # Configuration constants
├── data/
│   ├── stock_data.py         # Stock price fetching
│   ├── fii_dii.py            # FII/DII institutional data
│   ├── vix_data.py           # India VIX fetching
│   ├── news_sentiment.py     # Basic news sentiment
│   └── multi_sentiment.py    # Multi-source sentiment
├── models/
│   ├── hybrid_model.py       # XGBoost + LSTM + GRU + ARIMA + Prophet
│   ├── visual_analyst.py     # Chart pattern detection
│   └── backtester.py         # Vectorized backtesting
├── ui/
│   ├── charts.py             # Plotly chart functions
│   └── ai_analysis.py        # Gemini AI integration
├── utils/
│   ├── technical_indicators.py
│   └── risk_manager.py
├── docs/
│   ├── explain.md            # This file
│   └── accuracy_tips.md      # Tips to improve accuracy
└── requirements.txt
```

---

## Output Summary

After analysis completes, user sees:

| Tab | Content |
|-----|---------|
| **Dashboard** | Price chart, key metrics, AI analysis |
| **Dynamic Fusion** | Model weight allocation, feature importance |
| **Technicals & Risk** | Indicators, support/resistance, stop loss |
| **Fundamentals** | P/E, Market Cap, valuation ratios |
| **FII/DII Analysis** | Institutional flow charts |
| **Multi-Source Sentiment** | Breakdown by source (RSS, News, Reddit, Trends) |
| **Backtest** | Historical strategy performance |
| **Pattern Analysis** | Detected chart patterns with confidence |

---

## Accuracy Metrics

| Metric | Typical Range |
|--------|---------------|
| Directional Accuracy | 55-65% |
| RMSE | 0.015-0.025 |
| Sharpe Ratio | 0.8-1.5 |
