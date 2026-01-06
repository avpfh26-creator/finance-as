import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import re
import os
import xgboost as xgb
import shap
import plotly.graph_objects as go
from transformers import pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for server
import matplotlib.pyplot as plt
import cv2
try:
    from ultralytics import YOLO
    VISUAL_AI_AVAILABLE = True
except ImportError:
    VISUAL_AI_AVAILABLE = False
import tempfile
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, LSTM, Concatenate, BatchNormalization
# For TensorFlow 2.x (modern versions)
from tensorflow.keras.optimizers import Adam
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import CustomBusinessDay
import warnings
warnings.filterwarnings('ignore')

# Enhanced Imports
try:
    import optuna
except ImportError:
    optuna = None
import scipy.stats as stats

# Google Gemini API Integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =============================================
# GEMINI API CONFIGURATION
# =============================================
# Add your Gemini API key here or set as environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDyXJiIG64HQXzu6T2nSEzw76sNEbcCaBk")

def initialize_gemini():
    """Initialize Gemini API with safety settings"""
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("⚠️ google-generativeai library not installed")
        return None
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
        st.sidebar.warning("⚠️ No Gemini API key configured")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use gemini-2.5-flash - the stable text model
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.sidebar.error(f"Gemini init error: {e}")
        return None

def generate_gemini_analysis(stock_symbol, current_price, predicted_prices, metrics, 
                             fundamentals, sentiment_summary, technical_indicators, 
                             volatility_data, fusion_weights=None):
    """
    Generate comprehensive AI analysis using Google Gemini with expert-level prompt.
    Returns structured analysis with actionable insights.
    """
    model = initialize_gemini()
    if model is None:
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices, 
                                          metrics, sentiment_summary, technical_indicators)
    
    # Calculate key metrics for prompt
    price_forecast_end = predicted_prices['Predicted Price'].iloc[-1] if not predicted_prices.empty else current_price
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    forecast_days = len(predicted_prices) if not predicted_prices.empty else 0
    
    # Prepare sentiment summary
    sentiment_text = "Neutral (No recent news)"
    if sentiment_summary:
        positive_count = sum(1 for s in sentiment_summary.values() for label, _ in s if label == 'positive')
        negative_count = sum(1 for s in sentiment_summary.values() for label, _ in s if label == 'negative')
        total = positive_count + negative_count
        if total > 0:
            sentiment_ratio = positive_count / total
            if sentiment_ratio > 0.6:
                sentiment_text = f"Bullish ({positive_count}/{total} positive articles)"
            elif sentiment_ratio < 0.4:
                sentiment_text = f"Bearish ({negative_count}/{total} negative articles)"
            else:
                sentiment_text = f"Mixed ({positive_count} positive, {negative_count} negative)"
    
    # Fusion weights summary
    fusion_text = "Not available"
    if fusion_weights:
        fusion_text = f"Technical: {fusion_weights.get('technical', 0)*100:.1f}%, Sentiment: {fusion_weights.get('sentiment', 0)*100:.1f}%, Volatility: {fusion_weights.get('volatility', 0)*100:.1f}%"
    
    # Expert prompt engineering
    prompt = f"""
# EXPERT STOCK ANALYSIS REQUEST

You are a senior quantitative analyst at a top-tier investment bank with 20+ years of experience in Indian equity markets. Provide a comprehensive, actionable analysis for a sophisticated retail investor.

## STOCK DATA: {stock_symbol} (NSE)

### Current Market Data:
- **Current Price:** ₹{current_price:,.2f}
- **Today's Change:** Included in price action

### AI Model Predictions:
- **{forecast_days}-Day Price Forecast:** ₹{price_forecast_end:,.2f}
- **Predicted Return:** {forecast_return:+.2f}%
- **Model Directional Accuracy:** {metrics.get('accuracy', 0):.1f}% (on out-of-sample test data)
- **Prediction RMSE:** {metrics.get('rmse', 0):.4f}

### Technical Indicators:
- **RSI (14):** {technical_indicators.get('RSI', 'N/A')}
- **5-Day Volatility:** {technical_indicators.get('Volatility_5D', 0)*100:.2f}%
- **20-Day Volatility:** {technical_indicators.get('Volatility_20D', 0)*100:.2f}%
- **Price vs 20-MA:** {technical_indicators.get('Price_vs_MA20', 0)*100:+.2f}%
- **MACD Histogram:** {technical_indicators.get('MACD_Histogram', 'N/A')}

### Sentiment Analysis (FinBERT NLP):
- **News Sentiment:** {sentiment_text}

### Fundamental Data:
- **Forward P/E:** {fundamentals.get('Forward P/E', 'N/A')}
- **PEG Ratio:** {fundamentals.get('PEG Ratio', 'N/A')}
- **ROE:** {fundamentals.get('ROE', 'N/A')}
- **Debt/Equity:** {fundamentals.get('Debt/Equity', 'N/A')}

### Dynamic Fusion Model Weights:
{fusion_text}

---

## REQUIRED OUTPUT FORMAT (Be concise, max 300 words total):

### 🎯 VERDICT
[One of: STRONG BUY 🟢 | BUY 🟢 | HOLD 🟡 | SELL 🔴 | STRONG SELL 🔴]

### 📊 OUTLOOK
- **Short-term (1-5 days):** [Bullish/Bearish/Neutral + 1 sentence why]
- **Medium-term (1-4 weeks):** [Bullish/Bearish/Neutral + 1 sentence why]

### 💡 KEY INSIGHT
[Single most important factor driving this recommendation - 2 sentences max]

### ⚠️ RISK FACTORS
[2-3 bullet points of key risks to monitor]

### 📈 TRADE SETUP (if actionable)
- **Entry Zone:** [Price range or "Wait for..."]
- **Stop Loss:** [Price level or % from entry]
- **Target:** [Price level or % gain expected]

---

**IMPORTANT GUIDELINES:**
1. Be direct and actionable - avoid vague language
2. If model accuracy is below 55%, explicitly note low confidence
3. Weight technical signals more when sentiment is mixed
4. Consider Indian market hours and global cues
5. Never guarantee returns - use probabilistic language
6. If data is insufficient, say so clearly
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Gemini API call failed: {str(e)[:100]}. Using fallback analysis.")
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices,
                                          metrics, sentiment_summary, technical_indicators)

def generate_fallback_analysis(stock_symbol, current_price, predicted_prices, 
                               metrics, sentiment_summary, technical_indicators):
    """
    Generate structured analysis without Gemini API (template-based fallback)
    """
    price_forecast_end = predicted_prices['Predicted Price'].iloc[-1] if not predicted_prices.empty else current_price
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    accuracy = metrics.get('accuracy', 50)
    
    # Determine verdict
    if accuracy < 52:
        confidence = "Low Confidence"
    elif accuracy < 60:
        confidence = "Moderate Confidence"
    else:
        confidence = "Good Confidence"
    
    if forecast_return > 5 and accuracy > 60:
        verdict = "BUY 🟢"
        outlook = "Bullish"
    elif forecast_return > 2 and accuracy > 55:
        verdict = "HOLD (Positive Bias) 🟡"
        outlook = "Slightly Bullish"
    elif forecast_return < -5 and accuracy > 60:
        verdict = "SELL 🔴"
        outlook = "Bearish"
    elif forecast_return < -2 and accuracy > 55:
        verdict = "HOLD (Caution) 🟡"
        outlook = "Slightly Bearish"
    else:
        verdict = "HOLD 🟡"
        outlook = "Neutral"
    
    rsi = technical_indicators.get('RSI', 50)
    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    
    return f"""
### 🎯 VERDICT: {verdict}
**{confidence}** | Model Accuracy: {accuracy:.1f}%

### 📊 OUTLOOK
- **Short-term:** {outlook} | Predicted {forecast_return:+.1f}% move
- **RSI Signal:** {rsi_signal} ({rsi:.1f})

### 💡 KEY INSIGHT
The hybrid AI model (XGBoost + GRU) predicts a {'positive' if forecast_return > 0 else 'negative'} return over the forecast period. {'However, model accuracy is below 55%, suggesting low predictive confidence.' if accuracy < 55 else 'Model shows reasonable directional accuracy on test data.'}

### ⚠️ RISK FACTORS
- Model predictions are probabilistic, not guarantees
- {'High volatility detected - position size accordingly' if technical_indicators.get('Volatility_5D', 0) > 0.02 else 'Normal volatility levels'}
- External market factors may override technical signals

*Analysis generated using template mode.*
"""


# Custom Indian holiday calendar
class IndiaHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('Republic Day', month=1, day=26),
        Holiday('Independence Day', month=8, day=15),
        Holiday('Gandhi Jayanti', month=10, day=2),
        Holiday('Diwali', month=10, day=24),  # Example date - adjust as needed
        Holiday('Holi', month=3, day=25),     # Example date - adjust as needed
    ]

# Load FinBERT sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# Load Indian stock symbols
@st.cache_data
def get_indian_stocks():
    file_path = "indian_stocks.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8")
        df.columns = df.columns.str.strip()
        if "SYMBOL" in df.columns:
            return df["SYMBOL"].dropna().tolist()
        else:
            st.error("Error: 'SYMBOL' column not found.")
            return []
    else:
        st.error("File 'indian_stocks.csv' not found.")
        return ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]

# =============================================
# NEW: Fetch India VIX Data
# =============================================
def get_india_vix_data(start_date, end_date):
    """
    Fetch India VIX data from Yahoo Finance
    India VIX ticker: ^INDIAVIX
    """
    try:
        vix_ticker = "^INDIAVIX"
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date, progress=False)
        
        if vix_data.empty:
            # Fallback: Create synthetic VIX data based on NIFTY volatility
            nifty_data = yf.download("^NSEI", start=start_date, end=end_date, progress=False)
            if not nifty_data.empty:
                # Calculate rolling volatility as proxy for VIX
                returns = nifty_data['Close'].pct_change()
                vix_data = pd.DataFrame({
                    'Open': returns.rolling(20).std() * 100 * 16,  # Annualized volatility
                    'High': returns.rolling(20).std() * 100 * 16 * 1.1,
                    'Low': returns.rolling(20).std() * 100 * 16 * 0.9,
                    'Close': returns.rolling(20).std() * 100 * 16,
                    'Volume': nifty_data['Volume']
                })
                vix_data = vix_data.dropna()
        
        return vix_data
    except Exception as e:
        st.warning(f"Could not fetch India VIX data: {str(e)}")
        return pd.DataFrame()

# =============================================
# NEW: Technical Indicators Generator
# =============================================
def calculate_technical_indicators(df):
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Basic indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    
    # Moving Average Ratios
    df['MA_Ratio_5_20'] = df['MA5'] / df['MA20']
    df['MA_Ratio_5_50'] = df['MA5'] / df['MA50']
    df['MA_Ratio_20_200'] = df['MA20'] / df['MA200']
    
    # Volatility Measures
    df['Volatility_5D'] = df['Returns'].rolling(5, min_periods=1).std()
    df['Volatility_20D'] = df['Returns'].rolling(20, min_periods=1).std()
    df['ATR'] = calculate_atr(df)  # Average True Range
    
    # Volume Indicators
    df['Volume_MA5'] = df['Volume'].rolling(5, min_periods=1).mean()
    df['Volume_MA20'] = df['Volume'].rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA20']
    df['OBV'] = calculate_obv(df)  # On-Balance Volume
    
    # Momentum Indicators
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Support/Resistance Levels
    df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot_Point'] - df['Low']
    df['S1'] = 2 * df['Pivot_Point'] - df['High']
    
    # Price Position
    df['Price_vs_MA20'] = df['Close'] / df['MA20'] - 1
    df['Price_vs_MA50'] = df['Close'] / df['MA50'] - 1
    
    # Gap Analysis
    df['Gap'] = df['Open'] / df['Close'].shift(1) - 1
    
    return df.dropna()

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period, min_periods=1).mean()

def calculate_obv(df):
    """Calculate On-Balance Volume"""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# =============================================
# NEW: Dynamic Fusion Framework Models
# =============================================
class TechnicalExpertModel:
    """GRU-based model for technical data"""
    def __init__(self, lookback=30, n_features=25):
        self.lookback = lookback
        self.n_features = n_features
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = 10
        
    def build_model(self):
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=(self.lookback, self.n_features)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.3),
            GRU(32),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def prepare_data(self, X, y, lookback=30):
        """Prepare sequential data for GRU"""
        X_scaled = self.scaler.fit_transform(X)
        X_3d = []
        y_3d = []
        
        for i in range(lookback, len(X_scaled)):
            X_3d.append(X_scaled[i-lookback:i])
            y_3d.append(y[i])
        
        return np.array(X_3d), np.array(y_3d)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train the model"""
        X_seq, y_seq = self.prepare_data(X_train, y_train, self.lookback)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_data(X_val, y_val, self.lookback)
            validation_data = (X_val_seq, y_val_seq)
        else:
            validation_data = None
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        X_seq = []
        
        if len(X_scaled) >= self.lookback:
            X_seq = X_scaled[-self.lookback:].reshape(1, self.lookback, -1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
        else:
            # Pad with zeros if not enough data
            padding = np.zeros((self.lookback - len(X_scaled), X_scaled.shape[1]))
            X_padded = np.vstack([padding, X_scaled])
            X_seq = X_padded.reshape(1, self.lookback, -1)
            prediction = self.model.predict(X_seq, verbose=0)[0][0]
        
        return prediction
    
    def update_errors(self, true_value, predicted_value):
        """Update error tracking for uncertainty calculation"""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        # Keep only last N errors
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self):
        """Calculate uncertainty (variance of recent errors)"""
        if len(self.recent_errors) == 0:
            return 1.0  # Maximum uncertainty if no data
        
        return np.mean(self.recent_errors)

class SentimentExpertModel:
    """Transformer-based model for sentiment data"""
    def __init__(self):
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = 10
        
    def build_model(self):
        # Simplified transformer-like architecture using dense layers
        input_layer = Input(shape=(5,))  # Sentiment features
        x = Dense(64, activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output_layer = Dense(1)(x)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def extract_sentiment_features(self, sentiment_data):
        """Extract features from sentiment data"""
        features = []
        
        if len(sentiment_data) > 0:
            # Calculate various sentiment metrics
            sentiments = [s[0] for s in sentiment_data]  # Sentiment labels
            confidences = [s[1] for s in sentiment_data]  # Confidence scores
            
            # Convert sentiments to numerical values
            sentiment_values = []
            for sentiment, confidence in zip(sentiments, confidences):
                if sentiment == 'positive':
                    sentiment_values.append(confidence)
                elif sentiment == 'negative':
                    sentiment_values.append(-confidence)
                else:
                    sentiment_values.append(0)
            
            # Calculate features
            if sentiment_values:
                features = [
                    np.mean(sentiment_values),  # Average sentiment
                    np.std(sentiment_values),   # Sentiment volatility
                    len([v for v in sentiment_values if v > 0]) / len(sentiment_values),  # Positive ratio
                    len([v for v in sentiment_values if v < 0]) / len(sentiment_values),  # Negative ratio
                    np.max(sentiment_values)    # Maximum sentiment intensity
                ]
            else:
                features = [0, 0, 0.5, 0.5, 0]
        else:
            features = [0, 0, 0.5, 0.5, 0]  # Neutral if no sentiment data
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X_train, y_train, epochs=30, batch_size=16):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, sentiment_data):
        """Make prediction based on sentiment"""
        features = self.extract_sentiment_features(sentiment_data)
        prediction = self.model.predict(features, verbose=0)[0][0]
        return prediction
    
    def update_errors(self, true_value, predicted_value):
        """Update error tracking"""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self):
        """Calculate uncertainty"""
        if len(self.recent_errors) == 0:
            return 1.0
        
        return np.mean(self.recent_errors)

class VolatilityExpertModel:
    """MLP model for volatility (VIX) data"""
    def __init__(self):
        self.model = self.build_model()
        self.scaler = MinMaxScaler()
        self.recent_errors = []
        self.max_error_window = 10
        
    def build_model(self):
        model = Sequential([
            Dense(32, activation='relu', input_shape=(3,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def extract_volatility_features(self, vix_data, stock_data):
        """Extract volatility features from VIX and stock data"""
        if vix_data is None or stock_data is None or vix_data.empty or stock_data.empty:
            return np.array([[0.0, 0.0, 0.0]])  # Default features

        # Ensure we operate on the 'Close' series for VIX and the volatility column for stock
        if 'Close' in vix_data.columns:
            vix_close = vix_data['Close']
        else:
            vix_close = pd.Series(dtype=float)

        # Latest VIX close (scalar) - handle scalar or array-like safely
        latest_vix_close = 0.0
        if len(vix_close) > 0:
            last = vix_close.iloc[-1]
            try:
                if isinstance(last, (pd.Series, np.ndarray)):
                    arr = np.asarray(last).ravel()
                    if arr.size > 0 and not np.isnan(arr[-1]):
                        latest_vix_close = float(arr[-1])
                else:
                    if pd.notna(last):
                        latest_vix_close = float(last)
            except Exception:
                latest_vix_close = 0.0

        # VIX vs MA20 (safe computation)
        vix_vs_ma = 1.0
        if len(vix_close) >= 20:
            try:
                vix_ma20_raw = vix_close.rolling(20).mean().iloc[-1]
                # handle scalar or array-like
                if isinstance(vix_ma20_raw, (pd.Series, np.ndarray)):
                    vix_ma20_arr = np.asarray(vix_ma20_raw).ravel()
                    vix_ma20 = float(vix_ma20_arr[-1]) if vix_ma20_arr.size > 0 and not np.isnan(vix_ma20_arr[-1]) else None
                else:
                    vix_ma20 = float(vix_ma20_raw) if pd.notna(vix_ma20_raw) else None

                if vix_ma20 and vix_ma20 != 0:
                    vix_vs_ma = latest_vix_close / vix_ma20
                else:
                    vix_vs_ma = 1.0
            except Exception:
                vix_vs_ma = 1.0

        # Latest stock volatility (scalar)
        if 'Volatility_20D' in stock_data.columns and len(stock_data) > 0:
            latest_stock_vol = stock_data['Volatility_20D'].iloc[-1]
            latest_stock_vol = float(latest_stock_vol) if not pd.isna(latest_stock_vol) else 0.0
        else:
            latest_stock_vol = 0.0

        features = [latest_vix_close, vix_vs_ma, latest_stock_vol]
        return np.array(features, dtype=float).reshape(1, -1)
    
    def train(self, X_train, y_train, epochs=30, batch_size=16):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        return history
    
    def predict(self, vix_data, stock_data):
        """Make prediction based on volatility"""
        features = self.extract_volatility_features(vix_data, stock_data)
        prediction = self.model.predict(features, verbose=0)[0][0]
        return prediction
    
    def update_errors(self, true_value, predicted_value):
        """Update error tracking"""
        error = (true_value - predicted_value) ** 2
        self.recent_errors.append(error)
        
        if len(self.recent_errors) > self.max_error_window:
            self.recent_errors.pop(0)
    
    def get_uncertainty(self):
        """Calculate uncertainty"""
        if len(self.recent_errors) == 0:
            return 1.0
        
        return np.mean(self.recent_errors)

# =============================================
# NEW: Dynamic Fusion Framework
# =============================================
class DynamicFusionFramework:
    """Dynamic Fusion Framework with uncertainty-based weighting"""
    
    def __init__(self):
        self.technical_model = TechnicalExpertModel()
        self.sentiment_model = SentimentExpertModel()
        self.volatility_model = VolatilityExpertModel()
        
        # Track model performances
        self.model_predictions = {
            'technical': [],
            'sentiment': [],
            'volatility': []
        }
        self.true_values = []
        
    def calculate_dynamic_weights(self):
        """Calculate dynamic weights based on model uncertainties"""
        uncertainties = {
            'technical': self.technical_model.get_uncertainty(),
            'sentiment': self.sentiment_model.get_uncertainty(),
            'volatility': self.volatility_model.get_uncertainty()
        }
        
        # Apply Bayesian weighting formula: w_i = exp(-σ_i²) / Σ exp(-σ_j²)
        weights = {}
        total_weight = 0
        
        for model_name, uncertainty in uncertainties.items():
            # Avoid extreme values
            uncertainty = max(uncertainty, 1e-6)  # Prevent division by zero
            weight = np.exp(-uncertainty)
            weights[model_name] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight
        else:
            # Equal weights if all uncertainties are too high
            for model_name in weights:
                weights[model_name] = 1/3
        
        return weights, uncertainties
    
    def train_models(self, stock_data, sentiment_data, vix_data):
        """Train all three expert models"""
        
        # Prepare data
        stock_data_with_indicators = calculate_technical_indicators(stock_data)
        
        # Technical model features
        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA20', 'MA50', 'MA_Ratio_5_20',
            'Volatility_5D', 'Volatility_20D', 'ATR',
            'Volume_Ratio', 'RSI', 'MACD', 'MACD_Histogram',
            'Price_vs_MA20', 'Price_vs_MA50', 'Gap'
        ]
        
        # Ensure all features exist
        available_features = [f for f in tech_features if f in stock_data_with_indicators.columns]
        X_tech = stock_data_with_indicators[available_features]
        y_tech = stock_data_with_indicators['Returns'].shift(-1).dropna()
        X_tech = X_tech.iloc[:-1]  # Align with y
        
        # Train-test split
        split_idx = int(len(X_tech) * 0.8)
        X_tech_train, X_tech_test = X_tech.iloc[:split_idx], X_tech.iloc[split_idx:]
        y_tech_train, y_tech_test = y_tech.iloc[:split_idx], y_tech.iloc[split_idx:]
        
        # Train technical model
        # Ensure the technical model expects the correct number of features
        n_features_actual = X_tech.shape[1]
        if getattr(self.technical_model, 'n_features', None) != n_features_actual:
            self.technical_model.n_features = n_features_actual
            # Rebuild model to match the actual input feature count
            self.technical_model.model = self.technical_model.build_model()

        self.technical_model.train(X_tech_train, y_tech_train, X_tech_test, y_tech_test)
        
        # Prepare sentiment data
        sentiment_features = []
        sentiment_targets = []
        
        for date in stock_data_with_indicators.index:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in sentiment_data:
                daily_sentiments = sentiment_data[date_str]
                features = self.sentiment_model.extract_sentiment_features(daily_sentiments)
                sentiment_features.append(features[0])
                
                # Use next day's return as target
                if date in stock_data_with_indicators.index:
                    idx = stock_data_with_indicators.index.get_loc(date)
                    if idx + 1 < len(stock_data_with_indicators):
                        target = stock_data_with_indicators['Returns'].iloc[idx + 1]
                        sentiment_targets.append(target)
        
        if len(sentiment_features) > 10:
            X_sent = np.array(sentiment_features)[:len(sentiment_targets)]
            y_sent = np.array(sentiment_targets)[:len(sentiment_features)]
            self.sentiment_model.train(X_sent, y_sent)
        
        # Prepare volatility data
        volatility_features = []
        volatility_targets = []
        
        for i in range(len(stock_data_with_indicators)):
            if i >= 20:  # Need enough data for volatility calculation
                vix_slice = vix_data.iloc[:i+1] if len(vix_data) > i else vix_data
                stock_slice = stock_data_with_indicators.iloc[:i+1]
                
                features = self.volatility_model.extract_volatility_features(vix_slice, stock_slice)
                volatility_features.append(features[0])
                
                # Use next day's return as target
                if i + 1 < len(stock_data_with_indicators):
                    target = stock_data_with_indicators['Returns'].iloc[i + 1]
                    volatility_targets.append(target)
        
        if len(volatility_features) > 10:
            X_vol = np.array(volatility_features)[:len(volatility_targets)]
            y_vol = np.array(volatility_targets)[:len(volatility_features)]
            self.volatility_model.train(X_vol, y_vol)
    
    def predict(self, stock_data, sentiment_data, vix_data):
        """Make combined prediction using dynamic fusion"""
        
        # Get individual predictions
        stock_features = calculate_technical_indicators(stock_data)
        
        # Get technical features - CORRECTED VERSION
        if hasattr(self.technical_model.scaler, 'feature_names_in_'):
            tech_features = [f for f in self.technical_model.scaler.feature_names_in_ 
                            if f in stock_features.columns]
        else:
            # Use default features if feature_names_in_ not available
            tech_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'MA50', 'MA_Ratio_5_20',
                'Volatility_5D', 'Volatility_20D', 'ATR',
                'Volume_Ratio', 'RSI', 'MACD', 'MACD_Histogram',
                'Price_vs_MA20', 'Price_vs_MA50', 'Gap'
            ]
            # Filter to only include features that exist in the DataFrame
            tech_features = [f for f in tech_features if f in stock_features.columns]
        
        # Ensure we have enough features
        if len(tech_features) < 5:  # Minimum number of features needed
            # Use all available columns except the target
            tech_features = [col for col in stock_features.columns 
                            if col not in ['Returns', 'Target', 'Predicted']]
        
        # Limit to the model's expected number of features
        tech_features = tech_features[:self.technical_model.n_features]
        
        X_tech = stock_features[tech_features].iloc[-self.technical_model.lookback:]
        
        # If we have fewer than the required lookback days, warn but allow prediction by padding inside the model
        if len(X_tech) < self.technical_model.lookback:
            st.warning(f"Insufficient data for technical model. Need {self.technical_model.lookback} days, have {len(X_tech)} — padding will be applied to predict.")

        tech_pred = self.technical_model.predict(X_tech)
        
        # Get sentiment for latest date
        latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
        latest_sentiment = sentiment_data.get(latest_date, [])
        sentiment_pred = self.sentiment_model.predict(latest_sentiment)
        
        # Volatility prediction
        volatility_pred = self.volatility_model.predict(vix_data, stock_features)
        
        # Calculate dynamic weights
        weights, uncertainties = self.calculate_dynamic_weights()
        
        # Store predictions for tracking
        self.model_predictions['technical'].append(tech_pred)
        self.model_predictions['sentiment'].append(sentiment_pred)
        self.model_predictions['volatility'].append(volatility_pred)
        
        # Combine predictions with dynamic weights
        combined_pred = (
            weights['technical'] * tech_pred +
            weights['sentiment'] * sentiment_pred +
            weights['volatility'] * volatility_pred
        )
        
        return {
            'combined_prediction': combined_pred,
            'individual_predictions': {
                'technical': tech_pred,
                'sentiment': sentiment_pred,
                'volatility': volatility_pred
            },
            'weights': weights,
            'uncertainties': uncertainties
        }
    
    def update_model_performance(self, true_return):
        """Update models with true value for error calculation"""
        self.true_values.append(true_return)
        
        if len(self.true_values) > 1 and len(self.model_predictions['technical']) > 0:
            last_true = self.true_values[-2]  # Previous true value
            
            # Update each model's error tracking
            for model_name in ['technical', 'sentiment', 'volatility']:
                if len(self.model_predictions[model_name]) > 0:
                    last_pred = self.model_predictions[model_name][-1]
                    
                    if model_name == 'technical':
                        self.technical_model.update_errors(last_true, last_pred)
                    elif model_name == 'sentiment':
                        self.sentiment_model.update_errors(last_true, last_pred)
                    elif model_name == 'volatility':
                        self.volatility_model.update_errors(last_true, last_pred)

# =============================================
# NEW: Enhanced Visualization Functions
# =============================================
def create_dynamic_weights_visualization(weights_history):
    """Create visualization for dynamic weights over time"""
    fig = go.Figure()
    
    dates = list(weights_history.keys())
    tech_weights = [w['technical'] for w in weights_history.values()]
    sent_weights = [w['sentiment'] for w in weights_history.values()]
    vol_weights = [w['volatility'] for w in weights_history.values()]
    
    fig.add_trace(go.Scatter(
        x=dates, y=tech_weights,
        mode='lines+markers',
        name='Technical Model Weight',
        line=dict(color='blue', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=sent_weights,
        mode='lines+markers',
        name='Sentiment Model Weight',
        line=dict(color='green', width=2),
        stackgroup='one'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=vol_weights,
        mode='lines+markers',
        name='Volatility Model Weight',
        line=dict(color='red', width=2),
        stackgroup='one'
    ))
    
    fig.update_layout(
        title='Dynamic Model Weights Over Time',
        xaxis_title='Date',
        yaxis_title='Weight',
        hovermode='x unified',
        yaxis=dict(tickformat='.0%'),
        showlegend=True
    )
    
    return fig

def create_uncertainty_visualization(uncertainties_history):
    """Create visualization for model uncertainties"""
    fig = go.Figure()
    
    dates = list(uncertainties_history.keys())
    tech_unc = [u['technical'] for u in uncertainties_history.values()]
    sent_unc = [u['sentiment'] for u in uncertainties_history.values()]
    vol_unc = [u['volatility'] for u in uncertainties_history.values()]
    
    fig.add_trace(go.Scatter(
        x=dates, y=tech_unc,
        mode='lines',
        name='Technical Model Uncertainty',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=sent_unc,
        mode='lines',
        name='Sentiment Model Uncertainty',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=vol_unc,
        mode='lines',
        name='Volatility Model Uncertainty',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title='Model Uncertainties Over Time',
        xaxis_title='Date',
        yaxis_title='Uncertainty (σ²)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def create_model_performance_radar(weights, uncertainties):
    """Create radar chart for model performance comparison"""
    fig = go.Figure()
    
    models = ['Technical', 'Sentiment', 'Volatility']
    
    # Inverse of uncertainty = confidence
    confidences = [1/(u+1e-6) for u in uncertainties.values()]

    weight_vals = list(weights.values())

    fig.add_trace(go.Scatterpolar(
        r=weight_vals,
        theta=models,
        fill='toself',
        name='Model Weights',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=confidences,
        theta=models,
        fill='toself',
        name='Model Confidence',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(weight_vals), max(confidences))]
            )),
        showlegend=True,
        title='Model Performance Radar Chart'
    )
    
    return fig

# =============================================
# MODIFIED: Enhanced Streamlit UI Integration
# =============================================

# Fetch stock data (existing function)
def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    # Ensure start/end are passed as ISO date strings to yfinance
    try:
        start_str = pd.to_datetime(start).strftime('%Y-%m-%d')
    except Exception:
        start_str = start
    try:
        end_str = pd.to_datetime(end).strftime('%Y-%m-%d')
    except Exception:
        end_str = end

    data = stock.history(start=start_str, end=end_str)
    if not data.empty:
        data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        data.index = data.index.tz_localize(None)
        data = data.sort_index()
    return data

# =============================================
# NEW: Fundamental Analysis Module
# =============================================
def get_fundamental_data(ticker):
    """Fetch fundamental data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Safe extraction with default values
        fundamentals = {
            "Forward P/E": info.get("forwardPE", np.nan),
            "PEG Ratio": info.get("pegRatio", np.nan),
            "Price/Book": info.get("priceToBook", np.nan),
            "Debt/Equity": info.get("debtToEquity", np.nan),
            "ROE": info.get("returnOnEquity", np.nan),
            "Profit Margins": info.get("profitMargins", np.nan),
            "Revenue Growth": info.get("revenueGrowth", np.nan),
            "Free Cashflow": info.get("freeCashflow", np.nan),
            "Target Price (Analyst)": info.get("targetMeanPrice", np.nan)
        }
        return fundamentals
    except Exception as e:
        st.warning(f"Could not fetch fundamentals: {e}")
        return {}

# =============================================
# NEW: Risk Management Module
# =============================================
class RiskManager:
    """Professional Risk Management & Position Sizing"""
    
    @staticmethod
    def calculate_atr(df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1]

    @staticmethod
    def calculate_fibonacci_levels(df, lookback=90):
        recent_data = df.iloc[-lookback:]
        max_price = recent_data['High'].max()
        min_price = recent_data['Low'].min()
        diff = max_price - min_price
        
        levels = {
            "0.0% (Low)": min_price,
            "23.6%": min_price + 0.236 * diff,
            "38.2%": min_price + 0.382 * diff,
            "50.0%": min_price + 0.5 * diff,
            "61.8%": min_price + 0.618 * diff,
            "100.0% (High)": max_price
        }
        return levels

    @staticmethod
    def kelly_criterion(win_rate, win_loss_ratio):
        """
        Calculate optimal position size percentage.
        f* = (bp - q) / b
        b = odds received (win/loss ratio)
        p = probability of winning
        q = probability of losing (1-p)
        """
        if win_loss_ratio == 0: return 0
        return max(0, (win_loss_ratio * win_rate - (1 - win_rate)) / win_loss_ratio)

    @staticmethod
    def get_trade_setup(current_price, prediction, atr, confidence):
        direction = "LONG" if prediction > current_price else "SHORT"
        
        # Dynamic Stop Loss based on ATR
        # Tighter stop for lower confidence
        sl_multiplier = 2.0 if confidence > 0.7 else 1.5
        stop_dist = atr * sl_multiplier
        
        if direction == "LONG":
            stop_loss = current_price - stop_dist
            # Target: 1.5x risk at minimum
            target = current_price + (stop_dist * 1.5)
        else:
            stop_loss = current_price + stop_dist
            target = current_price - (stop_dist * 1.5)
            
        risk_reward = abs(target - current_price) / abs(current_price - stop_loss)
        
        return {
            "Direction": direction,
            "Entry": current_price,
            "Stop Loss": stop_loss,
            "Target": target,
            "Risk/Reward": risk_reward
        }

# =============================================
# NEW: Professional Backtester
# =============================================
class VectorizedBacktester:
    """Fast Vectorized Backtesting Engine"""
    
    def __init__(self, data, signals):
        self.data = data
        self.signals = signals
        
    def run_backtest(self, initial_capital=100000):
        """
        Runs a vectorized backtest based on RETURNS.
        """
        df = self.data.copy()
        
        # Ensure we have 'Actual_Return' (Next Day Return)
        # Logic: If Signal[T] is BUY, we get Return[T+1] (which is typically 'Target' or 'Next_Ret')
        # In our new pipeline, 'Actual_Return' in df is indeed the Return of T+1 (Target).
        # So Strategy Return = Signal[T] * Actual_Return[T]
        
        df['Strategy_Return'] = self.signals * df['Actual_Return']
        
        # Equity Curve
        df['Equity_Curve'] = initial_capital * (1 + df['Strategy_Return']).cumprod()
        
        # Metrics
        total_return = (df['Equity_Curve'].iloc[-1] / initial_capital) - 1
        sharpe_ratio = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252) if df['Strategy_Return'].std() != 0 else 0
        
        # Drawdown
        cumulative_returns = (1 + df['Strategy_Return']).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win Rate
        wins = df[df['Strategy_Return'] > 0]
        win_rate = len(wins) / len(df) if len(df) > 0 else 0
        
        return {
            "Total Return": total_return,
            "Sharpe Ratio": sharpe_ratio,
            "Max Drawdown": max_drawdown,
            "Win Rate": win_rate,
            "Equity Curve": df['Equity_Curve']
        }

# =============================================
# NEW: AutoML Hyperparameter Optimization
# =============================================
class ModelOptimizer:
    """Bayesian Hyperparameter Optimization using Optuna"""
    
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
    def optimize_xgb(self, n_trials=10):
        if optuna is None:
            return None # Fallback to default
        
        def objective(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'objective': 'reg:squarederror',
                'n_jobs': -1
            }
            
            model = xgb.XGBRegressor(**param)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            return rmse
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def optimize_gru(self, input_shape, X_train=None, X_test=None, n_trials=5):
        if optuna is None:
            return None
            
        # Use provided data if available (e.g. scaled/reshaped), else fallback to self (unlikely for GRU)
        train_x = X_train if X_train is not None else self.X_train
        test_x = X_test if X_test is not None else self.X_test
            
        def objective(trial):
            units1 = trial.suggest_int('units1', 32, 128)
            units2 = trial.suggest_int('units2', 16, 64)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            
            model = Sequential([
                GRU(units1, input_shape=input_shape, return_sequences=True),
                Dropout(dropout),
                GRU(units2),
                Dropout(dropout),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            
            # Short training for speed
            model.fit(train_x, self.y_train, epochs=10, batch_size=32, verbose=0)
            
            preds = model.predict(test_x, verbose=0)
            rmse = np.sqrt(mean_squared_error(self.y_test, preds))
            return rmse
            
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

# Fetch stock info (existing function)
def get_stock_info(ticker):

    stock = yf.Ticker(ticker)
    info = stock.info
    
    def format_value(value, format_str):
        if value == "N/A" or value is None:
            return "N/A"
        return format_str.format(value)
    
    return {
        "Market Cap": format_value(info.get("marketCap"), "{:,} INR"),
        "P/E Ratio": format_value(info.get("trailingPE"), "{}"),
        "ROCE": format_value(info.get("returnOnCapitalEmployed"), "{:.2f}%"),
        "Current Price": format_value(info.get("currentPrice"), "{:.2f} INR"),
        "Book Value": format_value(info.get("bookValue"), "{:.2f} INR"),
        "ROE": format_value(info.get("returnOnEquity"), "{:.2f}%"),
        "Dividend Yield": format_value(info.get("dividendYield"), "{:.2f}%"),
        "Face Value": format_value(info.get("faceValue"), "{:.2f} INR"),
        "High": format_value(info.get("dayHigh"), "{:.2f} INR"),
        "Low": format_value(info.get("dayLow"), "{:.2f} INR"),
    }

# News API (existing function)
NEWS_API_KEY = "563215a35c1a47968f46271e04083ea3"
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_news(stock_symbol):
    stock_name_mapping = {
        "RELIANCE": "Reliance Industries",
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "HDFCBANK": "HDFC Bank",
        "ICICIBANK": "ICICI Bank"
    }
    query = stock_name_mapping.get(stock_symbol, stock_symbol)
    params = {"q": query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt"}
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code != 200:
        st.error(f"Error fetching news: {response.json()}")
        return []
    return response.json().get("articles", [])

# Sentiment analysis (existing function)
def analyze_sentiment(text):
    if not text:
        return "neutral", 0.0
    result = sentiment_pipeline(text[:512])[0]
    return result['label'], result['score']

# Filter relevant news (existing function)
def filter_relevant_news(news_articles, stock_name):
    filtered_articles = []
    for article in news_articles:
        title = article.get('title', '')
        if title and re.search(stock_name, title, re.IGNORECASE):  
            filtered_articles.append(article)
    return filtered_articles

# =============================================
# REFACTORED: Strict Stationarity & No Leakage
# =============================================
def create_advanced_features(df):
    """
    Creates strictly stationary features for ML.
    AVOIDS: Absolute prices (Open, High, Low)
    USES: Returns, Volatility, Oscillators
    """
    df = df.copy()
    
    # 1. Target: Log Returns (Stationary)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Volatility (Normalized)
    df['Volatility_5D'] = df['Log_Ret'].rolling(window=5).std()
    
    # 3. Momentum (RSI is already 0-100, good)
    # Assuming 'RSI' exists from calculate_technical_indicators
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
    df['RSI_Norm'] = df['RSI'] / 100.0  # Scale to 0-1
    
    # 4. Volume Trend (Ratio)
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    # 5. Moving Average Divergence (Normalized by Price)
    df['MA_Div'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close']
    
    # Drop NaNs created by rolling windows
    df.dropna(inplace=True)
    
    return df

# Prophet forecasting (existing function)
def prophet_forecast(df, days=10):
    prophet_df = df.reset_index()[['Date', 'Close']].rename(
        columns={'Date': 'ds', 'Close': 'y'}
    )
    
    model = Prophet(
        daily_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode='additive'
    )
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=days, include_history=False)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat']].set_index('ds')

def adjust_predictions_for_market_closures(predictions_df):
    """
    Adjust predictions to show steady values on market closed days (weekends and Indian holidays).
    """
    # Create Indian business day calendar (Mon-Fri excluding holidays)
    india_bd = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    
    # Generate business days in the prediction range
    business_days = pd.date_range(
        start=predictions_df.index.min(),
        end=predictions_df.index.max(),
        freq=india_bd
    )
    
    # Mark non-business days
    predictions_df['is_market_day'] = predictions_df.index.isin(business_days)
    
    # Forward fill predictions for non-market days
    predictions_df['adjusted_prediction'] = np.where(
        predictions_df['is_market_day'],
        predictions_df['Predicted Price'],
        np.nan
    )
    predictions_df['adjusted_prediction'] = predictions_df['adjusted_prediction'].ffill()
    
    # Calculate daily changes based on adjusted predictions
    predictions_df['Daily Change (%)'] = predictions_df['adjusted_prediction'].pct_change().fillna(0) * 100
    
    return predictions_df[['adjusted_prediction', 'Daily Change (%)']].rename(
        columns={'adjusted_prediction': 'Predicted Price'}
    )

# REFACTORED: Hybrid Model Predicting RETURNS
def create_hybrid_model(df, sentiment_features, enable_automl=False):
    # 1. Prepare Data
    df_proc = create_advanced_features(df)
    
    # Merge Sentiment (Lagged by 1 day to prevent leakage - we predict T using news from T-1)
    # Actually, efficient markets react instantly. But for 'prediction', we use known info.
    # If we predict Close[T], we know Sentiment[T] if it's "Live", but for backtesting...
    # Let's align Sentiment[T-1] to predict Return[T] is safer, but usually we use Sentiment[T] to predict Close[T] intraday.
    # For daily close prediction, we usually use data up to T-1 to predict T.
    
    sentiment_df = pd.DataFrame(sentiment_features.items(), columns=["Date", "Sentiment"])
    if not sentiment_df.empty:
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date']).dt.tz_localize(None)
        df_proc = df_proc.reset_index().merge(sentiment_df, on="Date", how="left").set_index("Date")
        df_proc['Sentiment'] = pd.to_numeric(df_proc['Sentiment'], errors='coerce').fillna(0)
    else:
        df_proc['Sentiment'] = 0.0

    # 2. Define Features & Target
    # Predicting Next Day's Return
    df_proc['Target_Return'] = df_proc['Log_Ret'].shift(-1)
    df_proc.dropna(inplace=True)
    
    features = ['Log_Ret', 'Volatility_5D', 'RSI_Norm', 'Vol_Ratio', 'MA_Div', 'Sentiment']
    
    X = df_proc[features].values
    y = df_proc['Target_Return'].values
    
    # 3. Strict Time-Series Split (No random shuffle!)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 4. Strict Scaling (Fit on Train ONLY)
    scaler = MinMaxScaler(feature_range=(-1, 1)) # Returns can be negative
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Transform test using train stats
    
    # 5. XGBoost (Regressing Returns)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,      # Reduced complexity
        max_depth=3,           # Reduced depth to prevent overfitting
        learning_rate=0.05,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    
    # 6. GRU (Simplified)
    # Reshape for GRU: [Samples, Timesteps, Features]
    X_train_3d = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_3d = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    model_gru = Sequential([
        GRU(32, input_shape=(1, len(features)), return_sequences=False), # Single layer
        Dropout(0.2), # Dropout
        Dense(1) # Linear output for regression
    ])
    model_gru.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model_gru.fit(X_train_3d, y_train, epochs=20, batch_size=32, verbose=0, shuffle=False)
    
    gru_pred = model_gru.predict(X_test_3d, verbose=0).flatten()
    
    # 7. Simple Ensemble (Average)
    hybrid_pred = (xgb_pred + gru_pred) / 2.0
    
    # 8. Evaluation (Directional Accuracy & RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
    
    # Directional Accuracy: Did we predict the sign correctly?
    correct_direction = np.sign(hybrid_pred) == np.sign(y_test)
    accuracy = np.mean(correct_direction) * 100
    
    # Store predictions in DataFrame for analysis
    test_dates = df_proc.index[train_size:]
    results_df = pd.DataFrame({
        'Actual_Return': y_test,
        'Predicted_Return': hybrid_pred,
        'XGB_Return': xgb_pred,
        'GRU_Return': gru_pred
    }, index=test_dates)
    
    metrics = {
        'rmse': rmse,
        'accuracy': accuracy,
        'xgb_weight': 0.5, # Fixed weight for now
        'gru_weight': 0.5
    }
    
    return df_proc, results_df, {'xgb': xgb_model, 'gru': model_gru}, scaler, features, metrics

def hybrid_predict_next_day(models, scaler, last_data_point, features):
    """
    Predicts NEXT DAY return using the trained models.
    """
    # Prepare single sample
    x_input = list(last_data_point[features].values)
    x_scaled = scaler.transform([x_input])
    
    # XGB
    xgb_pred = models['xgb'].predict(x_scaled)[0]
    
    # GRU
    x_3d = x_scaled.reshape((1, 1, len(features)))
    gru_pred = models['gru'].predict(x_3d, verbose=0)[0][0]
    
    # Avg
    avg_return = (xgb_pred + gru_pred) / 2.0
    return avg_return

# REFACTORED: Recursive Price Forecast from Returns
def hybrid_predict_prices(models, scaler, last_known_data, features, days=10, weights=None):
    """
    Project future prices by recursively predicting returns.
    """
    future_dates = []
    future_prices = []
    daily_changes = []
    
    # Start from the very last known data point
    # We need to re-create features for the last window to ensure we have the correct inputs
    # But since create_advanced_features uses rolling windows, passing just the last row is insufficient.
    # We must pass enough history (e.g. 30 days) to calculate the features for the "current" point.
    
    current_data_window = last_known_data.copy()
    current_price = current_data_window['Close'].iloc[-1]
    last_date = current_data_window.index[-1]
    
    # Generate business days
    custom_business_day = CustomBusinessDay(calendar=IndiaHolidayCalendar())
    
    for i in range(days):
        next_date = last_date + custom_business_day
        last_date = next_date
        
        # Recalculate features on the FLY for the last row
        # This ensures 'Volatility', 'RSI' etc are updated if we were appending data
        # For this simplified version, we assume features are accessible.
        
        feat_df = create_advanced_features(current_data_window)
        if feat_df.empty:
             break
             
        current_features = feat_df.iloc[-1][features]
        
        # 1. Predict Return for Tomorrow
        pred_return = hybrid_predict_next_day(models, scaler, current_features, features)
        
        # 2. Calculate New Price
        new_price = current_price * np.exp(pred_return)
        
        # 3. Store
        future_dates.append(next_date)
        future_prices.append(new_price)
        pct_change = (new_price - current_price) / current_price * 100
        daily_changes.append(pct_change)
        
        # 4. Update State for Next Step (Append new row to window)
        # We MUST carry over 'Sentiment' and other non-technical features so create_advanced_features doesn't drop them
        new_row_dict = {
            'Open': new_price, 'High': new_price, 'Low': new_price, 'Close': new_price,
            'Volume': current_data_window['Volume'].iloc[-1]
        }
        if 'Sentiment' in current_data_window.columns:
            new_row_dict['Sentiment'] = current_data_window['Sentiment'].iloc[-1]
            
        new_row = pd.DataFrame(new_row_dict, index=[next_date])
        
        current_data_window = pd.concat([current_data_window, new_row])
        current_price = new_price
        
    return pd.DataFrame({
        'Predicted Price': future_prices,
        'Daily Change (%)': daily_changes
    }, index=future_dates)

# Candlestick chart (existing function)
def create_candlestick_chart(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    )])
    fig.update_layout(
        title="Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    return fig

# Generate investment recommendation (existing function)
def generate_recommendation(predicted_prices, current_price, accuracy, avg_sentiment):
    avg_prediction = predicted_prices['Predicted Price'].mean()
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    # Enhanced sentiment factor with confidence scaling
    sentiment_factor = 1 + (avg_sentiment * (accuracy/100))  # Scale sentiment impact by accuracy
    
    adjusted_change = price_change * sentiment_factor
    
    # Modified thresholds with confidence weighting
    confidence_weight = accuracy / 100
    if adjusted_change > 7 * confidence_weight and accuracy > 72:
        return "STRONG BUY", "High confidence in significant price increase"
    elif adjusted_change > 3 * confidence_weight and accuracy > 65:
        return "BUY", "Good confidence in moderate price increase"
    elif adjusted_change > 0 and accuracy > 60:
        return "HOLD (Positive)", "Potential for slight growth"
    elif adjusted_change < -7 * confidence_weight and accuracy > 72:
        return "STRONG SELL", "High confidence in significant price drop"
    elif adjusted_change < -3 * confidence_weight and accuracy > 65:
        return "SELL", "Good confidence in moderate price drop"
    elif adjusted_change < 0 and accuracy > 60:
        return "HOLD (Caution)", "Potential for slight decline"
    else:
        return "HOLD", "Unclear direction - consider other factors"

# =============================================
# MAIN STREAMLIT APP WITH DYNAMIC FUSION
# =============================================

# =============================================
# VISUAL PATTERN ANALYST (YOLOv8)
# =============================================
class VisualPatternAnalyst:
    def __init__(self):
        self.pattern_model = None
        self.future_model = None
        if VISUAL_AI_AVAILABLE:
            try:
                # Load models (will download from HF automatically if not present?) 
                # Ultralytics usually needs local path or 'model=' argument for HF?
                # Actually, ultralytics supports loading from HF hub via 'model=https://huggingface.co/...' or similar?
                # Or we assume user has to download? 
                # Let's try downloading if not exists or use direct loading.
                # Standard syntax: YOLO('model.pt')
                # If we don't have the weights locally, we might need to rely on 'foduucom/stockmarket-pattern-detection-yolov8' string support if library has it.
                # Ultralytics 8.1+ supports HF model loading directly? 
                # If not, I will point to local specific checks.
                # For this implementation, I will attempt to load by repo name if supported, else assume standardized path.
                # Limitation: HF download via ultralytics might not be seamless.
                # Workaround: We will try to download the weights file using huggingface_hub logic if needed, 
                # or just try the name string.
                
                # Using the repo name directly often works in newer versions or requires 'hf-hub:' prefix?
                # Documentation says: model = YOLO("https://huggingface.co/...")
                pass
            except Exception as e:
                st.error(f"Failed to initialize Visual AI: {e}")

    def generate_chart_image(self, df, filename="temp_chart.jpg"):
        """Generates a clean candlestick chart for computer vision"""
        df_slice = df.iloc[-50:] # Focus on recent
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple candlestick calculation
        width = 0.6
        width2 = 0.1
        
        up = df_slice[df_slice.Close >= df_slice.Open]
        down = df_slice[df_slice.Close < df_slice.Open]
        
        col1 = 'green'
        col2 = 'red'
        
        # Plot up prices
        ax.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
        ax.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
        ax.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)
        
        # Plot down prices
        ax.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
        ax.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
        ax.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)
        
        ax.axis('off') # Hide axis for pure pattern detection? 
        # Or models might expect grid? Usually pattern detection works best on pure geometry.
        # Let's keep it clean.
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return filename

    def analyze_patterns(self, df):
        if not VISUAL_AI_AVAILABLE:
            return None, [], "Visual AI Library not installed."
            
        # Create unique temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            chart_path = tmp.name
            
        self.generate_chart_image(df, chart_path)
        
        results_data = []
        annotated_img = None
        visual_bias = "Neutral"
        
        try:
            # 1. Pattern Detection
            # Note: We use the HF model identifiers. 
            # If this fails, we catch it.
            model_pat = YOLO('https://huggingface.co/foduucom/stockmarket-pattern-detection-yolov8/resolve/main/best.pt') 
            # Direct link to .pt file is safest
            
            res_pat = model_pat.predict(chart_path, conf=0.25)
            annotated_pat = res_pat[0].plot() # numpy array (BGR)
            
            # Extract detected classes
            for r in res_pat:
                for box in r.boxes:
                    cls_name = model_pat.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    results_data.append({"Type": "Pattern", "Name": cls_name, "Confidence": f"{conf:.2f}"})
            
            # 2. Future Prediction (Trend)
            model_fut = YOLO('https://huggingface.co/foduucom/stockmarket-future-prediction/resolve/main/best.pt')
            res_fut = model_fut.predict(chart_path, conf=0.25)
            
            bullish_score = 0
            bearish_score = 0
            
            for r in res_fut:
                for box in r.boxes:
                    cls_name = model_fut.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    results_data.append({"Type": "Trend Signal", "Name": cls_name, "Confidence": f"{conf:.2f}"})
                    
                    if "Up" in cls_name or "Bull" in cls_name:
                        bullish_score += conf
                    elif "Down" in cls_name or "Bear" in cls_name:
                        bearish_score += conf
                        
            if bullish_score > bearish_score:
                visual_bias = "Bullish"
            elif bearish_score > bullish_score:
                visual_bias = "Bearish"
                
            # Convert BGR to RGB for Streamlit
            annotated_img = cv2.cvtColor(annotated_pat, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            return None, [], f"Inference Error: {str(e)}"
            
        return annotated_img, results_data, visual_bias

# =============================================
# MAIN STREAMLIT APP WITH DYNAMIC FUSION
# =============================================

# Streamlit UI
st.set_page_config(page_title="Pro Stock AI", layout="wide")
st.title("🏆 ProTrader AI: Professional Stock Analytics")

# Sidebar Configuration
st.sidebar.header("Configuration")
indian_stocks = get_indian_stocks()
selected_stock = st.sidebar.selectbox("Select Stock", indian_stocks)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start", datetime.date(2023, 1, 1))
with col2:
    end_date = st.date_input("End", datetime.date.today())

st.sidebar.subheader("Advanced Settings")
enable_dynamic_fusion = st.sidebar.checkbox("Dynamic Fusion Framework", value=True)
enable_automl = st.sidebar.checkbox("AutoML Optimization (Slow)", value=False)
forecast_days = st.sidebar.slider("Forecast Horizon", 1, 30, 10)

chart_type = st.sidebar.radio("Primary Chart", ["Candlestick", "Line"])

# Track if we should show analysis - either button clicked or session state exists
show_analysis = False
df_stock = None
fundamentals = {}
news_articles = []

if st.sidebar.button("Launch Analysis", type="primary"):
    ticker = f"{selected_stock}.NS"
    
    # 1. Data Fetching
    with st.spinner(f"Fetching data for {ticker}..."):
        df_stock = get_stock_data(ticker, start_date, end_date)
        fundamentals = get_fundamental_data(ticker)
        news_articles = get_news(selected_stock)
        
        # Store in session state for persistence across reruns
        st.session_state['df_stock'] = df_stock
        st.session_state['fundamentals'] = fundamentals
        st.session_state['news_articles'] = news_articles
        st.session_state['selected_stock'] = selected_stock
        st.session_state['analysis_done'] = True
        st.session_state['forecast_days'] = forecast_days
        st.session_state['enable_dynamic_fusion'] = enable_dynamic_fusion
        st.session_state['chart_type'] = chart_type
        st.session_state['start_date'] = start_date
        st.session_state['end_date'] = end_date
        
    if not df_stock.empty:
        df_stock = df_stock.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
        st.session_state['df_stock'] = df_stock
        show_analysis = True

# Also show analysis if we have data in session state (e.g., after tab button clicks)
elif st.session_state.get('analysis_done') and st.session_state.get('df_stock') is not None:
    df_stock = st.session_state['df_stock']
    fundamentals = st.session_state.get('fundamentals', {})
    news_articles = st.session_state.get('news_articles', [])
    selected_stock = st.session_state.get('selected_stock', selected_stock)
    forecast_days = st.session_state.get('forecast_days', forecast_days)
    enable_dynamic_fusion = st.session_state.get('enable_dynamic_fusion', enable_dynamic_fusion)
    chart_type = st.session_state.get('chart_type', chart_type)
    start_date = st.session_state.get('start_date', start_date)
    end_date = st.session_state.get('end_date', end_date)
    show_analysis = True

if show_analysis and df_stock is not None and not df_stock.empty:
    # Tabs for better organization
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "🔬 Dynamic Fusion", 
        "📈 Technicals & Risk", 
        "🏛️ Fundamentals",
        "🛠️ Backtest",
        "👁️ Visual Analysis"
    ])

    # ======================================
    # TAB 1: Main Dashboard
    # ======================================
    with tab1:
        # Top Stats Row
        current_price = df_stock['Close'].iloc[-1]
        price_change = df_stock['Close'].iloc[-1] - df_stock['Close'].iloc[-2]
        pct_change = (price_change / df_stock['Close'].iloc[-2]) * 100
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{current_price:,.2f}", f"{pct_change:.2f}%")
        m2.metric("Market Cap", f"{fundamentals.get('MarketCap', 'N/A')}") # Need to parse safety
        m3.metric("P/E Ratio", f"{fundamentals.get('Forward P/E', 'N/A')}")
        m4.metric("Volume", f"{df_stock['Volume'].iloc[-1]:,}")

        # Main Chart
        st.subheader(f"Price Action: {selected_stock}")
        if chart_type == "Candlestick":
            st.plotly_chart(create_candlestick_chart(df_stock), use_container_width=True)
        else:
            st.line_chart(df_stock["Close"])
            
        # Sentiment Summary
        filtered_news = filter_relevant_news(news_articles, selected_stock)
        daily_sentiment = {}
        if filtered_news:
            st.info(f"Analyzed {len(filtered_news)} recent news articles for sentiment integration.")
            for article in filtered_news:
                text = f"{article.get('title','')} {article.get('description','')}".strip()
                sentiment, score = analyze_sentiment(text)
                date = article.get("publishedAt", "")[0:10]
                
                if date in daily_sentiment:
                    daily_sentiment[date].append((sentiment, score))
                else:
                    daily_sentiment[date] = [(sentiment, score)]
        else:
            st.warning("No specific news found. Using technicals only.")

        # Run Hybrid Model
        st.subheader("🤖 AI Price Forecast")
        with st.spinner("Running Hybrid AI Models (Sequential Train/Test)..."):
            st.header("📊 Hybrid AI Model Analysis (Professional Validation)")
            
            # Train hybrid model - NOW WITH COMPREHENSIVE METRICS
            df_proc, results_df, models, scaler, features, metrics = create_hybrid_model(df_stock, daily_sentiment if daily_sentiment else {})
            
            # Display Model Performance Metrics
            st.subheader("Strict Walk-Forward Validation Results")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Test RMSE (Lower is Better)", f"{metrics['rmse']:.4f}")
            with c2:
                st.metric("Directional Accuracy", f"{metrics['accuracy']:.2f}%")
                
            st.caption("Note: These metrics are calculated on strict out-of-sample data. They rely on predicting returns, not prices, making them a true test of market prediction ability.")
            
            # Display Scatter Plot of Predicted vs Actual Returns (True Validation)
            st.subheader("Predicted vs Actual Returns")
            
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=results_df.index, y=results_df['Actual_Return'], name='Actual Returns', mode='lines', line=dict(color='blue', width=1)))
            fig_val.add_trace(go.Scatter(x=results_df.index, y=results_df['Predicted_Return'], name='Predicted Returns', mode='lines', line=dict(color='orange', width=1)))
            st.plotly_chart(fig_val, use_container_width=True)

            # Store results in session state for other tabs
            st.session_state['results_df'] = results_df
            st.session_state['metrics'] = metrics
            st.session_state['df_proc'] = df_proc
            st.session_state['models'] = models
            st.session_state['scaler'] = scaler
            st.session_state['features'] = features
            
            # Forecast (using recursive returns)
            current_price = df_stock['Close'].iloc[-1]
            future_prices = hybrid_predict_prices(
                models, scaler, df_proc.iloc[-60:], features, 
                days=forecast_days, 
                weights={'xgb_weight': 0.5, 'gru_weight': 0.5}
            )
            st.session_state['future_prices'] = future_prices
            
            # =============================================
            # ENHANCED AI ANALYSIS WITH GEMINI
            # =============================================
            st.markdown("---")
            st.header("🤖 AI Expert Analysis")
            
            # Prepare technical indicators for Gemini
            technical_indicators = {
                'RSI': df_proc['RSI_Norm'].iloc[-1] * 100 if 'RSI_Norm' in df_proc.columns else 50,
                'Volatility_5D': df_proc['Volatility_5D'].iloc[-1] if 'Volatility_5D' in df_proc.columns else 0,
                'Volatility_20D': df_proc['Volatility_20D'].iloc[-1] if 'Volatility_20D' in df_proc.columns else 0,
                'Price_vs_MA20': df_proc['MA_Div'].iloc[-1] if 'MA_Div' in df_proc.columns else 0,
                'MACD_Histogram': df_proc['MACD_Histogram'].iloc[-1] if 'MACD_Histogram' in df_proc.columns else 0
            }
            
            # Generate Gemini Analysis
            with st.spinner("🧠 Generating AI Expert Analysis..."):
                gemini_result = generate_gemini_analysis(
                    stock_symbol=selected_stock,
                    current_price=current_price,
                    predicted_prices=future_prices,
                    metrics=metrics,
                    fundamentals=fundamentals,
                    sentiment_summary=daily_sentiment,
                    technical_indicators=technical_indicators,
                    volatility_data=df_proc['Volatility_5D'].iloc[-1] if 'Volatility_5D' in df_proc.columns else 0
                )
                # Check if we got Gemini response or fallback
                gemini_analysis = gemini_result
                gemini_used = "template mode" not in gemini_result.lower() and "Analysis generated using template" not in gemini_result
            
            # Display Analysis in Premium Card
            forecast_return = ((future_prices['Predicted Price'].iloc[-1] - current_price) / current_price) * 100 if not future_prices.empty else 0
            
            # Determine color scheme based on prediction
            if forecast_return > 3:
                gradient_colors = "linear-gradient(135deg, #1a472a 0%, #2d5a3d 50%, #3d6b4f 100%)"
                border_color = "#00ff88"
                verdict_emoji = "🟢"
                verdict_text = "BULLISH"
            elif forecast_return < -3:
                gradient_colors = "linear-gradient(135deg, #4a1a1a 0%, #5a2d2d 50%, #6b3d3d 100%)"
                border_color = "#ff4444"
                verdict_emoji = "🔴"
                verdict_text = "BEARISH"
            else:
                gradient_colors = "linear-gradient(135deg, #3a3a1a 0%, #4a4a2d 50%, #5a5a3d 100%)"
                border_color = "#ffaa00"
                verdict_emoji = "🟡"
                verdict_text = "NEUTRAL"
            
            # Top Quick Verdict Card
            st.markdown(f"""
            <div style="
                background: {gradient_colors};
                padding: 25px;
                border-radius: 15px;
                border: 2px solid {border_color};
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                margin-bottom: 20px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span style="font-size: 40px;">{verdict_emoji}</span>
                        <span style="font-size: 28px; font-weight: bold; color: white; margin-left: 10px;">{selected_stock}</span>
                        <span style="font-size: 18px; color: #aaa; margin-left: 10px;">Quick Verdict</span>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 32px; font-weight: bold; color: {border_color};">{verdict_text}</div>
                        <div style="font-size: 16px; color: #ccc;">Forecast: {forecast_return:+.2f}% | Accuracy: {metrics['accuracy']:.1f}%</div>
                    </div>
                </div>
                <hr style="border: 1px solid {border_color}33; margin: 15px 0;">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center;">
                    <div>
                        <div style="font-size: 12px; color: #888;">Current Price</div>
                        <div style="font-size: 18px; font-weight: bold; color: white;">₹{current_price:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 12px; color: #888;">{forecast_days}-Day Target</div>
                        <div style="font-size: 18px; font-weight: bold; color: {border_color};">₹{future_prices['Predicted Price'].iloc[-1]:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 12px; color: #888;">Model Confidence</div>
                        <div style="font-size: 18px; font-weight: bold; color: white;">{"High" if metrics['accuracy'] > 60 else "Medium" if metrics['accuracy'] > 52 else "Low"}</div>
                    </div>
                    <div>
                        <div style="font-size: 12px; color: #888;">Volatility</div>
                        <div style="font-size: 18px; font-weight: bold; color: white;">{df_proc['Volatility_5D'].iloc[-1]*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Gemini Detailed Analysis Card
            analysis_mode = "✨ Powered by Gemini AI" if gemini_used else "📝 Template Analysis"
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 25px;
                border-radius: 15px;
                border: 1px solid #0f3460;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            ">
                <h3 style="color: #e94560; margin-bottom: 15px;">🧠 AI Expert Analysis <span style="font-size: 14px; color: {'#00ff88' if gemini_used else '#ffaa00'};">({analysis_mode})</span></h3>
                <div style="color: #eee; line-height: 1.8;">
                    {gemini_analysis.replace(chr(10), '<br>')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Forecast Plot
            st.subheader("📈 Price Forecast Visualization")
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(
                x=df_stock.index[-60:], 
                y=df_stock['Close'][-60:], 
                name="Historical",
                line=dict(color='#00d4ff', width=2)
            ))
            fig_forecast.add_trace(go.Scatter(
                x=future_prices.index, 
                y=future_prices['Predicted Price'], 
                name="AI Forecast", 
                line=dict(dash='dot', color='#ff6b6b', width=2)
            ))
            fig_forecast.update_layout(
                template="plotly_dark",
                title=f"{selected_stock} - {forecast_days} Day Price Forecast",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            with st.expander("📊 View Detailed Forecast Table"):
                st.dataframe(future_prices.style.format("{:.2f}"))

    # ======================================
    # TAB 2: Dynamic Fusion
    # ======================================
    with tab2:
        st.header("🔬 Dynamic Fusion Framework")
        
        # Premium description card
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">Bayesian Multi-Expert System</h4>
            <p style="color: #eee;">This advanced system dynamically combines three specialized AI models:</p>
            <ul style="color: #aaa;">
                <li><strong style="color: #00d4ff;">Technical Expert</strong> - GRU neural network analyzing price patterns</li>
                <li><strong style="color: #00ff88;">Sentiment Expert</strong> - FinBERT transformer analyzing news</li>
                <li><strong style="color: #ff6b6b;">Volatility Expert</strong> - MLP analyzing India VIX & market fear</li>
            </ul>
            <p style="color: #888; font-size: 12px;">Weights are adjusted using Bayesian uncertainty: w = exp(-σ²) / Σ exp(-σ²)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if enable_dynamic_fusion:
            vix_data = get_india_vix_data(start_date, end_date)
            fusion_framework = DynamicFusionFramework()
            
            # Format sentiment
            fmt_sentiment = {k: [(s, c) for s, c in v] for k, v in daily_sentiment.items()}
            
            with st.spinner("🧠 Training Expert Models (this may take a moment)..."):
                try:
                    fusion_framework.train_models(df_stock, fmt_sentiment, vix_data)
                    
                    # Predict recent
                    recent_preds = []
                    
                    # Simulation loop for last 15 days
                    sim_range = df_stock.index[-15:]
                    
                    for date in sim_range:
                        try:
                            curr_idx = df_stock.index.get_loc(date)
                            if curr_idx < 30:
                                continue  # Need minimum data
                            stock_slice = df_stock.iloc[:curr_idx]
                            
                            res = fusion_framework.predict(stock_slice, fmt_sentiment, vix_data)
                            recent_preds.append({
                                'Date': date,
                                'Actual': df_stock.loc[date, 'Close'],
                                'Fusion': res['combined_prediction'],
                                'Tech_W': res['weights']['technical'],
                                'Sent_W': res['weights']['sentiment'],
                                'Vol_W': res['weights']['volatility']
                            })
                        except Exception as e:
                            continue
                    
                    if recent_preds:
                        res_df = pd.DataFrame(recent_preds).set_index('Date')
                        
                        # Current Weights Display
                        st.subheader("📊 Current Expert Weights")
                        latest_weights = recent_preds[-1]
                        
                        w1, w2, w3 = st.columns(3)
                        w1.metric("Technical 📈", f"{latest_weights['Tech_W']*100:.1f}%", 
                                 help="Weight assigned to price pattern analysis")
                        w2.metric("Sentiment 📰", f"{latest_weights['Sent_W']*100:.1f}%",
                                 help="Weight assigned to news sentiment")
                        w3.metric("Volatility ⚡", f"{latest_weights['Vol_W']*100:.1f}%",
                                 help="Weight assigned to VIX/fear analysis")
                        
                        st.markdown("---")
                        
                        # Weights Chart
                        st.subheader("📈 Weight Evolution (Last 15 Days)")
                        fig_w = go.Figure()
                        fig_w.add_trace(go.Scatter(
                            x=res_df.index, y=res_df['Tech_W'], 
                            name='Technical', stackgroup='one',
                            line=dict(color='#00d4ff')
                        ))
                        fig_w.add_trace(go.Scatter(
                            x=res_df.index, y=res_df['Sent_W'], 
                            name='Sentiment', stackgroup='one',
                            line=dict(color='#00ff88')
                        ))
                        fig_w.add_trace(go.Scatter(
                            x=res_df.index, y=res_df['Vol_W'], 
                            name='Volatility', stackgroup='one',
                            line=dict(color='#ff6b6b')
                        ))
                        fig_w.update_layout(
                            template="plotly_dark",
                            title="Dynamic Expert Influence Over Time",
                            yaxis_title="Weight",
                            yaxis=dict(tickformat='.0%'),
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig_w, use_container_width=True)
                        
                        # Interpretation
                        dominant = max(['Technical', 'Sentiment', 'Volatility'], 
                                      key=lambda x: latest_weights[f"{x[:4] if x != 'Volatility' else 'Vol'}_W"])
                        st.info(f"💡 **Insight:** The {dominant} expert currently has the highest influence, suggesting the model finds {dominant.lower()} signals most reliable for this stock right now.")
                    else:
                        st.warning("⚠️ Could not generate predictions. Insufficient data for dynamic fusion analysis.")
                except Exception as e:
                    st.error(f"❌ Dynamic Fusion training failed: {str(e)}")
                    st.info("This can happen with limited data or if the stock has unusual trading patterns. Try selecting a different date range or stock.")
        else:
            st.info("👆 Enable 'Dynamic Fusion Framework' in the sidebar to use this feature.")

    # ======================================
    # TAB 3: Technicals & Risk
    # ======================================
    with tab3:
        st.header("Risk Management & Technicals")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Key Levels (Fibonacci)")
            fibs = RiskManager.calculate_fibonacci_levels(df_stock)
            fib_df = pd.DataFrame.from_dict(fibs, orient='index', columns=['Price'])
            st.table(fib_df.style.format("₹{:.2f}"))
            
        with c2:
            st.subheader("Risk Metrics")
            atr = RiskManager.calculate_atr(df_stock)
            st.metric("ATR (Volatility)", f"₹{atr:.2f}")
            
            # Trade Setup Calculator
            st.markdown("### 🛡️ Trade Setup Calculator")
            
            # Get latest prediction direction
            pred_price = future_prices['Predicted Price'].iloc[-1] if not future_prices.empty else current_price
            
            setup = RiskManager.get_trade_setup(current_price, pred_price, atr, metrics['accuracy']/100)
            
            st.write(f"**Action:** {setup['Direction']}")
            st.write(f"**Entry:** ₹{setup['Entry']:.2f}")
            st.write(f"**Stop Loss:** ₹{setup['Stop Loss']:.2f} ({(setup['Stop Loss']-current_price)/current_price*100:.2f}%)")
            st.write(f"**Target:** ₹{setup['Target']:.2f} ({(setup['Target']-current_price)/current_price*100:.2f}%)")
            st.metric("Risk/Reward Ratio", f"1:{setup['Risk/Reward']:.2f}")

    # ======================================
    # TAB 4: Fundamentals
    # ======================================
    with tab4:
        st.header("Fundamental Health")
        
        f_cols = ["Forward P/E", "PEG Ratio", "Price/Book", "Debt/Equity", "ROE", "Profit Margins"]
        
        # create nice grid
        fc1, fc2, fc3 = st.columns(3)
        for i, key in enumerate(f_cols):
            val = fundamentals.get(key)
            if pd.isna(val): val = "N/A"
            elif isinstance(val, (int, float)): val = f"{val:.2f}"
            
            if i % 3 == 0: fc1.metric(key, val)
            elif i % 3 == 1: fc2.metric(key, val)
            else: fc3.metric(key, val)
            
        st.caption("Data source: Yahoo Finance")

    # ======================================
    # TAB 5: Backtesting
    # ======================================
    with tab5:
        st.header("🛠️ Strategy Backtest")
        st.write("Simulating trading strategy based on Hybrid Model signals over the selected period.")
        
        # Access results from session state
        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            backtest_data = st.session_state['results_df'].copy()
            
            if 'Predicted_Return' in backtest_data.columns and len(backtest_data) > 0:
                # Auto-run backtest
                with st.spinner("Running backtest simulation..."):
                    # Signal: 1 if Predicted Return > 0 (Positive), -1 if < 0
                    backtest_data['Signal'] = np.where(backtest_data['Predicted_Return'] > 0.001, 1, 0)
                    backtest_data['Signal'] = np.where(backtest_data['Predicted_Return'] < -0.001, -1, backtest_data['Signal'])
                    
                    # Run vector backtest
                    bt = VectorizedBacktester(backtest_data, backtest_data['Signal'])
                    res = bt.run_backtest()
                    
                    # Display Metrics in Premium Cards
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
                        <h3 style="color: #e94560; margin-bottom: 15px;">📊 Backtest Performance Summary</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    b1, b2, b3, b4 = st.columns(4)
                    
                    # Color code based on performance
                    ret_color = "normal" if res['Total Return'] > 0 else "inverse"
                    sr_color = "normal" if res['Sharpe Ratio'] > 1 else "off" if res['Sharpe Ratio'] > 0 else "inverse"
                    
                    b1.metric("Total Return", f"{res['Total Return']*100:.2f}%", delta=f"{res['Total Return']*100:.1f}%")
                    b2.metric("Sharpe Ratio", f"{res['Sharpe Ratio']:.2f}", help="Above 1 is good, above 2 is excellent")
                    b3.metric("Max Drawdown", f"{res['Max Drawdown']*100:.2f}%", delta=f"{res['Max Drawdown']*100:.1f}%", delta_color="inverse")
                    b4.metric("Win Rate", f"{res['Win Rate']*100:.1f}%", help="Percentage of profitable trades")
                    
                    st.markdown("---")
                    
                    # Equity Curve with Plotly
                    st.subheader("📈 Equity Curve")
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=res['Equity Curve'].index,
                        y=res['Equity Curve'].values,
                        name="Portfolio Value",
                        fill='tozeroy',
                        fillcolor='rgba(0, 212, 255, 0.2)',
                        line=dict(color='#00d4ff', width=2)
                    ))
                    fig_equity.update_layout(
                        template="plotly_dark",
                        title="Strategy Equity Curve (₹100,000 Initial Capital)",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value (₹)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    # Trading Signals Table
                    with st.expander("📋 View Trading Signals & Returns"):
                        display_df = backtest_data[['Actual_Return', 'Predicted_Return', 'Signal']].copy()
                        display_df['Signal_Label'] = display_df['Signal'].map({1: '🟢 LONG', 0: '⚪ HOLD', -1: '🔴 SHORT'})
                        st.dataframe(display_df.tail(50).style.format({
                            'Actual_Return': '{:.4f}',
                            'Predicted_Return': '{:.4f}'
                        }))
            else:
                st.warning("⚠️ Insufficient prediction data for backtesting. Please run analysis on a stock with more historical data.")
        else:
            st.info("👈 Please run stock analysis from the Dashboard tab first to generate backtest data.")
            st.markdown("""
            ### How Backtesting Works:
            1. The AI model generates predictions for each historical day
            2. Based on predicted returns, signals are generated (BUY/SELL/HOLD)
            3. Performance is calculated assuming trades are executed at next day's close
            4. Metrics like Sharpe Ratio and Max Drawdown help evaluate strategy quality
            """)

    # ======================================
    # TAB 6: Visual Analysis
    # ======================================
    with tab6:
        st.header("👁️ AI Visual Pattern Analysis")
        
        # Premium description card
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;">
            <h4 style="color: #e94560;">🧠 Computer Vision for Charts</h4>
            <p style="color: #eee;">This module uses YOLOv8 (You Only Look Once) neural networks trained on stock charts to detect:</p>
            <ul style="color: #aaa;">
                <li>Classic patterns: Head & Shoulders, Double Top/Bottom, Triangles</li>
                <li>Support & Resistance zones</li>
                <li>Trend direction signals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if not VISUAL_AI_AVAILABLE:
            st.error("⚠️ Visual AI libraries not installed.")
            st.markdown("""
            ### Installation Required:
            To enable Visual Pattern Analysis, install the following:
            ```bash
            pip install ultralytics opencv-python
            ```
            
            **Note:** This feature uses pre-trained YOLO models from Hugging Face:
            - `foduucom/stockmarket-pattern-detection-yolov8`
            - `foduucom/stockmarket-future-prediction`
            """)
        else:
            col_v1, col_v2 = st.columns([3, 1])
            
            with col_v1:
                st.subheader("📸 Chart Analysis")
                
                # Show preview chart before running inference
                st.markdown("**Current Price Action (Last 50 Days)**")
                preview_data = df_stock.tail(50)
                fig_preview = go.Figure(data=[go.Candlestick(
                    x=preview_data.index,
                    open=preview_data['Open'],
                    high=preview_data['High'],
                    low=preview_data['Low'],
                    close=preview_data['Close'],
                    name='Price'
                )])
                fig_preview.update_layout(
                    template="plotly_dark",
                    height=300,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_rangeslider_visible=False,
                    title="Click button below to detect patterns using AI"
                )
                st.plotly_chart(fig_preview, use_container_width=True)
                
                # Use a unique key for the button to prevent rerun issues
                if st.button("🔍 Run Visual Inference", type="primary", use_container_width=True, key="visual_inference_btn"):
                    st.session_state['run_visual_inference'] = True
                
                # Run inference if requested
                if st.session_state.get('run_visual_inference', False):
                    with st.spinner("Generating Chart & Running Neural Networks..."):
                        try:
                            analyst = VisualPatternAnalyst()
                            img, results, bias = analyst.analyze_patterns(df_stock)
                            
                            if img is not None:
                                st.session_state['visual_image'] = img
                                st.session_state['visual_bias'] = bias
                                st.session_state['visual_results'] = results
                                st.session_state['visual_error'] = None
                            else:
                                st.session_state['visual_error'] = bias
                                st.session_state['visual_image'] = None
                        except Exception as e:
                            st.session_state['visual_error'] = str(e)
                            st.session_state['visual_image'] = None
                        
                        st.session_state['run_visual_inference'] = False
                        st.rerun()
                
                # Display results from session state
                if st.session_state.get('visual_image') is not None:
                    st.image(st.session_state['visual_image'], caption="AI Annotated Chart (Detected Patterns)", use_column_width=True)
                    
                    if st.session_state.get('visual_results'):
                        st.subheader("🎯 Detected Patterns")
                        results_df_visual = pd.DataFrame(st.session_state['visual_results'])
                        st.dataframe(results_df_visual)
                
                if st.session_state.get('visual_error'):
                    st.error(f"Analysis failed: {st.session_state['visual_error']}")
                    st.info("""
                    **Troubleshooting:**
                    - The YOLO models need to be downloaded from HuggingFace
                    - This may fail due to network issues or firewall
                    - Try running: `pip install huggingface_hub` and restart the app
                    """)
                
                # Show conclusion if available
                if 'visual_bias' in st.session_state and st.session_state['visual_bias']:
                    bias_color = "#00ff88" if st.session_state['visual_bias'] == "Bullish" else "#ff4444" if st.session_state['visual_bias'] == "Bearish" else "#ffaa00"
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1a2e1a 0%, #2d3e2d 100%); padding: 15px; border-radius: 10px; border-left: 4px solid {bias_color};">
                        <h4 style="color: {bias_color};">Visual AI Conclusion: {st.session_state['visual_bias']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_v2:
                st.subheader("📊 Pattern Guide")
                st.markdown("""
                **Bullish Patterns:**
                - 🟢 Double Bottom
                - 🟢 Inverse H&S
                - 🟢 Ascending Triangle
                
                **Bearish Patterns:**
                - 🔴 Double Top
                - 🔴 Head & Shoulders
                - 🔴 Descending Triangle
                
                **Neutral:**
                - 🟡 Symmetrical Triangle
                - 🟡 Rectangle
            """)
else:
    st.error("❌ No data found for this stock. Please check the ticker symbol or try a different date range.")