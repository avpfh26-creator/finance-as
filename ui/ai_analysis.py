"""
AI Analysis utilities.
Gemini integration and fallback analysis generation.
"""

import streamlit as st

from config.settings import GEMINI_API_KEY, ModelConfig, TradingConfig

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


def initialize_gemini():
    """
    Initialize Gemini API with safety settings.
    
    Returns:
        Gemini model object or None if unavailable
    """
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("⚠️ google-generativeai library not installed")
        return None
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
        st.sidebar.warning("⚠️ No Gemini API key configured")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        return model
    except Exception as e:
        st.sidebar.error(f"Gemini init error: {e}")
        return None


def generate_gemini_analysis(stock_symbol: str, current_price: float, 
                             predicted_prices, metrics: dict, fundamentals: dict,
                             sentiment_summary: dict, technical_indicators: dict,
                             volatility_data, fusion_weights: dict = None,
                             fii_dii_data=None, vix_data=None, patterns: list = None) -> str:
    """
    Generate comprehensive AI analysis using Google Gemini.
    
    Args:
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        predicted_prices: DataFrame with predicted prices
        metrics: Dictionary with model metrics (accuracy, rmse)
        fundamentals: Dictionary with fundamental data
        sentiment_summary: Dictionary with sentiment data
        technical_indicators: Dictionary with technical indicators
        volatility_data: Volatility value
        fusion_weights: Dynamic fusion model weights (optional)
        fii_dii_data: DataFrame with FII/DII data (optional)
        vix_data: DataFrame/Float with VIX data (optional)
        patterns: List of detected chart patterns (optional)
    
    Returns:
        Markdown-formatted analysis string
    """
    model = initialize_gemini()
    if model is None:
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices, 
                                          metrics, sentiment_summary, technical_indicators)
    
    # Calculate key metrics for prompt
    if not predicted_prices.empty:
        price_forecast_end = predicted_prices['Predicted Price'].iloc[-1]
        forecast_days = len(predicted_prices)
    else:
        price_forecast_end = current_price
        forecast_days = 0
    
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    
    # Prepare sentiment summary text
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
    
    # Prepare FII/DII text
    fii_dii_text = "Data Unavailable"
    if fii_dii_data is not None and not fii_dii_data.empty:
        last_row = fii_dii_data.iloc[-1]
        fii_net = last_row.get('FII_Net', 0) / 1e7  # Convert to Cr
        dii_net = last_row.get('DII_Net', 0) / 1e7
        fii_dii_text = f"FII Net: ₹{fii_net:+.2f}Cr | DII Net: ₹{dii_net:+.2f}Cr (Latest Session)"
    
    # Prepare VIX text
    vix_text = "Data Unavailable"
    if vix_data is not None:
        try:
            val = float(vix_data.iloc[-1]['Close']) if hasattr(vix_data, 'iloc') else float(vix_data)
            vix_text = f"{val:.2f}"
            if val > 20: vix_text += " (High Volatility)"
            elif val < 12: vix_text += " (Low Volatility)"
            else: vix_text += " (Normal)"
        except: pass

    # Prepare Patterns text
    patterns_text = "No specific classic patterns detected."
    if patterns:
        p_list = [f"{p.get('Pattern')} ({p.get('Type')}, Conf: {p.get('Confidence')}%)" for p in patterns[:3]]
        patterns_text = ", ".join(p_list)
    
    # Fusion weights summary
    fusion_text = "Not available"
    if fusion_weights:
        fusion_text = f"Technical: {fusion_weights.get('technical', 0)*100:.1f}%, " \
                      f"Sentiment: {fusion_weights.get('sentiment', 0)*100:.1f}%, " \
                      f"Volatility: {fusion_weights.get('volatility', 0)*100:.1f}%"
    
    # Expert prompt - optimized for clean markdown output
    prompt = f"""You are a senior quantitative strategist. Analyze {stock_symbol} and provide a concise, professional analysis.

**MARKET DATA:**
- Stock: {stock_symbol} @ ₹{current_price:,.2f}
- India VIX: {vix_text}
- Institutional Flows: {fii_dii_text}

**AI MODEL FORECAST ({forecast_days} days):**
- Target: ₹{price_forecast_end:,.2f} ({forecast_return:+.2f}%)
- Accuracy: {metrics.get('accuracy', 0):.1f}% | RMSE: {metrics.get('rmse', 0):.4f}

**TECHNICALS:**
- Patterns: {patterns_text}
- RSI: {technical_indicators.get('RSI', 'N/A')} | MACD: {technical_indicators.get('MACD_Histogram', 'N/A')}
- Volatility: {technical_indicators.get('Volatility_20D', 0)*100:.2f}%

**SENTIMENT:** {sentiment_text}
**VALUATION:** P/E {fundamentals.get('Forward P/E', 'N/A')} | PEG {fundamentals.get('PEG Ratio', 'N/A')}

---

**PROVIDE YOUR ANALYSIS IN THIS EXACT FORMAT:**

### 🎯 Verdict: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
**Confidence:** [High/Medium/Low]

### 📊 Key Points
- **AI Signal:** [One sentence on what the model predicts]
- **Technical Setup:** [One sentence on chart structure]
- **Institutional Flow:** [One sentence on FII/DII implication]

### ⚠️ Risks
- [Risk 1 in one sentence]
- [Risk 2 in one sentence]

### 🎯 Trade Setup
- **Entry:** ₹[price]
- **Target:** ₹[price] ([X]% upside)
- **Stop Loss:** ₹[price] ([X]% downside)

**RULES:**
1. Use bullet points, not paragraphs
2. Be specific with numbers
3. Maximum 5 sentences per section
4. No disclaimers or caveats
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Gemini API call failed: {str(e)[:100]}. Using fallback analysis.")
        return generate_fallback_analysis(stock_symbol, current_price, predicted_prices,
                                          metrics, sentiment_summary, technical_indicators)


def generate_fallback_analysis(stock_symbol: str, current_price: float,
                               predicted_prices, metrics: dict,
                               sentiment_summary: dict, technical_indicators: dict) -> str:
    """
    Generate structured analysis without Gemini API (template-based fallback).
    
    Args:
        stock_symbol: Stock ticker symbol
        current_price: Current stock price
        predicted_prices: DataFrame with predicted prices
        metrics: Dictionary with model metrics
        sentiment_summary: Dictionary with sentiment data
        technical_indicators: Dictionary with technical indicators
    
    Returns:
        Markdown-formatted analysis string
    """
    if not predicted_prices.empty:
        price_forecast_end = predicted_prices['Predicted Price'].iloc[-1]
    else:
        price_forecast_end = current_price
    
    forecast_return = ((price_forecast_end - current_price) / current_price) * 100
    accuracy = metrics.get('accuracy', 50)
    
    # Determine verdict
    if accuracy < ModelConfig.LOW_CONFIDENCE_THRESHOLD:
        confidence = "Low Confidence"
    elif accuracy < ModelConfig.MEDIUM_CONFIDENCE_THRESHOLD:
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
    
    volatility = technical_indicators.get('Volatility_5D', 0)
    vol_text = "High volatility detected - position size accordingly" if volatility > 0.02 else "Normal volatility levels"
    
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
- {vol_text}
- External market factors may override technical signals

*Analysis generated using template mode.*
"""


def generate_recommendation(predicted_prices, current_price: float, 
                            accuracy: float, avg_sentiment: float) -> tuple:
    """
    Generate investment recommendation based on predictions.
    
    Args:
        predicted_prices: DataFrame with predicted prices
        current_price: Current stock price
        accuracy: Model directional accuracy
        avg_sentiment: Average sentiment score
    
    Returns:
        Tuple of (recommendation_label, reason_text)
    """
    avg_prediction = predicted_prices['Predicted Price'].mean()
    price_change = ((avg_prediction - current_price) / current_price) * 100
    
    # Enhanced sentiment factor with confidence scaling
    sentiment_factor = 1 + (avg_sentiment * (accuracy/100))
    adjusted_change = price_change * sentiment_factor
    
    # Modified thresholds with confidence weighting
    confidence_weight = accuracy / 100
    
    if adjusted_change > TradingConfig.STRONG_BUY_THRESHOLD * confidence_weight and accuracy > 72:
        return "STRONG BUY", "High confidence in significant price increase"
    elif adjusted_change > TradingConfig.BUY_THRESHOLD * confidence_weight and accuracy > 65:
        return "BUY", "Good confidence in moderate price increase"
    elif adjusted_change > 0 and accuracy > 60:
        return "HOLD (Positive)", "Potential for slight growth"
    elif adjusted_change < TradingConfig.STRONG_SELL_THRESHOLD * confidence_weight and accuracy > 72:
        return "STRONG SELL", "High confidence in significant price drop"
    elif adjusted_change < TradingConfig.SELL_THRESHOLD * confidence_weight and accuracy > 65:
        return "SELL", "Good confidence in moderate price drop"
    elif adjusted_change < 0 and accuracy > 60:
        return "HOLD (Caution)", "Potential for slight decline"
    else:
        return "HOLD", "Unclear direction - consider other factors"
