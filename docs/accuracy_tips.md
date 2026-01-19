# Accuracy Improvement Tips for ProTrader AI

This guide provides practical suggestions for improving the prediction accuracy of your stock analysis.

## 1. Use More Training Data

- **Current:** Uses data from your selected date range (typically 1-2 years)
- **Recommendation:** Use at least 3+ years of historical data
- **How:** Set the start date to 2021 or earlier in the sidebar

## 2. Enable AutoML Optimization

- Located in sidebar under "Advanced Settings"
- This performs hyperparameter tuning for XGBoost and GRU models
- **Tradeoff:** Takes longer to run but often improves accuracy by 5-10%

## 3. Ensure FII/DII Data is Available

- FII/DII institutional flow data significantly impacts prediction quality
- If you see "Could not fetch FII/DII data" warning, the model loses an important signal
- **Workaround:** Try running the analysis at different times (NSE API has rate limits)

## 4. Improve Sentiment Sources

- Add **NEWS_API_KEY** to `.env` file for NewsAPI integration
- Add **REDDIT_CLIENT_ID** and **REDDIT_CLIENT_SECRET** for Reddit sentiment
- More sentiment sources = better fusion model weights

## 5. Choose Appropriate Forecast Horizon

- Shorter horizons (1-5 days) are generally more accurate
- Longer horizons (15-30 days) have higher uncertainty
- **Recommendation:** Use 5-10 days for best balance

## 6. Feature Engineering Tips

The model already uses these features, but understanding them helps:

- **Technical:** RSI, MACD, Moving Averages, Volatility
- **Sentiment:** News sentiment scores, Reddit/Twitter sentiment
- **Institutional:** FII/DII net flows, cumulative positions
- **Market Regime:** India VIX levels

## 7. Avoid Overfitting

- The model uses walk-forward validation (train on past, test on future)
- Check the "Directional Accuracy" metric - above 55% is good
- If accuracy is very high (>80%), the model may be overfitting

## 8. Model-Specific Suggestions

### XGBoost
- Works best with clear trends
- Struggles in highly volatile, range-bound markets

### GRU (Neural Network)
- Captures sequential patterns
- Needs more data to train effectively

### Dynamic Fusion
- Automatically weights models based on recent performance
- Enable "Dynamic Fusion Framework" in sidebar for this

## 9. External Factors to Consider

The model cannot account for:
- Unexpected news events (earnings surprises, regulatory changes)
- Black swan events (market crashes, geopolitical events)
- Insider trading or manipulation
- Corporate actions (stock splits, buybacks)

## 10. Realistic Expectations

- Stock prediction is inherently uncertain
- Even professional quants achieve 55-60% directional accuracy
- Use predictions as one input, not the sole decision factor
- Always apply proper risk management (stop losses, position sizing)
