# Financial Market Prediction Models Comparison

## Project Overview
This project implements and compares various machine learning models for stock market prediction, as part of the TFG (Trabajo de Fin de Grado) on comparative analysis of ML models for financial markets.

## Prediction Targets
- **1-day return**: Next trading day return (regression & classification)
- **5-day return**: 5-trading-day forward return (regression)
- **Direction**: Binary classification of price movement direction

## Key Design Decisions

### Hybrid Approach with Sector Awareness
- Universal models trained on pooled data from all stocks
- Sector included as feature (one-hot encoded) for sector-specific patterns
- Stock ticker NOT included as feature to enable generalization to unseen stocks

### Data Split Strategy
- Time-based splits: 70% train, 15% validation, 15% test
- No random shuffling to maintain temporal integrity
- Same test period across all models for fair comparison

## Insights from Preprocessing Analysis
1. **Weak but non-random correlations**: Individual indicators show correlations between -0.24 and 0.10 with 5-day returns
2. **Sector-specific patterns**:
   - SMA performs better for automotive stocks
   - RSI for e-commerce
   - OBV for technology stocks
3. **Market regime effects**: High volatility (VIX > 25) shifts correlations negative
4. **Economic cycle impact**: Most indicators perform better during contractions

## Models Implemented
- **Traditional**: Linear Regression, ARIMA
- **Tree-based**: XGBoost
- **Deep Learning**: LSTM, CNN

## Technical Indicators Used
- RSI (14-day)
- MACD (12, 26, 9)
- Bollinger Bands (20-day, 2σ)
- SMA (50 and 200-day)
- OBV

## Usage
```bash
python main.py
```