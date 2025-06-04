# Financial Market Prediction Models Comparison

## Project Overview
This project implements and compares various machine learning models for stock market prediction, as part of the TFG (Trabajo de Fin de Grado) on comparative analysis of ML models for financial markets.

## Current Implementation Status

### ✅ Fully Implemented
- **Linear Regression/Logistic Regression**: Complete with feature scaling
- **Decision Tree**: Both regression and classification variants
- **XGBoost**: Complete with early stopping and hyperparameter tuning
- **Data Pipeline**: Full preprocessing, feature engineering, and target creation
- **CLI Interface**: Interactive command-line system for training and evaluation
- **Evaluation System**: Comprehensive metrics including regime-based analysis
- **ARIMA**: Statistical time series model
- **LSTM**: Recurrent neural network for sequence modeling
- **GRU**: Another recurrent neural network for sequence modeling
- **CNN**: Convolutional neural network for time series

## Prediction Targets
- **return_1d**: Next trading day return (regression)
- **direction_1d**: Binary classification of price movement direction
- **return_5d**: 5-trading-day forward return (regression)

## Key Design Decisions

### Hybrid Approach with Sector Awareness
- Universal models trained on pooled data from all stocks
- Sector included as feature (one-hot encoded) for sector-specific patterns

### Data Split Strategy
- Time-based splits: 70% train, 15% validation, 15% test
- Same test period across all models for fair comparison

## Technical Features Used
- **RSI** (14-day): Momentum oscillator
- **MACD** (12, 26, 9): Trend-following indicator
- **Bollinger Bands** (20-day, 2σ): Volatility bands
- **SMA** (50 and 200-day): Moving averages
- **OBV**: On-Balance Volume
- **Market Regime Features**: VIX-based volatility, GDP-based economic cycles
- **Sector Encoding**: One-hot encoded sectors

## Usage

### CLI Interface
```bash
python main.py
```

### Available Commands
- `models` - Show status of trained/untrained models
- `train <model_name> [target_name]` - Train specific model (e.g., `train xgboost return_1d`)
- `evaluate [model_names...]` - Evaluate trained models with regime analysis
- `help` - Show available commands
- `exit` - Exit program

### Example Session
```
> models                          # Check model status
> train linear                    # Train linear model on all targets
> train xgboost return_1d         # Train XGBoost on specific target
> evaluate                        # Evaluate all trained models
> evaluate linear xgboost         # Evaluate specific models
```

## Project Structure
```
src/
├── config/          # Configuration management
├── data/            # Data download, preprocessing, feature engineering
├── models/          # Model implementations (3 working, 3 planned)
├── evaluation/      # Metrics and regime-based analysis
├── utils/           # Model persistence utilities
└── main.py          # CLI interface
```

## Data Sources
- **Stock Data**: Yahoo Finance (AAPL, MSFT, AMZN, TSLA, ^GSPC, ^DJI, ^IXIC)
- **Economic Data**: FRED (VIX, GDP, 10-Year Inflation Expectations)
- **Time Period**: 2015-2024

## Current Capabilities
- ✅ Download and preprocess financial data
- ✅ Generate technical indicators and regime features
- ✅ Train and evaluate tabular ML models
- ✅ Regime-based performance analysis (volatility, sectors, economic cycles)
- ✅ Model persistence (save/load trained models)
- ✅ Trading metrics (Sharpe ratio, hit rate, directional accuracy)