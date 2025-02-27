# Stock Market Forecasting App

A modular desktop application built with Python and PyQt5 for forecasting stock prices. 
This app leverages multiple machine learning models—including deep learning (LSTM, GRU) and classical approaches (Ridge, Lasso, XGBoost, Random Forest), 
to predict future stock prices using historical data and log returns. 
The application provides interactive visualizations that display both full-range historical data and zoomed-in views of recent trends alongside future predictions.

## Features

- **Multi-Model Forecasting:**
  - **Deep Learning:** LSTM, GRU
  - **Linear Models:** Ridge, Lasso
  - **Tree-Based Models:** XGBoost, Random Forest
- **Interactive User Interface:**
  - Built using PyQt5 with an intuitive form to input stock tickers and select the forecasting model.
- **Data Visualization:**
  - Uses Matplotlib to display:
    - Full historical price charts with future predictions.
    - Zoomed-in charts of the last 6 months for detailed trend analysis.
- **Modular Architecture:**
  - Designed for easy integration of additional models and future enhancements.

## Technologies

- **Programming Language:** Python
- **Machine Learning Libraries:** PyTorch, scikit-learn, XGBoost
- **GUI Framework:** PyQt5
- **Data Visualization:** Matplotlib
- **Data Source:** yfinance
- **Others:** NumPy, Pandas

## Folder Structure:
StockPredictorApp/
├── app/
│   └── main_window.py       # PyQt5 GUI code
├── models/
│   ├── __init__.py
│   ├── lstm_model.py        # LSTM model definition
│   ├── gru_model.py         # GRU model definition
│   ├── regression_models.py # Ridge and Lasso implementations
│   ├── xgb_model.py         # XGBoost related functions
│   └── rf_model.py          # Random Forest related functions
├── utils/
│   ├── __init__.py
│   └── data_fetcher.py      # Data fetching, preprocessing, and sequence creation
├── requirements.txt         # List of required packages
└── main.py                  # Application entry point
