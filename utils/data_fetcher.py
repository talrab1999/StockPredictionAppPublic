import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fetch_stock_data(ticker, start_date="2018-01-01", end_date="2025-08-23"):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]
    df.dropna(inplace=True)
    return df

def compute_log_returns(df):
    df = df.copy()
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)
    return df

def scale_data(series, scaler=None):
    #shape (n_samples, 1)
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(series)
    else:
        scaled = scaler.transform(series)
    return scaled, scaler

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)
