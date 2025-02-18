import sys
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget, QApplication, QMainWindow, QComboBox
from PyQt5.QtCore import QFile
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils.data_fetcher import fetch_stock_data, compute_log_returns, scale_data, create_sequences
from models.lstm_model import LSTMModel
from models.gru_model import GRUModel
from models.train_and_predict import train_model, predict_future_LSTM_or_GRU, predict_future_linear_or_xgboost_or_rf
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_ui()
        # self.dark_mode = True
        # self.load_theme()
        
        #toggle Button
        # self.toggle_button = QPushButton("Switch to Light Mode")
        # self.toggle_button.clicked.connect(self.toggle_theme)

        # layout = QVBoxLayout()
        # layout.addWidget(self.toggle_button)

        # container = QWidget()
        # container.setLayout(layout)
        # self.setCentralWidget(container)

    def load_theme(self):
        theme_file = "dark_mode.qss" if self.dark_mode else "light_mode.qss"
        qss_file = QFile(theme_file)
        if qss_file.open(QFile.ReadOnly | QFile.Text):
            stream = qss_file.readAll()
            qss_file.close()
            self.setStyleSheet(str(stream, encoding='utf-8'))

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.load_theme()
        self.toggle_button.setText("Switch to Dark Mode" if not self.dark_mode else "Switch to Light Mode")
    
    def init_ui(self):
        layout = QVBoxLayout()
        form_layout = QHBoxLayout()

        self.stock_label = QLabel("Stock Ticker:")
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("e.g., AAPL")
        form_layout.addWidget(self.stock_label)
        form_layout.addWidget(self.stock_input)

        self.model_selector = QComboBox(self)
        self.model_selector.addItems(["LSTM", "GRU", "Ridge", "Lasso", "XGBoost", "Random Forest"])  # Extend with more models later if needed
        form_layout.addWidget(self.model_selector)

        self.run_button = QPushButton("Run Prediction")
        self.run_button.clicked.connect(self.run_prediction)
        form_layout.addWidget(self.run_button)
        
        layout.addLayout(form_layout)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def run_prediction(self):
        ticker = self.stock_input.text().strip().upper()
        selected_model = self.model_selector.currentText()
        if not ticker:
            return      
        print(f"Running prediction with {selected_model} model...")

        #fetch data from yahoo
        df = fetch_stock_data(ticker)
        if df.empty:
            print("Please enter a valid stock")
            return
        
        #using log_return (not prices)
        df = compute_log_returns(df)
        data = df[['log_return']].values
        data_scaled, scaler = scale_data(data)
        seq_length = 60
        X, y = create_sequences(data_scaled, seq_length)
        future_days = 10

        if selected_model in ["LSTM", "GRU"]:
            #for deep learning, convert data to torch tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            from torch.utils.data import DataLoader, TensorDataset
            train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

            if selected_model == "LSTM":
                model = LSTMModel().to(self.device)
            elif selected_model == "GRU":
                model = GRUModel().to(self.device)
 
            model = train_model(model, train_loader, num_epochs=30, lr=0.001, device=self.device)
            future_log_returns = predict_future_LSTM_or_GRU(model, data_scaled, seq_length, scaler, future_days=future_days, device=self.device)

        elif selected_model in ["Linear Regression", "Ridge", "Lasso"]:
            #reshape data for lin reg (2D)
            X_lr = X.reshape(X.shape[0], seq_length)
            if selected_model == "Linear Regression":
                lr_model = LinearRegression()
            elif selected_model == "Ridge":
                lr_model = Ridge(alpha=10000.0)
            elif selected_model == "Lasso":
                lr_model = Lasso(alpha= 0.1)
            lr_model.fit(X_lr, y)
            future_log_returns = predict_future_linear_or_xgboost_or_rf(lr_model, data_scaled, seq_length, future_days=future_days)

        elif selected_model == "XGBoost":
            #reshape data for XGBoost (2D)
            X_xgb = X.reshape(X.shape[0], seq_length)

            xgb_model = xgb.XGBRegressor(
                n_estimators=100,       #higher more complex, may overfit     
                learning_rate=0.001,    #higher it is it may overfit
                max_depth=6,            #higher may overfit
                subsample=0.6,          #higher (max 1) may overfit
                colsample_bytree=0.6,   #same as above
            )

            xgb_model.fit(X_xgb, y.ravel()) 
            future_log_returns = predict_future_linear_or_xgboost_or_rf(xgb_model, data_scaled, seq_length, future_days=future_days)

        elif selected_model == "Random Forest":
            #reshape data for rf (2D)
            X_rf = X.reshape(X.shape[0], seq_length)

            rf_model = RandomForestRegressor(
                n_estimators=200,      #higher improve performance but takes more time
                max_depth=5,           #dont even have to cap, but it may overfit the higher it is 
            )

            rf_model.fit(X_rf, y.ravel())
            future_log_returns = predict_future_linear_or_xgboost_or_rf(rf_model, data_scaled, seq_length, future_days=future_days)
        else:
            print("Invalid model selection.")
            return

        last_price = df['Close'].iloc[-1]
        future_prices = []
        price = last_price
        for log_ret in future_log_returns:
            price = price * np.exp(log_ret[0])
            future_prices.append(price)

        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        self.figure.clear()
        #create two subplots: one for the full view and one zoomed in on the last 6 months
        ax_full = self.figure.add_subplot(211)
        ax_zoom = self.figure.add_subplot(212)

        #adjust space between graphs, can make bigger
        self.figure.subplots_adjust(hspace=0.5)
        
        #plot full historical data and future forecast on the top subplot
        ax_full.plot(df.index, df['Close'], label="Historical Prices", color="blue")
        ax_full.plot(future_dates, future_prices, label="Future Prices", color="red", linestyle="dashed")
        ax_full.set_title(f"{ticker} Future Price Forecast (Full View)")
        ax_full.set_xlabel("Date")
        ax_full.set_ylabel("Price (USD)")
        ax_full.legend()
        ax_full.grid()
        ax_full.tick_params(axis='x', rotation=45)
        
        #define the zoom window, last 3 months from the last date in df
        zoom_start_date = last_date - pd.DateOffset(months=3)
        df_zoom = df[df.index >= zoom_start_date]
        
        #plot zoomed historical data and overlay future forecast (even if the forecast is partly outside the zoom window)
        ax_zoom.plot(df_zoom.index, df_zoom['Close'], label="Historical Prices (Last 3 Months)", color="blue")
        #only include future predictions that are relevant to the zoom window
        ax_zoom.plot(future_dates, future_prices, label="Future Prices", color="red", linestyle="dashed")
        ax_zoom.set_title(f"{ticker} Future Price Forecast (Zoomed In: Last 3 Months)")
        ax_zoom.set_xlabel("Date")
        ax_zoom.set_ylabel("Price (USD)")
        ax_zoom.legend()
        ax_zoom.grid()
        ax_zoom.tick_params(axis='x', rotation=45)
        
        self.canvas.draw()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

