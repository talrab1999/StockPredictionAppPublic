import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_model(model, train_loader, num_epochs=20, lr=0.001, device="cpu"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    return model

def predict_future_LSTM_or_GRU(model, data_scaled, seq_length, scaler, future_days=30, device="cpu"):
    future_input = data_scaled[-seq_length:].reshape(1, seq_length, 1)
    future_input = torch.tensor(future_input, dtype=torch.float32).to(device)
    future_log_returns = []
    model.eval()
    for _ in range(future_days):
        with torch.no_grad():
            pred = model(future_input)
            future_log_returns.append(pred.cpu().numpy())
            future_input = torch.cat((future_input[:, 1:, :], pred.view(1, 1, 1)), dim=1)
    future_log_returns = scaler.inverse_transform(
        np.array(future_log_returns).reshape(-1, 1)
    )
    return future_log_returns

def predict_future_linear_or_xgboost_or_rf(model, data_scaled, seq_length, future_days=30):
    future_input = data_scaled[-seq_length:].reshape(1, seq_length, 1)
    future_predictions = []
    for _ in range(future_days):
        X_flat = future_input.reshape(1, seq_length)
        pred = model.predict(X_flat)
        if pred.ndim == 1:
            next_val = pred[0]
        else:
            next_val = pred[0, 0]
        future_predictions.append(next_val)
        future_input = np.concatenate(
            (future_input[:, 1:, :], np.array(pred).reshape(1, 1, 1)),
            axis=1
        )
    return np.array(future_predictions).reshape(-1, 1)
