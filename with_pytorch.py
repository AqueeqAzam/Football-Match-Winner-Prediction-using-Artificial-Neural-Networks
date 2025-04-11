# PyTorch version of your TensorFlow code
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------- Logging Setup --------------------
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)

# -------------------- Load Data --------------------
def load_data(url: str):
    try:
        data = pd.read_csv(url)
        logging.info('✅ Data loaded successfully!')
        return data
    except Exception as e:
        logging.error("❌ Error loading data", exc_info=True)
        return None

# -------------------- Scale Features --------------------
def process_data(df):
    df.dropna(inplace=True)
    x = df[['team_a_avg_goals', 'team_b_avg_goals']]
    y = df['team_a_win']

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    logging.info(f"📊 Scaled X shape: {X_scaled.shape}")
    logging.info(f"🎯 Scaled y shape: {y_scaled.shape}")
    return X_scaled, y_scaled, x_scaler, y_scaler

# -------------------- Define Model --------------------
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# -------------------- Train Model --------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# -------------------- Predict --------------------
def make_predictions(model, X_val, y_scaler):
    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(X_val_tensor).numpy()
    pred_unscaled = y_scaler.inverse_transform(pred_scaled)
    return pred_unscaled

# -------------------- Custom Prediction --------------------
def predict_custom(model, input_values, x_scaler, y_scaler):
    input_scaled = x_scaler.transform([input_values])
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(input_tensor).numpy()
    pred_unscaled = y_scaler.inverse_transform(pred_scaled)[0][0]
    logging.info(f"🔮 Prediction: Team A Win Probability = {pred_unscaled:.2f}")
    print(f"\n🔮 Custom Prediction:\nInput: {input_values} ➡️ Predicted Team A Win: {pred_unscaled:.2f}")
    return pred_unscaled

# -------------------- Evaluate --------------------
def evaluate_model(y_true_scaled, y_pred_scaled, y_scaler):
    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 RMSE: {rmse:.4f}")
    print(f"📈 R² Score: {r2:.4f}")

# -------------------- Plot --------------------
def plot_predictions(y_val_scaled, y_pred_unscaled, y_scaler):
    y_true = y_scaler.inverse_transform(y_val_scaled.reshape(-1, 1))
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred_unscaled, label='Predicted')
    plt.title('📈 Actual vs Predicted - Team A Win Probability')
    plt.xlabel('Samples')
    plt.ylabel('Win Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------- Main --------------------
def main():
    url = "https://raw.githubusercontent.com/AqueeqAzam/data-science-and-machine-learning-datasets/refs/heads/main/classification_dl.csv"
    df = load_data(url)

    if df is not None:
        X_scaled, y_scaled, x_scaler, y_scaler = process_data(df)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

        model = NeuralNet(X_train.shape[1])
        train_model(model, X_train, y_train, X_val, y_val)

        predictions = make_predictions(model, X_val, y_scaler)
        evaluate_model(y_val, predictions, y_scaler)
        plot_predictions(y_val, predictions, y_scaler)

        custom_input = [2.1, 1.3]
        predict_custom(model, custom_input, x_scaler, y_scaler)

if __name__ == "__main__":
    main()
