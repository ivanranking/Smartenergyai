import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOT_FOLDER = 'static/plots'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define deep learning models
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x.float()).squeeze()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x.float())
        return self.fc(x[:, -1, :]).squeeze()

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x, _ = self.rnn(x.float())
        return self.fc(x[:, -1, :]).squeeze()

class CNNModel(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc = nn.Linear(32, output_dim)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.mean(x, dim=2)
        return self.fc(x).squeeze()

# Helper function to process data and train models
def train_models(filepath):
    df = pd.read_csv(filepath)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], errors='coerce')
        df = df.dropna(subset=['TimeStamp'])
        df['Hour'] = df['TimeStamp'].dt.hour
        df['Minute'] = df['TimeStamp'].dt.minute
        df['DayOfWeek'] = df['TimeStamp'].dt.dayofweek
        df = df.drop(columns=['TimeStamp'])
    
    df = pd.get_dummies(df)

    y = df['Total Power']
    X = df.drop('Total Power', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Tensor conversion
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    sequence_length = 1
    num_features = X_train.shape[1]
    X_train_tensor_seq = X_train_tensor.view(X_train_tensor.shape[0], sequence_length, num_features)
    X_test_tensor_seq = X_test_tensor.view(X_test_tensor.shape[0], sequence_length, num_features)
    X_train_tensor_cnn = X_train_tensor.view(X_train_tensor.shape[0], num_features, sequence_length)
    X_test_tensor_cnn = X_test_tensor.view(X_test_tensor.shape[0], num_features, sequence_length)

    models = {
        "MLP": MLP(num_features, 64, 1),
        "LSTM": LSTMModel(num_features, 64, 1),
        "RNN": RNNModel(num_features, 64, 1),
        "CNN": CNNModel(num_features, 1)
    }

    metrics = []

    for name, model in models.items():
        model = model.float()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(10):
            optimizer.zero_grad()
            if name in ["LSTM", "RNN"]:
                outputs = model(X_train_tensor_seq).squeeze()
            elif name == "CNN":
                outputs = model(X_train_tensor_cnn).squeeze()
            else:
                outputs = model(X_train_tensor).squeeze()
            loss = criterion(outputs, y_train_tensor.squeeze())
            loss.backward()
            optimizer.step()

        if name in ["LSTM", "RNN"]:
            y_pred_tensor = model(X_test_tensor_seq).squeeze()
        elif name == "CNN":
            y_pred_tensor = model(X_test_tensor_cnn).squeeze()
        else:
            y_pred_tensor = model(X_test_tensor).squeeze()

        y_pred = y_pred_tensor.detach().numpy().flatten()
        y_true = y_test.values.flatten()

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        msle = mean_squared_log_error(y_true, np.maximum(y_pred, 1e-6))

        metrics.append([name, mae, mse, rmse, r2, mape, msle])

    metrics_df = pd.DataFrame(metrics, columns=["Model", "MAE", "MSE", "RMSE", "R2", "MAPE", "MSLE"])

    # Plotting
    plot_path = os.path.join(PLOT_FOLDER, "plot.png")
    metrics_df.set_index("Model").plot(kind='bar', subplots=True, layout=(3, 2), figsize=(12, 12))
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return metrics_df

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            return redirect(request.url)
        file = request.files['dataset']
        if file.filename == '':
            return redirect(request.url)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        metrics_df = train_models(filepath)
        return render_template('training.html', tables=[metrics_df.to_html(classes='data')], plot_path='static/plots/plot.png')

    return render_template('index.html')

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html'), 500

if __name__ == "__main__":
    app.run(debug=True)
