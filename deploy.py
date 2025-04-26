from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Define models
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

# Initialize Flask app
app = Flask(__name__)

# Globals
model = None
scaler = None
features_list = []

def train_model(data):
    global model, scaler, features_list
    
    y = data['Total Power']
    X = data.drop('Total Power', axis=1)
    features_list = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    model = MLP(X.shape[1], 64, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor.squeeze())
        loss.backward()
        optimizer.step()

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = []
    predictions_text = ""

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            # File upload
            file = request.files['file']
            data = pd.read_csv(file)
            if 'Total Power' in data.columns:
                train_model(data)
            else:
                if model is None:
                    return "Please upload a training dataset with 'Total Power' column first.", 400
                X_input = scaler.transform(data)
                X_tensor = torch.tensor(X_input, dtype=torch.float32)
                y_pred = model(X_tensor).detach().numpy()
                predictions = y_pred.flatten().tolist()
                predictions_text = "\n".join([f"{p:.2f}" for p in predictions])

        elif request.form.getlist('manual_inputs'):
            # Manual inputs
            input_values = request.form.getlist('manual_inputs')
            input_values = [float(val) for val in input_values if val.strip() != '']
            if model is None:
                return "Please upload and train the model first.", 400
            if len(input_values) % len(features_list) != 0:
                return f"Each prediction needs {len(features_list)} feature values. Currently got {len(input_values)}.", 400
            X_manual = np.array(input_values).reshape(-1, len(features_list))
            X_manual_scaled = scaler.transform(X_manual)
            X_tensor = torch.tensor(X_manual_scaled, dtype=torch.float32)
            y_pred = model(X_tensor).detach().numpy()
            predictions = y_pred.flatten().tolist()
            predictions_text = "\n".join([f"{p:.2f}" for p in predictions])

    return render_template('home.html', features=features_list, predictions_text=predictions_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
