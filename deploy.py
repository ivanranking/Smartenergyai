from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

from models import MLP  # We'll create a simple MLP model like you had before

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load a default scaler and model (or retrain after CSV upload)
scaler = StandardScaler()
model = None
features_list = ['Hour', 'Minute', 'DayOfWeek']  # Extend with your dataset features

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/manual_input', methods=['GET', 'POST'])
def manual_input():
    if request.method == 'POST':
        data = [float(request.form.get(feature)) for feature in features_list]
        input_data = scaler.transform([data])
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor).item()

        return render_template('prediction.html', prediction=prediction)
    return render_template('manual_input.html', features=features_list)

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    global model, scaler

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            df = pd.read_csv(filepath)

            # Preprocessing
            y = df['Total Power']
            X = df.drop('Total Power', axis=1)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

            # Define and train a new MLP model
            model = MLP(X.shape[1], 64, 1)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            for epoch in range(20):
                model.train()
                optimizer.zero_grad()
                output = model(X_tensor)
                loss = criterion(output, y_tensor)
                loss.backward()
                optimizer.step()

            torch.save(model.state_dict(), 'models/trained_model.pth')

            return redirect(url_for('manual_input'))
    return render_template('upload_csv.html')

@app.route('/predict')
def predict():
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
