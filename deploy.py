from flask import Flask, render_template, request
import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename

from models import MLP  # Your MLP model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

scaler = StandardScaler()
model = None
features_list = ['Hour', 'Minute', 'DayOfWeek']  # update as per your features

@app.route('/', methods=['GET', 'POST'])
def home():
    global model, scaler
    prediction = None

    if request.method == 'POST':
        form_type = request.form.get('form_type')

        if form_type == 'manual':
            data = [float(request.form.get(feature)) for feature in features_list]
            input_data = scaler.transform([data])
            input_tensor = torch.tensor(input_data, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                prediction = model(input_tensor).item()

        elif form_type == 'csv':
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

                # Define and train new model
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

    return render_template('home.html', features=features_list, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
