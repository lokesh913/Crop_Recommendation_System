from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

# Train model on startup
def train_model():
    # Load the dataset
    data = pd.read_csv("Crop_recommendation.csv")
    
    # Separate features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Create label mapping
    labels = y.unique()
    label_dict = {label: i+1 for i, label in enumerate(labels)}
    reverse_label_dict = {i+1: label for i, label in enumerate(labels)}
    y = y.map(label_dict)
    
    # Scale the features
    minmax = MinMaxScaler()
    X_minmax = minmax.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_minmax)
    
    # Train the model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_scaled, y)
    
    return rf_model, scaler, minmax, reverse_label_dict

# Initialize model and scalers
model, sc, mx, crop_dict = train_model()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    mx_features = mx.transform(single_pred)
    sc_mx_features = sc.transform(mx_features)
    prediction = model.predict(sc_mx_features)

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)