import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")

# Separate features and target
X = data.drop('label', axis=1)
y = data['label']

# Convert labels to numeric values
label_dict = {label: i+1 for i, label in enumerate(y.unique())}
y = y.map(label_dict)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_minmax)
X_test_scaled = scaler.transform(X_test_minmax)

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Save the model and scalers
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(minmax, f)

with open('standscaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Print model accuracy
train_accuracy = rf_model.score(X_train_scaled, y_train)
test_accuracy = rf_model.score(X_test_scaled, y_test)
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy: {test_accuracy:.4f}") 