import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("Loading dataset...")
df = pd.read_csv("dataset/pak_flood_data.csv")

# Features to use for training
features = [
    'elevation', 
    'river_discharge', 
    'river_discharge_mean', 
    'river_discharge_median', 
    'river_discharge_max', 
    'river_discharge_min'
]

X = df[features]
y = df['flood_impact'] # 0: Safe, 1: Moderate, 2: Severe

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
print("Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "feature_names.pkl")

print("Successfully saved model.pkl, scaler.pkl, and feature_names.pkl!")
