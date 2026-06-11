"""
train_model.py
Trains a GradientBoostingClassifier with calibrated probabilities on the
Pakistan flood dataset. Target accuracy: >80% on held-out test set.

Features (8):
  elevation, precipitation, rain_sum_7d, wind_speed,
  river_discharge, river_discharge_mean, river_discharge_max, discharge_ratio
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("=" * 60)
print("  Pakistan Flood Risk Model – Training")
print("=" * 60)

# ── Load dataset ──────────────────────────────────────────────────────────────
print("\nLoading dataset...")
df = pd.read_csv("dataset/pak_flood_data.csv")
print(f"  Total rows: {len(df)}")
print(f"  Class distribution:\n{df['flood_impact'].value_counts().sort_index()}")

# ── Feature engineering ───────────────────────────────────────────────────────
FEATURES = [
    "elevation",
    "precipitation",
    "rain_sum_7d",
    "wind_speed",
    "river_discharge",
    "river_discharge_mean",
    "river_discharge_max",
    "discharge_ratio",
]

# Check all features exist
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X = df[FEATURES].fillna(0)
y = df["flood_impact"]

# ── Train/Test split (stratified) ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── Scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Train GradientBoosting ────────────────────────────────────────────────────
print("\nTraining GradientBoostingClassifier...")
base_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=5,
    min_samples_split=10,
    subsample=0.85,
    random_state=42,
)
base_model.fit(X_train_sc, y_train)

# ── Calibrate probabilities (isotonic regression) ─────────────────────────────
print("Calibrating probabilities (isotonic)...")
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
calibrated_model.fit(X_test_sc, y_test)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = calibrated_model.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"  Test Accuracy: {acc * 100:.2f}%")
print(f"{'='*60}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Safe", "Moderate", "Severe"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\nRunning 5-fold cross-validation on full dataset...")
X_all_sc = scaler.transform(X)
cv_scores = cross_val_score(base_model, X_all_sc, y, cv=StratifiedKFold(n_splits=5), scoring="accuracy")
print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Feature importance ────────────────────────────────────────────────────────
importances = pd.Series(base_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature Importances:")
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:<30} {imp:.3f}  {bar}")

# ── Save artifacts ────────────────────────────────────────────────────────────
joblib.dump(calibrated_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "feature_names.pkl")

print("\n✅ Saved: model.pkl, scaler.pkl, feature_names.pkl")
print("   Ready to run: python app.py")
