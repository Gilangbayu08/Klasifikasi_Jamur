# === API/predictor.py ===
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

model = joblib.load(os.path.join(PROJECT_DIR, "KNN_best_model.pkl"))
scaler = joblib.load(os.path.join(PROJECT_DIR, "KNN_scaler.pkl"))
encoder = joblib.load(os.path.join(PROJECT_DIR, "KNN_label_encoder.pkl"))


def predict_from_features_dict(features_dict: dict):
    """
    Menerima dict fitur (nama_fitur -> nilai),
    menyusun DataFrame sesuai urutan fitur saat training
    (pakai scaler.feature_names_in_), lalu prediksi.
    """
    # 1. Susun DataFrame dari dict
    X = pd.DataFrame([features_dict])

    # 2. Pastikan urutan kolom sama dengan training
    X = X[scaler.feature_names_in_]

    # 3. Scaling
    X_scaled = scaler.transform(X)

    # 4. Prediksi
    pred_encoded = model.predict(X_scaled)[0]
    probs = model.predict_proba(X_scaled)[0]

    # 5. Decode label
    label = encoder.inverse_transform([pred_encoded])[0]

    # 6. Susun dict probabilitas
    prob_dict = {
        cls: float(probs[i])
        for i, cls in enumerate(encoder.classes_)
    }

    return label, prob_dict
