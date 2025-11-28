import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib

# ==========================
# PATH DATASET
# ==========================
csv_path = r"E:\KLASIFIKASI_JAMUR\fitur_konsisten.csv"
model_path = "KNN_best_model.pkl"
scaler_path = "KNN_scaler.pkl"
label_path = "KNN_label_encoder.pkl"

# ==========================
# LOAD DATASET
# ==========================
print("=" * 80)
print("TRAINING MODEL K-NN JAMUR")
print("=" * 80)

df = pd.read_csv(csv_path)

print(f"\nTotal Data : {len(df)}")
print("Kolom:", list(df.columns))

print("\nDistribusi Label:")
print(df['label'].value_counts())

# ==========================
# PREPARE FEATURES & LABEL
# ==========================
X = df.drop(['filename', 'label'], axis=1)
y = df['label']

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# ==========================
# STANDARDIZATION
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================
# GRID SEARCH
# ==========================
print("\nMencari hyperparameter terbaik...")
print("=" * 80)

param_grid = {
    'n_neighbors': range(1, 21),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

# ==========================
# BEST MODEL
# ==========================
best_params = grid.best_params_
best_cv = grid.best_score_ * 100
test_acc = grid.score(X_test_scaled, y_test) * 100

print("\nHASIL TERBAIK")
print("=" * 80)
for k, v in best_params.items():
    print(f"{k}: {v}")

print(f"\nCV Accuracy  : {best_cv:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%")

# ==========================
# EVALUATION
# ==========================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("\nLAPORAN KLASIFIKASI")
print("=" * 80)
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(np.arange(len(le.classes_)), le.classes_)
plt.yticks(np.arange(len(le.classes_)), le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

# ==========================
# VISUALISASI AKURASI vs K
# ==========================
results = pd.DataFrame(grid.cv_results_)
results = results.sort_values('rank_test_score')

plt.figure()
for metric in ['euclidean', 'manhattan']:
    for weight in ['uniform', 'distance']:
        subset = results[
            (results['param_metric'] == metric) &
            (results['param_weights'] == weight)
        ].sort_values('param_n_neighbors')

        plt.plot(
            subset['param_n_neighbors'],
            subset['mean_test_score'] * 100,
            marker='o',
            label=f"{metric}-{weight}"
        )

plt.title("Akurasi CV vs Nilai K")
plt.xlabel("Nilai K")
plt.ylabel("Akurasi (%)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# SIMPAN MODEL
# ==========================
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le, label_path)

print("\nMODEL TERSIMPAN")
print("=" * 80)
print(f"Model  : {model_path}")
print(f"Scaler : {scaler_path}")
print(f"Label  : {label_path}")

# ==========================
# FEATURE IMPORTANCE (PROXY)
# ==========================
feature_std = X_train_scaled.std(axis=0)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'std': feature_std
}).sort_values('std', ascending=False)

print("\nTOP 10 FITUR PALING BERPENGARUH")
print("=" * 80)
print(feature_importance.head(10).to_string(index=False))

print("\nTRAINING SELESAI")
