# === app.py ===
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_DIR)

from API.feature_pipeline import extract_features_from_path
from API.predictor import predict_from_features_dict

app = FastAPI()

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/index")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Simpan file upload
    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Ekstraksi fitur (pipeline baru, konsisten dengan training)
    try:
        features_dict = extract_features_from_path(save_path)
    except FileNotFoundError:
        return {"status": "error", "message": "Gambar tidak ditemukan / gagal dibaca"}
    except Exception as e:
        return {"status": "error", "message": f"Gagal ekstraksi fitur: {str(e)}"}

    if features_dict is None:
        return {"status": "error", "message": "Objek jamur tidak terdeteksi / fitur kosong"}

    # Prediksi
    label, prob = predict_from_features_dict(features_dict)

    return {
        "status": "success",
        "file": file.filename,
        "prediction": label,
        "probabilities": prob,
        "features": features_dict,
    }
