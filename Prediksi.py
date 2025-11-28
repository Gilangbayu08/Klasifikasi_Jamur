# === Prediksi.py ===
import os

from API.feature_pipeline import extract_features_from_path
from API.predictor import predict_from_features_dict

if __name__ == "__main__":
    # Contoh path gambar untuk pengujian
    test_image = r"E:\KLASIFIKASI_JAMUR\API\uploads\A_00.jpg"

    if not os.path.exists(test_image):
        print(f"File tidak ditemukan: {test_image}")
        exit(1)

    print("\n==============================")
    print("PREDIKSI JAMUR (OFFLINE)")
    print("==============================")

    try:
        features_dict = extract_features_from_path(test_image)
        if features_dict is None:
            print("Objek jamur tidak terdeteksi / fitur kosong.")
            exit(1)

        label, prob = predict_from_features_dict(features_dict)

        print(f"File       : {os.path.basename(test_image)}")
        print(f"Prediksi   : {label}")
        print("Probabilitas:")
        for cls, p in prob.items():
            print(f"  {cls}: {p:.4f}")

    except Exception as e:
        print("\nERROR:", str(e))
