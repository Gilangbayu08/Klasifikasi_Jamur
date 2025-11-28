import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm
from scipy.stats import entropy


# FUNGSI EKSTRAKSI FITUR TERBAIK & KONSISTEN

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = gray.shape

    # ====================== MASK ======================
    mask = np.where(gray > 10, 255, 0).astype("uint8")

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ====================== SHAPE FEATURES ======================
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    x, y, w_box, h_box = cv2.boundingRect(cnt)
    aspect_ratio = w_box / h_box if h_box != 0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    rect_area = w_box * h_box
    extent = area / rect_area if rect_area > 0 else 0

    # ====================== HSV FEATURES ======================
    hsv_pixels = hsv[mask > 0]

    if len(hsv_pixels) > 0:
        mean_h = float(np.mean(hsv_pixels[:, 0]))
        std_h = float(np.std(hsv_pixels[:, 0]))
        mean_s = float(np.mean(hsv_pixels[:, 1]))
        std_s = float(np.std(hsv_pixels[:, 1]))
        mean_v = float(np.mean(hsv_pixels[:, 2]))
        std_v = float(np.std(hsv_pixels[:, 2]))
    else:
        mean_h = std_h = mean_s = std_s = mean_v = std_v = 0

    dark_pct = float(np.sum(hsv[:, :, 2] < 50) / (h * w))

    # ====================== GLCM FEATURES ======================
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    if np.sum(mask) > 200:
        glcm = graycomatrix(
            gray_masked,
            distances=[1],
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )

        glcm_features = {}
        for i, angle in enumerate(["0", "45", "90", "135"]):
            glcm_features[f"glcm_contrast_{angle}"] = float(graycoprops(glcm, "contrast")[0][i])
            glcm_features[f"glcm_homogeneity_{angle}"] = float(graycoprops(glcm, "homogeneity")[0][i])
            glcm_features[f"glcm_energy_{angle}"] = float(graycoprops(glcm, "energy")[0][i])
    else:
        glcm_features = {k: 0.0 for k in [
            "glcm_contrast_0", "glcm_homogeneity_0", "glcm_energy_0",
            "glcm_contrast_45", "glcm_homogeneity_45", "glcm_energy_45",
            "glcm_contrast_90", "glcm_homogeneity_90", "glcm_energy_90",
            "glcm_contrast_135", "glcm_homogeneity_135", "glcm_energy_135"
        ]}

    # ====================== LBP FEATURES ======================
    if np.sum(mask) > 200:
        lbp = local_binary_pattern(gray_masked, P=8, R=1, method="uniform")
        lbp_masked = lbp[mask > 0]

        hist, _ = np.histogram(lbp_masked, bins=10, range=(0, 10), density=True)

        lbp_mean = float(np.mean(lbp_masked))
        lbp_std = float(np.std(lbp_masked))
        lbp_entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        lbp_energy = float(np.sum(hist ** 2))

        hist_vals = hist[:6].tolist()
    else:
        lbp_mean = lbp_std = lbp_entropy = lbp_energy = 0
        hist_vals = [0] * 6

    # ====================== OUTPUT ======================
    features = {
        "area": area,
        "perimeter": perimeter,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "extent": extent,

        "mean_h": mean_h,
        "std_h": std_h,
        "mean_s": mean_s,
        "std_s": std_s,
        "mean_v": mean_v,
        "std_v": std_v,
        "dark_pct": dark_pct,

        **glcm_features,

        "lbp_mean": lbp_mean,
        "lbp_std": lbp_std,
        "lbp_entropy": lbp_entropy,
        "lbp_energy": lbp_energy,
        "lbp_hist_0": hist_vals[0],
        "lbp_hist_1": hist_vals[1],
        "lbp_hist_2": hist_vals[2],
        "lbp_hist_3": hist_vals[3],
        "lbp_hist_4": hist_vals[4],
        "lbp_hist_5": hist_vals[5],
    }

    return features

# GENERATE CSV PER GRADE
def extract_all(base_folder, output_csv):
    """
    Mengekstrak fitur dari gambar di subfolder RECON 
    untuk setiap Preprocessed_Grade_[A/B/C].
    """
    all_data = []
    grades = ["A", "B", "C"]

    for grade in grades:
        # 1. Tentukan path ke folder Grade yang sudah diproses (Preprocessed_Grade_A/B/C)
        grade_folder_name = f"Preprocessed_Grade_{grade}"
        grade_path = os.path.join(base_folder, grade_folder_name)
        
        # 2. Tentukan path ke subfolder RECON
        recon_folder = os.path.join(grade_path, "RECON")
        
        if not os.path.exists(recon_folder):
            print(f"⚠ Folder RECON tidak ditemukan: {recon_folder}")
            continue

        print(f"\n📂 Processing Grade {grade} di folder RECON")
        
        # Cari file di dalam folder RECON
        filenames = [f for f in os.listdir(recon_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

        for file in tqdm(filenames, desc=f"RECON-{grade}"):
            # Path lengkap ke file di dalam RECON
            path = os.path.join(recon_folder, file)
            
            # Panggil fungsi ekstraksi fitur
            feats = extract_features(path)

            if feats:
                feats["filename"] = file
                feats["label"] = grade
                all_data.append(feats)

    # 3. Buat dan simpan DataFrame
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Ekstraksi selesai! File disimpan di: {output_csv}")
    print("--- 5 baris data teratas ---")
    print(df.head())
    return df

if __name__ == "__main__":
    base = r"E:\KLASIFIKASI_JAMUR\Data_Train"  
    output = r"fitur_konsisten.csv"
    extract_all(base, output)
