# === API/feature_pipeline.py ===
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ---------------------------------------------------------
# 1. Preprocessing mirip batch_preprocessing.preprocess_mushroom_image
#    (dipakai untuk data baru: kamera / gambar mentah)
# ---------------------------------------------------------
def preprocess_mushroom_image_bgr(
    img_bgr,
    target_size=(256, 256),
    alpha=1.2,
    beta=20,
    lower_hsv=np.array([4, 34, 35]),
    upper_hsv=np.array([25, 255, 255]),
    min_area=500
):
    """
    Preprocessing jamur merang di background putih → segmentasi HSV,
    brightness/contrast, masking, hasil akhir jamur di atas background hitam.
    Input  : img_bgr (array BGR dari cv2 / webcam)
    Output : reconstructed_bgr, mask_filtered
    """
    if img_bgr is None:
        raise ValueError("img_bgr = None")

    # 1. Resize
    img_resized = cv2.resize(img_bgr, target_size)

    # 2. Brightness / Contrast
    img_bright = cv2.convertScaleAbs(img_resized, alpha=alpha, beta=beta)

    # 3. HSV
    img_hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)

    # 4. Mask awal (range jamur)
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # 5. Morphology Closing & Opening
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 6. Connected Components Filtering (buang noise kecil)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
    mask_filtered = np.zeros_like(mask_opened)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_filtered[labels == i] = 255

    # 7. Rekonstruksi gambar (jamur saja, background hitam)
    reconstructed_bgr = cv2.bitwise_and(img_bright, img_bright, mask=mask_filtered)

    return reconstructed_bgr, mask_filtered


# ---------------------------------------------------------
# 2. Ekstraksi fitur 35D (HARUS sama definisinya dengan Ekstraksi_Fitur.py)
# ---------------------------------------------------------
def extract_features_from_bgr(img_bgr):
    """
    Ekstraksi fitur 35 dimensi dari citra BGR jamur (sudah background hitam).
    Definisi fitur disamakan dengan Ekstraksi_Fitur.py (training).
    Return: features_dict (dict: nama_fitur -> nilai float)
    """
    if img_bgr is None:
        return None

    # Resize ke 256x256 (sama dengan training)
    img = cv2.resize(img_bgr, (256, 256))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, w = gray.shape

    # ===== MASK dari grayscale (gray > 10) + morphologi (training version) =====
    mask = np.where(gray > 10, 255, 0).astype("uint8")

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # ===== SHAPE FEATURES =====
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

    # ===== HSV FEATURES (hanya piksel di area mask) =====
    hsv_pixels = hsv[mask > 0]

    if len(hsv_pixels) > 0:
        mean_h = float(np.mean(hsv_pixels[:, 0]))
        std_h = float(np.std(hsv_pixels[:, 0]))
        mean_s = float(np.mean(hsv_pixels[:, 1]))
        std_s = float(np.std(hsv_pixels[:, 1]))
        mean_v = float(np.mean(hsv_pixels[:, 2]))
        std_v = float(np.std(hsv_pixels[:, 2]))
    else:
        mean_h = std_h = mean_s = std_s = mean_v = std_v = 0.0

    dark_pct = float(np.sum(hsv[:, :, 2] < 50) / (h * w))

    # ===== GLCM FEATURES =====
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    if np.sum(mask) > 200:
        glcm = graycomatrix(
            gray_masked,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
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

    # ===== LBP FEATURES =====
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
        lbp_mean = lbp_std = lbp_entropy = lbp_energy = 0.0
        hist_vals = [0.0] * 6

    # ===== KOMPILASI DICT FITUR (SAMA DENGAN CSV TRAINING) =====
    features_dict = {
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

    return features_dict


# ---------------------------------------------------------
# 3. Wrapper: ekstraksi fitur langsung dari path gambar
# ---------------------------------------------------------
def extract_features_from_path(image_path):
    """
    Digunakan oleh API dan Prediksi offline:
    - Baca gambar dari path
    - Preprocess (segmen background putih → jamur)
    - Ekstraksi fitur 35D
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # Untuk gambar raw (dari webcam / dataset mentah), pakai preprocessing penuh
    preprocessed_bgr, _ = preprocess_mushroom_image_bgr(img_bgr)
    features_dict = extract_features_from_bgr(preprocessed_bgr)

    return features_dict
