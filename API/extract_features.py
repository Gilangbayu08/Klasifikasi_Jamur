import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

FEATURE_ORDER = [
    "area","perimeter","circularity","aspect_ratio","solidity","extent",
    "mean_h","std_h","mean_s","std_s","mean_v","std_v","dark_pct",
    "glcm_contrast_0","glcm_homogeneity_0","glcm_energy_0",
    "glcm_contrast_45","glcm_homogeneity_45","glcm_energy_45",
    "glcm_contrast_90","glcm_homogeneity_90","glcm_energy_90",
    "glcm_contrast_135","glcm_homogeneity_135","glcm_energy_135",
    "lbp_mean","lbp_std","lbp_entropy","lbp_energy",
    "lbp_hist_0","lbp_hist_1","lbp_hist_2",
    "lbp_hist_3","lbp_hist_4","lbp_hist_5"
]

def preprocess_white_background(image_path, x1=80, y1=60, x2=280, y2=200):

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # Crop area
    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError("Hasil crop kosong. Periksa koordinat crop!")

    # Convert ke HSV
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Threshold jamur
    lower = np.array([5, 30, 30])
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background hitam
    result = cv2.bitwise_and(cropped, cropped, mask=mask)

    return image, cropped, mask, result


def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    try:
        _, cropped, pre_mask, pre_result = preprocess_white_background(image_path)
        img = pre_result  # pakai hasil background hitam untuk ekstraksi fitur
    except:
        pass
    # ===================================================

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    h, w = gray.shape

    mask = np.where(gray > 10, 255, 0).astype("uint8")
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    circularity = (4*np.pi*area)/(perimeter**2) if perimeter>0 else 0

    x,y,w_box,h_box = cv2.boundingRect(cnt)
    aspect_ratio = w_box/h_box if h_box!=0 else 0

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area/hull_area if hull_area>0 else 0
    rect_area = w_box*h_box
    extent = area/rect_area if rect_area>0 else 0

    hsv_pixels = hsv[mask>0]
    mean_h,std_h,mean_s,std_s,mean_v,std_v = (0,0,0,0,0,0)

    if len(hsv_pixels)>0:
        mean_h = float(np.mean(hsv_pixels[:,0])); std_h = float(np.std(hsv_pixels[:,0]))
        mean_s = float(np.mean(hsv_pixels[:,1])); std_s = float(np.std(hsv_pixels[:,1]))
        mean_v = float(np.mean(hsv_pixels[:,2])); std_v = float(np.std(hsv_pixels[:,2]))

    dark_pct = float(np.sum(hsv[:,:,2]<50)/(h*w))

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)

    glcm_data = {}
    if np.sum(mask)>200:
        glcm = graycomatrix(gray_masked,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],256,True,True)
        for i,ang in enumerate(["0","45","90","135"]):
            glcm_data[f"glcm_contrast_{ang}"] = float(graycoprops(glcm,"contrast")[0][i])
            glcm_data[f"glcm_homogeneity_{ang}"] = float(graycoprops(glcm,"homogeneity")[0][i])
            glcm_data[f"glcm_energy_{ang}"] = float(graycoprops(glcm,"energy")[0][i])
    else:
        for k in [
            "glcm_contrast_0","glcm_homogeneity_0","glcm_energy_0",
            "glcm_contrast_45","glcm_homogeneity_45","glcm_energy_45",
            "glcm_contrast_90","glcm_homogeneity_90","glcm_energy_90",
            "glcm_contrast_135","glcm_homogeneity_135","glcm_energy_135"
        ]:
            glcm_data[k]=0.0

    if np.sum(mask)>200:
        lbp = local_binary_pattern(gray_masked,8,1,"uniform")
        lbp_mask = lbp[mask>0]
        hist,_ = np.histogram(lbp_mask,bins=10,range=(0,10),density=True)

        lbp_mean = float(np.mean(lbp_mask))
        lbp_std = float(np.std(lbp_mask))
        lbp_entropy = float(-np.sum(hist*np.log2(hist+1e-10)))
        lbp_energy = float(np.sum(hist**2))
        hist = hist[:6].tolist()
    else:
        lbp_mean = lbp_std = lbp_entropy = lbp_energy = 0
        hist = [0]*6

    features_dict = {
        "area":area,"perimeter":perimeter,"circularity":circularity,"aspect_ratio":aspect_ratio,
        "solidity":solidity,"extent":extent,
        "mean_h":mean_h,"std_h":std_h,"mean_s":mean_s,"std_s":std_s,
        "mean_v":mean_v,"std_v":std_v,"dark_pct":dark_pct,
        **glcm_data,
        "lbp_mean":lbp_mean,"lbp_std":lbp_std,
        "lbp_entropy":lbp_entropy,"lbp_energy":lbp_energy,
        "lbp_hist_0":hist[0],"lbp_hist_1":hist[1],
        "lbp_hist_2":hist[2],"lbp_hist_3":hist[3],
        "lbp_hist_4":hist[4],"lbp_hist_5":hist[5],
    }

    feature_list = [features_dict[k] for k in FEATURE_ORDER]
    return feature_list, features_dict
