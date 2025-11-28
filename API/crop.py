import cv2
import matplotlib.pyplot as plt

# Path gambar
image_path = r"E:\klasifikasi_jamur\API\uploads\capture.jpg"

# Baca gambar
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ================================
# KOORDINAT CROP (UBAH SESUAI KEBUTUHAN)
# ================================
x1, y1 = 80, 60
x2, y2 = 280, 200

# ================================
# PROSES CROP
# ================================
crop = img_rgb[y1:y2, x1:x2]

# ================================
# TAMPILKAN PERBANDINGAN
# ================================
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Gambar Asli")
plt.imshow(img_rgb)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Hasil Crop")
plt.imshow(crop)
plt.axis("off")

plt.show()
