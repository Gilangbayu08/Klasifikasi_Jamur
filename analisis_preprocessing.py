import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_mushroom_preprocessing(image_path="B_042.jpg", grade_name="Grade B"):
 print("\n" + "="*70)
 print(f"ANALISIS PRA-PROCESSING JAMUR {grade_name}")
 print("="*70)
 print("\n[STEP 1] Membaca gambar...")

 img_bgr = cv2.imread(image_path)
 print(f"✓ Gambar berhasil dibaca")
 print(f"  Ukuran: {img_bgr.shape[1]} x {img_bgr.shape[0]} pixels")

 img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
 print("✓ Konversi BGR → RGB selesai")

 img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
 print("✓ Konversi ke Grayscale selesai")

 img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
 print("✓ Konversi RGB → HSV selesai")
 
 print("\n[STEP 5] Segmentasi warna jamur...")
# Threshold HSV untuk jamur merang
# H (Hue): 0-180 (semua warna)
# S (Saturation): 0-100 (saturasi rendah = warna pucat)
# V (Value): 50-255 (brightness cukup terang)

 lower_hsv = np.array([0, 0, 50])      # Batas bawah
 upper_hsv = np.array([180, 100, 255]) # Batas atas
# Buat mask (binary image: 0=background, 255=jamur)
 mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

 print("\n[STEP 6] Perbaiki mask dengan operasi morfologi...")
 kernel = np.ones((5, 5), np.uint8)

# MORPH_CLOSE: Tutup lubang kecil dalam objek
 mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# MORPH_OPEN: Hapus noise kecil di luar objek
 mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

# STEP 7: Terapkan mask pada gambar asli
 result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_clean)
 print("✓ Segmentasi selesai, background dihilangkan")

# Grayscale hasil segmentasi (untuk analisis)
 gray_segmented = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

# Hitung statistik intensitas
 print("\n[STEP 8] Menghitung statistik warna...")
 gray_masked = img_gray[mask_clean > 0] # ambil hanya area jamur

 if len(gray_masked) > 0:
        mean_intensity = np.mean(gray_masked)
        std_intensity = np.std(gray_masked)
        min_intensity = np.min(gray_masked)
        max_intensity = np.max(gray_masked)
        
        print("✓ Statistik Intensitas Grayscale (Area Jamur):")
        print(f"  Mean (Rata-rata): {mean_intensity:.2f}")
        print(f"  Std (Deviasi)   : {std_intensity:.2f}")
        print(f"  Min             : {min_intensity}")
        print(f"  Max             : {max_intensity}")
 else:
        print("⚠ Tidak ada area jamur terdeteksi!")

 plt.figure(figsize=(16, 10))
    
# Baris 1: Tahap konversi warna
 plt.subplot(2, 4, 1)
 plt.imshow(img_rgb)
 plt.title("1. Gambar Asli (RGB)", fontsize=11, fontweight='bold')
 plt.axis("off")
    
 plt.subplot(2, 4, 2)
 plt.imshow(img_gray, cmap='gray')
 plt.title("2. Grayscale", fontsize=11, fontweight='bold')
 plt.axis("off")
    
 plt.subplot(2, 4, 3)
 plt.imshow(img_hsv)
 plt.title("3. HSV (untuk segmentasi)", fontsize=11, fontweight='bold')
 plt.axis("off")
    
 plt.subplot(2, 4, 4)
 plt.imshow(mask, cmap='gray')
 plt.title("4. Mask Awal", fontsize=11, fontweight='bold')
 plt.axis("off")

# Baris 2: Hasil segmentasi
 plt.subplot(2, 4, 5)
 plt.imshow(mask_clean, cmap='gray')
 plt.title("5. Mask Setelah Morfologi", fontsize=11, fontweight='bold')
 plt.axis("off")
    
 plt.subplot(2, 4, 6)
 plt.imshow(result)
 plt.title("6. Hasil Segmentasi", fontsize=11, fontweight='bold')
 plt.axis("off")
    
 plt.subplot(2, 4, 7)
 plt.imshow(gray_segmented, cmap='gray')
 plt.title("7. Gray Segmented", fontsize=11, fontweight='bold')
 plt.axis("off")
    
 plt.subplot(2, 4, 8)
 if len(gray_masked) > 0:
        plt.hist(gray_masked, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.axvline(mean_intensity, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_intensity:.1f}')
        plt.title("8. Histogram Intensitas", fontsize=11, fontweight='bold')
        plt.xlabel("Intensitas", fontsize=10)
        plt.ylabel("Frekuensi", fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
 else:
        plt.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
        plt.title("8. Histogram Intensitas", fontsize=11, fontweight='bold')
        plt.axis("off")
    
 plt.suptitle(f"Analisis Preprocessing - {grade_name}", fontsize=14, fontweight='bold', y=0.98)
 plt.tight_layout()

 plt.show()

if __name__ == "__main__":
    analyze_mushroom_preprocessing()