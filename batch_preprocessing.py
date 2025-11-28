import cv2 
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# Fungsi Preprocessing untuk 1 gambar
def preprocess_mushroom_image(
    image_path,
    target_size=(256, 256),
    alpha=1.2,
    beta=20,
    lower_hsv=np.array([4, 34, 35]),
    upper_hsv=np.array([25, 255, 255]),
    min_area=500
):

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"⚠ Gagal membaca: {image_path}")
        return None, None

    # 1. Resize
    img_resized = cv2.resize(img_bgr, target_size)

    # 2. Brightness / Contrast
    img_bright = cv2.convertScaleAbs(img_resized, alpha=alpha, beta=beta)

    # 3. HSV
    img_hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)

    # 4. Mask awal
    mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

    # 5. Morphology Closing
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. Morphology Opening
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    # 7. Connected Components Filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_opened, connectivity=8)
    mask_filtered = np.zeros_like(mask_opened)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask_filtered[labels == i] = 255

    # 8. Rekonstruksi Gambar (Apply Mask)
    img_rgb = cv2.cvtColor(img_bright, cv2.COLOR_BGR2RGB)
    reconstructed = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_filtered)

    return reconstructed, mask_filtered


# Fungsi Batch Preprocessing untuk 1 folder
def batch_preprocess_folder(input_folder, output_folder):
    output_mask_folder = os.path.join(output_folder, "MASK")
    output_recon_folder = os.path.join(output_folder, "RECON")

    os.makedirs(output_mask_folder, exist_ok=True)
    os.makedirs(output_recon_folder, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    files = [f for f in os.listdir(input_folder) if f.endswith(valid_extensions)]

    if len(files) == 0:
        print(f"⚠ Tidak ada gambar ditemukan di: {input_folder}")
        return 0, 0

    print(f"\n{'=' * 70}")
    print(f"Memproses folder: {input_folder}")
    print(f"Jumlah gambar: {len(files)}")
    print(f"Output folder: {output_folder}")
    print(f"{'=' * 70}")

    success_count = 0
    failed_count = 0
    failed_files = []

    for filename in tqdm(files, desc="Processing", ncols=70):
        input_path = os.path.join(input_folder, filename)

        reconstructed, mask = preprocess_mushroom_image(input_path)

        if reconstructed is not None:
            # simpan mask
            mask_path = os.path.join(output_mask_folder, filename)
            cv2.imwrite(mask_path, mask)

            # simpan reconstructed
            recon_path = os.path.join(output_recon_folder, filename)
            cv2.imwrite(recon_path, cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR))

            success_count += 1
        else:
            failed_count += 1
            failed_files.append(filename)

    print(f"\n✓ Preprocessing selesai!")
    print(f"  Berhasil: {success_count} gambar")
    print(f"  Gagal   : {failed_count} gambar")

    return success_count, failed_count



# Fungsi Preprocessing untuk Semua Grade
def preprocess_all_grades(base_folder="Data_Train"):
    print("\n" + "=" * 70)
    print(" " * 15 + "BATCH PREPROCESSING - SEMUA GRADE")
    print("=" * 70)

    grades = {
        "Grade_A": {
            "input": os.path.join(base_folder, "Grade_A"),
            "output": os.path.join(base_folder, "Preprocessed_Grade_A")
        },
        "Grade_B": {
            "input": os.path.join(base_folder, "Grade_B"),
            "output": os.path.join(base_folder, "Preprocessed_Grade_B")
        },
        "Grade_C": {
            "input": os.path.join(base_folder, "Grade_C"),
            "output": os.path.join(base_folder, "Preprocessed_Grade_C")
        }
    }

    total_processed = 0
    total_failed = 0

    for grade_name, folders in grades.items():
        input_folder = folders["input"]
        output_folder = folders["output"]

        if not os.path.exists(input_folder):
            print(f"\n⚠ SKIP: Folder {input_folder} tidak ditemukan!")
            continue

        success, fail = batch_preprocess_folder(input_folder, output_folder)
        total_processed += success
        total_failed += fail

    print("\n" + "=" * 70)
    print("RINGKASAN PREPROCESSING")
    print("=" * 70)
    print(f"Total gambar berhasil: {total_processed}")
    print(f"Total gambar gagal   : {total_failed}")
    print("=" * 70)

    # show_sample_results(base_folder)
    print("\n✓ SEMUA PREPROCESSING SELESAI!")

    return total_processed, total_failed



# Fungsi Visualisasi
def show_sample_results(base_folder="Data_Train"):
    print("\n" + "=" * 70)
    print("MENAMPILKAN CONTOH HASIL PREPROCESSING")
    print("=" * 70)

    grades = ["Grade_A", "Grade_B", "Grade_C"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle("Contoh Hasil Preprocessing per Grade", fontsize=16, fontweight='bold')

    for idx, grade in enumerate(grades):
        input_folder = os.path.join(base_folder, grade)
        output_folder = os.path.join(base_folder, f"Preprocessed_{grade}")

        if not os.path.exists(input_folder) or not os.path.exists(output_folder):
            continue

        input_files = [f for f in os.listdir(input_folder)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(input_files) == 0:
            continue

        sample_file = input_files[0]
        input_path = os.path.join(input_folder, sample_file)
        output_path = os.path.join(output_folder, sample_file)

        img_original = cv2.imread(input_path)
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_preprocessed = cv2.imread(output_path)
        mask = preprocess_mushroom_image(input_path)[1]

        axes[idx, 0].imshow(img_original_rgb)
        axes[idx, 1].imshow(mask, cmap='gray')
        axes[idx, 2].imshow(img_preprocessed)
        axes[idx, 3].hist(mask.flatten(), bins=50)

        for j in range(4):
            axes[idx, j].axis('off')

    plt.tight_layout()
    plt.show()



# MAIN PROGRAM
if __name__ == "__main__":
    base_folder = "Data_Train"
    total_processed, total_failed = preprocess_all_grades(base_folder)
