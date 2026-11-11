
import cv2
import csv
import hashlib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASETS = {
    "Dataset_A": PROJECT_DIR / "data/raw/dataset_A",
    "Dataset_B": PROJECT_DIR / "data/raw/dataset_B"
}

OUTPUT_DIR = PROJECT_DIR / "data/metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "all_datasets_inventory.csv"
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff"
}

def calculate_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

results = []

for dataset_name, dataset_folder in DATASETS.items():

    print("\nDang quet:", dataset_name)

    if not dataset_folder.exists():
        print("Khong tim thay:", dataset_folder)
        continue

    for image_path in dataset_folder.rglob("*"):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label = image_path.parent.name.upper()

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED
        )

        if image is None:
            results.append([
                dataset_name, label, str(image_path),
                "ERROR", "", "", ""
            ])
            continue

        height, width = image.shape[:2]

        if len(image.shape) == 2:
            channels = 1
        else:
            channels = image.shape[2]

        image_hash = calculate_hash(image_path)

        results.append([
            dataset_name, label, str(image_path),
            "OK", width, height, channels, image_hash
        ])



with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "dataset", "label", "file_path", "status",
        "width", "height", "channels", "sha256"
    ])

    writer.writerows(results)

print("\nDA QUET XONG!")

print("Tong so anh:", len(results))

print(
    "Anh doc duoc:",
    sum(row[3] == "OK" for row in results)
)

print(
    "Anh loi:",
    sum(row[3] == "ERROR" for row in results)
)

print("File ket qua:", OUTPUT_FILE)
