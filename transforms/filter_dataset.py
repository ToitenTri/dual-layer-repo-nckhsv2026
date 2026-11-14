
import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data/metadata/all_datasets_inventory.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data/metadata/filtered_inventory.csv"
)

with open(INPUT_FILE, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

valid_images = []

seen_hashes = set()

error_count = 0
duplicate_count = 0

for row in rows:

    if row["status"] != "OK":
        error_count += 1
        continue

    image_hash = row["sha256"]

    if image_hash in seen_hashes:
        duplicate_count += 1
        continue

    seen_hashes.add(image_hash)
    valid_images.append(row)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = rows[0].keys() if rows else []
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(valid_images)
print("DA LOC XONG DATASET!")
print("Tong anh ban dau:", len(rows))
print("Anh loi:", error_count)
print("Anh trung lap:", duplicate_count)
print("Anh con lai:", len(valid_images))
print("File ket qua:", OUTPUT_FILE)

import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR / "data/metadata/all_datasets_inventory.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR / "data/metadata/filtered_inventory.csv"
)

with open(INPUT_FILE, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

valid_images = []

seen_hashes = set()

error_count = 0
duplicate_count = 0

for row in rows:

    if row["status"] != "OK":
        error_count += 1
        continue

    image_hash = row["sha256"]

    if image_hash in seen_hashes:
        duplicate_count += 1
        continue

    seen_hashes.add(image_hash)
    valid_images.append(row)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = rows[0].keys() if rows else []
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(valid_images)
print("DA LOC XONG DATASET!")
print("Tong anh ban dau:", len(rows))
print("Anh loi:", error_count)
print("Anh trung lap:", duplicate_count)
print("Anh con lai:", len(valid_images))
print("File ket qua:", OUTPUT_FILE)
