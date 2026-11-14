import os
import shutil
import pandas as pd

# 1. Đường dẫn thư mục
csv_path = 'data/metadata/filtered_inventory.csv'
final_ai_dir = 'data/processed/final/AI'
final_human_dir = 'data/processed/final/human'

print("Đang đọc danh sách ảnh, vui lòng đợi...")
df = pd.read_csv(csv_path)

count_ai = 0
count_human = 0

print("Bắt đầu phân loại và copy ảnh (Quá trình này có thể mất 5-10 phút)...")

# 2. Quét và copy ảnh
for index, row in df.iterrows():
    img_path = str(row['file_path'])
    label = str(row['label']).strip().lower()

    if not os.path.exists(img_path):
        continue

    # Phân loại dựa trên nhãn (AI/FAKE vào thư mục AI, REAL/HUMAN vào thư mục human)
    if 'ai' in label or 'fake' in label:
        dest_folder = final_ai_dir
        count_ai += 1
    elif 'human' in label or 'real' in label or 'người' in label:
        dest_folder = final_human_dir
        count_human += 1
    else:
        continue

    # Copy ảnh
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(dest_folder, img_name))

print(f"Xong! Đã chuyển thành công: {count_ai} ảnh AI và {count_human} ảnh Human.")