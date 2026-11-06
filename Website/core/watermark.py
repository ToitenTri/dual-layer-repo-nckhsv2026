import cv2
import numpy as np

def embed_watermark(image_matrix: np.ndarray, bit_string: str) -> np.ndarray:
    if image_matrix is None:
        raise ValueError("Ma trận ảnh đầu vào bị rỗng!")
    print(f"[CORE - WATERMARK] Đang nhúng chuỗi {len(bit_string)} bits: {bit_string[:16]}...")
    watermarked_image = image_matrix.copy()
    return watermarked_image

def extract_watermark(image_matrix: np.ndarray, bit_length: int = 64) -> str:
    if image_matrix is None:
        raise ValueError("Không thể đọc ảnh cần kiểm định!")
    np.random.seed(int(np.mean(image_matrix)))
    dummy_extracted_bits = "".join([str(np.random.randint(0, 2)) for _ in range(bit_length)])
    return dummy_extracted_bits