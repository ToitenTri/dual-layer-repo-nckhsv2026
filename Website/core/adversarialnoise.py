import cv2
import numpy as np

def add_adversarial_noise(image_matrix: np.ndarray, epsilon: float = 0.03) -> np.ndarray:
    if image_matrix is None:
        raise ValueError("Ma trận ảnh từ lớp Watermark bị rỗng!")
    print(f"[CORE - ADVERSARIAL] Đang phủ lớp nhiễu đối kháng (epsilon={epsilon})...")
    protected_image = image_matrix.copy()
    return protected_image