# Transforms

Thư mục này dành cho các phép biến đổi ảnh dùng trong huấn luyện, đánh giá độ
bền và mô phỏng các tác động lên watermark. Hiện tại các file Python bên dưới
là skeleton để thống nhất cấu trúc; thuật toán xử lý ảnh sẽ được bổ sung theo
từng giai đoạn.

## Cấu trúc

```text
transforms/
├── README.md
├── __init__.py
├── base.py
├── registry.py
├── pipeline.py
├── image/
│   ├── __init__.py
│   ├── resize.py
│   ├── crop.py
│   ├── rotate.py
│   ├── perspective.py
│   ├── color.py
│   └── blur.py
├── compression/
│   ├── __init__.py
│   ├── jpeg.py
│   ├── png.py
│   └── webp.py
├── noise/
│   ├── __init__.py
│   ├── gaussian.py
│   ├── salt_pepper.py
│   ├── poisson.py
│   └── speckle.py
├── print_capture/
│   ├── __init__.py
│   ├── print_simulation.py
│   ├── camera_noise.py
│   ├── lighting.py
│   └── screen_capture.py
├── watermark/
│   ├── __init__.py
│   ├── watermark_preserving.py
│   ├── watermark_destroying.py
│   └── watermark_attack.py
├── configs/
│   ├── default.yaml
│   ├── training.yaml
│   ├── evaluation.yaml
│   └── robustness.yaml
└── tests/
    ├── test_resize.py
    ├── test_compression.py
    ├── test_noise.py
    ├── test_pipeline.py
    └── test_reproducibility.py
```

## Vai trò các nhóm

- `image/`: resize, crop, rotate, perspective, màu sắc và blur.
- `compression/`: mô phỏng JPEG, PNG và WebP.
- `noise/`: nhiễu Gaussian, salt-and-pepper, Poisson và speckle.
- `print_capture/`: mô phỏng in ảnh, chụp camera, ánh sáng và chụp màn hình.
- `watermark/`: nhóm các biến đổi theo tác động lên watermark: giữ, phá hủy
  hoặc tấn công watermark.
- `configs/`: tham số cho training, evaluation và robustness; không hard-code
  tham số thí nghiệm trong các script.
- `tests/`: kiểm tra kích thước, phạm vi pixel, pipeline và khả năng tái lập.

## Thành phần điều phối

- `base.py`: giao diện chung của một transform.
- `registry.py`: đăng ký và tra cứu transform theo tên.
- `pipeline.py`: kết hợp nhiều transform theo thứ tự và xác suất áp dụng.

## Thứ tự triển khai

1. `image/resize.py`, `image/crop.py` và `image/blur.py`.
2. `compression/jpeg.py`.
3. `image/perspective.py`.
4. `noise/gaussian.py`.
5. Nhóm `print_capture/`.
6. Nhóm `watermark/`.
7. `pipeline.py`, `configs/` và `tests/`.

Các file hiện chỉ là khung tổ chức, chưa được xem là pipeline biến đổi ảnh
hoàn chỉnh.
