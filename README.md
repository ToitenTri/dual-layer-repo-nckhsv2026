# dual-layer-repo-nckhsv2026

Khung dự án nghiên cứu watermark ảnh hai lớp, gồm mô hình nhúng/giải mã,
mô phỏng biến đổi ảnh, nhiễu đối kháng và đánh giá chất lượng.

## Cấu trúc

```text
data/          Ảnh gốc và các tập train/val/test (không commit)
watermark/     Bộ nhúng và bộ giải mã
transforms/    Mô phỏng nén, resize và crop
adversarial/   Nhiễu đối kháng
evaluation/    PSNR, SSIM và tỷ lệ đọc đúng mã
web/           Ứng dụng Django
  config/      Cấu hình dự án
  database/    Thành phần truy cập dữ liệu
  protect/     Chức năng bảo vệ
  templates/   Giao diện HTML
reports/       Báo cáo và kết quả thí nghiệm
```

## Thiết lập môi trường với `venv`

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Danh sách thư viện trong `requirements.txt` được đồng bộ từ môi trường `.venv`
hiện tại bằng lệnh:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

Các nhóm thư viện chính:

- **Web:** Django
- **Deep learning:** PyTorch, TorchVision, TorchMetrics, Diffusers,
  Transformers, Accelerate, PEFT
- **Watermark và nhiễu:** Kornia, LPIPS, torchattacks, OpenCV
- **Xử lý và đánh giá ảnh:** NumPy, Pillow, SciPy, scikit-image,
  scikit-learn, Pandas, Matplotlib
- **Kiểm thử:** pytest

## Thư viện theo từng người

| Người | Thư viện sử dụng theo đầu việc | Tuần dùng |
|---|---|---|
| Tòng Đức Trí | torch, torchvision; numpy, pillow, opencv-python, tqdm; tensorboard; pywavelets; kornia, diffjpeg; scikit-image, lpips; diffusers, transformers, accelerate, safetensors, huggingface_hub | Tuần 4 trở đi; tensorboard tuần 4; pywavelets tuần 6-7; kornia, diffjpeg tuần 7; scikit-image, lpips tuần 8; diffusers, transformers, accelerate, safetensors, huggingface_hub tuần 8-14 |
| Nguyễn Thùy Linh | numpy, pillow, opencv-python; django; python-dotenv; torch, torchvision; tensorboard | Tuần 4 trở đi; django tuần 4-6 (CSDL), 14 (web); tensorboard tuần 8; python-dotenv tuần 14 |
| Đỗ Bích Diệp | numpy, pillow, opencv-python; django; torch, torchvision; tensorboard; peft | Tuần 4 trở đi; django tuần 4-6 (CSDL), 14 (web); tensorboard tuần 8; peft tuần 11 |
| Lê Gia Linh | numpy, pillow, opencv-python, matplotlib; scikit-learn; transformers (CLIP) | Tuần 4 trở đi; scikit-learn tuần 4-6; transformers (CLIP) tuần 11 |
| Vũ Uyên Linh | numpy, pillow, opencv-python, matplotlib; scikit-learn | Tuần 4 trở đi; scikit-learn tuần 4-6 |

Dữ liệu nghiên cứu đặt trong `data/` và không được đưa vào Git. Các thư mục
con dự kiến gồm `raw/`, `train/`, `val/` và `test/`.