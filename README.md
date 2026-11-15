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

## Tiến độ hiện tại

### Đã có

- Tạo khung thư mục nghiên cứu cho dữ liệu, watermark, biến đổi ảnh, nhiễu
  đối kháng, đánh giá và báo cáo.
- Có ứng dụng web prototype trong `Website/` để tải ảnh, tạo watermark ID,
  lưu metadata/key và hiển thị kết quả.
- Có giao diện manager để xem transaction, tải ảnh xác minh và xóa bản ghi.
- Có lớp tạo payload gồm watermark ID, author ID, timestamp, HMAC signature và
  chuỗi 64-bit.
- Đã tích hợp các baseline nghiên cứu:
  - HiDDeN trong `watermark/model/HiDDeN/`.
  - Picode trong `watermark/model/picode/`.
  - StegaStamp Original trong `watermark/stegastamp-original/`.
- Đã chuẩn bị các thư mục dữ liệu `raw/`, `train/`, `val/` và `test/`.

### Đang phát triển

- Hoàn thiện thuật toán nhúng watermark thực tế trong
  `Website/core/watermark.py`.
- Hoàn thiện thuật toán trích xuất và xác minh watermark.
- Thay thế lớp `adversarialnoise.py` hiện tại bằng pipeline nhiễu đối kháng
  thực tế.
- Bổ sung pipeline đo PSNR, SSIM, bit accuracy/BER trong `evaluation/`.
- Bổ sung các phép nén, resize, crop và biến đổi ảnh trong `transforms/`.
- Kết nối các model nghiên cứu với luồng web thay vì dùng prototype độc lập.
- Bổ sung test tự động cho các luồng tạo, nhúng, trích xuất và xác minh.

### Lưu ý trạng thái prototype

Ứng dụng web hiện chạy bằng Flask và dùng MongoDB, dù phần mô tả cấu trúc cũ
phía trên vẫn giữ nguyên để bảo toàn tài liệu ban đầu. Các hàm
`embed_watermark()`, `extract_watermark()` và `add_adversarial_noise()` hiện
chưa phải pipeline watermark hoàn chỉnh; không sử dụng hệ thống cho xác thực,
bảo mật hoặc bảo vệ bản quyền trong môi trường production.

## Cấu trúc thực tế hiện tại

```text
.
├── Website/                    # Flask prototype và giao diện web
│   ├── app.py                  # Route user và manager
│   ├── config.py               # Cấu hình ứng dụng, MongoDB, thư mục lưu trữ
│   ├── database.py             # Lưu transaction và đối sánh watermark
│   ├── core/
│   │   ├── crypto_manager.py   # Tạo watermark ID, HMAC và payload
│   │   ├── watermark.py        # API nhúng và trích xuất watermark
│   │   └── adversarialnoise.py # API thêm nhiễu đối kháng
│   ├── templates/              # Giao diện HTML
│   ├── static/uploads/         # Ảnh đầu vào
│   ├── static/results/         # Ảnh kết quả
│   └── storage/keys/           # File JSON metadata/key
├── watermark/
│   ├── model/HiDDeN/           # Baseline HiDDeN và experiment
│   ├── model/picode/           # Model, scraper, iOS scanner và tài liệu Picode
│   └── stegastamp-original/    # Baseline StegaStamp và checkpoint
├── data/                       # raw/train/val/test
├── transforms/                 # Biến đổi ảnh
├── adversarial/                # Nhiễu đối kháng
├── evaluation/                 # Đo lường chất lượng
├── reports/                    # Báo cáo và kết quả
├── main.py                     # Entrypoint mẫu
├── requirements.txt            # Danh sách dependency
└── .gitignore
```

### Chạy prototype web

Ứng dụng web yêu cầu Python environment có thêm `Flask` và `pymongo`, cùng
MongoDB local theo cấu hình trong `Website/config.py`:

```powershell
pip install Flask pymongo
cd Website
..\.venv\Scripts\python.exe app.py
```

Sau đó mở <http://127.0.0.1:5000/>.

Các thông tin mặc định hiện chỉ phù hợp cho local development. Secret key,
tài khoản manager và dữ liệu plaintext cần được chuyển sang cơ chế bảo mật
thích hợp trước khi triển khai.
