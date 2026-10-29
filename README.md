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

Dữ liệu nghiên cứu đặt trong `data/` và không được đưa vào Git. Các thư mục
con dự kiến gồm `raw/`, `train/`, `val/` và `test/`.