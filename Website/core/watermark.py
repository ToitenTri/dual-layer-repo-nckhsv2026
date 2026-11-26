from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WATERMARK_DIR = PROJECT_ROOT / "watermark"

# Thêm cả thư mục gốc và thư mục `watermark` để các import nội bộ như `from options import *` hoạt động
for p in (str(PROJECT_ROOT), str(WATERMARK_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layer.noiser import Noiser

# Khi nào có file weight đã train, điền đường dẫn vào đây, ví dụ:
# CHECKPOINT_PATH = WATERMARK_DIR / "runs" / "exp1" / "checkpoints" / "epoch_10.pth"
CHECKPOINT_PATH = None

# Cấu hình mặc định khớp với tham số trong model.py (điều chỉnh nếu bạn train thông số khác)
DEFAULT_IMAGE_SIZE = 128
DEFAULT_BIT_LENGTH = 30

_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint_if_needed(model: Hidden, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location=_DEVICE)
    if isinstance(ckpt, dict):
        if "enc-dec-model" in ckpt:
            model.encoder_decoder.load_state_dict(ckpt["enc-dec-model"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)


def _get_model(bit_length: int = DEFAULT_BIT_LENGTH, image_size: int = DEFAULT_IMAGE_SIZE) -> Hidden:
    global _MODEL
    if _MODEL is None:
        config = HiDDenConfiguration(
            H=image_size,
            W=image_size,
            message_length=bit_length,
            encoder_blocks=4,
            encoder_channels=64,
            decoder_blocks=7,
            decoder_channels=64,
            use_discriminator=False,
            use_vgg=False,
            discriminator_blocks=3,
            discriminator_channels=64,
            decoder_loss=1,
            encoder_loss=0.7,
            adversarial_loss=1e-3,
        )
        # Noiser rỗng = không thêm nhiễu khi nhúng/tách watermark
        noiser = Noiser([], _DEVICE)
        model = Hidden(config, _DEVICE, noiser, tb_logger=None)

        if CHECKPOINT_PATH is not None and Path(CHECKPOINT_PATH).exists():
            _load_checkpoint_if_needed(model, Path(CHECKPOINT_PATH))

        model.eval()
        _MODEL = model
    return _MODEL


def _bgr_to_tensor(image_bgr: np.ndarray, target_hw: tuple[int, int]) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
    # Chuẩn hóa về [-1, 1] (chuẩn phổ biến của HiDDeN); nếu model của bạn dùng [0, 1] thì đổi thành `resized.astype(np.float32) / 255.0`
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.unsqueeze(0).to(_DEVICE)


def _tensor_to_bgr(tensor: torch.Tensor, orig_hw: tuple[int, int]) -> np.ndarray:
    # Chuyển từ [-1, 1] về [0, 255] uint8; nếu đầu vào là [0, 1] thì bỏ `+ 1.0) / 2.0` và nhân trực tiếp 255.0
    img = ((tensor.squeeze(0).detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    rgb = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if (bgr.shape[0], bgr.shape[1]) != orig_hw:
        bgr = cv2.resize(bgr, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_LINEAR)
    return bgr


def _bits_to_tensor(bit_string: str, expected_len: int) -> torch.Tensor:
    cleaned = "".join(ch for ch in bit_string if ch in ("0", "1"))
    if len(cleaned) < expected_len:
        cleaned = cleaned.ljust(expected_len, "0")
    elif len(cleaned) > expected_len:
        cleaned = cleaned[:expected_len]
    bits = [float(ch) for ch in cleaned]
    return torch.tensor(bits, dtype=torch.float32, device=_DEVICE).unsqueeze(0)


def embed_watermark(image_matrix: np.ndarray, bit_string: str) -> np.ndarray:
    if image_matrix is None or image_matrix.size == 0:
        raise ValueError("Ma trận ảnh đầu vào bị rỗng!")

    bit_len = len(bit_string) if len(bit_string) > 0 else DEFAULT_BIT_LENGTH
    model = _get_model(bit_length=bit_len, image_size=DEFAULT_IMAGE_SIZE)

    orig_hw = (image_matrix.shape[0], image_matrix.shape[1])
    img_tensor = _bgr_to_tensor(image_matrix, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
    msg_tensor = _bits_to_tensor(bit_string, bit_len)

    print(f"[CORE - WATERMARK] Đang nhúng chuỗi {len(bit_string)} bits: {bit_string[:16]}...")
    with torch.no_grad():
        # Tùy implementation của EncoderDecoder: có thể gọi `model.encoder_decoder.encoder(img_tensor, msg_tensor)`
        # hoặc lấy `encoded_images` từ `model.encoder_decoder(img_tensor, msg_tensor)`
        encoded_images, _, _ = model.encoder_decoder(img_tensor, msg_tensor)

    return _tensor_to_bgr(encoded_images, orig_hw)


def extract_watermark(image_matrix: np.ndarray, bit_length: int = 64) -> str:
    if image_matrix is None or image_matrix.size == 0:
        raise ValueError("Không thể đọc ảnh cần kiểm định!")

    model = _get_model(bit_length=bit_length, image_size=DEFAULT_IMAGE_SIZE)
    img_tensor = _bgr_to_tensor(image_matrix, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    with torch.no_grad():
        # Nếu `EncoderDecoder` có phương thức `decoder(img_tensor)` riêng thì gọi trực tiếp sẽ sạch hơn;
        # nếu không có thì truyền dummy message vào `model.encoder_decoder(img_tensor, dummy_msg)` để lấy `decoded_messages`
        if hasattr(model.encoder_decoder, "decoder") and callable(getattr(model.encoder_decoder, "decoder")):
            decoded_messages = model.encoder_decoder.decoder(img_tensor)
        else:
            dummy_msg = torch.zeros((1, bit_length), dtype=torch.float32, device=_DEVICE)
            _, _, decoded_messages = model.encoder_decoder(img_tensor, dummy_msg)

    bits = (decoded_messages.squeeze(0).detach().cpu()[:bit_length] >= 0.5).int().tolist()
    return "".join(str(b) for b in bits)
