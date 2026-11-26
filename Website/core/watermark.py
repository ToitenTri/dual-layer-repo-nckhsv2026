from pathlib import Path
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F

WEBSITE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WATERMARK_DIR = PROJECT_ROOT / "watermark"

# Giữ thư mục `Website` ưu tiên trước để không xung đột module nội bộ,
# đồng thời bổ sung `watermark` và thư mục gốc để các import trong `watermark` hoạt động.
if str(WEBSITE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSITE_DIR))
for p in (str(WATERMARK_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.append(p)

from options import HiDDenConfiguration
from model.hidden import Hidden
from noise_layer.noiser import Noiser

# Khi nào có file weight đã train, điền đường dẫn vào đây (hoặc đặt biến môi trường WATERMARK_CHECKPOINT), ví dụ:
# CHECKPOINT_PATH = WATERMARK_DIR / "runs" / "exp1" / "checkpoints" / "epoch_10.pyt"
CHECKPOINT_PATH = None

# Cấu hình mặc định: khớp 64-bit payload của CryptoManager và kích thước 128x128 của HiDDeN
DEFAULT_IMAGE_SIZE = 128
DEFAULT_BIT_LENGTH = 64

_MODEL_CACHE: dict[tuple[int, int, str], Hidden] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Các cặp hệ số DCT trung tần đối xứng dùng cho bộ nhúng/tách ẩn dự phòng khi chưa có checkpoint HiDDeN
_DCT_PAIRS = (
    (3, 2, 2, 3),
    (4, 1, 1, 4),
    (4, 2, 2, 4),
    (3, 4, 4, 3),
    (2, 5, 5, 2),
    (1, 5, 5, 1),
    (2, 4, 4, 2),
    (1, 3, 3, 1),
)
_DCT_STRENGTH = 24.0


def _build_dct_matrix(n: int = 8) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    d = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    d[0, :] *= np.sqrt(1.0 / n)
    d[1:, :] *= np.sqrt(2.0 / n)
    return d.astype(np.float32)


_DCT_MAT = _build_dct_matrix(8)


def _resolve_checkpoint_path() -> Path | None:
    """Tìm đường dẫn checkpoint đã train (từ CHECKPOINT_PATH, biến môi trường, hoặc thư mục checkpoints)."""
    candidates: list[Path] = []
    if CHECKPOINT_PATH is not None:
        candidates.append(Path(CHECKPOINT_PATH))

    env_ckpt = os.environ.get("WATERMARK_CHECKPOINT", "").strip()
    if env_ckpt:
        candidates.append(Path(env_ckpt))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    search_roots = (
        WEBSITE_DIR / "storage" / "checkpoints",
        WATERMARK_DIR / "runs",
    )
    found: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            for ext in ("*.pth", "*.pyt", "*.pt", "*.ckpt"):
                found.extend(root.rglob(ext))
    if found:
        return max(found, key=lambda p: p.stat().st_mtime)
    return None


def _torch_load_compat(ckpt_path: Path):
    try:
        return torch.load(str(ckpt_path), map_location=_DEVICE, weights_only=False)
    except TypeError:
        return torch.load(str(ckpt_path), map_location=_DEVICE)


def _infer_ckpt_bit_length(ckpt, fallback: int) -> int:
    if not isinstance(ckpt, dict):
        return fallback
    state = ckpt.get("enc-dec-model") or ckpt.get("encoder_decoder") or ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
    if isinstance(state, dict):
        for key in ("decoder.linear.weight", "encoder_decoder.decoder.linear.weight"):
            if key in state and hasattr(state[key], "shape"):
                return int(state[key].shape[0])
    return fallback


def _load_checkpoint_into_model(model: Hidden, ckpt) -> None:
    if isinstance(ckpt, dict):
        if "enc-dec-model" in ckpt:
            model.encoder_decoder.load_state_dict(ckpt["enc-dec-model"])
        elif "encoder_decoder" in ckpt:
            model.encoder_decoder.load_state_dict(ckpt["encoder_decoder"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)


def _get_model(
    bit_length: int = DEFAULT_BIT_LENGTH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    ckpt_path: Path | None = None,
) -> Hidden:
    ckpt = _torch_load_compat(ckpt_path) if ckpt_path is not None else None
    effective_bits = _infer_ckpt_bit_length(ckpt, bit_length) if ckpt is not None else bit_length
    cache_key = (effective_bits, image_size, str(ckpt_path.resolve()) if ckpt_path else "")

    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    config = HiDDenConfiguration(
        H=image_size,
        W=image_size,
        message_length=effective_bits,
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
    noiser = Noiser([], _DEVICE)
    model = Hidden(config, _DEVICE, noiser, tb_logger=None)

    if ckpt is not None:
        _load_checkpoint_into_model(model, ckpt)

    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def _normalize_image_channels(image_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Chuẩn hóa ảnh đầu vào về BGR 3 kênh, giữ lại kênh alpha nếu là ảnh BGRA."""
    if image_matrix is None or image_matrix.size == 0:
        raise ValueError("Ma trận ảnh đầu vào bị rỗng!")

    if image_matrix.ndim == 2:
        return cv2.cvtColor(image_matrix, cv2.COLOR_GRAY2BGR), None
    if image_matrix.ndim == 3 and image_matrix.shape[2] == 4:
        return image_matrix[:, :, :3].copy(), image_matrix[:, :, 3].copy()
    if image_matrix.ndim == 3 and image_matrix.shape[2] == 3:
        return image_matrix.copy(), None
    raise ValueError(f"Định dạng số kênh ảnh không hợp lệ: shape={image_matrix.shape}")


def _restore_alpha(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return bgr
    return np.dstack([bgr, alpha])


def _bgr_to_tensor(image_bgr: np.ndarray, target_hw: tuple[int, int] | None = None) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if target_hw is not None and (rgb.shape[0], rgb.shape[1]) != target_hw:
        interp = cv2.INTER_AREA if rgb.shape[0] >= target_hw[0] and rgb.shape[1] >= target_hw[1] else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (target_hw[1], target_hw[0]), interpolation=interp)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.unsqueeze(0).to(_DEVICE)


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    img = ((tensor.squeeze(0).detach().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    rgb = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _normalize_bits(bit_string: str, expected_len: int) -> str:
    cleaned = "".join(ch for ch in bit_string if ch in ("0", "1"))
    if len(cleaned) < expected_len:
        cleaned = cleaned.ljust(expected_len, "0")
    elif len(cleaned) > expected_len:
        cleaned = cleaned[:expected_len]
    return cleaned


def _bits_to_tensor(bit_string: str, expected_len: int) -> torch.Tensor:
    cleaned = _normalize_bits(bit_string, expected_len)
    bits = [float(ch) for ch in cleaned]
    return torch.tensor(bits, dtype=torch.float32, device=_DEVICE).unsqueeze(0)


def _select_dct_pairs(n_blocks: int, n_bits: int) -> tuple[tuple[int, int, int, int], ...]:
    if n_blocks >= n_bits:
        return (_DCT_PAIRS[0],)
    pairs_needed = min(len(_DCT_PAIRS), (n_bits + n_blocks - 1) // n_blocks)
    return _DCT_PAIRS[:pairs_needed]


def _embed_dct_invisible(image_bgr: np.ndarray, bits: str) -> np.ndarray:
    """Nhúng watermark ẩn bằng DCT 8x8 trên kênh độ chói Y (giữ nguyên độ phân giải và màu sắc)."""
    n_bits = len(bits)
    h, w = image_bgr.shape[:2]
    if h < 8 or w < 8:
        raise ValueError("Ảnh quá nhỏ (cần tối thiểu 8x8 pixel để nhúng watermark)!")

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    bh, bw = h // 8, w // 8
    n_blocks = bh * bw
    pairs = _select_dct_pairs(n_blocks, n_bits)
    n_pairs = len(pairs)

    slot_indices = np.arange(n_blocks * n_pairs) % n_bits
    bit_signs = np.array([1.0 if bits[idx] == "1" else -1.0 for idx in slot_indices], dtype=np.float32)
    bit_signs = bit_signs.reshape(bh, bw, n_pairs)

    y_crop = y[: bh * 8, : bw * 8]
    for _ in range(2):
        blocks = y_crop.reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
        dct_blocks = _DCT_MAT @ blocks @ _DCT_MAT.T

        for p_idx, (u1, v1, u2, v2) in enumerate(pairs):
            c1 = dct_blocks[:, :, u1, v1]
            c2 = dct_blocks[:, :, u2, v2]
            signs = bit_signs[:, :, p_idx]
            diff = c1 - c2
            mask = (signs * diff) < _DCT_STRENGTH
            adj = np.where(mask, (signs * _DCT_STRENGTH - diff) * 0.5, 0.0)
            dct_blocks[:, :, u1, v1] = c1 + adj
            dct_blocks[:, :, u2, v2] = c2 - adj

        idct_blocks = _DCT_MAT.T @ dct_blocks @ _DCT_MAT
        y_crop = np.clip(np.round(idct_blocks.transpose(0, 2, 1, 3).reshape(bh * 8, bw * 8)), 0, 255).astype(np.float32)

    ycrcb[: bh * 8, : bw * 8, 0] = y_crop.astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _extract_dct_invisible(image_bgr: np.ndarray, bit_length: int) -> str:
    """Trích xuất chuỗi bit watermark từ các khối DCT 8x8 trên kênh Y bằng biểu quyết có trọng số."""
    h, w = image_bgr.shape[:2]
    if h < 8 or w < 8:
        raise ValueError("Ảnh quá nhỏ (cần tối thiểu 8x8 pixel để trích xuất watermark)!")

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32)
    bh, bw = h // 8, w // 8
    n_blocks = bh * bw
    pairs = _select_dct_pairs(n_blocks, bit_length)
    n_pairs = len(pairs)

    blocks = y[: bh * 8, : bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3)
    dct_blocks = _DCT_MAT @ blocks @ _DCT_MAT.T

    diffs = np.stack([dct_blocks[:, :, u1, v1] - dct_blocks[:, :, u2, v2] for u1, v1, u2, v2 in pairs], axis=-1)
    flat_diffs = diffs.reshape(-1)
    slot_indices = np.arange(flat_diffs.size) % bit_length
    scores = np.bincount(slot_indices, weights=flat_diffs, minlength=bit_length)
    return "".join("1" if s >= 0.0 else "0" for s in scores[:bit_length])


def embed_watermark(image_matrix: np.ndarray, bit_string: str) -> np.ndarray:
    bgr, alpha = _normalize_image_channels(image_matrix)
    bit_len = len(bit_string) if len(bit_string) > 0 else DEFAULT_BIT_LENGTH
    normalized_bits = _normalize_bits(bit_string, bit_len)

    ckpt_path = _resolve_checkpoint_path()
    if ckpt_path is not None:
        model = _get_model(bit_length=bit_len, image_size=DEFAULT_IMAGE_SIZE, ckpt_path=ckpt_path)
        model_bits = model.config.message_length
        orig_hw = (bgr.shape[0], bgr.shape[1])

        orig_tensor = _bgr_to_tensor(bgr, target_hw=None)
        img_128 = _bgr_to_tensor(bgr, target_hw=(model.config.H, model.config.W))
        msg_tensor = _bits_to_tensor(normalized_bits, model_bits)

        print(f"[CORE - WATERMARK] Đang nhúng bằng HiDDeN ({ckpt_path.name}) chuỗi {len(normalized_bits)} bits...")
        with torch.no_grad():
            encoded_128 = model.encoder_decoder.encoder(img_128, msg_tensor)
            # Áp dụng phần dư (residual) lên ảnh gốc để giữ nguyên độ phân giải sắc nét ban đầu
            delta_128 = encoded_128 - img_128
            delta_full = F.interpolate(delta_128, size=orig_hw, mode="bilinear", align_corners=False)
            out_tensor = (orig_tensor + delta_full).clamp(-1.0, 1.0)

        return _restore_alpha(_tensor_to_bgr(out_tensor), alpha)

    print(
        f"[CORE - WATERMARK] Chưa có checkpoint HiDDeN đã train (CHECKPOINT_PATH=None); "
        f"đang nhúng {bit_len} bits bằng DCT ẩn: {normalized_bits[:16]}..."
    )
    watermarked_bgr = _embed_dct_invisible(bgr, normalized_bits)
    return _restore_alpha(watermarked_bgr, alpha)


def extract_watermark(image_matrix: np.ndarray, bit_length: int = DEFAULT_BIT_LENGTH) -> str:
    if bit_length <= 0:
        raise ValueError("bit_length phải lớn hơn 0!")
    bgr, _ = _normalize_image_channels(image_matrix)

    ckpt_path = _resolve_checkpoint_path()
    if ckpt_path is not None:
        model = _get_model(bit_length=bit_length, image_size=DEFAULT_IMAGE_SIZE, ckpt_path=ckpt_path)
        img_tensor = _bgr_to_tensor(bgr, target_hw=(model.config.H, model.config.W))

        with torch.no_grad():
            if hasattr(model.encoder_decoder, "decoder") and callable(getattr(model.encoder_decoder, "decoder")):
                decoded_messages = model.encoder_decoder.decoder(img_tensor)
            else:
                dummy_msg = torch.zeros((1, model.config.message_length), dtype=torch.float32, device=_DEVICE)
                _, _, decoded_messages = model.encoder_decoder(img_tensor, dummy_msg)

        bits = (decoded_messages.squeeze(0).detach().cpu()[:bit_length] >= 0.5).int().tolist()
        return _normalize_bits("".join(str(b) for b in bits), bit_length)

    return _extract_dct_invisible(bgr, bit_length)
