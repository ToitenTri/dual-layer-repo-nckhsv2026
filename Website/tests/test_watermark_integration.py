import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

WEBSITE_DIR = Path(__file__).resolve().parents[1]
if str(WEBSITE_DIR) not in sys.path:
    sys.path.insert(0, str(WEBSITE_DIR))

from app import app
from core import watermark as wm_mod
from core.crypto_manager import CryptoManager


class WatermarkIntegrationTests(unittest.TestCase):
    def test_embed_and_extract_preserves_image_quality_and_bits(self):
        sample_path = WEBSITE_DIR / "static" / "uploads" / "in_WM-6419AC1F.jpg"
        image = cv2.imread(str(sample_path))
        self.assertIsNotNone(image)

        pkg = CryptoManager.generate_watermark_package("ban quyen thu nghiem", "Tester")
        bits = pkg["embedded_bits_str"]
        self.assertEqual(len(bits), 64)

        watermarked = wm_mod.embed_watermark(image, bits)
        self.assertEqual(watermarked.shape, image.shape)
        self.assertEqual(watermarked.dtype, np.uint8)

        ok, buf = cv2.imencode(".png", watermarked)
        self.assertTrue(ok)
        decoded_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        psnr = cv2.PSNR(image, decoded_img)
        self.assertGreater(psnr, 38.0)

        extracted = wm_mod.extract_watermark(decoded_img, bit_length=64)
        self.assertEqual(extracted, bits)

    def test_handles_grayscale_bgra_and_small_images(self):
        bits = "10100101" * 8
        rng = np.random.default_rng(42)

        gray = rng.integers(30, 225, (64, 64), dtype=np.uint8)
        wm_gray = wm_mod.embed_watermark(gray, bits)
        self.assertEqual(wm_gray.shape, (64, 64, 3))
        self.assertEqual(wm_mod.extract_watermark(wm_gray, 64), bits)

        bgra = rng.integers(30, 225, (64, 64, 4), dtype=np.uint8)
        wm_bgra = wm_mod.embed_watermark(bgra, bits)
        self.assertEqual(wm_bgra.shape, (64, 64, 4))
        self.assertTrue(np.array_equal(wm_bgra[:, :, 3], bgra[:, :, 3]))
        self.assertEqual(wm_mod.extract_watermark(wm_bgra, 64), bits)

        small = rng.integers(30, 225, (32, 32, 3), dtype=np.uint8)
        wm_small = wm_mod.embed_watermark(small, bits)
        self.assertEqual(wm_small.shape, (32, 32, 3))
        self.assertEqual(wm_mod.extract_watermark(wm_small, 64), bits)

    def test_hidden_checkpoint_mode_and_cache_isolation(self):
        model_64 = wm_mod._get_model(bit_length=64, image_size=128, ckpt_path=None)
        model_30 = wm_mod._get_model(bit_length=30, image_size=128, ckpt_path=None)
        self.assertIsNot(model_64, model_30)
        self.assertEqual(model_64.config.message_length, 64)
        self.assertEqual(model_30.config.message_length, 30)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = Path(tmpdir) / "test_hidden.pyt"
            torch.save({"enc-dec-model": model_64.encoder_decoder.state_dict()}, str(ckpt_file))

            image = np.random.default_rng(7).integers(0, 255, (200, 300, 3), dtype=np.uint8)
            bits = "01" * 32
            with patch.object(wm_mod, "CHECKPOINT_PATH", ckpt_file):
                out = wm_mod.embed_watermark(image, bits)
                self.assertEqual(out.shape, image.shape)
                ext = wm_mod.extract_watermark(out, bit_length=64)
                self.assertEqual(len(ext), 64)
                self.assertTrue(set(ext).issubset({"0", "1"}))

    def test_flask_user_and_manager_routes(self):
        sample_path = WEBSITE_DIR / "static" / "uploads" / "in_WM-6419AC1F.jpg"
        orig_img = cv2.imread(str(sample_path))
        self.assertIsNotNone(orig_img)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            upload_dir = tmp / "uploads"
            result_dir = tmp / "results"
            key_dir = tmp / "keys"
            upload_dir.mkdir()
            result_dir.mkdir()
            key_dir.mkdir()

            app.config["TESTING"] = True
            with (
                patch.dict(
                    app.config,
                    {
                        "UPLOAD_FOLDER": str(upload_dir),
                        "RESULT_FOLDER": str(result_dir),
                        "KEY_STORAGE_FOLDER": str(key_dir),
                    },
                ),
                patch("core.crypto_manager.Config.KEY_STORAGE_FOLDER", str(key_dir)),
            ):
                client = app.test_client()
                with open(sample_path, "rb") as fh:
                    resp = client.post(
                        "/",
                        data={
                            "author_id": "test_author",
                            "watermark_text": "test secret",
                            "image_file": (io.BytesIO(fh.read()), "dog.jpg"),
                        },
                        content_type="multipart/form-data",
                    )
                self.assertEqual(resp.status_code, 200)

                out_files = list(result_dir.glob("out_WM-*.png"))
                self.assertEqual(len(out_files), 1)
                out_img = cv2.imread(str(out_files[0]))
                self.assertIsNotNone(out_img)
                self.assertGreater(cv2.PSNR(orig_img, out_img), 38.0)

                key_files = list(key_dir.glob("WM-*.json"))
                self.assertEqual(len(key_files), 1)
                import json

                expected_bits = json.loads(key_files[0].read_text(encoding="utf-8"))["embedded_bits_str"]

                client.post(
                    "/manager/login",
                    data={
                        "username": app.config["ADMIN_USERNAME"],
                        "password": app.config["ADMIN_PASSWORD"],
                    },
                )
                with open(out_files[0], "rb") as out_fh:
                    verify_resp = client.post(
                        "/manager",
                        data={"verify_image": (io.BytesIO(out_fh.read()), "verify.png")},
                        content_type="multipart/form-data",
                    )
                self.assertEqual(verify_resp.status_code, 200)
                self.assertIn(expected_bits.encode("utf-8"), verify_resp.data)


if __name__ == "__main__":
    unittest.main()

