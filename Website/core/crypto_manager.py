import os
import json
import uuid
import hashlib
import hmac
import base64
from datetime import datetime
from config import Config

class CryptoManager:
    @staticmethod
    def generate_watermark_package(secret_text: str, author_id: str = "USER_DEFAULT") -> dict:
        wm_id = f"WM-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().isoformat()

        raw_data = f"{wm_id}|{author_id}|{secret_text}|{timestamp}"

        signature = hmac.new(
            Config.MASTER_CRYPTO_KEY,
            raw_data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        encoded_payload = base64.b64encode(raw_data.encode('utf-8')).decode('utf-8')

        hex_64bit = signature[:16]
        bit_string = bin(int(hex_64bit, 16))[2:].zfill(64)

        package = {
            "watermark_id": wm_id,
            "author_id": author_id,
            "secret_plaintext": secret_text,
            "encrypted_payload": encoded_payload,
            "crypto_signature": signature,
            "embedded_bits_str": bit_string,
            "embedded_bits_len": len(bit_string),
            "created_at": timestamp
        }
        return package

    @staticmethod
    def save_key_file(package: dict) -> str:
        filename = f"{package['watermark_id']}.json"
        filepath = os.path.join(Config.KEY_STORAGE_FOLDER, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(package, f, ensure_ascii=False, indent=4)
        return filepath