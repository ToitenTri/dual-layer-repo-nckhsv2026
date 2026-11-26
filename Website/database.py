import uuid
from datetime import datetime
from pymongo import MongoClient
from config import Config
from core.crypto_manager import CryptoManager


class Database:
    def __init__(self):
        try:
            self.client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=3000)
            self.db = self.client[Config.DB_NAME]
            # 1. authors: Bảng lưu thông tin tác giả
            self.authors_col = self.db["authors"]
            # 2. watermarks_vault: Bảng lưu thông tin bản quyền (FK -> author_id)
            self.vault_col = self.db["watermarks_vault"]
            # 3. experiments: Bảng lưu các lượt thử nghiệm/kiểm định (PK -> experiment_id, FK -> watermark_id)
            self.experiments_col = self.db["experiments"]

            self.client.server_info()
            print("[INFO] Kết nối MongoDB thành công")
        except Exception as e:
            print(f"[ERROR] Không thể kết nối MongoDB: {e}")
            self.authors_col = None
            self.vault_col = None
            self.experiments_col = None

    def save_full_transaction(self, crypto_package: dict, key_filename: str = None,
                              original_filename: str = "", saved_input_name: str = "",
                              output_name: str = "", process_time: float = 0.0,
                              epsilon: float = 0.03):
        """
        Lưu dữ liệu tuân thủ chuẩn 3NF:
        - Tách thông tin Tác giả (AUTHORS) - Quan hệ 1 : N
        - Lưu Thông tin Bản quyền (WATERMARKS) - Loại bỏ trường bắc cầu embedded_bits_str và key_filename
        - Lưu Lượt Thực nghiệm (EXPERIMENTS) - Khóa chính experiment_id riêng biệt, FK watermark_id
        """
        if self.vault_col is None or self.experiments_col is None:
            return None

        wm_id = crypto_package["watermark_id"]
        author_id = crypto_package.get("author_id", "Anonymous")
        now = datetime.now()

        # 1. Thực thể AUTHORS (Quan hệ 1 : N với WATERMARKS)
        if self.authors_col is not None:
            self.authors_col.update_one(
                {"author_id": author_id},
                {"$setOnInsert": {"author_id": author_id, "created_at": now}},
                upsert=True
            )

        # 2. Thực thể WATERMARKS_VAULT (Chuẩn 3NF: Loại bỏ embedded_bits_str và key_filename)
        self.vault_col.insert_one({
            "watermark_id": wm_id,
            "author_id": author_id,  # Khóa ngoại FK tham chiếu AUTHORS
            "secret_plaintext": crypto_package["secret_plaintext"],
            "encrypted_payload": crypto_package["encrypted_payload"],
            "crypto_signature": crypto_package["crypto_signature"],
            "created_at": now
        })

        # 3. Thực thể EXPERIMENTS (Chuẩn 3NF: Khóa chính experiment_id riêng, FK watermark_id)
        exp_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        ext = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'png'

        self.experiments_col.insert_one({
            "experiment_id": exp_id,
            "watermark_id": wm_id,  # Khóa ngoại FK tham chiếu WATERMARKS_VAULT
            "original_filename": original_filename,
            "file_ext": ext,
            "experiment_type": "Protect_DualLayer",
            "epsilon": epsilon,
            "processing_time_sec": round(process_time, 4),
            "created_at": now
        })

        return wm_id

    def get_all_joined_records(self):
        if self.experiments_col is None:
            return []

        pipeline = [
            {"$sort": {"created_at": -1}},
            {
                "$lookup": {
                    "from": "watermarks_vault",
                    "localField": "watermark_id",
                    "foreignField": "watermark_id",
                    "as": "vault_info"
                }
            },
            {"$unwind": {"path": "$vault_info", "preserveNullAndEmptyArrays": True}},
            {
                "$lookup": {
                    "from": "authors",
                    "localField": "vault_info.author_id",
                    "foreignField": "author_id",
                    "as": "author_info"
                }
            },
            {"$unwind": {"path": "$author_info", "preserveNullAndEmptyArrays": True}}
        ]

        records = list(self.experiments_col.aggregate(pipeline))

        # Khôi phục các trường dẫn xuất trên tầng hiển thị để tương thích tuyệt đối giao diện
        for r in records:
            wm_id = r.get("watermark_id", "")
            ext = r.get("file_ext", "png")

            if "input_image" not in r:
                r["input_image"] = f"in_{wm_id}.{ext}"
            if "output_image" not in r:
                r["output_image"] = f"out_{wm_id}.png"

            if r.get("vault_info"):
                sig = r["vault_info"].get("crypto_signature", "")
                if "embedded_bits_str" not in r["vault_info"] and sig:
                    r["vault_info"]["embedded_bits_str"] = CryptoManager.signature_to_bits(sig)
                if "key_filename" not in r["vault_info"]:
                    r["vault_info"]["key_filename"] = f"{wm_id}.json"

        return records

    def verify_extracted_bits(self, extracted_bits: str):
        """
        Đối soát chuỗi bits trích xuất từ ảnh với cơ sở dữ liệu.
        Chuỗi bit trong DB được sinh trực tiếp từ chữ ký crypto_signature.
        """
        if self.vault_col is None:
            return None

        all_vaults = list(self.vault_col.find())
        best_match = None
        lowest_ber = 1.0

        for item in all_vaults:
            # Lấy chuỗi bit: tính toán động từ crypto_signature (3NF) hoặc lấy trường cũ nếu có
            db_bits = item.get("embedded_bits_str")
            if not db_bits and item.get("crypto_signature"):
                db_bits = CryptoManager.signature_to_bits(item["crypto_signature"])

            if db_bits and len(db_bits) == len(extracted_bits):
                diff_bits = sum(b1 != b2 for b1, b2 in zip(extracted_bits, db_bits))
                ber = diff_bits / len(db_bits)
                if ber < lowest_ber:
                    lowest_ber = ber
                    best_match = {
                        "watermark_id": item["watermark_id"],
                        "author_id": item.get("author_id", "Unknown"),
                        "secret_plaintext": item.get("secret_plaintext", ""),
                        "ber_percent": round(ber * 100, 2),
                        "accuracy_percent": round((1 - ber) * 100, 2),
                        "matched_bits": len(db_bits) - diff_bits,
                        "total_bits": len(db_bits)
                    }
        return best_match

    def delete_record_by_wmid(self, wm_id: str):
        """
        Xóa bản quyền và thực hiện xóa lan truyền (Cascade Delete) toàn bộ
        các thực nghiệm liên quan theo quan hệ 1 : N.
        """
        if self.vault_col is not None and self.experiments_col is not None:
            self.vault_col.delete_one({"watermark_id": wm_id})
            # Xóa toàn bộ các lượt thử nghiệm thuộc watermark này (Cascade Delete 1:N)
            self.experiments_col.delete_many({"watermark_id": wm_id})
            return True
        return False
