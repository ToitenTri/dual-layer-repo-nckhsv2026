from pymongo import MongoClient
from datetime import datetime
from config import Config


class Database:
    def __init__(self):
        try:
            self.client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=3000)
            self.db = self.client[Config.DB_NAME]
            self.experiments_col = self.db["experiments"]
            self.vault_col = self.db["watermarks_vault"]
            self.client.server_info()
            print("[INFO] Kết nối MongoDB thành công!")
        except Exception as e:
            print(f"[ERROR] Không thể kết nối MongoDB: {e}")
            self.experiments_col = None
            self.vault_col = None

    def save_full_transaction(self, crypto_package: dict, key_filename: str, original_filename: str,
                              saved_input_name: str, output_name: str, process_time: float):
        if self.vault_col is None or self.experiments_col is None:
            return None

        wm_id = crypto_package["watermark_id"]
        now = datetime.now()

        # Lưu thông tin mã hóa vào watermarks_vault
        self.vault_col.insert_one({
            "watermark_id": wm_id,
            "author_id": crypto_package["author_id"],
            "secret_plaintext": crypto_package["secret_plaintext"],
            "encrypted_payload": crypto_package["encrypted_payload"],
            "crypto_signature": crypto_package["crypto_signature"],
            "embedded_bits_str": crypto_package["embedded_bits_str"],
            "key_filename": key_filename,
            "created_at": now
        })

        # Lưu thông tin thực nghiệm vào experiments
        self.experiments_col.insert_one({
            "watermark_id": wm_id,
            "original_filename": original_filename,
            "input_image": saved_input_name,
            "output_image": output_name,
            "processing_time_sec": round(process_time, 4),
            "status": "Protected (2 Layers)",
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
            {"$unwind": {"path": "$vault_info", "preserveNullAndEmptyArrays": True}}
        ]
        return list(self.experiments_col.aggregate(pipeline))

    def verify_extracted_bits(self, extracted_bits: str):
        if self.vault_col is None:
            return None
        all_vaults = list(self.vault_col.find())
        best_match = None
        lowest_ber = 1.0

        for item in all_vaults:
            db_bits = item.get("embedded_bits_str", "")
            if len(db_bits) == len(extracted_bits):
                diff_bits = sum(b1 != b2 for b1, b2 in zip(extracted_bits, db_bits))
                ber = diff_bits / len(db_bits)
                if ber < lowest_ber:
                    lowest_ber = ber
                    best_match = {
                        "watermark_id": item["watermark_id"],
                        "author_id": item["author_id"],
                        "secret_plaintext": item["secret_plaintext"],
                        "ber_percent": round(ber * 100, 2),
                        "accuracy_percent": round((1 - ber) * 100, 2),
                        "matched_bits": len(db_bits) - diff_bits,
                        "total_bits": len(db_bits)
                    }
        return best_match

    def delete_record_by_wmid(self, wm_id: str):
        if self.vault_col is not None and self.experiments_col is not None:
            self.vault_col.delete_one({"watermark_id": wm_id})
            self.experiments_col.delete_one({"watermark_id": wm_id})
            return True
        return False