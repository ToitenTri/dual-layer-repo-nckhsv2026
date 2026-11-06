import os


class Config:
    SECRET_KEY = 'nckh_secret_key_2026'
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Đường dẫn thư mục lưu trữ
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
    RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
    KEY_STORAGE_FOLDER = os.path.join(BASE_DIR, 'storage', 'keys')

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Giới hạn 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'jepg'}

    # Cấu hình MongoDB
    MONGO_URI = "mongodb://localhost:27017/"
    DB_NAME = "ImageProtectionDB"

    MASTER_CRYPTO_KEY = b'nckh_secret_master_key_32bytes!!'

    # Tài khoản quản trị Manager
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "123456"

    @staticmethod
    def init_Folders():
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULT_FOLDER, exist_ok=True)
        os.makedirs(Config.KEY_STORAGE_FOLDER, exist_ok=True)