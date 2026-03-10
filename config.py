import os
from datetime import timedelta

# Database Configuration
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///sign.db')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Secret Key
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Session Configuration
PERMANENT_SESSION_LIFETIME = timedelta(days=7)
SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# Upload Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# AI Model Configuration
MODEL_CONFIG = {
    'confidence_threshold': 0.7,
    'detection_fps': 30,
    'supported_signs': 'A-Z'
}
