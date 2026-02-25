"""
MemeShield Configuration Module
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration class."""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'memeshield-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    
    # MySQL Database settings
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'memeshield')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    
    # Model settings
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    CNN_MODEL_NAME = 'MobileNetV2'  # Options: 'MobileNetV2', 'ResNet50'
    IMAGE_SIZE = (224, 224)
    
    # OCR settings
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    # Text processing settings
    MAX_TEXT_LENGTH = 512
    EMBEDDING_DIM = 128
    
    @staticmethod
    def get_db_uri():
        """Generate MySQL connection URI."""
        return f"mysql+pymysql://{Config.MYSQL_USER}:{Config.MYSQL_PASSWORD}@{Config.MYSQL_HOST}:{Config.MYSQL_PORT}/{Config.MYSQL_DATABASE}"


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


# Configuration dictionary
config_dict = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
