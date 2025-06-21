import os
from typing import List

class Settings:
    """Application settings loaded from environment variables"""
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./modelship.db")
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # File Upload
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    EXPORT_DIR: str = os.getenv("EXPORT_DIR", "exports")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_IMAGE_EXTENSIONS", 
        "jpg,jpeg,png,gif,webp"
    ).split(",")
    
    ALLOWED_TEXT_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_TEXT_EXTENSIONS", 
        "txt,csv,json"
    ).split(",")
    
    # ML Models
    HUGGINGFACE_CACHE_DIR: str = os.getenv("HUGGINGFACE_CACHE_DIR", "./models_cache")
    IMAGE_MODEL_NAME: str = os.getenv("IMAGE_MODEL_NAME", "microsoft/resnet-50")
    
    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:8080"
    ).split(",")
    
    # Create required directories
    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.EXPORT_DIR, exist_ok=True)
        os.makedirs(self.HUGGINGFACE_CACHE_DIR, exist_ok=True)

# Global settings instance
settings = Settings() 