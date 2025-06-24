import os
from typing import List, Optional
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"

class ProductionSettings:
    """Production-ready application settings"""
    
    # Environment
    ENVIRONMENT: Environment = Environment(os.getenv("ENVIRONMENT", "development"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./modelship.db")
    DATABASE_POOL_SIZE: int = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "0"))
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "CHANGE-THIS-IN-PRODUCTION")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    PASSWORD_MIN_LENGTH: int = int(os.getenv("PASSWORD_MIN_LENGTH", "8"))
    
    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "100"))
    
    # File Upload Limits
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "storage/uploads")
    EXPORT_DIR: str = os.getenv("EXPORT_DIR", "storage/exports")
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "52428800"))  # 50MB
    MAX_FILES_PER_BATCH: int = int(os.getenv("MAX_FILES_PER_BATCH", "100"))
    MAX_TOTAL_UPLOAD_SIZE: int = int(os.getenv("MAX_TOTAL_UPLOAD_SIZE", "524288000"))  # 500MB
    
    # Processing Limits
    MAX_CONCURRENT_CLASSIFICATIONS: int = int(os.getenv("MAX_CONCURRENT_CLASSIFICATIONS", "10"))
    CLASSIFICATION_TIMEOUT_SECONDS: int = int(os.getenv("CLASSIFICATION_TIMEOUT_SECONDS", "300"))  # 5 minutes
    BATCH_PROCESSING_CHUNK_SIZE: int = int(os.getenv("BATCH_PROCESSING_CHUNK_SIZE", "50"))
    
    # File Types
    ALLOWED_IMAGE_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_IMAGE_EXTENSIONS", 
        "jpg,jpeg,png,gif,webp,bmp,tiff"
    ).split(",")
    
    ALLOWED_TEXT_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_TEXT_EXTENSIONS", 
        "txt,csv,json,tsv"
    ).split(",")
    
    ALLOWED_DOCUMENT_EXTENSIONS: List[str] = os.getenv(
        "ALLOWED_DOCUMENT_EXTENSIONS",
        "pdf,doc,docx"
    ).split(",")
    
    # ML Models
    HUGGINGFACE_CACHE_DIR: str = os.getenv("HUGGINGFACE_CACHE_DIR", "./models_cache")
    IMAGE_MODEL_NAME: str = os.getenv("IMAGE_MODEL_NAME", "microsoft/resnet-50")
    TEXT_MODEL_NAME: str = os.getenv("TEXT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    OBJECT_DETECTION_MODEL: str = os.getenv("OBJECT_DETECTION_MODEL", "yolov8n.pt")
    MODEL_CACHE_TTL_HOURS: int = int(os.getenv("MODEL_CACHE_TTL_HOURS", "24"))
    
    # API Server
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "4"))
    
    # CORS
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS", 
        "http://localhost:3000,http://localhost:8080,https://modelship.com"
    ).split(",")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO" if ENVIRONMENT == Environment.PRODUCTION else "DEBUG")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json" if ENVIRONMENT == Environment.PRODUCTION else "text")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)
    
    # Monitoring & Health
    HEALTH_CHECK_INTERVAL: int = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN", None)
    
    # Redis (for caching and task queue)
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL", None)
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    # Background Tasks
    TASK_QUEUE_ENABLED: bool = os.getenv("TASK_QUEUE_ENABLED", "true").lower() == "true"
    MAX_BACKGROUND_TASKS: int = int(os.getenv("MAX_BACKGROUND_TASKS", "50"))
    
    # Storage
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")  # local, s3, gcs
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID", None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY", None)
    AWS_S3_BUCKET: Optional[str] = os.getenv("AWS_S3_BUCKET", None)
    AWS_REGION: Optional[str] = os.getenv("AWS_REGION", "us-east-1")
    
    # Email (for notifications)
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST", None)
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER", None)
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD", None)
    FROM_EMAIL: Optional[str] = os.getenv("FROM_EMAIL", None)
    
    # Feature Flags
    ENABLE_OBJECT_DETECTION: bool = os.getenv("ENABLE_OBJECT_DETECTION", "true").lower() == "true"
    ENABLE_TEXT_CLASSIFICATION: bool = os.getenv("ENABLE_TEXT_CLASSIFICATION", "true").lower() == "true"
    ENABLE_ACTIVE_LEARNING: bool = os.getenv("ENABLE_ACTIVE_LEARNING", "true").lower() == "true"
    ENABLE_EXPERT_REVIEW: bool = os.getenv("ENABLE_EXPERT_REVIEW", "true").lower() == "true"
    ENABLE_ADVANCED_EXPORTS: bool = os.getenv("ENABLE_ADVANCED_EXPORTS", "true").lower() == "true"
    
    # Performance
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    ENABLE_COMPRESSION: bool = os.getenv("ENABLE_COMPRESSION", "true").lower() == "true"
    
    def __init__(self):
        """Initialize and validate settings"""
        self._create_directories()
        self._validate_settings()
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.UPLOAD_DIR,
            self.EXPORT_DIR,
            self.HUGGINGFACE_CACHE_DIR,
            "storage/projects",
            "storage/schemas",
            "storage/results",
            "storage/logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _validate_settings(self):
        """Validate critical production settings"""
        if self.ENVIRONMENT == Environment.PRODUCTION:
            if self.SECRET_KEY == "CHANGE-THIS-IN-PRODUCTION":
                raise ValueError("SECRET_KEY must be changed in production!")
            
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production!")
                
            if len(self.SECRET_KEY) < 32:
                raise ValueError("SECRET_KEY must be at least 32 characters long!")
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == Environment.DEVELOPMENT
    
    def get_database_url(self) -> str:
        """Get database URL with connection pooling for production"""
        if self.is_production and "sqlite" in self.DATABASE_URL:
            # In production, recommend PostgreSQL
            print("⚠️  WARNING: Using SQLite in production. Consider PostgreSQL for better performance.")
        return self.DATABASE_URL

# Global settings instance
settings = ProductionSettings() 