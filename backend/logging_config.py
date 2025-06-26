import logging
import logging.handlers
import json
import sys
from contextlib import contextmanager
from os import getenv
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
        if hasattr(record, 'project_id'):
            log_entry["project_id"] = record.project_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'endpoint'):
            log_entry["endpoint"] = record.endpoint
        if hasattr(record, 'method'):
            log_entry["method"] = record.method
        if hasattr(record, 'processing_time'):
            log_entry["processing_time"] = record.processing_time
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    """Colored formatter for development console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted

class ProductionLogger:
    """Production-ready logging configuration"""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_format: str = "text",
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 environment: str = getenv("ENV", "development")):
        self.log_level = getattr(logging, log_level.upper())
        self.log_format = log_format
        self.log_file = log_file
        self.enable_console = enable_console

        # Adjust log levels based on environment
        self.log_level = logging.DEBUG if environment == "development" else self.log_level

        # Setup logging
    
    def setup_logging(self):
        """Configure logging with proper handlers and formatters"""
        
        # Create logs directory
        log_dir = Path("storage/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers = []
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            
            if self.log_format == "json":
                console_handler.setFormatter(JSONFormatter())
            else:
                console_formatter = ColoredFormatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                console_handler.setFormatter(console_formatter)
            
            root_logger.addHandler(console_handler)
        
        # File handler with rotation
        if self.log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.log_file,
                maxBytes=10_000_000,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(self.log_level)
            
            if self.log_format == "json":
                file_handler.setFormatter(JSONFormatter())
            else:
                file_formatter = logging.Formatter(
                    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
            
            root_logger.addHandler(file_handler)
        
        # Application-specific handlers
        self.setup_app_loggers()
        
        # Third-party library log levels
        self.configure_third_party_loggers()
    
    def setup_app_loggers(self):
        """Setup application-specific loggers"""
        
        # API access logger
        access_logger = logging.getLogger("api.access")
        access_handler = logging.handlers.RotatingFileHandler(
            filename="storage/logs/access.log",
            maxBytes=50_000_000,  # 50MB
            backupCount=10
        )
        access_handler.setFormatter(JSONFormatter())
        access_logger.addHandler(access_handler)
        access_logger.setLevel(logging.INFO)
        
        # ML processing logger
        ml_logger = logging.getLogger("ml.processing")
        ml_handler = logging.handlers.RotatingFileHandler(
            filename="storage/logs/ml_processing.log",
            maxBytes=20_000_000,  # 20MB
            backupCount=5
        )
        ml_handler.setFormatter(JSONFormatter())
        ml_logger.addHandler(ml_handler)
        ml_logger.setLevel(logging.INFO)
        
        # Auth logger
        auth_logger = logging.getLogger("auth")
        auth_handler = logging.handlers.RotatingFileHandler(
            filename="storage/logs/auth.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        auth_handler.setFormatter(JSONFormatter())
        auth_logger.addHandler(auth_handler)
        auth_logger.setLevel(logging.INFO)

        # Review logger
        review_logger = logging.getLogger("review")
        review_handler = logging.handlers.RotatingFileHandler(
            filename="storage/logs/review.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=5
        )
        review_handler.setFormatter(JSONFormatter())
        review_logger.addHandler(review_handler)
        review_logger.setLevel(logging.INFO)

        # Error logger
        error_handler = logging.handlers.RotatingFileHandler(
            filename="storage/logs/errors.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=10
        )
        error_handler.setFormatter(JSONFormatter())
        error_logger.addHandler(error_handler)
        error_logger.setLevel(logging.ERROR)
    
    # Summary logging
    @staticmethod
    def log_summary(logger_name: str, message: str, summary: Dict[str, Any]):
        logger = logging.getLogger(logger_name)
        logger.info(
            f"{message}",
            extra=summary
        )

    def configure_third_party_loggers(self):
        """Configure third-party library loggers to reduce noise"""
        
        # Suppress verbose transformers logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        
        # Suppress torch warnings
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        # Suppress PIL warnings
        logging.getLogger("PIL").setLevel(logging.WARNING)
        
        # Suppress HTTP client logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        # Suppress SQLAlchemy info logs
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
        
        # Suppress uvicorn access logs (we handle our own)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)

def log_api_access(method: str, endpoint: str, status_code: int, 
                  processing_time: float, user_id: Optional[str] = None,
                  request_id: Optional[str] = None):
    """Log API access with structured data"""
    logger = logging.getLogger("api.access")
    logger.info(
        f"{method} {endpoint} - {status_code}",
        extra={
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "processing_time": processing_time,
            "user_id": user_id,
            "request_id": request_id
        }
    )

def log_ml_processing(operation: str, duration: float, success: bool,
                     model_name: str, input_size: int, 
                     project_id: Optional[str] = None,
                     error_message: Optional[str] = None):
    """Log ML processing operations"""
    logger = logging.getLogger("ml.processing")
    level = logging.INFO if success else logging.ERROR
    
    logger.log(
        level,
        f"ML {operation} - {'Success' if success else 'Failed'}",
        extra={
            "operation": operation,
            "duration": duration,
            "success": success,
            "model_name": model_name,
            "input_size": input_size,
            "project_id": project_id,
            "error_message": error_message
        }
    )

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Log errors with context"""
    logger = logging.getLogger("errors")
    logger.error(
        f"Error: {str(error)}",
        exc_info=True,
        extra=context or {}
    )

# Context manager for request logging
class RequestLoggingContext:
    def __init__(self, method: str, endpoint: str, request_id: str):
        self.method = method
        self.endpoint = endpoint
        self.request_id = request_id
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            status_code = 500 if exc_type else 200
            log_api_access(
                self.method, 
                self.endpoint, 
                status_code, 
                duration, 
                request_id=self.request_id
            ) 