#!/usr/bin/env python3
"""
ModelShip Production Startup Script
Handles production server startup with proper configuration and error handling
"""

import os
import sys
import logging
import signal
import asyncio
from pathlib import Path
from typing import Optional

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """Setup production environment and validate configuration"""
    
    # Ensure we're in production mode
    os.environ.setdefault('ENVIRONMENT', 'production')
    
    # Load environment variables
    from production_config import settings
    
    # Validate critical settings
    if settings.SECRET_KEY == "CHANGE-THIS-IN-PRODUCTION":
        print("❌ ERROR: SECRET_KEY must be changed in production!")
        print("   Set SECRET_KEY environment variable or update .env file")
        sys.exit(1)
    
    if settings.DEBUG:
        print("❌ ERROR: DEBUG must be False in production!")
        print("   Set DEBUG=false in environment variables or .env file")
        sys.exit(1)
    
    # Create required directories
    required_dirs = [
        settings.UPLOAD_DIR,
        settings.EXPORT_DIR,
        settings.HUGGINGFACE_CACHE_DIR,
        "storage/logs",
        "storage/projects",
        "storage/schemas",
        "storage/results"
    ]
    
    for directory in required_dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return settings

def check_dependencies():
    """Check if all required dependencies are available"""
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'transformers',
        'torch',
        'pillow',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ ERROR: Missing required packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r production_requirements.txt")
        sys.exit(1)
    
    print("✅ All required packages are available")

def check_database_connection(settings):
    """Check database connectivity"""
    try:
        from database import create_tables
        create_tables()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("   Please check your DATABASE_URL configuration")
        return False

def check_ml_models(settings):
    """Check if ML models are available"""
    model_issues = []
    
    # Check YOLO models
    yolo_models = ["yolov8n.pt", "yolov8s.pt"]
    for model in yolo_models:
        if not os.path.exists(model):
            model_issues.append(f"YOLO model not found: {model}")
    
    # Check Hugging Face cache directory
    if not os.path.exists(settings.HUGGINGFACE_CACHE_DIR):
        model_issues.append(f"Hugging Face cache directory not found: {settings.HUGGINGFACE_CACHE_DIR}")
    
    if model_issues:
        print("⚠️  WARNING: Some ML models are missing:")
        for issue in model_issues:
            print(f"   - {issue}")
        print("   Models will be downloaded on first use")
    else:
        print("✅ ML models are available")

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    
    def signal_handler(signum, frame):
        print(f"\n📡 Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

def run_health_checks(settings):
    """Run comprehensive health checks before starting"""
    
    print("🔍 Running production health checks...")
    
    # Check dependencies
    check_dependencies()
    
    # Check database
    if not check_database_connection(settings):
        print("❌ CRITICAL: Database connection failed!")
        print("   Server will not start without database connectivity")
        sys.exit(1)
    
    # Check ML models
    check_ml_models(settings)
    
    # Check disk space
    import psutil
    disk = psutil.disk_usage('/')
    if disk.percent > 90:
        print(f"⚠️  WARNING: Disk usage is {disk.percent}% - consider freeing space")
    
    # Check memory
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        print(f"⚠️  WARNING: Memory usage is {memory.percent}% - monitor performance")
    
    print("✅ Health checks completed")

def start_production_server(settings):
    """Start the production server with optimized configuration"""
    
    import uvicorn
    
    # Production uvicorn configuration
    config = {
        "app": "main_production:app",
        "host": settings.API_HOST,
        "port": settings.API_PORT,
        "workers": settings.WORKERS,
        "loop": "uvloop",
        "http": "h11",
        "interface": "asgi3",
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": False,  # We handle our own access logging
        "server_header": False,
        "date_header": True,
        "forwarded_allow_ips": "*",
        "proxy_headers": True,
    }
    
    # SSL configuration if certificates are provided
    if hasattr(settings, 'SSL_CERT_PATH') and hasattr(settings, 'SSL_KEY_PATH'):
        if os.path.exists(settings.SSL_CERT_PATH) and os.path.exists(settings.SSL_KEY_PATH):
            config.update({
                "ssl_certfile": settings.SSL_CERT_PATH,
                "ssl_keyfile": settings.SSL_KEY_PATH
            })
            print(f"🔐 SSL enabled with certificates")
    
    print("🚀 Starting ModelShip Production Server...")
    print(f"📡 Server: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"🏃 Environment: {settings.ENVIRONMENT}")
    print(f"👥 Workers: {settings.WORKERS}")
    print(f"📊 Database: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else 'SQLite'}")
    print(f"🧠 ML Models: {settings.HUGGINGFACE_CACHE_DIR}")
    print(f"📁 Storage: {settings.UPLOAD_DIR}")
    print("─" * 50)
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

def main():
    """Main production startup function"""
    
    print("=" * 50)
    print("🚀 ModelShip Production Server Startup")
    print("=" * 50)
    
    try:
        # Setup signal handlers
        setup_signal_handlers()
        
        # Setup environment
        settings = setup_environment()
        
        # Run health checks
        run_health_checks(settings)
        
        # Start server
        start_production_server(settings)
        
    except KeyboardInterrupt:
        print("\n🛑 Startup interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ CRITICAL ERROR during startup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 