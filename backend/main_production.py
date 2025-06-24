"""
ModelShip Production API Server
Production-ready FastAPI application with comprehensive logging, monitoring, and security
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

# Production imports
from production_config import settings
from logging_config import ProductionLogger, get_logger
from middleware import setup_middleware
from health_monitoring import router as health_router

# Core application imports
from file_handler import router as file_router
from classification import router as classify_router
from text_classification import router as text_classify_router
from export import router as export_router
from review_system import router as review_router
from project_management import router as project_router
from simple_project_api import router as simple_project_router

# Database
from database import get_db, create_tables

# Schema management
from label_schema_manager import label_schema_manager, LabelSchema

# Advanced feature imports (with error handling)
OPTIONAL_ROUTERS = []

try:
    from advanced_export import router as ml_export_router
    OPTIONAL_ROUTERS.append(("advanced_export", ml_export_router))
except ImportError as e:
    logging.warning(f"Advanced export not available: {e}")

try:
    from annotation_quality_dashboard import router as quality_router
    OPTIONAL_ROUTERS.append(("quality_dashboard", quality_router))
except ImportError as e:
    logging.warning(f"Quality dashboard not available: {e}")

try:
    from gold_standard_testing import router as gold_standard_router
    OPTIONAL_ROUTERS.append(("gold_standard", gold_standard_router))
except ImportError as e:
    logging.warning(f"Gold standard testing not available: {e}")

try:
    from expert_qa_system import router as expert_qa_router
    OPTIONAL_ROUTERS.append(("expert_qa", expert_qa_router))
except ImportError as e:
    logging.warning(f"Expert QA system not available: {e}")

try:
    from active_learning import router as active_learning_router
    OPTIONAL_ROUTERS.append(("active_learning", active_learning_router))
except ImportError as e:
    logging.warning(f"Active learning not available: {e}")

try:
    from analytics_dashboard import router as analytics_router
    OPTIONAL_ROUTERS.append(("analytics", analytics_router))
except ImportError as e:
    logging.warning(f"Analytics dashboard not available: {e}")

try:
    from mlops_integration import router as mlops_router
    OPTIONAL_ROUTERS.append(("mlops", mlops_router))
except ImportError as e:
    logging.warning(f"MLOps integration not available: {e}")

try:
    from data_versioning import router as versioning_router
    OPTIONAL_ROUTERS.append(("data_versioning", versioning_router))
except ImportError as e:
    logging.warning(f"Data versioning not available: {e}")

try:
    from vertical_templates import router as vertical_router
    OPTIONAL_ROUTERS.append(("vertical_templates", vertical_router))
except ImportError as e:
    logging.warning(f"Vertical templates not available: {e}")

try:
    from expert_in_loop import router as expert_router
    OPTIONAL_ROUTERS.append(("expert_in_loop", expert_router))
except ImportError as e:
    logging.warning(f"Expert in loop not available: {e}")

try:
    from bias_fairness_reports import router as bias_router
    OPTIONAL_ROUTERS.append(("bias_fairness", bias_router))
except ImportError as e:
    logging.warning(f"Bias fairness reports not available: {e}")

try:
    from security_compliance import router as security_router
    OPTIONAL_ROUTERS.append(("security_compliance", security_router))
except ImportError as e:
    logging.warning(f"Security compliance not available: {e}")

try:
    from ml_assisted_prelabeling import router as prelabeling_router
    OPTIONAL_ROUTERS.append(("ml_prelabeling", prelabeling_router))
except ImportError as e:
    logging.warning(f"ML assisted prelabeling not available: {e}")

try:
    from consensus_controls import router as consensus_router
    OPTIONAL_ROUTERS.append(("consensus_controls", consensus_router))
except ImportError as e:
    logging.warning(f"Consensus controls not available: {e}")

# Initialize production logging
prod_logger = ProductionLogger(
    log_level=settings.LOG_LEVEL,
    log_format=settings.LOG_FORMAT,
    log_file=settings.LOG_FILE,
    enable_console=True
)

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting ModelShip API server...")
    
    try:
        # Initialize database
        logger.info("📊 Initializing database...")
        create_tables()
        logger.info("✅ Database initialized successfully")
        
        # Initialize ML models (optional)
        try:
            from object_detection_service import object_detection_service
            logger.info("🧠 Object detection service loaded")
        except ImportError as e:
            logger.warning(f"⚠️ Object detection service not available: {e}")
        
        # Log startup configuration
        logger.info(f"🏃 Environment: {settings.ENVIRONMENT}")
        logger.info(f"🔐 Debug mode: {settings.DEBUG}")
        logger.info(f"📡 API server starting on {settings.API_HOST}:{settings.API_PORT}")
        logger.info(f"📁 Upload directory: {settings.UPLOAD_DIR}")
        logger.info(f"📤 Export directory: {settings.EXPORT_DIR}")
        logger.info(f"🧠 ML models cache: {settings.HUGGINGFACE_CACHE_DIR}")
        
        # Log enabled features
        enabled_features = []
        if settings.ENABLE_OBJECT_DETECTION:
            enabled_features.append("Object Detection")
        if settings.ENABLE_TEXT_CLASSIFICATION:
            enabled_features.append("Text Classification")
        if settings.ENABLE_ACTIVE_LEARNING:
            enabled_features.append("Active Learning")
        if settings.ENABLE_EXPERT_REVIEW:
            enabled_features.append("Expert Review")
        if settings.ENABLE_ADVANCED_EXPORTS:
            enabled_features.append("Advanced Exports")
        
        logger.info(f"🔧 Enabled features: {', '.join(enabled_features)}")
        logger.info(f"📦 Optional modules loaded: {len(OPTIONAL_ROUTERS)}")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("🛑 Shutting down ModelShip API server...")

# Create FastAPI application
app = FastAPI(
    title="ModelShip API",
    description="AI-powered auto-labeling platform for images and text - Production Ready",
    version="1.0.0",
    docs_url="/docs" if not settings.is_production else None,  # Disable docs in production
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
    debug=settings.DEBUG
)

# Setup middleware
app = setup_middleware(app, settings)

# Include health monitoring (always first)
app.include_router(health_router)

# Include core routers
app.include_router(simple_project_router, prefix="/api/v1")
app.include_router(file_router, prefix="/api/v1")
app.include_router(classify_router, prefix="/api/v1")
app.include_router(text_classify_router, prefix="/api/v1")
app.include_router(export_router, prefix="/api/v1")
app.include_router(review_router, prefix="/api/v1")
app.include_router(project_router, prefix="/api/v1")

# Include optional routers
for router_name, router in OPTIONAL_ROUTERS:
    try:
        app.include_router(router, prefix="/api/v1")
        logger.info(f"✅ Loaded {router_name} router")
    except Exception as e:
        logger.error(f"❌ Failed to load {router_name} router: {e}")

# Import project file manager
try:
    from project_file_manager import project_file_manager
    logger.info("✅ Project file manager loaded")
except ImportError as e:
    logger.warning(f"⚠️ Project file manager not available: {e}")

# Root endpoint
@app.get("/")
def read_root():
    """API root endpoint with system information"""
    return {
        "message": "ModelShip API is running!",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "object_detection": settings.ENABLE_OBJECT_DETECTION,
            "text_classification": settings.ENABLE_TEXT_CLASSIFICATION,
            "active_learning": settings.ENABLE_ACTIVE_LEARNING,
            "expert_review": settings.ENABLE_EXPERT_REVIEW,
            "advanced_exports": settings.ENABLE_ADVANCED_EXPORTS
        },
        "documentation": "/docs" if not settings.is_production else "disabled",
        "health_check": "/health"
    }

# File serving for project files
@app.get("/api/v1/projects/{project_id}/files/{file_type}/{filename}")
async def serve_project_file(project_id: int, file_type: str, filename: str):
    """Serve project files with proper security checks"""
    try:
        # Validate file type
        allowed_types = ["uploads", "exports", "results"]
        if file_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Construct file path
        file_path = f"storage/projects/{project_id}/{file_type}/{filename}"
        
        # Security check - ensure path doesn't escape project directory
        import os.path
        normalized_path = os.path.normpath(file_path)
        if not normalized_path.startswith(f"storage/projects/{project_id}/"):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve file")

# Schema management endpoints
@app.get("/api/v1/schemas", tags=["schemas"])
async def list_schemas(
    project_id: Optional[int] = Query(None),
    include_public: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List available schemas"""
    try:
        schemas = label_schema_manager.list_schemas(
            project_id=project_id,
            include_public=include_public
        )
        return {"schemas": schemas}
    except Exception as e:
        logger.error(f"Error listing schemas: {e}")
        raise HTTPException(status_code=500, detail="Failed to list schemas")

@app.get("/api/v1/schemas/{schema_id}", tags=["schemas"])
async def get_schema(schema_id: str, db: Session = Depends(get_db)):
    """Get specific schema"""
    try:
        schema = label_schema_manager.get_schema(schema_id)
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found")
        return schema
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting schema: {e}")
        raise HTTPException(status_code=500, detail="Failed to get schema")

@app.post("/api/v1/schemas", tags=["schemas"])
async def create_schema(schema_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create new schema"""
    try:
        schema_id = label_schema_manager.create_schema(schema_data)
        return {"schema_id": schema_id, "message": "Schema created successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating schema: {e}")
        raise HTTPException(status_code=500, detail="Failed to create schema")

@app.put("/api/v1/schemas/{schema_id}", tags=["schemas"])
async def update_schema(
    schema_id: str, 
    updates: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """Update existing schema"""
    try:
        success = label_schema_manager.update_schema(schema_id, updates)
        if not success:
            raise HTTPException(status_code=404, detail="Schema not found")
        return {"message": "Schema updated successfully"}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating schema: {e}")
        raise HTTPException(status_code=500, detail="Failed to update schema")

@app.delete("/api/v1/schemas/{schema_id}", tags=["schemas"])
async def delete_schema(schema_id: str, db: Session = Depends(get_db)):
    """Delete schema"""
    try:
        success = label_schema_manager.delete_schema(schema_id)
        if not success:
            raise HTTPException(status_code=404, detail="Schema not found")
        return {"message": "Schema deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting schema: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete schema")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}",
        exc_info=True,
        extra={
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Production-specific endpoints
if settings.is_production:
    
    @app.get("/api/v1/status", tags=["monitoring"])
    async def production_status():
        """Production status endpoint"""
        return {
            "status": "production",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT
        }

if __name__ == "__main__":
    import uvicorn
    
    # Production server configuration
    uvicorn_config = {
        "app": "main_production:app" if settings.is_production else app,
        "host": settings.API_HOST,
        "port": settings.API_PORT,
        "log_level": settings.LOG_LEVEL.lower(),
        "access_log": False,  # We handle our own access logging
        "server_header": False,  # Don't reveal server info
        "date_header": True,
    }
    
    if settings.is_production:
        uvicorn_config.update({
            "workers": settings.WORKERS,
            "loop": "uvloop",  # Better performance
            "http": "h11",     # HTTP/1.1 implementation
            "interface": "asgi3"
        })
    else:
        uvicorn_config.update({
            "reload": True,
            "reload_dirs": ["backend"]
        })
    
    logger.info("🚀 Starting ModelShip API server...")
    uvicorn.run(**uvicorn_config) 