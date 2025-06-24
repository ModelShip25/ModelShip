from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from file_handler import router as file_router
from classification import router as classify_router
from text_classification import router as text_classify_router
from export import router as export_router
# from billing import router as billing_router  # Temporarily disabled - focusing on core functionality
from review_system import router as review_router

# Production imports
try:
    from production_config import settings
    from logging_config import ProductionLogger, get_logger
    from middleware import setup_middleware
    from health_monitoring import router as health_router
    PRODUCTION_MODE = True
except ImportError:
    from config import settings
    PRODUCTION_MODE = False
    print("⚠️  Running in development mode - production modules not available")

try:
    from advanced_export import router as ml_export_router
except ImportError:
    ml_export_router = None
import uvicorn
import logging
import warnings
import os
from datetime import datetime
from label_schema_manager import label_schema_manager, LabelSchema
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from database import get_db
from models import User
# Auth temporarily removed for development
from project_management import router as project_router
from vertical_templates import router as vertical_router
from expert_in_loop import router as expert_router
from bias_fairness_reports import router as bias_router
from security_compliance import router as security_router
from ml_assisted_prelabeling import router as prelabeling_router
from consensus_controls import router as consensus_router

# ✅ ADD SIMPLE PROJECT API - NO DATABASE REQUIRED
from simple_project_api import router as simple_project_router

# Initialize production logging if available
if PRODUCTION_MODE:
    prod_logger = ProductionLogger(
        log_level=settings.LOG_LEVEL,
        log_format=settings.LOG_FORMAT,
        log_file=settings.LOG_FILE,
        enable_console=True
    )
    logger = get_logger(__name__)
    logger.info("🚀 Starting ModelShip in PRODUCTION mode")
else:
    # Configure logging to reduce verbosity (development mode)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("ml_service").setLevel(logging.WARNING)
    logging.getLogger("advanced_ml_service").setLevel(logging.WARNING)
    
    # Suppress specific transformers warnings
    warnings.filterwarnings("ignore", message=".*use_fast.*")
    warnings.filterwarnings("ignore", message=".*slow processor.*")
    logger = logging.getLogger(__name__)

# Create FastAPI app with metadata
app = FastAPI(
    title="ModelShip API",
    description="AI-powered auto-labeling platform for images and text - Production Ready" if PRODUCTION_MODE else "AI-powered auto-labeling platform for images and text",
    version="1.0.0",
    docs_url="/docs" if not (PRODUCTION_MODE and settings.is_production) else None,
    redoc_url="/redoc" if not (PRODUCTION_MODE and settings.is_production) else None,
    debug=not (PRODUCTION_MODE and settings.is_production)
)

# Initialize database tables
try:
    from database import create_tables
    create_tables()
    logger.info("✅ Database tables initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize database tables: {e}")

# Setup middleware
if PRODUCTION_MODE:
    app = setup_middleware(app, settings)
    logger.info("✅ Production middleware configured")
else:
    # Add CORS middleware for frontend development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include health monitoring (always first)
if PRODUCTION_MODE:
    app.include_router(health_router)

# Include all routers with API versioning
API_PREFIX = "/api/v1" if PRODUCTION_MODE else ""

app.include_router(simple_project_router, prefix=API_PREFIX)  # ✅ SIMPLE PROJECT API FIRST
app.include_router(file_router, prefix=API_PREFIX)
app.include_router(classify_router, prefix=API_PREFIX)
app.include_router(text_classify_router, prefix=API_PREFIX)
app.include_router(export_router, prefix=API_PREFIX)
# app.include_router(billing_router)  # Temporarily disabled - focusing on core functionality
app.include_router(review_router, prefix=API_PREFIX)
app.include_router(project_router, prefix=API_PREFIX)

# Include Phase 2 & 3 QA routers (with error handling)
try:
    from annotation_quality_dashboard import router as quality_router
    app.include_router(quality_router)
    logging.info("✅ Quality dashboard router loaded")
except ImportError as e:
    logging.warning(f"❌ Quality dashboard module not found: {e}")

try:
    from gold_standard_testing import router as gold_standard_router
    app.include_router(gold_standard_router)
    logging.info("✅ Gold standard router loaded")
except ImportError as e:
    logging.warning(f"❌ Gold standard module not found: {e}")

try:
    from expert_qa_system import router as expert_qa_router
    app.include_router(expert_qa_router)
    logging.info("✅ Expert QA router loaded")
except ImportError as e:
    logging.warning(f"❌ Expert QA module not found: {e}")

# Include advanced ML export router if available
if ml_export_router:
    app.include_router(ml_export_router)

# Note: Project management router already included above

# Object detection service integration
try:
    from object_detection_service import object_detection_service
    logging.info("✅ Object detection service loaded")
except ImportError as e:
    logging.warning(f"❌ Object detection service not found: {e}")

# Additional optional routers
try:
    from active_learning import router as active_learning_router
    app.include_router(active_learning_router)
    logging.info("✅ Active learning router loaded")
except ImportError as e:
    logging.warning(f"❌ Active learning module not found: {e}")

try:
    from analytics_dashboard import router as analytics_router
    app.include_router(analytics_router)
    logging.info("✅ Analytics dashboard router loaded")
except ImportError as e:
    logging.warning(f"❌ Analytics dashboard module not found: {e}")

# Import project file manager
from project_file_manager import project_file_manager

# Duplicate imports removed - routers already imported above

# Phase 2 routers (imported above with error handling)
try:
    from mlops_integration import router as mlops_router
    app.include_router(mlops_router)
    logging.info("✅ MLOps integration router loaded")
except ImportError as e:
    logging.warning(f"❌ MLOps integration module not found: {e}")

try:
    from data_versioning import router as versioning_router
    app.include_router(versioning_router)
    logging.info("✅ Data versioning router loaded")
except ImportError as e:
    logging.warning(f"❌ Data versioning module not found: {e}")

# Include Phase 3 routers (with error handling)
try:
    from vertical_templates import router as vertical_router
    app.include_router(vertical_router)
    logging.info("✅ Vertical templates router loaded")
except ImportError as e:
    logging.warning(f"❌ Vertical templates module not found: {e}")

try:
    from expert_in_loop import router as expert_router
    app.include_router(expert_router)
    logging.info("✅ Expert in loop router loaded")
except ImportError as e:
    logging.warning(f"❌ Expert in loop module not found: {e}")

try:
    from bias_fairness_reports import router as bias_router
    app.include_router(bias_router)
    logging.info("✅ Bias fairness router loaded")
except ImportError as e:
    logging.warning(f"❌ Bias fairness module not found: {e}")

try:
    from security_compliance import router as security_router
    app.include_router(security_router)
    logging.info("✅ Security compliance router loaded")
except ImportError as e:
    logging.warning(f"❌ Security compliance module not found: {e}")

# Include new feature routers
try:
    from ml_assisted_prelabeling import router as prelabeling_router
    app.include_router(prelabeling_router)
    logging.info("✅ ML assisted prelabeling router loaded")
except ImportError as e:
    logging.warning(f"❌ ML assisted prelabeling module not found: {e}")

try:
    from consensus_controls import router as consensus_router
    app.include_router(consensus_router)
    logging.info("✅ Consensus controls router loaded")
except ImportError as e:
    logging.warning(f"❌ Consensus controls module not found: {e}")

# Global exception handler for production
if PRODUCTION_MODE:
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

@app.get("/")
def read_root():
    """API root endpoint with system information"""
    response = {
        "message": "ModelShip API is running!",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if PRODUCTION_MODE:
        response.update({
            "environment": settings.ENVIRONMENT,
            "features": {
                "object_detection": settings.ENABLE_OBJECT_DETECTION,
                "text_classification": settings.ENABLE_TEXT_CLASSIFICATION,
                "active_learning": settings.ENABLE_ACTIVE_LEARNING,
                "expert_review": settings.ENABLE_EXPERT_REVIEW,
                "advanced_exports": settings.ENABLE_ADVANCED_EXPORTS
            },
            "documentation": "/docs" if not settings.is_production else "disabled",
            "health_check": "/health"
        })
    
    return response
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy",
        "features": {
            "phase_1_complete": True,
            "auto_labeling": True,
            "object_detection": True,
            "image_classification": True,
            "text_classification": True,
            "batch_processing": True,
            "project_management": True,
            "human_review": True,
            "advanced_export": True,
            "coco_yolo_export": True,
            "team_management": True,
            "confidence_thresholds": True,
            "billing_system": False,  # Phase 2 feature
            "analytics": True
        },
        "phase_1_workflows": {
            "project_setup": "/api/projects/create",
            "auto_label_loop": "/api/classify/image/detect",
            "human_review": "/api/review/tasks/pending",
            "export": "/api/export/project/{project_id}"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "service": "ModelShip API",
        "version": "1.0.0"
    }

# Project file serving endpoint
@app.get("/api/projects/{project_id}/files/{file_type}/{filename}")
async def serve_project_file(project_id: int, file_type: str, filename: str):
    """Serve files from project storage (originals or annotated)"""
    
    if file_type not in ["originals", "annotated"]:
        raise HTTPException(status_code=400, detail="File type must be 'originals' or 'annotated'")
    
    file_path = project_file_manager.get_absolute_file_path(project_id, file_type, filename)
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"}  # Cache for 1 day
    )

# ========================
# LABEL SCHEMA MANAGEMENT
# ========================

@app.get("/api/schemas", tags=["schemas"])
async def list_schemas(
    project_id: Optional[int] = Query(None),
    include_public: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List available label schemas"""
    
    try:
        schemas = label_schema_manager.list_schemas(
            project_id=project_id, 
            include_public=include_public
        )
        
        return {
            "schemas": schemas,
            "total_count": len(schemas),
            "project_id": project_id,
            "include_public": include_public
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list schemas: {str(e)}")

@app.get("/api/schemas/{schema_id}", tags=["schemas"])
async def get_schema(
    schema_id: str,
    db: Session = Depends(get_db)
):
    """Get a specific label schema by ID"""
    
    try:
        schema = label_schema_manager.get_schema(schema_id)
        
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found")
        
        return {
            "schema": schema.dict(),
            "validation": label_schema_manager.validate_schema(schema)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")

@app.post("/api/schemas", tags=["schemas"])
async def create_schema(
    schema_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create a new label schema"""
    
    try:
        # Add default creator information for development
        schema_data["created_by"] = "dev_user"
        
        schema = label_schema_manager.create_schema(schema_data)
        
        return {
            "message": "Schema created successfully",
            "schema": schema.dict(),
            "validation": label_schema_manager.validate_schema(schema)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create schema: {str(e)}")

@app.put("/api/schemas/{schema_id}", tags=["schemas"])
async def update_schema(
    schema_id: str,
    updates: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update an existing label schema"""
    
    try:
        schema = label_schema_manager.update_schema(schema_id, updates)
        
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found")
        
        return {
            "message": "Schema updated successfully",
            "schema": schema.dict(),
            "validation": label_schema_manager.validate_schema(schema)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update schema: {str(e)}")

@app.delete("/api/schemas/{schema_id}", tags=["schemas"])
async def delete_schema(
    schema_id: str,
    db: Session = Depends(get_db)
):
    """Delete a label schema"""
    
    try:
        success = label_schema_manager.delete_schema(schema_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Schema not found or could not be deleted")
        
        return {"message": "Schema deleted successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete schema: {str(e)}")

@app.post("/api/schemas/{schema_id}/validate", tags=["schemas"])
async def validate_schema(
    schema_id: str,
    db: Session = Depends(get_db)
):
    """Validate a label schema"""
    
    try:
        schema = label_schema_manager.get_schema(schema_id)
        
        if not schema:
            raise HTTPException(status_code=404, detail="Schema not found")
        
        validation_result = label_schema_manager.validate_schema(schema)
        
        return {
            "schema_id": schema_id,
            "validation": validation_result,
            "schema_info": {
                "name": schema.name,
                "version": schema.version,
                "categories_count": len(schema.categories),
                "label_type": schema.label_type
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate schema: {str(e)}")

@app.get("/api/schemas/{schema_id}/export", tags=["schemas"])
async def export_schema(
    schema_id: str,
    format: str = Query("json", description="Export format (json, coco, yolo)"),
    db: Session = Depends(get_db)
):
    """Export a label schema in specified format"""
    
    try:
        exported_data = label_schema_manager.export_schema(schema_id, format)
        
        if not exported_data:
            raise HTTPException(status_code=404, detail="Schema not found")
        
        return {
            "schema_id": schema_id,
            "format": format,
            "data": exported_data
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export schema: {str(e)}")

@app.get("/api/schemas/templates/built-in", tags=["schemas"])
async def get_built_in_schemas(
    db: Session = Depends(get_db)
):
    """Get all built-in schema templates"""
    
    try:
        built_in_schemas = []
        
        for schema_id, schema in label_schema_manager.built_in_schemas.items():
            built_in_schemas.append({
                "id": schema.id,
                "name": schema.name,
                "description": schema.description,
                "label_type": schema.label_type,
                "categories_count": len(schema.categories),
                "auto_approval_threshold": schema.auto_approval_threshold,
                "categories": [
                    {
                        "id": cat.id,
                        "name": cat.name,
                        "description": cat.description,
                        "color": cat.color
                    }
                    for cat in schema.categories
                ]
            })
        
        return {
            "built_in_schemas": built_in_schemas,
            "total_count": len(built_in_schemas)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get built-in schemas: {str(e)}")

# End of schema management endpoints

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for frontend connection testing"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "phase": "Phase 1A",
        "features": [
            "text_classification",
            "ner_classification", 
            "image_classification",
            "object_detection",
            "auto_approval_workflow",
            "label_schema_management"
        ]
    }

# Duplicate router inclusions removed - all routers loaded above with error handling

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)