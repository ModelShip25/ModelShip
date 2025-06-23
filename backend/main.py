from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from auth import router as auth_router
from file_handler import router as file_router
from classification import router as classify_router
from text_classification import router as text_classify_router
from export import router as export_router
# from billing import router as billing_router  # Temporarily disabled - focusing on core functionality
from review_system import router as review_router
try:
    from advanced_export import router as ml_export_router
except ImportError:
    ml_export_router = None
import uvicorn
import logging
import warnings
import os
from label_schema_manager import label_schema_manager, LabelSchema
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
from project_management import router as project_router

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ml_service").setLevel(logging.WARNING)
logging.getLogger("advanced_ml_service").setLevel(logging.WARNING)

# Suppress specific transformers warnings
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow processor.*")

# Create FastAPI app with metadata
app = FastAPI(
    title="ModelShip API",
    description="AI-powered auto-labeling platform for images and text",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize database tables
try:
    from database import create_tables
    create_tables()
    logging.info("✅ Database tables initialized successfully")
except Exception as e:
    logging.error(f"❌ Failed to initialize database tables: {e}")

# Add CORS middleware for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # Add frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(auth_router)
app.include_router(file_router)
app.include_router(classify_router)
app.include_router(text_classify_router)
app.include_router(export_router)
# app.include_router(billing_router)  # Temporarily disabled - focusing on core functionality
app.include_router(review_router)
app.include_router(project_router)

# Include advanced ML export router if available
if ml_export_router:
    app.include_router(ml_export_router)

# Add Phase 1 MVP routers
try:
    from project_management import router as project_router
    app.include_router(project_router)
    logging.info("✅ Project management router loaded")
except ImportError as e:
    logging.warning(f"❌ Project management module not found: {e}")

try:
    from advanced_export import router as advanced_export_router
    app.include_router(advanced_export_router)
    logging.info("✅ Advanced export router loaded")
except ImportError as e:
    logging.warning(f"❌ Advanced export module not found: {e}")

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

# Import routers
from auth import router as auth_router
from classification import router as classification_router
from project_management import router as project_router
from review_system import router as review_router
from export import router as export_router
from file_handler import router as file_router

# Phase 2 routers
from annotation_quality_dashboard import router as quality_router
from mlops_integration import router as mlops_router
from data_versioning import router as versioning_router
from gold_standard_testing import router as gold_standard_router

@app.get("/")
def read_root():
    return {
        "message": "ModelShip API is running!",
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new label schema"""
    
    try:
        # Add creator information
        schema_data["created_by"] = str(current_user.id)
        
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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
    current_user: User = Depends(get_current_user),
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

# Include routers
app.include_router(auth_router)
app.include_router(classification_router)
app.include_router(project_router)
app.include_router(review_router)
app.include_router(export_router)
app.include_router(file_router)

# Phase 2 routers
app.include_router(quality_router)
app.include_router(mlops_router)
app.include_router(versioning_router)
app.include_router(gold_standard_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)