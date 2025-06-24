#!/usr/bin/env python3
"""
Simple Project API using file storage - no database required
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import simple_storage
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_storage import storage

router = APIRouter(prefix="/api/simple-projects", tags=["simple_projects"])

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    project_type: str = "image_classification"
    confidence_threshold: float = 0.8
    auto_approve_threshold: float = 0.95
    guidelines: Optional[str] = None

class ProjectResponse(BaseModel):
    project_id: int
    name: str
    description: str
    project_type: str
    status: str
    created_at: str
    total_items: int
    labeled_items: int
    default_labels: List[str]

@router.post("/", response_model=Dict[str, Any])
async def create_project(project_data: ProjectCreate):
    """Create a new project using simple file storage"""
    
    try:
        # Convert Pydantic model to dict
        project_dict = project_data.dict()
        
        # Create project using simple storage
        project = storage.create_project(project_dict)
        
        return {
            "success": True,
            "message": "Project created successfully",
            "project": project
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@router.get("/", response_model=Dict[str, Any])
async def list_projects():
    """List all projects"""
    
    try:
        projects = storage.list_projects()
        
        return {
            "success": True,
            "projects": projects,
            "total_count": len(projects)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")

@router.get("/{project_id}", response_model=Dict[str, Any])
async def get_project(project_id: int):
    """Get project by ID"""
    
    try:
        project = storage.get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {
            "success": True,
            "project": project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")

@router.put("/{project_id}", response_model=Dict[str, Any])
async def update_project(project_id: int, updates: Dict[str, Any]):
    """Update project"""
    
    try:
        project = storage.update_project(project_id, updates)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {
            "success": True,
            "message": "Project updated successfully",
            "project": project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@router.delete("/{project_id}", response_model=Dict[str, Any])
async def delete_project(project_id: int):
    """Delete project"""
    
    try:
        success = storage.delete_project(project_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {
            "success": True,
            "message": "Project deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@router.get("/stats/storage", response_model=Dict[str, Any])
async def get_storage_stats():
    """Get storage statistics"""
    
    try:
        stats = storage.get_storage_stats()
        
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

@router.post("/test-create", response_model=Dict[str, Any])
async def create_test_project():
    """Create a test project for quick testing"""
    
    try:
        test_data = {
            "name": f"Test Project {storage.get_next_project_id()}",
            "description": "Auto-generated test project",
            "project_type": "image_classification"
        }
        
        project = storage.create_project(test_data)
        
        return {
            "success": True,
            "message": "Test project created successfully",
            "project": project
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create test project: {str(e)}")

@router.get("/types/supported")
async def get_supported_project_types():
    """Get supported project types and their details"""
    
    return {
        "success": True,
        "supported_types": {
            "image_classification": {
                "description": "Single label per image",
                "default_labels": ["positive", "negative"],
                "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"]
            },
            "object_detection": {
                "description": "Multiple objects with bounding boxes",
                "default_labels": ["person", "car", "bike", "animal", "object"],
                "supported_formats": ["jpg", "jpeg", "png", "gif", "webp"]
            },
            "text_classification": {
                "description": "Text document classification",
                "default_labels": ["positive", "negative", "neutral"],
                "supported_formats": ["txt", "csv", "json"]
            },
            "mixed_dataset": {
                "description": "Images and text combined",
                "default_labels": ["category_a", "category_b", "category_c"],
                "supported_formats": ["jpg", "jpeg", "png", "txt", "csv"]
            }
        }
    } 