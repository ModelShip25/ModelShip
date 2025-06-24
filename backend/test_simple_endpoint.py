#!/usr/bin/env python3
"""
Test a simple project creation endpoint without any authentication dependencies
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import json
import requests

# Create a simple test app
test_app = FastAPI()

class SimpleProject(BaseModel):
    name: str
    project_type: str
    description: Optional[str] = None

@test_app.post("/test/projects/")
async def create_simple_project(project: SimpleProject):
    """Create a project without any authentication"""
    return {
        "project_id": 999,
        "name": project.name,
        "project_type": project.project_type,
        "description": project.description,
        "message": "Test project created successfully"
    }

@test_app.get("/test/health")
async def test_health():
    """Health check endpoint"""
    return {"status": "ok", "message": "Test endpoint working"}

if __name__ == "__main__":
    import uvicorn
    print("Starting test server on http://localhost:8001")
    uvicorn.run(test_app, host="0.0.0.0", port=8001, log_level="info") 