#!/usr/bin/env python3
"""
Simple File-Based Storage System for ModelShip
No database required - stores everything in JSON files
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

class SimpleStorage:
    """Simple file-based storage for projects and data"""
    
    def __init__(self, storage_root: str = "storage"):
        self.storage_root = Path(storage_root)
        self.projects_dir = self.storage_root / "projects"
        self.uploads_dir = self.storage_root / "uploads" 
        self.results_dir = self.storage_root / "results"
        
        # Create directories
        for dir_path in [self.projects_dir, self.uploads_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Counter files
        self.project_counter_file = self.projects_dir / "next_id.json"
        self.init_counters()
    
    def init_counters(self):
        """Initialize ID counters"""
        if not self.project_counter_file.exists():
            self.save_json(self.project_counter_file, {"next_id": 1})
    
    def get_next_project_id(self) -> int:
        """Get next available project ID"""
        counter_data = self.load_json(self.project_counter_file)
        project_id = counter_data["next_id"]
        counter_data["next_id"] = project_id + 1
        self.save_json(self.project_counter_file, counter_data)
        return project_id
    
    def save_json(self, file_path: Path, data: Dict[str, Any]):
        """Save data to JSON file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    
    def load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file"""
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def create_project(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new project"""
        project_id = self.get_next_project_id()
        
        project = {
            "project_id": project_id,
            "name": project_data.get("name", f"Project {project_id}"),
            "description": project_data.get("description", ""),
            "project_type": project_data.get("project_type", "image_classification"),
            "confidence_threshold": project_data.get("confidence_threshold", 0.8),
            "auto_approve_threshold": project_data.get("auto_approve_threshold", 0.95),
            "guidelines": project_data.get("guidelines", ""),
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_items": 0,
            "labeled_items": 0,
            "reviewed_items": 0,
            "approved_items": 0,
            "files": [],
            "results": [],
            "default_labels": self.get_default_labels(project_data.get("project_type", "image_classification"))
        }
        
        # Save project file
        project_file = self.projects_dir / f"project_{project_id}.json"
        self.save_json(project_file, project)
        
        # Create project upload directory
        project_upload_dir = self.uploads_dir / f"project_{project_id}"
        project_upload_dir.mkdir(exist_ok=True)
        
        return project
    
    def get_project(self, project_id: int) -> Optional[Dict[str, Any]]:
        """Get project by ID"""
        project_file = self.projects_dir / f"project_{project_id}.json"
        if project_file.exists():
            return self.load_json(project_file)
        return None
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        projects = []
        for project_file in self.projects_dir.glob("project_*.json"):
            project_data = self.load_json(project_file)
            if project_data:  # Only include non-empty projects
                projects.append(project_data)
        
        # Sort by creation date (newest first)
        projects.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return projects
    
    def update_project(self, project_id: int, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update project"""
        project = self.get_project(project_id)
        if not project:
            return None
        
        # Update fields
        project.update(updates)
        project["updated_at"] = datetime.now().isoformat()
        
        # Save updated project
        project_file = self.projects_dir / f"project_{project_id}.json"
        self.save_json(project_file, project)
        
        return project
    
    def delete_project(self, project_id: int) -> bool:
        """Delete project and associated files"""
        project_file = self.projects_dir / f"project_{project_id}.json"
        if project_file.exists():
            project_file.unlink()
            
            # Clean up upload directory
            project_upload_dir = self.uploads_dir / f"project_{project_id}"
            if project_upload_dir.exists():
                import shutil
                shutil.rmtree(project_upload_dir)
            
            return True
        return False
    
    def get_default_labels(self, project_type: str) -> List[str]:
        """Get default labels for project type"""
        defaults = {
            "image_classification": ["positive", "negative"],
            "object_detection": ["person", "car", "bike", "animal", "object"],
            "text_classification": ["positive", "negative", "neutral"],
            "mixed_dataset": ["category_a", "category_b", "category_c"]
        }
        return defaults.get(project_type, ["label_1", "label_2", "label_3"])
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        projects = self.list_projects()
        
        total_projects = len(projects)
        active_projects = len([p for p in projects if p.get("status") == "active"])
        total_files = sum(len(p.get("files", [])) for p in projects)
        
        # Calculate storage size
        total_size = 0
        for dir_path in [self.projects_dir, self.uploads_dir, self.results_dir]:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return {
            "total_projects": total_projects,
            "active_projects": active_projects,
            "total_files": total_files,
            "storage_size_bytes": total_size,
            "storage_size_mb": round(total_size / (1024 * 1024), 2),
            "storage_directories": {
                "projects": str(self.projects_dir),
                "uploads": str(self.uploads_dir),
                "results": str(self.results_dir)
            }
        }

# Global storage instance
storage = SimpleStorage()

if __name__ == "__main__":
    # Test the storage system
    print("🧪 Testing Simple Storage System...")
    
    # Create test project
    test_project_data = {
        "name": "Test Image Project",
        "description": "Testing the storage system",
        "project_type": "image_classification"
    }
    
    project = storage.create_project(test_project_data)
    print(f"✅ Created project: {project['name']} (ID: {project['project_id']})")
    
    # List projects
    projects = storage.list_projects()
    print(f"📂 Total projects: {len(projects)}")
    
    # Get storage stats
    stats = storage.get_storage_stats()
    print(f"📊 Storage stats: {stats}")
    
    print("✅ Simple storage system working perfectly!") 