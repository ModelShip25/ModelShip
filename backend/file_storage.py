"""
Simple file-based storage system for ModelShip testing
Replaces database with JSON files in storage folders
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FileStorage:
    """Simple file-based storage system"""
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = base_path
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create storage directories if they don't exist"""
        directories = [
            "projects",
            "users", 
            "results",
            "uploads",
            "uploads/annotated"
        ]
        
        for directory in directories:
            path = os.path.join(self.base_path, directory)
            os.makedirs(path, exist_ok=True)
    
    def save_json(self, category: str, filename: str, data: Dict) -> bool:
        """Save data as JSON file"""
        try:
            filepath = os.path.join(self.base_path, category, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save {category}/{filename}: {e}")
            return False
    
    def load_json(self, category: str, filename: str) -> Optional[Dict]:
        """Load data from JSON file"""
        try:
            filepath = os.path.join(self.base_path, category, f"{filename}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to load {category}/{filename}: {e}")
            return None
    
    def list_files(self, category: str) -> List[str]:
        """List all files in a category"""
        try:
            path = os.path.join(self.base_path, category)
            if os.path.exists(path):
                return [f.replace('.json', '') for f in os.listdir(path) if f.endswith('.json')]
            return []
        except Exception as e:
            logger.error(f"Failed to list {category}: {e}")
            return []
    
    def delete_file(self, category: str, filename: str) -> bool:
        """Delete a file"""
        try:
            filepath = os.path.join(self.base_path, category, f"{filename}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {category}/{filename}: {e}")
            return False

class ProjectStorage:
    """Project-specific storage operations"""
    
    def __init__(self, storage: FileStorage):
        self.storage = storage
        self.next_id_file = "next_project_id"
    
    def get_next_id(self) -> int:
        """Get next available project ID"""
        next_id_data = self.storage.load_json("projects", self.next_id_file)
        if next_id_data:
            next_id = next_id_data.get("next_id", 1)
        else:
            next_id = 1
        
        # Save incremented ID
        self.storage.save_json("projects", self.next_id_file, {"next_id": next_id + 1})
        return next_id
    
    def create_project(self, project_data: Dict, user_id: int = 1) -> Dict:
        """Create a new project"""
        project_id = self.get_next_id()
        
        project = {
            "project_id": project_id,
            "name": project_data["name"],
            "description": project_data.get("description", ""),
            "project_type": project_data["project_type"],
            "confidence_threshold": project_data.get("confidence_threshold", 0.8),
            "auto_approve_threshold": project_data.get("auto_approve_threshold", 0.95),
            "status": "active",
            "owner_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "total_items": 0,
            "labeled_items": 0,
            "reviewed_items": 0,
            "statistics": {
                "total_items": 0,
                "labeled_items": 0,
                "reviewed_items": 0
            }
        }
        
        if self.storage.save_json("projects", str(project_id), project):
            return project
        else:
            raise Exception("Failed to save project")
    
    def get_project(self, project_id: int) -> Optional[Dict]:
        """Get project by ID"""
        return self.storage.load_json("projects", str(project_id))
    
    def list_projects(self, user_id: int = None) -> List[Dict]:
        """List all projects (optionally filtered by user)"""
        project_ids = self.storage.list_files("projects")
        projects = []
        
        for project_id in project_ids:
            if project_id == self.next_id_file:
                continue
                
            project = self.storage.load_json("projects", project_id)
            if project:
                if user_id is None or project.get("owner_id") == user_id:
                    projects.append(project)
        
        return sorted(projects, key=lambda p: p.get("created_at", ""), reverse=True)
    
    def update_project(self, project_id: int, updates: Dict) -> bool:
        """Update project data"""
        project = self.get_project(project_id)
        if project:
            project.update(updates)
            project["updated_at"] = datetime.utcnow().isoformat()
            return self.storage.save_json("projects", str(project_id), project)
        return False

class UserStorage:
    """User-specific storage operations"""
    
    def __init__(self, storage: FileStorage):
        self.storage = storage
        self.test_user_id = 1
    
    def get_or_create_test_user(self) -> Dict:
        """Get or create the test user"""
        user = self.storage.load_json("users", str(self.test_user_id))
        
        if not user:
            user = {
                "user_id": self.test_user_id,
                "email": "test@modelship.ai",
                "role": "admin",
                "subscription_tier": "unlimited",
                "credits_remaining": 999999,
                "created_at": datetime.utcnow().isoformat()
            }
            self.storage.save_json("users", str(self.test_user_id), user)
        
        return user

class ReviewStorage:
    """Review task storage operations"""
    
    def __init__(self, storage: FileStorage):
        self.storage = storage
    
    def get_pending_tasks(self, project_id: int) -> List[Dict]:
        """Get mock review tasks for a project"""
        # Generate mock review tasks
        mock_tasks = [
            {
                "result_id": f"task_{project_id}_1",
                "project_id": project_id,
                "filename": "sample_image_1.jpg",
                "predicted_label": "person",
                "confidence": 85.2,
                "status": "pending_review",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "result_id": f"task_{project_id}_2", 
                "project_id": project_id,
                "filename": "sample_image_2.jpg",
                "predicted_label": "car",
                "confidence": 78.9,
                "status": "pending_review",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "result_id": f"task_{project_id}_3",
                "project_id": project_id, 
                "filename": "sample_image_3.jpg",
                "predicted_label": "dog",
                "confidence": 92.1,
                "status": "pending_review",
                "created_at": datetime.utcnow().isoformat()
            }
        ]
        
        return mock_tasks
    
    def submit_review(self, result_id: str, action: str, notes: str = "") -> Dict:
        """Submit a review for a task"""
        review_result = {
            "result_id": result_id,
            "action": action,
            "notes": notes,
            "reviewed_at": datetime.utcnow().isoformat(),
            "reviewer_id": 1,
            "status": "completed"
        }
        
        # Save review result
        self.storage.save_json("results", f"review_{result_id}", review_result)
        
        return {
            "message": f"Review {action}d successfully",
            "result_id": result_id,
            "action": action,
            "timestamp": review_result["reviewed_at"]
        }

# Global storage instances
file_storage = FileStorage()
project_storage = ProjectStorage(file_storage)
user_storage = UserStorage(file_storage)
review_storage = ReviewStorage(file_storage) 