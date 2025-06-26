"""
Simple file-based storage system for ModelShip testing
Replaces database with JSON files in storage folders
"""

import json
import os
import uuid
import shutil
import hashlib
import mimetypes
import zipfile
import io
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, BinaryIO, Tuple
import logging
import magic  # python-magic for mime type detection

logger = logging.getLogger(__name__)

class FileStorage:
    """Enhanced file-based storage system"""
    
    def __init__(self, base_path: str = "storage"):
        self.base_path = base_path
        self.chunk_size = 10 * 1024 * 1024  # 10MB chunk size for large files
        self.ensure_directories()
        self.audit_log_enabled = True
        
        # Allowed MIME types
        self.allowed_image_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/tiff'
        ]
        self.allowed_document_types = [
            'text/plain', 'text/csv', 'application/json', 'application/pdf',
            'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ]
        
        # Audit logs directory
        self.audit_log_path = os.path.join(self.base_path, "audit_logs")
    
    def ensure_directories(self):
        """Create storage directories if they don't exist"""
        directories = [
            "projects",
            "users", 
            "results",
            "uploads",
            "uploads/annotated",
            "uploads/chunks",  # For large file uploads
            "backups",         # For backups
            "audit_logs",      # For audit logs
            "archives"         # For archived projects
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
            
    def validate_file_type(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """Validate file type using content inspection"""
        try:
            # Use python-magic for content-based MIME detection
            mime_type = magic.from_buffer(file_content, mime=True)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Check against allowed types
            if mime_type in self.allowed_image_types:
                valid_type = "image"
            elif mime_type in self.allowed_document_types:
                valid_type = "document"
            else:
                return False, f"Unsupported file type: {mime_type}"
                
            # Validate extension against actual content
            expected_extensions = {
                'image/jpeg': ['.jpg', '.jpeg'],
                'image/png': ['.png'],
                'image/gif': ['.gif'],
                'image/webp': ['.webp'],
                'text/plain': ['.txt'],
                'text/csv': ['.csv'],
                'application/json': ['.json'],
                'application/pdf': ['.pdf']
            }
            
            if mime_type in expected_extensions and file_ext not in expected_extensions.get(mime_type, []):
                return False, f"File extension {file_ext} doesn't match content type {mime_type}"
                
            return True, valid_type
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def save_file_in_chunks(self, file_obj: BinaryIO, chunk_path: str, chunk_size: int = None) -> str:
        """Save large file in chunks"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        # Create a unique ID for this chunked upload
        upload_id = str(uuid.uuid4())
        upload_dir = os.path.join(self.base_path, "uploads/chunks", upload_id)
        os.makedirs(upload_dir, exist_ok=True)
        
        chunk_index = 0
        file_hash = hashlib.md5()
        
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
                
            # Update hash
            file_hash.update(chunk)
            
            # Save chunk
            chunk_file = os.path.join(upload_dir, f"chunk_{chunk_index}")
            with open(chunk_file, 'wb') as f:
                f.write(chunk)
                
            chunk_index += 1
            
        # Save metadata
        metadata = {
            "upload_id": upload_id,
            "chunks": chunk_index,
            "total_size": file_obj.tell(),
            "hash": file_hash.hexdigest(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(os.path.join(upload_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
            
        return upload_id
    
    def assemble_chunks(self, upload_id: str, target_path: str) -> Optional[str]:
        """Assemble chunks into final file"""
        upload_dir = os.path.join(self.base_path, "uploads/chunks", upload_id)
        metadata_path = os.path.join(upload_dir, "metadata.json")
        
        if not os.path.exists(metadata_path):
            logger.error(f"Metadata not found for upload {upload_id}")
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Create target directory if needed
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Assemble chunks
        with open(target_path, 'wb') as target_file:
            for i in range(metadata["chunks"]):
                chunk_path = os.path.join(upload_dir, f"chunk_{i}")
                if not os.path.exists(chunk_path):
                    logger.error(f"Chunk {i} missing for upload {upload_id}")
                    return None
                    
                with open(chunk_path, 'rb') as chunk_file:
                    target_file.write(chunk_file.read())
                    
        # Validate assembled file
        file_hash = hashlib.md5()
        with open(target_path, 'rb') as f:
            while True:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                file_hash.update(chunk)
                
        if file_hash.hexdigest() != metadata["hash"]:
            logger.error(f"Hash mismatch for assembled file {upload_id}")
            os.remove(target_path)
            return None
            
        # Clean up chunks
        shutil.rmtree(upload_dir)
        
        return target_path
    
    def create_backup(self, categories: List[str]) -> Optional[str]:
        """Create a backup of specified data categories"""
        try:
            backup_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.base_path, "backups", f"backup_{backup_id}.zip")
            
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as backup_zip:
                for category in categories:
                    category_path = os.path.join(self.base_path, category)
                    if os.path.exists(category_path):
                        for root, _, files in os.walk(category_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.base_path)
                                backup_zip.write(file_path, arcname)
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def restore_backup(self, backup_path: str, target_path: Optional[str] = None) -> bool:
        """Restore from a backup"""
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False
                
            if target_path is None:
                target_path = self.base_path
                
            with zipfile.ZipFile(backup_path, 'r') as backup_zip:
                backup_zip.extractall(target_path)
                
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    def log_audit_event(self, event_type: str, user_id: str, details: Dict[str, Any]) -> bool:
        """Log an audit event"""
        if not self.audit_log_enabled:
            return True
            
        try:
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": event_type,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details
            }
            
            # Create daily audit log file
            log_date = datetime.utcnow().strftime("%Y-%m-%d")
            log_file = os.path.join(self.audit_log_path, f"audit_{log_date}.jsonl")
            
            os.makedirs(self.audit_log_path, exist_ok=True)
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
                
            return True
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
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
            # Handle status transitions
            if "status" in updates and updates["status"] != project.get("status"):
                self._handle_status_transition(project_id, project["status"], updates["status"])
                
            project.update(updates)
            project["updated_at"] = datetime.utcnow().isoformat()
            return self.storage.save_json("projects", str(project_id), project)
        return False
        
    def _handle_status_transition(self, project_id: int, old_status: str, new_status: str) -> None:
        """Handle project status transitions"""
        # Log status change
        self.storage.log_audit_event(
            "project_status_change",
            "system",
            {
                "project_id": project_id,
                "old_status": old_status,
                "new_status": new_status
            }
        )
        
        # Handle specific transitions
        if new_status == "archived":
            self._archive_project(project_id)
        elif old_status == "archived" and new_status != "archived":
            self._unarchive_project(project_id)
            
    def _archive_project(self, project_id: int) -> None:
        """Archive project data and files"""
        project = self.get_project(project_id)
        if not project:
            return
            
        # Create archive timestamp
        archive_time = datetime.utcnow()
        archive_id = f"project_{project_id}_{archive_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Copy project data
        archive_data = project.copy()
        archive_data["archived_at"] = archive_time.isoformat()
        archive_data["archive_id"] = archive_id
        
        # Save archived data
        self.storage.save_json("archives", archive_id, archive_data)
        
        # Create backup of project files if needed
        # This could be implemented to save files to cold storage
        
    def _unarchive_project(self, project_id: int) -> None:
        """Restore project from archive if needed"""
        # For now, just log the unarchive event
        # Actual file restoration would be implemented here
        
    def export_project(self, project_id: int, export_format: str = "json") -> Optional[str]:
        """Export project data and configs"""
        project = self.get_project(project_id)
        if not project:
            return None
            
        try:
            export_time = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            export_filename = f"project_{project_id}_export_{export_time}"
            export_dir = os.path.join(self.storage.base_path, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            if export_format == "json":
                # Export as JSON
                export_path = os.path.join(export_dir, f"{export_filename}.json")
                with open(export_path, 'w') as f:
                    json.dump(project, f, indent=2, default=str)
                return export_path
                
            elif export_format == "zip":
                # Export as ZIP with data and files
                export_path = os.path.join(export_dir, f"{export_filename}.zip")
                
                with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as export_zip:
                    # Add project data
                    project_json = json.dumps(project, indent=2, default=str)
                    export_zip.writestr("project.json", project_json)
                    
                    # TODO: Add project files to the ZIP
                    
                return export_path
                
            else:
                logger.error(f"Unsupported export format: {export_format}")
                return None
                
        except Exception as e:
            logger.error(f"Project export failed: {e}")
            return None
            
    def duplicate_project(self, project_id: int, new_name: Optional[str] = None) -> Optional[int]:
        """Create a duplicate of a project"""
        project = self.get_project(project_id)
        if not project:
            return None
            
        try:
            # Create new project data
            new_project = project.copy()
            new_project_id = self.get_next_id()
            
            # Update project details
            if new_name:
                new_project["name"] = new_name
            else:
                new_project["name"] = f"{project['name']} (Copy)"
                
            new_project["project_id"] = new_project_id
            new_project["created_at"] = datetime.utcnow().isoformat()
            new_project["updated_at"] = datetime.utcnow().isoformat()
            new_project["total_items"] = 0
            new_project["labeled_items"] = 0
            new_project["reviewed_items"] = 0
            
            # Save new project
            self.storage.save_json("projects", str(new_project_id), new_project)
            
            # TODO: Copy project files if needed
            
            return new_project_id
            
        except Exception as e:
            logger.error(f"Project duplication failed: {e}")
            return None

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