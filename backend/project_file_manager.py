"""
Project File Manager - Organizes files by project
Creates folder structure: project_storage/project_{id}/originals/ and /annotated/
"""

import os
import shutil
from typing import Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProjectFileManager:
    def __init__(self, base_storage_path: str = "project_storage"):
        self.base_path = base_storage_path
        self.ensure_base_directory()
    
    def ensure_base_directory(self):
        """Ensure the base storage directory exists"""
        os.makedirs(self.base_path, exist_ok=True)
    
    def get_project_path(self, project_id: int) -> str:
        """Get the main project directory path"""
        return os.path.join(self.base_path, f"project_{project_id}")
    
    def get_originals_path(self, project_id: int) -> str:
        """Get the originals directory path for a project"""
        return os.path.join(self.get_project_path(project_id), "originals")
    
    def get_annotated_path(self, project_id: int) -> str:
        """Get the annotated directory path for a project"""
        return os.path.join(self.get_project_path(project_id), "annotated")
    
    def ensure_project_directories(self, project_id: int):
        """Create project directories if they don't exist"""
        try:
            project_path = self.get_project_path(project_id)
            originals_path = self.get_originals_path(project_id)
            annotated_path = self.get_annotated_path(project_id)
            
            os.makedirs(project_path, exist_ok=True)
            os.makedirs(originals_path, exist_ok=True)
            os.makedirs(annotated_path, exist_ok=True)
            
            logger.info(f"Created directories for project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directories for project {project_id}: {e}")
            return False
    
    def save_original_file(self, project_id: int, file_content: bytes, filename: str) -> Optional[str]:
        """Save an original uploaded file to the project's originals folder"""
        try:
            self.ensure_project_directories(project_id)
            originals_path = self.get_originals_path(project_id)
            
            # Create unique filename with timestamp if needed
            base_name, ext = os.path.splitext(filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{base_name}_{timestamp}{ext}"
            
            file_path = os.path.join(originals_path, unique_filename)
            
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved original file: {unique_filename} for project {project_id}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save original file for project {project_id}: {e}")
            return None
    
    def save_annotated_file(self, project_id: int, source_image_path: str, annotated_filename: str) -> Optional[str]:
        """Save an annotated image to the project's annotated folder"""
        try:
            self.ensure_project_directories(project_id)
            annotated_path = self.get_annotated_path(project_id)
            
            # Create timestamped filename
            base_name, ext = os.path.splitext(annotated_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_filename = f"{base_name}_annotated_{timestamp}{ext}"
            
            destination_path = os.path.join(annotated_path, final_filename)
            
            # If source is from temp location, move it
            if os.path.exists(source_image_path):
                shutil.copy2(source_image_path, destination_path)
                logger.info(f"Saved annotated file: {final_filename} for project {project_id}")
                return destination_path
            else:
                logger.error(f"Source annotated image not found: {source_image_path}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save annotated file for project {project_id}: {e}")
            return None
    
    def get_project_files(self, project_id: int) -> dict:
        """Get all files for a project"""
        try:
            originals_path = self.get_originals_path(project_id)
            annotated_path = self.get_annotated_path(project_id)
            
            originals = []
            annotated = []
            
            if os.path.exists(originals_path):
                originals = [f for f in os.listdir(originals_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            
            if os.path.exists(annotated_path):
                annotated = [f for f in os.listdir(annotated_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            
            return {
                "originals": originals,
                "annotated": annotated,
                "total_originals": len(originals),
                "total_annotated": len(annotated)
            }
            
        except Exception as e:
            logger.error(f"Failed to get files for project {project_id}: {e}")
            return {"originals": [], "annotated": [], "total_originals": 0, "total_annotated": 0}
    
    def get_file_url(self, project_id: int, file_type: str, filename: str) -> str:
        """Generate URL for accessing a project file"""
        return f"/api/projects/{project_id}/files/{file_type}/{filename}"
    
    def get_absolute_file_path(self, project_id: int, file_type: str, filename: str) -> Optional[str]:
        """Get absolute file path for a project file"""
        try:
            if file_type == "originals":
                base_path = self.get_originals_path(project_id)
            elif file_type == "annotated":
                base_path = self.get_annotated_path(project_id)
            else:
                return None
            
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                return file_path
            return None
            
        except Exception as e:
            logger.error(f"Failed to get file path for {file_type}/{filename} in project {project_id}: {e}")
            return None
    
    def delete_project_files(self, project_id: int) -> bool:
        """Delete all files for a project"""
        try:
            project_path = self.get_project_path(project_id)
            if os.path.exists(project_path):
                shutil.rmtree(project_path)
                logger.info(f"Deleted all files for project {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete files for project {project_id}: {e}")
            return False

# Global instance
project_file_manager = ProjectFileManager() 