from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from database import get_db
from models import User, File as FileModel
from auth import get_current_user, get_optional_user
import os
import shutil
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import mimetypes
import logging
from datetime import datetime
import json
from PIL import Image
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "txt", "csv"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

logger = logging.getLogger(__name__)

class FileHandler:
    def __init__(self, base_upload_dir: str = "uploads"):
        self.base_upload_dir = Path(base_upload_dir)
        self.base_upload_dir.mkdir(exist_ok=True)
        
        # Supported file types
        self.supported_image_types = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        self.supported_text_types = {'.txt', '.csv', '.json', '.xml', '.md', '.rtf'}
        self.supported_document_types = {'.pdf', '.docx', '.doc', '.odt'}
        
        # File size limits (in bytes)
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_batch_size = 100  # Maximum files per batch
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get comprehensive file information"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        file_size = stat.st_size
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Determine file category
        file_ext = file_path.suffix.lower()
        if file_ext in self.supported_image_types:
            category = "image"
        elif file_ext in self.supported_text_types:
            category = "text"
        elif file_ext in self.supported_document_types:
            category = "document"
        else:
            category = "other"
        
        # Calculate file hash for duplicate detection
        file_hash = self._calculate_file_hash(file_path)
        
        info = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": file_size,
            "file_size_human": self._format_file_size(file_size),
            "mime_type": mime_type,
            "file_extension": file_ext,
            "category": category,
            "file_hash": file_hash,
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_supported": file_ext in (self.supported_image_types | self.supported_text_types | self.supported_document_types)
        }
        
        # Add image-specific info
        if category == "image":
            try:
                with Image.open(file_path) as img:
                    info.update({
                        "image_width": img.width,
                        "image_height": img.height,
                        "image_mode": img.mode,
                        "image_format": img.format
                    })
            except Exception as e:
                logger.warning(f"Could not read image metadata for {file_path}: {e}")
        
        return info

    async def upload_files_batch(
        self, 
        files: List[Any], 
        project_id: int,
        user_id: int,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Upload multiple files with progress tracking"""
        
        if len(files) > self.max_batch_size:
            raise ValueError(f"Batch size exceeds maximum of {self.max_batch_size} files")
        
        # Create project upload directory
        project_dir = self.base_upload_dir / f"project_{project_id}"
        project_dir.mkdir(exist_ok=True)
        
        # Initialize results
        results = {
            "successful_uploads": [],
            "failed_uploads": [],
            "duplicate_files": [],
            "unsupported_files": [],
            "total_files": len(files),
            "total_size": 0,
            "processing_time": 0
        }
        
        start_time = datetime.now()
        
        # Process files concurrently
        tasks = []
        for i, file in enumerate(files):
            task = self._process_single_file(
                file, project_dir, project_id, user_id, i, len(files)
            )
            tasks.append(task)
        
        # Execute tasks and collect results
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                results["failed_uploads"].append({
                    "filename": getattr(files[i], 'filename', f'file_{i}'),
                    "error": str(result),
                    "index": i
                })
            else:
                # Categorize result
                if result["status"] == "success":
                    results["successful_uploads"].append(result)
                    results["total_size"] += result["file_size"]
                elif result["status"] == "duplicate":
                    results["duplicate_files"].append(result)
                elif result["status"] == "unsupported":
                    results["unsupported_files"].append(result)
                else:
                    results["failed_uploads"].append(result)
            
            # Call progress callback
            if progress_callback:
                progress = (i + 1) / len(files)
                progress_callback(progress, f"Processed {i + 1}/{len(files)} files")
        
        results["processing_time"] = (datetime.now() - start_time).total_seconds()
        
        # Generate upload summary
        results["summary"] = {
            "success_rate": len(results["successful_uploads"]) / len(files) * 100,
            "total_size_human": self._format_file_size(results["total_size"]),
            "duplicates_found": len(results["duplicate_files"]),
            "unsupported_found": len(results["unsupported_files"]),
            "errors_found": len(results["failed_uploads"])
        }
        
        return results

    async def _process_single_file(
        self, 
        file: Any, 
        project_dir: Path, 
        project_id: int, 
        user_id: int,
        file_index: int,
        total_files: int
    ) -> Dict[str, Any]:
        """Process a single file upload"""
        
        try:
            # Validate file
            if not hasattr(file, 'filename') or not hasattr(file, 'read'):
                raise ValueError("Invalid file object")
            
            filename = file.filename
            if not filename:
                raise ValueError("Empty filename")
            
            # Check file size
            file_content = await self._read_file_content(file)
            file_size = len(file_content)
            
            if file_size > self.max_file_size:
                raise ValueError(f"File size {self._format_file_size(file_size)} exceeds maximum of {self._format_file_size(self.max_file_size)}")
            
            # Check file type
            file_ext = Path(filename).suffix.lower()
            is_supported = file_ext in (self.supported_image_types | self.supported_text_types | self.supported_document_types)
            
            if not is_supported:
                return {
                    "status": "unsupported",
                    "filename": filename,
                    "file_extension": file_ext,
                    "message": f"Unsupported file type: {file_ext}",
                    "index": file_index
                }
            
            # Generate unique filename to avoid conflicts
            file_hash = hashlib.md5(file_content).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{file_hash}_{filename}"
            
            # Check for duplicates
            existing_files = list(project_dir.glob(f"*_{file_hash}_*"))
            if existing_files:
                return {
                    "status": "duplicate",
                    "filename": filename,
                    "existing_file": existing_files[0].name,
                    "file_hash": file_hash,
                    "message": "File already exists (duplicate detected)",
                    "index": file_index
                }
            
            # Save file
            file_path = project_dir / unique_filename
            await self._save_file_content(file_content, file_path)
            
            # Get file info
            file_info = self.get_file_info(file_path)
            
            # Create upload record
            upload_record = {
                "status": "success",
                "filename": filename,
                "saved_filename": unique_filename,
                "file_path": str(file_path),
                "file_size": file_size,
                "file_size_human": self._format_file_size(file_size),
                "file_hash": file_hash,
                "file_type": file_info["category"],
                "mime_type": file_info["mime_type"],
                "project_id": project_id,
                "user_id": user_id,
                "uploaded_at": datetime.now().isoformat(),
                "index": file_index
            }
            
            # Add image-specific metadata
            if file_info["category"] == "image":
                upload_record.update({
                    "image_width": file_info.get("image_width"),
                    "image_height": file_info.get("image_height"),
                    "image_format": file_info.get("image_format")
                })
            
            logger.info(f"Successfully uploaded file {file_index + 1}/{total_files}: {filename}")
            return upload_record
            
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")
            return {
                "status": "error",
                "filename": getattr(file, 'filename', f'file_{file_index}'),
                "error": str(e),
                "index": file_index
            }

    async def _read_file_content(self, file: Any) -> bytes:
        """Read file content asynchronously"""
        if hasattr(file, 'read'):
            # Reset file pointer if possible
            if hasattr(file, 'seek'):
                file.seek(0)
            content = file.read()
            if isinstance(content, str):
                content = content.encode('utf-8')
            return content
        else:
            raise ValueError("File object does not support reading")

    async def _save_file_content(self, content: bytes, file_path: Path):
        """Save file content asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._write_file_sync,
            content,
            file_path
        )

    def _write_file_sync(self, content: bytes, file_path: Path):
        """Synchronous file writing"""
        with open(file_path, 'wb') as f:
            f.write(content)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"

    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file safely"""
        try:
            file_path = Path(file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False

    def move_file(self, source_path: Union[str, Path], dest_path: Union[str, Path]) -> bool:
        """Move file from source to destination"""
        try:
            source_path = Path(source_path)
            dest_path = Path(dest_path)
            
            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"Moved file from {source_path} to {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to move file from {source_path} to {dest_path}: {e}")
            return False

    def get_project_files(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all files for a project"""
        project_dir = self.base_upload_dir / f"project_{project_id}"
        
        if not project_dir.exists():
            return []
        
        files = []
        for file_path in project_dir.iterdir():
            if file_path.is_file():
                try:
                    file_info = self.get_file_info(file_path)
                    files.append(file_info)
                except Exception as e:
                    logger.error(f"Error getting info for file {file_path}: {e}")
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created_at'], reverse=True)
        return files

    def cleanup_old_files(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up files older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        deleted_files = []
        total_size_freed = 0
        
        for project_dir in self.base_upload_dir.iterdir():
            if project_dir.is_dir() and project_dir.name.startswith("project_"):
                for file_path in project_dir.iterdir():
                    if file_path.is_file():
                        if file_path.stat().st_mtime < cutoff_time:
                            try:
                                file_size = file_path.stat().st_size
                                file_path.unlink()
                                deleted_files.append(str(file_path))
                                total_size_freed += file_size
                            except Exception as e:
                                logger.error(f"Failed to delete old file {file_path}: {e}")
        
        return {
            "deleted_files": deleted_files,
            "files_deleted": len(deleted_files),
            "total_size_freed": total_size_freed,
            "total_size_freed_human": self._format_file_size(total_size_freed)
        }

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_files = 0
        total_size = 0
        projects_with_files = 0
        
        file_types = {}
        
        for project_dir in self.base_upload_dir.iterdir():
            if project_dir.is_dir() and project_dir.name.startswith("project_"):
                project_files = 0
                for file_path in project_dir.iterdir():
                    if file_path.is_file():
                        total_files += 1
                        project_files += 1
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        
                        # Track file types
                        file_ext = file_path.suffix.lower()
                        if file_ext not in file_types:
                            file_types[file_ext] = {"count": 0, "size": 0}
                        file_types[file_ext]["count"] += 1
                        file_types[file_ext]["size"] += file_size
                
                if project_files > 0:
                    projects_with_files += 1
        
        return {
            "total_files": total_files,
            "total_size": total_size,
            "total_size_human": self._format_file_size(total_size),
            "projects_with_files": projects_with_files,
            "file_types": file_types,
            "average_file_size": total_size / max(1, total_files),
            "average_file_size_human": self._format_file_size(total_size / max(1, total_files))
        }

# Create global instance
file_handler = FileHandler()

router = APIRouter(prefix="/api", tags=["files"])

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    # Validate file type
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Check file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    # Generate unique filename to avoid conflicts
    import uuid
    file_ext = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(contents)
    
    # Save file info to database (optional for unauthenticated users)
    db_file = None
    if current_user:
        db_file = FileModel(
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            file_size=len(contents),
            file_type=file_ext,
            status="uploaded"
        )
        
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
    
    return {
        "file_id": db_file.id if db_file else None,
        "filename": file.filename,
        "file_size": len(contents),
        "file_path": file_path,
        "message": "File uploaded successfully"
    }

@router.get("/files")
def get_user_files(
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    files = db.query(FileModel).filter(FileModel.user_id == current_user.id).all()
    return {
        "files": [
            {
                "id": f.id,
                "filename": f.filename,
                "file_size": f.file_size,
                "file_type": f.file_type,
                "status": f.status,
                "created_at": f.created_at
            }
            for f in files
        ]
    }

@router.delete("/files/{file_id}")
def delete_file(
    file_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    # Get file from database
    db_file = db.query(FileModel).filter(
        FileModel.id == file_id,
        FileModel.user_id == current_user.id
    ).first()
    
    if not db_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Delete physical file
    try:
        if os.path.exists(db_file.file_path):
            os.remove(db_file.file_path)
    except Exception as e:
        # Log error but continue with database deletion
        pass
    
    # Delete from database
    db.delete(db_file)
    db.commit()
    
    return {"message": "File deleted successfully"} 