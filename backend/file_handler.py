from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, status
from sqlalchemy.orm import Session
from database import get_db
from models import User, File as FileModel
from auth import get_current_user
import os
import shutil
from typing import List

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "txt", "csv"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

router = APIRouter(prefix="/api", tags=["files"])

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    
    # Save file info to database
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
        "file_id": db_file.id,
        "filename": file.filename,
        "file_size": len(contents),
        "message": "File uploaded successfully"
    }

@router.get("/files")
def get_user_files(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
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
    current_user: User = Depends(get_current_user)
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