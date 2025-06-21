from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result, File as FileModel
from auth import get_current_user, get_optional_user
from advanced_ml_service import AdvancedMLService, advanced_ml_service
from typing import List, Dict, Any, Optional
import asyncio
import time
import uuid
import os
from datetime import datetime

router = APIRouter(prefix="/api/classify", tags=["classification"])

class ClassificationService:
    """Service class handling classification business logic"""
    
    def __init__(self):
        self.advanced_ml_service = advanced_ml_service
    
    async def process_classification_job(self, job_id: int, files_data: List[Dict], job_type: str, db: Session):
        """Process classification job in background"""
        try:
            # Update job status to processing
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "processing"
            db.commit()
            
            results = []
            processed_count = 0
            
            # Extract file paths for batch processing
            image_paths = [file_data["file_path"] for file_data in files_data if job_type == "image"]
            
            if job_type == "image" and image_paths:
                # Use advanced batch processing
                def progress_callback(progress: float, message: str):
                    # Update job progress in database
                    job.completed_items = int(progress * len(files_data))
                    db.commit()
                
                batch_results = await self.advanced_ml_service.classify_image_batch(
                    image_paths=image_paths,
                    model_name="resnet50",
                    batch_size=8,
                    progress_callback=progress_callback
                )
                
                # Save results to database
                for i, result in enumerate(batch_results):
                    file_data = files_data[i]
                    
                    db_result = Result(
                        job_id=job_id,
                        file_id=file_data.get("file_id"),
                        filename=file_data["filename"],
                        predicted_label=result["predicted_label"],
                        confidence=result["confidence"],
                        processing_time=result["processing_time"],
                        status=result["status"],
                        error_message=result.get("error_message")
                    )
                    
                    db.add(db_result)
                    processed_count += 1
                
            else:
                # Fallback to individual processing for text or other types
                for file_data in files_data:
                    try:
                        start_time = time.time()
                        
                        # Classify content based on type
                        if job_type == "image":
                            result = await self.advanced_ml_service.classify_image_single(
                                image_path=file_data["file_path"],
                                model_name="resnet50",
                                include_metadata=False
                            )
                        else:
                            # Add text classification when implemented
                            result = {"predicted_label": "text_classification_pending", "confidence": 0.0}
                        
                        processing_time = round(time.time() - start_time, 3)
                        
                        # Save result to database
                        db_result = Result(
                            job_id=job_id,
                            file_id=file_data.get("file_id"),
                            filename=file_data["filename"],
                            predicted_label=result["predicted_label"],
                            confidence=result["confidence"],
                            processing_time=processing_time,
                            status="success"
                        )
                        
                        db.add(db_result)
                        processed_count += 1
                        
                        # Update job progress
                        job.completed_items = processed_count
                        db.commit()
                        
                    except Exception as file_error:
                        # Log individual file errors but continue processing
                        error_result = Result(
                            job_id=job_id,
                            file_id=file_data.get("file_id"),
                            filename=file_data["filename"],
                            predicted_label=None,
                            confidence=0.0,
                            status="error",
                            error_message=str(file_error)
                        )
                        db.add(error_result)
                        db.commit()
            
            # Mark job as completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.total_items = len(files_data)
            job.completed_items = processed_count
            db.commit()
            
        except Exception as e:
            # Mark job as failed
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "failed"
            job.error_message = str(e)
            db.commit()

# Global service instance
classification_service = ClassificationService()

@router.post("/image")
async def classify_single_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Classify a single image - for testing and quick classification"""
    
    # Check user credits
    if current_user.credits_remaining < 1:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    try:
        # Save file temporarily
        temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join("uploads", temp_filename)
        os.makedirs("uploads", exist_ok=True)
        
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Run classification using advanced service
        result = await classification_service.advanced_ml_service.classify_image_single(
            image_path=temp_path,
            model_name="resnet50",
            include_metadata=True
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Deduct credit
        current_user.credits_remaining -= 1
        db.commit()
        
        return {
            "predicted_label": result["predicted_label"],
            "confidence": round(result["confidence"] * 100, 2),
            "processing_time": result["processing_time"],
            "classification_id": result["classification_id"],
            "model_used": result["model_used"],
            "credits_remaining": current_user.credits_remaining,
            "metadata": result.get("processing_metadata", {}),
            "quality_metrics": result.get("quality_metrics", {})
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/image/quick")
async def classify_quick_image(
    file: UploadFile = File(...),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Quick image classification without authentication - for frictionless experience"""
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    # Check file size (limit to 10MB for quick endpoint)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
    
    try:
        # Save file temporarily
        temp_filename = f"quick_{uuid.uuid4()}_{file.filename}"
        temp_path = os.path.join("uploads", temp_filename)
        os.makedirs("uploads", exist_ok=True)
        
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Run classification using advanced service
        result = await classification_service.advanced_ml_service.classify_image_single(
            image_path=temp_path,
            model_name="resnet50",
            include_metadata=False
        )
        
        # Clean up temp file
        os.remove(temp_path)
        
        return {
            "predicted_label": result["predicted_label"],
            "confidence": round(result["confidence"] * 100, 2),
            "processing_time": result["processing_time"],
            "classification_id": result["classification_id"],
            "status": result["status"],
            "note": "Quick classification - register for advanced features"
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/batch/quick")
async def quick_batch_classification(
    files: List[UploadFile] = File(...),
    job_type: str = "image"
):
    """Quick batch classification without authentication - limited to 5 files for demo"""
    
    # Limit to 5 files for demo
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Demo batch limit is 5 files. Sign up for more!")
    
    # Validate file types
    valid_image_types = ["image/jpeg", "image/png", "image/gif"]
    
    for file in files:
        if job_type == "image" and file.content_type not in valid_image_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type for image classification: {file.content_type}"
            )
    
    try:
        results = []
        
        for i, file in enumerate(files):
            # Save file temporarily
            temp_filename = f"quick_batch_{uuid.uuid4()}_{file.filename}"
            temp_path = os.path.join("uploads", temp_filename)
            os.makedirs("uploads", exist_ok=True)
            
            contents = await file.read()
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            try:
                # Run classification
                result = await classification_service.advanced_ml_service.classify_image_single(
                    image_path=temp_path,
                    model_name="resnet50",
                    include_metadata=False
                )
                
                results.append({
                    "filename": file.filename,
                    "predicted_label": result["predicted_label"],
                    "confidence": round(result["confidence"] * 100, 2),
                    "processing_time": result["processing_time"],
                    "status": "success"
                })
                
            except Exception as file_error:
                results.append({
                    "filename": file.filename,
                    "predicted_label": None,
                    "confidence": 0,
                    "status": "error",
                    "error_message": str(file_error)
                })
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return {
            "results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "note": "Sign up for batch processing of larger datasets and background processing"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@router.post("/batch")
async def create_batch_classification_job(
    files: List[UploadFile] = File(...),
    job_type: str = "image",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new batch classification job"""
    
    # Validate job type
    if job_type not in ["image", "text"]:
        raise HTTPException(status_code=400, detail="Invalid job type. Use 'image' or 'text'")
    
    # Check batch size limits
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Batch size exceeds limit of 100 files")
    
    # Check user credits
    if current_user.credits_remaining < len(files):
        raise HTTPException(
            status_code=402, 
            detail=f"Insufficient credits. Need {len(files)}, have {current_user.credits_remaining}"
        )
    
    # Validate file types
    valid_image_types = ["image/jpeg", "image/png", "image/gif"]
    valid_text_types = ["text/plain", "text/csv"]
    
    for file in files:
        if job_type == "image" and file.content_type not in valid_image_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type for image classification: {file.content_type}"
            )
        elif job_type == "text" and file.content_type not in valid_text_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type for text classification: {file.content_type}"
            )
    
    try:
        # Create job record
        job = Job(
            user_id=current_user.id,
            job_type=job_type,
            total_items=len(files),
            status="queued"
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Save files and prepare for processing
        files_data = []
        
        for file in files:
            # Save file
            filename = f"{job.id}_{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join("uploads", filename)
            
            contents = await file.read()
            with open(file_path, "wb") as f:
                f.write(contents)
            
            files_data.append({
                "filename": file.filename,
                "file_path": file_path,
                "file_id": None  # We're not storing in FileModel for batch jobs
            })
        
        # Start background processing
        background_tasks.add_task(
            classification_service.process_classification_job,
            job.id, files_data, job_type, db
        )
        
        # Deduct credits
        current_user.credits_remaining -= len(files)
        db.commit()
        
        return {
            "job_id": job.id,
            "status": job.status,
            "total_items": job.total_items,
            "message": f"Batch classification job created with {len(files)} files",
            "credits_remaining": current_user.credits_remaining
        }
        
    except Exception as e:
        # Clean up any saved files on error
        for file_data in files_data:
            if os.path.exists(file_data["file_path"]):
                os.remove(file_data["file_path"])
        
        raise HTTPException(status_code=500, detail=f"Failed to create batch job: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get information about available ML models"""
    try:
        models_info = classification_service.advanced_ml_service.get_available_models()
        performance_stats = classification_service.advanced_ml_service.get_performance_stats()
        
        return {
            "available_models": models_info,
            "performance_stats": performance_stats,
            "service_status": "operational"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models info: {str(e)}")

@router.get("/jobs/{job_id}")
def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of a classification job"""
    
    # Get job with user verification
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate progress percentage
    progress_percentage = 0
    if job.total_items > 0:
        progress_percentage = round((job.completed_items / job.total_items) * 100, 2)
    
    return {
        "job_id": job.id,
        "status": job.status,
        "job_type": job.job_type,
        "total_items": job.total_items,
        "completed_items": job.completed_items,
        "progress_percentage": progress_percentage,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message if job.status == "failed" else None
    }

@router.get("/jobs")
def get_user_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get all classification jobs for the current user"""
    
    jobs = db.query(Job).filter(
        Job.user_id == current_user.id
    ).order_by(Job.created_at.desc()).offset(offset).limit(limit).all()
    
    jobs_data = []
    for job in jobs:
        progress_percentage = 0
        if job.total_items > 0:
            progress_percentage = round((job.completed_items / job.total_items) * 100, 2)
        
        jobs_data.append({
                "job_id": job.id,
                "status": job.status,
                "job_type": job.job_type,
                "total_items": job.total_items,
                "completed_items": job.completed_items,
            "progress_percentage": progress_percentage,
                "created_at": job.created_at,
                "completed_at": job.completed_at
        })
    
    return {
        "jobs": jobs_data,
        "total_count": len(jobs_data),
        "limit": limit,
        "offset": offset
    }

@router.get("/results/{job_id}")
def get_job_results(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    include_errors: bool = Query(default=False),
    confidence_threshold: float = Query(default=0.0, ge=0.0, le=1.0)
):
    """Get the results of a completed classification job"""
    
    # Verify job ownership
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Build query for results
    results_query = db.query(Result).filter(Result.job_id == job_id)
    
    # Apply filters
    if not include_errors:
        results_query = results_query.filter(Result.status == "success")
    
    if confidence_threshold > 0:
        results_query = results_query.filter(Result.confidence >= confidence_threshold)
    
    results = results_query.all()
    
    # Format results
    formatted_results = []
    for result in results:
        formatted_results.append({
            "result_id": result.id,
            "filename": result.filename,
            "predicted_label": result.predicted_label,
            "confidence": round(result.confidence * 100, 2) if result.confidence else 0,
            "processing_time": result.processing_time,
            "status": result.status,
            "error_message": result.error_message,
            "reviewed": result.reviewed,
            "ground_truth": result.ground_truth
        })
    
    # Calculate summary statistics
    successful_results = [r for r in results if r.status == "success" and r.confidence is not None]
    
    summary = {
        "total_results": len(results),
        "successful_results": len(successful_results),
        "failed_results": len(results) - len(successful_results),
        "average_confidence": round(
            sum(r.confidence for r in successful_results) / len(successful_results) * 100, 2
        ) if successful_results else 0,
        "average_processing_time": round(
            sum(r.processing_time for r in successful_results) / len(successful_results), 3
        ) if successful_results else 0
    }
    
    return {
        "job_id": job_id,
        "job_status": job.status,
        "results": formatted_results,
        "summary": summary,
        "filters_applied": {
            "include_errors": include_errors,
            "confidence_threshold": confidence_threshold
        }
    } 