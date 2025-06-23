from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, Query, Form
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result, File as FileModel
from auth import get_current_user, get_optional_user
from advanced_ml_service import AdvancedMLService, advanced_ml_service
from object_detection_service import object_detection_service
from typing import List, Dict, Any, Optional
import asyncio
import time
import uuid
import os
from datetime import datetime
from project_file_manager import project_file_manager
from text_ml_service import text_ml_service

router = APIRouter(prefix="/api/classify", tags=["classification"])

class ClassificationService:
    """Service class handling classification business logic"""
    
    def __init__(self):
        self.advanced_ml_service = advanced_ml_service
        
        # Auto-approval configuration
        self.auto_approval_config = {
            "enabled": True,
            "confidence_thresholds": {
                "image_classification": 0.85,   # 85% confidence for image classification
                "object_detection": 0.75,       # 75% confidence for object detection
                "text_classification": 0.80,    # 80% confidence for text classification
                "ner": 0.70,                     # 70% confidence for named entity recognition
                "sentiment": 0.75,               # 75% confidence for sentiment analysis
                "default": 0.80                  # Default threshold for other types
            },
            "min_results_for_batch_auto_approval": 5,  # Minimum results needed for batch auto-approval
            "quality_score_threshold": 0.7,     # Quality score threshold for auto-approval
            "max_auto_approval_per_job": 1000   # Maximum auto-approvals per job (safety limit)
        }
    
    def should_auto_approve(self, result: Dict[str, Any], classification_type: str = "default") -> bool:
        """
        Determine if a classification result should be auto-approved
        
        Args:
            result: Classification result with confidence and other metrics
            classification_type: Type of classification (image_classification, object_detection, etc.)
            
        Returns:
            Boolean indicating if result should be auto-approved
        """
        
        if not self.auto_approval_config["enabled"]:
            return False
        
        # Get confidence threshold for this classification type
        threshold = self.auto_approval_config["confidence_thresholds"].get(
            classification_type, 
            self.auto_approval_config["confidence_thresholds"]["default"]
        )
        
        # Check confidence score
        confidence = result.get("confidence", 0.0)
        if isinstance(confidence, (int, float)) and confidence < threshold * 100:  # Convert to percentage
            return False
        
        # Check quality score if available
        quality_score = result.get("quality_score", result.get("quality_metrics", {}).get("overall_score", 1.0))
        if quality_score < self.auto_approval_config["quality_score_threshold"]:
            return False
        
        # Check for error status
        if result.get("status") == "error":
            return False
        
        # Additional checks for object detection
        if classification_type == "object_detection":
            # Require high-quality detections for auto-approval
            if result.get("summary", {}).get("high_quality_detections", 0) == 0:
                return False
            
            # Check detection density (not too many or too few objects)
            detection_density = result.get("summary", {}).get("detection_density", 0)
            if detection_density > 50 or detection_density < 0.1:  # Reasonable density range
                return False
        
        return True
    
    async def apply_auto_approval_workflow(self, results: List[Dict], classification_type: str, job_id: int, db: Session) -> Dict[str, int]:
        """
        Apply auto-approval workflow to classification results
        
        Args:
            results: List of classification results
            classification_type: Type of classification
            job_id: Job ID for tracking
            db: Database session
            
        Returns:
            Dictionary with approval statistics
        """
        
        auto_approved_count = 0
        requires_review_count = 0
        max_auto_approvals = self.auto_approval_config["max_auto_approval_per_job"]
        
        for result in results:
            # Check if we've reached the auto-approval limit
            if auto_approved_count >= max_auto_approvals:
                requires_review_count += 1
                continue
            
            # Determine if result should be auto-approved
            should_approve = self.should_auto_approve(result, classification_type)
            
            if should_approve:
                # Mark as auto-approved in the result
                result["auto_approved"] = True
                result["review_status"] = "auto_approved"
                result["reviewed_at"] = datetime.utcnow().isoformat()
                result["reviewer"] = "system_auto_approval"
                auto_approved_count += 1
            else:
                # Mark as requiring human review
                result["auto_approved"] = False
                result["review_status"] = "pending_review"
                result["review_priority"] = self._calculate_review_priority(result, classification_type)
                requires_review_count += 1
        
        # Log auto-approval statistics
        approval_stats = {
            "auto_approved": auto_approved_count,
            "requires_review": requires_review_count,
            "total_processed": len(results),
            "auto_approval_rate": round((auto_approved_count / len(results)) * 100, 2) if results else 0,
            "classification_type": classification_type,
            "job_id": job_id
        }
        
        return approval_stats
    
    def _calculate_review_priority(self, result: Dict[str, Any], classification_type: str) -> str:
        """
        Calculate review priority for results that require human review
        
        Args:
            result: Classification result
            classification_type: Type of classification
            
        Returns:
            Priority level: "high", "medium", "low"
        """
        
        confidence = result.get("confidence", 0.0)
        
        # High priority: Very low confidence or errors
        if result.get("status") == "error" or confidence < 30:
            return "high"
        
        # Medium priority: Moderate confidence
        if confidence < 60:
            return "medium"
        
        # Low priority: High confidence but below auto-approval threshold
        return "low"
    
    async def process_classification_job(self, job_id: int, files_data: List[Dict], job_type: str, db: Session):
        """Process classification job in background with auto-approval workflow"""
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
                
                # Apply auto-approval workflow to batch results
                approval_stats = await self.apply_auto_approval_workflow(
                    results=batch_results,
                    classification_type="image_classification",
                    job_id=job_id,
                    db=db
                )
                
                # Save results to database with auto-approval status
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
                        error_message=result.get("error_message"),
                        # Auto-approval fields
                        auto_approved=result.get("auto_approved", False),
                        review_status=result.get("review_status", "pending_review"),
                        review_priority=result.get("review_priority", "medium"),
                        reviewed_at=result.get("reviewed_at"),
                        reviewer=result.get("reviewer")
                    )
                    
                    db.add(db_result)
                    processed_count += 1
                
                # Store approval statistics in job metadata
                job.metadata = {
                    "auto_approval_stats": approval_stats,
                    "processing_completed_at": datetime.utcnow().isoformat()
                }
                db.commit()
                
            else:
                # Fallback to individual processing for text or other types
                individual_results = []
                
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
                        result["processing_time"] = processing_time
                        individual_results.append(result)
                        
                        processed_count += 1
                        
                        # Update job progress
                        job.completed_items = processed_count
                        db.commit()
                        
                    except Exception as file_error:
                        # Add error result
                        error_result = {
                            "predicted_label": None,
                            "confidence": 0.0,
                            "status": "error",
                            "error_message": str(file_error),
                            "processing_time": 0.0
                        }
                        individual_results.append(error_result)
                
                # Apply auto-approval workflow to individual results
                classification_type = "image_classification" if job_type == "image" else "text_classification"
                approval_stats = await self.apply_auto_approval_workflow(
                    results=individual_results,
                    classification_type=classification_type,
                    job_id=job_id,
                    db=db
                )
                
                # Save individual results to database with auto-approval status
                for i, result in enumerate(individual_results):
                    file_data = files_data[i]
                    
                    db_result = Result(
                        job_id=job_id,
                        file_id=file_data.get("file_id"),
                        filename=file_data["filename"],
                        predicted_label=result["predicted_label"],
                        confidence=result["confidence"],
                        processing_time=result["processing_time"],
                        status=result.get("status", "success"),
                        error_message=result.get("error_message"),
                        # Auto-approval fields
                        auto_approved=result.get("auto_approved", False),
                        review_status=result.get("review_status", "pending_review"),
                        review_priority=result.get("review_priority", "medium"),
                        reviewed_at=result.get("reviewed_at"),
                        reviewer=result.get("reviewer")
                    )
                    
                    db.add(db_result)
                
                # Store approval statistics in job metadata
                job.metadata = {
                    "auto_approval_stats": approval_stats,
                    "processing_completed_at": datetime.utcnow().isoformat()
                }
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

    async def process_text_classification_job(self, job_id: int, texts_data: List[Dict], classification_type: str, db: Session):
        """Process text classification job in background with auto-approval workflow"""
        try:
            # Update job status to processing
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "processing"
            db.commit()
            
            results = []
            processed_count = 0
            
            # Process texts individually for now (can be optimized for batch later)
            for text_data in texts_data:
                try:
                    start_time = time.time()
                    
                    # Run text classification
                    result = await text_ml_service.classify_text_single(
                        text=text_data["text"],
                        classification_type=text_data["classification_type"],
                        custom_categories=text_data.get("custom_categories"),
                        include_metadata=text_data.get("include_metadata", False)
                    )
                    
                    processing_time = round(time.time() - start_time, 3)
                    result["processing_time"] = processing_time
                    results.append(result)
                    
                    processed_count += 1
                    
                    # Update job progress
                    job.completed_items = processed_count
                    db.commit()
                    
                except Exception as text_error:
                    # Add error result
                    error_result = {
                        "predicted_label": "error",
                        "confidence": 0.0,
                        "status": "error",
                        "error_message": str(text_error),
                        "processing_time": 0.0,
                        "classification_type": classification_type
                    }
                    results.append(error_result)
                    processed_count += 1
            
            # Apply auto-approval workflow to text results
            approval_stats = await self.apply_auto_approval_workflow(
                results=results,
                classification_type=classification_type,
                job_id=job_id,
                db=db
            )
            
            # Save results to database with auto-approval status
            for i, result in enumerate(results):
                text_data_item = texts_data[i]
                
                db_result = Result(
                    job_id=job_id,
                    file_id=None,  # No file for text classification
                    filename=f"text_{i+1}",  # Use index as filename
                    predicted_label=result["predicted_label"],
                    confidence=result["confidence"],
                    processing_time=result["processing_time"],
                    status=result.get("status", "success"),
                    error_message=result.get("error_message"),
                    # Auto-approval fields
                    auto_approved=result.get("auto_approved", False),
                    review_status=result.get("review_status", "pending_review"),
                    review_priority=result.get("review_priority", "medium"),
                    reviewed_at=result.get("reviewed_at"),
                    reviewer=result.get("reviewer"),
                    # Text-specific metadata
                    metadata={
                        "original_text": text_data_item["text"],
                        "text_index": i,
                        "classification_type": classification_type,
                        "entities": result.get("entities", []) if classification_type in ["ner", "named_entity"] else None,
                        "entity_summary": result.get("entity_summary", {}) if classification_type in ["ner", "named_entity"] else None
                    }
                )
                
                db.add(db_result)
            
            # Store approval statistics in job metadata
            job.metadata = {
                "auto_approval_stats": approval_stats,
                "classification_type": classification_type,
                "processing_completed_at": datetime.utcnow().isoformat()
            }
            db.commit()
            
            # Mark job as completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.total_items = len(texts_data)
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

@router.post("/image/detect")
async def detect_objects_in_image(
    file: UploadFile = File(...),
    project_id: int = Form(...),
    model_name: str = Form("yolo8n"),
    confidence_threshold: float = Form(0.25),
    annotate_image: bool = Form(True)
):
    """Detect objects in a single image for a specific project"""
    
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    try:
        # Save original file to project storage
        file_content = await file.read()
        original_file_path = project_file_manager.save_original_file(
            project_id=project_id,
            file_content=file_content, 
            filename=file.filename
        )
        
        if not original_file_path:
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Process image with object detection
        result = await object_detection_service.detect_objects(
            image_path=original_file_path,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            annotate_image=annotate_image,
            save_annotated=True,
            project_id=project_id  # Pass project_id to the detection service
        )
        
        return {
            "project_id": project_id,
            "filename": file.filename,
            "detection_results": result,
            "total_objects_detected": result.get("total_objects_detected", 0),
            "processing_time": result.get("processing_time", 0),
            "annotated_image_path": result.get("annotated_image_path"),
            "detections": result.get("detections", []),
            "summary": result.get("summary", {}),
            "quality_score": result.get("quality_score"),
            "message": "Object detection completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@router.post("/batch/detect")
async def detect_objects_batch(
    files: List[UploadFile] = File(...),
    project_id: int = Form(...),
    model_name: str = Form("yolo8n"),
    confidence_threshold: float = Form(0.25),
    annotate_images: bool = Form(True)
):
    """Detect objects in multiple images for a specific project"""
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files per batch.")
    
    try:
        batch_results = []
        total_objects = 0
        
        for i, file in enumerate(files):
            try:
                # Validate file type
                if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/gif"]:
                    batch_results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error_message": "Invalid file type"
                    })
                    continue
                
                # Save original file to project storage
                file_content = await file.read()
                original_file_path = project_file_manager.save_original_file(
                    project_id=project_id,
                    file_content=file_content,
                    filename=file.filename
                )
                
                if not original_file_path:
                    batch_results.append({
                        "filename": file.filename,
                        "status": "error", 
                        "error_message": "Failed to save file"
                    })
                    continue
                
                # Run object detection
                result = await object_detection_service.detect_objects(
                    image_path=original_file_path,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    annotate_image=annotate_images,
                    save_annotated=True,
                    project_id=project_id
                )
                
                # Format result for batch response
                file_result = {
                    "filename": file.filename,
                    "status": "success",
                    "total_objects_detected": result.get("total_objects_detected", 0),
                    "detections": result.get("detections", []),
                    "processing_time": result.get("processing_time", 0),
                    "annotated_image_path": result.get("annotated_image_path"),
                    "quality_score": result.get("quality_score"),
                    "summary": result.get("summary", {})
                }
                
                batch_results.append(file_result)
                total_objects += result.get("total_objects_detected", 0)
                
            except Exception as e:
                logger.error(f"Failed to process {file.filename}: {str(e)}")
                batch_results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error_message": str(e)
                })
        
        # Calculate summary statistics
        successful_results = [r for r in batch_results if r["status"] == "success"]
        failed_results = [r for r in batch_results if r["status"] == "error"]
        
        return {
            "project_id": project_id,
            "total_processed": len(files),
            "successful_detections": len(successful_results),
            "failed_detections": len(failed_results),
            "total_objects_detected": total_objects,
            "average_objects_per_image": total_objects / max(len(successful_results), 1),
            "results": batch_results,
            "summary": {
                "model_used": model_name,
                "confidence_threshold": confidence_threshold,
                "annotated_images": annotate_images
            }
        }
        
    except Exception as e:
        logger.error(f"Batch detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")

@router.get("/annotated/{image_filename}")
async def get_annotated_image(image_filename: str):
    """Serve annotated images with bounding boxes"""
    
    # Check if file exists in annotated directory
    annotated_path = os.path.join("uploads", "annotated", image_filename)
    
    if not os.path.exists(annotated_path):
        raise HTTPException(status_code=404, detail="Annotated image not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        path=annotated_path,
        media_type="image/jpeg",
        filename=image_filename
    )

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
        # Get classification models
        classification_models = classification_service.advanced_ml_service.get_available_models()
        classification_stats = classification_service.advanced_ml_service.get_performance_stats()
        
        # Get object detection models
        detection_models = object_detection_service.get_available_models()
        detection_stats = object_detection_service.get_performance_stats()
        
        return {
            "classification_models": classification_models,
            "object_detection_models": detection_models,
            "classification_stats": classification_stats,
            "detection_stats": detection_stats,
            "supported_image_formats": ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
            "capabilities": {
                "image_classification": "Classify entire image into single category",
                "object_detection": "Detect and locate multiple objects with bounding boxes",
                "batch_processing": "Process multiple images simultaneously",
                "annotated_images": "Generate images with visual annotations"
            },
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

@router.post("/text")
async def classify_single_text(
    text: str = Form(...),
    classification_type: str = Form("sentiment"),
    custom_categories: List[str] = Form(default=None),
    include_metadata: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Classify a single text - supports sentiment, emotion, topic, spam, toxicity, language, ner, named_entity"""
    
    # Check user credits
    if current_user.credits_remaining < 1:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    # Validate classification type
    valid_types = ["sentiment", "emotion", "topic", "spam", "toxicity", "language", "ner", "named_entity"]
    if classification_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid classification type. Must be one of: {valid_types}")
    
    # Validate text input
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Run text classification using the text ML service
        result = await text_ml_service.classify_text_single(
            text=text,
            classification_type=classification_type,
            custom_categories=custom_categories,
            include_metadata=include_metadata
        )
        
        # Deduct credit
        current_user.credits_remaining -= 1
        db.commit()
        
        # Enhanced response for different classification types
        response = {
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "classification_id": result["classification_id"],
            "classification_type": classification_type,
            "credits_remaining": current_user.credits_remaining,
            "status": result["status"]
        }
        
        # Add NER-specific fields
        if classification_type in ["ner", "named_entity"]:
            response.update({
                "entities_found": result.get("entities_found", 0),
                "entities": result.get("entities", []),
                "entity_summary": result.get("entity_summary", {})
            })
        
        # Add metadata if requested
        if include_metadata:
            response["metadata"] = {
                "text_metadata": result.get("text_metadata", {}),
                "model_metadata": result.get("model_metadata", {}),
                "processing_metadata": result.get("processing_metadata", {}),
                "all_predictions": result.get("all_predictions", [])
            }
            
            # Add NER-specific statistics
            if classification_type in ["ner", "named_entity"]:
                response["metadata"]["ner_statistics"] = result.get("ner_statistics", {})
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text classification failed: {str(e)}")

@router.post("/text/batch")
async def classify_text_batch(
    texts: List[str] = Form(...),
    classification_type: str = Form("sentiment"),
    custom_categories: List[str] = Form(default=None),
    include_metadata: bool = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Batch text classification with auto-approval workflow"""
    
    # Validate input
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    # Check user credits
    required_credits = len(texts)
    if current_user.credits_remaining < required_credits:
        raise HTTPException(status_code=402, detail=f"Insufficient credits. Need {required_credits}, have {current_user.credits_remaining}")
    
    # Validate classification type
    valid_types = ["sentiment", "emotion", "topic", "spam", "toxicity", "language", "ner", "named_entity"]
    if classification_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid classification type. Must be one of: {valid_types}")
    
    try:
        # Create batch job
        job = Job(
            user_id=current_user.id,
            job_type="text",
            status="created",
            total_items=len(texts),
            completed_items=0,
            metadata={"classification_type": classification_type, "custom_categories": custom_categories}
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Prepare texts data for processing
        texts_data = [
            {
                "text": text,
                "index": i,
                "classification_type": classification_type,
                "custom_categories": custom_categories,
                "include_metadata": include_metadata
            }
            for i, text in enumerate(texts)
        ]
        
        # Deduct credits upfront
        current_user.credits_remaining -= required_credits
        db.commit()
        
        # Start background processing
        background_tasks.add_task(
            classification_service.process_text_classification_job,
            job.id,
            texts_data,
            classification_type,
            db
        )
        
        return {
            "job_id": job.id,
            "status": "created",
            "total_items": len(texts),
            "message": f"Batch text classification job created for {len(texts)} texts",
            "classification_type": classification_type,
            "credits_deducted": required_credits,
            "credits_remaining": current_user.credits_remaining
        }
        
    except Exception as e:
        # Refund credits if job creation failed
        current_user.credits_remaining += required_credits
        db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to create batch job: {str(e)}")

@router.get("/text/models")
async def get_available_text_models():
    """Get available text classification models and their metadata"""
    
    try:
        models_info = text_ml_service.get_model_info()
        
        return {
            "available_models": models_info,
            "classification_types": [
                {
                    "type": "sentiment",
                    "description": "Classify text sentiment as positive, negative, or neutral",
                    "use_cases": ["Social media monitoring", "Customer feedback", "Content moderation"]
                },
                {
                    "type": "emotion", 
                    "description": "Detect emotional tone in text",
                    "use_cases": ["Emotional analysis", "Mental health monitoring", "Customer sentiment"]
                },
                {
                    "type": "topic",
                    "description": "Classify text into custom topic categories",
                    "use_cases": ["Content categorization", "Document classification", "Research analysis"]
                },
                {
                    "type": "spam",
                    "description": "Detect spam, toxic, or harmful content",
                    "use_cases": ["Email filtering", "Comment moderation", "Content safety"]
                },
                {
                    "type": "toxicity",
                    "description": "Advanced toxicity detection for comments",
                    "use_cases": ["Comment moderation", "Social media safety", "Content filtering"]
                },
                {
                    "type": "language",
                    "description": "Automatically detect the language of text",
                    "use_cases": ["Multilingual support", "Content routing", "Research analysis"]
                },
                {
                    "type": "ner",
                    "description": "Named Entity Recognition - extract persons, organizations, locations, and misc entities",
                    "use_cases": ["Information extraction", "Document analysis", "Knowledge graphs"]
                },
                {
                    "type": "named_entity",
                    "description": "Advanced Named Entity Recognition with high accuracy",
                    "use_cases": ["Entity extraction", "Content analysis", "Data mining"]
                }
            ],
            "auto_approval_config": classification_service.auto_approval_config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}") 