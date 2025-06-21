from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result, File
from auth import get_current_user
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

router = APIRouter(prefix="/api/review", tags=["review"])

logger = logging.getLogger(__name__)

@router.get("/jobs/{job_id}")
async def get_review_interface_data(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    confidence_filter: Optional[float] = Query(None, ge=0.0, le=1.0),
    status_filter: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(50, le=200)
):
    """Get data for human review interface"""
    
    # Verify job ownership
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Build query for results
    query = db.query(Result).filter(Result.job_id == job_id)
    
    # Apply filters
    if confidence_filter is not None:
        query = query.filter(Result.confidence <= confidence_filter)
    
    if status_filter:
        query = query.filter(Result.status == status_filter)
    
    # Get total count for pagination
    total_results = query.count()
    
    # Apply pagination
    offset = (page - 1) * limit
    results = query.offset(offset).limit(limit).all()
    
    # Calculate review statistics
    total_items = db.query(Result).filter(Result.job_id == job_id).count()
    reviewed_items = db.query(Result).filter(
        Result.job_id == job_id,
        Result.reviewed == True
    ).count()
    
    low_confidence_items = db.query(Result).filter(
        Result.job_id == job_id,
        Result.confidence < 0.8,
        Result.status == "success"
    ).count()
    
    error_items = db.query(Result).filter(
        Result.job_id == job_id,
        Result.status == "error"
    ).count()
    
    # Format results for frontend
    formatted_results = []
    for result in results:
        formatted_result = {
            "id": result.id,
            "filename": result.filename,
            "predicted_label": result.predicted_label,
            "confidence": round(result.confidence * 100, 1) if result.confidence else 0,
            "confidence_raw": result.confidence,
            "status": result.status,
            "reviewed": result.reviewed,
            "ground_truth": result.ground_truth,
            "error_message": result.error_message,
            "processing_time": result.processing_time,
            "created_at": result.created_at.isoformat(),
            "needs_review": result.confidence < 0.8 if result.confidence else True,
            "file_path": f"/api/files/{result.file_id}" if result.file_id else None
        }
        formatted_results.append(formatted_result)
    
    return {
        "job_info": {
            "id": job.id,
            "job_type": job.job_type,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        },
        "review_stats": {
            "total_items": total_items,
            "reviewed_items": reviewed_items,
            "pending_review": total_items - reviewed_items,
            "low_confidence_items": low_confidence_items,
            "error_items": error_items,
            "review_progress": (reviewed_items / total_items * 100) if total_items > 0 else 0
        },
        "results": formatted_results,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total_results,
            "total_pages": (total_results + limit - 1) // limit
        },
        "filters_applied": {
            "confidence_filter": confidence_filter,
            "status_filter": status_filter
        }
    }

@router.post("/results/{result_id}/approve")
async def approve_prediction(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Approve an AI prediction"""
    
    result = db.query(Result).join(Job).filter(
        Result.id == result_id,
        Job.user_id == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Mark as reviewed and approved
    result.reviewed = True
    result.ground_truth = result.predicted_label
    result.review_action = "approved"
    result.reviewed_at = datetime.utcnow()
    result.reviewed_by = current_user.id
    
    db.commit()
    
    logger.info(f"User {current_user.id} approved result {result_id}")
    
    return {
        "success": True,
        "message": "Prediction approved",
        "result_id": result_id,
        "action": "approved"
    }

@router.post("/results/{result_id}/correct")
async def correct_prediction(
    result_id: int,
    correct_label: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Correct an AI prediction with the true label"""
    
    result = db.query(Result).join(Job).filter(
        Result.id == result_id,
        Job.user_id == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Mark as reviewed and corrected
    result.reviewed = True
    result.ground_truth = correct_label
    result.review_action = "corrected"
    result.reviewed_at = datetime.utcnow()
    result.reviewed_by = current_user.id
    
    db.commit()
    
    logger.info(f"User {current_user.id} corrected result {result_id}: {result.predicted_label} -> {correct_label}")
    
    return {
        "success": True,
        "message": "Prediction corrected",
        "result_id": result_id,
        "action": "corrected",
        "original_label": result.predicted_label,
        "corrected_label": correct_label
    }

@router.post("/results/{result_id}/reject")
async def reject_prediction(
    result_id: int,
    reason: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Reject an AI prediction"""
    
    result = db.query(Result).join(Job).filter(
        Result.id == result_id,
        Job.user_id == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Mark as reviewed and rejected
    result.reviewed = True
    result.review_action = "rejected"
    result.rejection_reason = reason
    result.reviewed_at = datetime.utcnow()
    result.reviewed_by = current_user.id
    
    db.commit()
    
    logger.info(f"User {current_user.id} rejected result {result_id}: {reason}")
    
    return {
        "success": True,
        "message": "Prediction rejected",
        "result_id": result_id,
        "action": "rejected",
        "reason": reason
    }

@router.post("/jobs/{job_id}/bulk-approve")
async def bulk_approve_high_confidence(
    job_id: int,
    confidence_threshold: float = Query(0.9, ge=0.0, le=1.0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Bulk approve all high-confidence predictions"""
    
    # Verify job ownership
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Find high-confidence unreviewed results
    high_confidence_results = db.query(Result).filter(
        Result.job_id == job_id,
        Result.confidence >= confidence_threshold,
        Result.status == "success",
        Result.reviewed == False
    ).all()
    
    approved_count = 0
    for result in high_confidence_results:
        result.reviewed = True
        result.ground_truth = result.predicted_label
        result.review_action = "bulk_approved"
        result.reviewed_at = datetime.utcnow()
        result.reviewed_by = current_user.id
        approved_count += 1
    
    db.commit()
    
    logger.info(f"User {current_user.id} bulk approved {approved_count} results for job {job_id}")
    
    return {
        "success": True,
        "message": f"Bulk approved {approved_count} high-confidence predictions",
        "approved_count": approved_count,
        "confidence_threshold": confidence_threshold
    }

@router.get("/suggestions/{result_id}")
async def get_label_suggestions(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get label suggestions for correction"""
    
    result = db.query(Result).join(Job).filter(
        Result.id == result_id,
        Job.user_id == current_user.id
    ).first()
    
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Get common labels from this user's other results
    common_labels = db.query(Result.ground_truth).filter(
        Result.ground_truth.isnot(None),
        Result.job_id.in_(
            db.query(Job.id).filter(Job.user_id == current_user.id)
        )
    ).distinct().limit(10).all()
    
    suggestions = [label[0] for label in common_labels if label[0]]
    
    # Add some common ImageNet categories as fallback
    default_suggestions = [
        "cat", "dog", "car", "person", "building", "food", "nature", "animal", "object", "other"
    ]
    
    # Combine and deduplicate
    all_suggestions = list(set(suggestions + default_suggestions))[:10]
    
    return {
        "suggestions": all_suggestions,
        "current_prediction": result.predicted_label,
        "confidence": result.confidence
    }

@router.get("/analytics/{job_id}")
async def get_review_analytics(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analytics for review process"""
    
    # Verify job ownership
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Calculate accuracy metrics
    reviewed_results = db.query(Result).filter(
        Result.job_id == job_id,
        Result.reviewed == True,
        Result.ground_truth.isnot(None)
    ).all()
    
    if not reviewed_results:
        return {
            "message": "No reviewed results yet",
            "analytics": None
        }
    
    total_reviewed = len(reviewed_results)
    correct_predictions = sum(1 for r in reviewed_results if r.predicted_label == r.ground_truth)
    accuracy = (correct_predictions / total_reviewed * 100) if total_reviewed > 0 else 0
    
    # Confidence distribution
    confidence_buckets = {
        "high": sum(1 for r in reviewed_results if r.confidence >= 0.9),
        "medium": sum(1 for r in reviewed_results if 0.7 <= r.confidence < 0.9),
        "low": sum(1 for r in reviewed_results if r.confidence < 0.7)
    }
    
    # Review actions distribution
    review_actions = {}
    for result in reviewed_results:
        action = getattr(result, 'review_action', 'unknown')
        review_actions[action] = review_actions.get(action, 0) + 1
    
    return {
        "job_id": job_id,
        "review_summary": {
            "total_reviewed": total_reviewed,
            "accuracy": round(accuracy, 2),
            "correct_predictions": correct_predictions,
            "incorrect_predictions": total_reviewed - correct_predictions
        },
        "confidence_distribution": confidence_buckets,
        "review_actions": review_actions,
        "recommendations": {
            "model_performance": "good" if accuracy > 85 else "needs_improvement",
            "suggested_confidence_threshold": 0.9 if accuracy > 90 else 0.95,
            "review_efficiency": "high" if confidence_buckets["high"] > total_reviewed * 0.7 else "medium"
        }
    } 