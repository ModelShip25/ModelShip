from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result, File, Project, UserRole
from auth import get_current_user
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import statistics

# Import file storage system
from file_storage import review_storage, project_storage

router = APIRouter(prefix="/api/review", tags=["review"])

logger = logging.getLogger(__name__)

class ReviewService:
    """Service for managing human review tasks and quality control"""
    
    def __init__(self):
        self.review_stats = {
            "total_reviews_completed": 0,
            "average_review_time": 0.0,
            "inter_annotator_agreement": 0.0
        }
    
    def assign_review_tasks(self, db: Session, project_id: int, reviewer_id: int, batch_size: int = 10) -> List[Dict]:
        """Assign review tasks to a reviewer based on priority and uncertainty"""
        
        # Get pending results that need review
        pending_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.status == "success",
            Result.reviewed == False,
            Result.confidence.between(0.5, 0.9)  # Uncertain predictions need review
        ).order_by(Result.confidence.asc()).limit(batch_size).all()
        
        # Mark as assigned for review
        assigned_tasks = []
        for result in pending_results:
            result.status = "pending_review"
            result.reviewed_by = reviewer_id
            
            assigned_tasks.append({
                "result_id": result.id,
                "filename": result.filename,
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "file_id": result.file_id,
                "detections": result.all_predictions if hasattr(result, 'detections') else None
            })
        
        db.commit()
        return assigned_tasks
    
    def calculate_inter_annotator_agreement(self, db: Session, project_id: int) -> float:
        """Calculate actual inter-annotator agreement for quality control"""
        
        # Get all results that have been reviewed for this project
        reviewed_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed == True,
            Result.ground_truth.isnot(None)
        ).all()
        
        if len(reviewed_results) < 2:
            return 0.0
        
        # Group results by file/item to find items reviewed by multiple people
        item_reviews = {}
        for result in reviewed_results:
            item_key = f"{result.job_id}_{result.filename}"
            if item_key not in item_reviews:
                item_reviews[item_key] = []
            item_reviews[item_key].append({
                'reviewer_id': result.reviewed_by,
                'predicted_label': result.predicted_label,
                'ground_truth': result.ground_truth,
                'review_action': result.review_action
            })
        
        # Find items with multiple reviews
        multi_reviewed_items = {k: v for k, v in item_reviews.items() if len(v) > 1}
        
        if not multi_reviewed_items:
            # If no items have multiple reviews, calculate agreement based on 
            # consistency between predicted labels and ground truth
            correct_predictions = sum(1 for r in reviewed_results 
                                    if r.predicted_label == r.ground_truth)
            total_predictions = len(reviewed_results)
            return round(correct_predictions / total_predictions, 3) if total_predictions > 0 else 0.0
        
        # Calculate agreement for items with multiple reviews
        total_agreements = 0
        total_comparisons = 0
        
        for item_key, reviews in multi_reviewed_items.items():
            # Calculate pairwise agreement for this item
            for i in range(len(reviews)):
                for j in range(i + 1, len(reviews)):
                    review1 = reviews[i]
                    review2 = reviews[j]
                    
                    # Check agreement on ground truth labels
                    if review1['ground_truth'] == review2['ground_truth']:
                        total_agreements += 1
                    
                    # Also check agreement on review actions
                    if review1['review_action'] == review2['review_action']:
                        total_agreements += 0.5  # Partial weight for action agreement
                    
                    total_comparisons += 1
        
        # Calculate overall agreement rate
        if total_comparisons == 0:
            # Fallback to prediction accuracy
            correct_predictions = sum(1 for r in reviewed_results 
                                    if r.predicted_label == r.ground_truth)
            return round(correct_predictions / len(reviewed_results), 3)
        
        agreement_rate = total_agreements / total_comparisons
        return round(min(1.0, agreement_rate), 3)  # Cap at 1.0

review_service = ReviewService()

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

@router.get("/tasks/pending")
def get_pending_review_tasks(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = Query(default=20, le=50)
):
    """Get pending review tasks for the current user"""
    
    # Check if user has reviewer role
    if current_user.role not in [UserRole.REVIEWER, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions for review tasks")
    
    # Get assigned or assignable tasks
    pending_tasks = db.query(Result).join(Job).filter(
        Job.project_id == project_id,
        Result.status.in_(["pending_review", "success"]),
        Result.reviewed == False,
        Result.confidence < 0.9  # Only uncertain predictions
    ).order_by(Result.confidence.asc()).limit(limit).all()
    
    tasks = []
    for result in pending_tasks:
        task_data = {
            "result_id": result.id,
            "job_id": result.job_id,
            "filename": result.filename,
            "predicted_label": result.predicted_label,
            "confidence": round(result.confidence * 100, 2) if result.confidence else 0,
            "processing_time": result.processing_time,
            "created_at": result.created_at,
            "file_id": result.file_id,
            "needs_review": result.confidence < 0.8 if result.confidence else True
        }
        
        # Add detection data if available
        if hasattr(result, 'all_predictions') and result.all_predictions:
            task_data["all_predictions"] = result.all_predictions
        
        tasks.append(task_data)
    
    return {
        "tasks": tasks,
        "total_pending": len(tasks),
        "project_id": project_id,
        "reviewer": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role.value
        }
    }

@router.post("/tasks/{result_id}/review")
def submit_review(
    result_id: int,
    review_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit a review decision for a result"""
    
    # Validate reviewer permissions
    if current_user.role not in [UserRole.REVIEWER, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Insufficient permissions for review")
    
    # Get the result
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Extract review decision
    action = review_data.get("action")  # "approve", "reject", "modify"
    corrected_label = review_data.get("corrected_label")
    notes = review_data.get("notes", "")
    review_time = review_data.get("review_time_seconds", 0)
    
    # Validate action
    if action not in ["approve", "reject", "modify"]:
        raise HTTPException(status_code=400, detail="Invalid review action")
    
    # Update result based on review
    result.reviewed = True
    result.reviewed_by = current_user.id
    result.reviewed_at = datetime.utcnow()
    result.review_action = action
    result.correction_reason = notes
    
    if action == "approve":
        result.ground_truth = result.predicted_label
        result.status = "approved"
    elif action == "reject":
        result.status = "rejected"
        result.ground_truth = None
    elif action == "modify":
        if not corrected_label:
            raise HTTPException(status_code=400, detail="Corrected label required for modify action")
        result.ground_truth = corrected_label
        result.status = "corrected"
    
    db.commit()
    
    # Update review statistics
    review_service.review_stats["total_reviews_completed"] += 1
    if review_time > 0:
        # Update average review time (simple moving average)
        current_avg = review_service.review_stats["average_review_time"]
        total_reviews = review_service.review_stats["total_reviews_completed"]
        new_avg = ((current_avg * (total_reviews - 1)) + review_time) / total_reviews
        review_service.review_stats["average_review_time"] = round(new_avg, 2)
    
    return {
        "message": "Review submitted successfully",
        "result_id": result_id,
        "action": action,
        "reviewer": current_user.email,
        "reviewed_at": result.reviewed_at
    }

@router.get("/tasks/{result_id}")
def get_review_task_details(
    result_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed information for a specific review task"""
    
    result = db.query(Result).filter(Result.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Get job and project info
    job = db.query(Job).filter(Job.id == result.job_id).first()
    project = db.query(Project).filter(Project.id == job.project_id).first() if job else None
    
    task_details = {
        "result_id": result.id,
        "filename": result.filename,
        "predicted_label": result.predicted_label,
        "confidence": round(result.confidence * 100, 2) if result.confidence else 0,
        "all_predictions": result.all_predictions,
        "processing_time": result.processing_time,
        "model_version": result.model_version,
        "created_at": result.created_at,
        "file_id": result.file_id,
        "job_info": {
            "job_id": job.id if job else None,
            "job_type": job.job_type if job else None,
            "model_name": job.model_name if job else None
        },
        "project_info": {
            "project_id": project.id if project else None,
            "project_name": project.name if project else None,
            "confidence_threshold": project.confidence_threshold if project else None
        },
        "review_history": {
            "reviewed": result.reviewed,
            "reviewed_by": result.reviewed_by,
            "reviewed_at": result.reviewed_at,
            "review_action": result.review_action,
            "ground_truth": result.ground_truth,
            "correction_reason": result.correction_reason
        }
    }
    
    return task_details

@router.get("/projects/{project_id}/stats")
def get_project_review_stats(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get review statistics for a project"""
    
    # Get project review data
    all_results = db.query(Result).join(Job).filter(
        Job.project_id == project_id
    ).all()
    
    reviewed_results = [r for r in all_results if r.reviewed]
    pending_results = [r for r in all_results if not r.reviewed and r.confidence and r.confidence < 0.9]
    
    # Calculate statistics
    total_items = len(all_results)
    reviewed_items = len(reviewed_results)
    pending_review = len(pending_results)
    
    # Review action breakdown
    approved = len([r for r in reviewed_results if r.review_action == "approve"])
    modified = len([r for r in reviewed_results if r.review_action == "modify"])
    rejected = len([r for r in reviewed_results if r.review_action == "reject"])
    
    # Calculate inter-annotator agreement
    agreement_rate = review_service.calculate_inter_annotator_agreement(db, project_id)
    
    # Confidence distribution
    confidences = [r.confidence for r in all_results if r.confidence]
    confidence_stats = {
        "average": round(statistics.mean(confidences), 3) if confidences else 0,
        "median": round(statistics.median(confidences), 3) if confidences else 0,
        "low_confidence_count": len([c for c in confidences if c < 0.7]),
        "high_confidence_count": len([c for c in confidences if c > 0.9])
    } if confidences else {}
    
    return {
        "project_id": project_id,
        "summary": {
            "total_items": total_items,
            "reviewed_items": reviewed_items,
            "pending_review": pending_review,
            "review_completion_rate": round((reviewed_items / max(1, total_items)) * 100, 2)
        },
        "review_actions": {
            "approved": approved,
            "modified": modified,
            "rejected": rejected
        },
        "quality_metrics": {
            "inter_annotator_agreement": agreement_rate,
            "average_review_time": review_service.review_stats["average_review_time"],
            "confidence_distribution": confidence_stats
        }
    }

@router.post("/projects/{project_id}/assign-tasks")
def assign_review_tasks(
    project_id: int,
    assignment_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Assign review tasks to reviewers (admin only)"""
    
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Extract assignment parameters
    reviewer_id = assignment_data.get("reviewer_id")
    batch_size = assignment_data.get("batch_size", 10)
    
    # Validate reviewer
    reviewer = db.query(User).filter(User.id == reviewer_id).first()
    if not reviewer or reviewer.role not in [UserRole.REVIEWER, UserRole.ADMIN]:
        raise HTTPException(status_code=400, detail="Invalid reviewer")
    
    # Assign tasks
    assigned_tasks = review_service.assign_review_tasks(db, project_id, reviewer_id, batch_size)
    
    return {
        "message": f"Assigned {len(assigned_tasks)} tasks to reviewer",
        "project_id": project_id,
        "reviewer": {
            "id": reviewer.id,
            "email": reviewer.email
        },
        "assigned_tasks": len(assigned_tasks),
        "task_ids": [task["result_id"] for task in assigned_tasks]
    }

@router.get("/dashboard")
def get_review_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get review dashboard data for the current user"""
    
    if current_user.role not in [UserRole.REVIEWER, UserRole.ADMIN]:
        raise HTTPException(status_code=403, detail="Reviewer access required")
    
    # Get user's review activity
    user_reviews = db.query(Result).filter(
        Result.reviewed_by == current_user.id,
        Result.reviewed_at >= datetime.utcnow() - timedelta(days=30)
    ).all()
    
    # Get pending assignments
    pending_assignments = db.query(Result).filter(
        Result.reviewed_by == current_user.id,
        Result.reviewed == False
    ).count()
    
    # Calculate performance metrics
    total_reviews = len(user_reviews)
    reviews_this_week = len([r for r in user_reviews if r.reviewed_at >= datetime.utcnow() - timedelta(days=7)])
    
    # Review time analysis
    review_times = []
    for result in user_reviews:
        if result.reviewed_at and result.created_at:
            time_diff = (result.reviewed_at - result.created_at).total_seconds()
            review_times.append(time_diff)
    
    avg_review_time = round(statistics.mean(review_times) / 60, 2) if review_times else 0  # in minutes
    
    return {
        "reviewer": {
            "id": current_user.id,
            "email": current_user.email,
            "role": current_user.role.value
        },
        "activity": {
            "total_reviews_30_days": total_reviews,
            "reviews_this_week": reviews_this_week,
            "pending_assignments": pending_assignments,
            "average_review_time_minutes": avg_review_time
        },
        "performance": {
            "reviews_per_day": round(reviews_this_week / 7, 1),
            "consistency_score": min(100, round((reviews_this_week / max(1, total_reviews)) * 100, 1))
        }
    }

@router.get("/tasks/pending-test")
def get_pending_review_tasks_test(
    project_id: int = Query(...),
    limit: int = Query(default=20, le=50)
):
    """Get pending review tasks for testing - no authentication required"""
    
    try:
        # Get mock review tasks using file storage
        tasks = review_storage.get_pending_tasks(project_id)
        
        # Apply limit
        limited_tasks = tasks[:limit]
        
        return {
            "tasks": limited_tasks,
            "total_pending": len(tasks),
            "project_id": project_id,
            "limit": limit,
            "note": "These are mock review tasks for Phase 1 testing"
        }
        
    except Exception as e:
        logger.error(f"Failed to get pending review tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get review tasks: {str(e)}")

@router.post("/tasks/{result_id}/review-test")
def submit_review_test(
    result_id: str,
    review_data: Dict[str, Any]
):
    """Submit review for testing - no authentication required"""
    
    try:
        action = review_data.get("action", "approve")
        notes = review_data.get("notes", "")
        
        # Submit review using file storage
        result = review_storage.submit_review(result_id, action, notes)
        
        return {
            "message": result["message"],
            "result_id": result_id,
            "action": action,
            "reviewed_at": result["timestamp"],
            "note": "Review submitted via Phase 1 testing interface"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit review: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}") 