from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from database import get_db
from models import Project, Job, Result, User, Review
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Data class for annotation quality metrics"""
    total_annotations: int
    auto_approved: int
    human_reviewed: int
    rejected_annotations: int
    average_confidence: float
    inter_annotator_agreement: float
    processing_speed: float
    accuracy_rate: float

class AnnotationQualityDashboard:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
    
    def get_project_quality_metrics(self, db: Session, project_id: int, days: int = 7) -> QualityMetrics:
        """Get comprehensive quality metrics for a project"""
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all jobs for the project in date range
        jobs = db.query(Job).filter(
            Job.project_id == project_id,
            Job.created_at >= start_date,
            Job.created_at <= end_date
        ).all()
        
        if not jobs:
            return QualityMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        
        job_ids = [job.id for job in jobs]
        
        # Get all results for these jobs
        results = db.query(Result).filter(Result.job_id.in_(job_ids)).all()
        
        if not results:
            return QualityMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        
        # Calculate metrics
        total_annotations = len(results)
        auto_approved = len([r for r in results if not r.reviewed])
        human_reviewed = len([r for r in results if r.reviewed])
        rejected_annotations = len([r for r in results if r.reviewed and r.review_action == "reject"])
        
        # Average confidence
        confidences = [r.confidence for r in results if r.confidence is not None]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Inter-annotator agreement (simplified)
        inter_annotator_agreement = self._calculate_agreement(db, project_id, start_date, end_date)
        
        # Processing speed (annotations per hour)
        total_hours = max(1, (end_date - start_date).total_seconds() / 3600)
        processing_speed = total_annotations / total_hours
        
        # Accuracy rate (non-rejected / total reviewed)
        accuracy_rate = ((human_reviewed - rejected_annotations) / human_reviewed * 100) if human_reviewed > 0 else 100.0
        
        return QualityMetrics(
            total_annotations=total_annotations,
            auto_approved=auto_approved,
            human_reviewed=human_reviewed,
            rejected_annotations=rejected_annotations,
            average_confidence=round(average_confidence, 2),
            inter_annotator_agreement=inter_annotator_agreement,
            processing_speed=round(processing_speed, 2),
            accuracy_rate=round(accuracy_rate, 2)
        )
    
    def _calculate_agreement(self, db: Session, project_id: int, start_date: datetime, end_date: datetime) -> float:
        """Calculate inter-annotator agreement for the time period"""
        
        # Get results that have been reviewed by multiple people
        reviewed_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed == True,
            Result.created_at >= start_date,
            Result.created_at <= end_date
        ).all()
        
        if len(reviewed_results) < 2:
            return 0.0
        
        # Group by filename to find items reviewed by multiple people
        item_reviews = {}
        for result in reviewed_results:
            key = f"{result.job_id}_{result.filename}"
            if key not in item_reviews:
                item_reviews[key] = []
            item_reviews[key].append(result)
        
        # Calculate agreement for items with multiple reviews
        agreements = 0
        total_comparisons = 0
        
        for reviews in item_reviews.values():
            if len(reviews) > 1:
                for i in range(len(reviews)):
                    for j in range(i + 1, len(reviews)):
                        if reviews[i].ground_truth == reviews[j].ground_truth:
                            agreements += 1
                        total_comparisons += 1
        
        return round(agreements / max(1, total_comparisons), 3)
    
    def get_annotator_performance(self, db: Session, project_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance metrics for each annotator"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all reviews for the project
        reviews = db.query(Review).join(Job).filter(
            Job.project_id == project_id,
            Review.created_at >= start_date,
            Review.created_at <= end_date
        ).all()
        
        # Group by reviewer
        reviewer_stats = {}
        for review in reviews:
            reviewer_id = review.reviewer_id
            if reviewer_id not in reviewer_stats:
                reviewer_stats[reviewer_id] = {
                    "total_reviews": 0,
                    "approved": 0,
                    "rejected": 0,
                    "modified": 0,
                    "avg_time": 0,
                    "review_times": []
                }
            
            stats = reviewer_stats[reviewer_id]
            stats["total_reviews"] += 1
            
            if review.action == "approve":
                stats["approved"] += 1
            elif review.action == "reject":
                stats["rejected"] += 1
            elif review.action == "modify":
                stats["modified"] += 1
            
            # Calculate review time if available
            if review.created_at and review.updated_at:
                review_time = (review.updated_at - review.created_at).total_seconds()
                stats["review_times"].append(review_time)
        
        # Calculate averages and get user info
        annotator_performance = []
        for reviewer_id, stats in reviewer_stats.items():
            user = db.query(User).filter(User.id == reviewer_id).first()
            
            avg_review_time = sum(stats["review_times"]) / len(stats["review_times"]) if stats["review_times"] else 0
            
            performance = {
                "user_id": reviewer_id,
                "username": user.email if user else f"User_{reviewer_id}",
                "total_reviews": stats["total_reviews"],
                "approved_rate": round(stats["approved"] / stats["total_reviews"] * 100, 2),
                "rejected_rate": round(stats["rejected"] / stats["total_reviews"] * 100, 2),
                "modified_rate": round(stats["modified"] / stats["total_reviews"] * 100, 2),
                "avg_review_time_seconds": round(avg_review_time, 2),
                "reviews_per_day": round(stats["total_reviews"] / days, 2)
            }
            annotator_performance.append(performance)
        
        # Sort by total reviews (most active first)
        annotator_performance.sort(key=lambda x: x["total_reviews"], reverse=True)
        
        return annotator_performance
    
    def get_annotation_trends(self, db: Session, project_id: int, days: int = 30) -> Dict[str, Any]:
        """Get annotation trends over time"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily annotation counts
        daily_stats = {}
        current_date = start_date
        
        while current_date <= end_date:
            day_start = current_date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            
            # Count annotations for this day
            day_results = db.query(Result).join(Job).filter(
                Job.project_id == project_id,
                Result.created_at >= day_start,
                Result.created_at < day_end
            ).all()
            
            daily_stats[day_start.strftime("%Y-%m-%d")] = {
                "total_annotations": len(day_results),
                "auto_approved": len([r for r in day_results if not r.reviewed]),
                "human_reviewed": len([r for r in day_results if r.reviewed]),
                "average_confidence": round(
                    sum(r.confidence for r in day_results if r.confidence) / max(1, len(day_results)), 2
                )
            }
            
            current_date += timedelta(days=1)
        
        return {
            "daily_stats": daily_stats,
            "trend_analysis": self._analyze_trends(daily_stats)
        }
    
    def _analyze_trends(self, daily_stats: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze trends in the daily statistics"""
        
        if not daily_stats:
            return {"trend": "no_data", "change_rate": 0}
        
        dates = sorted(daily_stats.keys())
        if len(dates) < 2:
            return {"trend": "insufficient_data", "change_rate": 0}
        
        # Calculate trend for total annotations
        first_week = dates[:7] if len(dates) >= 7 else dates[:len(dates)//2]
        last_week = dates[-7:] if len(dates) >= 7 else dates[len(dates)//2:]
        
        first_week_avg = sum(daily_stats[date]["total_annotations"] for date in first_week) / len(first_week)
        last_week_avg = sum(daily_stats[date]["total_annotations"] for date in last_week) / len(last_week)
        
        if first_week_avg == 0:
            change_rate = 0
        else:
            change_rate = ((last_week_avg - first_week_avg) / first_week_avg) * 100
        
        if change_rate > 10:
            trend = "increasing"
        elif change_rate < -10:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "change_rate": round(change_rate, 2),
            "first_period_avg": round(first_week_avg, 2),
            "last_period_avg": round(last_week_avg, 2)
        }
    
    def get_quality_alerts(self, db: Session, project_id: int) -> List[Dict[str, Any]]:
        """Get quality alerts for the project"""
        
        alerts = []
        
        # Get recent metrics
        metrics = self.get_project_quality_metrics(db, project_id, days=1)
        
        # Low confidence alert
        if metrics.average_confidence < 70:
            alerts.append({
                "type": "low_confidence",
                "severity": "warning",
                "message": f"Average confidence is low: {metrics.average_confidence}%",
                "recommendation": "Review confidence thresholds or retrain models"
            })
        
        # High rejection rate alert
        rejection_rate = (metrics.rejected_annotations / max(1, metrics.human_reviewed)) * 100
        if rejection_rate > 20:
            alerts.append({
                "type": "high_rejection_rate",
                "severity": "critical",
                "message": f"High rejection rate: {rejection_rate:.1f}%",
                "recommendation": "Review annotation quality and model performance"
            })
        
        # Low agreement alert
        if metrics.inter_annotator_agreement < 0.7:
            alerts.append({
                "type": "low_agreement",
                "severity": "warning",
                "message": f"Low inter-annotator agreement: {metrics.inter_annotator_agreement}",
                "recommendation": "Provide additional training or clearer guidelines"
            })
        
        # Slow processing alert
        if metrics.processing_speed < 10:  # Less than 10 annotations per hour
            alerts.append({
                "type": "slow_processing",
                "severity": "info",
                "message": f"Processing speed is slow: {metrics.processing_speed} annotations/hour",
                "recommendation": "Consider optimizing workflow or adding more annotators"
            })
        
        return alerts

# Create service instance
quality_dashboard = AnnotationQualityDashboard()

# FastAPI Router
router = APIRouter(prefix="/api/quality", tags=["quality"])

@router.get("/metrics/{project_id}")
async def get_quality_metrics(
    project_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get quality metrics for a project"""
    try:
        metrics = quality_dashboard.get_project_quality_metrics(db, project_id, days)
        return {
            "status": "success",
            "project_id": project_id,
            "period_days": days,
            "metrics": {
                "total_annotations": metrics.total_annotations,
                "auto_approved": metrics.auto_approved,
                "human_reviewed": metrics.human_reviewed,
                "rejected_annotations": metrics.rejected_annotations,
                "average_confidence": metrics.average_confidence,
                "inter_annotator_agreement": metrics.inter_annotator_agreement,
                "processing_speed": metrics.processing_speed,
                "accuracy_rate": metrics.accuracy_rate
            }
        }
    except Exception as e:
        logger.error(f"Failed to get quality metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/annotators/{project_id}")
async def get_annotator_performance(
    project_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get annotator performance metrics"""
    try:
        performance = quality_dashboard.get_annotator_performance(db, project_id, days)
        return {
            "status": "success",
            "project_id": project_id,
            "period_days": days,
            "annotators": performance
        }
    except Exception as e:
        logger.error(f"Failed to get annotator performance: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/trends/{project_id}")
async def get_annotation_trends(
    project_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get annotation trends over time"""
    try:
        trends = quality_dashboard.get_annotation_trends(db, project_id, days)
        return {
            "status": "success",
            "project_id": project_id,
            "period_days": days,
            "trends": trends
        }
    except Exception as e:
        logger.error(f"Failed to get annotation trends: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/alerts/{project_id}")
async def get_quality_alerts(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get quality alerts for a project"""
    try:
        alerts = quality_dashboard.get_quality_alerts(db, project_id)
        return {
            "status": "success",
            "project_id": project_id,
            "alerts": alerts,
            "alert_count": len(alerts)
        }
    except Exception as e:
        logger.error(f"Failed to get quality alerts: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboard/{project_id}")
async def get_complete_dashboard(
    project_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get complete dashboard data for a project"""
    try:
        metrics = quality_dashboard.get_project_quality_metrics(db, project_id, days)
        annotators = quality_dashboard.get_annotator_performance(db, project_id, days * 4)  # Longer period for annotators
        trends = quality_dashboard.get_annotation_trends(db, project_id, days)
        alerts = quality_dashboard.get_quality_alerts(db, project_id)
        
        return {
            "status": "success",
            "project_id": project_id,
            "generated_at": datetime.now().isoformat(),
            "dashboard": {
                "metrics": {
                    "total_annotations": metrics.total_annotations,
                    "auto_approved": metrics.auto_approved,
                    "human_reviewed": metrics.human_reviewed,
                    "rejected_annotations": metrics.rejected_annotations,
                    "average_confidence": metrics.average_confidence,
                    "inter_annotator_agreement": metrics.inter_annotator_agreement,
                    "processing_speed": metrics.processing_speed,
                    "accuracy_rate": metrics.accuracy_rate
                },
                "top_annotators": annotators[:5],  # Top 5 annotators
                "trends": trends,
                "alerts": alerts,
                "summary": {
                    "health_score": self._calculate_health_score(metrics, alerts),
                    "recommendations": self._generate_recommendations(metrics, alerts)
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get complete dashboard: {e}")
        raise HTTPException(status_code=400, detail=str(e))

def _calculate_health_score(self, metrics: QualityMetrics, alerts: List[Dict]) -> int:
    """Calculate overall project health score (0-100)"""
    score = 100
    
    # Deduct points for issues
    if metrics.average_confidence < 70:
        score -= 20
    elif metrics.average_confidence < 80:
        score -= 10
    
    if metrics.accuracy_rate < 80:
        score -= 25
    elif metrics.accuracy_rate < 90:
        score -= 15
    
    if metrics.inter_annotator_agreement < 0.7:
        score -= 20
    elif metrics.inter_annotator_agreement < 0.8:
        score -= 10
    
    # Deduct for alerts
    critical_alerts = len([a for a in alerts if a["severity"] == "critical"])
    warning_alerts = len([a for a in alerts if a["severity"] == "warning"])
    
    score -= critical_alerts * 15
    score -= warning_alerts * 5
    
    return max(0, score)

def _generate_recommendations(self, metrics: QualityMetrics, alerts: List[Dict]) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if metrics.average_confidence < 80:
        recommendations.append("Consider retraining models with more diverse data")
    
    if metrics.accuracy_rate < 90:
        recommendations.append("Review annotation guidelines and provide additional training")
    
    if metrics.inter_annotator_agreement < 0.8:
        recommendations.append("Implement regular calibration sessions for annotators")
    
    if metrics.processing_speed < 20:
        recommendations.append("Optimize annotation workflow or increase team size")
    
    if len(alerts) > 3:
        recommendations.append("Address critical quality issues before scaling annotation volume")
    
    if not recommendations:
        recommendations.append("Quality metrics look good! Consider expanding annotation capacity")
    
    return recommendations 