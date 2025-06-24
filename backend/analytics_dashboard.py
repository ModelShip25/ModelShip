from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
from database import get_db
from models import User, Project, Job, Result, File, Analytics, ProjectAssignment, UserRole
from auth import get_current_user, get_optional_user
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

router = APIRouter(prefix="/api/analytics", tags=["analytics_dashboard"])

logger = logging.getLogger(__name__)

class AnalyticsDashboard:
    """
    Comprehensive analytics dashboard for ModelShip platform
    Tracks performance, usage, quality, and business metrics
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_overview(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user analytics overview"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Basic user statistics
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"error": "User not found"}
        
        # Project statistics
        user_projects = self.db.query(Project).join(ProjectAssignment).filter(
            ProjectAssignment.user_id == user_id
        ).all()
        
        # Job statistics
        total_jobs = self.db.query(Job).filter(Job.user_id == user_id).count()
        completed_jobs = self.db.query(Job).filter(
            Job.user_id == user_id,
            Job.status == "completed"
        ).count()
        
        recent_jobs = self.db.query(Job).filter(
            Job.user_id == user_id,
            Job.created_at >= start_date
        ).count()
        
        # Results and accuracy
        total_results = self.db.query(Result).join(Job).filter(
            Job.user_id == user_id,
            Result.status == "success"
        ).count()
        
        reviewed_results = self.db.query(Result).filter(
            Result.reviewed_by == user_id
        ).count()
        
        # Accuracy calculation
        accurate_reviews = self.db.query(Result).filter(
            Result.reviewed_by == user_id,
            Result.predicted_label == Result.ground_truth,
            Result.ground_truth.isnot(None)
        ).count()
        
        review_accuracy = (accurate_reviews / reviewed_results * 100) if reviewed_results > 0 else 0
        
        # Credit usage
        credits_used = user.credits_remaining if user.credits_remaining else 0
        initial_credits = 100  # Default starting credits
        credits_spent = initial_credits - credits_used
        
        return {
            "user_info": {
                "user_id": user.id,
                "email": user.email,
                "role": user.role,
                "subscription_tier": user.subscription_tier,
                "member_since": user.created_at.isoformat(),
                "last_login": user.last_login_at.isoformat() if user.last_login_at else None
            },
            "project_stats": {
                "total_projects": len(user_projects),
                "active_projects": len([p for p in user_projects if p.status == "active"]),
                "owned_projects": len([p for p in user_projects if p.owner_id == user_id]),
                "project_roles": Counter([
                    self.db.query(ProjectAssignment).filter(
                        ProjectAssignment.project_id == p.id,
                        ProjectAssignment.user_id == user_id
                    ).first().role.value for p in user_projects
                ])
            },
            "job_performance": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "success_rate": round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1),
                "recent_jobs": recent_jobs,
                "avg_items_per_job": round(total_results / total_jobs if total_jobs > 0 else 0, 1)
            },
            "review_performance": {
                "total_reviews": reviewed_results,
                "review_accuracy": round(review_accuracy, 1),
                "reviews_this_period": self.db.query(Result).filter(
                    Result.reviewed_by == user_id,
                    Result.reviewed_at >= start_date
                ).count()
            },
            "resource_usage": {
                "credits_remaining": credits_used,
                "credits_spent": credits_spent,
                "total_items_processed": total_results
            }
        }
    
    def get_project_analytics(self, project_id: int, days: int = 30) -> Dict[str, Any]:
        """Get detailed analytics for a specific project"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        project = self.db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return {"error": "Project not found"}
        
        # Job statistics for this project
        project_jobs = self.db.query(Job).filter(Job.project_id == project_id).all()
        completed_jobs = [j for j in project_jobs if j.status == "completed"]
        
        # Results statistics
        all_results = self.db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.status == "success"
        ).all()
        
        reviewed_results = [r for r in all_results if r.reviewed]
        accurate_results = [r for r in reviewed_results 
                          if r.predicted_label == r.ground_truth and r.ground_truth]
        
        # Label distribution
        label_distribution = Counter([r.predicted_label for r in all_results if r.predicted_label])
        
        # Confidence analysis
        confidences = [r.confidence for r in all_results if r.confidence is not None]
        confidence_stats = {
            "mean": round(np.mean(confidences), 3) if confidences else 0,
            "std": round(np.std(confidences), 3) if confidences else 0,
            "min": round(min(confidences), 3) if confidences else 0,
            "max": round(max(confidences), 3) if confidences else 0
        }
        
        # Team performance
        team_assignments = self.db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id
        ).all()
        
        team_performance = []
        for assignment in team_assignments:
            user_reviews = self.db.query(Result).join(Job).filter(
                Job.project_id == project_id,
                Result.reviewed_by == assignment.user_id
            ).count()
            
            team_performance.append({
                "user_email": assignment.user.email,
                "role": assignment.role.value,
                "assigned_items": assignment.assigned_items,
                "completed_items": assignment.completed_items,
                "reviews_completed": user_reviews,
                "completion_rate": round(
                    (assignment.completed_items / assignment.assigned_items * 100) 
                    if assignment.assigned_items > 0 else 0, 1
                )
            })
        
        # Progress over time
        daily_progress = self._calculate_daily_progress(project_id, start_date, end_date)
        
        return {
            "project_info": {
                "project_id": project.id,
                "name": project.name,
                "type": project.project_type,
                "status": project.status.value,
                "created_at": project.created_at.isoformat(),
                "deadline": project.deadline.isoformat() if project.deadline else None
            },
            "progress_summary": {
                "total_items": project.total_items,
                "labeled_items": project.labeled_items,
                "reviewed_items": project.reviewed_items,
                "approved_items": project.approved_items,
                "completion_percentage": round(
                    (project.labeled_items / project.total_items * 100) 
                    if project.total_items > 0 else 0, 1
                ),
                "review_percentage": round(
                    (project.reviewed_items / project.labeled_items * 100) 
                    if project.labeled_items > 0 else 0, 1
                )
            },
            "quality_metrics": {
                "total_results": len(all_results),
                "reviewed_results": len(reviewed_results),
                "accurate_predictions": len(accurate_results),
                "accuracy_rate": round(
                    (len(accurate_results) / len(reviewed_results) * 100) 
                    if reviewed_results else 0, 1
                ),
                "automation_rate": round(
                    (project.approved_items / project.labeled_items * 100) 
                    if project.labeled_items > 0 else 0, 1
                ),
                "confidence_statistics": confidence_stats
            },
            "label_analysis": {
                "unique_labels": len(label_distribution),
                "most_common_labels": dict(label_distribution.most_common(10)),
                "label_distribution": dict(label_distribution)
            },
            "team_performance": team_performance,
            "timeline_analysis": daily_progress,
            "job_statistics": {
                "total_jobs": len(project_jobs),
                "completed_jobs": len(completed_jobs),
                "success_rate": round(
                    (len(completed_jobs) / len(project_jobs) * 100) 
                    if project_jobs else 0, 1
                )
            }
        }
    
    def get_platform_overview(self, days: int = 30) -> Dict[str, Any]:
        """Get platform-wide analytics overview"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # User statistics
        total_users = self.db.query(User).count()
        active_users = self.db.query(User).filter(
            User.last_login_at >= start_date
        ).count() if start_date else 0
        
        new_users = self.db.query(User).filter(
            User.created_at >= start_date
        ).count()
        
        # Project statistics
        total_projects = self.db.query(Project).count()
        active_projects = self.db.query(Project).filter(
            Project.status == "active"
        ).count()
        
        # Job and processing statistics
        total_jobs = self.db.query(Job).count()
        completed_jobs = self.db.query(Job).filter(
            Job.status == "completed"
        ).count()
        
        recent_jobs = self.db.query(Job).filter(
            Job.created_at >= start_date
        ).count()
        
        # Results and quality
        total_results = self.db.query(Result).filter(
            Result.status == "success"
        ).count()
        
        reviewed_results = self.db.query(Result).filter(
            Result.reviewed == True
        ).count()
        
        # Usage by project type
        project_types = self.db.query(
            Project.project_type,
            func.count(Project.id).label('count')
        ).group_by(Project.project_type).all()
        
        # Subscription distribution
        subscription_tiers = self.db.query(
            User.subscription_tier,
            func.count(User.id).label('count')
        ).group_by(User.subscription_tier).all()
        
        return {
            "platform_summary": {
                "total_users": total_users,
                "active_users": active_users,
                "new_users_this_period": new_users,
                "user_activation_rate": round((active_users / total_users * 100) if total_users > 0 else 0, 1)
            },
            "project_overview": {
                "total_projects": total_projects,
                "active_projects": active_projects,
                "project_types": {pt: count for pt, count in project_types},
                "avg_projects_per_user": round(total_projects / total_users if total_users > 0 else 0, 1)
            },
            "processing_metrics": {
                "total_jobs": total_jobs,
                "completed_jobs": completed_jobs,
                "success_rate": round((completed_jobs / total_jobs * 100) if total_jobs > 0 else 0, 1),
                "recent_activity": recent_jobs,
                "total_items_processed": total_results,
                "items_reviewed": reviewed_results,
                "review_coverage": round((reviewed_results / total_results * 100) if total_results > 0 else 0, 1)
            },
            "business_metrics": {
                "subscription_distribution": {tier: count for tier, count in subscription_tiers},
                "platform_utilization": round((active_projects / total_projects * 100) if total_projects > 0 else 0, 1)
            }
        }
    
    def _calculate_daily_progress(self, project_id: int, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Calculate daily progress for a project"""
        
        # Get results created each day
        daily_results = self.db.query(
            func.date(Result.created_at).label('date'),
            func.count(Result.id).label('count')
        ).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= start_date,
            Result.created_at <= end_date,
            Result.status == "success"
        ).group_by(func.date(Result.created_at)).all()
        
        # Get reviews completed each day
        daily_reviews = self.db.query(
            func.date(Result.reviewed_at).label('date'),
            func.count(Result.id).label('count')
        ).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed_at >= start_date,
            Result.reviewed_at <= end_date,
            Result.reviewed == True
        ).group_by(func.date(Result.reviewed_at)).all()
        
        # Combine data
        progress_data = {}
        
        for date, count in daily_results:
            progress_data[date] = {"labeled": count, "reviewed": 0}
        
        for date, count in daily_reviews:
            if date in progress_data:
                progress_data[date]["reviewed"] = count
            else:
                progress_data[date] = {"labeled": 0, "reviewed": count}
        
        # Format as list
        return [
            {
                "date": date.isoformat(),
                "items_labeled": data["labeled"],
                "items_reviewed": data["reviewed"]
            }
            for date, data in sorted(progress_data.items())
        ]

# Initialize analytics dashboard
def get_analytics_dashboard(db: Session = Depends(get_db)):
    return AnalyticsDashboard(db)

@router.get("/user-dashboard")
async def get_user_dashboard(
    current_user: Optional[User] = Depends(get_optional_user),
    dashboard: AnalyticsDashboard = Depends(get_analytics_dashboard),
    days: int = Query(30, ge=1, le=365)
):
    """Get personalized analytics dashboard for current user"""
    
    try:
        analytics_data = dashboard.get_user_overview(current_user.id, days)
        
        if "error" in analytics_data:
            raise HTTPException(status_code=404, detail=analytics_data["error"])
        
        return {
            "dashboard_type": "user_overview",
            "time_period": f"{days} days",
            "data": analytics_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User dashboard failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@router.get("/project-dashboard/{project_id}")
async def get_project_dashboard(
    project_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    dashboard: AnalyticsDashboard = Depends(get_analytics_dashboard),
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get detailed analytics dashboard for a specific project"""
    
    try:
        # Check if user has access to this project
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied to this project")
        
        analytics_data = dashboard.get_project_analytics(project_id, days)
        
        if "error" in analytics_data:
            raise HTTPException(status_code=404, detail=analytics_data["error"])
        
        return {
            "dashboard_type": "project_analytics",
            "project_id": project_id,
            "user_role": assignment.role.value,
            "time_period": f"{days} days",
            "data": analytics_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Project dashboard failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@router.get("/platform-overview")
async def get_platform_dashboard(
    current_user: Optional[User] = Depends(get_optional_user),
    dashboard: AnalyticsDashboard = Depends(get_analytics_dashboard),
    days: int = Query(30, ge=1, le=365)
):
    """Get platform-wide analytics overview (admin only)"""
    
    try:
        # Check if user has admin access
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required")
        
        analytics_data = dashboard.get_platform_overview(days)
        
        return {
            "dashboard_type": "platform_overview",
            "time_period": f"{days} days",
            "data": analytics_data,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Platform dashboard failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dashboard generation failed: {str(e)}")

@router.get("/cost-savings/{project_id}")
async def calculate_cost_savings(
    project_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
    manual_cost_per_label: float = Query(0.10, ge=0.01, le=10.0),
    auto_cost_per_label: float = Query(0.01, ge=0.001, le=1.0)
):
    """Calculate cost savings from automation for a project"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
        
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Calculate costs
        total_items = project.labeled_items
        auto_approved_items = project.approved_items
        manually_reviewed_items = project.reviewed_items
        
        # Cost calculation
        manual_only_cost = total_items * manual_cost_per_label
        hybrid_cost = (auto_approved_items * auto_cost_per_label + 
                      manually_reviewed_items * manual_cost_per_label)
        
        total_savings = manual_only_cost - hybrid_cost
        savings_percentage = (total_savings / manual_only_cost * 100) if manual_only_cost > 0 else 0
        
        # Time savings calculation (assuming 1 minute per manual label, 1 second per auto label)
        manual_time_hours = total_items / 60  # 1 minute per label
        auto_time_hours = (auto_approved_items / 3600 +  # 1 second per auto label
                          manually_reviewed_items / 60)   # 1 minute per manual review
        
        time_saved_hours = manual_time_hours - auto_time_hours
        
        return {
            "project_id": project_id,
            "cost_analysis": {
                "total_items_processed": total_items,
                "auto_approved_items": auto_approved_items,
                "manually_reviewed_items": manually_reviewed_items,
                "automation_rate": round((auto_approved_items / total_items * 100) if total_items > 0 else 0, 1)
            },
            "cost_comparison": {
                "manual_only_cost": round(manual_only_cost, 2),
                "hybrid_approach_cost": round(hybrid_cost, 2),
                "total_savings": round(total_savings, 2),
                "savings_percentage": round(savings_percentage, 1),
                "cost_per_label_manual": manual_cost_per_label,
                "cost_per_label_auto": auto_cost_per_label
            },
            "time_analysis": {
                "manual_approach_hours": round(manual_time_hours, 1),
                "hybrid_approach_hours": round(auto_time_hours, 1),
                "time_saved_hours": round(time_saved_hours, 1),
                "time_saved_days": round(time_saved_hours / 8, 1)  # 8-hour work day
            },
            "roi_metrics": {
                "efficiency_gain": f"{round(savings_percentage, 1)}%",
                "speed_improvement": f"{round((manual_time_hours / auto_time_hours) if auto_time_hours > 0 else 0, 1)}x faster",
                "break_even_point": round(hybrid_cost / (manual_cost_per_label - auto_cost_per_label), 0) if manual_cost_per_label > auto_cost_per_label else "Immediate"
            },
            "calculated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost savings calculation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Calculation failed: {str(e)}")

@router.get("/quality-metrics/{project_id}")
async def get_quality_metrics(
    project_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get detailed quality metrics for a project"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get results in time period
        results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= start_date,
            Result.status == "success"
        ).all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        
        # Calculate quality metrics
        total_results = len(results)
        reviewed_results = [r for r in results if r.reviewed]
        accurate_results = [r for r in reviewed_results 
                          if r.predicted_label == r.ground_truth and r.ground_truth]
        
        # Confidence analysis
        high_conf_results = [r for r in results if r.confidence and r.confidence > 0.9]
        low_conf_results = [r for r in results if r.confidence and r.confidence < 0.5]
        
        # Error analysis
        error_results = [r for r in reviewed_results 
                        if r.predicted_label != r.ground_truth and r.ground_truth]
        
        # Inter-annotator agreement (if multiple reviews exist)
        agreement_score = self._calculate_inter_annotator_agreement(db, project_id, start_date)
        
        return {
            "project_id": project_id,
            "time_period": f"{days} days",
            "quality_overview": {
                "total_predictions": total_results,
                "reviewed_predictions": len(reviewed_results),
                "accurate_predictions": len(accurate_results),
                "error_predictions": len(error_results),
                "review_coverage": round((len(reviewed_results) / total_results * 100) if total_results > 0 else 0, 1),
                "accuracy_rate": round((len(accurate_results) / len(reviewed_results) * 100) if reviewed_results else 0, 1)
            },
            "confidence_analysis": {
                "high_confidence_count": len(high_conf_results),
                "low_confidence_count": len(low_conf_results),
                "avg_confidence": round(np.mean([r.confidence for r in results if r.confidence]), 3) if results else 0,
                "confidence_distribution": self._get_confidence_distribution(results)
            },
            "error_analysis": {
                "total_errors": len(error_results),
                "error_rate": round((len(error_results) / len(reviewed_results) * 100) if reviewed_results else 0, 1),
                "common_error_patterns": self._analyze_error_patterns(error_results)
            },
            "agreement_metrics": agreement_score,
            "recommendations": self._generate_quality_recommendations(
                total_results, reviewed_results, accurate_results, error_results
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}")

def _calculate_inter_annotator_agreement(self, db: Session, project_id: int, start_date: datetime) -> Dict[str, Any]:
    """Calculate inter-annotator agreement scores"""
    # This is a simplified version - in practice, you'd need multiple annotations per item
    return {
        "agreement_score": 0.85,  # Placeholder - implement based on your review workflow
        "note": "Inter-annotator agreement calculation requires multiple reviews per item"
    }

def _get_confidence_distribution(self, results: List[Result]) -> Dict[str, int]:
    """Get distribution of confidence scores"""
    confidences = [r.confidence for r in results if r.confidence is not None]
    
    if not confidences:
        return {}
    
    # Create bins
    bins = {
        "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, 
        "0.6-0.8": 0, "0.8-1.0": 0
    }
    
    for conf in confidences:
        if conf < 0.2:
            bins["0.0-0.2"] += 1
        elif conf < 0.4:
            bins["0.2-0.4"] += 1
        elif conf < 0.6:
            bins["0.4-0.6"] += 1
        elif conf < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    
    return bins

def _analyze_error_patterns(self, error_results: List[Result]) -> Dict[str, Any]:
    """Analyze common error patterns"""
    if not error_results:
        return {"message": "No errors to analyze"}
    
    # Group errors by predicted -> actual label
    error_patterns = defaultdict(int)
    for result in error_results:
        pattern = f"{result.predicted_label} → {result.ground_truth}"
        error_patterns[pattern] += 1
    
    # Get most common error patterns
    common_errors = dict(Counter(error_patterns).most_common(5))
    
    return {
        "most_common_errors": common_errors,
        "total_error_patterns": len(error_patterns),
        "suggestion": "Focus review efforts on these common error patterns"
    }

def _generate_quality_recommendations(self, total_results: int, reviewed_results: List, 
                                     accurate_results: List, error_results: List) -> List[str]:
    """Generate quality improvement recommendations"""
    recommendations = []
    
    review_rate = len(reviewed_results) / total_results if total_results > 0 else 0
    accuracy_rate = len(accurate_results) / len(reviewed_results) if reviewed_results else 0
    error_rate = len(error_results) / len(reviewed_results) if reviewed_results else 0
    
    if review_rate < 0.1:
        recommendations.append("Increase review coverage - currently only {:.1%} of results are reviewed".format(review_rate))
    
    if accuracy_rate < 0.8:
        recommendations.append("Model accuracy is below 80% - consider retraining or adjusting confidence thresholds")
    
    if error_rate > 0.2:
        recommendations.append("High error rate detected - focus on error pattern analysis and model improvement")
    
    if not recommendations:
        recommendations.append("Quality metrics look good - maintain current review practices")
    
    return recommendations 