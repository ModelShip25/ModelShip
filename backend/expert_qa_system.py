from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, JSON, and_, or_, func, desc
from database import get_db
from database_base import Base
from models import Project, Job, Result, User
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import statistics
import random

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/expert-qa", tags=["expert-qa"])

class ExpertTier(Enum):
    """Expert reviewer tiers"""
    JUNIOR = "junior"
    SENIOR = "senior" 
    SPECIALIST = "specialist"
    DOMAIN_EXPERT = "domain_expert"
    GOLD_STANDARD = "gold_standard"

class TaskComplexity(Enum):
    """Task complexity levels for expert routing"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT_ONLY = "expert_only"

@dataclass
class AccuracyBenchmark:
    """Industry accuracy benchmarks for comparison"""
    task_type: str
    industry_average: float
    industry_best: float
    our_current: float
    our_target: float
    confidence_interval: Tuple[float, float]

class ExpertReviewer(Base):
    __tablename__ = "expert_reviewers"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tier = Column(String(50), nullable=False)  # ExpertTier enum
    specializations = Column(JSON)  # List of domains/task types
    hourly_rate = Column(Float)
    
    # Performance metrics
    total_reviews = Column(Integer, default=0)
    accuracy_score = Column(Float, default=0.0)  # 0-100
    average_review_time = Column(Float, default=0.0)  # seconds
    reliability_score = Column(Float, default=100.0)  # 0-100
    
    # Availability
    is_active = Column(Boolean, default=True)
    max_concurrent_tasks = Column(Integer, default=10)
    current_tasks = Column(Integer, default=0)
    timezone = Column(String(50), default="UTC")
    
    # Quality control
    calibration_score = Column(Float, default=0.0)  # How well confidence matches accuracy
    bias_flags = Column(JSON, default=list)  # Any detected biases
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ExpertTask(Base):
    __tablename__ = "expert_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    result_id = Column(Integer, ForeignKey("results.id"), nullable=False)
    expert_id = Column(Integer, ForeignKey("expert_reviewers.id"), nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    # Task details
    task_type = Column(String(50), nullable=False)
    complexity_level = Column(String(20), nullable=False)
    priority = Column(Integer, default=5)  # 1-10 scale
    
    # Expert review
    expert_label = Column(String(255))
    expert_confidence = Column(Float)
    expert_annotations = Column(JSON)  # Bounding boxes, entities, etc.
    review_notes = Column(Text)
    
    # Quality metrics
    disagreement_with_model = Column(Boolean, default=False)
    review_time_seconds = Column(Float)
    difficulty_rating = Column(Integer)  # 1-5 scale from expert
    
    # Status tracking
    status = Column(String(20), default="assigned")  # assigned, in_progress, completed, escalated
    assigned_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    deadline = Column(DateTime)

class QualityAlert(Base):
    __tablename__ = "quality_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    alert_type = Column(String(50), nullable=False)  # accuracy_drop, bias_detected, expert_disagreement
    severity = Column(String(20), nullable=False)  # low, medium, high, critical
    
    message = Column(Text, nullable=False)
    details = Column(JSON)
    threshold_value = Column(Float)
    current_value = Column(Float)
    
    # Resolution
    status = Column(String(20), default="active")  # active, acknowledged, resolved
    acknowledged_by = Column(Integer, ForeignKey("users.id"))
    resolved_by = Column(Integer, ForeignKey("users.id"))
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)

class BiasDetection(Base):
    __tablename__ = "bias_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    # Bias details
    bias_type = Column(String(50), nullable=False)  # class_imbalance, demographic_skew, temporal_drift
    affected_classes = Column(JSON)
    confidence_score = Column(Float)  # How confident we are in this bias detection
    
    # Statistical measures
    effect_size = Column(Float)
    p_value = Column(Float)
    statistical_significance = Column(Boolean, default=False)
    
    # Recommendations
    recommendations = Column(JSON)
    severity_score = Column(Float)  # 0-100
    
    # Tracking
    first_detected = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = Column(String(20), default="active")

class ExpertQASystem:
    """Enterprise-grade expert-in-the-loop QA system"""
    
    def __init__(self):
        self.accuracy_benchmarks = {
            "image_classification": AccuracyBenchmark(
                task_type="image_classification",
                industry_average=0.85,
                industry_best=0.95,
                our_current=0.0,
                our_target=0.97,
                confidence_interval=(0.94, 0.99)
            ),
            "object_detection": AccuracyBenchmark(
                task_type="object_detection", 
                industry_average=0.75,
                industry_best=0.90,
                our_current=0.0,
                our_target=0.92,
                confidence_interval=(0.88, 0.95)
            ),
            "text_classification": AccuracyBenchmark(
                task_type="text_classification",
                industry_average=0.82,
                industry_best=0.92,
                our_current=0.0,
                our_target=0.94,
                confidence_interval=(0.91, 0.96)
            ),
            "ner": AccuracyBenchmark(
                task_type="ner",
                industry_average=0.78,
                industry_best=0.88,
                our_current=0.0,
                our_target=0.90,
                confidence_interval=(0.87, 0.93)
            )
        }
        
        # Expert routing thresholds
        self.complexity_thresholds = {
            TaskComplexity.SIMPLE: {"confidence_min": 0.9, "agreement_min": 0.95},
            TaskComplexity.MODERATE: {"confidence_min": 0.7, "agreement_min": 0.85},
            TaskComplexity.COMPLEX: {"confidence_min": 0.5, "agreement_min": 0.70},
            TaskComplexity.EXPERT_ONLY: {"confidence_min": 0.0, "agreement_min": 0.0}
        }
    
    async def route_to_expert(
        self,
        db: Session,
        result_id: int,
        complexity_level: TaskComplexity = None,
        required_specializations: List[str] = None
    ) -> Dict[str, Any]:
        """Intelligently route tasks to appropriate expert reviewers"""
        
        # Get the result and determine complexity if not provided
        result = db.query(Result).filter(Result.id == result_id).first()
        if not result:
            raise ValueError(f"Result {result_id} not found")
        
        if complexity_level is None:
            complexity_level = self._assess_task_complexity(result)
        
        # Find available experts
        available_experts = self._find_available_experts(
            db, 
            complexity_level, 
            required_specializations or []
        )
        
        if not available_experts:
            # No experts available - create alert
            await self._create_alert(
                db,
                result.job_id,  # Assuming job has project_id
                "expert_unavailable",
                "high",
                f"No experts available for task complexity: {complexity_level.value}"
            )
            return {"status": "queued", "message": "No experts currently available"}
        
        # Select best expert using scoring algorithm
        selected_expert = self._select_best_expert(available_experts, result)
        
        # Create expert task
        expert_task = ExpertTask(
            result_id=result_id,
            expert_id=selected_expert.id,
            project_id=self._get_project_id_from_result(db, result_id),
            task_type=result.task_type or "unknown",
            complexity_level=complexity_level.value,
            priority=self._calculate_priority(result),
            deadline=datetime.utcnow() + timedelta(hours=24)  # Default 24h deadline
        )
        
        db.add(expert_task)
        
        # Update expert's current task count
        selected_expert.current_tasks += 1
        
        db.commit()
        db.refresh(expert_task)
        
        # Send notification to expert (implement as needed)
        await self._notify_expert(selected_expert, expert_task)
        
        logger.info(f"Routed task {expert_task.id} to expert {selected_expert.id}")
        
        return {
            "status": "assigned",
            "expert_task_id": expert_task.id,
            "expert_id": selected_expert.id,
            "estimated_completion": expert_task.deadline.isoformat(),
            "complexity_level": complexity_level.value
        }
    
    def _assess_task_complexity(self, result: Result) -> TaskComplexity:
        """Assess task complexity based on model confidence and other factors"""
        
        confidence = result.confidence or 0.0
        
        # Get prediction distribution if available
        predictions = getattr(result, 'all_predictions', None)
        entropy = 0.0
        
        if predictions and isinstance(predictions, dict):
            probs = list(predictions.values())
            if probs:
                # Calculate entropy
                entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
        
        # Complexity scoring
        if confidence >= 0.9 and entropy < 0.5:
            return TaskComplexity.SIMPLE
        elif confidence >= 0.7 and entropy < 1.0:
            return TaskComplexity.MODERATE
        elif confidence >= 0.5:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.EXPERT_ONLY
    
    def _find_available_experts(
        self,
        db: Session,
        complexity_level: TaskComplexity,
        required_specializations: List[str]
    ) -> List[ExpertReviewer]:
        """Find available experts matching requirements"""
        
        # Base query for active experts with capacity
        query = db.query(ExpertReviewer).filter(
            ExpertReviewer.is_active == True,
            ExpertReviewer.current_tasks < ExpertReviewer.max_concurrent_tasks
        )
        
        # Filter by required specializations
        if required_specializations:
            # This would need proper JSON querying based on your DB
            query = query.filter(
                ExpertReviewer.specializations.contains(required_specializations)
            )
        
        # Filter by tier based on complexity
        if complexity_level == TaskComplexity.EXPERT_ONLY:
            query = query.filter(ExpertReviewer.tier.in_([
                ExpertTier.SPECIALIST.value,
                ExpertTier.DOMAIN_EXPERT.value,
                ExpertTier.GOLD_STANDARD.value
            ]))
        elif complexity_level == TaskComplexity.COMPLEX:
            query = query.filter(ExpertReviewer.tier.in_([
                ExpertTier.SENIOR.value,
                ExpertTier.SPECIALIST.value,
                ExpertTier.DOMAIN_EXPERT.value,
                ExpertTier.GOLD_STANDARD.value
            ]))
        
        return query.all()
    
    def _select_best_expert(self, experts: List[ExpertReviewer], result: Result) -> ExpertReviewer:
        """Select the best expert using multi-criteria scoring"""
        
        def score_expert(expert: ExpertReviewer) -> float:
            score = 0.0
            
            # Accuracy weight (40%)
            score += expert.accuracy_score * 0.4
            
            # Reliability weight (25%)
            score += expert.reliability_score * 0.25
            
            # Availability weight (20%) - favor less busy experts
            availability_score = max(0, 100 - (expert.current_tasks / expert.max_concurrent_tasks * 100))
            score += availability_score * 0.2
            
            # Speed weight (15%) - favor faster reviewers
            if expert.average_review_time > 0:
                # Normalize to 0-100 scale (lower time = higher score)
                speed_score = max(0, 100 - (expert.average_review_time / 3600 * 10))  # Hours to score
                score += speed_score * 0.15
            
            return score
        
        # Score all experts and return the best
        expert_scores = [(expert, score_expert(expert)) for expert in experts]
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        return expert_scores[0][0]
    
    def _calculate_priority(self, result: Result) -> int:
        """Calculate task priority (1-10 scale)"""
        
        priority = 5  # Default medium priority
        
        # Lower confidence = higher priority
        if result.confidence:
            if result.confidence < 0.3:
                priority += 3
            elif result.confidence < 0.6:
                priority += 2
            elif result.confidence < 0.8:
                priority += 1
        
        # Error status = highest priority
        if result.status == "error":
            priority = 10
        
        return min(10, priority)
    
    def _get_project_id_from_result(self, db: Session, result_id: int) -> int:
        """Get project ID from result via job"""
        result = db.query(Result).join(Job).filter(Result.id == result_id).first()
        return result.job.project_id if result and result.job else 1
    
    async def _notify_expert(self, expert: ExpertReviewer, task: ExpertTask):
        """Send notification to expert about new task"""
        # Implement notification logic (email, Slack, etc.)
        logger.info(f"Notifying expert {expert.id} about task {task.id}")
        pass
    
    async def monitor_accuracy_benchmarks(self, db: Session, project_id: int) -> Dict[str, Any]:
        """Monitor and compare accuracy against industry benchmarks"""
        
        # Get recent results for the project
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= start_date,
            Result.reviewed == True,
            Result.ground_truth.isnot(None)
        ).all()
        
        if not results:
            return {"status": "insufficient_data", "message": "Not enough reviewed data for benchmarking"}
        
        # Calculate current accuracy by task type
        task_accuracies = {}
        for result in results:
            task_type = result.task_type or "unknown"
            if task_type not in task_accuracies:
                task_accuracies[task_type] = {"correct": 0, "total": 0}
            
            if result.predicted_label == result.ground_truth:
                task_accuracies[task_type]["correct"] += 1
            task_accuracies[task_type]["total"] += 1
        
        # Compare against benchmarks
        benchmark_comparison = {}
        alerts_to_create = []
        
        for task_type, accuracy_data in task_accuracies.items():
            current_accuracy = accuracy_data["correct"] / accuracy_data["total"]
            
            if task_type in self.accuracy_benchmarks:
                benchmark = self.accuracy_benchmarks[task_type]
                benchmark.our_current = current_accuracy
                
                comparison = {
                    "task_type": task_type,
                    "current_accuracy": round(current_accuracy, 3),
                    "industry_average": benchmark.industry_average,
                    "industry_best": benchmark.industry_best,
                    "our_target": benchmark.our_target,
                    "vs_industry_avg": round(current_accuracy - benchmark.industry_average, 3),
                    "vs_industry_best": round(current_accuracy - benchmark.industry_best, 3),
                    "vs_our_target": round(current_accuracy - benchmark.our_target, 3),
                    "sample_size": accuracy_data["total"]
                }
                
                # Create alerts for significant underperformance
                if current_accuracy < benchmark.industry_average - 0.05:
                    alerts_to_create.append({
                        "type": "accuracy_below_industry",
                        "severity": "high",
                        "message": f"{task_type} accuracy ({current_accuracy:.1%}) is significantly below industry average ({benchmark.industry_average:.1%})"
                    })
                elif current_accuracy < benchmark.our_target - 0.03:
                    alerts_to_create.append({
                        "type": "accuracy_below_target", 
                        "severity": "medium",
                        "message": f"{task_type} accuracy ({current_accuracy:.1%}) is below target ({benchmark.our_target:.1%})"
                    })
                
                benchmark_comparison[task_type] = comparison
        
        # Create alerts
        for alert_data in alerts_to_create:
            await self._create_alert(
                db, 
                project_id, 
                alert_data["type"], 
                alert_data["severity"], 
                alert_data["message"]
            )
        
        return {
            "status": "success",
            "benchmark_comparison": benchmark_comparison,
            "overall_health": self._calculate_overall_health(benchmark_comparison),
            "recommendations": self._generate_accuracy_recommendations(benchmark_comparison),
            "alerts_created": len(alerts_to_create)
        }
    
    def _calculate_overall_health(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall project health score"""
        
        if not comparisons:
            return {"score": 0, "status": "unknown"}
        
        # Weight by sample size and calculate weighted average vs target
        total_weight = 0
        weighted_performance = 0
        
        for task_type, data in comparisons.items():
            weight = min(data["sample_size"], 1000)  # Cap weight at 1000 samples
            performance = max(0, 100 + (data["vs_our_target"] * 100))  # Convert to 0-100+ scale
            
            weighted_performance += performance * weight
            total_weight += weight
        
        overall_score = weighted_performance / total_weight if total_weight > 0 else 0
        
        # Determine status
        if overall_score >= 100:
            status = "excellent"
        elif overall_score >= 95:
            status = "good"
        elif overall_score >= 85:
            status = "acceptable"
        elif overall_score >= 70:
            status = "needs_improvement"
        else:
            status = "critical"
        
        return {
            "score": round(overall_score, 1),
            "status": status,
            "total_samples": total_weight
        }
    
    def _generate_accuracy_recommendations(self, comparisons: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations for improving accuracy"""
        
        recommendations = []
        
        for task_type, data in comparisons.items():
            if data["vs_industry_avg"] < -0.05:
                recommendations.append(
                    f"Critical: {task_type} accuracy is {abs(data['vs_industry_avg']):.1%} below industry average. "
                    f"Consider model retraining or expert review for this task type."
                )
            elif data["vs_our_target"] < -0.03:
                recommendations.append(
                    f"Consider increasing expert review rate for {task_type} tasks to improve accuracy."
                )
            elif data["sample_size"] < 100:
                recommendations.append(
                    f"Collect more labeled data for {task_type} to improve statistical confidence."
                )
        
        return recommendations
    
    async def detect_bias_patterns(self, db: Session, project_id: int) -> Dict[str, Any]:
        """Detect bias patterns in annotations and model predictions"""
        
        # Get all results for analysis
        results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed == True
        ).all()
        
        if len(results) < 50:  # Need sufficient data
            return {"status": "insufficient_data", "message": "Need at least 50 reviewed samples for bias detection"}
        
        detected_biases = []
        
        # 1. Class imbalance detection
        class_distribution = {}
        for result in results:
            label = result.ground_truth or result.predicted_label
            class_distribution[label] = class_distribution.get(label, 0) + 1
        
        total_samples = len(results)
        expected_frequency = total_samples / len(class_distribution)
        
        for class_label, count in class_distribution.items():
            frequency_ratio = count / expected_frequency
            if frequency_ratio < 0.5 or frequency_ratio > 2.0:  # More than 2x deviation
                detected_biases.append({
                    "type": "class_imbalance",
                    "affected_class": class_label,
                    "frequency_ratio": round(frequency_ratio, 2),
                    "sample_count": count,
                    "severity": "high" if frequency_ratio < 0.3 or frequency_ratio > 3.0 else "medium"
                })
        
        # 2. Temporal drift detection
        temporal_bias = self._detect_temporal_drift(results)
        if temporal_bias:
            detected_biases.extend(temporal_bias)
        
        # 3. Confidence calibration bias
        calibration_bias = self._detect_confidence_bias(results)
        if calibration_bias:
            detected_biases.append(calibration_bias)
        
        # Store detected biases
        for bias in detected_biases:
            bias_detection = BiasDetection(
                project_id=project_id,
                bias_type=bias["type"],
                affected_classes=[bias.get("affected_class")] if bias.get("affected_class") else [],
                confidence_score=0.8,  # Default confidence
                severity_score=80 if bias.get("severity") == "high" else 60,
                recommendations=self._generate_bias_recommendations(bias)
            )
            db.add(bias_detection)
        
        db.commit()
        
        return {
            "status": "success",
            "biases_detected": len(detected_biases),
            "bias_details": detected_biases,
            "recommendations": [self._generate_bias_recommendations(bias) for bias in detected_biases]
        }
    
    def _detect_temporal_drift(self, results: List[Result]) -> List[Dict[str, Any]]:
        """Detect temporal drift in model performance"""
        
        # Sort by creation time
        sorted_results = sorted(results, key=lambda r: r.created_at)
        
        # Split into early and recent periods
        split_point = len(sorted_results) // 2
        early_results = sorted_results[:split_point]
        recent_results = sorted_results[split_point:]
        
        # Calculate accuracy for each period
        early_correct = sum(1 for r in early_results if r.predicted_label == r.ground_truth)
        recent_correct = sum(1 for r in recent_results if r.predicted_label == r.ground_truth)
        
        early_accuracy = early_correct / len(early_results)
        recent_accuracy = recent_correct / len(recent_results)
        
        accuracy_change = recent_accuracy - early_accuracy
        
        if abs(accuracy_change) > 0.05:  # 5% change threshold
            return [{
                "type": "temporal_drift",
                "early_accuracy": round(early_accuracy, 3),
                "recent_accuracy": round(recent_accuracy, 3),
                "accuracy_change": round(accuracy_change, 3),
                "severity": "high" if abs(accuracy_change) > 0.1 else "medium"
            }]
        
        return []
    
    def _detect_confidence_bias(self, results: List[Result]) -> Optional[Dict[str, Any]]:
        """Detect confidence calibration bias"""
        
        # Group results by confidence bins
        confidence_bins = {}
        for result in results:
            if result.confidence is not None:
                bin_key = int(result.confidence * 10) / 10  # 0.1 bin size
                if bin_key not in confidence_bins:
                    confidence_bins[bin_key] = {"correct": 0, "total": 0}
                
                if result.predicted_label == result.ground_truth:
                    confidence_bins[bin_key]["correct"] += 1
                confidence_bins[bin_key]["total"] += 1
        
        # Calculate calibration error
        calibration_errors = []
        for confidence, data in confidence_bins.items():
            if data["total"] >= 5:  # Minimum samples per bin
                actual_accuracy = data["correct"] / data["total"]
                calibration_error = abs(confidence - actual_accuracy)
                calibration_errors.append(calibration_error)
        
        if calibration_errors:
            avg_calibration_error = statistics.mean(calibration_errors)
            if avg_calibration_error > 0.1:  # 10% threshold
                return {
                    "type": "confidence_bias",
                    "calibration_error": round(avg_calibration_error, 3),
                    "severity": "high" if avg_calibration_error > 0.2 else "medium"
                }
        
        return None
    
    def _generate_bias_recommendations(self, bias: Dict[str, Any]) -> List[str]:
        """Generate recommendations for addressing detected bias"""
        
        recommendations = []
        
        if bias["type"] == "class_imbalance":
            recommendations.extend([
                f"Collect more samples for underrepresented class: {bias['affected_class']}",
                "Consider data augmentation techniques for minority classes",
                "Implement class weighting in model training"
            ])
        elif bias["type"] == "temporal_drift":
            recommendations.extend([
                "Retrain model with recent data",
                "Implement continuous learning pipeline",
                "Monitor for distribution shift in input data"
            ])
        elif bias["type"] == "confidence_bias":
            recommendations.extend([
                "Recalibrate model confidence scores",
                "Implement temperature scaling for confidence calibration",
                "Adjust review thresholds based on calibration analysis"
            ])
        
        return recommendations
    
    async def _create_alert(
        self,
        db: Session,
        project_id: int,
        alert_type: str,
        severity: str,
        message: str,
        details: Dict[str, Any] = None
    ):
        """Create a quality alert"""
        
        alert = QualityAlert(
            project_id=project_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details or {}
        )
        
        db.add(alert)
        db.commit()
        
        # Send notifications for high severity alerts
        if severity in ["high", "critical"]:
            await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: QualityAlert):
        """Send alert notification to relevant stakeholders"""
        # Implement notification logic (email, Slack, webhooks)
        logger.warning(f"Quality Alert [{alert.severity.upper()}]: {alert.message}")
    
    async def simulate_cost_quality_tradeoff(
        self,
        db: Session,
        project_id: int,
        confidence_thresholds: List[float] = None
    ) -> Dict[str, Any]:
        """Simulate cost vs quality tradeoffs for different review thresholds"""
        
        if confidence_thresholds is None:
            confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        
        # Get recent results for analysis
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        
        results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= start_date,
            Result.confidence.isnot(None)
        ).all()
        
        if not results:
            return {"status": "insufficient_data", "message": "No data available for simulation"}
        
        simulations = []
        
        for threshold in confidence_thresholds:
            # Calculate metrics for this threshold
            auto_approved = [r for r in results if r.confidence >= threshold]
            needs_review = [r for r in results if r.confidence < threshold]
            
            # Estimate accuracy for auto-approved (based on confidence)
            auto_accuracy = np.mean([r.confidence for r in auto_approved]) if auto_approved else 0.0
            
            # Estimate human review accuracy (typically higher)
            human_accuracy = 0.95  # Assumption: human reviewers achieve 95% accuracy
            
            # Calculate overall expected accuracy
            total_items = len(results)
            if total_items > 0:
                overall_accuracy = (
                    len(auto_approved) * auto_accuracy + 
                    len(needs_review) * human_accuracy
                ) / total_items
            else:
                overall_accuracy = 0.0
            
            # Calculate costs (assuming $0.10 per human review)
            human_review_cost = len(needs_review) * 0.10
            total_cost_per_1k = (human_review_cost / total_items * 1000) if total_items > 0 else 0
            
            # Calculate expert routing needs
            expert_tasks = len([r for r in needs_review if r.confidence < 0.5])
            
            simulation = {
                "confidence_threshold": threshold,
                "auto_approved_pct": round(len(auto_approved) / total_items * 100, 1) if total_items > 0 else 0,
                "human_review_pct": round(len(needs_review) / total_items * 100, 1) if total_items > 0 else 0,
                "expert_review_pct": round(expert_tasks / total_items * 100, 1) if total_items > 0 else 0,
                "expected_accuracy": round(overall_accuracy, 3),
                "cost_per_1k_items": round(total_cost_per_1k, 2),
                "items_analyzed": total_items,
                "quality_grade": self._calculate_quality_grade(overall_accuracy)
            }
            
            simulations.append(simulation)
        
        # Find optimal balance point
        optimal_threshold = self._find_optimal_threshold(simulations)
        
        return {
            "status": "success",
            "simulations": simulations,
            "optimal_threshold": optimal_threshold,
            "recommendations": self._generate_threshold_recommendations(simulations),
            "current_performance": self._calculate_current_performance(results)
        }
    
    def _calculate_quality_grade(self, accuracy: float) -> str:
        """Calculate quality grade based on accuracy"""
        if accuracy >= 0.95:
            return "A+"
        elif accuracy >= 0.92:
            return "A"
        elif accuracy >= 0.88:
            return "B+"
        elif accuracy >= 0.85:
            return "B"
        elif accuracy >= 0.80:
            return "C+"
        elif accuracy >= 0.75:
            return "C"
        else:
            return "D"
    
    def _find_optimal_threshold(self, simulations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find optimal threshold based on cost-quality balance"""
        
        # Score each simulation (balance cost and quality)
        scored_sims = []
        for sim in simulations:
            # Normalize metrics to 0-100 scale
            accuracy_score = sim["expected_accuracy"] * 100
            cost_score = max(0, 100 - sim["cost_per_1k_items"])  # Lower cost = higher score
            
            # Weighted average (70% accuracy, 30% cost)
            overall_score = (accuracy_score * 0.7) + (cost_score * 0.3)
            
            scored_sims.append({
                "threshold": sim["confidence_threshold"],
                "score": overall_score,
                "accuracy": sim["expected_accuracy"],
                "cost": sim["cost_per_1k_items"]
            })
        
        # Find best scoring threshold
        best_sim = max(scored_sims, key=lambda x: x["score"])
        
        return {
            "threshold": best_sim["threshold"],
            "score": round(best_sim["score"], 1),
            "expected_accuracy": best_sim["accuracy"],
            "cost_per_1k": best_sim["cost"]
        }
    
    def _generate_threshold_recommendations(self, simulations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on simulation results"""
        
        recommendations = []
        
        # Find high accuracy simulations
        high_accuracy_sims = [s for s in simulations if s["expected_accuracy"] >= 0.95]
        
        if high_accuracy_sims:
            min_cost_sim = min(high_accuracy_sims, key=lambda x: x["cost_per_1k_items"])
            recommendations.append(
                f"For 95%+ accuracy: Use threshold {min_cost_sim['confidence_threshold']} "
                f"(${min_cost_sim['cost_per_1k_items']:.2f} per 1K items)"
            )
        
        # Find cost-effective options
        low_cost_sims = [s for s in simulations if s["cost_per_1k_items"] <= 10.0]
        
        if low_cost_sims:
            best_quality_low_cost = max(low_cost_sims, key=lambda x: x["expected_accuracy"])
            recommendations.append(
                f"For budget-conscious approach: Use threshold {best_quality_low_cost['confidence_threshold']} "
                f"({best_quality_low_cost['expected_accuracy']:.1%} accuracy, "
                f"${best_quality_low_cost['cost_per_1k_items']:.2f} per 1K items)"
            )
        
        return recommendations
    
    def _calculate_current_performance(self, results: List[Result]) -> Dict[str, Any]:
        """Calculate current system performance"""
        
        total_items = len(results)
        if total_items == 0:
            return {"message": "No data available"}
        
        reviewed_items = [r for r in results if r.reviewed]
        auto_approved_items = [r for r in results if not r.reviewed]
        
        # Calculate accuracy where we have ground truth
        accurate_predictions = [r for r in reviewed_items if r.predicted_label == r.ground_truth]
        current_accuracy = len(accurate_predictions) / len(reviewed_items) if reviewed_items else 0
        
        return {
            "total_items": total_items,
            "auto_approved_pct": round(len(auto_approved_items) / total_items * 100, 1),
            "human_reviewed_pct": round(len(reviewed_items) / total_items * 100, 1),
            "measured_accuracy": round(current_accuracy, 3),
            "avg_confidence": round(np.mean([r.confidence for r in results if r.confidence]), 3)
        }
    
    async def get_real_time_qa_dashboard(self, db: Session, project_id: int) -> Dict[str, Any]:
        """Get real-time QA dashboard with live metrics"""
        
        # Get data for last 24 hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        
        # Recent results
        recent_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= start_time
        ).all()
        
        # Expert tasks in progress
        active_expert_tasks = db.query(ExpertTask).filter(
            ExpertTask.project_id == project_id,
            ExpertTask.status.in_(["assigned", "in_progress"])
        ).all()
        
        # Quality alerts
        active_alerts = db.query(QualityAlert).filter(
            QualityAlert.project_id == project_id,
            QualityAlert.status == "active"
        ).all()
        
        # Calculate metrics
        dashboard_data = {
            "timestamp": end_time.isoformat(),
            "processing_metrics": self._calculate_processing_metrics(recent_results),
            "quality_metrics": self._calculate_quality_metrics(recent_results),
            "expert_queue": self._calculate_expert_queue_metrics(active_expert_tasks),
            "alerts": self._format_active_alerts(active_alerts),
            "trending": self._calculate_trending_metrics(db, project_id),
            "recommendations": self._generate_realtime_recommendations(recent_results, active_alerts)
        }
        
        return dashboard_data
    
    def _calculate_processing_metrics(self, results: List[Result]) -> Dict[str, Any]:
        """Calculate real-time processing metrics"""
        
        if not results:
            return {"message": "No recent activity"}
        
        total = len(results)
        completed = len([r for r in results if r.status == "success"])
        errors = len([r for r in results if r.status == "error"])
        processing = len([r for r in results if r.status == "processing"])
        
        avg_processing_time = np.mean([
            r.processing_time for r in results 
            if r.processing_time and r.status == "success"
        ]) if any(r.processing_time for r in results) else 0
        
        return {
            "total_items": total,
            "completed": completed,
            "errors": errors,
            "processing": processing,
            "success_rate": round(completed / total * 100, 1) if total > 0 else 0,
            "avg_processing_time": round(avg_processing_time, 2)
        }
    
    def _calculate_quality_metrics(self, results: List[Result]) -> Dict[str, Any]:
        """Calculate real-time quality metrics"""
        
        if not results:
            return {"message": "No recent activity"}
        
        # Confidence distribution
        confidences = [r.confidence for r in results if r.confidence is not None]
        
        if not confidences:
            return {"message": "No confidence data available"}
        
        # Review queue metrics
        low_confidence = len([r for r in results if r.confidence and r.confidence < 0.8])
        needs_expert = len([r for r in results if r.confidence and r.confidence < 0.5])
        
        return {
            "avg_confidence": round(np.mean(confidences), 3),
            "confidence_std": round(np.std(confidences), 3),
            "low_confidence_items": low_confidence,
            "expert_review_queue": needs_expert,
            "confidence_distribution": {
                "high (>0.9)": len([c for c in confidences if c > 0.9]),
                "medium (0.7-0.9)": len([c for c in confidences if 0.7 <= c <= 0.9]),
                "low (<0.7)": len([c for c in confidences if c < 0.7])
            }
        }
    
    def _calculate_expert_queue_metrics(self, expert_tasks: List[ExpertTask]) -> Dict[str, Any]:
        """Calculate expert queue metrics"""
        
        if not expert_tasks:
            return {"queue_size": 0, "avg_wait_time": 0}
        
        # Calculate wait times
        current_time = datetime.utcnow()
        wait_times = [
            (current_time - task.assigned_at).total_seconds() / 3600 
            for task in expert_tasks
        ]
        
        # Group by complexity
        complexity_counts = {}
        for task in expert_tasks:
            complexity = task.complexity_level
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        return {
            "queue_size": len(expert_tasks),
            "avg_wait_time_hours": round(np.mean(wait_times), 1),
            "max_wait_time_hours": round(max(wait_times), 1),
            "by_complexity": complexity_counts,
            "overdue_tasks": len([t for t in expert_tasks if t.deadline and current_time > t.deadline])
        }
    
    def _format_active_alerts(self, alerts: List[QualityAlert]) -> List[Dict[str, Any]]:
        """Format active alerts for dashboard"""
        
        return [
            {
                "id": alert.id,
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "age_hours": round((datetime.utcnow() - alert.created_at).total_seconds() / 3600, 1)
            }
            for alert in alerts
        ]
    
    def _calculate_trending_metrics(self, db: Session, project_id: int) -> Dict[str, Any]:
        """Calculate trending metrics (comparing last 24h vs previous 24h)"""
        
        now = datetime.utcnow()
        
        # Current period (last 24h)
        current_start = now - timedelta(hours=24)
        current_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= current_start
        ).all()
        
        # Previous period (24h before that)
        prev_start = now - timedelta(hours=48)
        prev_end = current_start
        prev_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.created_at >= prev_start,
            Result.created_at < prev_end
        ).all()
        
        # Calculate trends
        current_volume = len(current_results)
        prev_volume = len(prev_results)
        volume_change = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
        
        current_avg_conf = np.mean([r.confidence for r in current_results if r.confidence]) if current_results else 0
        prev_avg_conf = np.mean([r.confidence for r in prev_results if r.confidence]) if prev_results else 0
        confidence_change = current_avg_conf - prev_avg_conf
        
        return {
            "volume_change_pct": round(volume_change, 1),
            "confidence_change": round(confidence_change, 3),
            "current_volume": current_volume,
            "previous_volume": prev_volume
        }
    
    def _generate_realtime_recommendations(self, results: List[Result], alerts: List[QualityAlert]) -> List[str]:
        """Generate real-time recommendations"""
        
        recommendations = []
        
        # Check for high error rates
        if results:
            error_rate = len([r for r in results if r.status == "error"]) / len(results)
            if error_rate > 0.05:  # >5% error rate
                recommendations.append(f"High error rate detected ({error_rate:.1%}). Check model performance.")
        
        # Check for low confidence trends
        low_conf_items = [r for r in results if r.confidence and r.confidence < 0.7]
        if len(low_conf_items) > len(results) * 0.3:  # >30% low confidence
            recommendations.append("High volume of low-confidence predictions. Consider model retraining.")
        
        # Check alert severity
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        if critical_alerts:
            recommendations.append(f"{len(critical_alerts)} critical alerts require immediate attention.")
        
        return recommendations

# Initialize the service
expert_qa_service = ExpertQASystem()

# API Endpoints

@router.post("/register-expert")
async def register_expert_reviewer(
    user_id: int,
    tier: str,
    specializations: List[str],
    hourly_rate: Optional[float] = None,
    max_concurrent_tasks: int = 10,
    db: Session = Depends(get_db)
):
    """Register a new expert reviewer"""
    
    # Validate tier
    try:
        expert_tier = ExpertTier(tier)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier. Must be one of: {[t.value for t in ExpertTier]}")
    
    # Check if user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check if already registered as expert
    existing_expert = db.query(ExpertReviewer).filter(ExpertReviewer.user_id == user_id).first()
    if existing_expert:
        raise HTTPException(status_code=400, detail="User is already registered as an expert reviewer")
    
    expert = ExpertReviewer(
        user_id=user_id,
        tier=tier,
        specializations=specializations,
        hourly_rate=hourly_rate,
        max_concurrent_tasks=max_concurrent_tasks
    )
    
    db.add(expert)
    db.commit()
    db.refresh(expert)
    
    return {
        "status": "success",
        "expert_id": expert.id,
        "tier": tier,
        "specializations": specializations
    }

@router.post("/route-task/{result_id}")
async def route_task_to_expert(
    result_id: int,
    complexity_level: Optional[str] = None,
    required_specializations: Optional[List[str]] = None,
    db: Session = Depends(get_db)
):
    """Route a specific task to an expert reviewer"""
    
    complexity = None
    if complexity_level:
        try:
            complexity = TaskComplexity(complexity_level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid complexity level. Must be one of: {[c.value for c in TaskComplexity]}")
    
    result = await expert_qa_service.route_to_expert(
        db, 
        result_id, 
        complexity, 
        required_specializations or []
    )
    
    return result

@router.get("/benchmarks/{project_id}")
async def get_accuracy_benchmarks(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get accuracy benchmarks comparison for a project"""
    
    result = await expert_qa_service.monitor_accuracy_benchmarks(db, project_id)
    return result

@router.get("/bias-detection/{project_id}")
async def detect_bias(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Detect bias patterns in project annotations"""
    
    result = await expert_qa_service.detect_bias_patterns(db, project_id)
    return result

@router.get("/experts")
async def list_expert_reviewers(
    tier: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """List all expert reviewers"""
    
    query = db.query(ExpertReviewer)
    
    if tier:
        query = query.filter(ExpertReviewer.tier == tier)
    
    if active_only:
        query = query.filter(ExpertReviewer.is_active == True)
    
    experts = query.all()
    
    return {
        "experts": [
            {
                "id": expert.id,
                "user_id": expert.user_id,
                "tier": expert.tier,
                "specializations": expert.specializations,
                "accuracy_score": expert.accuracy_score,
                "total_reviews": expert.total_reviews,
                "current_tasks": expert.current_tasks,
                "max_concurrent_tasks": expert.max_concurrent_tasks,
                "is_active": expert.is_active
            }
            for expert in experts
        ]
    }

@router.get("/alerts/{project_id}")
async def get_quality_alerts(
    project_id: int,
    severity: Optional[str] = None,
    status: str = "active",
    db: Session = Depends(get_db)
):
    """Get quality alerts for a project"""
    
    query = db.query(QualityAlert).filter(
        QualityAlert.project_id == project_id,
        QualityAlert.status == status
    )
    
    if severity:
        query = query.filter(QualityAlert.severity == severity)
    
    alerts = query.order_by(desc(QualityAlert.created_at)).all()
    
    return {
        "alerts": [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "details": alert.details,
                "created_at": alert.created_at.isoformat()
            }
            for alert in alerts
        ]
    }

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Acknowledge a quality alert"""
    
    alert = db.query(QualityAlert).filter(QualityAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.status = "acknowledged"
    alert.acknowledged_by = user_id
    
    db.commit()
    
    return {"status": "acknowledged"}

@router.get("/expert-tasks/{expert_id}")
async def get_expert_tasks(
    expert_id: int,
    status: str = "assigned",
    limit: int = Query(20, le=100),
    db: Session = Depends(get_db)
):
    """Get tasks assigned to an expert"""
    
    tasks = db.query(ExpertTask).filter(
        ExpertTask.expert_id == expert_id,
        ExpertTask.status == status
    ).limit(limit).all()
    
    return {
        "tasks": [
            {
                "id": task.id,
                "result_id": task.result_id,
                "task_type": task.task_type,
                "complexity_level": task.complexity_level,
                "priority": task.priority,
                "assigned_at": task.assigned_at.isoformat(),
                "deadline": task.deadline.isoformat() if task.deadline else None,
                "status": task.status
            }
            for task in tasks
        ]
    }

@router.post("/expert-tasks/{task_id}/complete")
async def complete_expert_task(
    task_id: int,
    expert_label: str,
    expert_confidence: float,
    review_notes: Optional[str] = None,
    difficulty_rating: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Complete an expert review task"""
    
    task = db.query(ExpertTask).filter(ExpertTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task
    task.expert_label = expert_label
    task.expert_confidence = expert_confidence
    task.review_notes = review_notes
    task.difficulty_rating = difficulty_rating
    task.status = "completed"
    task.completed_at = datetime.utcnow()
    
    if task.started_at:
        task.review_time_seconds = (task.completed_at - task.started_at).total_seconds()
    
    # Update the original result
    result = db.query(Result).filter(Result.id == task.result_id).first()
    if result:
        result.ground_truth = expert_label
        result.reviewed = True
        result.reviewed_by = task.expert_id
        
        # Check for disagreement
        if result.predicted_label != expert_label:
            task.disagreement_with_model = True
    
    # Update expert statistics
    expert = db.query(ExpertReviewer).filter(ExpertReviewer.id == task.expert_id).first()
    if expert:
        expert.current_tasks = max(0, expert.current_tasks - 1)
        expert.total_reviews += 1
        
        # Update average review time
        if task.review_time_seconds:
            if expert.average_review_time == 0:
                expert.average_review_time = task.review_time_seconds
            else:
                expert.average_review_time = (expert.average_review_time + task.review_time_seconds) / 2
    
    db.commit()
    
    return {
        "status": "completed",
        "task_id": task_id,
        "expert_label": expert_label,
        "disagreement_detected": task.disagreement_with_model
    }

@router.get("/cost-quality-simulation/{project_id}")
async def simulate_cost_quality_tradeoff(
    project_id: int,
    thresholds: Optional[str] = Query(None, description="Comma-separated confidence thresholds (e.g., '0.5,0.7,0.9')"),
    db: Session = Depends(get_db)
):
    """Simulate cost vs quality tradeoffs for different confidence thresholds"""
    
    confidence_thresholds = None
    if thresholds:
        try:
            confidence_thresholds = [float(t.strip()) for t in thresholds.split(',')]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid threshold format. Use comma-separated floats.")
    
    result = await expert_qa_service.simulate_cost_quality_tradeoff(db, project_id, confidence_thresholds)
    return result

@router.get("/realtime-dashboard/{project_id}")
async def get_realtime_qa_dashboard(
    project_id: int,
    db: Session = Depends(get_db)
):
    """Get real-time QA dashboard with live metrics"""
    
    result = await expert_qa_service.get_real_time_qa_dashboard(db, project_id)
    return result

@router.post("/expert-tasks/{task_id}/start")
async def start_expert_task(
    task_id: int,
    db: Session = Depends(get_db)
):
    """Mark an expert task as started"""
    
    task = db.query(ExpertTask).filter(ExpertTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != "assigned":
        raise HTTPException(status_code=400, detail="Task is not in assigned status")
    
    task.status = "in_progress"
    task.started_at = datetime.utcnow()
    
    db.commit()
    
    return {"status": "started", "task_id": task_id}

@router.post("/automated-qa-trigger/{project_id}")
async def trigger_automated_qa_analysis(
    project_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger comprehensive automated QA analysis"""
    
    async def run_qa_analysis():
        """Background task to run comprehensive QA analysis"""
        try:
            # Run accuracy benchmarking
            benchmark_result = await expert_qa_service.monitor_accuracy_benchmarks(db, project_id)
            
            # Run bias detection
            bias_result = await expert_qa_service.detect_bias_patterns(db, project_id)
            
            # Run cost-quality simulation
            simulation_result = await expert_qa_service.simulate_cost_quality_tradeoff(db, project_id)
            
            logger.info(f"Automated QA analysis completed for project {project_id}")
            
        except Exception as e:
            logger.error(f"Automated QA analysis failed for project {project_id}: {str(e)}")
    
    background_tasks.add_task(run_qa_analysis)
    
    return {
        "status": "triggered",
        "message": "Comprehensive QA analysis started in background",
        "project_id": project_id
    }

@router.get("/qa-health-report/{project_id}")
async def get_qa_health_report(
    project_id: int,
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """Get comprehensive QA health report"""
    
    # Get multiple metrics in parallel
    benchmark_result = await expert_qa_service.monitor_accuracy_benchmarks(db, project_id)
    bias_result = await expert_qa_service.detect_bias_patterns(db, project_id)
    simulation_result = await expert_qa_service.simulate_cost_quality_tradeoff(db, project_id)
    dashboard_result = await expert_qa_service.get_real_time_qa_dashboard(db, project_id)
    
    # Get active alerts
    alerts = db.query(QualityAlert).filter(
        QualityAlert.project_id == project_id,
        QualityAlert.status == "active"
    ).all()
    
    # Calculate overall health score
    health_components = {
        "accuracy_health": benchmark_result.get("overall_health", {}).get("score", 0),
        "bias_health": 100 - (bias_result.get("biases_detected", 0) * 10),  # Penalize detected biases
        "cost_efficiency": simulation_result.get("optimal_threshold", {}).get("score", 0),
        "alert_health": max(0, 100 - len(alerts) * 20)  # Penalize active alerts
    }
    
    overall_health = sum(health_components.values()) / len(health_components)
    
    # Generate executive summary
    executive_summary = {
        "overall_health_score": round(overall_health, 1),
        "health_grade": expert_qa_service._calculate_quality_grade(overall_health / 100),
        "key_metrics": {
            "accuracy_vs_industry": benchmark_result.get("benchmark_comparison", {}),
            "detected_biases": bias_result.get("biases_detected", 0),
            "active_alerts": len(alerts),
            "cost_efficiency": simulation_result.get("optimal_threshold", {})
        },
        "top_recommendations": [
            *benchmark_result.get("recommendations", [])[:2],
            *bias_result.get("recommendations", [])[:2],
            *simulation_result.get("recommendations", [])[:2]
        ][:5]  # Top 5 recommendations
    }
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "project_id": project_id,
        "period_days": days,
        "executive_summary": executive_summary,
        "detailed_reports": {
            "accuracy_benchmarks": benchmark_result,
            "bias_analysis": bias_result,
            "cost_quality_simulation": simulation_result,
            "realtime_dashboard": dashboard_result
        },
        "health_components": health_components
    } 