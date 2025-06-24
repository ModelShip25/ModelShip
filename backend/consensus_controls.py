from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Float
from database import get_db
from database_base import Base
from typing import Dict, List, Any, Optional, Set
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import Counter, defaultdict
import statistics
import numpy as np

logger = logging.getLogger(__name__)

class ConsensusMethod(str, Enum):
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    EXPERT_OVERRIDE = "expert_override"
    UNANIMOUS = "unanimous"
    CONFIDENCE_WEIGHTED = "confidence_weighted"

class AnnotatorTier(str, Enum):
    NOVICE = "novice"          # New annotators, lower weight
    EXPERIENCED = "experienced" # Regular annotators
    EXPERT = "expert"          # Domain experts, higher weight
    GOLD_STANDARD = "gold_standard"  # Trusted annotators for ground truth

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_CONSENSUS = "awaiting_consensus"
    CONSENSUS_REACHED = "consensus_reached"
    EXPERT_REVIEW_NEEDED = "expert_review_needed"
    COMPLETED = "completed"
    DISPUTED = "disputed"

class ConsensusTask(Base):
    __tablename__ = "consensus_tasks"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    # Task configuration
    item_id = Column(String(255), nullable=False)
    item_type = Column(String(50))  # image, text, document
    item_data = Column(JSON)
    
    # Consensus configuration
    required_annotators = Column(Integer, default=2)
    consensus_method = Column(String(50), default=ConsensusMethod.MAJORITY_VOTE.value)
    confidence_threshold = Column(Float, default=0.8)
    
    # Assignment strategy
    annotator_tiers_required = Column(JSON)  # List of required tiers
    max_time_per_annotator_hours = Column(Integer, default=24)
    
    # Progress tracking
    status = Column(String(50), default=TaskStatus.PENDING.value)
    assigned_annotators = Column(JSON, default=list)
    completed_annotations = Column(Integer, default=0)
    
    # Results
    consensus_label = Column(String(255))
    consensus_confidence = Column(Float)
    agreement_score = Column(Float)
    annotation_results = Column(JSON, default=list)
    
    # Quality metrics
    inter_annotator_agreement = Column(Float)
    annotation_time_avg_minutes = Column(Float)
    difficulty_score = Column(Float)  # Based on disagreement
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_at = Column(DateTime)
    first_annotation_at = Column(DateTime)
    consensus_reached_at = Column(DateTime)
    completed_at = Column(DateTime)

class AnnotatorAssignment(Base):
    __tablename__ = "annotator_assignments"
    
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer, ForeignKey("consensus_tasks.id"), nullable=False)
    annotator_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Assignment details
    assigned_at = Column(DateTime, default=datetime.utcnow)
    due_at = Column(DateTime)
    annotator_tier = Column(String(50))
    assignment_priority = Column(Integer, default=1)  # 1=high, 3=low
    
    # Completion tracking
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    time_spent_minutes = Column(Float)
    
    # Annotation result
    annotation_label = Column(String(255))
    annotation_confidence = Column(Float)
    annotation_notes = Column(Text)
    annotation_metadata = Column(JSON)
    
    # Status
    status = Column(String(50), default="assigned")  # assigned, started, completed, expired
    is_active = Column(Boolean, default=True)

class AnnotatorProfile(Base):
    __tablename__ = "annotator_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Annotator classification
    tier = Column(String(50), default=AnnotatorTier.NOVICE.value)
    specializations = Column(JSON, default=list)  # Domain specializations
    languages = Column(JSON, default=lambda: ["English"])
    
    # Performance metrics
    total_annotations = Column(Integer, default=0)
    accuracy_score = Column(Float, default=0.0)
    consistency_score = Column(Float, default=0.0)
    average_time_per_task_minutes = Column(Float, default=0.0)
    
    # Reliability metrics
    agreement_with_consensus = Column(Float, default=0.0)
    agreement_with_experts = Column(Float, default=0.0)
    gold_standard_accuracy = Column(Float, default=0.0)
    
    # Availability
    max_concurrent_tasks = Column(Integer, default=5)
    preferred_task_types = Column(JSON, default=list)
    availability_hours = Column(JSON)  # Weekly schedule
    time_zone = Column(String(50))
    
    # Status
    is_active = Column(Boolean, default=True)
    last_active = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class ConsensusConfig:
    required_annotators: int = 2
    consensus_method: ConsensusMethod = ConsensusMethod.MAJORITY_VOTE
    confidence_threshold: float = 0.8
    annotator_tiers_required: List[AnnotatorTier] = None
    max_time_per_annotator_hours: int = 24
    enable_expert_escalation: bool = True
    require_unanimous_for_difficult: bool = False
    weight_by_annotator_performance: bool = True

class ConsensusControlService:
    def __init__(self):
        self.annotator_weights = {
            AnnotatorTier.NOVICE: 0.7,
            AnnotatorTier.EXPERIENCED: 1.0,
            AnnotatorTier.EXPERT: 1.5,
            AnnotatorTier.GOLD_STANDARD: 2.0
        }
    
    def create_consensus_task(
        self,
        db: Session,
        project_id: int,
        item_data: Dict[str, Any],
        config: ConsensusConfig,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Create a new consensus annotation task"""
        
        logger.info(f"Creating consensus task for project {project_id}")
        
        # Create task record
        task = ConsensusTask(
            project_id=project_id,
            item_id=item_data.get("id", f"item_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            item_type=item_data.get("type", "unknown"),
            item_data=item_data,
            required_annotators=config.required_annotators,
            consensus_method=config.consensus_method.value,
            confidence_threshold=config.confidence_threshold,
            annotator_tiers_required=[tier.value for tier in (config.annotator_tiers_required or [])],
            max_time_per_annotator_hours=config.max_time_per_annotator_hours
        )
        
        db.add(task)
        db.commit()
        db.refresh(task)
        
        # Assign annotators in background
        background_tasks.add_task(
            self._assign_annotators_async,
            db,
            task.id,
            config
        )
        
        return {
            "task_id": task.id,
            "item_id": task.item_id,
            "required_annotators": config.required_annotators,
            "consensus_method": config.consensus_method.value,
            "status": "created",
            "estimated_completion": datetime.utcnow() + timedelta(hours=config.max_time_per_annotator_hours)
        }
    
    async def _assign_annotators_async(
        self,
        db: Session,
        task_id: int,
        config: ConsensusConfig
    ):
        """Assign annotators to a consensus task"""
        
        try:
            task = db.query(ConsensusTask).filter(ConsensusTask.id == task_id).first()
            if not task:
                logger.error(f"Task {task_id} not found")
                return
            
            # Find suitable annotators
            suitable_annotators = self._find_suitable_annotators(
                db, task.project_id, config
            )
            
            if len(suitable_annotators) < config.required_annotators:
                logger.warning(f"Only {len(suitable_annotators)} suitable annotators found for task {task_id}")
            
            # Select annotators based on strategy
            selected_annotators = self._select_annotators(
                suitable_annotators, config.required_annotators, config
            )
            
            # Create assignments
            assignments = []
            due_time = datetime.utcnow() + timedelta(hours=config.max_time_per_annotator_hours)
            
            for i, annotator in enumerate(selected_annotators):
                assignment = AnnotatorAssignment(
                    task_id=task_id,
                    annotator_id=annotator["user_id"],
                    due_at=due_time,
                    annotator_tier=annotator["tier"],
                    assignment_priority=1 if annotator["tier"] in ["expert", "gold_standard"] else 2
                )
                
                db.add(assignment)
                assignments.append(assignment)
            
            # Update task
            task.status = TaskStatus.IN_PROGRESS.value
            task.assigned_annotators = [a["user_id"] for a in selected_annotators]
            task.assigned_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Assigned {len(selected_annotators)} annotators to task {task_id}")
            
            # Send notifications (would be implemented)
            self._notify_annotators(selected_annotators, task)
            
        except Exception as e:
            logger.error(f"Annotator assignment failed for task {task_id}: {e}")
    
    def _find_suitable_annotators(
        self,
        db: Session,
        project_id: int,
        config: ConsensusConfig
    ) -> List[Dict[str, Any]]:
        """Find annotators suitable for the task"""
        
        # Get annotator profiles
        profiles = db.query(AnnotatorProfile).filter(
            AnnotatorProfile.is_active == True
        ).all()
        
        suitable_annotators = []
        
        for profile in profiles:
            # Check tier requirements
            if config.annotator_tiers_required:
                if profile.tier not in [tier.value for tier in config.annotator_tiers_required]:
                    continue
            
            # Check availability (simplified)
            current_assignments = db.query(AnnotatorAssignment).filter(
                AnnotatorAssignment.annotator_id == profile.user_id,
                AnnotatorAssignment.status.in_(["assigned", "started"]),
                AnnotatorAssignment.is_active == True
            ).count()
            
            if current_assignments >= profile.max_concurrent_tasks:
                continue
            
            # Check performance thresholds
            if profile.accuracy_score < 0.7 and profile.total_annotations > 10:
                continue
            
            suitable_annotators.append({
                "user_id": profile.user_id,
                "tier": profile.tier,
                "accuracy_score": profile.accuracy_score,
                "consistency_score": profile.consistency_score,
                "total_annotations": profile.total_annotations,
                "current_load": current_assignments,
                "specializations": profile.specializations
            })
        
        return suitable_annotators
    
    def _select_annotators(
        self,
        suitable_annotators: List[Dict[str, Any]],
        required_count: int,
        config: ConsensusConfig
    ) -> List[Dict[str, Any]]:
        """Select the best annotators for the task"""
        
        if len(suitable_annotators) <= required_count:
            return suitable_annotators
        
        # Score annotators based on multiple criteria
        for annotator in suitable_annotators:
            score = 0.0
            
            # Performance score (40% weight)
            performance_score = (
                annotator["accuracy_score"] * 0.6 +
                annotator["consistency_score"] * 0.4
            )
            score += performance_score * 0.4
            
            # Experience score (30% weight)
            experience_score = min(annotator["total_annotations"] / 100, 1.0)
            score += experience_score * 0.3
            
            # Tier bonus (20% weight)
            tier_weights = {
                "novice": 0.5,
                "experienced": 0.7,
                "expert": 0.9,
                "gold_standard": 1.0
            }
            tier_score = tier_weights.get(annotator["tier"], 0.5)
            score += tier_score * 0.2
            
            # Availability score (10% weight)
            availability_score = 1.0 - (annotator["current_load"] / 10)
            score += max(0, availability_score) * 0.1
            
            annotator["selection_score"] = score
        
        # Sort by score and select top annotators
        suitable_annotators.sort(key=lambda x: x["selection_score"], reverse=True)
        
        # Ensure diversity in tiers if possible
        selected = []
        tiers_selected = set()
        
        # First pass: select one from each tier
        for annotator in suitable_annotators:
            if len(selected) >= required_count:
                break
            
            tier = annotator["tier"]
            if tier not in tiers_selected or len(selected) == 0:
                selected.append(annotator)
                tiers_selected.add(tier)
        
        # Second pass: fill remaining slots with best available
        for annotator in suitable_annotators:
            if len(selected) >= required_count:
                break
            
            if annotator not in selected:
                selected.append(annotator)
        
        return selected[:required_count]
    
    def _notify_annotators(
        self,
        annotators: List[Dict[str, Any]],
        task: ConsensusTask
    ):
        """Send notifications to assigned annotators"""
        
        # In a real implementation, this would send emails/notifications
        for annotator in annotators:
            logger.info(f"Notifying annotator {annotator['user_id']} about task {task.id}")
    
    def submit_annotation(
        self,
        db: Session,
        assignment_id: int,
        annotation_label: str,
        annotation_confidence: float,
        annotation_notes: Optional[str] = None,
        annotation_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Submit an annotation for a consensus task"""
        
        assignment = db.query(AnnotatorAssignment).filter(
            AnnotatorAssignment.id == assignment_id
        ).first()
        
        if not assignment:
            raise ValueError("Assignment not found")
        
        if assignment.status == "completed":
            raise ValueError("Assignment already completed")
        
        # Record start time if not already started
        if not assignment.started_at:
            assignment.started_at = datetime.utcnow()
            assignment.status = "started"
        
        # Calculate time spent
        time_spent = (datetime.utcnow() - assignment.started_at).total_seconds() / 60
        
        # Update assignment
        assignment.completed_at = datetime.utcnow()
        assignment.time_spent_minutes = time_spent
        assignment.annotation_label = annotation_label
        assignment.annotation_confidence = annotation_confidence
        assignment.annotation_notes = annotation_notes
        assignment.annotation_metadata = annotation_metadata or {}
        assignment.status = "completed"
        
        # Update task progress
        task = db.query(ConsensusTask).filter(
            ConsensusTask.id == assignment.task_id
        ).first()
        
        if task:
            task.completed_annotations += 1
            
            if not task.first_annotation_at:
                task.first_annotation_at = datetime.utcnow()
            
            # Check if ready for consensus calculation
            if task.completed_annotations >= task.required_annotators:
                task.status = TaskStatus.AWAITING_CONSENSUS.value
                
                # Calculate consensus in background
                self._calculate_consensus(db, task.id)
        
        db.commit()
        
        return {
            "assignment_id": assignment_id,
            "task_id": assignment.task_id,
            "annotation_submitted": True,
            "time_spent_minutes": time_spent,
            "task_progress": f"{task.completed_annotations}/{task.required_annotators}" if task else "unknown"
        }
    
    def _calculate_consensus(
        self,
        db: Session,
        task_id: int
    ) -> Dict[str, Any]:
        """Calculate consensus for a completed task"""
        
        task = db.query(ConsensusTask).filter(ConsensusTask.id == task_id).first()
        if not task:
            raise ValueError("Task not found")
        
        # Get all completed assignments
        assignments = db.query(AnnotatorAssignment).filter(
            AnnotatorAssignment.task_id == task_id,
            AnnotatorAssignment.status == "completed"
        ).all()
        
        if len(assignments) < task.required_annotators:
            logger.warning(f"Task {task_id} has insufficient annotations for consensus")
            return {"status": "insufficient_annotations"}
        
        # Extract annotations
        annotations = []
        for assignment in assignments:
            # Get annotator profile for weighting
            profile = db.query(AnnotatorProfile).filter(
                AnnotatorProfile.user_id == assignment.annotator_id
            ).first()
            
            weight = self.annotator_weights.get(
                AnnotatorTier(profile.tier) if profile else AnnotatorTier.EXPERIENCED,
                1.0
            )
            
            annotations.append({
                "label": assignment.annotation_label,
                "confidence": assignment.annotation_confidence,
                "weight": weight,
                "annotator_tier": profile.tier if profile else "experienced",
                "time_spent": assignment.time_spent_minutes
            })
        
        # Calculate consensus based on method
        consensus_method = ConsensusMethod(task.consensus_method)
        
        if consensus_method == ConsensusMethod.MAJORITY_VOTE:
            consensus_result = self._calculate_majority_vote(annotations)
        elif consensus_method == ConsensusMethod.WEIGHTED_AVERAGE:
            consensus_result = self._calculate_weighted_consensus(annotations)
        elif consensus_method == ConsensusMethod.CONFIDENCE_WEIGHTED:
            consensus_result = self._calculate_confidence_weighted_consensus(annotations)
        elif consensus_method == ConsensusMethod.UNANIMOUS:
            consensus_result = self._calculate_unanimous_consensus(annotations)
        else:
            consensus_result = self._calculate_majority_vote(annotations)  # Default
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(annotations)
        
        # Update task with results
        task.consensus_label = consensus_result["label"]
        task.consensus_confidence = consensus_result["confidence"]
        task.agreement_score = quality_metrics["agreement_score"]
        task.inter_annotator_agreement = quality_metrics["inter_annotator_agreement"]
        task.annotation_time_avg_minutes = quality_metrics["avg_time_minutes"]
        task.difficulty_score = quality_metrics["difficulty_score"]
        task.annotation_results = [
            {
                "label": ann["label"],
                "confidence": ann["confidence"],
                "tier": ann["annotator_tier"],
                "time_spent": ann["time_spent"]
            }
            for ann in annotations
        ]
        
        # Determine final status
        if consensus_result["needs_expert_review"]:
            task.status = TaskStatus.EXPERT_REVIEW_NEEDED.value
        elif quality_metrics["agreement_score"] < 0.6:
            task.status = TaskStatus.DISPUTED.value
        else:
            task.status = TaskStatus.CONSENSUS_REACHED.value
            task.consensus_reached_at = datetime.utcnow()
            
            # Mark as completed if no expert review needed
            if not consensus_result["needs_expert_review"]:
                task.completed_at = datetime.utcnow()
                task.status = TaskStatus.COMPLETED.value
        
        db.commit()
        
        # Update annotator performance metrics
        self._update_annotator_metrics(db, assignments, consensus_result["label"])
        
        return {
            "task_id": task_id,
            "consensus_label": consensus_result["label"],
            "consensus_confidence": consensus_result["confidence"],
            "agreement_score": quality_metrics["agreement_score"],
            "status": task.status,
            "needs_expert_review": consensus_result["needs_expert_review"]
        }
    
    def _calculate_majority_vote(
        self,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate consensus using majority vote"""
        
        labels = [ann["label"] for ann in annotations]
        label_counts = Counter(labels)
        
        # Get most common label
        most_common_label, count = label_counts.most_common(1)[0]
        
        # Calculate confidence based on agreement
        confidence = count / len(annotations)
        
        # Check if expert review needed (low confidence or tie)
        needs_expert_review = confidence < 0.6 or (
            len(label_counts) > 1 and 
            label_counts.most_common(2)[0][1] == label_counts.most_common(2)[1][1]
        )
        
        return {
            "label": most_common_label,
            "confidence": confidence,
            "needs_expert_review": needs_expert_review
        }
    
    def _calculate_weighted_consensus(
        self,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate consensus using weighted voting"""
        
        label_weights = defaultdict(float)
        total_weight = 0.0
        
        for ann in annotations:
            label_weights[ann["label"]] += ann["weight"]
            total_weight += ann["weight"]
        
        # Find label with highest weighted score
        best_label = max(label_weights.keys(), key=lambda x: label_weights[x])
        confidence = label_weights[best_label] / total_weight
        
        needs_expert_review = confidence < 0.7
        
        return {
            "label": best_label,
            "confidence": confidence,
            "needs_expert_review": needs_expert_review
        }
    
    def _calculate_confidence_weighted_consensus(
        self,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate consensus weighted by annotator confidence"""
        
        label_scores = defaultdict(list)
        
        for ann in annotations:
            label_scores[ann["label"]].append(ann["confidence"] * ann["weight"])
        
        # Calculate average confidence for each label
        label_avg_confidence = {}
        for label, scores in label_scores.items():
            label_avg_confidence[label] = sum(scores) / len(scores)
        
        # Select label with highest average confidence
        best_label = max(label_avg_confidence.keys(), key=lambda x: label_avg_confidence[x])
        confidence = label_avg_confidence[best_label]
        
        needs_expert_review = confidence < 0.8
        
        return {
            "label": best_label,
            "confidence": confidence,
            "needs_expert_review": needs_expert_review
        }
    
    def _calculate_unanimous_consensus(
        self,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate consensus requiring unanimous agreement"""
        
        labels = [ann["label"] for ann in annotations]
        unique_labels = set(labels)
        
        if len(unique_labels) == 1:
            # Unanimous agreement
            label = labels[0]
            confidence = min(ann["confidence"] for ann in annotations)
            needs_expert_review = False
        else:
            # No unanimous agreement - needs expert review
            most_common = Counter(labels).most_common(1)[0]
            label = most_common[0]
            confidence = most_common[1] / len(labels)
            needs_expert_review = True
        
        return {
            "label": label,
            "confidence": confidence,
            "needs_expert_review": needs_expert_review
        }
    
    def _calculate_quality_metrics(
        self,
        annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate quality metrics for the annotation set"""
        
        labels = [ann["label"] for ann in annotations]
        confidences = [ann["confidence"] for ann in annotations]
        times = [ann["time_spent"] for ann in annotations]
        
        # Agreement score (proportion of majority)
        label_counts = Counter(labels)
        most_common_count = label_counts.most_common(1)[0][1]
        agreement_score = most_common_count / len(labels)
        
        # Inter-annotator agreement (simplified Fleiss' kappa approximation)
        unique_labels = list(set(labels))
        n_annotators = len(labels)
        n_categories = len(unique_labels)
        
        if n_categories == 1:
            inter_annotator_agreement = 1.0
        else:
            # Simplified calculation
            observed_agreement = agreement_score
            expected_agreement = 1.0 / n_categories
            
            if expected_agreement == 1.0:
                inter_annotator_agreement = 1.0
            else:
                inter_annotator_agreement = (
                    (observed_agreement - expected_agreement) / 
                    (1.0 - expected_agreement)
                )
        
        # Average annotation time
        avg_time_minutes = statistics.mean(times) if times else 0.0
        
        # Difficulty score (higher = more difficult)
        confidence_variance = statistics.variance(confidences) if len(confidences) > 1 else 0.0
        time_variance = statistics.variance(times) if len(times) > 1 else 0.0
        
        difficulty_score = (
            (1.0 - agreement_score) * 0.5 +
            confidence_variance * 0.3 +
            min(time_variance / 100, 1.0) * 0.2
        )
        
        return {
            "agreement_score": agreement_score,
            "inter_annotator_agreement": max(0.0, inter_annotator_agreement),
            "avg_time_minutes": avg_time_minutes,
            "difficulty_score": difficulty_score
        }
    
    def _update_annotator_metrics(
        self,
        db: Session,
        assignments: List[AnnotatorAssignment],
        consensus_label: str
    ):
        """Update performance metrics for annotators"""
        
        for assignment in assignments:
            profile = db.query(AnnotatorProfile).filter(
                AnnotatorProfile.user_id == assignment.annotator_id
            ).first()
            
            if not profile:
                continue
            
            # Update basic counters
            profile.total_annotations += 1
            
            # Update accuracy (agreement with consensus)
            was_correct = assignment.annotation_label == consensus_label
            
            # Running average update
            if profile.total_annotations == 1:
                profile.accuracy_score = 1.0 if was_correct else 0.0
            else:
                # Exponential moving average with alpha = 0.1
                alpha = 0.1
                new_accuracy = 1.0 if was_correct else 0.0
                profile.accuracy_score = (
                    alpha * new_accuracy + (1 - alpha) * profile.accuracy_score
                )
            
            # Update average time
            if assignment.time_spent_minutes:
                if profile.total_annotations == 1:
                    profile.average_time_per_task_minutes = assignment.time_spent_minutes
                else:
                    alpha = 0.1
                    profile.average_time_per_task_minutes = (
                        alpha * assignment.time_spent_minutes + 
                        (1 - alpha) * profile.average_time_per_task_minutes
                    )
            
            # Update tier if performance warrants it
            self._update_annotator_tier(profile)
            
            profile.last_active = datetime.utcnow()
            profile.updated_at = datetime.utcnow()
        
        db.commit()
    
    def _update_annotator_tier(self, profile: AnnotatorProfile):
        """Update annotator tier based on performance"""
        
        if profile.total_annotations < 10:
            return  # Not enough data
        
        current_tier = AnnotatorTier(profile.tier)
        
        # Promotion criteria
        if (current_tier == AnnotatorTier.NOVICE and 
            profile.accuracy_score > 0.85 and 
            profile.total_annotations >= 25):
            profile.tier = AnnotatorTier.EXPERIENCED.value
            
        elif (current_tier == AnnotatorTier.EXPERIENCED and 
              profile.accuracy_score > 0.92 and 
              profile.total_annotations >= 100):
            profile.tier = AnnotatorTier.EXPERT.value
        
        # Demotion criteria (rare, but possible)
        elif (profile.accuracy_score < 0.7 and 
              profile.total_annotations >= 20):
            if current_tier != AnnotatorTier.NOVICE:
                profile.tier = AnnotatorTier.NOVICE.value
    
    def get_task_status(
        self,
        db: Session,
        task_id: int
    ) -> Dict[str, Any]:
        """Get detailed status of a consensus task"""
        
        task = db.query(ConsensusTask).filter(ConsensusTask.id == task_id).first()
        if not task:
            raise ValueError("Task not found")
        
        assignments = db.query(AnnotatorAssignment).filter(
            AnnotatorAssignment.task_id == task_id
        ).all()
        
        assignment_details = []
        for assignment in assignments:
            profile = db.query(AnnotatorProfile).filter(
                AnnotatorProfile.user_id == assignment.annotator_id
            ).first()
            
            assignment_details.append({
                "assignment_id": assignment.id,
                "annotator_id": assignment.annotator_id,
                "annotator_tier": assignment.annotator_tier,
                "status": assignment.status,
                "assigned_at": assignment.assigned_at.isoformat(),
                "due_at": assignment.due_at.isoformat() if assignment.due_at else None,
                "completed_at": assignment.completed_at.isoformat() if assignment.completed_at else None,
                "annotation_label": assignment.annotation_label,
                "annotation_confidence": assignment.annotation_confidence,
                "time_spent_minutes": assignment.time_spent_minutes,
                "annotator_accuracy": profile.accuracy_score if profile else None
            })
        
        return {
            "task_id": task.id,
            "item_id": task.item_id,
            "status": task.status,
            "progress": f"{task.completed_annotations}/{task.required_annotators}",
            "consensus_method": task.consensus_method,
            "consensus_label": task.consensus_label,
            "consensus_confidence": task.consensus_confidence,
            "agreement_score": task.agreement_score,
            "inter_annotator_agreement": task.inter_annotator_agreement,
            "difficulty_score": task.difficulty_score,
            "assignments": assignment_details,
            "created_at": task.created_at.isoformat(),
            "consensus_reached_at": task.consensus_reached_at.isoformat() if task.consensus_reached_at else None
        }

# Create service instance
consensus_service = ConsensusControlService()

# FastAPI Router
router = APIRouter(prefix="/api/consensus", tags=["consensus"])

@router.get("/status")
async def get_consensus_status():
    """Get consensus controls status"""
    return {
        "status": "available",
        "message": "Consensus controls endpoints",
        "features": ["multi_annotator", "agreement_metrics", "conflict_resolution"]
    }

@router.get("/metrics/{project_id}")
async def get_consensus_metrics(project_id: int):
    """Get consensus metrics for a project"""
    return {
        "project_id": project_id,
        "agreement_rate": 0.0,
        "conflicts": [],
        "message": "Consensus metrics - coming soon"
    }

@router.post("/create-task")
async def create_consensus_task(
    project_id: int,
    item_data: Dict[str, Any],
    required_annotators: int = 2,
    consensus_method: str = "majority_vote",
    confidence_threshold: float = 0.8,
    annotator_tiers_required: Optional[List[str]] = None,
    max_time_per_annotator_hours: int = 24,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Create a new consensus annotation task"""
    
    try:
        consensus_method_enum = ConsensusMethod(consensus_method.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid consensus method: {consensus_method}")
    
    if required_annotators < 1 or required_annotators > 5:
        raise HTTPException(status_code=400, detail="Required annotators must be between 1 and 5")
    
    # Parse annotator tiers
    tiers = []
    if annotator_tiers_required:
        for tier in annotator_tiers_required:
            try:
                tiers.append(AnnotatorTier(tier.lower()))
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid annotator tier: {tier}")
    
    config = ConsensusConfig(
        required_annotators=required_annotators,
        consensus_method=consensus_method_enum,
        confidence_threshold=confidence_threshold,
        annotator_tiers_required=tiers,
        max_time_per_annotator_hours=max_time_per_annotator_hours
    )
    
    try:
        result = consensus_service.create_consensus_task(
            db, project_id, item_data, config, background_tasks
        )
        
        return {
            "status": "success",
            "task": result
        }
        
    except Exception as e:
        logger.error(f"Consensus task creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/submit-annotation/{assignment_id}")
async def submit_annotation(
    assignment_id: int,
    annotation_label: str,
    annotation_confidence: float,
    annotation_notes: Optional[str] = None,
    annotation_metadata: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Submit an annotation for a consensus task"""
    
    if annotation_confidence < 0.0 or annotation_confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
    
    try:
        result = consensus_service.submit_annotation(
            db, assignment_id, annotation_label, annotation_confidence,
            annotation_notes, annotation_metadata
        )
        
        return {
            "status": "success",
            "submission": result
        }
        
    except Exception as e:
        logger.error(f"Annotation submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/task-status/{task_id}")
async def get_task_status(
    task_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed status of a consensus task"""
    
    try:
        status = consensus_service.get_task_status(db, task_id)
        
        return {
            "status": "success",
            "task_status": status
        }
        
    except Exception as e:
        logger.error(f"Task status retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/project-tasks/{project_id}")
async def get_project_consensus_tasks(
    project_id: int,
    status_filter: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get consensus tasks for a project"""
    
    query = db.query(ConsensusTask).filter(
        ConsensusTask.project_id == project_id
    )
    
    if status_filter:
        query = query.filter(ConsensusTask.status == status_filter)
    
    tasks = query.order_by(ConsensusTask.created_at.desc()).limit(limit).all()
    
    return {
        "status": "success",
        "project_id": project_id,
        "total_tasks": len(tasks),
        "tasks": [
            {
                "task_id": task.id,
                "item_id": task.item_id,
                "status": task.status,
                "progress": f"{task.completed_annotations}/{task.required_annotators}",
                "consensus_method": task.consensus_method,
                "consensus_label": task.consensus_label,
                "agreement_score": task.agreement_score,
                "created_at": task.created_at.isoformat()
            }
            for task in tasks
        ]
    }

@router.get("/annotator-performance/{annotator_id}")
async def get_annotator_performance(
    annotator_id: int,
    db: Session = Depends(get_db)
):
    """Get performance metrics for an annotator"""
    
    profile = db.query(AnnotatorProfile).filter(
        AnnotatorProfile.user_id == annotator_id
    ).first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Annotator profile not found")
    
    # Get recent assignments
    recent_assignments = db.query(AnnotatorAssignment).filter(
        AnnotatorAssignment.annotator_id == annotator_id,
        AnnotatorAssignment.status == "completed"
    ).order_by(AnnotatorAssignment.completed_at.desc()).limit(10).all()
    
    return {
        "status": "success",
        "annotator_id": annotator_id,
        "performance": {
            "tier": profile.tier,
            "total_annotations": profile.total_annotations,
            "accuracy_score": profile.accuracy_score,
            "consistency_score": profile.consistency_score,
            "average_time_per_task_minutes": profile.average_time_per_task_minutes,
            "agreement_with_consensus": profile.agreement_with_consensus,
            "specializations": profile.specializations,
            "is_active": profile.is_active,
            "last_active": profile.last_active.isoformat() if profile.last_active else None
        },
        "recent_assignments": [
            {
                "task_id": assignment.task_id,
                "completed_at": assignment.completed_at.isoformat(),
                "time_spent_minutes": assignment.time_spent_minutes,
                "annotation_confidence": assignment.annotation_confidence
            }
            for assignment in recent_assignments
        ]
    } 