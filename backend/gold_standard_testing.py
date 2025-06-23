from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, JSON
from database import get_db, Base
from models import Project, Job, Result, User
from typing import Dict, List, Any, Optional
import logging
import json
import random
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)

class GoldStandardSample(Base):
    __tablename__ = "gold_standard_samples"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    
    # Ground truth annotation
    correct_label = Column(String(255), nullable=False)
    correct_bounding_boxes = Column(JSON)  # For object detection
    correct_entities = Column(JSON)  # For NER
    
    # Metadata
    difficulty_level = Column(String(50), default="medium")  # easy, medium, hard
    sample_type = Column(String(50), nullable=False)  # image_classification, object_detection, text_ner, etc.
    description = Column(Text)
    
    # Usage tracking
    times_used = Column(Integer, default=0)
    last_used = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))

class GoldStandardTest(Base):
    __tablename__ = "gold_standard_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    sample_id = Column(Integer, ForeignKey("gold_standard_samples.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    result_id = Column(Integer, ForeignKey("results.id"))
    
    # Test results
    annotator_id = Column(Integer, ForeignKey("users.id"))
    predicted_label = Column(String(255))
    predicted_bounding_boxes = Column(JSON)
    predicted_entities = Column(JSON)
    confidence_score = Column(Float)
    
    # Scoring
    is_correct = Column(Boolean)
    accuracy_score = Column(Float)  # 0.0 to 1.0
    response_time_seconds = Column(Float)
    
    # Test metadata
    test_type = Column(String(50), default="blind")  # blind, calibration, drift_detection
    injected_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

class GoldStandardService:
    def __init__(self):
        self.injection_rates = {
            "low": 0.05,    # 5% of annotations
            "medium": 0.10, # 10% of annotations
            "high": 0.15    # 15% of annotations
        }
    
    def create_gold_sample(
        self,
        db: Session,
        project_id: int,
        filename: str,
        file_path: str,
        correct_label: str,
        sample_type: str,
        difficulty_level: str = "medium",
        description: str = "",
        user_id: int = 1,
        correct_bounding_boxes: Optional[List[Dict]] = None,
        correct_entities: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Create a new gold standard sample"""
        
        # Validate project exists
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Create gold standard sample
        gold_sample = GoldStandardSample(
            project_id=project_id,
            filename=filename,
            file_path=file_path,
            correct_label=correct_label,
            correct_bounding_boxes=correct_bounding_boxes,
            correct_entities=correct_entities,
            difficulty_level=difficulty_level,
            sample_type=sample_type,
            description=description,
            created_by=user_id
        )
        
        db.add(gold_sample)
        db.commit()
        db.refresh(gold_sample)
        
        logger.info(f"Created gold standard sample {gold_sample.id} for project {project_id}")
        
        return {
            "status": "created",
            "sample_id": gold_sample.id,
            "filename": filename,
            "correct_label": correct_label,
            "difficulty_level": difficulty_level,
            "sample_type": sample_type
        }
    
    def get_gold_samples(
        self,
        db: Session,
        project_id: int,
        sample_type: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get gold standard samples for a project"""
        
        query = db.query(GoldStandardSample).filter(
            GoldStandardSample.project_id == project_id
        )
        
        if sample_type:
            query = query.filter(GoldStandardSample.sample_type == sample_type)
        
        if difficulty_level:
            query = query.filter(GoldStandardSample.difficulty_level == difficulty_level)
        
        if active_only:
            query = query.filter(GoldStandardSample.is_active == True)
        
        samples = query.all()
        
        return [self._format_gold_sample(sample) for sample in samples]
    
    def _format_gold_sample(self, sample: GoldStandardSample) -> Dict[str, Any]:
        """Format gold sample for API response"""
        return {
            "id": sample.id,
            "filename": sample.filename,
            "file_path": sample.file_path,
            "correct_label": sample.correct_label,
            "difficulty_level": sample.difficulty_level,
            "sample_type": sample.sample_type,
            "description": sample.description,
            "times_used": sample.times_used,
            "last_used": sample.last_used.isoformat() if sample.last_used else None,
            "is_active": sample.is_active,
            "created_at": sample.created_at.isoformat()
        }
    
    async def inject_gold_samples(
        self,
        db: Session,
        project_id: int,
        job_id: int,
        injection_rate: str = "medium",
        total_samples: int = 100
    ) -> Dict[str, Any]:
        """Inject gold standard samples into a regular annotation job"""
        
        rate = self.injection_rates.get(injection_rate, 0.10)
        num_to_inject = max(1, int(total_samples * rate))
        
        # Get available gold samples for this project
        available_samples = db.query(GoldStandardSample).filter(
            GoldStandardSample.project_id == project_id,
            GoldStandardSample.is_active == True
        ).all()
        
        if not available_samples:
            raise ValueError("No gold standard samples available for injection")
        
        # Select samples to inject (random selection with difficulty balancing)
        samples_to_inject = self._select_balanced_samples(available_samples, num_to_inject)
        
        injected_tests = []
        for sample in samples_to_inject:
            # Create a test record
            test = GoldStandardTest(
                project_id=project_id,
                sample_id=sample.id,
                job_id=job_id,
                test_type="blind"
            )
            
            db.add(test)
            injected_tests.append(test)
            
            # Update sample usage
            sample.times_used += 1
            sample.last_used = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Injected {len(injected_tests)} gold samples into job {job_id}")
        
        return {
            "status": "injected",
            "job_id": job_id,
            "samples_injected": len(injected_tests),
            "injection_rate": rate,
            "total_samples": total_samples,
            "test_ids": [test.id for test in injected_tests]
        }
    
    def _select_balanced_samples(
        self,
        available_samples: List[GoldStandardSample],
        num_to_select: int
    ) -> List[GoldStandardSample]:
        """Select samples with balanced difficulty distribution"""
        
        # Group by difficulty
        by_difficulty = {"easy": [], "medium": [], "hard": []}
        for sample in available_samples:
            difficulty = sample.difficulty_level
            if difficulty in by_difficulty:
                by_difficulty[difficulty].append(sample)
        
        # Calculate distribution (40% medium, 30% easy, 30% hard)
        distribution = {
            "medium": int(num_to_select * 0.4),
            "easy": int(num_to_select * 0.3),
            "hard": int(num_to_select * 0.3)
        }
        
        # Adjust if we don't have enough samples in any category
        total_allocated = sum(distribution.values())
        if total_allocated < num_to_select:
            distribution["medium"] += num_to_select - total_allocated
        
        selected = []
        for difficulty, count in distribution.items():
            available = by_difficulty[difficulty]
            if available:
                # Prefer less-used samples
                available.sort(key=lambda s: (s.times_used, s.last_used or datetime.min))
                selected.extend(available[:count])
        
        # If we still need more, add any remaining samples
        if len(selected) < num_to_select:
            remaining = [s for s in available_samples if s not in selected]
            remaining.sort(key=lambda s: s.times_used)
            selected.extend(remaining[:num_to_select - len(selected)])
        
        return selected[:num_to_select]
    
    def score_annotation(
        self,
        db: Session,
        test_id: int,
        predicted_label: str,
        annotator_id: int,
        confidence_score: Optional[float] = None,
        predicted_bounding_boxes: Optional[List[Dict]] = None,
        predicted_entities: Optional[List[Dict]] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """Score an annotation against gold standard"""
        
        test = db.query(GoldStandardTest).filter(
            GoldStandardTest.id == test_id
        ).first()
        
        if not test:
            raise ValueError(f"Test {test_id} not found")
        
        # Get the gold standard sample
        gold_sample = db.query(GoldStandardSample).filter(
            GoldStandardSample.id == test.sample_id
        ).first()
        
        # Calculate accuracy score
        accuracy_score = self._calculate_accuracy(
            gold_sample, predicted_label, predicted_bounding_boxes, predicted_entities
        )
        
        # Update test record
        test.annotator_id = annotator_id
        test.predicted_label = predicted_label
        test.predicted_bounding_boxes = predicted_bounding_boxes
        test.predicted_entities = predicted_entities
        test.confidence_score = confidence_score
        test.is_correct = accuracy_score >= 0.8  # 80% threshold for "correct"
        test.accuracy_score = accuracy_score
        test.response_time_seconds = response_time
        test.completed_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "test_id": test_id,
            "accuracy_score": accuracy_score,
            "is_correct": test.is_correct,
            "expected_label": gold_sample.correct_label,
            "predicted_label": predicted_label,
            "confidence_score": confidence_score,
            "response_time": response_time
        }
    
    def _calculate_accuracy(
        self,
        gold_sample: GoldStandardSample,
        predicted_label: str,
        predicted_bounding_boxes: Optional[List[Dict]],
        predicted_entities: Optional[List[Dict]]
    ) -> float:
        """Calculate accuracy score for the prediction"""
        
        if gold_sample.sample_type == "image_classification":
            # Simple label matching
            return 1.0 if predicted_label == gold_sample.correct_label else 0.0
        
        elif gold_sample.sample_type == "object_detection":
            # Calculate IoU for bounding boxes
            if not predicted_bounding_boxes or not gold_sample.correct_bounding_boxes:
                return 0.0
            
            return self._calculate_detection_accuracy(
                gold_sample.correct_bounding_boxes,
                predicted_bounding_boxes
            )
        
        elif gold_sample.sample_type in ["text_ner", "text_classification"]:
            # Entity matching for NER, label matching for classification
            if gold_sample.sample_type == "text_classification":
                return 1.0 if predicted_label == gold_sample.correct_label else 0.0
            else:
                return self._calculate_ner_accuracy(
                    gold_sample.correct_entities or [],
                    predicted_entities or []
                )
        
        return 0.0
    
    def _calculate_detection_accuracy(
        self,
        correct_boxes: List[Dict],
        predicted_boxes: List[Dict]
    ) -> float:
        """Calculate mAP-style accuracy for object detection"""
        
        if not correct_boxes or not predicted_boxes:
            return 0.0
        
        # Simplified IoU calculation
        total_iou = 0.0
        matches = 0
        
        for correct_box in correct_boxes:
            best_iou = 0.0
            for pred_box in predicted_boxes:
                if correct_box.get("label") == pred_box.get("label"):
                    iou = self._calculate_iou(correct_box, pred_box)
                    best_iou = max(best_iou, iou)
            
            if best_iou > 0.5:  # IoU threshold
                total_iou += best_iou
                matches += 1
        
        return total_iou / len(correct_boxes) if correct_boxes else 0.0
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        
        x1_min, y1_min = box1.get("x", 0), box1.get("y", 0)
        x1_max, y1_max = x1_min + box1.get("width", 0), y1_min + box1.get("height", 0)
        
        x2_min, y2_min = box2.get("x", 0), box2.get("y", 0)
        x2_max, y2_max = x2_min + box2.get("width", 0), y2_min + box2.get("height", 0)
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_ner_accuracy(
        self,
        correct_entities: List[Dict],
        predicted_entities: List[Dict]
    ) -> float:
        """Calculate F1 score for named entity recognition"""
        
        if not correct_entities and not predicted_entities:
            return 1.0
        
        if not correct_entities or not predicted_entities:
            return 0.0
        
        # Convert to sets of (text, label, start, end) tuples
        correct_set = set()
        for entity in correct_entities:
            correct_set.add((
                entity.get("text", ""),
                entity.get("label", ""),
                entity.get("start", 0),
                entity.get("end", 0)
            ))
        
        predicted_set = set()
        for entity in predicted_entities:
            predicted_set.add((
                entity.get("text", ""),
                entity.get("label", ""),
                entity.get("start", 0),
                entity.get("end", 0)
            ))
        
        # Calculate precision, recall, F1
        true_positives = len(correct_set & predicted_set)
        false_positives = len(predicted_set - correct_set)
        false_negatives = len(correct_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1
    
    def get_annotator_scores(
        self,
        db: Session,
        project_id: int,
        days: int = 30,
        annotator_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get annotator performance on gold standard tests"""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        query = db.query(GoldStandardTest).filter(
            GoldStandardTest.project_id == project_id,
            GoldStandardTest.completed_at >= start_date,
            GoldStandardTest.completed_at <= end_date,
            GoldStandardTest.accuracy_score.isnot(None)
        )
        
        if annotator_id:
            query = query.filter(GoldStandardTest.annotator_id == annotator_id)
        
        tests = query.all()
        
        if not tests:
            return {
                "project_id": project_id,
                "period_days": days,
                "annotator_id": annotator_id,
                "total_tests": 0,
                "performance": {}
            }
        
        # Group by annotator
        by_annotator = {}
        for test in tests:
            aid = test.annotator_id
            if aid not in by_annotator:
                by_annotator[aid] = []
            by_annotator[aid].append(test)
        
        # Calculate performance metrics
        performance = {}
        for aid, annotator_tests in by_annotator.items():
            scores = [t.accuracy_score for t in annotator_tests]
            correct_count = sum(1 for t in annotator_tests if t.is_correct)
            
            # Get user info
            user = db.query(User).filter(User.id == aid).first()
            username = user.username if user else f"User_{aid}"
            
            performance[aid] = {
                "annotator_id": aid,
                "username": username,
                "total_tests": len(annotator_tests),
                "correct_annotations": correct_count,
                "accuracy_rate": correct_count / len(annotator_tests),
                "average_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
                "avg_response_time": sum(
                    t.response_time_seconds for t in annotator_tests if t.response_time_seconds
                ) / len([t for t in annotator_tests if t.response_time_seconds]) if any(
                    t.response_time_seconds for t in annotator_tests
                ) else None
            }
        
        return {
            "project_id": project_id,
            "period_days": days,
            "annotator_filter": annotator_id,
            "total_tests": len(tests),
            "annotators_evaluated": len(performance),
            "performance": performance
        }
    
    def detect_model_drift(
        self,
        db: Session,
        project_id: int,
        days: int = 7
    ) -> Dict[str, Any]:
        """Detect model performance drift using gold standard results"""
        
        end_date = datetime.utcnow()
        current_period_start = end_date - timedelta(days=days)
        previous_period_start = current_period_start - timedelta(days=days)
        
        # Get current period results
        current_tests = db.query(GoldStandardTest).filter(
            GoldStandardTest.project_id == project_id,
            GoldStandardTest.completed_at >= current_period_start,
            GoldStandardTest.completed_at <= end_date,
            GoldStandardTest.accuracy_score.isnot(None)
        ).all()
        
        # Get previous period results
        previous_tests = db.query(GoldStandardTest).filter(
            GoldStandardTest.project_id == project_id,
            GoldStandardTest.completed_at >= previous_period_start,
            GoldStandardTest.completed_at < current_period_start,
            GoldStandardTest.accuracy_score.isnot(None)
        ).all()
        
        if not current_tests or not previous_tests:
            return {
                "project_id": project_id,
                "drift_detected": False,
                "message": "Insufficient data for drift detection",
                "current_period_tests": len(current_tests),
                "previous_period_tests": len(previous_tests)
            }
        
        # Calculate performance metrics
        current_avg = sum(t.accuracy_score for t in current_tests) / len(current_tests)
        previous_avg = sum(t.accuracy_score for t in previous_tests) / len(previous_tests)
        
        performance_change = current_avg - previous_avg
        change_percentage = (performance_change / previous_avg) * 100 if previous_avg > 0 else 0
        
        # Drift detection thresholds
        drift_detected = abs(change_percentage) > 10  # 10% change threshold
        drift_severity = "critical" if abs(change_percentage) > 20 else "warning" if drift_detected else "normal"
        
        return {
            "project_id": project_id,
            "period_days": days,
            "drift_detected": drift_detected,
            "drift_severity": drift_severity,
            "current_period": {
                "tests": len(current_tests),
                "average_accuracy": round(current_avg, 3),
                "period_start": current_period_start.isoformat()
            },
            "previous_period": {
                "tests": len(previous_tests),
                "average_accuracy": round(previous_avg, 3),
                "period_start": previous_period_start.isoformat()
            },
            "change_analysis": {
                "absolute_change": round(performance_change, 3),
                "percentage_change": round(change_percentage, 2),
                "trend": "improving" if performance_change > 0 else "declining" if performance_change < 0 else "stable"
            },
            "recommendations": self._generate_drift_recommendations(drift_severity, change_percentage)
        }
    
    def _generate_drift_recommendations(self, severity: str, change_percentage: float) -> List[str]:
        """Generate recommendations based on drift analysis"""
        
        recommendations = []
        
        if severity == "critical":
            recommendations.extend([
                "Immediate model retraining recommended",
                "Review recent data quality and annotation guidelines",
                "Consider expanding gold standard test set"
            ])
        elif severity == "warning":
            if change_percentage < 0:
                recommendations.extend([
                    "Monitor model performance closely",
                    "Review annotation consistency",
                    "Consider incremental model updates"
                ])
            else:
                recommendations.extend([
                    "Performance improvement detected",
                    "Validate improvements with additional testing",
                    "Consider updating baseline metrics"
                ])
        else:
            recommendations.append("Model performance is stable")
        
        return recommendations

# Create service instance
gold_standard_service = GoldStandardService()

# FastAPI Router
router = APIRouter(prefix="/api/gold-standard", tags=["gold_standard"])

@router.post("/samples/create/{project_id}")
async def create_gold_sample(
    project_id: int,
    filename: str,
    file_path: str,
    correct_label: str,
    sample_type: str,
    difficulty_level: str = "medium",
    description: str = "",
    user_id: int = 1,
    db: Session = Depends(get_db)
):
    """Create a new gold standard sample"""
    try:
        result = gold_standard_service.create_gold_sample(
            db, project_id, filename, file_path, correct_label,
            sample_type, difficulty_level, description, user_id
        )
        return result
    except Exception as e:
        logger.error(f"Failed to create gold sample: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/samples/{project_id}")
async def get_gold_samples(
    project_id: int,
    sample_type: Optional[str] = None,
    difficulty_level: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get gold standard samples for a project"""
    try:
        samples = gold_standard_service.get_gold_samples(
            db, project_id, sample_type, difficulty_level
        )
        return {
            "status": "success",
            "project_id": project_id,
            "samples": samples,
            "total_samples": len(samples)
        }
    except Exception as e:
        logger.error(f"Failed to get gold samples: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/inject/{project_id}/{job_id}")
async def inject_gold_samples(
    project_id: int,
    job_id: int,
    injection_rate: str = "medium",
    total_samples: int = 100,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Inject gold standard samples into annotation job"""
    try:
        result = await gold_standard_service.inject_gold_samples(
            db, project_id, job_id, injection_rate, total_samples
        )
        return result
    except Exception as e:
        logger.error(f"Failed to inject gold samples: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/score/{test_id}")
async def score_annotation(
    test_id: int,
    predicted_label: str,
    annotator_id: int,
    confidence_score: Optional[float] = None,
    response_time: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """Score an annotation against gold standard"""
    try:
        result = gold_standard_service.score_annotation(
            db, test_id, predicted_label, annotator_id, confidence_score, None, None, response_time
        )
        return result
    except Exception as e:
        logger.error(f"Failed to score annotation: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/performance/{project_id}")
async def get_annotator_performance(
    project_id: int,
    days: int = 30,
    annotator_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get annotator performance on gold standard tests"""
    try:
        performance = gold_standard_service.get_annotator_scores(
            db, project_id, days, annotator_id
        )
        return {
            "status": "success",
            "performance": performance
        }
    except Exception as e:
        logger.error(f"Failed to get performance: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/drift-detection/{project_id}")
async def detect_model_drift(
    project_id: int,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Detect model performance drift"""
    try:
        drift_analysis = gold_standard_service.detect_model_drift(db, project_id, days)
        return {
            "status": "success",
            "drift_analysis": drift_analysis
        }
    except Exception as e:
        logger.error(f"Failed to detect drift: {e}")
        raise HTTPException(status_code=400, detail=str(e)) 