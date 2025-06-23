from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from models import User, Project, Job, Result, File
from auth import get_current_user
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from collections import Counter
import math
import random

router = APIRouter(prefix="/api/active-learning", tags=["active_learning"])

logger = logging.getLogger(__name__)

class ActiveLearningEngine:
    """
    Active Learning Engine for intelligent sample selection
    Implements multiple sampling strategies for optimal model improvement
    """
    
    def __init__(self):
        self.sampling_strategies = {
            "uncertainty": self._uncertainty_sampling,
            "margin": self._margin_sampling, 
            "entropy": self._entropy_sampling,
            "diverse": self._diversity_sampling,
            "disagreement": self._disagreement_sampling,
            "random": self._random_sampling
        }
    
    def _uncertainty_sampling(self, results: List[Result], n_samples: int) -> List[int]:
        """
        Select samples with lowest confidence scores
        Most uncertain predictions need human review
        """
        # Sort by confidence (ascending - lowest first)
        sorted_results = sorted(results, key=lambda x: x.confidence if x.confidence else 0)
        
        # Return indices of most uncertain samples
        return [results.index(result) for result in sorted_results[:n_samples]]
    
    def _margin_sampling(self, results: List[Result], n_samples: int) -> List[int]:
        """
        Select samples with smallest margin between top 2 predictions
        Requires all_predictions field with top-k predictions
        """
        margin_scores = []
        
        for i, result in enumerate(results):
            if result.all_predictions and isinstance(result.all_predictions, list):
                predictions = result.all_predictions
                if len(predictions) >= 2:
                    # Margin = difference between top 2 confidence scores
                    margin = predictions[0].get('confidence', 0) - predictions[1].get('confidence', 0)
                    margin_scores.append((i, margin))
                else:
                    # If only one prediction, use uncertainty
                    margin_scores.append((i, 1 - result.confidence if result.confidence else 1))
            else:
                # Fallback to uncertainty
                margin_scores.append((i, 1 - result.confidence if result.confidence else 1))
        
        # Sort by margin (ascending - smallest margins first)
        margin_scores.sort(key=lambda x: x[1])
        
        return [idx for idx, _ in margin_scores[:n_samples]]
    
    def _entropy_sampling(self, results: List[Result], n_samples: int) -> List[int]:
        """
        Select samples with highest prediction entropy
        High entropy indicates model uncertainty across all classes
        """
        entropy_scores = []
        
        for i, result in enumerate(results):
            if result.all_predictions and isinstance(result.all_predictions, list):
                predictions = result.all_predictions
                
                # Calculate entropy
                confidences = [pred.get('confidence', 0) for pred in predictions]
                
                # Normalize to sum to 1 (probability distribution)
                total = sum(confidences)
                if total > 0:
                    probs = [conf / total for conf in confidences]
                    # Calculate entropy: -Σ(p * log(p))
                    entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
                else:
                    entropy = 0
                
                entropy_scores.append((i, entropy))
            else:
                # Fallback: use inverse confidence as entropy proxy
                conf = result.confidence if result.confidence else 0.5
                entropy = -conf * math.log(conf + 1e-10) - (1-conf) * math.log(1-conf + 1e-10)
                entropy_scores.append((i, entropy))
        
        # Sort by entropy (descending - highest entropy first)
        entropy_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in entropy_scores[:n_samples]]
    
    def _diversity_sampling(self, results: List[Result], n_samples: int) -> List[int]:
        """
        Select diverse samples to ensure balanced coverage
        Aims for representative sampling across different classes
        """
        # Group results by predicted label
        label_groups = {}
        for i, result in enumerate(results):
            label = result.predicted_label or "unknown"
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(i)
        
        # Calculate samples per class for balanced selection
        n_classes = len(label_groups)
        if n_classes == 0:
            return []
        
        base_samples_per_class = n_samples // n_classes
        remaining_samples = n_samples % n_classes
        
        selected_indices = []
        
        # Select samples from each class
        for label, indices in label_groups.items():
            class_samples = base_samples_per_class
            if remaining_samples > 0:
                class_samples += 1
                remaining_samples -= 1
            
            # For this class, select most uncertain samples
            class_results = [results[i] for i in indices]
            uncertain_indices = self._uncertainty_sampling(class_results, min(class_samples, len(class_results)))
            
            # Map back to original indices
            selected_indices.extend([indices[i] for i in uncertain_indices])
        
        return selected_indices[:n_samples]
    
    def _disagreement_sampling(self, results: List[Result], n_samples: int) -> List[int]:
        """
        Select samples where different models would disagree
        Useful when multiple models are available
        """
        # For now, simulate disagreement using prediction spread
        disagreement_scores = []
        
        for i, result in enumerate(results):
            if result.all_predictions and isinstance(result.all_predictions, list):
                predictions = result.all_predictions
                
                if len(predictions) >= 2:
                    # Calculate standard deviation of confidence scores
                    confidences = [pred.get('confidence', 0) for pred in predictions]
                    if len(confidences) > 1:
                        mean_conf = np.mean(confidences)
                        std_conf = np.std(confidences)
                        disagreement = std_conf
                    else:
                        disagreement = 0
                else:
                    disagreement = 1 - result.confidence if result.confidence else 1
                
                disagreement_scores.append((i, disagreement))
            else:
                # Fallback to uncertainty
                disagreement_scores.append((i, 1 - result.confidence if result.confidence else 1))
        
        # Sort by disagreement (descending - highest disagreement first)
        disagreement_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [idx for idx, _ in disagreement_scores[:n_samples]]
    
    def _random_sampling(self, results: List[Result], n_samples: int) -> List[int]:
        """
        Select samples randomly
        Useful for initial exploration or when no specific strategy is available
        """
        return random.sample(range(len(results)), n_samples)
    
    def select_samples_for_review(
        self, 
        results: List[Result], 
        strategy: str = "uncertainty",
        n_samples: int = 10,
        exclude_reviewed: bool = True
    ) -> List[int]:
        """
        Select samples for human review using specified strategy
        
        Args:
            results: List of Result objects
            strategy: Sampling strategy ('uncertainty', 'margin', 'entropy', 'diverse', 'disagreement')
            n_samples: Number of samples to select
            exclude_reviewed: Whether to exclude already reviewed samples
        
        Returns:
            List of indices of selected samples
        """
        if not results:
            return []
        
        # Filter out reviewed samples if requested
        if exclude_reviewed:
            filtered_results = [r for r in results if not r.reviewed]
            if not filtered_results:
                return []
            
            # Map indices back to original list
            index_mapping = {i: results.index(r) for i, r in enumerate(filtered_results)}
            results = filtered_results
        else:
            index_mapping = {i: i for i in range(len(results))}
        
        # Apply sampling strategy
        if strategy not in self.sampling_strategies:
            strategy = "uncertainty"  # Default fallback
        
        selected_indices = self.sampling_strategies[strategy](results, n_samples)
        
        # Map back to original indices if filtering was applied
        if exclude_reviewed:
            selected_indices = [index_mapping[i] for i in selected_indices if i in index_mapping]
        
        return selected_indices

    def calculate_information_gain(self, results_before: List[Result], results_after: List[Result]) -> Dict[str, float]:
        """
        Calculate information gain from active learning iteration
        Measures improvement in model uncertainty and accuracy
        """
        if not results_before or not results_after:
            return {"error": "Insufficient data for calculation"}
        
        # Calculate average confidence before and after
        conf_before = np.mean([r.confidence for r in results_before if r.confidence is not None])
        conf_after = np.mean([r.confidence for r in results_after if r.confidence is not None])
        
        # Calculate accuracy if ground truth available
        accurate_before = sum(1 for r in results_before 
                            if r.reviewed and r.predicted_label == r.ground_truth)
        total_reviewed_before = sum(1 for r in results_before if r.reviewed)
        
        accurate_after = sum(1 for r in results_after 
                           if r.reviewed and r.predicted_label == r.ground_truth)
        total_reviewed_after = sum(1 for r in results_after if r.reviewed)
        
        accuracy_before = accurate_before / total_reviewed_before if total_reviewed_before > 0 else 0
        accuracy_after = accurate_after / total_reviewed_after if total_reviewed_after > 0 else 0
        
        return {
            "confidence_improvement": conf_after - conf_before,
            "accuracy_improvement": accuracy_after - accuracy_before,
            "samples_reviewed_before": total_reviewed_before,
            "samples_reviewed_after": total_reviewed_after,
            "information_gain_score": (conf_after - conf_before) + (accuracy_after - accuracy_before)
        }

# Initialize global active learning engine
active_learning_engine = ActiveLearningEngine()

@router.post("/suggest-samples/{job_id}")
async def suggest_samples_for_review(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    strategy: str = Query("uncertainty", description="Sampling strategy"),
    n_samples: int = Query(10, ge=1, le=100),
    exclude_reviewed: bool = Query(True),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_confidence: float = Query(1.0, ge=0.0, le=1.0)
):
    """
    Suggest samples for human review using active learning
    """
    
    try:
        # Verify job ownership
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == current_user.id
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get results with confidence filtering
        results_query = db.query(Result).filter(
            Result.job_id == job_id,
            Result.status == "success"
        )
        
        if min_confidence > 0 or max_confidence < 1:
            results_query = results_query.filter(
                Result.confidence >= min_confidence,
                Result.confidence <= max_confidence
            )
        
        results = results_query.all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found for active learning")
        
        # Select samples using active learning
        selected_indices = active_learning_engine.select_samples_for_review(
            results=results,
            strategy=strategy,
            n_samples=n_samples,
            exclude_reviewed=exclude_reviewed
        )
        
        if not selected_indices:
            return {
                "job_id": job_id,
                "strategy": strategy,
                "suggested_samples": [],
                "message": "No samples available for review with current criteria",
                "total_available": len(results)
            }
        
        # Format suggested samples
        suggested_samples = []
        for idx in selected_indices:
            result = results[idx]
            sample_data = {
                "result_id": result.id,
                "filename": result.filename,
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "already_reviewed": result.reviewed,
                "created_at": result.created_at.isoformat(),
                "reason_for_selection": self._get_selection_reason(result, strategy)
            }
            
            # Add additional predictions if available
            if result.all_predictions:
                sample_data["all_predictions"] = result.all_predictions
            
            suggested_samples.append(sample_data)
        
        # Calculate strategy effectiveness metrics
        confidence_stats = self._calculate_confidence_stats(results, selected_indices)
        
        return {
            "job_id": job_id,
            "strategy": strategy,
            "suggested_samples": suggested_samples,
            "selection_statistics": {
                "total_available_samples": len(results),
                "samples_requested": n_samples,
                "samples_selected": len(selected_indices),
                "strategy_used": strategy,
                "confidence_range": {
                    "min_confidence": min(r.confidence for r in results if r.confidence is not None),
                    "max_confidence": max(r.confidence for r in results if r.confidence is not None),
                    "avg_confidence": np.mean([r.confidence for r in results if r.confidence is not None]),
                    "selected_avg_confidence": np.mean([results[i].confidence for i in selected_indices 
                                                       if results[i].confidence is not None])
                }
            },
            "next_steps": {
                "review_url": f"/api/review/batch",
                "bulk_review_tip": "Use batch review endpoints to efficiently process multiple samples",
                "rerun_suggestion": "Run active learning again after reviewing to get next batch"
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Active learning suggestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Active learning failed: {str(e)}")

def _get_selection_reason(self, result: Result, strategy: str) -> str:
    """Generate human-readable reason for sample selection"""
    reasons = {
        "uncertainty": f"Low confidence ({result.confidence:.3f}) - model uncertain about prediction",
        "margin": "Small margin between top predictions - borderline case",
        "entropy": "High prediction entropy - model confused across multiple classes",
        "diverse": f"Selected for class diversity ({result.predicted_label})",
        "disagreement": "High disagreement between different model predictions"
    }
    return reasons.get(strategy, "Selected by active learning algorithm")

def _calculate_confidence_stats(self, results: List[Result], selected_indices: List[int]) -> Dict[str, float]:
    """Calculate confidence statistics for selected samples"""
    all_confidences = [r.confidence for r in results if r.confidence is not None]
    selected_confidences = [results[i].confidence for i in selected_indices 
                          if results[i].confidence is not None]
    
    if not all_confidences or not selected_confidences:
        return {}
    
    return {
        "overall_mean_confidence": float(np.mean(all_confidences)),
        "overall_std_confidence": float(np.std(all_confidences)),
        "selected_mean_confidence": float(np.mean(selected_confidences)),
        "selected_std_confidence": float(np.std(selected_confidences)),
        "confidence_reduction": float(np.mean(all_confidences) - np.mean(selected_confidences))
    }

@router.get("/strategies")
async def get_active_learning_strategies():
    """Get information about available active learning strategies"""
    
    return {
        "strategies": {
            "uncertainty": {
                "name": "Uncertainty Sampling",
                "description": "Select samples with lowest confidence scores",
                "best_for": ["Binary classification", "Initial labeling rounds"],
                "advantages": ["Simple and effective", "Works with any model"],
                "disadvantages": ["May focus on outliers", "Can be noisy"]
            },
            "margin": {
                "name": "Margin Sampling", 
                "description": "Select samples with smallest margin between top predictions",
                "best_for": ["Multi-class problems", "Well-calibrated models"],
                "advantages": ["More robust than uncertainty", "Focuses on decision boundaries"],
                "disadvantages": ["Requires top-k predictions", "More computationally expensive"]
            },
            "entropy": {
                "name": "Entropy Sampling",
                "description": "Select samples with highest prediction entropy",
                "best_for": ["Multi-class classification", "Complex decision boundaries"],
                "advantages": ["Considers full probability distribution", "Theoretically grounded"],
                "disadvantages": ["Requires probability outputs", "Can be sensitive to calibration"]
            },
            "diverse": {
                "name": "Diversity Sampling",
                "description": "Select diverse samples for balanced coverage",
                "best_for": ["Imbalanced datasets", "Ensuring representative coverage"],
                "advantages": ["Ensures class balance", "Reduces bias"],
                "disadvantages": ["May miss important edge cases", "Requires class labels"]
            },
            "disagreement": {
                "name": "Disagreement Sampling",
                "description": "Select samples where different models disagree",
                "best_for": ["Ensemble models", "Model comparison"],
                "advantages": ["Leverages multiple models", "Finds challenging cases"],
                "disadvantages": ["Requires multiple models", "More complex setup"]
            }
        },
        "recommendations": {
            "getting_started": "uncertainty",
            "small_datasets": "diverse",
            "large_datasets": "margin",
            "multi_class": "entropy",
            "imbalanced_data": "diverse",
            "ensemble_models": "disagreement"
        },
        "best_practices": [
            "Start with uncertainty sampling for simplicity",
            "Use diversity sampling for imbalanced datasets",
            "Combine strategies for optimal results",
            "Review 10-20% of data in each iteration",
            "Monitor accuracy improvements to validate strategy effectiveness",
            "Consider domain expertise when interpreting suggestions"
        ]
    }

@router.post("/analyze-effectiveness/{job_id}")
async def analyze_active_learning_effectiveness(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    comparison_days: int = Query(7, ge=1, le=30)
):
    """
    Analyze the effectiveness of active learning for a job
    Compare metrics before and after human review
    """
    
    try:
        # Verify job ownership
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == current_user.id
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get all results for this job
        all_results = db.query(Result).filter(
            Result.job_id == job_id,
            Result.status == "success"
        ).order_by(Result.created_at).all()
        
        if not all_results:
            raise HTTPException(status_code=404, detail="No results found")
        
        # Split results into before/after based on review timeline
        cutoff_date = datetime.now() - timedelta(days=comparison_days)
        
        results_before = [r for r in all_results if r.created_at < cutoff_date]
        results_after = [r for r in all_results if r.created_at >= cutoff_date]
        
        # Calculate improvement metrics
        improvement_metrics = active_learning_engine.calculate_information_gain(
            results_before, results_after
        )
        
        # Additional analysis
        review_stats = {
            "total_results": len(all_results),
            "reviewed_results": sum(1 for r in all_results if r.reviewed),
            "review_percentage": round(sum(1 for r in all_results if r.reviewed) / len(all_results) * 100, 1),
            "results_before_cutoff": len(results_before),
            "results_after_cutoff": len(results_after)
        }
        
        # Label distribution analysis
        all_labels = [r.predicted_label for r in all_results if r.predicted_label]
        reviewed_labels = [r.predicted_label for r in all_results if r.reviewed and r.predicted_label]
        
        label_coverage = {
            "total_unique_labels": len(set(all_labels)),
            "reviewed_unique_labels": len(set(reviewed_labels)),
            "coverage_percentage": round(len(set(reviewed_labels)) / len(set(all_labels)) * 100, 1) if all_labels else 0
        }
        
        # Confidence evolution
        confidence_evolution = self._analyze_confidence_evolution(all_results)
        
        return {
            "job_id": job_id,
            "analysis_period": f"{comparison_days} days",
            "improvement_metrics": improvement_metrics,
            "review_statistics": review_stats,
            "label_coverage": label_coverage,
            "confidence_evolution": confidence_evolution,
            "recommendations": self._generate_recommendations(
                improvement_metrics, review_stats, label_coverage
            ),
            "analyzed_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Active learning analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def _analyze_confidence_evolution(self, results: List[Result]) -> Dict[str, Any]:
    """Analyze how confidence scores evolve over time"""
    
    if not results:
        return {}
    
    # Sort by creation time
    sorted_results = sorted(results, key=lambda x: x.created_at)
    
    # Group by time periods (daily)
    daily_confidence = {}
    for result in sorted_results:
        day = result.created_at.date()
        if day not in daily_confidence:
            daily_confidence[day] = []
        if result.confidence is not None:
            daily_confidence[day].append(result.confidence)
    
    # Calculate daily averages
    confidence_trend = []
    for day in sorted(daily_confidence.keys()):
        confidences = daily_confidence[day]
        if confidences:
            confidence_trend.append({
                "date": day.isoformat(),
                "avg_confidence": round(np.mean(confidences), 3),
                "std_confidence": round(np.std(confidences), 3),
                "sample_count": len(confidences)
            })
    
    return {
        "daily_trends": confidence_trend,
        "overall_trend": "improving" if len(confidence_trend) > 1 and 
                        confidence_trend[-1]["avg_confidence"] > confidence_trend[0]["avg_confidence"]
                        else "stable"
    }

def _generate_recommendations(self, improvement_metrics: Dict, review_stats: Dict, label_coverage: Dict) -> List[str]:
    """Generate actionable recommendations based on analysis"""
    
    recommendations = []
    
    # Review coverage recommendations
    review_pct = review_stats.get("review_percentage", 0)
    if review_pct < 10:
        recommendations.append("Consider reviewing more samples (currently {:.1f}%) to improve model accuracy".format(review_pct))
    elif review_pct > 50:
        recommendations.append("Good review coverage ({:.1f}%) - focus on quality over quantity".format(review_pct))
    
    # Label coverage recommendations
    coverage_pct = label_coverage.get("coverage_percentage", 0)
    if coverage_pct < 80:
        recommendations.append("Improve label coverage (currently {:.1f}%) by reviewing samples from underrepresented classes".format(coverage_pct))
    
    # Improvement recommendations
    confidence_improvement = improvement_metrics.get("confidence_improvement", 0)
    if confidence_improvement < 0.05:
        recommendations.append("Low confidence improvement - consider different sampling strategy or more diverse training data")
    
    accuracy_improvement = improvement_metrics.get("accuracy_improvement", 0)
    if accuracy_improvement < 0.05:
        recommendations.append("Limited accuracy gains - focus on reviewing borderline cases and difficult samples")
    
    if not recommendations:
        recommendations.append("Active learning is working well - continue current strategy")
    
    return recommendations 