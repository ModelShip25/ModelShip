from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON, Float
from database import get_db
from database_base import Base
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class BiasType(str, Enum):
    CLASS_IMBALANCE = "class_imbalance"
    DEMOGRAPHIC_BIAS = "demographic_bias"
    PREDICTION_BIAS = "prediction_bias"
    SELECTION_BIAS = "selection_bias"
    ANNOTATION_BIAS = "annotation_bias"
    ALGORITHMIC_BIAS = "algorithmic_bias"

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class BiasReport(Base):
    __tablename__ = "bias_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    
    # Report metadata
    report_type = Column(String(50), nullable=False)  # "automated", "manual", "scheduled"
    analysis_date = Column(DateTime, default=datetime.utcnow)
    data_period_start = Column(DateTime)
    data_period_end = Column(DateTime)
    
    # Bias analysis results
    detected_biases = Column(JSON)  # List of bias types detected
    severity_scores = Column(JSON)  # Severity for each bias type
    bias_metrics = Column(JSON)    # Detailed metrics
    
    # Recommendations
    recommendations = Column(JSON)  # List of recommended actions
    risk_assessment = Column(JSON)  # Risk levels and impact analysis
    
    # Compliance status
    fairness_score = Column(Float)  # Overall fairness score 0-1
    compliance_status = Column(JSON)  # Compliance with various standards
    
    # Visualization data
    charts_data = Column(JSON)  # Base64 encoded charts
    
    # Status
    is_active = Column(Boolean, default=True)
    reviewed_by = Column(Integer, ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    action_taken = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)

@dataclass
class BiasAnalysisConfig:
    check_class_balance: bool = True
    check_demographic_parity: bool = True
    check_equalized_odds: bool = True
    check_prediction_fairness: bool = True
    check_annotation_consistency: bool = True
    
    # Thresholds
    class_imbalance_threshold: float = 0.1  # Min proportion for minority class
    demographic_parity_threshold: float = 0.05  # Max difference in positive rates
    equalized_odds_threshold: float = 0.05  # Max difference in TPR/FPR
    annotation_agreement_threshold: float = 0.8  # Min agreement between annotators
    
    # Protected attributes
    protected_attributes: List[str] = None
    fairness_metrics: List[str] = None

class BiasFairnessService:
    def __init__(self):
        self.analysis_config = BiasAnalysisConfig()
        sns.set_style("whitegrid")
        plt.style.use('seaborn-v0_8')
    
    def analyze_project_bias(
        self,
        db: Session,
        project_id: int,
        data: List[Dict[str, Any]],
        config: Optional[BiasAnalysisConfig] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive bias analysis on project data"""
        
        if config:
            self.analysis_config = config
        
        logger.info(f"Starting bias analysis for project {project_id}")
        
        # Convert data to DataFrame for analysis
        df = pd.DataFrame(data)
        
        analysis_results = {
            "project_id": project_id,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "data_summary": self._generate_data_summary(df),
            "bias_detection": {},
            "fairness_metrics": {},
            "recommendations": [],
            "risk_assessment": {},
            "visualizations": {}
        }
        
        # 1. Class Balance Analysis
        if self.analysis_config.check_class_balance:
            analysis_results["bias_detection"]["class_imbalance"] = self._analyze_class_balance(df)
        
        # 2. Demographic Bias Analysis
        if self.analysis_config.check_demographic_parity:
            analysis_results["bias_detection"]["demographic_bias"] = self._analyze_demographic_bias(df)
        
        # 3. Prediction Fairness Analysis
        if self.analysis_config.check_prediction_fairness:
            analysis_results["bias_detection"]["prediction_bias"] = self._analyze_prediction_fairness(df)
        
        # 4. Annotation Consistency Analysis
        if self.analysis_config.check_annotation_consistency:
            analysis_results["bias_detection"]["annotation_bias"] = self._analyze_annotation_consistency(df)
        
        # 5. Generate Fairness Metrics
        analysis_results["fairness_metrics"] = self._calculate_fairness_metrics(df)
        
        # 6. Generate Recommendations
        analysis_results["recommendations"] = self._generate_recommendations(analysis_results["bias_detection"])
        
        # 7. Risk Assessment
        analysis_results["risk_assessment"] = self._assess_bias_risks(analysis_results["bias_detection"])
        
        # 8. Generate Visualizations
        analysis_results["visualizations"] = self._generate_bias_visualizations(df, analysis_results)
        
        # 9. Calculate Overall Fairness Score
        fairness_score = self._calculate_overall_fairness_score(analysis_results)
        analysis_results["fairness_score"] = fairness_score
        
        # 10. Save report to database
        report_id = self._save_bias_report(db, project_id, analysis_results)
        analysis_results["report_id"] = report_id
        
        logger.info(f"Bias analysis completed for project {project_id}. Fairness score: {fairness_score:.3f}")
        
        return analysis_results
    
    def _generate_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics of the dataset"""
        
        summary = {
            "total_samples": len(df),
            "total_features": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        # Label distribution
        if 'label' in df.columns:
            label_counts = df['label'].value_counts().to_dict()
            summary["label_distribution"] = label_counts
            summary["num_classes"] = len(label_counts)
        
        # Confidence score statistics
        if 'confidence' in df.columns:
            summary["confidence_stats"] = {
                "mean": float(df['confidence'].mean()),
                "std": float(df['confidence'].std()),
                "min": float(df['confidence'].min()),
                "max": float(df['confidence'].max()),
                "median": float(df['confidence'].median())
            }
        
        return summary
    
    def _analyze_class_balance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze class balance and detect imbalances"""
        
        if 'label' not in df.columns:
            return {"status": "skipped", "reason": "No label column found"}
        
        label_counts = df['label'].value_counts()
        total_samples = len(df)
        
        class_proportions = (label_counts / total_samples).to_dict()
        
        # Detect imbalance
        min_proportion = min(class_proportions.values())
        is_imbalanced = min_proportion < self.analysis_config.class_imbalance_threshold
        
        imbalance_ratio = max(class_proportions.values()) / min_proportion
        
        # Determine severity
        if min_proportion < 0.05:
            severity = SeverityLevel.CRITICAL
        elif min_proportion < 0.1:
            severity = SeverityLevel.HIGH
        elif min_proportion < 0.2:
            severity = SeverityLevel.MEDIUM
        else:
            severity = SeverityLevel.LOW
        
        return {
            "detected": is_imbalanced,
            "severity": severity.value,
            "metrics": {
                "class_proportions": class_proportions,
                "imbalance_ratio": float(imbalance_ratio),
                "minority_class_proportion": float(min_proportion),
                "total_classes": len(class_proportions)
            },
            "affected_classes": [
                cls for cls, prop in class_proportions.items() 
                if prop < self.analysis_config.class_imbalance_threshold
            ]
        }
    
    def _analyze_demographic_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze bias across demographic groups"""
        
        # Look for demographic columns
        demographic_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['gender', 'age', 'race', 'ethnicity', 'location'])
        ]
        
        if not demographic_columns:
            return {"status": "skipped", "reason": "No demographic columns detected"}
        
        bias_results = {
            "detected": False,
            "severity": SeverityLevel.LOW.value,
            "metrics": {},
            "affected_groups": []
        }
        
        for demo_col in demographic_columns:
            if demo_col not in df.columns:
                continue
            
            # Calculate representation
            group_counts = df[demo_col].value_counts()
            group_proportions = (group_counts / len(df)).to_dict()
            
            # If we have labels, check for disparate impact
            if 'label' in df.columns:
                bias_metrics = self._calculate_demographic_parity(df, demo_col, 'label')
                
                # Check if bias detected
                max_disparity = max(bias_metrics.get('parity_differences', [0]))
                if max_disparity > self.analysis_config.demographic_parity_threshold:
                    bias_results["detected"] = True
                    if max_disparity > 0.2:
                        bias_results["severity"] = SeverityLevel.HIGH.value
                    elif max_disparity > 0.1:
                        bias_results["severity"] = SeverityLevel.MEDIUM.value
            
            bias_results["metrics"][demo_col] = {
                "group_proportions": group_proportions,
                "representation_balance": min(group_proportions.values()) / max(group_proportions.values()) if group_proportions else 0
            }
            
            if 'label' in df.columns:
                bias_results["metrics"][demo_col].update(bias_metrics)
        
        return bias_results
    
    def _calculate_demographic_parity(
        self, 
        df: pd.DataFrame, 
        protected_attr: str, 
        outcome_attr: str
    ) -> Dict[str, Any]:
        """Calculate demographic parity metrics"""
        
        # Get positive outcome rates by group
        positive_rates = {}
        parity_differences = []
        
        for group in df[protected_attr].unique():
            if pd.isna(group):
                continue
            
            group_data = df[df[protected_attr] == group]
            if len(group_data) == 0:
                continue
            
            # Assuming binary classification with positive class
            positive_outcomes = group_data[outcome_attr].value_counts()
            total_in_group = len(group_data)
            
            if len(positive_outcomes) > 0:
                # Take the most frequent class as "positive" for this analysis
                positive_rate = positive_outcomes.iloc[0] / total_in_group
                positive_rates[str(group)] = float(positive_rate)
        
        # Calculate parity differences
        if len(positive_rates) > 1:
            rates = list(positive_rates.values())
            max_rate = max(rates)
            min_rate = min(rates)
            parity_differences = [max_rate - rate for rate in rates]
        
        return {
            "positive_rates_by_group": positive_rates,
            "parity_differences": parity_differences,
            "max_disparity": max(parity_differences) if parity_differences else 0
        }
    
    def _analyze_prediction_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fairness of predictions across groups"""
        
        if 'predicted_label' not in df.columns or 'actual_label' not in df.columns:
            return {"status": "skipped", "reason": "Prediction columns not found"}
        
        # Calculate confusion matrix metrics by group
        fairness_metrics = {
            "detected": False,
            "severity": SeverityLevel.LOW.value,
            "metrics": {}
        }
        
        # Overall performance
        accuracy = (df['predicted_label'] == df['actual_label']).mean()
        fairness_metrics["metrics"]["overall_accuracy"] = float(accuracy)
        
        # If we have demographic info, check fairness across groups
        demographic_columns = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['gender', 'age', 'race', 'ethnicity'])
        ]
        
        for demo_col in demographic_columns:
            if demo_col not in df.columns:
                continue
            
            group_metrics = {}
            accuracies = []
            
            for group in df[demo_col].unique():
                if pd.isna(group):
                    continue
                
                group_data = df[df[demo_col] == group]
                if len(group_data) == 0:
                    continue
                
                group_accuracy = (group_data['predicted_label'] == group_data['actual_label']).mean()
                group_metrics[str(group)] = float(group_accuracy)
                accuracies.append(group_accuracy)
            
            # Check for significant differences in accuracy
            if len(accuracies) > 1:
                accuracy_gap = max(accuracies) - min(accuracies)
                if accuracy_gap > 0.1:  # 10% difference threshold
                    fairness_metrics["detected"] = True
                    if accuracy_gap > 0.2:
                        fairness_metrics["severity"] = SeverityLevel.HIGH.value
                    elif accuracy_gap > 0.15:
                        fairness_metrics["severity"] = SeverityLevel.MEDIUM.value
            
            fairness_metrics["metrics"][f"{demo_col}_accuracy"] = group_metrics
            fairness_metrics["metrics"][f"{demo_col}_accuracy_gap"] = max(accuracies) - min(accuracies) if len(accuracies) > 1 else 0
        
        return fairness_metrics
    
    def _analyze_annotation_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze consistency of human annotations"""
        
        # Look for multiple annotator columns
        annotator_columns = [col for col in df.columns if 'annotator' in col.lower()]
        
        if len(annotator_columns) < 2:
            return {"status": "skipped", "reason": "Multiple annotators not found"}
        
        consistency_metrics = {
            "detected": False,
            "severity": SeverityLevel.LOW.value,
            "metrics": {}
        }
        
        # Calculate inter-annotator agreement
        agreements = []
        for i in range(len(df)):
            row_annotations = [df.iloc[i][col] for col in annotator_columns if not pd.isna(df.iloc[i][col])]
            if len(row_annotations) > 1:
                # Calculate agreement as proportion of matching annotations
                most_common = Counter(row_annotations).most_common(1)[0][1]
                agreement = most_common / len(row_annotations)
                agreements.append(agreement)
        
        if agreements:
            avg_agreement = np.mean(agreements)
            consistency_metrics["metrics"]["inter_annotator_agreement"] = float(avg_agreement)
            
            if avg_agreement < self.analysis_config.annotation_agreement_threshold:
                consistency_metrics["detected"] = True
                if avg_agreement < 0.6:
                    consistency_metrics["severity"] = SeverityLevel.HIGH.value
                elif avg_agreement < 0.7:
                    consistency_metrics["severity"] = SeverityLevel.MEDIUM.value
        
        # Analyze bias by annotator
        annotator_bias = {}
        for col in annotator_columns:
            if 'label' in df.columns:
                annotator_labels = df[col].dropna()
                if len(annotator_labels) > 0:
                    label_distribution = annotator_labels.value_counts(normalize=True).to_dict()
                    annotator_bias[col] = label_distribution
        
        consistency_metrics["metrics"]["annotator_bias"] = annotator_bias
        
        return consistency_metrics
    
    def _calculate_fairness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive fairness metrics"""
        
        metrics = {}
        
        # Representation metrics
        if 'label' in df.columns:
            label_entropy = self._calculate_entropy(df['label'].value_counts().values)
            metrics["label_entropy"] = float(label_entropy)
            metrics["label_balance_score"] = float(min(df['label'].value_counts(normalize=True)))
        
        # Confidence distribution metrics
        if 'confidence' in df.columns:
            confidence_stats = df['confidence'].describe()
            metrics["confidence_distribution"] = {
                "mean": float(confidence_stats['mean']),
                "std": float(confidence_stats['std']),
                "skewness": float(df['confidence'].skew()) if 'skew' in dir(df['confidence']) else 0
            }
        
        # Quality metrics
        if 'predicted_label' in df.columns and 'actual_label' in df.columns:
            accuracy = (df['predicted_label'] == df['actual_label']).mean()
            metrics["prediction_accuracy"] = float(accuracy)
        
        return metrics
    
    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate Shannon entropy"""
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    def _generate_recommendations(self, bias_detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on detected biases"""
        
        recommendations = []
        
        # Class imbalance recommendations
        class_bias = bias_detection.get("class_imbalance", {})
        if class_bias.get("detected", False):
            severity = class_bias.get("severity", "low")
            
            if severity in ["high", "critical"]:
                recommendations.append({
                    "type": "data_augmentation",
                    "priority": "high",
                    "title": "Address Severe Class Imbalance",
                    "description": "Implement data augmentation, synthetic data generation, or weighted sampling to balance classes",
                    "implementation": [
                        "Use SMOTE or ADASYN for synthetic minority oversampling",
                        "Implement class weights in model training",
                        "Consider ensemble methods like BalancedRandomForest",
                        "Collect more data for underrepresented classes"
                    ],
                    "expected_impact": "Improve model performance on minority classes by 15-30%"
                })
            else:
                recommendations.append({
                    "type": "monitoring",
                    "priority": "medium",
                    "title": "Monitor Class Balance",
                    "description": "Set up automated monitoring for class distribution changes",
                    "implementation": [
                        "Implement class balance alerts",
                        "Regular model retraining with balanced sampling",
                        "Track performance metrics by class"
                    ]
                })
        
        # Demographic bias recommendations
        demo_bias = bias_detection.get("demographic_bias", {})
        if demo_bias.get("detected", False):
            recommendations.append({
                "type": "fairness_intervention",
                "priority": "high",
                "title": "Mitigate Demographic Bias",
                "description": "Implement fairness-aware ML techniques to ensure equitable outcomes",
                "implementation": [
                    "Apply demographic parity constraints during training",
                    "Use adversarial debiasing techniques",
                    "Implement post-processing calibration by group",
                    "Regular bias audits and fairness testing"
                ],
                "expected_impact": "Reduce demographic disparities by 50-80%"
            })
        
        # Prediction fairness recommendations
        pred_bias = bias_detection.get("prediction_bias", {})
        if pred_bias.get("detected", False):
            recommendations.append({
                "type": "model_improvement",
                "priority": "high",
                "title": "Improve Prediction Fairness",
                "description": "Ensure consistent model performance across all demographic groups",
                "implementation": [
                    "Implement group-specific model calibration",
                    "Use fairness-aware loss functions",
                    "Regular performance monitoring by group",
                    "Consider multi-task learning approaches"
                ]
            })
        
        # Annotation bias recommendations
        annotation_bias = bias_detection.get("annotation_bias", {})
        if annotation_bias.get("detected", False):
            recommendations.append({
                "type": "process_improvement",
                "priority": "medium",
                "title": "Improve Annotation Quality",
                "description": "Enhance annotation guidelines and training to reduce bias",
                "implementation": [
                    "Develop clearer annotation guidelines",
                    "Implement annotator training programs",
                    "Use consensus-based labeling for difficult cases",
                    "Regular annotator performance reviews"
                ]
            })
        
        # General recommendations
        recommendations.append({
            "type": "governance",
            "priority": "medium",
            "title": "Establish Bias Monitoring Framework",
            "description": "Create systematic processes for ongoing bias detection and mitigation",
            "implementation": [
                "Schedule regular bias audits",
                "Implement automated bias alerts",
                "Create bias incident response procedures",
                "Establish fairness metrics dashboard"
            ]
        })
        
        return recommendations
    
    def _assess_bias_risks(self, bias_detection: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks associated with detected biases"""
        
        risk_assessment = {
            "overall_risk_level": "low",
            "risk_factors": [],
            "business_impact": {},
            "compliance_risks": {},
            "mitigation_urgency": "low"
        }
        
        high_risk_biases = 0
        medium_risk_biases = 0
        
        for bias_type, bias_info in bias_detection.items():
            if not isinstance(bias_info, dict) or not bias_info.get("detected", False):
                continue
            
            severity = bias_info.get("severity", "low")
            
            risk_factor = {
                "bias_type": bias_type,
                "severity": severity,
                "business_risk": "medium",
                "compliance_risk": "medium"
            }
            
            if bias_type == "demographic_bias":
                risk_factor["business_risk"] = "high"
                risk_factor["compliance_risk"] = "high"
                risk_factor["legal_implications"] = [
                    "Potential discrimination lawsuits",
                    "Regulatory compliance violations",
                    "Reputational damage"
                ]
            
            elif bias_type == "class_imbalance" and severity in ["high", "critical"]:
                risk_factor["business_risk"] = "high"
                risk_factor["operational_impact"] = [
                    "Poor model performance",
                    "Incorrect business decisions",
                    "Customer dissatisfaction"
                ]
            
            if severity == "high":
                high_risk_biases += 1
            elif severity == "medium":
                medium_risk_biases += 1
            
            risk_assessment["risk_factors"].append(risk_factor)
        
        # Determine overall risk level
        if high_risk_biases > 0:
            risk_assessment["overall_risk_level"] = "high"
            risk_assessment["mitigation_urgency"] = "immediate"
        elif medium_risk_biases > 1:
            risk_assessment["overall_risk_level"] = "medium"
            risk_assessment["mitigation_urgency"] = "high"
        elif medium_risk_biases > 0:
            risk_assessment["overall_risk_level"] = "medium"
            risk_assessment["mitigation_urgency"] = "medium"
        
        return risk_assessment
    
    def _generate_bias_visualizations(
        self, 
        df: pd.DataFrame, 
        analysis_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate visualizations for bias analysis"""
        
        visualizations = {}
        
        try:
            # 1. Class Distribution Chart
            if 'label' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                label_counts = df['label'].value_counts()
                
                bars = ax.bar(range(len(label_counts)), label_counts.values)
                ax.set_xlabel('Class Labels')
                ax.set_ylabel('Count')
                ax.set_title('Class Distribution')
                ax.set_xticks(range(len(label_counts)))
                ax.set_xticklabels(label_counts.index, rotation=45)
                
                # Color bars based on imbalance severity
                class_bias = analysis_results["bias_detection"].get("class_imbalance", {})
                if class_bias.get("detected", False):
                    severity = class_bias.get("severity", "low")
                    if severity == "critical":
                        colors = ['red' if count < len(df) * 0.05 else 'blue' for count in label_counts.values]
                    elif severity == "high":
                        colors = ['orange' if count < len(df) * 0.1 else 'blue' for count in label_counts.values]
                    else:
                        colors = ['yellow' if count < len(df) * 0.2 else 'blue' for count in label_counts.values]
                    
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                
                plt.tight_layout()
                visualizations["class_distribution"] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 2. Confidence Distribution
            if 'confidence' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(df['confidence'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Confidence Score Distribution')
                ax.axvline(df['confidence'].mean(), color='red', linestyle='--', label=f'Mean: {df["confidence"].mean():.3f}')
                ax.legend()
                
                plt.tight_layout()
                visualizations["confidence_distribution"] = self._fig_to_base64(fig)
                plt.close(fig)
            
            # 3. Demographic Analysis Charts
            demographic_columns = [
                col for col in df.columns 
                if any(keyword in col.lower() for keyword in ['gender', 'age', 'race', 'ethnicity'])
            ]
            
            for demo_col in demographic_columns[:2]:  # Limit to first 2 demographic columns
                if demo_col in df.columns:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Demographic distribution
                    demo_counts = df[demo_col].value_counts()
                    ax1.pie(demo_counts.values, labels=demo_counts.index, autopct='%1.1f%%')
                    ax1.set_title(f'{demo_col.title()} Distribution')
                    
                    # Performance by demographic group (if available)
                    if 'predicted_label' in df.columns and 'actual_label' in df.columns:
                        group_accuracies = []
                        group_names = []
                        
                        for group in df[demo_col].unique():
                            if not pd.isna(group):
                                group_data = df[df[demo_col] == group]
                                if len(group_data) > 0:
                                    accuracy = (group_data['predicted_label'] == group_data['actual_label']).mean()
                                    group_accuracies.append(accuracy)
                                    group_names.append(str(group))
                        
                        if group_accuracies:
                            bars = ax2.bar(group_names, group_accuracies)
                            ax2.set_ylabel('Accuracy')
                            ax2.set_title(f'Model Accuracy by {demo_col.title()}')
                            ax2.set_ylim(0, 1)
                            
                            # Color bars based on fairness
                            mean_accuracy = np.mean(group_accuracies)
                            for bar, acc in zip(bars, group_accuracies):
                                if abs(acc - mean_accuracy) > 0.1:
                                    bar.set_color('red')
                                elif abs(acc - mean_accuracy) > 0.05:
                                    bar.set_color('orange')
                                else:
                                    bar.set_color('green')
                    
                    plt.tight_layout()
                    visualizations[f"{demo_col}_analysis"] = self._fig_to_base64(fig)
                    plt.close(fig)
            
            # 4. Bias Severity Heatmap
            bias_data = analysis_results["bias_detection"]
            if bias_data:
                severity_mapping = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                
                bias_types = []
                severity_scores = []
                
                for bias_type, bias_info in bias_data.items():
                    if isinstance(bias_info, dict) and bias_info.get("detected", False):
                        bias_types.append(bias_type.replace("_", " ").title())
                        severity = bias_info.get("severity", "low")
                        severity_scores.append(severity_mapping.get(severity, 1))
                
                if bias_types:
                    fig, ax = plt.subplots(figsize=(10, max(3, len(bias_types) * 0.5)))
                    
                    # Create heatmap data
                    heatmap_data = np.array(severity_scores).reshape(-1, 1)
                    
                    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
                    ax.set_xticks([0])
                    ax.set_xticklabels(['Severity'])
                    ax.set_yticks(range(len(bias_types)))
                    ax.set_yticklabels(bias_types)
                    ax.set_title('Detected Bias Severity Levels')
                    
                    # Add text annotations
                    for i, score in enumerate(severity_scores):
                        severity_text = {1: "Low", 2: "Medium", 3: "High", 4: "Critical"}[score]
                        ax.text(0, i, severity_text, ha="center", va="center", fontweight='bold')
                    
                    plt.tight_layout()
                    visualizations["bias_severity_heatmap"] = self._fig_to_base64(fig)
                    plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error generating bias visualizations: {e}")
        
        return visualizations
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return img_base64
    
    def _calculate_overall_fairness_score(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall fairness score (0-1, higher is better)"""
        
        score = 1.0
        
        bias_detection = analysis_results.get("bias_detection", {})
        
        # Penalties for detected biases
        severity_penalties = {"low": 0.05, "medium": 0.15, "high": 0.3, "critical": 0.5}
        
        for bias_type, bias_info in bias_detection.items():
            if isinstance(bias_info, dict) and bias_info.get("detected", False):
                severity = bias_info.get("severity", "low")
                penalty = severity_penalties.get(severity, 0.1)
                
                # Additional penalties for certain bias types
                if bias_type == "demographic_bias":
                    penalty *= 1.5  # Demographic bias is more serious
                elif bias_type == "prediction_bias":
                    penalty *= 1.3
                
                score -= penalty
        
        return max(0.0, score)
    
    def _save_bias_report(
        self,
        db: Session,
        project_id: int,
        analysis_results: Dict[str, Any]
    ) -> int:
        """Save bias analysis report to database"""
        
        detected_biases = []
        severity_scores = {}
        
        for bias_type, bias_info in analysis_results["bias_detection"].items():
            if isinstance(bias_info, dict) and bias_info.get("detected", False):
                detected_biases.append(bias_type)
                severity_scores[bias_type] = bias_info.get("severity", "low")
        
        report = BiasReport(
            project_id=project_id,
            report_type="automated",
            detected_biases=detected_biases,
            severity_scores=severity_scores,
            bias_metrics=analysis_results["fairness_metrics"],
            recommendations=analysis_results["recommendations"],
            risk_assessment=analysis_results["risk_assessment"],
            fairness_score=analysis_results["fairness_score"],
            charts_data=analysis_results["visualizations"]
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        return report.id
    
    def generate_bias_alert(
        self,
        db: Session,
        project_id: int,
        bias_type: BiasType,
        severity: SeverityLevel,
        details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate automated bias alert"""
        
        alert = {
            "alert_id": f"bias_{project_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "project_id": project_id,
            "bias_type": bias_type.value,
            "severity": severity.value,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details,
            "recommended_actions": self._get_quick_recommendations(bias_type, severity),
            "escalation_required": severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        }
        
        logger.warning(f"Bias alert generated for project {project_id}: {bias_type.value} ({severity.value})")
        
        return alert
    
    def _get_quick_recommendations(
        self,
        bias_type: BiasType,
        severity: SeverityLevel
    ) -> List[str]:
        """Get quick recommendations for specific bias types"""
        
        recommendations = {
            BiasType.CLASS_IMBALANCE: [
                "Review data collection strategy",
                "Consider data augmentation techniques",
                "Implement class-weighted training",
                "Monitor minority class performance"
            ],
            BiasType.DEMOGRAPHIC_BIAS: [
                "Conduct immediate fairness audit",
                "Review data for representation issues",
                "Implement fairness constraints",
                "Consider legal/compliance review"
            ],
            BiasType.PREDICTION_BIAS: [
                "Analyze model performance by group",
                "Implement bias-aware training",
                "Review feature engineering",
                "Consider model recalibration"
            ],
            BiasType.ANNOTATION_BIAS: [
                "Review annotation guidelines",
                "Provide additional annotator training",
                "Implement consensus labeling",
                "Audit annotator performance"
            ]
        }
        
        base_recommendations = recommendations.get(bias_type, ["Investigate and monitor"])
        
        if severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            base_recommendations.insert(0, "URGENT: Immediate action required")
            if bias_type == BiasType.DEMOGRAPHIC_BIAS:
                base_recommendations.insert(1, "Consider pausing deployment until resolved")
        
        return base_recommendations

# Create service instance
bias_service = BiasFairnessService()

# FastAPI Router
router = APIRouter(prefix="/api/bias-fairness", tags=["bias_fairness"])

@router.post("/analyze/{project_id}")
async def analyze_project_bias(
    project_id: int,
    data: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """Perform comprehensive bias analysis on project data"""
    
    try:
        # Convert config if provided
        analysis_config = None
        if config:
            analysis_config = BiasAnalysisConfig(**config)
        
        results = bias_service.analyze_project_bias(db, project_id, data, analysis_config)
        
        return {
            "status": "success",
            "analysis": results
        }
        
    except Exception as e:
        logger.error(f"Bias analysis failed for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{project_id}")
async def get_bias_reports(
    project_id: int,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get bias analysis reports for a project"""
    
    reports = db.query(BiasReport).filter(
        BiasReport.project_id == project_id,
        BiasReport.is_active == True
    ).order_by(BiasReport.created_at.desc()).limit(limit).all()
    
    return {
        "status": "success",
        "project_id": project_id,
        "total_reports": len(reports),
        "reports": [
            {
                "report_id": report.id,
                "analysis_date": report.analysis_date.isoformat(),
                "detected_biases": report.detected_biases,
                "severity_scores": report.severity_scores,
                "fairness_score": report.fairness_score,
                "recommendations_count": len(report.recommendations or [])
            }
            for report in reports
        ]
    }

@router.get("/report/{report_id}")
async def get_detailed_bias_report(
    report_id: int,
    include_charts: bool = True,
    db: Session = Depends(get_db)
):
    """Get detailed bias analysis report"""
    
    report = db.query(BiasReport).filter(BiasReport.id == report_id).first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report_data = {
        "report_id": report.id,
        "project_id": report.project_id,
        "analysis_date": report.analysis_date.isoformat(),
        "detected_biases": report.detected_biases,
        "severity_scores": report.severity_scores,
        "bias_metrics": report.bias_metrics,
        "recommendations": report.recommendations,
        "risk_assessment": report.risk_assessment,
        "fairness_score": report.fairness_score,
        "compliance_status": report.compliance_status
    }
    
    if include_charts and report.charts_data:
        report_data["visualizations"] = report.charts_data
    
    return {
        "status": "success",
        "report": report_data
    }

@router.post("/alert")
async def create_bias_alert(
    project_id: int,
    bias_type: str,
    severity: str,
    details: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Create manual bias alert"""
    
    try:
        bias_type_enum = BiasType(bias_type.lower())
        severity_enum = SeverityLevel(severity.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid bias type or severity: {e}")
    
    alert = bias_service.generate_bias_alert(
        db, project_id, bias_type_enum, severity_enum, details
    )
    
    return {
        "status": "success",
        "alert": alert
    }

@router.get("/dashboard/{project_id}")
async def get_bias_dashboard(
    project_id: int,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get bias monitoring dashboard data"""
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get recent reports
    reports = db.query(BiasReport).filter(
        BiasReport.project_id == project_id,
        BiasReport.created_at >= start_date,
        BiasReport.is_active == True
    ).order_by(BiasReport.created_at.desc()).all()
    
    # Calculate trends
    fairness_scores = [report.fairness_score for report in reports if report.fairness_score]
    bias_trends = defaultdict(list)
    
    for report in reports:
        for bias_type in report.detected_biases or []:
            bias_trends[bias_type].append({
                "date": report.analysis_date.isoformat(),
                "severity": report.severity_scores.get(bias_type, "low")
            })
    
    dashboard_data = {
        "project_id": project_id,
        "analysis_period": f"Last {days} days",
        "summary": {
            "total_reports": len(reports),
            "current_fairness_score": fairness_scores[0] if fairness_scores else None,
            "fairness_trend": "improving" if len(fairness_scores) > 1 and fairness_scores[0] > fairness_scores[-1] else "stable",
            "active_biases": len(set(bias for report in reports[:3] for bias in (report.detected_biases or []))),
            "high_severity_alerts": sum(1 for report in reports if any(
                severity in ["high", "critical"] for severity in (report.severity_scores or {}).values()
            ))
        },
        "fairness_score_history": [
            {
                "date": report.analysis_date.isoformat(),
                "score": report.fairness_score
            }
            for report in reports if report.fairness_score
        ],
        "bias_trends": dict(bias_trends),
        "recent_recommendations": []
    }
    
    # Get latest recommendations
    if reports:
        latest_report = reports[0]
        dashboard_data["recent_recommendations"] = latest_report.recommendations or []
    
    return {
        "status": "success",
        "dashboard": dashboard_data
    } 