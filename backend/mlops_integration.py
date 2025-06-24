from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Project, Job, Result
from typing import Dict, List, Any, Optional
import logging
import json
import requests
import asyncio
from datetime import datetime
import os
from pathlib import Path
import tempfile
import zipfile

logger = logging.getLogger(__name__)

class MLOpsIntegrator:
    def __init__(self):
        self.supported_platforms = {
            "mlflow": MLflowConnector(),
            "kubeflow": KubeflowConnector(), 
            "sagemaker": SageMakerConnector(),
            "custom": CustomConnector()
        }
    
    async def export_for_training(
        self,
        db: Session,
        project_id: int,
        platform: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export annotated data for ML training platform"""
        
        if platform not in self.supported_platforms:
            raise ValueError(f"Unsupported platform: {platform}")
        
        connector = self.supported_platforms[platform]
        
        # Get project and validate
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Get completed annotations
        completed_jobs = db.query(Job).filter(
            Job.project_id == project_id,
            Job.status == "completed"
        ).all()
        
        if not completed_jobs:
            raise ValueError("No completed annotations found")
        
        # Prepare dataset
        dataset = await self._prepare_training_dataset(db, completed_jobs, project.project_type)
        
        # Export to platform
        export_result = await connector.export_dataset(dataset, config)
        
        return {
            "status": "success",
            "platform": platform,
            "project_id": project_id,
            "dataset_info": {
                "total_samples": len(dataset["samples"]),
                "train_samples": len(dataset.get("train_split", [])),
                "val_samples": len(dataset.get("val_split", [])),
                "test_samples": len(dataset.get("test_split", []))
            },
            "export_result": export_result
        }
    
    async def _prepare_training_dataset(
        self,
        db: Session,
        jobs: List[Job],
        project_type: str
    ) -> Dict[str, Any]:
        """Prepare dataset in training-ready format"""
        
        samples = []
        labels = set()
        
        for job in jobs:
            results = db.query(Result).filter(Result.job_id == job.id).all()
            
            for result in results:
                # Use ground truth if available, otherwise predicted label
                label = result.ground_truth or result.predicted_label
                if not label:
                    continue
                
                sample = {
                    "id": f"{job.id}_{result.id}",
                    "file_path": result.file_path,
                    "filename": result.filename,
                    "label": label,
                    "confidence": result.confidence,
                    "project_type": project_type,
                    "metadata": {
                        "job_id": job.id,
                        "result_id": result.id,
                        "reviewed": result.reviewed,
                        "review_action": result.review_action
                    }
                }
                
                # Add type-specific data
                if project_type == "object_detection" and result.bounding_boxes:
                    sample["bounding_boxes"] = json.loads(result.bounding_boxes)
                elif project_type == "text_classification" and result.entities:
                    sample["entities"] = json.loads(result.entities)
                
                samples.append(sample)
                labels.add(label)
        
        # Create train/val/test splits (70/20/10)
        import random
        random.shuffle(samples)
        
        total = len(samples)
        train_end = int(total * 0.7)
        val_end = int(total * 0.9)
        
        return {
            "samples": samples,
            "labels": list(labels),
            "train_split": samples[:train_end],
            "val_split": samples[train_end:val_end],
            "test_split": samples[val_end:],
            "project_type": project_type,
            "total_samples": total
        }
    
    async def trigger_training_pipeline(
        self,
        platform: str,
        dataset_info: Dict[str, Any],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trigger training pipeline on the specified platform"""
        
        if platform not in self.supported_platforms:
            raise ValueError(f"Unsupported platform: {platform}")
        
        connector = self.supported_platforms[platform]
        return await connector.start_training(dataset_info, training_config)

class MLflowConnector:
    def __init__(self):
        self.base_url = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.experiment_name = "ModelShip_Annotations"
    
    async def export_dataset(self, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Export dataset to MLflow"""
        
        try:
            import mlflow
            import mlflow.data
            from mlflow.data.pandas_dataset import PandasDataset
            import pandas as pd
            
            # Convert to pandas DataFrame
            df_data = []
            for sample in dataset["samples"]:
                row = {
                    "file_path": sample["file_path"],
                    "label": sample["label"],
                    "confidence": sample["confidence"],
                    "split": "train" if sample in dataset["train_split"] else "val" if sample in dataset["val_split"] else "test"
                }
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.base_url)
            
            # Create or get experiment
            try:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            except:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                experiment_id = experiment.experiment_id
            
            # Log dataset
            with mlflow.start_run(experiment_id=experiment_id):
                dataset_source = mlflow.data.from_pandas(df, source="ModelShip_Annotations")
                mlflow.log_input(dataset_source, context="training")
                
                # Log metadata
                mlflow.log_params({
                    "total_samples": dataset["total_samples"],
                    "num_labels": len(dataset["labels"]),
                    "project_type": dataset["project_type"]
                })
                
                run_id = mlflow.active_run().info.run_id
            
            return {
                "platform": "mlflow",
                "experiment_id": experiment_id,
                "run_id": run_id,
                "dataset_uri": f"{self.base_url}/#/experiments/{experiment_id}/runs/{run_id}",
                "status": "exported"
            }
            
        except ImportError:
            logger.warning("MLflow not installed, using mock export")
            return {
                "platform": "mlflow",
                "status": "mock_export",
                "message": "MLflow not available - install with: pip install mlflow"
            }
        except Exception as e:
            logger.error(f"MLflow export failed: {e}")
            raise HTTPException(status_code=400, detail=f"MLflow export failed: {e}")
    
    async def start_training(self, dataset_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Start training job in MLflow"""
        
        try:
            import mlflow
            
            # This would typically trigger a training script
            training_script = config.get("training_script", "train.py")
            parameters = config.get("parameters", {})
            
            # In a real implementation, this would submit a job to MLflow Projects
            return {
                "platform": "mlflow",
                "status": "training_started",
                "job_id": f"mlflow_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "message": f"Training job submitted with script: {training_script}"
            }
            
        except ImportError:
            return {
                "platform": "mlflow",
                "status": "mock_training",
                "message": "MLflow not available for training"
            }

class KubeflowConnector:
    def __init__(self):
        self.api_url = os.getenv("KUBEFLOW_API_URL", "http://localhost:8080")
        self.namespace = os.getenv("KUBEFLOW_NAMESPACE", "kubeflow")
    
    async def export_dataset(self, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Export dataset to Kubeflow"""
        
        # Create dataset manifest for Kubeflow
        manifest = {
            "apiVersion": "data.kubeflow.org/v1alpha1",
            "kind": "Dataset",
            "metadata": {
                "name": f"modelship-dataset-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "namespace": self.namespace
            },
            "spec": {
                "description": "Dataset from ModelShip annotation platform",
                "source": "ModelShip",
                "format": dataset["project_type"],
                "samples": dataset["total_samples"],
                "labels": dataset["labels"]
            }
        }
        
        # In a real implementation, this would create the dataset in Kubeflow
        return {
            "platform": "kubeflow",
            "status": "exported",
            "dataset_name": manifest["metadata"]["name"],
            "namespace": self.namespace,
            "manifest": manifest
        }
    
    async def start_training(self, dataset_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Start training pipeline in Kubeflow"""
        
        pipeline_config = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "name": f"modelship-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "namespace": self.namespace
            },
            "spec": {
                "entrypoint": "training-pipeline",
                "templates": [{
                    "name": "training-pipeline",
                    "steps": [
                        {"name": "data-preparation", "template": "prepare-data"},
                        {"name": "model-training", "template": "train-model"},
                        {"name": "model-evaluation", "template": "evaluate-model"}
                    ]
                }]
            }
        }
        
        return {
            "platform": "kubeflow",
            "status": "pipeline_submitted",
            "workflow_name": pipeline_config["metadata"]["name"],
            "namespace": self.namespace
        }

class SageMakerConnector:
    def __init__(self):
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.bucket = os.getenv("SAGEMAKER_BUCKET", "modelship-sagemaker")
    
    async def export_dataset(self, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Export dataset to SageMaker format"""
        
        # Create SageMaker-compatible dataset structure
        s3_path = f"s3://{self.bucket}/datasets/modelship-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # In a real implementation, this would upload to S3
        return {
            "platform": "sagemaker",
            "status": "exported",
            "s3_path": s3_path,
            "dataset_format": "sagemaker_format",
            "total_samples": dataset["total_samples"]
        }
    
    async def start_training(self, dataset_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Start SageMaker training job"""
        
        job_name = f"modelship-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        training_config = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": config.get("training_image", "382416733822.dkr.ecr.us-east-1.amazonaws.com/image_classification:latest"),
                "TrainingInputMode": "File"
            },
            "RoleArn": config.get("role_arn", "arn:aws:iam::123456789012:role/SageMakerRole"),
            "InputDataConfig": [{
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": dataset_info.get("s3_path", ""),
                        "S3DataDistributionType": "FullyReplicated"
                    }
                }
            }],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{self.bucket}/models/"
            },
            "ResourceConfig": {
                "InstanceType": config.get("instance_type", "ml.m5.large"),
                "InstanceCount": 1,
                "VolumeSizeInGB": 30
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": config.get("max_runtime", 3600)
            }
        }
        
        return {
            "platform": "sagemaker",
            "status": "training_job_submitted", 
            "job_name": job_name,
            "config": training_config
        }

class CustomConnector:
    def __init__(self):
        self.webhook_url = os.getenv("CUSTOM_WEBHOOK_URL")
    
    async def export_dataset(self, dataset: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Export to custom endpoint"""
        
        if not self.webhook_url:
            return {
                "platform": "custom",
                "status": "no_webhook_configured",
                "message": "Set CUSTOM_WEBHOOK_URL environment variable"
            }
        
        # Send dataset info to custom webhook
        payload = {
            "action": "dataset_export",
            "dataset": {
                "total_samples": dataset["total_samples"],
                "labels": dataset["labels"],
                "project_type": dataset["project_type"]
            },
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                "platform": "custom",
                "status": "exported",
                "webhook_response": response.json() if response.content else None
            }
        except Exception as e:
            logger.error(f"Custom webhook failed: {e}")
            return {
                "platform": "custom",
                "status": "webhook_failed",
                "error": str(e)
            }
    
    async def start_training(self, dataset_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger custom training pipeline"""
        
        if not self.webhook_url:
            return {
                "platform": "custom",
                "status": "no_webhook_configured"
            }
        
        payload = {
            "action": "start_training",
            "dataset_info": dataset_info,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            
            return {
                "platform": "custom",
                "status": "training_started",
                "webhook_response": response.json() if response.content else None
            }
        except Exception as e:
            return {
                "platform": "custom",
                "status": "webhook_failed",
                "error": str(e)
            }

# Create service instance
mlops_integrator = MLOpsIntegrator()

# FastAPI Router
router = APIRouter(prefix="/api/mlops", tags=["mlops"])

@router.get("/platforms")
async def get_supported_platforms():
    """Get list of supported MLOps platforms"""
    return {
        "status": "success",
        "platforms": {
            "mlflow": {
                "name": "MLflow",
                "description": "Open source ML lifecycle management",
                "features": ["experiment_tracking", "model_registry", "deployment"]
            },
            "kubeflow": {
                "name": "Kubeflow",
                "description": "ML workflows on Kubernetes",
                "features": ["pipelines", "distributed_training", "serving"]
            },
            "sagemaker": {
                "name": "Amazon SageMaker",
                "description": "AWS managed ML platform",
                "features": ["managed_training", "auto_scaling", "built_in_algorithms"]
            },
            "custom": {
                "name": "Custom Webhook",
                "description": "Integration via custom webhook",
                "features": ["flexible_integration", "custom_workflows"]
            }
        }
    }

@router.post("/export/{project_id}")
async def export_for_training(
    project_id: int,
    platform: str,
    config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Export annotated data for ML training"""
    try:
        result = await mlops_integrator.export_for_training(db, project_id, platform, config)
        return result
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/train")
async def start_training_pipeline(
    platform: str,
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any]
):
    """Start training pipeline on specified platform"""
    try:
        result = await mlops_integrator.trigger_training_pipeline(
            platform, dataset_info, training_config
        )
        return result
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{platform}/{job_id}")
async def get_training_status(platform: str, job_id: str):
    """Get training job status (placeholder for now)"""
    return {
        "status": "success",
        "platform": platform,
        "job_id": job_id,
        "training_status": "running",
        "message": "Status tracking will be implemented based on platform APIs"
    }

@router.get("/status")
async def get_mlops_status():
    """Get MLOps integration status"""
    return {
        "status": "available",
        "message": "MLOps integration endpoints",
        "features": ["model_versioning", "deployment_tracking", "performance_monitoring"]
    }

@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "models": [],
        "message": "MLOps model management - coming soon"
    } 