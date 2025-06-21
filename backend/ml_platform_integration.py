from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from database import get_db
from models import User, Project, Job, Result, File, ProjectAssignment
from auth import get_current_user, get_api_key_user
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import asyncio
from datetime import datetime, timedelta
import logging
import hashlib
import hmac
from pydantic import BaseModel

router = APIRouter(prefix="/api/ml-integration", tags=["ml_platform_integration"])

logger = logging.getLogger(__name__)

# Pydantic models for ML platform integration
class DatasetStreamConfig(BaseModel):
    format: str = "json"  # json, tensorflow, pytorch, huggingface
    batch_size: int = 32
    include_confidence: bool = True
    confidence_threshold: float = 0.0
    reviewed_only: bool = False
    image_size: Optional[tuple] = None
    preprocessing: Optional[Dict[str, Any]] = None

class WebhookConfig(BaseModel):
    url: str
    events: List[str]  # ["data.labeled", "project.completed", "quality.alert"]
    secret: str
    headers: Optional[Dict[str, str]] = None

class ModelRegistration(BaseModel):
    name: str
    version: str
    framework: str  # tensorflow, pytorch, huggingface, sklearn
    model_type: str  # classification, detection, segmentation
    metrics: Dict[str, float]
    description: Optional[str] = None

# In-memory webhook storage (in production, use Redis or database)
registered_webhooks = {}

@router.get("/datasets/{project_id}/info")
async def get_dataset_info(
    project_id: int,
    current_user: User = Depends(get_api_key_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive dataset information for ML platforms"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied to this project")
        
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Get dataset statistics
        total_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.status == "success"
        ).count()
        
        reviewed_results = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.reviewed == True
        ).count()
        
        # Get label distribution
        from sqlalchemy import func
        label_distribution = db.query(
            Result.predicted_label,
            func.count(Result.id).label('count')
        ).join(Job).filter(
            Job.project_id == project_id,
            Result.status == "success",
            Result.predicted_label.isnot(None)
        ).group_by(Result.predicted_label).all()
        
        # Get file types and sizes
        file_stats = db.query(
            File.file_type,
            func.count(File.id).label('count'),
            func.avg(File.file_size).label('avg_size')
        ).filter(File.project_id == project_id).group_by(File.file_type).all()
        
        return {
            "project_info": {
                "id": project.id,
                "name": project.name,
                "type": project.project_type,
                "created_at": project.created_at.isoformat(),
                "status": project.status.value
            },
            "dataset_statistics": {
                "total_samples": total_results,
                "labeled_samples": total_results,
                "reviewed_samples": reviewed_results,
                "review_coverage": round(reviewed_results / total_results * 100, 1) if total_results > 0 else 0
            },
            "label_distribution": {
                "unique_labels": len(label_distribution),
                "label_counts": {label: count for label, count in label_distribution},
                "is_balanced": _check_label_balance(label_distribution)
            },
            "file_statistics": {
                "file_types": {ftype: {"count": count, "avg_size_mb": round(avg_size / (1024*1024), 2) if avg_size else 0}
                             for ftype, count, avg_size in file_stats}
            },
            "ml_compatibility": {
                "tensorflow": True,
                "pytorch": True,
                "huggingface": project.project_type.startswith("text"),
                "sklearn": True,
                "formats_available": ["json", "csv", "tensorflow", "pytorch", "coco", "yolo"]
            },
            "streaming_available": True,
            "realtime_updates": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset info failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")

def _check_label_balance(label_distribution: List[tuple]) -> bool:
    """Check if dataset is reasonably balanced"""
    if not label_distribution:
        return True
    
    counts = [count for _, count in label_distribution]
    max_count = max(counts)
    min_count = min(counts)
    
    # Consider balanced if ratio is less than 10:1
    return (max_count / min_count) < 10 if min_count > 0 else False

@router.get("/datasets/{project_id}/stream")
async def stream_dataset(
    project_id: int,
    format: str = Query("json", description="Output format: json, tensorflow, pytorch, huggingface"),
    batch_size: int = Query(32, description="Batch size for streaming"),
    include_confidence: bool = Query(True, description="Include confidence scores"),
    confidence_threshold: float = Query(0.0, description="Minimum confidence threshold"),
    reviewed_only: bool = Query(False, description="Only include reviewed samples"),
    current_user: User = Depends(get_api_key_user),
    db: Session = Depends(get_db)
):
    """Stream dataset in real-time for ML training"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
        
        async def generate_data_stream():
            """Generate streaming data in specified format"""
            
            # Get initial dataset
            results_query = db.query(Result).join(Job).filter(
                Job.project_id == project_id,
                Result.status == "success"
            )
            
            if reviewed_only:
                results_query = results_query.filter(Result.reviewed == True)
            
            if confidence_threshold > 0:
                results_query = results_query.filter(Result.confidence >= confidence_threshold)
            
            results = results_query.all()
            
            # Stream in batches
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                
                if format == "tensorflow":
                    yield _format_tensorflow_batch(batch)
                elif format == "pytorch":
                    yield _format_pytorch_batch(batch)
                elif format == "huggingface":
                    yield _format_huggingface_batch(batch)
                else:  # Default JSON format
                    yield _format_json_batch(batch, include_confidence)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
            
            # Continue streaming new data (polling approach)
            last_check = datetime.utcnow()
            while True:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get new results since last check
                new_results = db.query(Result).join(Job).filter(
                    Job.project_id == project_id,
                    Result.status == "success",
                    Result.created_at > last_check
                ).all()
                
                if new_results:
                    logger.info(f"Streaming {len(new_results)} new samples to ML platform")
                    
                    for i in range(0, len(new_results), batch_size):
                        batch = new_results[i:i + batch_size]
                        
                        if format == "tensorflow":
                            yield _format_tensorflow_batch(batch)
                        elif format == "pytorch":
                            yield _format_pytorch_batch(batch)
                        else:
                            yield _format_json_batch(batch, include_confidence)
                
                last_check = datetime.utcnow()
        
        return StreamingResponse(
            generate_data_stream(),
            media_type="application/x-ndjson",
            headers={
                "X-ModelShip-Project-ID": str(project_id),
                "X-ModelShip-Format": format,
                "X-ModelShip-Batch-Size": str(batch_size)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset streaming failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

def _format_json_batch(batch: List[Result], include_confidence: bool) -> str:
    """Format batch as JSON lines"""
    batch_data = []
    
    for result in batch:
        item = {
            "id": result.id,
            "filename": result.filename,
            "label": result.predicted_label,
            "created_at": result.created_at.isoformat()
        }
        
        if include_confidence:
            item["confidence"] = result.confidence
            
        if result.all_predictions:
            item["all_predictions"] = result.all_predictions
            
        if result.reviewed:
            item["reviewed"] = True
            item["ground_truth"] = result.ground_truth
            
        batch_data.append(item)
    
    return "\n".join(json.dumps(item) for item in batch_data) + "\n"

def _format_tensorflow_batch(batch: List[Result]) -> str:
    """Format batch for TensorFlow consumption"""
    tf_batch = {
        "features": [],
        "labels": [],
        "metadata": []
    }
    
    for result in batch:
        # For images, include file path for loading
        if result.file:
            tf_batch["features"].append(result.file.file_path)
        else:
            tf_batch["features"].append(f"uploads/{result.filename}")
            
        tf_batch["labels"].append(result.predicted_label)
        tf_batch["metadata"].append({
            "confidence": result.confidence,
            "reviewed": result.reviewed,
            "created_at": result.created_at.isoformat()
        })
    
    return json.dumps(tf_batch) + "\n"

def _format_pytorch_batch(batch: List[Result]) -> str:
    """Format batch for PyTorch consumption"""
    pytorch_batch = {
        "data": [],
        "targets": [],
        "info": []
    }
    
    for result in batch:
        pytorch_batch["data"].append({
            "file_path": result.file.file_path if result.file else f"uploads/{result.filename}",
            "filename": result.filename
        })
        pytorch_batch["targets"].append(result.predicted_label)
        pytorch_batch["info"].append({
            "confidence": result.confidence,
            "reviewed": result.reviewed
        })
    
    return json.dumps(pytorch_batch) + "\n"

def _format_huggingface_batch(batch: List[Result]) -> str:
    """Format batch for Hugging Face datasets"""
    hf_batch = []
    
    for result in batch:
        # For text data, try to get actual text content
        text_content = ""
        if result.file and result.file.file_type in ["txt", "csv", "json"]:
            try:
                with open(result.file.file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except:
                text_content = result.filename  # Fallback
        else:
            text_content = result.filename
            
        hf_batch.append({
            "text": text_content,
            "label": result.predicted_label,
            "confidence": result.confidence,
            "reviewed": result.reviewed
        })
    
    return json.dumps(hf_batch) + "\n"

@router.post("/webhooks/register")
async def register_webhook(
    webhook_config: WebhookConfig,
    current_user: User = Depends(get_api_key_user),
    db: Session = Depends(get_db)
):
    """Register webhook for real-time ML platform notifications"""
    
    try:
        # Validate webhook URL
        import requests
        try:
            response = requests.head(webhook_config.url, timeout=5)
            if response.status_code >= 400:
                raise HTTPException(status_code=400, detail="Webhook URL is not accessible")
        except requests.RequestException:
            raise HTTPException(status_code=400, detail="Cannot reach webhook URL")
        
        # Generate webhook ID
        webhook_id = hashlib.md5(f"{current_user.id}_{webhook_config.url}".encode()).hexdigest()
        
        # Store webhook configuration
        registered_webhooks[webhook_id] = {
            "id": webhook_id,
            "user_id": current_user.id,
            "url": webhook_config.url,
            "events": webhook_config.events,
            "secret": webhook_config.secret,
            "headers": webhook_config.headers or {},
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        logger.info(f"Webhook registered: {webhook_id} for user {current_user.id}")
        
        return {
            "webhook_id": webhook_id,
            "url": webhook_config.url,
            "events": webhook_config.events,
            "status": "registered",
            "test_endpoint": f"/api/ml-integration/webhooks/{webhook_id}/test"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.get("/webhooks/")
async def list_webhooks(
    current_user: User = Depends(get_api_key_user)
):
    """List all registered webhooks for the user"""
    
    user_webhooks = [
        webhook for webhook in registered_webhooks.values()
        if webhook["user_id"] == current_user.id and webhook["active"]
    ]
    
    return {
        "webhooks": user_webhooks,
        "total_count": len(user_webhooks)
    }

@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    current_user: User = Depends(get_api_key_user)
):
    """Delete a registered webhook"""
    
    if webhook_id not in registered_webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook = registered_webhooks[webhook_id]
    if webhook["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    webhook["active"] = False
    
    return {"message": "Webhook deleted successfully"}

@router.post("/webhooks/{webhook_id}/test")
async def test_webhook(
    webhook_id: str,
    current_user: User = Depends(get_api_key_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Test webhook endpoint with sample data"""
    
    if webhook_id not in registered_webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook = registered_webhooks[webhook_id]
    if webhook["user_id"] != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Send test event
    test_event = {
        "type": "webhook.test",
        "data": {
            "message": "This is a test webhook event from ModelShip",
            "timestamp": datetime.utcnow().isoformat(),
            "webhook_id": webhook_id
        }
    }
    
    background_tasks.add_task(send_webhook_event, webhook, test_event)
    
    return {"message": "Test webhook event sent"}

async def send_webhook_event(webhook: Dict[str, Any], event: Dict[str, Any]):
    """Send webhook event to registered URL"""
    
    try:
        import requests
        
        # Create signature
        payload = json.dumps(event)
        signature = hmac.new(
            webhook["secret"].encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "Content-Type": "application/json",
            "X-ModelShip-Signature": signature,
            "X-ModelShip-Event": event["type"],
            **webhook.get("headers", {})
        }
        
        response = requests.post(
            webhook["url"],
            data=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code >= 400:
            logger.error(f"Webhook delivery failed: {response.status_code} - {response.text}")
        else:
            logger.info(f"Webhook delivered successfully to {webhook['url']}")
            
    except Exception as e:
        logger.error(f"Webhook delivery error: {str(e)}")

@router.post("/models/register")
async def register_model(
    model_info: ModelRegistration,
    project_id: int = Query(...),
    current_user: User = Depends(get_api_key_user),
    db: Session = Depends(get_db)
):
    """Register a trained model back to ModelShip"""
    
    try:
        # Check access to project
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied to this project")
        
        # Store model information (in production, use proper model registry)
        model_record = {
            "id": hashlib.md5(f"{project_id}_{model_info.name}_{model_info.version}".encode()).hexdigest(),
            "project_id": project_id,
            "user_id": current_user.id,
            "name": model_info.name,
            "version": model_info.version,
            "framework": model_info.framework,
            "model_type": model_info.model_type,
            "metrics": model_info.metrics,
            "description": model_info.description,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        # In production, store in database
        logger.info(f"Model registered: {model_record['id']} for project {project_id}")
        
        # Trigger webhook for model registration
        for webhook in registered_webhooks.values():
            if webhook["user_id"] == current_user.id and "model.registered" in webhook["events"]:
                event = {
                    "type": "model.registered",
                    "project_id": project_id,
                    "model": model_record,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await send_webhook_event(webhook, event)
        
        return {
            "model_id": model_record["id"],
            "status": "registered",
            "message": "Model registered successfully",
            "model_info": model_record
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.get("/datasets/{project_id}/updates")
async def get_dataset_updates(
    project_id: int,
    since: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    current_user: User = Depends(get_api_key_user),
    db: Session = Depends(get_db)
):
    """Get dataset updates since a specific timestamp"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Build query
        query = db.query(Result).join(Job).filter(
            Job.project_id == project_id,
            Result.status == "success"
        )
        
        if since:
            query = query.filter(Result.created_at > since)
        
        # Get updates
        updates = query.order_by(Result.created_at.desc()).limit(limit).all()
        
        update_data = []
        for result in updates:
            update_data.append({
                "id": result.id,
                "filename": result.filename,
                "label": result.predicted_label,
                "confidence": result.confidence,
                "reviewed": result.reviewed,
                "ground_truth": result.ground_truth,
                "created_at": result.created_at.isoformat(),
                "file_path": result.file.file_path if result.file else None
            })
        
        return {
            "project_id": project_id,
            "updates": update_data,
            "count": len(update_data),
            "has_more": len(update_data) == limit,
            "latest_timestamp": update_data[0]["created_at"] if update_data else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dataset updates failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Updates failed: {str(e)}")

@router.post("/analytics/log-metrics")
async def log_training_metrics(
    project_id: int,
    metrics: Dict[str, float],
    epoch: Optional[int] = None,
    model_name: Optional[str] = None,
    current_user: User = Depends(get_api_key_user),
    db: Session = Depends(get_db)
):
    """Log training metrics from external ML platforms"""
    
    try:
        # Check access
        assignment = db.query(ProjectAssignment).filter(
            ProjectAssignment.project_id == project_id,
            ProjectAssignment.user_id == current_user.id
        ).first()
        
        if not assignment:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Store metrics in analytics table
        from models import Analytics
        
        for metric_name, metric_value in metrics.items():
            analytics_record = Analytics(
                project_id=project_id,
                user_id=current_user.id,
                metric_type=f"training.{metric_name}",
                metric_value=metric_value,
                metric_data={
                    "epoch": epoch,
                    "model_name": model_name,
                    "source": "external_ml_platform"
                },
                period_start=datetime.utcnow(),
                period_end=datetime.utcnow()
            )
            db.add(analytics_record)
        
        db.commit()
        
        logger.info(f"Training metrics logged for project {project_id}: {metrics}")
        
        return {
            "status": "logged",
            "project_id": project_id,
            "metrics_count": len(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics logging failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Logging failed: {str(e)}")

@router.get("/status")
async def get_integration_status():
    """Get status of ML platform integration features"""
    
    return {
        "ml_integration": {
            "status": "active",
            "version": "1.0",
            "features": {
                "real_time_streaming": True,
                "webhook_support": True,
                "model_registration": True,
                "analytics_logging": True,
                "multi_format_export": True
            },
            "supported_frameworks": {
                "tensorflow": "2.x",
                "pytorch": "1.x+",
                "huggingface": "4.x+",
                "scikit_learn": "1.x+",
                "mlflow": "2.x+"
            },
            "export_formats": [
                "json", "csv", "tensorflow", "pytorch", 
                "huggingface", "coco", "yolo", "pascal_voc"
            ],
            "streaming_protocols": ["HTTP/JSON", "WebSocket", "Polling"],
            "webhook_events": [
                "data.labeled", "project.completed", "quality.alert",
                "model.registered", "training.completed"
            ]
        },
        "rate_limits": {
            "streaming": "unlimited",
            "webhook_delivery": "10/second",
            "api_calls": "1000/minute",
            "dataset_access": "500/minute"
        },
        "documentation": {
            "api_guide": "/api/docs",
            "integration_examples": "/api/ml-integration/examples",
            "sdk_downloads": {
                "python": "pip install modelship-python",
                "javascript": "npm install modelship-sdk",
                "r": "install.packages('modelship')"
            }
        }
    } 