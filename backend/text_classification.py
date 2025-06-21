from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result
from auth import get_current_user, get_optional_user
from text_ml_service import text_ml_service, TextClassificationType
from typing import List, Dict, Any, Optional
import asyncio
import time
import uuid
import os
import pandas as pd
from datetime import datetime
import logging

router = APIRouter(prefix="/api/classify/text", tags=["text_classification"])

logger = logging.getLogger(__name__)

class TextClassificationService:
    """Service class handling text classification business logic"""
    
    def __init__(self):
        self.text_ml_service = text_ml_service
    
    async def process_text_classification_job(self, job_id: int, texts_data: List[Dict], classification_type: str, custom_categories: Optional[List[str]], db: Session):
        """Process text classification job in background"""
        try:
            # Update job status to processing
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "processing"
            db.commit()
            
            processed_count = 0
            
            # Extract texts for batch processing
            texts = [text_data["text"] for text_data in texts_data]
            
            def progress_callback(progress: float, message: str):
                # Update job progress in database
                job.completed_items = int(progress * len(texts_data))
                db.commit()
            
            # Use batch text classification
            batch_results = await self.text_ml_service.classify_text_batch(
                texts=texts,
                classification_type=classification_type,
                custom_categories=custom_categories,
                batch_size=16,
                progress_callback=progress_callback
            )
            
            # Save results to database
            for i, result in enumerate(batch_results):
                text_data = texts_data[i]
                
                db_result = Result(
                    job_id=job_id,
                    file_id=text_data.get("file_id"),
                    filename=text_data.get("filename", f"text_{i+1}.txt"),
                    predicted_label=result["predicted_label"],
                    confidence=result["confidence"] / 100.0,  # Convert percentage to decimal
                    processing_time=result["processing_time"],
                    status=result["status"],
                    error_message=result.get("error_message")
                )
                
                db.add(db_result)
                processed_count += 1
            
            # Mark job as completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.total_items = len(texts_data)
            job.completed_items = processed_count
            db.commit()
            
            logger.info(f"Completed text classification job {job_id}: {processed_count} texts processed")
            
        except Exception as e:
            # Mark job as failed
            job = db.query(Job).filter(Job.id == job_id).first()
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            logger.error(f"Text classification job {job_id} failed: {str(e)}")

# Global service instance
text_classification_service = TextClassificationService()

@router.get("/types")
async def get_text_classification_types():
    """Get available text classification types for research labs and users"""
    try:
        available_types = text_ml_service.get_available_classifications()
        
        return {
            "available_types": available_types,
            "use_cases": {
                "sentiment": "Customer feedback analysis, social media monitoring, product reviews",
                "emotion": "Mental health research, customer service optimization, content analysis",
                "topic": "Content categorization, research paper sorting, news classification", 
                "spam": "Email filtering, content moderation, spam detection",
                "language": "Multilingual support, content routing, research analysis",
                "toxicity": "Social media safety, content moderation, community management"
            },
            "research_applications": {
                "sentiment": "Psychology studies, market research, political analysis",
                "emotion": "Mental health diagnostics, therapy effectiveness, behavioral studies",
                "topic": "Literature review automation, research categorization, knowledge mining",
                "language": "Linguistic research, multilingual corpus analysis, translation studies"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get classification types: {str(e)}")

@router.post("/single")
async def classify_single_text(
    text: str = Form(...),
    classification_type: str = Form(default="sentiment"),
    custom_categories: Optional[str] = Form(default=None),  # Comma-separated categories
    include_metadata: bool = Form(default=False),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Classify a single text - perfect for research labs and quick testing"""
    
    # Check user credits
    if current_user.credits_remaining < 1:
        raise HTTPException(status_code=402, detail="Insufficient credits")
    
    # Validate classification type
    available_types = text_ml_service.get_available_classifications()
    if classification_type not in available_types["available_types"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid classification type. Available: {available_types['available_types']}"
        )
    
    try:
        # Parse custom categories if provided
        categories_list = None
        if custom_categories and classification_type == "topic":
            categories_list = [cat.strip() for cat in custom_categories.split(",")]
        
        # Run classification
        result = await text_ml_service.classify_text_single(
            text=text,
            classification_type=classification_type,
            custom_categories=categories_list,
            include_metadata=include_metadata
        )
        
        # Deduct credit
        current_user.credits_remaining -= 1
        db.commit()
        
        return {
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "classification_id": result["classification_id"],
            "classification_type": classification_type,
            "credits_remaining": current_user.credits_remaining,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            **({"metadata": result.get("text_metadata", {})} if include_metadata else {}),
            **({"all_predictions": result.get("all_predictions", [])} if include_metadata else {})
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text classification failed: {str(e)}")

@router.post("/quick")
async def classify_quick_text(
    text: str = Form(...),
    classification_type: str = Form(default="sentiment"),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Quick text classification without authentication - for frictionless research experience"""
    
    # Validate text length for free tier
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long for quick classification (max 1000 characters)")
    
    try:
        # Run classification
        result = await text_ml_service.classify_text_single(
            text=text,
            classification_type=classification_type,
            include_metadata=False
        )
        
        return {
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "processing_time": result["processing_time"],
            "classification_type": classification_type,
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "note": "Sign up for batch processing and advanced features"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text classification failed: {str(e)}")

@router.post("/batch")
async def create_batch_text_classification_job(
    classification_type: str = Form(default="sentiment"),
    custom_categories: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    texts: Optional[str] = Form(default=None),  # JSON array of texts
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create batch text classification job for research datasets"""
    
    if not file and not texts:
        raise HTTPException(status_code=400, detail="Either file or texts must be provided")
    
    try:
        texts_data = []
        
        # Process file upload
        if file:
            # Validate file type
            if file.content_type not in ["text/plain", "text/csv", "application/json"]:
                raise HTTPException(status_code=400, detail="Invalid file type. Use .txt, .csv, or .json")
            
            # Save file temporarily
            temp_filename = f"temp_{uuid.uuid4()}_{file.filename}"
            temp_path = os.path.join("uploads", temp_filename)
            os.makedirs("uploads", exist_ok=True)
            
            contents = await file.read()
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            # Parse file based on type
            if file.filename.endswith('.csv'):
                df = pd.read_csv(temp_path)
                if 'text' not in df.columns:
                    raise HTTPException(status_code=400, detail="CSV must have a 'text' column")
                
                for idx, row in df.iterrows():
                    texts_data.append({
                        "text": str(row['text']),
                        "filename": f"{file.filename}_row_{idx+1}"
                    })
            
            elif file.filename.endswith('.json'):
                import json
                with open(temp_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        if isinstance(item, dict) and 'text' in item:
                            texts_data.append({
                                "text": str(item['text']),
                                "filename": f"{file.filename}_item_{idx+1}"
                            })
                        elif isinstance(item, str):
                            texts_data.append({
                                "text": item,
                                "filename": f"{file.filename}_item_{idx+1}"
                            })
            
            else:  # .txt file
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split by lines for batch processing
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                for idx, line in enumerate(lines):
                    texts_data.append({
                        "text": line,
                        "filename": f"{file.filename}_line_{idx+1}"
                    })
            
            # Clean up temp file
            os.remove(temp_path)
        
        # Process direct text input
        elif texts:
            import json
            try:
                texts_list = json.loads(texts)
                for idx, text in enumerate(texts_list):
                    texts_data.append({
                        "text": str(text),
                        "filename": f"direct_input_{idx+1}"
                    })
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for texts")
        
        if not texts_data:
            raise HTTPException(status_code=400, detail="No valid texts found to process")
        
        # Check credits
        if current_user.credits_remaining < len(texts_data):
            raise HTTPException(
                status_code=402, 
                detail=f"Insufficient credits. Need {len(texts_data)}, have {current_user.credits_remaining}"
            )
        
        # Create job
        job = Job(
            user_id=current_user.id,
            job_type="text",
            status="queued",
            total_items=len(texts_data)
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        # Parse custom categories
        categories_list = None
        if custom_categories and classification_type == "topic":
            categories_list = [cat.strip() for cat in custom_categories.split(",")]
        
        # Start background processing
        background_tasks.add_task(
            text_classification_service.process_text_classification_job,
            job.id, texts_data, classification_type, categories_list, db
        )
        
        # Deduct credits
        current_user.credits_remaining -= len(texts_data)
        db.commit()
        
        return {
            "job_id": job.id,
            "status": job.status,
            "total_items": job.total_items,
            "classification_type": classification_type,
            "custom_categories": categories_list,
            "message": f"Batch text classification job created with {len(texts_data)} texts",
            "credits_remaining": current_user.credits_remaining
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch job: {str(e)}")

@router.get("/research-templates")
async def get_research_templates():
    """Get pre-defined templates for common research use cases"""
    
    return {
        "templates": {
            "sentiment_analysis": {
                "name": "Sentiment Analysis Research",
                "description": "Analyze sentiment in social media, reviews, or survey responses",
                "classification_type": "sentiment",
                "categories": ["POSITIVE", "NEGATIVE", "NEUTRAL"],
                "use_cases": ["Social media research", "Customer feedback analysis", "Political sentiment"],
                "sample_input": "I love this new product! It works perfectly and saves me so much time."
            },
            "emotion_detection": {
                "name": "Emotion Detection Study",
                "description": "Detect emotional states in text for psychological research",
                "classification_type": "emotion",
                "categories": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "neutral"],
                "use_cases": ["Mental health research", "Therapy analysis", "Behavioral studies"],
                "sample_input": "I'm feeling overwhelmed and anxious about the upcoming presentation."
            },
            "topic_classification": {
                "name": "Custom Topic Classification",
                "description": "Classify texts into custom research categories",
                "classification_type": "topic",
                "categories": "User-defined (e.g., 'Politics, Technology, Health, Education')",
                "use_cases": ["Content analysis", "Literature review", "Research categorization"],
                "sample_input": "The new AI model shows promising results in medical diagnosis applications."
            },
            "content_moderation": {
                "name": "Content Moderation Research",
                "description": "Detect harmful or inappropriate content for safety research",
                "classification_type": "toxicity",
                "categories": ["TOXIC", "NON_TOXIC"],
                "use_cases": ["Online safety research", "Content policy development", "Digital wellbeing"],
                "sample_input": "This is a neutral comment about technology research."
            },
            "multilingual_analysis": {
                "name": "Language Detection Study",
                "description": "Identify languages in multilingual datasets",
                "classification_type": "language",
                "categories": "100+ languages supported",
                "use_cases": ["Multilingual corpus analysis", "Translation studies", "Global communication research"],
                "sample_input": "Hello world! Bonjour le monde! Hola mundo!"
            }
        },
        "getting_started": {
            "step_1": "Choose a template that matches your research needs",
            "step_2": "Upload your text data (CSV, JSON, or TXT format)",
            "step_3": "Configure classification parameters",
            "step_4": "Review auto-generated labels using our human-in-the-loop system",
            "step_5": "Export labeled data in your preferred format"
        }
    }

@router.get("/performance")
async def get_text_classification_performance():
    """Get performance statistics for text classification service"""
    try:
        performance_stats = text_ml_service.get_performance_stats()
        available_types = text_ml_service.get_available_classifications()
        
        return {
            "performance_stats": performance_stats,
            "service_status": "operational",
            "available_classifications": available_types["available_types"],
            "supported_languages": "100+ languages for language detection",
            "max_text_length": "2048 characters per text",
            "batch_size_limit": "1000 texts per batch job"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}") 