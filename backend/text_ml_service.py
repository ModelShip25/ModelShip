from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import time
import asyncio
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import uuid
import logging
from enum import Enum
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationType(Enum):
    """Types of text classification available"""
    SENTIMENT = "sentiment"                    # Positive/Negative/Neutral
    EMOTION = "emotion"                       # Joy/Anger/Fear/Sadness/etc
    TOPIC = "topic"                          # Business/Tech/Sports/Politics/etc
    SPAM = "spam"                            # Spam/Ham classification
    TOXICITY = "toxicity"                    # Toxic/Non-toxic content
    LANGUAGE = "language"                    # Language detection
    INTENT = "intent"                        # User intent classification
    NAMED_ENTITY = "named_entity"            # Person/Organization/Location/etc
    CUSTOM = "custom"                        # Custom categories

class TextMLService:
    """Advanced text classification and labeling service for research and production"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_metadata = {}
        
        # Initialize different text classification models
        self._initialize_text_models()
        
        # Performance tracking
        self.performance_stats = {
            "total_classifications": 0,
            "total_processing_time": 0.0,
            "model_usage": {},
            "error_count": 0
        }
    
    def _initialize_text_models(self):
        """Initialize various text classification models"""
        try:
            # Sentiment Analysis
            self.models["sentiment"] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_metadata["sentiment"] = {
                "name": "Sentiment Analysis",
                "type": "sentiment_classification",
                "categories": ["NEGATIVE", "NEUTRAL", "POSITIVE"],
                "description": "Classify text sentiment as positive, negative, or neutral",
                "use_cases": ["Social media monitoring", "Customer feedback", "Product reviews"]
            }
            
            # Emotion Detection
            self.models["emotion"] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_metadata["emotion"] = {
                "name": "Emotion Detection",
                "type": "emotion_classification", 
                "categories": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
                "description": "Detect emotional tone in text",
                "use_cases": ["Mental health research", "Customer service", "Content moderation"]
            }
            
            # Topic Classification
            self.models["topic"] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_metadata["topic"] = {
                "name": "Topic Classification",
                "type": "topic_classification",
                "categories": "Custom (user-defined)",
                "description": "Classify text into custom topic categories",
                "use_cases": ["Content categorization", "News classification", "Research paper sorting"]
            }
            
            # Spam Detection
            self.models["spam"] = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_metadata["spam"] = {
                "name": "Spam/Toxicity Detection",
                "type": "toxicity_classification",
                "categories": ["TOXIC", "NON_TOXIC"],
                "description": "Detect spam, toxic, or harmful content",
                "use_cases": ["Content moderation", "Email filtering", "Social media safety"]
            }
            
            # Language Detection
            self.models["language"] = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=0 if torch.cuda.is_available() else -1
            )
            self.model_metadata["language"] = {
                "name": "Language Detection",
                "type": "language_detection",
                "categories": "100+ languages",
                "description": "Automatically detect the language of text",
                "use_cases": ["Multilingual support", "Content routing", "Research analysis"]
            }
            
            logger.info("Text classification models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize text models: {str(e)}")
            raise
    
    async def classify_text_single(self,
                                 text: str,
                                 classification_type: str = "sentiment",
                                 custom_categories: Optional[List[str]] = None,
                                 include_metadata: bool = False) -> Dict[str, Any]:
        """
        Classify a single text with comprehensive analysis
        
        Args:
            text: Text to classify
            classification_type: Type of classification to perform
            custom_categories: Custom categories for topic classification
            include_metadata: Include detailed metadata in response
        """
        
        start_time = time.time()
        classification_id = str(uuid.uuid4())
        
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Text cannot be empty")
            
            if classification_type not in self.models and classification_type != "topic":
                raise ValueError(f"Classification type '{classification_type}' not available")
            
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Run classification based on type
            if classification_type == "topic" and custom_categories:
                # Zero-shot topic classification with custom categories
                model = self.models["topic"]
                results = model(processed_text, custom_categories)
                
                # Format results
                predictions = []
                for label, score in zip(results["labels"], results["scores"]):
                    predictions.append({
                        "label": label,
                        "confidence": float(score)
                    })
                
            else:
                # Standard classification
                model = self.models[classification_type]
                raw_results = model(processed_text)
                
                # Handle different result formats
                if isinstance(raw_results, list):
                    predictions = [
                        {
                            "label": result["label"],
                            "confidence": float(result["score"])
                        }
                        for result in raw_results
                    ]
                else:
                    predictions = [{
                        "label": raw_results["label"],
                        "confidence": float(raw_results["score"])
                    }]
            
            # Sort by confidence
            predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(classification_type, processing_time, True)
            
            # Build response
            response = {
                "classification_id": classification_id,
                "predicted_label": predictions[0]["label"],
                "confidence": round(predictions[0]["confidence"] * 100, 2),
                "processing_time": round(processing_time, 3),
                "classification_type": classification_type,
                "status": "success"
            }
            
            # Add detailed metadata for research/enterprise users
            if include_metadata:
                response.update({
                    "all_predictions": [
                        {
                            "label": pred["label"],
                            "confidence": round(pred["confidence"] * 100, 2)
                        }
                        for pred in predictions
                    ],
                    "text_metadata": {
                        "original_length": len(text),
                        "processed_length": len(processed_text),
                        "word_count": len(processed_text.split()),
                        "character_count": len(processed_text),
                        "language_detected": self._detect_basic_language(processed_text)
                    },
                    "model_metadata": self.model_metadata.get(classification_type, {}),
                    "processing_metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "preprocessing_applied": True,
                        "device_used": "gpu" if torch.cuda.is_available() else "cpu"
                    }
                })
            
            return response
            
        except Exception as e:
            self._update_performance_stats(classification_type, time.time() - start_time, False)
            logger.error(f"Text classification failed: {str(e)}")
            
            return {
                "classification_id": classification_id,
                "predicted_label": "classification_error",
                "confidence": 0.0,
                "processing_time": round(time.time() - start_time, 3),
                "classification_type": classification_type,
                "status": "error",
                "error_message": str(e)
            }
    
    async def classify_text_batch(self,
                                texts: List[str],
                                classification_type: str = "sentiment",
                                custom_categories: Optional[List[str]] = None,
                                batch_size: int = 16,
                                progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Classify multiple texts in batches for efficiency
        """
        
        total_texts = len(texts)
        results = []
        
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            # Process each text in the batch
            for text in batch_texts:
                try:
                    result = await self.classify_text_single(
                        text=text,
                        classification_type=classification_type,
                        custom_categories=custom_categories,
                        include_metadata=False
                    )
                    batch_results.append(result)
                    
                except Exception as e:
                    error_result = {
                        "classification_id": str(uuid.uuid4()),
                        "predicted_label": "error",
                        "confidence": 0.0,
                        "classification_type": classification_type,
                        "status": "error",
                        "error_message": str(e),
                        "text_preview": text[:100] + "..." if len(text) > 100 else text
                    }
                    batch_results.append(error_result)
            
            results.extend(batch_results)
            
            # Report progress
            if progress_callback:
                progress = min(i + batch_size, total_texts) / total_texts
                progress_callback(progress, f"Processed {min(i + batch_size, total_texts)}/{total_texts} texts")
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better classification"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Limit length to prevent memory issues
        max_length = 2048
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    def _detect_basic_language(self, text: str) -> str:
        """Basic language detection for metadata"""
        try:
            # Use language detection model if available
            if "language" in self.models:
                result = self.models["language"](text)
                return result[0]["label"] if result else "unknown"
        except:
            pass
        
        # Fallback to simple heuristics
        if re.search(r'[а-яё]', text.lower()):
            return "russian"
        elif re.search(r'[àâäéèêëïîôöùûüÿç]', text.lower()):
            return "french"
        elif re.search(r'[áéíóúñü]', text.lower()):
            return "spanish"
        else:
            return "english"
    
    def _update_performance_stats(self, classification_type: str, processing_time: float, success: bool):
        """Update performance tracking statistics"""
        self.performance_stats["total_classifications"] += 1
        self.performance_stats["total_processing_time"] += processing_time
        
        if classification_type not in self.performance_stats["model_usage"]:
            self.performance_stats["model_usage"][classification_type] = 0
        self.performance_stats["model_usage"][classification_type] += 1
        
        if not success:
            self.performance_stats["error_count"] += 1
    
    def get_available_classifications(self) -> Dict[str, Any]:
        """Get information about available text classification types"""
        return {
            "available_types": list(self.model_metadata.keys()),
            "type_details": self.model_metadata,
            "custom_categories_supported": ["topic"],
            "service_status": "operational"
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        total_classifications = self.performance_stats["total_classifications"]
        
        if total_classifications == 0:
            return {"message": "No text classifications performed yet"}
        
        return {
            "total_classifications": total_classifications,
            "average_processing_time": round(
                self.performance_stats["total_processing_time"] / total_classifications, 3
            ),
            "classification_type_usage": self.performance_stats["model_usage"],
            "error_rate": round(
                self.performance_stats["error_count"] / total_classifications * 100, 2
            ),
            "success_rate": round(
                (total_classifications - self.performance_stats["error_count"]) / 
                total_classifications * 100, 2
            )
        }

# Global text ML service instance
text_ml_service = TextMLService() 