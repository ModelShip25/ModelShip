from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import time
import asyncio
from typing import List, Dict, Any, Union, Optional, Callable
from datetime import datetime
import uuid
import logging
from text_ml_service_fixed import FixedTextMLService
from sqlalchemy.orm import Session
from datetime import timedelta
from config import settings
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for model outputs to avoid redundant processing
MODEL_CACHE = {}
MODEL_CACHE_SIZE = 100  # Max number of results to cache
MODEL_CACHE_EXPIRY = 3600  # Cache expiry in seconds

class ModelTier(Enum):
    """Model tiers for different customer segments"""
    STARTUP = "startup"      # Basic pre-trained models
    ENTERPRISE = "enterprise" # Advanced models + custom training
    INDUSTRY = "industry"    # Domain-specific specialized models

class ConfidenceCalibrator:
    """Calibrate model confidence scores for better reliability"""
    
    @staticmethod
    def calibrate_confidence(raw_confidence: float, model_name: str) -> float:
        """Calibrate confidence based on model characteristics"""
        # Temperature scaling based on model type
        temperature_map = {
            "microsoft/resnet-50": 1.2,
            "google/vit-base-patch16-224": 1.1,
            "facebook/convnext-base-224": 1.15
        }
        
        temperature = temperature_map.get(model_name, 1.0)
        calibrated = raw_confidence ** (1/temperature)
        
        # Ensure bounds
        return max(0.0, min(1.0, calibrated))

class ModelHealthMonitor:
    """Monitor model health and performance"""
    
    def __init__(self):
        self.model_status = {}
        self.health_check_interval = 60 * 60  # 1 hour
        self.last_checked = {}
    
    def record_prediction_result(self, model_name: str, success: bool, latency: float):
        """Record prediction success/failure for health monitoring"""
        if model_name not in self.model_status:
            self.model_status[model_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "latencies": [],
                "health_status": "healthy"
            }
        
        status = self.model_status[model_name]
        status["total_requests"] += 1
        
        if success:
            status["successful_requests"] += 1
        else:
            status["failed_requests"] += 1
        
        status["latencies"].append(latency)
        if len(status["latencies"]) > 100:
            status["latencies"].pop(0)  # Keep only last 100 latencies
        
        # Update health status
        self._update_health_status(model_name)
    
    def _update_health_status(self, model_name: str):
        """Update health status based on recent performance"""
        status = self.model_status[model_name]
        
        if status["total_requests"] < 5:
            return  # Not enough data
        
        # Calculate error rate
        error_rate = status["failed_requests"] / status["total_requests"]
        
        # Calculate latency p95
        latencies = sorted(status["latencies"])
        p95_index = int(len(latencies) * 0.95)
        p95_latency = latencies[p95_index] if latencies else 0
        
        # Update health status
        if error_rate > 0.5:
            status["health_status"] = "critical"
        elif error_rate > 0.2:
            status["health_status"] = "unhealthy"
        elif p95_latency > 10.0:  # More than 10 seconds for p95
            status["health_status"] = "degraded"
        else:
            status["health_status"] = "healthy"
    
    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get health status for a specific model"""
        if model_name not in self.model_status:
            return {"health_status": "unknown", "message": "No data available"}
        
        status = self.model_status[model_name]
        
        # Calculate average latency
        avg_latency = sum(status["latencies"]) / len(status["latencies"]) if status["latencies"] else 0
        
        return {
            "health_status": status["health_status"],
            "total_requests": status["total_requests"],
            "error_rate": round(status["failed_requests"] / status["total_requests"] * 100, 2) if status["total_requests"] > 0 else 0,
            "average_latency": round(avg_latency, 3),
            "p95_latency": round(sorted(status["latencies"])[int(len(status["latencies"]) * 0.95)] if len(status["latencies"]) > 0 else 0, 3)
        }
    
    def get_all_model_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all models"""
        return {model: self.get_model_health(model) for model in self.model_status}

class AdvancedMLService:
    """Advanced ML service with enterprise features and multi-model support"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.model_metadata = {}
        self.calibrator = ConfidenceCalibrator()
        self.health_monitor = ModelHealthMonitor()
        
        # Initialize models for different tiers
        self._initialize_startup_models()
        self._initialize_enterprise_models()
        
        # Performance tracking
        self.performance_stats = {
            "total_classifications": 0,
            "total_processing_time": 0.0,
            "model_usage": {},
            "error_count": 0
        }
    
    def _initialize_startup_models(self):
        """Initialize models for startup tier"""
        try:
            # Set device and suppress warnings
            device = 0 if torch.cuda.is_available() else -1
            if device == -1:
                print("Device set to use cpu")
            
            # Primary image classification model with fast processor
            self.models["resnet50"] = pipeline(
                "image-classification", 
                model="microsoft/resnet-50",
                device=device,
                use_fast=True  # Use fast processor to avoid warnings
            )
        
            self.model_metadata["resnet50"] = {
                "name": "ResNet-50",
                "type": "image_classification",
                "tier": ModelTier.STARTUP,
                "categories": 1000,
                "accuracy": 0.76,
                "avg_processing_time": 0.5,
                "supported_formats": ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
                "description": "General-purpose image classification with 1000 ImageNet categories"
            }
            
            logger.info("Startup models initialized successfully")
            
        except Exception as e:
            error_logger = logging.getLogger("errors")
            error_logger.error(f"Failed to initialize startup models: {str(e)}")
            raise
    
    def _initialize_enterprise_models(self):
        """Initialize additional models for enterprise tier"""
        try:
            # Set device
            device = 0 if torch.cuda.is_available() else -1
            
            # Vision Transformer for higher accuracy with fast processor
            self.models["vit"] = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=device,
                use_fast=True  # Use fast processor to avoid warnings
            )
            
            self.model_metadata["vit"] = {
                "name": "Vision Transformer",
                "type": "image_classification", 
                "tier": ModelTier.ENTERPRISE,
                "categories": 1000,
                "accuracy": 0.85,
                "avg_processing_time": 0.8,
                "supported_formats": ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
                "description": "State-of-the-art transformer-based image classification"
            }
            
            logger.info("Enterprise models initialized successfully")
            
        except Exception as e:
            logger.warning(f"Enterprise models initialization failed: {str(e)}")
    
    async def classify_text_single(self, text: str, 
                                   classification_type: str = "sentiment", 
                                   custom_categories: Optional[List[str]] = None, 
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        try:
            response = await text_ml_service.classify_text_single(text, classification_type, custom_categories)
            if progress_callback:
                progress_callback(1.0, f"Processed text for {classification_type}")
            return response
        except Exception as e:
            logger.error(f"Error during text classification: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def classify_text_batch(self, texts: List[str], 
                                 classification_type: str = "sentiment", 
                                 custom_categories: Optional[List[str]] = None, 
                                 batch_size: int = 16, 
                                 progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        results = await text_ml_service.classify_text_batch(texts, 
                                classification_type, custom_categories, batch_size, progress_callback)
        return results
    async def classify_image_single(self, 
                                  image_path: str, 
                                  model_name: str = "resnet50",
                                  include_metadata: bool = False,
                                  confidence_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Classify a single image with advanced features
        
        Args:
            image_path: Path to the image file
            model_name: Model to use for classification
            include_metadata: Include detailed metadata in response
            confidence_threshold: Minimum confidence for predictions
        """
        
        start_time = time.time()
        classification_id = str(uuid.uuid4())
        
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not available")
            
            # Load and preprocess image
            image = await self._preprocess_image(image_path)
            
            # Check cache for this image
            cache_key = f"{image_path}:{model_name}"
            if cache_key in MODEL_CACHE:
                cache_entry = MODEL_CACHE[cache_key]
                # Check if cache entry is still valid
                if time.time() - cache_entry["timestamp"] < MODEL_CACHE_EXPIRY:
                    logger.info(f"Using cached result for {image_path}")
                    raw_results = cache_entry["results"]
                else:
                    # Cache expired, remove it
                    del MODEL_CACHE[cache_key]
            
            # If not in cache or expired, run the model
            if cache_key not in MODEL_CACHE:
                model = self.models[model_name]
                raw_results = model(image)
                
                # Update cache
                MODEL_CACHE[cache_key] = {
                    "results": raw_results,
                    "timestamp": time.time()
                }
                
                # Manage cache size
                if len(MODEL_CACHE) > MODEL_CACHE_SIZE:
                    # Remove oldest entry
                    oldest_key = min(MODEL_CACHE.keys(), key=lambda k: MODEL_CACHE[k]["timestamp"])
                    del MODEL_CACHE[oldest_key]
            
            # Process results with confidence calibration
            processed_results = self._process_classification_results(
                raw_results, model_name, confidence_threshold
            )
            
            processing_time = time.time() - start_time
            
            # Update performance stats and health monitoring
            self._update_performance_stats(model_name, processing_time, True)
            self.health_monitor.record_prediction_result(model_name, True, processing_time)
            
            # Build response
            response = {
                "classification_id": classification_id,
                "predicted_label": processed_results["top_prediction"]["label"],
                "confidence": processed_results["top_prediction"]["confidence"],
                "processing_time": round(processing_time, 3),
                "model_used": model_name,
                "status": "success"
            }
            
            # Add detailed metadata for enterprise customers
            if include_metadata:
                response.update({
                    "top_5_predictions": processed_results["top_5"],
                    "model_metadata": self.model_metadata[model_name],
                    "image_metadata": await self._get_image_metadata(image_path),
                    "processing_metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "confidence_calibrated": True,
                        "preprocessing_steps": ["resize", "normalize", "tensor_conversion"],
                        "device_used": "gpu" if torch.cuda.is_available() else "cpu"
                    },
                    "quality_metrics": {
                        "prediction_entropy": self._calculate_entropy(processed_results["all_predictions"]),
                        "confidence_calibration_score": self._get_calibration_score(processed_results),
                        "prediction_stability": "high"  # TODO: Implement actual stability check
                    }
                })
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_performance_stats(model_name, processing_time, False)
            self.health_monitor.record_prediction_result(model_name, False, processing_time)
            logger.error(f"Classification failed for {image_path}: {str(e)}")
            
            return {
                "classification_id": classification_id,
                "predicted_label": "classification_error",
                "confidence": 0.0,
                "processing_time": round(time.time() - start_time, 3),
                "model_used": model_name,
                "status": "error",
                "error_message": str(e)
            }
    
    async def classify_image_batch(self,
                                 image_paths: List[str],
                                 model_name: str = "resnet50",
                                 batch_size: int = 8,
                                 progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Classify multiple images in optimized batches
        
        Args:
            image_paths: List of image file paths
            model_name: Model to use for classification
            batch_size: Number of images to process simultaneously
            progress_callback: Optional callback for progress updates
        """
        
        results = []
        total_images = len(image_paths)
        
        # Process in batches for memory efficiency
        for i in range(0, total_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.classify_image_single(path, model_name, include_metadata=False)
                for path in batch_paths
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions in batch
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_result = {
                        "classification_id": str(uuid.uuid4()),
                        "predicted_label": "batch_processing_error",
                        "confidence": 0.0,
                        "processing_time": 0.0,
                        "model_used": model_name,
                        "status": "error",
                        "error_message": str(result),
                        "image_path": batch_paths[j]
                    }
                    results.append(error_result)
                else:
                    results.append(result)
            
            # Report progress
            if progress_callback:
                progress = min(i + batch_size, total_images) / total_images
                progress_callback(progress, f"Processed {min(i + batch_size, total_images)}/{total_images} images")
        
        return results
    
    async def _preprocess_image(self, image_path: str) -> Image.Image:
        """Advanced image preprocessing with error handling"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image dimensions (prevent memory issues)
            max_dimension = 4096
            if image.width > max_dimension or image.height > max_dimension:
                # Resize while maintaining aspect ratio
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            raise
    
    def _process_classification_results(self, 
                                      raw_results: List[Dict],
                                      model_name: str,
                                      confidence_threshold: float) -> Dict[str, Any]:
        """Process and calibrate classification results"""
        
        # Calibrate confidence scores
        calibrated_results = []
        for result in raw_results:
            calibrated_confidence = self.calibrator.calibrate_confidence(
                result["score"], model_name
            )
            
            if calibrated_confidence >= confidence_threshold:
                calibrated_results.append({
                    "label": result["label"],
                    "confidence": calibrated_confidence,
                    "raw_confidence": result["score"]
                })
        
        # Sort by calibrated confidence
        calibrated_results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "top_prediction": calibrated_results[0] if calibrated_results else {
                "label": "low_confidence_prediction", 
                "confidence": 0.0,
                "raw_confidence": 0.0
            },
            "top_5": calibrated_results[:5],
            "all_predictions": calibrated_results
        }
    
    async def _get_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract detailed image metadata"""
        try:
            image = Image.open(image_path)
            file_size = os.path.getsize(image_path)
            
            return {
                "filename": os.path.basename(image_path),
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }
        except Exception:
            return {"error": "Could not extract image metadata"}
    
    def _calculate_entropy(self, predictions: List[Dict]) -> float:
        """Calculate prediction entropy for uncertainty estimation"""
        if not predictions:
            return 0.0
        
        import math
        entropy = 0.0
        for pred in predictions:
            conf = pred["confidence"]
            if conf > 0:
                entropy -= conf * math.log2(conf)
        
        return round(entropy, 3)
    
    def _get_calibration_score(self, results: Dict) -> float:
        """Get confidence calibration quality score"""
        # Simple calibration score based on top prediction confidence
        top_conf = results["top_prediction"]["confidence"]
        raw_conf = results["top_prediction"].get("raw_confidence", top_conf)
        
        calibration_adjustment = abs(top_conf - raw_conf)
        return round(1.0 - calibration_adjustment, 3)
    
    def _update_performance_stats(self, model_name: str, processing_time: float, success: bool):
        """Update performance tracking statistics"""
        self.performance_stats["total_classifications"] += 1
        self.performance_stats["total_processing_time"] += processing_time
        
        if model_name not in self.performance_stats["model_usage"]:
            self.performance_stats["model_usage"][model_name] = 0
        self.performance_stats["model_usage"][model_name] += 1
        
        if not success:
            self.performance_stats["error_count"] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        total_classifications = self.performance_stats["total_classifications"]
        
        if total_classifications == 0:
            return {"message": "No classifications performed yet"}
        
        return {
            "total_classifications": total_classifications,
            "average_processing_time": round(
                self.performance_stats["total_processing_time"] / total_classifications, 3
            ),
            "model_usage": self.performance_stats["model_usage"],
            "error_rate": round(
                self.performance_stats["error_count"] / total_classifications * 100, 2
            ),
            "success_rate": round(
                (total_classifications - self.performance_stats["error_count"]) / 
                total_classifications * 100, 2
            )
        }
    
    def get_health_check(self) -> Dict[str, Any]:
        """Get health status for all models"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",  # TODO: Aggregate from individual models
            "models": self.health_monitor.get_all_model_health()
        }

    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models with health status"""
        models_info = {}
        
        for model_key, metadata in self.model_metadata.items():
            health_status = "unknown"
            if model_key in self.models:
                health_status = self.health_monitor.get_model_health(model_key).get("health_status", "unknown")
            
            models_info[model_key] = {
                **metadata,
                "status": "available" if model_key in self.models else "unavailable",
                "health_status": health_status
            }
        
        return models_info

# Legacy compatibility wrapper
class MLService:
    """Legacy wrapper for backward compatibility"""
    
    def __init__(self):
        self.advanced_service = AdvancedMLService()
    
    def classify_image(self, image_path: str) -> dict:
        """Legacy method - synchronous image classification"""
        try:
            # Use asyncio to run the async method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.advanced_service.classify_image_single(image_path)
            )
            loop.close()
            
            return {
                "label": result["predicted_label"],
                "confidence": result["confidence"]
            }
        except Exception as e:
            return {
                "label": "classification_error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def classify_text(self, text: str) -> dict:
        """Legacy text classification - placeholder"""
        return {
            "label": "text_classification_not_implemented",
            "confidence": 0.0
        }
    
    def get_supported_image_formats(self) -> list:
        """Return list of supported image formats"""
        return ["jpg", "jpeg", "png", "gif", "webp", "bmp"]
    
    def get_supported_text_formats(self) -> list:
        """Return list of supported text formats"""
        return ["txt", "csv"]

# Global ML service instances
# Add Text Classification Service
text_ml_service = FixedTextMLService()  # Use the fixed, lightweight service for text

ml_service = MLService()  # Legacy compatibility
advanced_ml_service = AdvancedMLService()  # New advanced service
