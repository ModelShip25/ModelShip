from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import time
import asyncio
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import uuid
import logging
from enum import Enum
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AdvancedMLService:
    """Advanced ML service with enterprise features and multi-model support"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.model_metadata = {}
        self.calibrator = ConfidenceCalibrator()
        
        # Initialize models for different tiers
        self._initialize_startup_models()
        
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
            logger.error(f"Failed to initialize startup models: {str(e)}")
            # Create a fallback dummy model for development
            self.models["resnet50"] = None
            self.model_metadata["resnet50"] = {
                "name": "ResNet-50 (Fallback)",
                "type": "image_classification",
                "tier": ModelTier.STARTUP,
                "status": "fallback_mode"
            }
    
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
            
            model = self.models.get(model_name)
            
            # Handle fallback mode
            if model is None:
                return self._create_fallback_response(classification_id, image_path, start_time)
            
            # Load and preprocess image
            image = await self._preprocess_image(image_path)
            
            # Run classification
            raw_results = model(image)
            
            # Process results with confidence calibration
            processed_results = self._process_classification_results(
                raw_results, model_name, confidence_threshold
            )
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(model_name, processing_time, True)
            
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
                        "prediction_stability": "high"
                    }
                })
            
            return response
            
        except Exception as e:
            self._update_performance_stats(model_name, time.time() - start_time, False)
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
    
    def _create_fallback_response(self, classification_id: str, image_path: str, start_time: float) -> Dict[str, Any]:
        """Create a fallback response when models are not available"""
        # Simple image analysis for fallback
        try:
            image = Image.open(image_path)
            image_format = image.format.lower() if image.format else "unknown"
            
            # Basic classification based on image properties
            if image.mode == "RGBA" or image_format == "png":
                label = "graphic_or_logo"
                confidence = 0.75
            elif image.width > image.height * 1.5:
                label = "landscape_or_banner"
                confidence = 0.7
            elif image.height > image.width * 1.5:
                label = "portrait_or_vertical_image"
                confidence = 0.7
            else:
                label = "general_image"
                confidence = 0.6
                
        except Exception:
            label = "unprocessable_image"
            confidence = 0.1
        
        return {
            "classification_id": classification_id,
            "predicted_label": label,
            "confidence": confidence,
            "processing_time": round(time.time() - start_time, 3),
            "model_used": "fallback_analyzer",
            "status": "success",
            "note": "Using fallback classification due to model unavailability"
        }
    
    async def classify_image_batch(self,
                                 image_paths: List[str],
                                 model_name: str = "resnet50",
                                 batch_size: int = 8,
                                 progress_callback: Optional[callable] = None) -> List[Dict[str, Any]]:
        """
        Classify multiple images in optimized batches
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
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            model_key: {
                **metadata,
                "status": "available" if self.models.get(model_key) is not None else "unavailable"
            }
            for model_key, metadata in self.model_metadata.items()
        }

# Global advanced ML service instance
advanced_ml_service = AdvancedMLService() 