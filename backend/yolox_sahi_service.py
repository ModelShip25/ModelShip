import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging
import json

# YOLOX + SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoloxSahiService:
    """Advanced object detection service using YOLOX + SAHI for small object detection"""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self._initialize_models()
        
        # Performance tracking
        self.stats = {
            "total_detections": 0,
            "total_objects_detected": 0,
            "total_processing_time": 0.0,
            "model_usage": {}
        }
        
        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Class-specific confidence thresholds for better filtering
        self.class_confidence_thresholds = {
            'person': 0.3, 'car': 0.25, 'truck': 0.25, 'bus': 0.25,
            'bicycle': 0.2, 'motorcycle': 0.2, 'dog': 0.3, 'cat': 0.3,
            'bird': 0.2, 'bottle': 0.15, 'cup': 0.15, 'cell phone': 0.2
        }
    
    def _initialize_models(self):
        """Initialize YOLOX models with SAHI support"""
        try:
            # Set device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # YOLOX Nano - fast and lightweight
            self.models["yolox-nano"] = UltralyticsDetectionModel(
                model_path="yolox_nano.pth",
                config_path=None,
                device=device,
                mask_threshold=0.5,
                confidence_threshold=0.25,
                category_mapping=None,
                category_remapping=None,
                load_at_init=True,
                image_size=416
            )
            
            self.model_info["yolox-nano"] = {
                "name": "YOLOX Nano",
                "speed": "very_fast",
                "accuracy": "good",
                "classes": 80,
                "description": "Ultra-fast object detection optimized for real-time applications",
                "recommended_confidence": 0.25,
                "image_size": 416
            }
            
            # YOLOX Small - balanced speed and accuracy
            self.models["yolox-small"] = UltralyticsDetectionModel(
                model_path="yolox_s.pth",
                config_path=None,
                device=device,
                mask_threshold=0.5,
                confidence_threshold=0.2,
                category_mapping=None,
                category_remapping=None,
                load_at_init=True,
                image_size=640
            )
            
            self.model_info["yolox-small"] = {
                "name": "YOLOX Small",
                "speed": "fast",
                "accuracy": "very_good",
                "classes": 80,
                "description": "Balanced speed and accuracy for general-purpose detection",
                "recommended_confidence": 0.2,
                "image_size": 640
            }
            
            # YOLOX Medium - higher accuracy
            self.models["yolox-medium"] = UltralyticsDetectionModel(
                model_path="yolox_m.pth",
                config_path=None,
                device=device,
                mask_threshold=0.5,
                confidence_threshold=0.15,
                category_mapping=None,
                category_remapping=None,
                load_at_init=True,
                image_size=640
            )
            
            self.model_info["yolox-medium"] = {
                "name": "YOLOX Medium",
                "speed": "medium",
                "accuracy": "excellent",
                "classes": 80,
                "description": "High accuracy detection for demanding applications",
                "recommended_confidence": 0.15,
                "image_size": 640
            }
            
            logger.info("YOLOX+SAHI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLOX models: {str(e)}")
            # Initialize with placeholder for development
            self.models = {}
            self.model_info = {}
    
    async def detect_objects(self, 
                           image_path: str,
                           model_name: str = "yolox-small",
                           confidence_threshold: float = 0.25,
                           annotate_image: bool = True,
                           save_annotated: bool = True,
                           use_sahi_slicing: bool = True,
                           slice_height: int = 640,
                           slice_width: int = 640,
                           overlap_height_ratio: float = 0.2,
                           overlap_width_ratio: float = 0.2,
                           project_id: int = 1) -> Dict[str, Any]:
        """
        Enhanced object detection using YOLOX + SAHI with sliced inference
        
        Args:
            image_path: Path to input image
            model_name: Model to use (yolox-nano, yolox-small, yolox-medium)
            confidence_threshold: Confidence threshold for filtering detections
            annotate_image: Whether to draw bounding boxes and labels
            save_annotated: Whether to save annotated image
            use_sahi_slicing: Whether to use SAHI sliced inference
            slice_height: Height of each slice for SAHI
            slice_width: Width of each slice for SAHI
            overlap_height_ratio: Overlap ratio for height
            overlap_width_ratio: Overlap ratio for width
            project_id: Project ID for saving annotated images
        """
        
        start_time = time.time()
        detection_id = str(uuid.uuid4())
        
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if model_name not in self.models and len(self.models) > 0:
                raise ValueError(f"Model '{model_name}' not available")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Get model or use mock for development
            if len(self.models) == 0:
                # Mock response for development when models aren't loaded
                return await self._create_mock_response(image_path, detection_id, start_time)
            
            model = self.models[model_name]
            
            # Perform detection with SAHI
            if use_sahi_slicing:
                # Use SAHI sliced prediction for better small object detection
                result = get_sliced_prediction(
                    image=image_path,
                    detection_model=model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    postprocess_type="GREEDYNMMS",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=False,
                    verbose=1
                )
            else:
                # Standard inference without slicing
                result = get_sliced_prediction(
                    image=image_path,
                    detection_model=model,
                    slice_height=height,
                    slice_width=width,
                    overlap_height_ratio=0,
                    overlap_width_ratio=0,
                    verbose=1
                )
            
            # Process detections
            detections = []
            total_objects = 0
            
            for detection in result.object_prediction_list:
                # Extract detection information
                bbox = detection.bbox
                class_name = detection.category.name
                confidence = detection.score.value
                class_id = detection.category.id
                
                # Apply confidence threshold
                if confidence < confidence_threshold:
                    continue
                
                # Apply class-specific thresholds
                if class_name in self.class_confidence_thresholds:
                    if confidence < self.class_confidence_thresholds[class_name]:
                        continue
                
                # Calculate box metrics
                x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                box_width = int(x2 - x1)
                box_height = int(y2 - y1)
                area = box_width * box_height
                relative_area = area / (width * height)
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(
                    confidence, relative_area, box_width, box_height
                )
                
                detection_dict = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "quality_score": round(quality_score, 3),
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2),
                        "center_x": center_x, "center_y": center_y,
                        "width": box_width, "height": box_height
                    },
                    "area": area,
                    "relative_area": round(relative_area, 4),
                    "is_high_quality": quality_score > 0.7
                }
                
                detections.append(detection_dict)
                total_objects += 1
            
            # Sort detections by quality score
            detections.sort(key=lambda x: x["quality_score"], reverse=True)
            
            # Create annotated image
            annotated_image_path = None
            if annotate_image and detections:
                annotated_image_path = await self._create_enhanced_annotated_image(
                    image_path, detections, save_annotated, project_id
                )
            
            processing_time = time.time() - start_time
            
            # Update stats
            self._update_stats(model_name, processing_time, total_objects)
            
            # Build response
            response = {
                "detection_id": detection_id,
                "filename": os.path.basename(image_path),
                "total_objects_detected": total_objects,
                "detections": detections,
                "processing_time": round(processing_time, 3),
                "model_used": model_name,
                "sahi_enabled": use_sahi_slicing,
                "slice_config": {
                    "slice_height": slice_height,
                    "slice_width": slice_width,
                    "overlap_height_ratio": overlap_height_ratio,
                    "overlap_width_ratio": overlap_width_ratio
                } if use_sahi_slicing else None,
                "image_dimensions": {"width": width, "height": height}
            }
            
            if annotated_image_path:
                response["annotated_image_path"] = annotated_image_path
            
            # Add summary statistics
            if detections:
                class_counts = {}
                confidences = []
                quality_scores = []
                high_quality_count = 0
                
                for det in detections:
                    class_name = det["class_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidences.append(det["confidence"])
                    quality_scores.append(det["quality_score"])
                    if det["is_high_quality"]:
                        high_quality_count += 1
                
                response["summary"] = {
                    "unique_classes": len(class_counts),
                    "class_distribution": class_counts,
                    "average_confidence": round(np.mean(confidences), 3),
                    "max_confidence": round(max(confidences), 3),
                    "min_confidence": round(min(confidences), 3),
                    "average_quality_score": round(np.mean(quality_scores), 3),
                    "high_quality_detections": high_quality_count,
                    "quality_percentage": round((high_quality_count / total_objects) * 100, 1) if total_objects > 0 else 0
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                "detection_id": detection_id,
                "filename": os.path.basename(image_path) if image_path else "unknown",
                "total_objects_detected": 0,
                "detections": [],
                "error": str(e),
                "processing_time": round(processing_time, 3),
                "model_used": model_name,
                "success": False
            }
    
    async def _create_mock_response(self, image_path: str, detection_id: str, start_time: float) -> Dict[str, Any]:
        """Create mock response when models aren't loaded (for development)"""
        processing_time = time.time() - start_time
        
        # Mock detections for testing
        mock_detections = [
            {
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.85,
                "quality_score": 0.8,
                "bbox": {
                    "x1": 100, "y1": 50, "x2": 300, "y2": 400,
                    "center_x": 200, "center_y": 225,
                    "width": 200, "height": 350
                },
                "area": 70000,
                "relative_area": 0.15,
                "is_high_quality": True
            },
            {
                "class_name": "car",
                "class_id": 2,
                "confidence": 0.75,
                "quality_score": 0.7,
                "bbox": {
                    "x1": 350, "y1": 200, "x2": 600, "y2": 350,
                    "center_x": 475, "center_y": 275,
                    "width": 250, "height": 150
                },
                "area": 37500,
                "relative_area": 0.12,
                "is_high_quality": True
            }
        ]
        
        return {
            "detection_id": detection_id,
            "filename": os.path.basename(image_path),
            "total_objects_detected": len(mock_detections),
            "detections": mock_detections,
            "processing_time": round(processing_time, 3),
            "model_used": "yolox-small",
            "sahi_enabled": True,
            "image_dimensions": {"width": 640, "height": 480},
            "mock_response": True,
            "summary": {
                "unique_classes": 2,
                "class_distribution": {"person": 1, "car": 1},
                "average_confidence": 0.8,
                "max_confidence": 0.85,
                "min_confidence": 0.75,
                "average_quality_score": 0.75,
                "high_quality_detections": 2,
                "quality_percentage": 100.0
            }
        }
    
    def _calculate_quality_score(self, confidence: float, relative_area: float, 
                                width: int, height: int) -> float:
        """Calculate object quality score based on multiple factors"""
        # Base score from confidence
        score = confidence
        
        # Area bonus (objects with reasonable size get higher scores)
        if 0.001 < relative_area < 0.3:  # Not too small, not too large
            score += 0.1
        
        # Aspect ratio bonus (reasonable object proportions)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if 1 < aspect_ratio < 5:  # Reasonable aspect ratio
            score += 0.05
        
        # Size bonus (larger objects within reason are more reliable)
        if width > 20 and height > 20:
            score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _create_enhanced_annotated_image(self, 
                                             image_path: str, 
                                             detections: List[Dict],
                                             save_image: bool = True,
                                             project_id: int = 1) -> Optional[str]:
        """Create enhanced annotated image with YOLOX+SAHI detections"""
        
        try:
            # Load image with PIL
            pil_image = Image.open(image_path)
            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            
            # Load fonts
            try:
                if os.name == 'nt':  # Windows
                    font = ImageFont.truetype("arial.ttf", max(16, img_width // 80))
                    small_font = ImageFont.truetype("arial.ttf", max(12, img_width // 100))
                else:  # Linux/Mac
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(16, img_width // 80))
                    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(12, img_width // 100))
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Enhanced color palette
            colors = [
                (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
                (175, 82, 222), (255, 204, 0), (255, 45, 85), (88, 86, 214),
                (90, 200, 250), (255, 95, 0), (191, 90, 242), (102, 217, 239),
                (255, 176, 64), (46, 204, 113), (231, 76, 60), (155, 89, 182),
                (52, 152, 219), (241, 196, 15), (230, 126, 34), (26, 188, 156)
            ]
            
            # Draw detections
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                is_high_quality = detection["is_high_quality"]
                
                # Select color
                color = colors[detection["class_id"] % len(colors)]
                line_width = 4 if is_high_quality else 2
                
                # Draw bounding box
                draw.rectangle(
                    [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], 
                    outline=color, 
                    width=line_width
                )
                
                # Add quality indicators
                if is_high_quality:
                    corner_size = 8
                    draw.rectangle([bbox["x1"], bbox["y1"], bbox["x1"] + corner_size, bbox["y1"] + corner_size], fill=color)
                    draw.rectangle([bbox["x2"] - corner_size, bbox["y1"], bbox["x2"], bbox["y1"] + corner_size], fill=color)
                
                # Enhanced label
                quality_indicator = "★" if is_high_quality else "○"
                label = f"{quality_indicator} {class_name}: {confidence:.2f}"
                
                # Draw label background and text
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                label_x = max(0, min(bbox["x1"], img_width - text_width - 8))
                label_y = max(text_height + 8, bbox["y1"])
                
                # Semi-transparent background
                overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(
                    [label_x - 4, label_y - text_height - 8, label_x + text_width + 8, label_y - 2],
                    fill=(*color, 200)
                )
                pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(pil_image)
                
                # Draw text with shadow
                draw.text((label_x + 1, label_y - text_height - 3), label, fill=(0, 0, 0), font=font)
                draw.text((label_x, label_y - text_height - 4), label, fill=(255, 255, 255), font=font)
            
            # Add SAHI watermark
            sahi_text = "🎯 YOLOX+SAHI Detection"
            draw.text((10, 10), sahi_text, fill=(255, 255, 255), font=small_font)
            draw.text((11, 11), sahi_text, fill=(0, 0, 0), font=small_font)
            
            # Save annotated image
            if save_image:
                try:
                    from project_file_manager import project_file_manager
                    
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    temp_annotated_filename = f"{base_name}_yolox_sahi_temp.jpg"
                    temp_annotated_path = os.path.join("uploads", temp_annotated_filename)
                    
                    os.makedirs("uploads", exist_ok=True)
                    pil_image.save(temp_annotated_path, "JPEG", quality=95)
                    
                    annotated_filename = f"{base_name}_yolox_sahi.jpg"
                    final_annotated_path = project_file_manager.save_annotated_file(
                        project_id=project_id,
                        source_image_path=temp_annotated_path,
                        annotated_filename=annotated_filename
                    )
                    
                    if os.path.exists(temp_annotated_path):
                        os.remove(temp_annotated_path)
                    
                    return final_annotated_path
                    
                except Exception as e:
                    logger.error(f"Failed to save annotated image: {str(e)}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create annotated image: {str(e)}")
            return None
    
    def _update_stats(self, model_name: str, processing_time: float, objects_detected: int):
        """Update performance statistics"""
        self.stats["total_detections"] += 1
        self.stats["total_objects_detected"] += objects_detected
        self.stats["total_processing_time"] += processing_time
        
        if model_name not in self.stats["model_usage"]:
            self.stats["model_usage"][model_name] = 0
        self.stats["model_usage"][model_name] += 1
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "models": self.model_info,
            "total_models": len(self.model_info),
            "supported_classes": len(self.class_names),
            "class_names": self.class_names[:10],  # First 10 classes
            "features": [
                "SAHI sliced inference for small objects",
                "Multiple YOLOX model sizes",
                "Enhanced visualization",
                "Quality scoring",
                "Class-specific thresholds"
            ]
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_processing_time = (
            self.stats["total_processing_time"] / max(self.stats["total_detections"], 1)
        )
        avg_objects_per_image = (
            self.stats["total_objects_detected"] / max(self.stats["total_detections"], 1)
        )
        
        return {
            **self.stats,
            "average_processing_time": round(avg_processing_time, 3),
            "average_objects_per_image": round(avg_objects_per_image, 2)
        }

# Initialize the service
yolox_sahi_service = YoloxSahiService() 