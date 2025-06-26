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

# SAHI imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SahiEnhancedDetectionService:
    """Enhanced object detection service using SAHI + Ultralytics YOLO for superior small object detection"""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self._initialize_models()
        
        # Performance tracking
        self.stats = {
            "total_detections": 0,
            "total_objects_detected": 0,
            "total_processing_time": 0.0,
            "model_usage": {},
            "sahi_usage": 0,
            "standard_usage": 0
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
        
        # Enhanced class-specific confidence thresholds
        self.class_confidence_thresholds = {
            'person': 0.3, 'car': 0.25, 'truck': 0.25, 'bus': 0.25,
            'bicycle': 0.2, 'motorcycle': 0.2, 'dog': 0.3, 'cat': 0.3,
            'bird': 0.15, 'bottle': 0.1, 'cup': 0.1, 'cell phone': 0.15,
            'laptop': 0.2, 'tv': 0.2, 'mouse': 0.1, 'keyboard': 0.15,
            'book': 0.1, 'scissors': 0.1, 'teddy bear': 0.2
        }
    
    def _initialize_models(self):
        """Initialize SAHI-enhanced YOLO models"""
        try:
            # Set device
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # SAHI + YOLOv8 Nano - Ultra-fast with SAHI enhancement
            try:
                self.models["sahi-yolo8n"] = UltralyticsDetectionModel(
                    model_path="yolov8n.pt",
                    confidence_threshold=0.2,
                    device=device,
                    category_mapping=None,
                    category_remapping=None,
                    load_at_init=True,
                    image_size=640
                )
                
                self.model_info["sahi-yolo8n"] = {
                    "name": "SAHI + YOLOv8 Nano",
                    "speed": "very_fast",
                    "accuracy": "excellent_with_sahi",
                    "classes": 80,
                    "description": "Ultra-fast object detection with SAHI enhancement for small objects",
                    "recommended_confidence": 0.2,
                    "image_size": 640,
                    "sahi_enabled": True
                }
                
                logger.info("✅ SAHI + YOLOv8 Nano model initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load YOLOv8n model: {e}")
            
            # SAHI + YOLOv8 Small - Balanced performance with SAHI
            try:
                self.models["sahi-yolo8s"] = UltralyticsDetectionModel(
                    model_path="yolov8s.pt",
                    confidence_threshold=0.15,
                    device=device,
                    category_mapping=None,
                    category_remapping=None,
                    load_at_init=True,
                    image_size=640
                )
                
                self.model_info["sahi-yolo8s"] = {
                    "name": "SAHI + YOLOv8 Small",
                    "speed": "fast",
                    "accuracy": "superior_with_sahi",
                    "classes": 80,
                    "description": "Balanced speed and superior accuracy with SAHI enhancement",
                    "recommended_confidence": 0.15,
                    "image_size": 640,
                    "sahi_enabled": True
                }
                
                logger.info("✅ SAHI + YOLOv8 Small model initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load YOLOv8s model: {e}")
            
            if len(self.models) > 0:
                logger.info(f"🎯 SAHI Enhanced Detection Service initialized with {len(self.models)} models")
            else:
                logger.warning("⚠️ No models loaded - using mock mode")
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize SAHI models: {str(e)}")
            self.models = {}
            self.model_info = {}
    
    async def detect_objects(self, 
                           image_path: str,
                           model_name: str = "sahi-yolo8n",
                           confidence_threshold: float = 0.25,
                           annotate_image: bool = True,
                           save_annotated: bool = True,
                           use_sahi_slicing: bool = True,
                           slice_height: int = 640,
                           slice_width: int = 640,
                           overlap_height_ratio: float = 0.2,
                           overlap_width_ratio: float = 0.2,
                           postprocess_type: str = "GREEDYNMMS",
                           project_id: int = 1) -> Dict[str, Any]:
        """
        Enhanced object detection using SAHI + YOLO with sliced inference for superior small object detection
        
        Args:
            image_path: Path to input image
            model_name: Model to use (sahi-yolo8n, sahi-yolo8s)
            confidence_threshold: Confidence threshold for filtering detections
            annotate_image: Whether to draw bounding boxes and labels
            save_annotated: Whether to save annotated image
            use_sahi_slicing: Whether to use SAHI sliced inference (recommended)
            slice_height: Height of each slice for SAHI
            slice_width: Width of each slice for SAHI
            overlap_height_ratio: Overlap ratio for height (0.1-0.8)
            overlap_width_ratio: Overlap ratio for width (0.1-0.8)
            postprocess_type: SAHI postprocessing type (GREEDYNMMS, NMS)
            project_id: Project ID for saving annotated images
        """
        
        start_time = time.time()
        detection_id = str(uuid.uuid4())
        
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and analyze image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            height, width = image.shape[:2]
            
            # Auto-select model if not available
            if model_name not in self.models and len(self.models) > 0:
                model_name = list(self.models.keys())[0]
                logger.info(f"Auto-selected model: {model_name}")
            
            # Mock response if no models loaded (development mode)
            if len(self.models) == 0:
                return await self._create_mock_response(image_path, detection_id, start_time, use_sahi_slicing)
            
            model = self.models[model_name]
            
            # Perform SAHI-enhanced detection
            if use_sahi_slicing:
                # SAHI sliced prediction for superior small object detection
                result = get_sliced_prediction(
                    image=image_path,
                    detection_model=model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio,
                    postprocess_type=postprocess_type,
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.5,
                    postprocess_class_agnostic=False,
                    verbose=0
                )
                self.stats["sahi_usage"] += 1
            else:
                # Standard inference without slicing
                result = get_sliced_prediction(
                    image=image_path,
                    detection_model=model,
                    slice_height=height,
                    slice_width=width,
                    overlap_height_ratio=0,
                    overlap_width_ratio=0,
                    verbose=0
                )
                self.stats["standard_usage"] += 1
            
            # Process and filter detections
            detections = []
            total_objects = 0
            
            for detection in result.object_prediction_list:
                bbox = detection.bbox
                class_name = detection.category.name
                confidence = detection.score.value
                class_id = detection.category.id
                
                # Apply confidence filtering
                if confidence < confidence_threshold:
                    continue
                
                # Apply class-specific thresholds for better precision
                effective_threshold = self.class_confidence_thresholds.get(class_name, confidence_threshold)
                if confidence < effective_threshold:
                    continue
                
                # Calculate enhanced metrics
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                box_width = int(x2 - x1)
                box_height = int(y2 - y1)
                area = box_width * box_height
                relative_area = area / (width * height)
                
                # Enhanced quality scoring
                quality_score = self._calculate_enhanced_quality_score(
                    confidence, relative_area, box_width, box_height, class_name
                )
                
                # Determine detection quality level
                quality_level = self._get_quality_level(quality_score)
                
                detection_dict = {
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": round(confidence, 3),
                    "quality_score": round(quality_score, 3),
                    "quality_level": quality_level,
                    "bbox": {
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "center_x": center_x, "center_y": center_y,
                        "width": box_width, "height": box_height
                    },
                    "area": area,
                    "relative_area": round(relative_area, 4),
                    "effective_threshold": round(effective_threshold, 3),
                    "is_high_quality": quality_score > 0.7,
                    "sahi_detected": use_sahi_slicing
                }
                
                detections.append(detection_dict)
                total_objects += 1
            
            # Sort by quality score for best results first
            detections.sort(key=lambda x: x["quality_score"], reverse=True)
            
            # Create enhanced annotated image
            annotated_image_path = None
            if annotate_image and detections:
                annotated_image_path = await self._create_sahi_annotated_image(
                    image_path, detections, save_annotated, project_id, use_sahi_slicing
                )
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self._update_stats(model_name, processing_time, total_objects)
            
            # Build comprehensive response
            response = {
                "detection_id": detection_id,
                "filename": os.path.basename(image_path),
                "total_objects_detected": total_objects,
                "detections": detections,
                "processing_time": round(processing_time, 3),
                "model_used": model_name,
                "sahi_enhanced": True,
                "sahi_slicing_enabled": use_sahi_slicing,
                "detection_method": "SAHI + YOLOv8" if use_sahi_slicing else "Standard YOLOv8",
                "slice_configuration": {
                    "slice_height": slice_height,
                    "slice_width": slice_width,
                    "overlap_height_ratio": overlap_height_ratio,
                    "overlap_width_ratio": overlap_width_ratio,
                    "postprocess_type": postprocess_type
                } if use_sahi_slicing else None,
                "image_dimensions": {"width": width, "height": height},
                "success": True
            }
            
            if annotated_image_path:
                response["annotated_image_path"] = annotated_image_path
            
            # Add enhanced summary statistics
            if detections:
                response["summary"] = self._generate_detection_summary(detections, total_objects)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ SAHI detection failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return {
                "detection_id": detection_id,
                "filename": os.path.basename(image_path) if image_path else "unknown",
                "total_objects_detected": 0,
                "detections": [],
                "error": str(e),
                "processing_time": round(processing_time, 3),
                "model_used": model_name,
                "sahi_enhanced": True,
                "success": False
            }
    
    def _calculate_enhanced_quality_score(self, confidence: float, relative_area: float, 
                                        width: int, height: int, class_name: str) -> float:
        """Calculate enhanced quality score with class-specific considerations"""
        # Base score from confidence
        score = confidence
        
        # Area-based scoring (optimal size range)
        if 0.0005 < relative_area < 0.4:  # Good size range
            score += 0.1
        elif relative_area > 0.4:  # Very large objects
            score -= 0.05
        
        # Aspect ratio scoring
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if 1 < aspect_ratio < 4:  # Good proportions
            score += 0.05
        elif aspect_ratio > 8:  # Too elongated
            score -= 0.1
        
        # Minimum size requirements
        if width > 15 and height > 15:  # Reasonable minimum size
            score += 0.05
        elif width < 10 or height < 10:  # Too small, likely noise
            score -= 0.2
        
        # Class-specific bonuses
        if class_name in ['person', 'car', 'dog', 'cat']:  # Common important objects
            score += 0.02
        elif class_name in ['cell phone', 'book', 'cup', 'bottle']:  # Small objects SAHI excels at
            score += 0.05
        
        return max(0.0, min(score, 1.0))  # Clamp between 0 and 1
    
    def _get_quality_level(self, quality_score: float) -> str:
        """Determine quality level based on score"""
        if quality_score >= 0.8:
            return "excellent"
        elif quality_score >= 0.6:
            return "good"
        elif quality_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_detection_summary(self, detections: List[Dict], total_objects: int) -> Dict[str, Any]:
        """Generate comprehensive detection summary"""
        class_counts = {}
        confidences = []
        quality_scores = []
        quality_levels = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        sahi_detections = 0
        areas = []
        
        for det in detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(det["confidence"])
            quality_scores.append(det["quality_score"])
            quality_levels[det["quality_level"]] += 1
            areas.append(det["relative_area"])
            
            if det.get("sahi_detected", False):
                sahi_detections += 1
        
        return {
            "unique_classes": len(class_counts),
            "class_distribution": dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)),
            "confidence_stats": {
                "average": round(np.mean(confidences), 3),
                "max": round(max(confidences), 3),
                "min": round(min(confidences), 3),
                "median": round(np.median(confidences), 3)
            },
            "quality_stats": {
                "average_quality_score": round(np.mean(quality_scores), 3),
                "quality_distribution": quality_levels,
                "high_quality_count": quality_levels["excellent"] + quality_levels["good"],
                "quality_percentage": round(((quality_levels["excellent"] + quality_levels["good"]) / total_objects) * 100, 1)
            },
            "size_stats": {
                "average_relative_area": round(np.mean(areas), 4),
                "largest_object": round(max(areas), 4),
                "smallest_object": round(min(areas), 4)
            },
            "sahi_impact": {
                "sahi_enhanced_detections": sahi_detections,
                "sahi_enhancement_rate": round((sahi_detections / total_objects) * 100, 1) if total_objects > 0 else 0
            }
        }
    
    async def _create_mock_response(self, image_path: str, detection_id: str, start_time: float, 
                                  use_sahi_slicing: bool) -> Dict[str, Any]:
        """Create enhanced mock response for development"""
        processing_time = time.time() - start_time
        
        mock_detections = [
            {
                "class_name": "person",
                "class_id": 0,
                "confidence": 0.92,
                "quality_score": 0.89,
                "quality_level": "excellent",
                "bbox": {"x1": 120, "y1": 80, "x2": 280, "y2": 420, "center_x": 200, "center_y": 250, "width": 160, "height": 340},
                "area": 54400,
                "relative_area": 0.176,
                "effective_threshold": 0.3,
                "is_high_quality": True,
                "sahi_detected": use_sahi_slicing
            },
            {
                "class_name": "cell phone",
                "class_id": 67,
                "confidence": 0.78,
                "quality_score": 0.82,
                "quality_level": "excellent",
                "bbox": {"x1": 450, "y1": 150, "x2": 480, "y2": 200, "center_x": 465, "center_y": 175, "width": 30, "height": 50},
                "area": 1500,
                "relative_area": 0.0049,
                "effective_threshold": 0.15,
                "is_high_quality": True,
                "sahi_detected": use_sahi_slicing
            },
            {
                "class_name": "cup",
                "class_id": 41,
                "confidence": 0.65,
                "quality_score": 0.71,
                "quality_level": "good",
                "bbox": {"x1": 350, "y1": 280, "x2": 390, "y2": 330, "center_x": 370, "center_y": 305, "width": 40, "height": 50},
                "area": 2000,
                "relative_area": 0.0065,
                "effective_threshold": 0.1,
                "is_high_quality": True,
                "sahi_detected": use_sahi_slicing
            }
        ]
        
        return {
            "detection_id": detection_id,
            "filename": os.path.basename(image_path),
            "total_objects_detected": len(mock_detections),
            "detections": mock_detections,
            "processing_time": round(processing_time, 3),
            "model_used": "sahi-yolo8n",
            "sahi_enhanced": True,
            "sahi_slicing_enabled": use_sahi_slicing,
            "detection_method": "SAHI + YOLOv8 (Mock)" if use_sahi_slicing else "Standard YOLOv8 (Mock)",
            "image_dimensions": {"width": 640, "height": 480},
            "mock_response": True,
            "success": True,
            "summary": {
                "unique_classes": 3,
                "class_distribution": {"person": 1, "cell phone": 1, "cup": 1},
                "confidence_stats": {"average": 0.783, "max": 0.92, "min": 0.65, "median": 0.78},
                "quality_stats": {
                    "average_quality_score": 0.807,
                    "quality_distribution": {"excellent": 2, "good": 1, "fair": 0, "poor": 0},
                    "high_quality_count": 3,
                    "quality_percentage": 100.0
                },
                "sahi_impact": {
                    "sahi_enhanced_detections": 3 if use_sahi_slicing else 0,
                    "sahi_enhancement_rate": 100.0 if use_sahi_slicing else 0.0
                }
            }
        }
    
    async def _create_sahi_annotated_image(self, image_path: str, detections: List[Dict],
                                         save_image: bool, project_id: int, 
                                         sahi_enabled: bool) -> Optional[str]:
        """Create SAHI-enhanced annotated image with quality indicators"""
        try:
            pil_image = Image.open(image_path)
            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            
            # Load fonts
            try:
                if os.name == 'nt':
                    font = ImageFont.truetype("arial.ttf", max(14, img_width // 90))
                    small_font = ImageFont.truetype("arial.ttf", max(10, img_width // 120))
                else:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(14, img_width // 90))
                    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(10, img_width // 120))
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Quality-based color palette
            quality_colors = {
                "excellent": (46, 204, 113),    # Emerald green
                "good": (52, 152, 219),         # Peter river blue
                "fair": (241, 196, 15),         # Sun flower yellow
                "poor": (231, 76, 60)           # Alizarin red
            }
            
            # Class-based colors as fallback
            class_colors = [
                (255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 149, 0),
                (175, 82, 222), (255, 204, 0), (255, 45, 85), (88, 86, 214),
                (90, 200, 250), (255, 95, 0), (191, 90, 242), (102, 217, 239)
            ]
            
            # Draw detections with quality-based styling
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                quality_level = detection["quality_level"]
                quality_score = detection["quality_score"]
                
                # Select color based on quality
                color = quality_colors.get(quality_level, class_colors[detection["class_id"] % len(class_colors)])
                
                # Line width based on quality
                line_width = 3 if quality_level in ["excellent", "good"] else 2
                
                # Draw bounding box
                draw.rectangle([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], 
                             outline=color, width=line_width)
                
                # Quality indicators
                if quality_level == "excellent":
                    # Double corner markers for excellent
                    corner_size = 8
                    draw.rectangle([bbox["x1"], bbox["y1"], bbox["x1"] + corner_size, bbox["y1"] + corner_size], fill=color)
                    draw.rectangle([bbox["x2"] - corner_size, bbox["y1"], bbox["x2"], bbox["y1"] + corner_size], fill=color)
                    draw.rectangle([bbox["x1"], bbox["y2"] - corner_size, bbox["x1"] + corner_size, bbox["y2"]], fill=color)
                    draw.rectangle([bbox["x2"] - corner_size, bbox["y2"] - corner_size, bbox["x2"], bbox["y2"]], fill=color)
                elif quality_level == "good":
                    # Single corner markers for good
                    corner_size = 6
                    draw.rectangle([bbox["x1"], bbox["y1"], bbox["x1"] + corner_size, bbox["y1"] + corner_size], fill=color)
                    draw.rectangle([bbox["x2"] - corner_size, bbox["y1"], bbox["x2"], bbox["y1"] + corner_size], fill=color)
                
                # Enhanced label with quality and SAHI indicator
                quality_icon = {"excellent": "★", "good": "●", "fair": "◐", "poor": "○"}[quality_level]
                sahi_icon = " 🔍" if detection.get("sahi_detected", False) else ""
                label = f"{quality_icon} {class_name}: {confidence:.2f}{sahi_icon}"
                
                # Calculate label position
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
                    fill=color + (220,) if color else (255, 255, 255, 220)
                )
                pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(pil_image)
                # Draw text with shadow
                draw.text((label_x + 1, label_y - text_height - 3), label, fill=(0, 0, 0), font=font)
                draw.text((label_x, label_y - text_height - 4), label, fill=(255, 255, 255), font=font)
                
                # Add detection number
                num_text = f"#{i+1}"
                draw.text((bbox["x2"] - 25, bbox["y2"] - 20), num_text, fill=(255, 255, 255), font=small_font)
                draw.text((bbox["x2"] - 24, bbox["y2"] - 19), num_text, fill=(0, 0, 0), font=small_font)
            
            # Enhanced header with SAHI branding
            method_text = "🎯 SAHI Enhanced Detection" if sahi_enabled else "🎯 Standard YOLO Detection"
            header_y = 10
            
            # Header background
            header_bbox = draw.textbbox((0, 0), method_text, font=font)
            header_width = header_bbox[2] - header_bbox[0]
            header_height = header_bbox[3] - header_bbox[1]
            
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([8, header_y - 4, header_width + 16, header_y + header_height + 4], 
                                 fill=(0, 0, 0, 180))
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_image)
            
            # Header text
            draw.text((10, header_y), method_text, fill=(255, 255, 255), font=font)
            
            # Detection summary
            summary_text = f"Objects: {len(detections)} | Quality: ★{sum(1 for d in detections if d['quality_level'] == 'excellent')} ●{sum(1 for d in detections if d['quality_level'] == 'good')}"
            draw.text((10, header_y + header_height + 8), summary_text, fill=(255, 255, 255), font=small_font)
            draw.text((11, header_y + header_height + 9), summary_text, fill=(0, 0, 0), font=small_font)
            
            # Save enhanced annotated image
            if save_image:
                try:
                    from project_file_manager import project_file_manager
                    
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    method_suffix = "sahi_enhanced" if sahi_enabled else "standard"
                    temp_filename = f"{base_name}_{method_suffix}_temp.jpg"
                    temp_path = os.path.join("uploads", temp_filename)
                    
                    os.makedirs("uploads", exist_ok=True)
                    pil_image.save(temp_path, "JPEG", quality=95)
                    
                    final_filename = f"{base_name}_{method_suffix}.jpg"
                    final_path = project_file_manager.save_annotated_file(
                        project_id=project_id,
                        source_image_path=temp_path,
                        annotated_filename=final_filename
                    )
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return final_path
                    
                except Exception as e:
                    logger.error(f"Failed to save SAHI annotated image: {str(e)}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create SAHI annotated image: {str(e)}")
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
        """Get information about available SAHI-enhanced models"""
        return {
            "models": self.model_info,
            "total_models": len(self.model_info),
            "supported_classes": len(self.class_names),
            "class_names": self.class_names,
            "sahi_features": [
                "Sliced inference for small object detection",
                "Enhanced quality scoring system",
                "Multiple YOLOv8 model sizes with SAHI",
                "Quality-based visualization",
                "Class-specific confidence thresholds",
                "Advanced postprocessing options"
            ],
            "recommended_settings": {
                "slice_height": 640,
                "slice_width": 640,
                "overlap_ratios": "0.2 for balanced performance",
                "postprocess_type": "GREEDYNMMS for best results"
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get enhanced performance statistics"""
        total_detections = max(self.stats["total_detections"], 1)
        
        return {
            **self.stats,
            "average_processing_time": round(self.stats["total_processing_time"] / total_detections, 3),
            "average_objects_per_image": round(self.stats["total_objects_detected"] / total_detections, 2),
            "sahi_usage_rate": round((self.stats["sahi_usage"] / total_detections) * 100, 1),
            "enhancement_ratio": f"SAHI: {self.stats['sahi_usage']} | Standard: {self.stats['standard_usage']}"
        }

# Initialize the enhanced service
sahi_enhanced_detection_service = SahiEnhancedDetectionService() 