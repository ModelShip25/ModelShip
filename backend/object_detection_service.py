import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObjectDetectionService:
    """Advanced object detection service with enhanced accuracy and visualization"""
    
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
        
        # Enhanced class-specific confidence thresholds for better accuracy
        self.class_confidence_thresholds = {
            # People & Animals
            'person': 0.3, 'cat': 0.3, 'dog': 0.3, 'bird': 0.25, 'horse': 0.4,
            'sheep': 0.35, 'cow': 0.4, 'elephant': 0.5, 'bear': 0.4, 'zebra': 0.45, 'giraffe': 0.5,
            
            # Vehicles
            'car': 0.4, 'bicycle': 0.35, 'motorcycle': 0.4, 'bus': 0.5, 'truck': 0.5,
            'airplane': 0.6, 'train': 0.6, 'boat': 0.4,
            
            # Furniture & Household
            'chair': 0.3, 'couch': 0.4, 'bed': 0.4, 'dining table': 0.35, 'toilet': 0.5,
            'tv': 0.4, 'laptop': 0.4, 'microwave': 0.25, 'oven': 0.25, 'toaster': 0.25,
            'sink': 0.25, 'refrigerator': 0.25,
            
            # Food & Kitchen
            'bottle': 0.2, 'wine glass': 0.25, 'cup': 0.25, 'fork': 0.15, 'knife': 0.15,
            'spoon': 0.15, 'bowl': 0.25, 'banana': 0.3, 'apple': 0.25, 'sandwich': 0.3,
            'orange': 0.25, 'pizza': 0.4, 'donut': 0.3, 'cake': 0.35, 'broccoli': 0.25,
            'carrot': 0.25, 'hot dog': 0.3,
            
            # Sports & Recreation
            'frisbee': 0.25, 'skis': 0.25, 'snowboard': 0.25, 'sports ball': 0.25,
            'kite': 0.25, 'baseball bat': 0.25, 'baseball glove': 0.25, 'skateboard': 0.25,
            'surfboard': 0.25, 'tennis racket': 0.25,
            
            # Personal Items
            'backpack': 0.25, 'umbrella': 0.25, 'handbag': 0.25, 'tie': 0.25,
            'suitcase': 0.25, 'cell phone': 0.25, 'book': 0.2, 'clock': 0.3,
            'vase': 0.3, 'scissors': 0.15, 'teddy bear': 0.2, 'hair drier': 0.2,
            'toothbrush': 0.15, 'mouse': 0.25, 'remote': 0.25, 'keyboard': 0.25,
            
            # Outdoor & Street
            'traffic light': 0.25, 'fire hydrant': 0.2, 'stop sign': 0.25,
            'parking meter': 0.2, 'bench': 0.3, 'potted plant': 0.25
        }
    
    def _initialize_models(self):
        """Initialize YOLO object detection models with enhanced settings"""
        try:
            # YOLOv8 nano - fast and lightweight
            self.models["yolo8n"] = YOLO('yolov8n.pt')
            self.model_info["yolo8n"] = {
                "name": "YOLOv8 Nano",
                "speed": "very_fast",
                "accuracy": "good",
                "classes": 80,
                "description": "Fast object detection optimized for real-time applications",
                "recommended_confidence": 0.25
            }
            
            # YOLOv8 small - balanced speed and accuracy
            self.models["yolo8s"] = YOLO('yolov8s.pt')
            self.model_info["yolo8s"] = {
                "name": "YOLOv8 Small",
                "speed": "fast", 
                "accuracy": "very_good",
                "classes": 80,
                "description": "Balanced speed and accuracy for general-purpose detection",
                "recommended_confidence": 0.2
            }
            
            # Enhanced class names with better descriptions
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
                'toothbrush'
            ]
            
            logger.info("Enhanced object detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize object detection models: {str(e)}")
            raise
    
    async def detect_objects(self, 
                           image_path: str,
                           model_name: str = "yolo8n",
                           confidence_threshold: float = 0.25,
                           annotate_image: bool = True,
                           save_annotated: bool = True,
                           use_class_specific_thresholds: bool = True,
                           project_id: int = 1) -> Dict[str, Any]:
        """
        Enhanced object detection with improved accuracy and visualization
        
        Args:
            image_path: Path to input image
            model_name: Model to use (yolo8n, yolo8s)
            confidence_threshold: Base confidence threshold
            annotate_image: Whether to draw bounding boxes and labels
            save_annotated: Whether to save annotated image
            use_class_specific_thresholds: Use optimized thresholds per class
            project_id: Project ID for saving annotated images
        """
        
        start_time = time.time()
        detection_id = str(uuid.uuid4())
        
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not available")
            
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Enhanced detection with multiple scales and augmentation
            model = self.models[model_name]
            
            # Run detection with enhanced parameters
            results = model(
                image_path,
                conf=confidence_threshold * 0.8,  # Lower initial threshold
                iou=0.45,  # Non-maximum suppression threshold
                imgsz=640,  # Input image size
                augment=True,  # Test-time augmentation for better accuracy
                agnostic_nms=False,  # Class-specific NMS
                max_det=100  # Maximum detections per image
            )
            
            # Process detections with enhanced filtering
            detections = []
            total_objects = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        # Apply class-specific confidence thresholds
                        effective_threshold = confidence_threshold
                        if use_class_specific_thresholds and class_name in self.class_confidence_thresholds:
                            effective_threshold = max(
                                confidence_threshold,
                                self.class_confidence_thresholds[class_name]
                            )
                        
                        # Filter by effective threshold
                        if confidence < effective_threshold:
                            continue
                        
                        # Calculate enhanced box metrics
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        box_width = int(x2 - x1)
                        box_height = int(y2 - y1)
                        area = box_width * box_height
                        relative_area = area / (width * height)
                        
                        # Calculate object quality score
                        quality_score = self._calculate_quality_score(
                            confidence, relative_area, box_width, box_height
                        )
                        
                        detection = {
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
                            "effective_threshold": round(effective_threshold, 3),
                            "is_high_quality": quality_score > 0.7
                        }
                        
                        detections.append(detection)
                        total_objects += 1
            
            # Sort detections by quality score (best first)
            detections.sort(key=lambda x: x["quality_score"], reverse=True)
            
            # Create enhanced annotated image
            annotated_image_path = None
            if annotate_image and detections:
                annotated_image_path = await self._create_enhanced_annotated_image(
                    image_path, detections, save_annotated, project_id
                )
            
            processing_time = time.time() - start_time
            
            # Update stats
            self._update_stats(model_name, processing_time, total_objects)
            
            # Build enhanced response
            response = {
                "detection_id": detection_id,
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "model_used": model_name,
                "model_info": self.model_info[model_name],
                "processing_time": round(processing_time, 3),
                "image_dimensions": {"width": width, "height": height},
                "confidence_threshold": confidence_threshold,
                "used_class_specific_thresholds": use_class_specific_thresholds,
                "total_objects_detected": total_objects,
                "detections": detections,
                "annotated_image_path": annotated_image_path,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add enhanced summary statistics
            if detections:
                class_counts = {}
                confidences = []
                quality_scores = []
                areas = []
                high_quality_count = 0
                
                for det in detections:
                    class_name = det["class_name"]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidences.append(det["confidence"])
                    quality_scores.append(det["quality_score"])
                    areas.append(det["relative_area"])
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
                    "quality_percentage": round((high_quality_count / total_objects) * 100, 1) if total_objects > 0 else 0,
                    "average_object_size": round(np.mean(areas), 4),
                    "detection_density": round(total_objects / (width * height / 1000000), 2)  # Objects per megapixel
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Object detection failed: {str(e)}")
            return {
                "detection_id": detection_id,
                "image_path": image_path,
                "filename": os.path.basename(image_path),
                "status": "error",
                "error_message": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _calculate_quality_score(self, confidence: float, relative_area: float, 
                               width: int, height: int) -> float:
        """Calculate detection quality score based on multiple factors"""
        
        # Base score from confidence
        confidence_score = confidence
        
        # Size score (prefer medium-sized objects)
        if relative_area < 0.001:  # Very small
            size_score = 0.3
        elif relative_area < 0.01:  # Small
            size_score = 0.7
        elif relative_area < 0.1:  # Medium
            size_score = 1.0
        elif relative_area < 0.5:  # Large
            size_score = 0.9
        else:  # Very large
            size_score = 0.6
        
        # Aspect ratio score (prefer reasonable aspect ratios)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio < 3:
            aspect_score = 1.0
        elif aspect_ratio < 5:
            aspect_score = 0.8
        else:
            aspect_score = 0.6
        
        # Resolution score (prefer objects with sufficient resolution)
        min_dimension = min(width, height)
        if min_dimension >= 50:
            resolution_score = 1.0
        elif min_dimension >= 30:
            resolution_score = 0.8
        elif min_dimension >= 20:
            resolution_score = 0.6
        else:
            resolution_score = 0.4
        
        # Weighted combination
        quality_score = (
            confidence_score * 0.4 +
            size_score * 0.25 +
            aspect_score * 0.2 +
            resolution_score * 0.15
        )
        
        return min(1.0, quality_score)

    async def _create_enhanced_annotated_image(self, 
                                             image_path: str, 
                                             detections: List[Dict],
                                             save_image: bool = True,
                                             project_id: int = 1) -> Optional[str]:
        """Create enhanced annotated image with improved visualization"""
        
        try:
            # Load image with PIL for better text rendering
            pil_image = Image.open(image_path)
            draw = ImageDraw.Draw(pil_image)
            img_width, img_height = pil_image.size
            
            # Enhanced font loading with fallbacks
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
            
            # Enhanced color palette with better visibility
            colors = [
                (255, 59, 48),   # Red
                (52, 199, 89),   # Green  
                (0, 122, 255),   # Blue
                (255, 149, 0),   # Orange
                (175, 82, 222),  # Purple
                (255, 204, 0),   # Yellow
                (255, 45, 85),   # Pink
                (88, 86, 214),   # Indigo
                (90, 200, 250),  # Light Blue
                (255, 95, 0),    # Red Orange
                (191, 90, 242),  # Light Purple
                (102, 217, 239), # Cyan
                (255, 176, 64),  # Light Orange
                (46, 204, 113),  # Emerald
                (231, 76, 60),   # Alizarin
                (155, 89, 182),  # Amethyst
                (52, 152, 219),  # Peter River
                (241, 196, 15),  # Sun Flower
                (230, 126, 34),  # Carrot
                (26, 188, 156)   # Turquoise
            ]
            
            # Draw enhanced detections
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                class_name = detection["class_name"]
                confidence = detection["confidence"]
                quality_score = detection["quality_score"]
                is_high_quality = detection["is_high_quality"]
                
                # Select color based on class and quality
                color = colors[detection["class_id"] % len(colors)]
                
                # Adjust line width based on quality
                line_width = 4 if is_high_quality else 2
                
                # Draw bounding box with quality-based styling
                draw.rectangle(
                    [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]], 
                    outline=color, 
                    width=line_width
                )
                
                # Add corner markers for high-quality detections
                if is_high_quality:
                    corner_size = 8
                    # Top-left corner
                    draw.rectangle([bbox["x1"], bbox["y1"], bbox["x1"] + corner_size, bbox["y1"] + corner_size], fill=color)
                    # Top-right corner  
                    draw.rectangle([bbox["x2"] - corner_size, bbox["y1"], bbox["x2"], bbox["y1"] + corner_size], fill=color)
                
                # Enhanced label with quality indicator
                quality_indicator = "★" if is_high_quality else "○"
                label = f"{quality_indicator} {class_name}: {confidence:.2f}"
                
                # Get text size for background
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
                
                # Calculate label position (avoid going outside image)
                label_x = max(0, min(bbox["x1"], img_width - text_width - 8))
                label_y = max(text_height + 8, bbox["y1"])
                
                # Draw enhanced label background with gradient effect
                label_bg = [
                    label_x - 4, 
                    label_y - text_height - 8,
                    label_x + text_width + 8, 
                    label_y - 2
                ]
                
                # Semi-transparent background
                overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(label_bg, fill=(*color, 200))
                pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(pil_image)
                
                # Draw label text with shadow for better readability
                shadow_offset = 1
                draw.text(
                    (label_x + shadow_offset, label_y - text_height - 4 + shadow_offset),
                    label, fill=(0, 0, 0), font=font
                )
                draw.text(
                    (label_x, label_y - text_height - 4), 
                    label, fill=(255, 255, 255), font=font
                )
                
                # Add object ID for human review
                object_id = f"#{i+1}"
                id_bbox = draw.textbbox((0, 0), object_id, font=small_font)
                id_width = id_bbox[2] - id_bbox[0]
                
                # Draw object ID in bottom-right of bounding box
                id_x = bbox["x2"] - id_width - 4
                id_y = bbox["y2"] - 16
                
                draw.rectangle([id_x - 2, id_y - 2, id_x + id_width + 2, id_y + 12], fill=(0, 0, 0, 180))
                draw.text((id_x, id_y), object_id, fill=(255, 255, 255), font=small_font)
            
            # Enhanced image metadata overlay
            filename = os.path.basename(image_path)
            detection_count = len(detections)
            high_quality_count = sum(1 for d in detections if d["is_high_quality"])
            
            # Create info panel
            info_lines = [
                f"📸 {filename}",
                f"🎯 {detection_count} objects detected",
                f"★ {high_quality_count} high quality",
                f"📏 {img_width}×{img_height}px"
            ]
            
            # Calculate info panel size
            max_text_width = 0
            total_text_height = 0
            line_heights = []
            
            for line in info_lines:
                bbox_line = draw.textbbox((0, 0), line, font=small_font)
                line_width = bbox_line[2] - bbox_line[0]
                line_height = bbox_line[3] - bbox_line[1]
                max_text_width = max(max_text_width, line_width)
                line_heights.append(line_height)
                total_text_height += line_height + 2
            
            # Position info panel at bottom-left
            panel_x = 10
            panel_y = img_height - total_text_height - 20
            panel_width = max_text_width + 16
            panel_height = total_text_height + 16
            
            # Draw info panel background
            overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
                fill=(0, 0, 0, 160)
            )
            pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(pil_image)
            
            # Draw info text
            current_y = panel_y + 8
            for i, line in enumerate(info_lines):
                draw.text((panel_x + 8, current_y), line, fill=(255, 255, 255), font=small_font)
                current_y += line_heights[i] + 2
            
            # Save annotated image if requested
            if save_image:
                try:
                    from project_file_manager import project_file_manager
                    
                    # First save the annotated PIL image to a temporary location
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    temp_annotated_filename = f"{base_name}_detected_temp.jpg"
                    temp_annotated_path = os.path.join("uploads", temp_annotated_filename)
                    
                    # Ensure uploads directory exists
                    os.makedirs("uploads", exist_ok=True)
                    
                    # Save the annotated PIL image
                    pil_image.save(temp_annotated_path, "JPEG", quality=95)
                    
                    # Now move it to the project storage using the file manager
                    annotated_filename = f"{base_name}_detected.jpg"
                    final_annotated_path = project_file_manager.save_annotated_file(
                        project_id=project_id,
                        source_image_path=temp_annotated_path,
                        annotated_filename=annotated_filename
                    )
                    
                    # Clean up temp file
                    if os.path.exists(temp_annotated_path):
                        os.remove(temp_annotated_path)
                    
                    if final_annotated_path:
                        # Return just the filename for the API endpoint
                        annotated_filename_only = os.path.basename(final_annotated_path)
                        logger.info(f"Annotated image saved to project {project_id}: {annotated_filename_only}")
                        return annotated_filename_only
                    else:
                        logger.error("Failed to save annotated image to project storage")
                        return None
                    
                except ImportError:
                    logger.warning("Project file manager not available, using fallback method")
                    # Fallback to original method
                    annotated_dir = os.path.join("uploads", "annotated")
                    os.makedirs("uploads", exist_ok=True)
                    os.makedirs(annotated_dir, exist_ok=True)
                    
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    annotated_filename = f"{base_name}_annotated_{timestamp}.jpg"
                    annotated_path = os.path.join(annotated_dir, annotated_filename)
                    
                    pil_image.save(annotated_path, "JPEG", quality=95)
                    return annotated_filename
                except Exception as e:
                    logger.error(f"Error saving annotated image: {str(e)}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to create enhanced annotated image: {str(e)}")
            return None

    async def detect_objects_batch(self, 
                                 image_paths: List[str],
                                 model_name: str = "yolo8n",
                                 confidence_threshold: float = 0.25,
                                 annotate_images: bool = True,
                                 progress_callback: Optional[callable] = None,
                                 project_id: int = 1) -> List[Dict[str, Any]]:
        """Enhanced batch object detection for multiple images"""
        
        results = []
        total_images = len(image_paths)
        
        for i, image_path in enumerate(image_paths):
            try:
                result = await self.detect_objects(
                    image_path=image_path,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    annotate_image=annotate_images,
                    save_annotated=True,
                    use_class_specific_thresholds=True,
                    project_id=project_id
                )
                results.append(result)
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / total_images
                    progress_callback(progress, f"Processed {i + 1}/{total_images} images")
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    "image_path": image_path,
                    "filename": os.path.basename(image_path),
                    "status": "error",
                    "error_message": str(e)
                })
        
        return results
    
    def _update_stats(self, model_name: str, processing_time: float, objects_detected: int):
        """Update performance statistics"""
        self.stats["total_detections"] += 1
        self.stats["total_objects_detected"] += objects_detected
        self.stats["total_processing_time"] += processing_time
        
        if model_name not in self.stats["model_usage"]:
            self.stats["model_usage"][model_name] = 0
        self.stats["model_usage"][model_name] += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            **self.stats,
            "average_processing_time": (
                self.stats["total_processing_time"] / max(1, self.stats["total_detections"])
            ),
            "average_objects_per_image": (
                self.stats["total_objects_detected"] / max(1, self.stats["total_detections"])
            )
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "models": self.model_info,
            "supported_classes": len(self.class_names),
            "class_names": self.class_names,
            "class_confidence_thresholds": self.class_confidence_thresholds
        }

# Global instance
object_detection_service = ObjectDetectionService() 