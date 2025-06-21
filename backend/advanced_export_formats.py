from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from sqlalchemy.orm import Session
from database import get_db
from models import User, Project, Job, Result, File
from auth import get_current_user
from typing import List, Dict, Any, Optional
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
import uuid
import logging

router = APIRouter(prefix="/api/export/formats", tags=["advanced_export"])

logger = logging.getLogger(__name__)

class COCOExporter:
    """Export data in COCO format for object detection and segmentation"""
    
    def __init__(self):
        self.coco_format = {
            "info": {
                "description": "ModelShip Auto-Labeled Dataset",
                "url": "https://modelship.ai",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "ModelShip Platform",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{
                "id": 1,
                "name": "ModelShip License",
                "url": "https://modelship.ai/license"
            }],
            "images": [],
            "annotations": [],
            "categories": []
        }
    
    def add_categories(self, label_categories: List[str]):
        """Add label categories to COCO format"""
        for idx, category in enumerate(label_categories):
            self.coco_format["categories"].append({
                "id": idx + 1,
                "name": category,
                "supercategory": "object"
            })
    
    def add_image_annotation(self, result: Result, file_path: str, image_id: int, annotation_id: int):
        """Add image and annotation to COCO format"""
        # Add image info
        try:
            from PIL import Image
            img = Image.open(file_path)
            width, height = img.size
        except:
            width, height = 640, 480  # Default if image can't be opened
        
        self.coco_format["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": result.filename,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": result.created_at.isoformat()
        })
        
        # Add annotation (for classification, we create a full-image bounding box)
        category_id = self._get_category_id(result.predicted_label)
        
        self.coco_format["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [],
            "area": width * height,
            "bbox": [0, 0, width, height],  # Full image bbox for classification
            "iscrowd": 0,
            "confidence": result.confidence,
            "review_status": "reviewed" if result.reviewed else "auto_labeled"
        })
    
    def _get_category_id(self, category_name: str) -> int:
        """Get category ID for a given category name"""
        for cat in self.coco_format["categories"]:
            if cat["name"] == category_name:
                return cat["id"]
        return 1  # Default category

class YOLOExporter:
    """Export data in YOLO format for object detection"""
    
    def __init__(self):
        self.class_names = []
        self.annotations = {}
    
    def add_classes(self, label_categories: List[str]):
        """Add class names for YOLO format"""
        self.class_names = label_categories
    
    def add_annotation(self, result: Result, file_path: str):
        """Add annotation in YOLO format"""
        try:
            from PIL import Image
            img = Image.open(file_path)
            width, height = img.size
        except:
            width, height = 640, 480
        
        # Get class index
        class_id = self._get_class_id(result.predicted_label)
        
        # For classification, we use full image as bounding box
        # YOLO format: class_id center_x center_y width height (normalized)
        center_x = 0.5
        center_y = 0.5
        bbox_width = 1.0
        bbox_height = 1.0
        
        annotation_line = f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}"
        
        # Store annotation for this image
        image_name = os.path.splitext(result.filename)[0]
        self.annotations[image_name] = annotation_line
    
    def _get_class_id(self, class_name: str) -> int:
        """Get class ID for YOLO format"""
        try:
            return self.class_names.index(class_name)
        except ValueError:
            return 0  # Default class

class PascalVOCExporter:
    """Export data in Pascal VOC XML format"""
    
    def __init__(self):
        self.annotations = {}
    
    def add_annotation(self, result: Result, file_path: str):
        """Add annotation in Pascal VOC XML format"""
        try:
            from PIL import Image
            img = Image.open(file_path)
            width, height = img.size
            depth = len(img.getbands())
        except:
            width, height, depth = 640, 480, 3
        
        # Create XML structure
        annotation = ET.Element("annotation")
        
        # Add folder and filename
        ET.SubElement(annotation, "folder").text = "images"
        ET.SubElement(annotation, "filename").text = result.filename
        ET.SubElement(annotation, "path").text = file_path
        
        # Add source
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "ModelShip"
        
        # Add size
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)
        
        # Add segmented
        ET.SubElement(annotation, "segmented").text = "0"
        
        # Add object (for classification, we create a full-image object)
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = result.predicted_label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "confidence").text = str(result.confidence)
        
        # Add bounding box (full image for classification)
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = "1"
        ET.SubElement(bndbox, "ymin").text = "1"
        ET.SubElement(bndbox, "xmax").text = str(width)
        ET.SubElement(bndbox, "ymax").text = str(height)
        
        # Store XML for this image
        image_name = os.path.splitext(result.filename)[0]
        self.annotations[image_name] = ET.tostring(annotation, encoding='unicode')

@router.post("/coco/{job_id}")
async def export_coco_format(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    include_reviewed_only: bool = Query(False),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    """Export job results in COCO format"""
    
    try:
        # Verify job ownership
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == current_user.id
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get results
        results_query = db.query(Result).filter(Result.job_id == job_id)
        
        if include_reviewed_only:
            results_query = results_query.filter(Result.reviewed == True)
        
        if confidence_threshold > 0:
            results_query = results_query.filter(Result.confidence >= confidence_threshold)
        
        results = results_query.all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found with specified criteria")
        
        # Create COCO exporter
        coco_exporter = COCOExporter()
        
        # Get unique categories
        categories = list(set(result.predicted_label for result in results if result.predicted_label))
        coco_exporter.add_categories(categories)
        
        # Add annotations
        for idx, result in enumerate(results):
            if result.file:
                file_path = result.file.file_path
            else:
                file_path = os.path.join("uploads", result.filename)
            
            if os.path.exists(file_path):
                coco_exporter.add_image_annotation(result, file_path, idx + 1, idx + 1)
        
        # Create export directory
        export_dir = os.path.join("exports", f"coco_{job_id}_{uuid.uuid4().hex[:8]}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Save COCO JSON
        coco_file = os.path.join(export_dir, "annotations.json")
        with open(coco_file, 'w') as f:
            json.dump(coco_exporter.coco_format, f, indent=2)
        
        # Create README
        readme_content = f"""# COCO Format Export - ModelShip

## Dataset Information
- Job ID: {job_id}
- Export Date: {datetime.now().isoformat()}
- Total Images: {len(coco_exporter.coco_format['images'])}
- Total Annotations: {len(coco_exporter.coco_format['annotations'])}
- Categories: {len(categories)}

## File Structure
- annotations.json: COCO format annotations
- README.md: This file

## Categories
{chr(10).join(f"- {cat['name']} (ID: {cat['id']})" for cat in coco_exporter.coco_format['categories'])}

## Usage
This dataset is in COCO format and can be used with frameworks like:
- Detectron2
- MMDetection
- YOLOv5/v8 (with conversion)
- TensorFlow Object Detection API

Generated by ModelShip Auto-Labeling Platform
"""
        
        readme_file = os.path.join(export_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create ZIP file
        zip_filename = f"coco_export_{job_id}.zip"
        zip_path = os.path.join("exports", zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        return {
            "export_format": "COCO",
            "job_id": job_id,
            "download_url": f"/api/export/download/{zip_filename}",
            "file_size_mb": round(os.path.getsize(zip_path) / (1024*1024), 2),
            "statistics": {
                "total_images": len(coco_exporter.coco_format['images']),
                "total_annotations": len(coco_exporter.coco_format['annotations']),
                "categories": categories,
                "filters_applied": {
                    "reviewed_only": include_reviewed_only,
                    "confidence_threshold": confidence_threshold
                }
            },
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"COCO export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"COCO export failed: {str(e)}")

@router.post("/yolo/{job_id}")
async def export_yolo_format(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    include_reviewed_only: bool = Query(False),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    """Export job results in YOLO format"""
    
    try:
        # Verify job ownership  
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == current_user.id
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get results
        results_query = db.query(Result).filter(Result.job_id == job_id)
        
        if include_reviewed_only:
            results_query = results_query.filter(Result.reviewed == True)
        
        if confidence_threshold > 0:
            results_query = results_query.filter(Result.confidence >= confidence_threshold)
        
        results = results_query.all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        
        # Create YOLO exporter
        yolo_exporter = YOLOExporter()
        
        # Get unique categories
        categories = list(set(result.predicted_label for result in results if result.predicted_label))
        yolo_exporter.add_classes(categories)
        
        # Add annotations
        for result in results:
            if result.file:
                file_path = result.file.file_path
            else:
                file_path = os.path.join("uploads", result.filename)
            
            if os.path.exists(file_path):
                yolo_exporter.add_annotation(result, file_path)
        
        # Create export directory
        export_dir = os.path.join("exports", f"yolo_{job_id}_{uuid.uuid4().hex[:8]}")
        os.makedirs(export_dir, exist_ok=True)
        
        # Save class names
        classes_file = os.path.join(export_dir, "classes.txt")
        with open(classes_file, 'w') as f:
            for class_name in yolo_exporter.class_names:
                f.write(f"{class_name}\n")
        
        # Save annotations
        labels_dir = os.path.join(export_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        
        for image_name, annotation in yolo_exporter.annotations.items():
            label_file = os.path.join(labels_dir, f"{image_name}.txt")
            with open(label_file, 'w') as f:
                f.write(annotation)
        
        # Create data.yaml for YOLOv5/v8
        data_yaml = f"""# YOLO Dataset Configuration - ModelShip Export

path: ./  # Root directory
train: images/train/  # Training images
val: images/val/      # Validation images 
test: images/test/    # Test images (optional)

# Number of classes
nc: {len(categories)}

# Class names
names: {categories}

# Dataset info
job_id: {job_id}
export_date: {datetime.now().isoformat()}
total_images: {len(yolo_exporter.annotations)}
"""
        
        yaml_file = os.path.join(export_dir, "data.yaml")
        with open(yaml_file, 'w') as f:
            f.write(data_yaml)
        
        # Create README
        readme_content = f"""# YOLO Format Export - ModelShip

## Dataset Information
- Job ID: {job_id}
- Export Date: {datetime.now().isoformat()}
- Total Images: {len(yolo_exporter.annotations)}
- Categories: {len(categories)}

## File Structure
- classes.txt: List of class names
- labels/: Directory containing YOLO format label files
- data.yaml: YOLOv5/v8 configuration file
- README.md: This file

## Format
Each label file contains lines in the format:
`class_id center_x center_y width height`

All coordinates are normalized (0-1).

## Categories
{chr(10).join(f"{i}: {cat}" for i, cat in enumerate(categories))}

## Usage
This dataset is compatible with:
- YOLOv5/v8
- Ultralytics YOLO
- Darknet
- Other YOLO implementations

Generated by ModelShip Auto-Labeling Platform
"""
        
        readme_file = os.path.join(export_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create ZIP file
        zip_filename = f"yolo_export_{job_id}.zip"
        zip_path = os.path.join("exports", zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        return {
            "export_format": "YOLO",
            "job_id": job_id,
            "download_url": f"/api/export/download/{zip_filename}",
            "file_size_mb": round(os.path.getsize(zip_path) / (1024*1024), 2),
            "statistics": {
                "total_images": len(yolo_exporter.annotations),
                "total_labels": len(yolo_exporter.annotations),
                "categories": categories,
                "filters_applied": {
                    "reviewed_only": include_reviewed_only,
                    "confidence_threshold": confidence_threshold
                }
            },
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"YOLO export failed: {str(e)}")

@router.post("/pascal-voc/{job_id}")
async def export_pascal_voc_format(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    include_reviewed_only: bool = Query(False),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    """Export job results in Pascal VOC XML format"""
    
    try:
        # Verify job ownership
        job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == current_user.id
        ).first()
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Get results
        results_query = db.query(Result).filter(Result.job_id == job_id)
        
        if include_reviewed_only:
            results_query = results_query.filter(Result.reviewed == True)
        
        if confidence_threshold > 0:
            results_query = results_query.filter(Result.confidence >= confidence_threshold)
        
        results = results_query.all()
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found")
        
        # Create Pascal VOC exporter
        voc_exporter = PascalVOCExporter()
        
        # Add annotations
        for result in results:
            if result.file:
                file_path = result.file.file_path
            else:
                file_path = os.path.join("uploads", result.filename)
            
            if os.path.exists(file_path):
                voc_exporter.add_annotation(result, file_path)
        
        # Create export directory
        export_dir = os.path.join("exports", f"pascal_voc_{job_id}_{uuid.uuid4().hex[:8]}")
        annotations_dir = os.path.join(export_dir, "Annotations")
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Save XML annotations
        for image_name, xml_content in voc_exporter.annotations.items():
            xml_file = os.path.join(annotations_dir, f"{image_name}.xml")
            with open(xml_file, 'w') as f:
                f.write(xml_content)
        
        # Create README
        categories = list(set(result.predicted_label for result in results if result.predicted_label))
        
        readme_content = f"""# Pascal VOC Format Export - ModelShip

## Dataset Information
- Job ID: {job_id}
- Export Date: {datetime.now().isoformat()}
- Total Images: {len(voc_exporter.annotations)}
- Categories: {len(categories)}

## File Structure
- Annotations/: Directory containing Pascal VOC XML files
- README.md: This file

## Format
Each XML file contains structured annotation data including:
- Image metadata (size, filename, path)
- Object annotations with bounding boxes
- Class labels and confidence scores

## Categories
{chr(10).join(f"- {cat}" for cat in categories)}

## Usage
This dataset is compatible with:
- TensorFlow Object Detection API
- PyTorch Vision
- OpenCV
- Custom training scripts expecting Pascal VOC format

Generated by ModelShip Auto-Labeling Platform
"""
        
        readme_file = os.path.join(export_dir, "README.md")
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Create ZIP file
        zip_filename = f"pascal_voc_export_{job_id}.zip"
        zip_path = os.path.join("exports", zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(export_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, export_dir)
                    zipf.write(file_path, arcname)
        
        return {
            "export_format": "Pascal VOC",
            "job_id": job_id,
            "download_url": f"/api/export/download/{zip_filename}",
            "file_size_mb": round(os.path.getsize(zip_path) / (1024*1024), 2),
            "statistics": {
                "total_images": len(voc_exporter.annotations),
                "total_annotations": len(voc_exporter.annotations),
                "categories": categories,
                "filters_applied": {
                    "reviewed_only": include_reviewed_only,
                    "confidence_threshold": confidence_threshold
                }
            },
            "created_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pascal VOC export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pascal VOC export failed: {str(e)}")

@router.get("/formats")
async def get_available_export_formats():
    """Get information about available export formats"""
    
    return {
        "available_formats": {
            "coco": {
                "name": "COCO Format",
                "description": "Microsoft Common Objects in Context format",
                "file_types": ["JSON"],
                "use_cases": ["Object detection", "Instance segmentation", "Keypoint detection"],
                "compatible_frameworks": ["Detectron2", "MMDetection", "TensorFlow Object Detection API"],
                "endpoint": "/api/export/formats/coco/{job_id}"
            },
            "yolo": {
                "name": "YOLO Format", 
                "description": "You Only Look Once annotation format",
                "file_types": ["TXT", "YAML"],
                "use_cases": ["Real-time object detection", "Edge deployment"],
                "compatible_frameworks": ["YOLOv5", "YOLOv8", "Ultralytics", "Darknet"],
                "endpoint": "/api/export/formats/yolo/{job_id}"
            },
            "pascal_voc": {
                "name": "Pascal VOC Format",
                "description": "Pascal Visual Object Classes XML format",
                "file_types": ["XML"],
                "use_cases": ["Object detection", "Classification", "Segmentation"],
                "compatible_frameworks": ["TensorFlow", "PyTorch", "OpenCV", "Custom training"],
                "endpoint": "/api/export/formats/pascal-voc/{job_id}"
            },
            "csv": {
                "name": "CSV Format",
                "description": "Comma-separated values format",
                "file_types": ["CSV"],
                "use_cases": ["Data analysis", "Spreadsheet import", "Custom processing"],
                "compatible_frameworks": ["Pandas", "Excel", "R", "Custom scripts"],
                "endpoint": "/api/export/create/{job_id}"
            },
            "json": {
                "name": "JSON Format",
                "description": "JavaScript Object Notation format",
                "file_types": ["JSON"],
                "use_cases": ["Web applications", "API integration", "NoSQL databases"],
                "compatible_frameworks": ["Web frameworks", "REST APIs", "MongoDB"],
                "endpoint": "/api/export/create/{job_id}"
            }
        },
        "format_recommendations": {
            "computer_vision": {
                "object_detection": ["COCO", "YOLO", "Pascal VOC"],
                "image_classification": ["CSV", "JSON", "COCO"],
                "instance_segmentation": ["COCO"],
                "real_time_inference": ["YOLO"]
            },
            "natural_language_processing": {
                "text_classification": ["CSV", "JSON"],
                "named_entity_recognition": ["JSON"],
                "sentiment_analysis": ["CSV", "JSON"]
            }
        }
    } 