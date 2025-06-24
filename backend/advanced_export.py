from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse  
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result, Project, File as FileModel, ProjectAssignment
from auth import get_current_user, get_optional_user
import json
import os
import zipfile
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import csv
import uuid
from io import StringIO

router = APIRouter(prefix="/api/export", tags=["export"])

class ExportService:
    """Advanced export service supporting multiple formats"""
    
    def __init__(self):
        self.supported_formats = {
            "json": {
                "description": "JSON format with full metadata",
                "file_extension": ".json",
                "content_type": "application/json"
            },
            "csv": {
                "description": "CSV format for tabular data",
                "file_extension": ".csv", 
                "content_type": "text/csv"
            },
            "coco": {
                "description": "COCO format for object detection",
                "file_extension": ".json",
                "content_type": "application/json"
            },
            "yolo": {
                "description": "YOLO format with text annotations",
                "file_extension": ".zip",
                "content_type": "application/zip"
            }
        }
    
    def export_to_json(self, results: List[Result], include_metadata: bool = True) -> Dict[str, Any]:
        """Export results to comprehensive JSON format"""
        
        export_data = {
            "export_info": {
                "format": "json",
                "created_at": datetime.utcnow().isoformat(),
                "total_items": len(results),
                "include_metadata": include_metadata
            },
            "results": []
        }
        
        for result in results:
            result_data = {
                "id": result.id,
                "filename": result.filename,
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "ground_truth": result.ground_truth,
                "reviewed": result.reviewed,
                "review_action": result.review_action
            }
            
            # Add detection data if available
            if hasattr(result, 'all_predictions') and result.all_predictions:
                result_data["all_predictions"] = result.all_predictions
            
            # Add metadata if requested
            if include_metadata:
                result_data["metadata"] = {
                    "processing_time": result.processing_time,
                    "model_version": result.model_version,
                    "created_at": result.created_at.isoformat() if result.created_at else None,
                    "reviewed_at": result.reviewed_at.isoformat() if result.reviewed_at else None,
                    "file_id": result.file_id,
                    "job_id": result.job_id
                }
            
            export_data["results"].append(result_data)
        
        return export_data
    
    def export_to_csv(self, results: List[Result]) -> str:
        """Export results to CSV format"""
        
        output = StringIO()
        fieldnames = [
            'id', 'filename', 'predicted_label', 'confidence', 
            'ground_truth', 'reviewed', 'review_action', 
            'processing_time', 'created_at'
        ]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'id': result.id,
                'filename': result.filename,
                'predicted_label': result.predicted_label or '',
                'confidence': result.confidence or 0,
                'ground_truth': result.ground_truth or '',
                'reviewed': result.reviewed,
                'review_action': result.review_action or '',
                'processing_time': result.processing_time or 0,
                'created_at': result.created_at.isoformat() if result.created_at else ''
            })
        
        return output.getvalue()
    
    def export_to_coco(self, results: List[Result], project_info: Dict = None) -> Dict[str, Any]:
        """Export object detection results to COCO format"""
        
        # COCO format structure
        coco_data = {
            "info": {
                "description": f"ModelShip Export - {project_info.get('name', 'Unknown Project') if project_info else 'Unknown Project'}",
                "version": "1.0",
                "year": datetime.utcnow().year,
                "contributor": "ModelShip Platform",
                "date_created": datetime.utcnow().isoformat()
            },
            "licenses": [{
                "id": 1,
                "name": "Custom License",
                "url": ""
            }],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Get unique categories from results
        categories = set()
        for result in results:
            if result.predicted_label:
                categories.add(result.predicted_label)
            if result.ground_truth:
                categories.add(result.ground_truth)
        
        # Add categories to COCO format
        for i, category in enumerate(sorted(categories), 1):
            coco_data["categories"].append({
                "id": i,
                "name": category,
                "supercategory": "object"
            })
        
        # Create category mapping
        category_map = {cat["name"]: cat["id"] for cat in coco_data["categories"]}
        
        # Process results
        annotation_id = 1
        for result in results:
            # Add image info
            image_info = {
                "id": result.id,
                "file_name": result.filename,
                "width": 640,  # Default, should be extracted from actual image
                "height": 480,  # Default, should be extracted from actual image
                "license": 1
            }
            coco_data["images"].append(image_info)
            
            # Add annotations (if detection data available)
            if hasattr(result, 'all_predictions') and result.all_predictions:
                try:
                    predictions = json.loads(result.all_predictions) if isinstance(result.all_predictions, str) else result.all_predictions
                    
                    if isinstance(predictions, list):
                        for pred in predictions:
                            if isinstance(pred, dict) and 'bbox' in pred:
                                bbox = pred['bbox']
                                annotation = {
                                    "id": annotation_id,
                                    "image_id": result.id,
                                    "category_id": category_map.get(pred.get('class_name', result.predicted_label), 1),
                                    "bbox": [bbox.get('x1', 0), bbox.get('y1', 0), 
                                           bbox.get('width', 100), bbox.get('height', 100)],
                                    "area": bbox.get('width', 100) * bbox.get('height', 100),
                                    "iscrowd": 0
                                }
                                coco_data["annotations"].append(annotation)
                                annotation_id += 1
                except:
                    # Fallback for simple predictions
                    if result.predicted_label and result.predicted_label in category_map:
                        annotation = {
                            "id": annotation_id,
                            "image_id": result.id,
                            "category_id": category_map[result.predicted_label],
                            "bbox": [0, 0, 100, 100],  # Default bbox
                            "area": 10000,
                            "iscrowd": 0
                        }
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
        
        return coco_data
    
    def export_to_yolo(self, results: List[Result], export_dir: str) -> str:
        """Export results to YOLO format (text files + zip)"""
        
        # Create temporary directory for YOLO files
        yolo_dir = os.path.join(export_dir, "yolo_export")
        os.makedirs(yolo_dir, exist_ok=True)
        
        # Create classes.txt file
        classes = set()
        for result in results:
            if result.predicted_label:
                classes.add(result.predicted_label)
            if result.ground_truth:
                classes.add(result.ground_truth)
        
        classes_list = sorted(classes)
        with open(os.path.join(yolo_dir, "classes.txt"), "w") as f:
            for cls in classes_list:
                f.write(f"{cls}\n")
        
        # Create class mapping
        class_map = {cls: i for i, cls in enumerate(classes_list)}
        
        # Process each result
        for result in results:
            if not result.filename:
                continue
            
            # Create annotation file (same name as image, but .txt extension)
            base_name = os.path.splitext(result.filename)[0]
            annotation_file = os.path.join(yolo_dir, f"{base_name}.txt")
            
            with open(annotation_file, "w") as f:
                # Use ground truth if available, otherwise predicted label
                label = result.ground_truth or result.predicted_label
                
                if label and label in class_map:
                    class_id = class_map[label]
                    
                    # Extract bounding box if available
                    if hasattr(result, 'all_predictions') and result.all_predictions:
                        try:
                            predictions = json.loads(result.all_predictions) if isinstance(result.all_predictions, str) else result.all_predictions
                            
                            if isinstance(predictions, list):
                                for pred in predictions:
                                    if isinstance(pred, dict) and 'bbox' in pred:
                                        bbox = pred['bbox']
                                        # Convert to YOLO format (normalized coordinates)
                                        x_center = (bbox.get('x1', 0) + bbox.get('x2', 100)) / 2 / 640  # Normalize by image width
                                        y_center = (bbox.get('y1', 0) + bbox.get('y2', 100)) / 2 / 480  # Normalize by image height
                                        width = abs(bbox.get('x2', 100) - bbox.get('x1', 0)) / 640
                                        height = abs(bbox.get('y2', 100) - bbox.get('y1', 0)) / 480
                                        
                                        pred_class_id = class_map.get(pred.get('class_name', label), class_id)
                                        f.write(f"{pred_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            else:
                                # Default bounding box for simple classification
                                f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")
                        except:
                            # Default bounding box
                            f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")
                    else:
                        # Default bounding box for classification results
                        f.write(f"{class_id} 0.5 0.5 0.3 0.3\n")
        
        # Create zip file
        zip_path = os.path.join(export_dir, "yolo_export.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(yolo_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, yolo_dir)
                    zipf.write(file_path, arcname)
        
        return zip_path

export_service = ExportService()

@router.get("/formats")
def get_supported_formats():
    """Get list of supported export formats"""
    return {
        "supported_formats": export_service.supported_formats,
        "recommendations": {
            "object_detection": ["coco", "yolo"],
            "image_classification": ["json", "csv"],
            "text_classification": ["json", "csv"],
            "research": ["json"],
            "production": ["coco", "yolo"]
        }
    }

@router.post("/project/{project_id}")
def export_project_results(
    project_id: int,
    export_config: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Export project results in specified format"""
    
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Check access permissions
    has_access = (project.owner_id == current_user.id) or db.query(ProjectAssignment).filter(
        ProjectAssignment.project_id == project_id,
        ProjectAssignment.user_id == current_user.id
    ).first()
    
    if not has_access:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Extract export configuration
    export_format = export_config.get("format", "json")
    include_metadata = export_config.get("include_metadata", True)
    confidence_threshold = export_config.get("confidence_threshold", 0.0)
    include_reviewed_only = export_config.get("include_reviewed_only", False)
    
    # Validate format
    if export_format not in export_service.supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {export_format}")
    
    # Get results based on filters
    query = db.query(Result).join(Job).filter(Job.project_id == project_id)
    
    if confidence_threshold > 0:
        query = query.filter(Result.confidence >= confidence_threshold)
    
    if include_reviewed_only:
        query = query.filter(Result.reviewed == True)
    
    results = query.all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No results found matching criteria")
    
    # Create export directory
    export_dir = os.path.join("uploads", "exports", str(uuid.uuid4()))
    os.makedirs(export_dir, exist_ok=True)
    
    try:
        # Generate export based on format
        if export_format == "json":
            export_data = export_service.export_to_json(results, include_metadata)
            export_filename = f"project_{project_id}_export.json"
            export_path = os.path.join(export_dir, export_filename)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif export_format == "csv":
            csv_data = export_service.export_to_csv(results)
            export_filename = f"project_{project_id}_export.csv"
            export_path = os.path.join(export_dir, export_filename)
            
            with open(export_path, 'w', newline='') as f:
                f.write(csv_data)
        
        elif export_format == "coco":
            project_info = {"name": project.name, "id": project.id}
            coco_data = export_service.export_to_coco(results, project_info)
            export_filename = f"project_{project_id}_coco.json"
            export_path = os.path.join(export_dir, export_filename)
            
            with open(export_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
        
        elif export_format == "yolo":
            export_path = export_service.export_to_yolo(results, export_dir)
            export_filename = "yolo_export.zip"
        
        # Get file size
        file_size = os.path.getsize(export_path)
        
        return {
            "export_id": os.path.basename(export_dir),
            "project_id": project_id,
            "format": export_format,
            "filename": export_filename,
            "file_size": file_size,
            "total_items": len(results),
            "download_url": f"/api/export/download/{os.path.basename(export_dir)}/{export_filename}",
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow().replace(hour=23, minute=59, second=59)).isoformat(),
            "export_config": {
                "format": export_format,
                "include_metadata": include_metadata,
                "confidence_threshold": confidence_threshold,
                "include_reviewed_only": include_reviewed_only
            }
        }
        
    except Exception as e:
        # Clean up on error
        import shutil
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/download/{export_id}/{filename}")
async def download_export(
    export_id: str,
    filename: str,
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Download exported file"""
    
    export_path = os.path.join("uploads", "exports", export_id, filename)
    
    if not os.path.exists(export_path):
        raise HTTPException(status_code=404, detail="Export file not found or expired")
    
    from fastapi.responses import FileResponse
    
    # Determine content type based on file extension
    if filename.endswith('.json'):
        media_type = "application/json"
    elif filename.endswith('.csv'):
        media_type = "text/csv"
    elif filename.endswith('.zip'):
        media_type = "application/zip"
    else:
        media_type = "application/octet-stream"
    
    return FileResponse(
        path=export_path,
        media_type=media_type,
        filename=filename
    )

@router.get("/job/{job_id}")
def export_job_results(
    job_id: int,
    export_format: str = Query(default="json"),
    include_metadata: bool = Query(default=True),
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Export results from a specific job"""
    
    # Verify job ownership
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get job results
    results = db.query(Result).filter(Result.job_id == job_id).all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this job")
    
    # Validate format
    if export_format not in export_service.supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {export_format}")
    
    try:
        if export_format == "json":
            export_data = export_service.export_to_json(results, include_metadata)
            return export_data
        
        elif export_format == "csv":
            csv_data = export_service.export_to_csv(results)
            from fastapi.responses import Response
            return Response(
                content=csv_data,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=job_{job_id}_results.csv"}
            )
        
        else:
            raise HTTPException(status_code=400, detail="Format not supported for job export. Use project export for COCO/YOLO formats.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/project/{project_id}/summary")
def get_export_summary(
    project_id: int,
    current_user: Optional[User] = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Get export summary and statistics for a project"""
    
    # Verify project access
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Get all results
    all_results = db.query(Result).join(Job).filter(Job.project_id == project_id).all()
    reviewed_results = [r for r in all_results if r.reviewed]
    high_confidence = [r for r in all_results if r.confidence and r.confidence > 0.9]
    
    # Calculate statistics
    summary = {
        "project": {
            "id": project.id,
            "name": project.name,
            "project_type": project.project_type
        },
        "data_summary": {
            "total_items": len(all_results),
            "reviewed_items": len(reviewed_results),
            "high_confidence_items": len(high_confidence),
            "exportable_items": len([r for r in all_results if r.status == "success"])
        },
        "recommended_formats": [],
        "export_options": {
            "confidence_thresholds": [0.0, 0.5, 0.7, 0.9],
            "include_metadata": [True, False],
            "reviewed_only": [True, False]
        }
    }
    
    # Add format recommendations based on project type
    if project.project_type == "object_detection":
        summary["recommended_formats"] = ["coco", "yolo", "json"]
    elif project.project_type in ["image_classification", "text_classification"]:
        summary["recommended_formats"] = ["csv", "json"]
    else:
        summary["recommended_formats"] = ["json"]
    
    return summary

@router.post("/project/{project_id}/test")
def export_project_results_test(
    project_id: int,
    export_config: Dict[str, Any]
):
    """Export test project results for Phase 1 testing - no authentication required"""
    
    try:
        # Extract export configuration
        export_format = export_config.get("format", "json")
        include_metadata = export_config.get("include_metadata", True)
        confidence_threshold = export_config.get("confidence_threshold", 0.0)
        
        # Validate format
        if export_format not in export_service.supported_formats:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {export_format}")
        
        # Create enhanced mock export data for testing
        mock_export_data = {
            "export_info": {
                "project_id": project_id,
                "format": export_format,
                "generated_at": datetime.utcnow().isoformat(),
                "include_metadata": include_metadata,
                "confidence_threshold": confidence_threshold,
                "exported_by": "Phase 1 Testing System"
            },
            "results": [
                {
                    "filename": "test_family_photo.jpg",
                    "predicted_label": "person",
                    "confidence": 0.94,
                    "detections": [
                        {
                            "class_name": "person",
                            "confidence": 0.94,
                            "bbox": {"x1": 120, "y1": 80, "x2": 280, "y2": 420}
                        },
                        {
                            "class_name": "person", 
                            "confidence": 0.88,
                            "bbox": {"x1": 350, "y1": 60, "x2": 480, "y2": 380}
                        }
                    ],
                    "status": "success",
                    "reviewed": True,
                    "processing_time": 1.2
                },
                {
                    "filename": "street_scene.jpg", 
                    "predicted_label": "car",
                    "confidence": 0.91,
                    "detections": [
                        {
                            "class_name": "car",
                            "confidence": 0.91,
                            "bbox": {"x1": 50, "y1": 180, "x2": 320, "y2": 380}
                        },
                        {
                            "class_name": "person",
                            "confidence": 0.76,
                            "bbox": {"x1": 400, "y1": 120, "x2": 450, "y2": 280}
                        }
                    ],
                    "status": "success",
                    "reviewed": False,
                    "processing_time": 0.9
                },
                {
                    "filename": "living_room.jpg",
                    "predicted_label": "couch",
                    "confidence": 0.85,
                    "detections": [
                        {
                            "class_name": "couch",
                            "confidence": 0.85,
                            "bbox": {"x1": 80, "y1": 200, "x2": 520, "y2": 400}
                        },
                        {
                            "class_name": "potted plant",
                            "confidence": 0.79,
                            "bbox": {"x1": 550, "y1": 150, "x2": 600, "y2": 300}
                        }
                    ],
                    "status": "success", 
                    "reviewed": True,
                    "processing_time": 1.1
                }
            ],
            "statistics": {
                "total_items": 3,
                "reviewed_items": 2,
                "average_confidence": 0.897,
                "classes_detected": ["person", "car", "couch", "potted plant"],
                "total_detections": 6,
                "processing_summary": {
                    "total_processing_time": 3.2,
                    "average_processing_time": 1.07,
                    "files_processed": 3
                }
            }
        }
        
        # Create a mock download URL
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_export_project_{project_id}_{export_format}_{timestamp}.{export_service.supported_formats[export_format]['file_extension'].lstrip('.')}"
        download_url = f"/api/export/download/{filename}"
        
        return {
            "message": "Test export generated successfully",
            "project_id": project_id,
            "format": export_format,
            "filename": filename,
            "download_url": download_url,
            "file_size_mb": 3.2,
            "total_items": mock_export_data["statistics"]["total_items"],
            "export_data": mock_export_data if export_format == "json" else None,
            "created_at": datetime.utcnow().isoformat(),
            "note": "This is enhanced test export data for Phase 1 testing",
            "export_preview": {
                "format_description": export_service.supported_formats[export_format]["description"],
                "content_type": export_service.supported_formats[export_format]["content_type"],
                "includes_annotations": True,
                "includes_metadata": include_metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test export failed: {str(e)}") 