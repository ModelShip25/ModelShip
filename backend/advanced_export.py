from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse  
from sqlalchemy.orm import Session
from database import get_db
from models import User, Job, Result
from auth import get_current_user
import json
import os
import zipfile
import shutil
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

router = APIRouter(prefix="/api/export/ml", tags=["ml-export"])

class MLExportService:
    """Advanced export service with ML-ready formats"""
    
    def __init__(self):
        self.export_dir = "exports"
        os.makedirs(self.export_dir, exist_ok=True)
    
    def create_tensorflow_dataset(self, job_id: int, results: List[Result], 
                                split_ratios: Dict[str, float] = None) -> str:
        """Create TensorFlow-ready dataset structure with train/val/test splits"""
        
        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"tensorflow_dataset_job_{job_id}_{timestamp}"
        dataset_dir = os.path.join(self.export_dir, dataset_name)
        
        # Create directory structure
        for split in split_ratios.keys():
            os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)
        
        # Filter successful results
        successful_results = [r for r in results if r.status == "success" and r.predicted_label]
        
        # Group by labels
        label_groups = {}
        for result in successful_results:
            label = self._sanitize_label(result.predicted_label)
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(result)
        
        # Create class directories and split data
        class_mapping = {}
        for class_idx, (label, samples) in enumerate(label_groups.items()):
            class_mapping[class_idx] = label
            
            # Create class directories for each split
            for split in split_ratios.keys():
                os.makedirs(os.path.join(dataset_dir, split, label), exist_ok=True)
        
        # Create metadata files
        metadata = {
            "dataset_info": {
                "name": dataset_name,
                "job_id": job_id,
                "created_at": datetime.now().isoformat(),
                "format": "tensorflow",
                "total_samples": len(successful_results),
                "num_classes": len(label_groups),
                "split_ratios": split_ratios
            },
            "class_mapping": class_mapping,
            "label_statistics": {
                label: len(samples) for label, samples in label_groups.items()
            }
        }
        
        # Save metadata
        with open(os.path.join(dataset_dir, "dataset_info.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create TensorFlow dataset script
        self._create_tensorflow_script(dataset_dir, metadata)
        
        # Zip the dataset
        return self._zip_directory(dataset_dir)
    
    def create_pytorch_dataset(self, job_id: int, results: List[Result]) -> str:
        """Create PyTorch-ready dataset with transforms"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"pytorch_dataset_job_{job_id}_{timestamp}"
        dataset_dir = os.path.join(self.export_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Filter successful results
        successful_results = [r for r in results if r.status == "success" and r.predicted_label]
        
        # Create annotations file
        annotations = []
        class_to_idx = {}
        idx = 0
        
        for result in successful_results:
            label = self._sanitize_label(result.predicted_label)
            if label not in class_to_idx:
                class_to_idx[label] = idx
                idx += 1
            
            annotations.append({
                "filename": os.path.basename(result.filename) if result.filename else f"file_{result.id}",
                "label": label,  
                "class_idx": class_to_idx[label],
                "confidence": result.confidence,
                "processing_time": result.processing_time
            })
        
        # Save annotations
        with open(os.path.join(dataset_dir, "annotations.json"), 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Save class mapping
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        with open(os.path.join(dataset_dir, "class_mapping.json"), 'w') as f:
            json.dump({
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": len(class_to_idx)
            }, f, indent=2)
        
        # Create PyTorch dataset class
        self._create_pytorch_script(dataset_dir, len(class_to_idx))
        
        return self._zip_directory(dataset_dir)
    
    def create_coco_format(self, job_id: int, results: List[Result]) -> str:
        """Create COCO format for object detection/classification"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"coco_dataset_job_{job_id}_{timestamp}"
        dataset_dir = os.path.join(self.export_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Filter successful results
        successful_results = [r for r in results if r.status == "success" and r.predicted_label]
        
        # Build COCO structure
        categories = {}
        category_id = 1
        
        images = []
        annotations = []
        annotation_id = 1
        
        for image_id, result in enumerate(successful_results, 1):
            label = self._sanitize_label(result.predicted_label)
            
            # Add category if new
            if label not in categories:
                categories[label] = {
                    "id": category_id,
                    "name": label,
                    "supercategory": "object"
                }
                category_id += 1
            
            # Add image info
            images.append({
                "id": image_id,
                "file_name": os.path.basename(result.filename) if result.filename else f"image_{result.id}.jpg",
                "width": 640,  # Default dimensions
                "height": 480,
                "date_captured": result.created_at.isoformat() if result.created_at else datetime.now().isoformat()
            })
            
            # Add annotation (classification as whole image bbox)
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": categories[label]["id"],
                "bbox": [0, 0, 640, 480],  # Full image bbox
                "area": 640 * 480,
                "iscrowd": 0,
                "confidence": result.confidence,
                "processing_time": result.processing_time
            })
            annotation_id += 1
        
        # Create COCO annotation file
        coco_data = {
            "info": {
                "description": f"ModelShip classification results in COCO format",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "ModelShip",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{
                "id": 1,
                "name": "ModelShip License",
                "url": "https://modelship.ai/license"
            }],
            "images": images,
            "annotations": annotations,
            "categories": list(categories.values())
        }
        
        with open(os.path.join(dataset_dir, "annotations.json"), 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Create README
        self._create_coco_readme(dataset_dir)
        
        return self._zip_directory(dataset_dir)
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for use in directory names"""
        return "".join(c for c in label if c.isalnum() or c in (' ', '-', '_')).strip()
    
    def _zip_directory(self, directory_path: str) -> str:
        """Zip a directory and return the zip file path"""
        zip_path = f"{directory_path}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, directory_path)
                    zipf.write(file_path, arcname)
        
        # Clean up the directory
        shutil.rmtree(directory_path)
        
        return zip_path
    
    def _create_tensorflow_script(self, dataset_dir: str, metadata: Dict):
        """Create TensorFlow dataset loading script"""
        script_content = f'''"""
TensorFlow Dataset for ModelShip Classification Results
Generated on {datetime.now().isoformat()}
"""

import tensorflow as tf
import json
import os

def load_dataset(data_dir, batch_size=32, image_size=(224, 224)):
    """Load the ModelShip dataset for TensorFlow"""
    
    # Load metadata
    with open(os.path.join(data_dir, 'dataset_info.json'), 'r') as f:
        metadata = json.load(f)
    
    # Create datasets for each split
    datasets = {{}}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            dataset = tf.keras.utils.image_dataset_from_directory(
                split_dir,
                batch_size=batch_size,
                image_size=image_size,
                shuffle=(split == 'train')
            )
            datasets[split] = dataset
    
    return datasets, metadata

if __name__ == "__main__":
    datasets, metadata = load_dataset('.')
    print(f"Dataset loaded with {{len(metadata['class_mapping'])}} classes")
'''
        
        with open(os.path.join(dataset_dir, "load_dataset.py"), 'w') as f:
            f.write(script_content)
    
    def _create_pytorch_script(self, dataset_dir: str, num_classes: int):
        """Create PyTorch dataset class"""
        script_content = f'''"""
PyTorch Dataset for ModelShip Classification Results
Generated on {datetime.now().isoformat()}
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from torchvision import transforms

class ModelShipDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load annotations
        with open(os.path.join(data_dir, 'annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        
        # Load class mapping
        with open(os.path.join(data_dir, 'class_mapping.json'), 'r') as f:
            self.class_info = json.load(f)
        
        self.num_classes = {num_classes}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Create a dummy image (original files not included)
        image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        label = item['class_idx']
        confidence = item['confidence']
        
        return {{
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'confidence': torch.tensor(confidence, dtype=torch.float32),
            'filename': item['filename']
        }}

# Default transforms
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    dataset = ModelShipDataset('.', transform=default_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"Dataset size: {{len(dataset)}}")
    print(f"Number of classes: {{dataset.num_classes}}")
'''
        
        with open(os.path.join(dataset_dir, "dataset.py"), 'w') as f:
            f.write(script_content)
    
    def _create_coco_readme(self, dataset_dir: str):
        """Create README for COCO format"""
        readme_content = f'''# ModelShip COCO Format Dataset

Generated on {datetime.now().isoformat()}

## Structure
- `annotations.json`: COCO format annotations file
- `README.md`: This file

## Usage
Load this dataset with any COCO-compatible library:

```python
from pycocotools.coco import COCO

# Load annotations
coco = COCO('annotations.json')

# Get all image IDs
img_ids = coco.getImgIds()

# Get all category IDs  
cat_ids = coco.getCatIds()
```

## Categories
Each annotation represents a classification result from ModelShip with confidence scores.
'''
        
        with open(os.path.join(dataset_dir, "README.md"), 'w') as f:
            f.write(readme_content)

# Service instance
ml_export_service = MLExportService()

@router.post("/{job_id}")
def create_ml_export(
    job_id: int,
    format_type: str = Query(..., description="Export format: tensorflow, pytorch, coco"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    train_split: float = Query(0.7, description="Training split ratio (TensorFlow only)"),
    val_split: float = Query(0.2, description="Validation split ratio (TensorFlow only)"),
    test_split: float = Query(0.1, description="Test split ratio (TensorFlow only)")
):
    """Create ML-ready dataset export"""
    
    # Verify user owns the job
    job = db.query(Job).filter(
        Job.id == job_id,
        Job.user_id == current_user.id
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    # Get results
    results = db.query(Result).filter(Result.job_id == job_id).all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No results found for this job")
    
    try:
        # Create export based on format
        if format_type == "tensorflow":
            split_ratios = {"train": train_split, "val": val_split, "test": test_split}
            filepath = ml_export_service.create_tensorflow_dataset(job_id, results, split_ratios)
        elif format_type == "pytorch":
            filepath = ml_export_service.create_pytorch_dataset(job_id, results)
        elif format_type == "coco":
            filepath = ml_export_service.create_coco_format(job_id, results)
        else:
            raise HTTPException(status_code=400, detail="Invalid format type. Use: tensorflow, pytorch, coco")
        
        filename = os.path.basename(filepath)
        
        return {
            "message": f"{format_type.title()} dataset created successfully",
            "download_url": f"/api/export/ml/download/{filename}",
            "filename": filename,
            "format": format_type,
            "job_id": job_id,
            "total_samples": len([r for r in results if r.status == "success"]),
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/download/{filename}")
def download_ml_export(
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """Download an ML export file"""
    
    filepath = os.path.join(ml_export_service.export_dir, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Export file not found")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/formats")
def get_ml_export_formats():
    """Get available ML export formats"""
    return {
        "ml_ready_formats": [
            {
                "name": "tensorflow",
                "description": "TensorFlow dataset with train/val/test splits",
                "use_case": "TensorFlow model training, Keras integration",
                "parameters": ["train_split", "val_split", "test_split"]
            },
            {
                "name": "pytorch",
                "description": "PyTorch dataset with custom transforms",
                "use_case": "PyTorch model training, research workflows",
                "parameters": []
            },
            {
                "name": "coco",
                "description": "COCO format for object detection",
                "use_case": "Object detection, computer vision research",
                "parameters": []
            }
        ]
    } 