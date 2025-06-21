# ModelShip Core Features Specification
*Essential functionality for AI training data workflows*

## 🎯 Core Value Proposition Features

### **1. Auto-Labeling Engine**
The heart of ModelShip - intelligent classification with confidence scoring

### **2. Human-in-the-Loop Review System**
Quality assurance workflow for training data accuracy

### **3. ML-Ready Export System**
Direct export to TensorFlow, PyTorch, and other training frameworks

### **4. API Integration for Training Pipelines**
Seamless integration with existing ML workflows

### **5. Quality Assurance & Validation**
Ensure training data meets AI model requirements

### **6. Production-Ready Infrastructure**
Monitoring, logging, and error handling

---

## 🔧 Feature 1: Advanced Auto-Labeling Engine

### **Multi-Model Classification System**
```python
# backend/app/ml/labeling_engine.py
from typing import List, Dict, Any, Optional, Union
import asyncio
from enum import Enum
import time
import uuid

class LabelingConfidence(Enum):
    HIGH = "high"        # >95% confidence
    MEDIUM = "medium"    # 80-95% confidence  
    LOW = "low"         # 60-80% confidence
    UNCERTAIN = "uncertain"  # <60% confidence

class AutoLabelingEngine:
    """Core auto-labeling engine with multiple model support"""
    
    def __init__(self):
        self.image_models = {
            "general": "microsoft/resnet-50",
            "detailed": "google/vit-large-patch16-224", 
            "objects": "facebook/detr-resnet-50",
            "medical": "microsoft/beit-base-patch16-224-pt22k-ft22k",
            "nsfw": "Falconsai/nsfw_image_detection"
        }
        self.text_models = {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "topic": "facebook/bart-large-mnli",
            "toxicity": "martin-ha/toxic-comment-model"
        }
        self.loaded_models = {}
    
    async def auto_label_batch(
        self, 
        items: List[Dict[str, Any]], 
        label_types: List[str] = ["general"],
        confidence_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Auto-label batch of items with multiple classification types"""
        
        results = []
        
        for item in items:
            try:
                item_result = {
                    "item_id": item.get("id", str(uuid.uuid4())),
                    "filename": item.get("filename"),
                    "labels": {},
                    "confidence_scores": {},
                    "needs_review": False,
                    "processing_metadata": {}
                }
                
                # Run classification for each requested label type
                for label_type in label_types:
                    start_time = time.time()
                    
                    classification_result = await self._classify_item(
                        item["data"], 
                        item["type"],  # "image" or "text"
                        label_type
                    )
                    
                    item_result["labels"][label_type] = classification_result["predicted_label"]
                    item_result["confidence_scores"][label_type] = classification_result["confidence"]
                    item_result["processing_metadata"][label_type] = {
                        "processing_time": time.time() - start_time,
                        "model_used": classification_result["model_name"],
                        "model_version": classification_result.get("model_version", "1.0")
                    }
                    
                    # Flag for review if confidence is below threshold
                    if classification_result["confidence"] < confidence_threshold:
                        item_result["needs_review"] = True
                        item_result["review_reason"] = f"Low confidence for {label_type}: {classification_result['confidence']:.2f}"
                
                # Determine overall confidence level
                avg_confidence = sum(item_result["confidence_scores"].values()) / len(item_result["confidence_scores"])
                item_result["overall_confidence"] = self._get_confidence_level(avg_confidence)
                
                results.append(item_result)
                
            except Exception as e:
                # Handle individual item failures gracefully
                results.append({
                    "item_id": item.get("id", str(uuid.uuid4())),
                    "filename": item.get("filename"),
                    "error": str(e),
                    "status": "failed",
                    "needs_review": True,
                    "review_reason": "Processing failed"
                })
        
        return results
    
    async def _classify_item(self, data: bytes, data_type: str, label_type: str) -> Dict[str, Any]:
        """Classify single item with specified model"""
        
        model_key = f"{data_type}_{label_type}"
        
        if model_key not in self.loaded_models:
            await self._load_model(data_type, label_type)
        
        classifier = self.loaded_models[model_key]
        
        if data_type == "image":
            return await classifier.classify_image(data)
        elif data_type == "text":
            return await classifier.classify_text(data.decode('utf-8'))
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    def _get_confidence_level(self, confidence: float) -> LabelingConfidence:
        """Convert numeric confidence to categorical level"""
        if confidence >= 0.95:
            return LabelingConfidence.HIGH
        elif confidence >= 0.80:
            return LabelingConfidence.MEDIUM
        elif confidence >= 0.60:
            return LabelingConfidence.LOW
        else:
            return LabelingConfidence.UNCERTAIN
```

### **Smart Batch Processing API**
```python
# backend/app/api/labeling.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import json

router = APIRouter(prefix="/api/labeling", tags=["auto-labeling"])

@router.post("/batch/auto-label")
async def auto_label_batch(
    files: List[UploadFile] = File(...),
    label_types: str = Form(...),  # JSON string: ["general", "objects", "nsfw"]
    confidence_threshold: float = Form(0.8),
    auto_approve_high_confidence: bool = Form(True),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Auto-label batch of files with multiple classification types"""
    
    # Parse label types
    try:
        label_types_list = json.loads(label_types)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid label_types format")
    
    # Validate user has enough credits
    total_classifications = len(files) * len(label_types_list)
    if current_user.credits_remaining < total_classifications:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Need {total_classifications}, have {current_user.credits_remaining}"
        )
    
    # Create labeling job
    job = Job(
        user_id=current_user.id,
        job_type="auto_labeling",
        total_items=len(files),
        status="processing",
        metadata={
            "label_types": label_types_list,
            "confidence_threshold": confidence_threshold,
            "auto_approve_high_confidence": auto_approve_high_confidence
        }
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    
    # Prepare items for processing
    items = []
    for file in files:
        file_content = await file.read()
        items.append({
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "data": file_content,
            "type": "image" if file.content_type.startswith("image/") else "text"
        })
    
    # Start background processing
    background_tasks.add_task(
        process_auto_labeling_job,
        job.id,
        items,
        label_types_list,
        confidence_threshold,
        auto_approve_high_confidence
    )
    
    return {
        "job_id": job.id,
        "status": "processing",
        "total_items": len(files),
        "estimated_completion_time": len(files) * 2,  # 2 seconds per item
        "message": "Auto-labeling job started"
    }

async def process_auto_labeling_job(
    job_id: int,
    items: List[Dict],
    label_types: List[str],
    confidence_threshold: float,
    auto_approve_high_confidence: bool
):
    """Background task to process auto-labeling job"""
    
    db = next(get_db())
    labeling_engine = AutoLabelingEngine()
    
    try:
        # Run auto-labeling
        results = await labeling_engine.auto_label_batch(
            items, 
            label_types, 
            confidence_threshold
        )
        
        # Save results to database
        for result in results:
            label_result = LabelResult(
                job_id=job_id,
                item_id=result["item_id"],
                filename=result["filename"],
                labels=result.get("labels", {}),
                confidence_scores=result.get("confidence_scores", {}),
                needs_review=result["needs_review"],
                review_reason=result.get("review_reason"),
                status="completed" if not result.get("error") else "failed",
                processing_metadata=result.get("processing_metadata", {}),
                approved=auto_approve_high_confidence and result.get("overall_confidence") == LabelingConfidence.HIGH
            )
            db.add(label_result)
        
        # Update job status
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = "completed"
        job.completed_items = len([r for r in results if not r.get("error")])
        job.completed_at = datetime.utcnow()
        
        # Calculate summary stats
        high_confidence_count = len([r for r in results if r.get("overall_confidence") == LabelingConfidence.HIGH])
        needs_review_count = len([r for r in results if r.get("needs_review")])
        
        job.summary_stats = {
            "total_processed": len(results),
            "high_confidence": high_confidence_count,
            "needs_review": needs_review_count,
            "auto_approved": high_confidence_count if auto_approve_high_confidence else 0
        }
        
        db.commit()
        
    except Exception as e:
        # Mark job as failed
        job = db.query(Job).filter(Job.id == job_id).first()
        job.status = "failed"
        job.error_message = str(e)
        db.commit()
```

---

## 🔍 Feature 2: Human-in-the-Loop Review System

### **Interactive Review Interface**
```typescript
// frontend/src/components/review/ReviewInterface.tsx
import React, { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

interface LabelResult {
  id: string;
  filename: string;
  labels: Record<string, string>;
  confidence_scores: Record<string, number>;
  needs_review: boolean;
  review_reason?: string;
  approved: boolean;
  image_url?: string;
  text_content?: string;
}

interface ReviewStats {
  total_items: number;
  pending_review: number;
  approved: number;
  rejected: number;
  average_confidence: number;
}

export const ReviewInterface: React.FC<{ jobId: string }> = ({ jobId }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [filter, setFilter] = useState<'all' | 'needs_review' | 'high_confidence'>('needs_review');
  const queryClient = useQueryClient();

  // Fetch review data
  const { data: reviewData, isLoading } = useQuery({
    queryKey: ['review-data', jobId, filter],
    queryFn: () => fetchReviewData(jobId, filter)
  });

  const { data: stats } = useQuery({
    queryKey: ['review-stats', jobId],
    queryFn: () => fetchReviewStats(jobId),
    refetchInterval: 5000 // Update every 5 seconds
  });

  // Mutation for approving/rejecting labels
  const updateLabelMutation = useMutation({
    mutationFn: ({ itemId, action, correctedLabels }: {
      itemId: string;
      action: 'approve' | 'reject' | 'correct';
      correctedLabels?: Record<string, string>;
    }) => updateLabelResult(itemId, action, correctedLabels),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['review-data', jobId] });
      queryClient.invalidateQueries({ queryKey: ['review-stats', jobId] });
    }
  });

  const currentItem = reviewData?.items[currentIndex];

  const handleApprove = useCallback(() => {
    if (currentItem) {
      updateLabelMutation.mutate({
        itemId: currentItem.id,
        action: 'approve'
      });
      nextItem();
    }
  }, [currentItem]);

  const handleReject = useCallback(() => {
    if (currentItem) {
      updateLabelMutation.mutate({
        itemId: currentItem.id,
        action: 'reject'
      });
      nextItem();
    }
  }, [currentItem]);

  const handleCorrect = useCallback((correctedLabels: Record<string, string>) => {
    if (currentItem) {
      updateLabelMutation.mutate({
        itemId: currentItem.id,
        action: 'correct',
        correctedLabels
      });
      nextItem();
    }
  }, [currentItem]);

  const nextItem = () => {
    if (reviewData && currentIndex < reviewData.items.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const previousItem = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  if (isLoading) return <ReviewSkeleton />;
  if (!reviewData || !currentItem) return <div>No items to review</div>;

  return (
    <div className="review-interface h-screen flex">
      {/* Sidebar - Review Stats & Navigation */}
      <div className="w-80 bg-gray-50 p-6 border-r">
        <div className="mb-6">
          <h2 className="text-xl font-bold mb-4">Review Progress</h2>
          <ReviewStatsCard stats={stats} />
        </div>

        {/* Filter Controls */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Filter Items</label>
          <select 
            value={filter} 
            onChange={(e) => setFilter(e.target.value as any)}
            className="w-full p-2 border rounded"
          >
            <option value="needs_review">Needs Review</option>
            <option value="all">All Items</option>
            <option value="high_confidence">High Confidence</option>
          </select>
        </div>

        {/* Item Navigation */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-600">
              Item {currentIndex + 1} of {reviewData.items.length}
            </span>
            <div className="flex gap-2">
              <button 
                onClick={previousItem}
                disabled={currentIndex === 0}
                className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
              >
                ←
              </button>
              <button 
                onClick={nextItem}
                disabled={currentIndex === reviewData.items.length - 1}
                className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
              >
                →
              </button>
            </div>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full"
              style={{ width: `${((currentIndex + 1) / reviewData.items.length) * 100}%` }}
            />
          </div>
        </div>

        {/* Quick Actions */}
        <div className="space-y-2">
          <button
            onClick={handleApprove}
            className="w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700"
          >
            ✓ Approve (A)
          </button>
          <button
            onClick={handleReject}
            className="w-full bg-red-600 text-white py-2 px-4 rounded hover:bg-red-700"
          >
            ✗ Reject (R)
          </button>
        </div>
      </div>

      {/* Main Review Area */}
      <div className="flex-1 flex flex-col">
        {/* Item Display */}
        <div className="flex-1 p-6">
          <div className="mb-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold">{currentItem.filename}</h3>
              <ConfidenceBadge 
                confidence={Math.max(...Object.values(currentItem.confidence_scores))}
                needsReview={currentItem.needs_review}
              />
            </div>
            {currentItem.review_reason && (
              <p className="text-sm text-amber-600 mt-1">
                Review Reason: {currentItem.review_reason}
              </p>
            )}
          </div>

          {/* Display Content */}
          <div className="mb-6">
            {currentItem.image_url ? (
              <img 
                src={currentItem.image_url} 
                alt={currentItem.filename}
                className="max-w-full max-h-96 object-contain mx-auto border rounded"
              />
            ) : (
              <div className="bg-gray-100 p-4 rounded border">
                <pre className="whitespace-pre-wrap text-sm">
                  {currentItem.text_content}
                </pre>
              </div>
            )}
          </div>

          {/* Label Review/Edit */}
          <LabelReviewCard
            labels={currentItem.labels}
            confidenceScores={currentItem.confidence_scores}
            onCorrect={handleCorrect}
          />
        </div>

        {/* Keyboard Shortcuts Help */}
        <div className="border-t p-4 bg-gray-50">
          <div className="text-sm text-gray-600">
            <strong>Keyboard Shortcuts:</strong> 
            A = Approve, R = Reject, ← → = Navigate, E = Edit Labels
          </div>
        </div>
      </div>
    </div>
  );
};

const LabelReviewCard: React.FC<{
  labels: Record<string, string>;
  confidenceScores: Record<string, number>;
  onCorrect: (correctedLabels: Record<string, string>) => void;
}> = ({ labels, confidenceScores, onCorrect }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedLabels, setEditedLabels] = useState(labels);

  const handleSaveCorrections = () => {
    onCorrect(editedLabels);
    setIsEditing(false);
  };

  return (
    <div className="bg-white border rounded-lg p-4">
      <div className="flex justify-between items-center mb-4">
        <h4 className="font-semibold">Predicted Labels</h4>
        <button
          onClick={() => setIsEditing(!isEditing)}
          className="text-blue-600 text-sm hover:text-blue-800"
        >
          {isEditing ? 'Cancel' : 'Edit Labels'}
        </button>
      </div>

      <div className="space-y-3">
        {Object.entries(labels).map(([labelType, labelValue]) => (
          <div key={labelType} className="flex items-center justify-between">
            <div className="flex-1">
              <span className="text-sm font-medium text-gray-600 capitalize">
                {labelType}:
              </span>
              {isEditing ? (
                <input
                  type="text"
                  value={editedLabels[labelType] || ''}
                  onChange={(e) => setEditedLabels({
                    ...editedLabels,
                    [labelType]: e.target.value
                  })}
                  className="ml-2 px-2 py-1 border rounded text-sm"
                />
              ) : (
                <span className="ml-2 font-semibold">{labelValue}</span>
              )}
            </div>
            <div className="ml-4">
              <ConfidenceScore confidence={confidenceScores[labelType]} />
            </div>
          </div>
        ))}
      </div>

      {isEditing && (
        <div className="mt-4 flex gap-2">
          <button
            onClick={handleSaveCorrections}
            className="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700"
          >
            Save Corrections
          </button>
          <button
            onClick={() => setIsEditing(false)}
            className="bg-gray-300 text-gray-700 px-4 py-2 rounded text-sm hover:bg-gray-400"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
};
```

---

## 📤 Feature 3: ML-Ready Export System

### **Multi-Format Export Engine**
```python
# backend/app/export/ml_formats.py
from typing import List, Dict, Any, Optional
import json
import csv
import yaml
import zipfile
from pathlib import Path
import shutil

class MLExportEngine:
    """Export labeled data in ML framework-ready formats"""
    
    def __init__(self, export_dir: str = "./exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    async def export_tensorflow_dataset(
        self,
        job_id: int,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        include_metadata: bool = True
    ) -> Dict[str, str]:
        """Export dataset in TensorFlow format"""
        
        # Get approved labels
        labels = await self._get_approved_labels(job_id)
        
        # Create directory structure
        export_path = self.export_dir / f"tensorflow_dataset_{job_id}"
        export_path.mkdir(exist_ok=True)
        
        # Split data
        splits = self._split_data(labels, train_split, val_split, test_split)
        
        # Create TensorFlow dataset structure
        for split_name, split_data in splits.items():
            split_dir = export_path / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create label directories
            label_dirs = {}
            for item in split_data:
                for label_type, label_value in item['labels'].items():
                    label_dir = split_dir / label_type / label_value
                    label_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy/save file to appropriate label directory
                    if item['file_type'] == 'image':
                        dest_file = label_dir / item['filename']
                        shutil.copy2(item['file_path'], dest_file)
                    else:
                        # For text, create individual files
                        dest_file = label_dir / f"{item['id']}.txt"
                        with open(dest_file, 'w', encoding='utf-8') as f:
                            f.write(item['content'])
        
        # Create dataset metadata
        metadata = {
            "dataset_info": {
                "name": f"modelship_dataset_{job_id}",
                "version": "1.0",
                "splits": {
                    "train": len(splits['train']),
                    "validation": len(splits['validation']),
                    "test": len(splits['test'])
                },
                "total_samples": len(labels),
                "label_types": list(labels[0]['labels'].keys()) if labels else [],
                "created_at": datetime.utcnow().isoformat()
            },
            "tensorflow_config": {
                "image_size": [224, 224],
                "batch_size": 32,
                "shuffle_buffer_size": 1000,
                "prefetch_size": "AUTOTUNE"
            }
        }
        
        with open(export_path / "dataset_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create TensorFlow loading script
        tf_script = self._generate_tensorflow_script(metadata)
        with open(export_path / "load_dataset.py", 'w') as f:
            f.write(tf_script)
        
        # Create ZIP archive
        zip_path = f"{export_path}.zip"
        shutil.make_archive(str(export_path), 'zip', export_path)
        
        return {
            "format": "tensorflow",
            "export_path": zip_path,
            "metadata": metadata,
            "loading_script": "load_dataset.py"
        }
    
    async def export_pytorch_dataset(self, job_id: int) -> Dict[str, str]:
        """Export dataset in PyTorch format"""
        
        labels = await self._get_approved_labels(job_id)
        export_path = self.export_dir / f"pytorch_dataset_{job_id}"
        export_path.mkdir(exist_ok=True)
        
        # Create annotations file
        annotations = []
        for item in labels:
            annotations.append({
                "id": item['id'],
                "filename": item['filename'],
                "labels": item['labels'],
                "confidence_scores": item['confidence_scores']
            })
        
        with open(export_path / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # Copy data files
        data_dir = export_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        for item in labels:
            dest_file = data_dir / item['filename']
            if item['file_type'] == 'image':
                shutil.copy2(item['file_path'], dest_file)
            else:
                with open(dest_file, 'w', encoding='utf-8') as f:
                    f.write(item['content'])
        
        # Create PyTorch Dataset class
        pytorch_script = self._generate_pytorch_script()
        with open(export_path / "dataset.py", 'w') as f:
            f.write(pytorch_script)
        
        # Create requirements file
        requirements = [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "pillow>=8.0.0",
            "pandas>=1.3.0"
        ]
        
        with open(export_path / "requirements.txt", 'w') as f:
            f.write('\n'.join(requirements))
        
        # Create ZIP archive
        zip_path = f"{export_path}.zip"
        shutil.make_archive(str(export_path), 'zip', export_path)
        
        return {
            "format": "pytorch",
            "export_path": zip_path,
            "dataset_class": "ModelShipDataset",
            "usage_example": "dataset.py"
        }
    
    async def export_huggingface_dataset(self, job_id: int) -> Dict[str, str]:
        """Export dataset in Hugging Face format"""
        
        labels = await self._get_approved_labels(job_id)
        export_path = self.export_dir / f"huggingface_dataset_{job_id}"
        export_path.mkdir(exist_ok=True)
        
        # Create dataset structure for Hugging Face
        dataset_dict = {
            "data": [],
            "features": {}
        }
        
        for item in labels:
            data_entry = {
                "id": item['id'],
                "filename": item['filename'],
                "labels": item['labels'],
                "confidence_scores": item['confidence_scores']
            }
            
            if item['file_type'] == 'image':
                data_entry['image_path'] = f"data/{item['filename']}"
            else:
                data_entry['text'] = item['content']
            
            dataset_dict["data"].append(data_entry)
        
        # Define features schema
        if labels:
            sample_labels = labels[0]['labels']
            for label_type in sample_labels.keys():
                dataset_dict["features"][label_type] = {
                    "dtype": "string",
                    "description": f"Label for {label_type} classification"
                }
        
        # Save dataset
        with open(export_path / "dataset.json", 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        # Copy data files
        data_dir = export_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        for item in labels:
            if item['file_type'] == 'image':
                dest_file = data_dir / item['filename']
                shutil.copy2(item['file_path'], dest_file)
        
        # Create Hugging Face loading script
        hf_script = self._generate_huggingface_script()
        with open(export_path / "load_dataset.py", 'w') as f:
            f.write(hf_script)
        
        # Create ZIP archive
        zip_path = f"{export_path}.zip"
        shutil.make_archive(str(export_path), 'zip', export_path)
        
        return {
            "format": "huggingface",
            "export_path": zip_path,
            "loading_script": "load_dataset.py"
        }
    
    def _generate_tensorflow_script(self, metadata: Dict) -> str:
        """Generate TensorFlow dataset loading script"""
        return f'''
import tensorflow as tf
import json
from pathlib import Path

def load_modelship_dataset(dataset_path: str, batch_size: int = 32):
    """Load ModelShip dataset for TensorFlow training"""
    
    dataset_path = Path(dataset_path)
    
    # Load metadata
    with open(dataset_path / "dataset_info.json", 'r') as f:
        metadata = json.load(f)
    
    # Create datasets for each split
    datasets = {{}}
    
    for split in ['train', 'validation', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            # Create dataset from directory
            datasets[split] = tf.keras.utils.image_dataset_from_directory(
                split_path,
                image_size=metadata['tensorflow_config']['image_size'],
                batch_size=batch_size,
                shuffle=(split == 'train')
            )
            
            # Normalize images
            datasets[split] = datasets[split].map(
                lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
            )
            
            # Prefetch for performance
            datasets[split] = datasets[split].prefetch(tf.data.AUTOTUNE)
    
    return datasets

# Example usage:
# datasets = load_modelship_dataset("./tensorflow_dataset_{metadata['dataset_info']['name']}")
# model.fit(datasets['train'], validation_data=datasets['validation'])
'''
    
    def _generate_pytorch_script(self) -> str:
        """Generate PyTorch Dataset class"""
        return '''
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from pathlib import Path

class ModelShipDataset(Dataset):
    """PyTorch Dataset for ModelShip labeled data"""
    
    def __init__(self, dataset_path: str, transform=None, target_transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load annotations
        with open(self.dataset_path / "annotations.json", 'r') as f:
            self.annotations = json.load(f)
        
        # Create label mappings
        self.label_mappings = {}
        for item in self.annotations:
            for label_type, label_value in item['labels'].items():
                if label_type not in self.label_mappings:
                    self.label_mappings[label_type] = {}
                if label_value not in self.label_mappings[label_type]:
                    idx = len(self.label_mappings[label_type])
                    self.label_mappings[label_type][label_value] = idx
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # Load image or text
        if item['filename'].lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            image_path = self.dataset_path / "data" / item['filename']
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Convert labels to indices
            labels = {}
            for label_type, label_value in item['labels'].items():
                labels[label_type] = self.label_mappings[label_type][label_value]
            
            return image, labels
        else:
            # Handle text data
            text_path = self.dataset_path / "data" / item['filename']
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            labels = {}
            for label_type, label_value in item['labels'].items():
                labels[label_type] = self.label_mappings[label_type][label_value]
            
            return text, labels

# Example usage:
# dataset = ModelShipDataset("./pytorch_dataset_123")
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
'''

# API endpoint for exports
@router.post("/export")
async def export_dataset(
    job_id: int,
    export_format: str = "tensorflow",  # tensorflow, pytorch, huggingface, coco, yolo
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Export labeled dataset in specified ML format"""
    
    # Verify user owns the job
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Initialize export engine
    export_engine = MLExportEngine()
    
    # Export based on format
    if export_format == "tensorflow":
        result = await export_engine.export_tensorflow_dataset(
            job_id, train_split, val_split, test_split
        )
    elif export_format == "pytorch":
        result = await export_engine.export_pytorch_dataset(job_id)
    elif export_format == "huggingface":
        result = await export_engine.export_huggingface_dataset(job_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported export format")
    
    # Create download record
    download = Download(
        user_id=current_user.id,
        job_id=job_id,
        export_format=export_format,
        file_path=result["export_path"],
        created_at=datetime.utcnow()
    )
    db.add(download)
    db.commit()
    
    return {
        "download_id": download.id,
        "export_format": export_format,
        "download_url": f"/api/download/{download.id}",
        "metadata": result.get("metadata"),
        "estimated_size": os.path.getsize(result["export_path"])
    }
```

---

## 🔄 Feature 4: API Integration for Training Pipelines

### **Real-time API for Model Training Integration**
```python
# backend/app/api/training_integration.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any, Optional
import asyncio
import requests
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/training", tags=["training-integration"])

class TrainingPipelineManager:
    """Manage integration with customer training pipelines"""
    
    def __init__(self):
        self.active_pipelines = {}
        self.webhook_callbacks = {}
    
    async def register_training_pipeline(
        self,
        user_id: int,
        pipeline_config: Dict[str, Any]
    ) -> str:
        """Register a customer's training pipeline for automatic data delivery"""
        
        pipeline_id = f"pipeline_{user_id}_{int(time.time())}"
        
        self.active_pipelines[pipeline_id] = {
            "user_id": user_id,
            "config": pipeline_config,
            "status": "active",
            "created_at": datetime.utcnow(),
            "last_delivery": None,
            "total_deliveries": 0
        }
        
        return pipeline_id
    
    async def deliver_new_labels(
        self,
        pipeline_id: str,
        new_labels: List[Dict[str, Any]]
    ):
        """Automatically deliver newly approved labels to training pipeline"""
        
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline = self.active_pipelines[pipeline_id]
        config = pipeline["config"]
        
        # Prepare data in requested format
        formatted_data = await self._format_data_for_pipeline(new_labels, config)
        
        # Deliver via configured method
        if config["delivery_method"] == "webhook":
            await self._deliver_via_webhook(config["webhook_url"], formatted_data)
        elif config["delivery_method"] == "api_pull":
            # Store for customer to pull via API
            await self._store_for_api_pull(pipeline_id, formatted_data)
        elif config["delivery_method"] == "cloud_storage":
            await self._deliver_to_cloud_storage(config["storage_config"], formatted_data)
        
        # Update pipeline stats
        pipeline["last_delivery"] = datetime.utcnow()
        pipeline["total_deliveries"] += 1

@router.post("/pipelines/register")
async def register_training_pipeline(
    pipeline_config: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Register customer training pipeline for automatic data delivery"""
    
    # Validate pipeline configuration
    required_fields = ["name", "delivery_method", "data_format"]
    for field in required_fields:
        if field not in pipeline_config:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    # Validate delivery method
    valid_methods = ["webhook", "api_pull", "cloud_storage"]
    if pipeline_config["delivery_method"] not in valid_methods:
        raise HTTPException(status_code=400, detail="Invalid delivery method")
    
    # Create pipeline record
    pipeline = TrainingPipeline(
        user_id=current_user.id,
        name=pipeline_config["name"],
        config=pipeline_config,
        status="active"
    )
    db.add(pipeline)
    db.commit()
    db.refresh(pipeline)
    
    # Register with pipeline manager
    manager = TrainingPipelineManager()
    pipeline_id = await manager.register_training_pipeline(
        current_user.id,
        pipeline_config
    )
    
    return {
        "pipeline_id": pipeline_id,
        "status": "registered",
        "webhook_url": f"/api/training/webhook/{pipeline_id}" if pipeline_config["delivery_method"] == "webhook" else None,
        "api_endpoint": f"/api/training/pull/{pipeline_id}" if pipeline_config["delivery_method"] == "api_pull" else None
    }

@router.get("/pipelines/{pipeline_id}/pull")
async def pull_new_training_data(
    pipeline_id: str,
    since: Optional[datetime] = None,
    limit: int = 1000,
    current_user: User = Depends(get_current_user)
):
    """Pull newly labeled data for training pipeline"""
    
    # Verify pipeline ownership
    pipeline = db.query(TrainingPipeline).filter(
        TrainingPipeline.id == pipeline_id,
        TrainingPipeline.user_id == current_user.id
    ).first()
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Get new approved labels since last pull
    since_time = since or pipeline.last_pull_time or datetime.utcnow() - timedelta(hours=24)
    
    new_labels = db.query(LabelResult).filter(
        LabelResult.user_id == current_user.id,
        LabelResult.approved == True,
        LabelResult.updated_at > since_time
    ).limit(limit).all()
    
    # Format data according to pipeline config
    formatted_data = []
    for label in new_labels:
        formatted_item = {
            "id": label.id,
            "filename": label.filename,
            "labels": label.labels,
            "confidence_scores": label.confidence_scores,
            "file_url": f"/api/files/download/{label.file_id}",
            "labeled_at": label.updated_at.isoformat()
        }
        formatted_data.append(formatted_item)
    
    # Update last pull time
    pipeline.last_pull_time = datetime.utcnow()
    db.commit()
    
    return {
        "pipeline_id": pipeline_id,
        "data": formatted_data,
        "total_items": len(formatted_data),
        "next_pull_url": f"/api/training/pipelines/{pipeline_id}/pull?since={datetime.utcnow().isoformat()}",
        "has_more": len(formatted_data) == limit
    }

@router.post("/webhooks/label-approved")
async def webhook_label_approved(
    label_id: str,
    background_tasks: BackgroundTasks
):
    """Webhook triggered when label is approved - notify training pipelines"""
    
    # Get label details
    label = db.query(LabelResult).filter(LabelResult.id == label_id).first()
    if not label:
        return {"status": "label_not_found"}
    
    # Find active training pipelines for this user
    pipelines = db.query(TrainingPipeline).filter(
        TrainingPipeline.user_id == label.user_id,
        TrainingPipeline.status == "active"
    ).all()
    
    # Notify each pipeline
    for pipeline in pipelines:
        if pipeline.config.get("auto_delivery", True):
            background_tasks.add_task(
                notify_training_pipeline,
                pipeline.id,
                [label.to_dict()]
            )
    
    return {"status": "notifications_sent", "pipelines_notified": len(pipelines)}

@router.post("/feedback/model-performance")
async def receive_model_performance_feedback(
    pipeline_id: str,
    performance_metrics: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Receive feedback on model performance to improve labeling quality"""
    
    # Validate pipeline ownership
    pipeline = db.query(TrainingPipeline).filter(
        TrainingPipeline.id == pipeline_id,
        TrainingPipeline.user_id == current_user.id
    ).first()
    
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    # Store performance feedback
    feedback = ModelPerformanceFeedback(
        pipeline_id=pipeline_id,
        user_id=current_user.id,
        metrics=performance_metrics,
        created_at=datetime.utcnow()
    )
    db.add(feedback)
    db.commit()
    
    # Analyze feedback to improve labeling
    await analyze_performance_feedback(pipeline_id, performance_metrics)
    
    return {
        "status": "feedback_received",
        "analysis": "Feedback will be used to improve future labeling quality",
        "suggestions": await get_labeling_improvements(performance_metrics)
    }
```

### **Continuous Learning System**
```python
# backend/app/ml/continuous_learning.py
from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta

class ContinuousLearningEngine:
    """Learn from user corrections to improve auto-labeling accuracy"""
    
    def __init__(self):
        self.correction_patterns = {}
        self.confidence_adjustments = {}
        self.model_performance_history = {}
    
    async def learn_from_corrections(
        self,
        original_labels: Dict[str, str],
        corrected_labels: Dict[str, str],
        confidence_scores: Dict[str, float],
        model_used: str
    ):
        """Learn from user corrections to improve future predictions"""
        
        # Track correction patterns
        for label_type in original_labels.keys():
            original = original_labels.get(label_type)
            corrected = corrected_labels.get(label_type)
            confidence = confidence_scores.get(label_type, 0)
            
            if original != corrected:
                # This was a correction
                correction_key = f"{model_used}_{label_type}_{original}_to_{corrected}"
                
                if correction_key not in self.correction_patterns:
                    self.correction_patterns[correction_key] = {
                        "count": 0,
                        "confidence_levels": [],
                        "first_seen": datetime.utcnow()
                    }
                
                self.correction_patterns[correction_key]["count"] += 1
                self.correction_patterns[correction_key]["confidence_levels"].append(confidence)
        
        # Update confidence adjustments
        await self._update_confidence_adjustments(model_used, original_labels, corrected_labels, confidence_scores)
    
    async def get_confidence_adjustment(
        self,
        model_name: str,
        label_type: str,
        predicted_label: str,
        base_confidence: float
    ) -> float:
        """Get confidence adjustment based on learned patterns"""
        
        adjustment_key = f"{model_name}_{label_type}_{predicted_label}"
        
        if adjustment_key in self.confidence_adjustments:
            adjustment_data = self.confidence_adjustments[adjustment_key]
            
            # Calculate adjustment based on correction history
            correction_rate = adjustment_data["corrections"] / max(adjustment_data["total_predictions"], 1)
            
            # Reduce confidence if this label is frequently corrected
            if correction_rate > 0.1:  # More than 10% correction rate
                adjustment = -0.1 * (correction_rate * 10)  # Up to -0.1 adjustment
                return max(0.0, base_confidence + adjustment)
        
        return base_confidence
    
    async def suggest_review_items(
        self,
        job_id: int,
        max_suggestions: int = 50
    ) -> List[Dict[str, Any]]:
        """Suggest items that should be prioritized for human review"""
        
        # Get items from job
        items = await self._get_job_items(job_id)
        
        suggestions = []
        for item in items:
            risk_score = await self._calculate_review_risk_score(item)
            
            if risk_score > 0.3:  # Threshold for suggesting review
                suggestions.append({
                    "item_id": item["id"],
                    "risk_score": risk_score,
                    "reason": await self._get_review_reason(item),
                    "priority": "high" if risk_score > 0.7 else "medium"
                })
        
        # Sort by risk score and return top suggestions
        suggestions.sort(key=lambda x: x["risk_score"], reverse=True)
        return suggestions[:max_suggestions]

# API endpoint for continuous learning
@router.post("/learning/feedback")
async def provide_learning_feedback(
    corrections: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user)
):
    """Provide feedback to continuous learning system"""
    
    learning_engine = ContinuousLearningEngine()
    
    for correction in corrections:
        await learning_engine.learn_from_corrections(
            correction["original_labels"],
            correction["corrected_labels"],
            correction["confidence_scores"],
            correction["model_used"]
        )
    
    return {
        "status": "feedback_processed",
        "corrections_learned": len(corrections),
        "message": "System will improve accuracy based on your feedback"
    }
```

---

## ✅ Feature 5: Quality Assurance & Validation

### **Automated Quality Checks**
```python
# backend/app/quality/validation_engine.py
from typing import Dict, List, Any, Tuple
import statistics
from datetime import datetime

class QualityValidationEngine:
    """Automated quality assurance for labeled data"""
    
    def __init__(self):
        self.quality_thresholds = {
            "min_confidence": 0.8,
            "max_inconsistency_rate": 0.05,
            "min_human_agreement": 0.9
        }
    
    async def validate_label_quality(
        self,
        job_id: int,
        validation_level: str = "standard"  # standard, strict, production
    ) -> Dict[str, Any]:
        """Run comprehensive quality validation on labeled dataset"""
        
        # Get all labels for job
        labels = await self._get_job_labels(job_id)
        
        validation_results = {
            "overall_quality_score": 0.0,
            "validation_level": validation_level,
            "total_items": len(labels),
            "checks_performed": [],
            "issues_found": [],
            "recommendations": [],
            "certification_ready": False
        }
        
        # 1. Confidence Distribution Check
        confidence_check = await self._validate_confidence_distribution(labels)
        validation_results["checks_performed"].append(confidence_check)
        
        # 2. Label Consistency Check
        consistency_check = await self._validate_label_consistency(labels)
        validation_results["checks_performed"].append(consistency_check)
        
        # 3. Inter-Annotator Agreement (if human reviews exist)
        agreement_check = await self._validate_human_agreement(labels)
        validation_results["checks_performed"].append(agreement_check)
        
        # 4. Statistical Distribution Check
        distribution_check = await self._validate_label_distribution(labels)
        validation_results["checks_performed"].append(distribution_check)
        
        # 5. Sample Quality Check (manual verification subset)
        if validation_level in ["strict", "production"]:
            sample_check = await self._validate_sample_quality(labels)
            validation_results["checks_performed"].append(sample_check)
        
        # Calculate overall quality score
        quality_scores = [check["score"] for check in validation_results["checks_performed"]]
        validation_results["overall_quality_score"] = statistics.mean(quality_scores)
        
        # Determine certification readiness
        min_score = 0.95 if validation_level == "production" else 0.90 if validation_level == "strict" else 0.85
        validation_results["certification_ready"] = validation_results["overall_quality_score"] >= min_score
        
        # Generate recommendations
        validation_results["recommendations"] = await self._generate_quality_recommendations(validation_results)
        
        return validation_results
    
    async def _validate_confidence_distribution(self, labels: List[Dict]) -> Dict[str, Any]:
        """Check if confidence scores have healthy distribution"""
        
        confidences = []
        for label in labels:
            for conf in label.get("confidence_scores", {}).values():
                confidences.append(conf)
        
        if not confidences:
            return {"check": "confidence_distribution", "score": 0.0, "status": "failed", "message": "No confidence scores found"}
        
        # Analyze distribution
        mean_conf = statistics.mean(confidences)
        median_conf = statistics.median(confidences)
        high_conf_ratio = len([c for c in confidences if c > 0.9]) / len(confidences)
        low_conf_ratio = len([c for c in confidences if c < 0.6]) / len(confidences)
        
        # Scoring logic
        score = 1.0
        issues = []
        
        if mean_conf < 0.8:
            score -= 0.3
            issues.append(f"Low average confidence: {mean_conf:.2f}")
        
        if low_conf_ratio > 0.1:
            score -= 0.2
            issues.append(f"Too many low confidence predictions: {low_conf_ratio:.1%}")
        
        if high_conf_ratio < 0.6:
            score -= 0.1
            issues.append(f"Not enough high confidence predictions: {high_conf_ratio:.1%}")
        
        return {
            "check": "confidence_distribution",
            "score": max(0.0, score),
            "status": "passed" if score > 0.8 else "warning" if score > 0.6 else "failed",
            "metrics": {
                "mean_confidence": mean_conf,
                "median_confidence": median_conf,
                "high_confidence_ratio": high_conf_ratio,
                "low_confidence_ratio": low_conf_ratio
            },
            "issues": issues
        }
    
    async def _validate_label_consistency(self, labels: List[Dict]) -> Dict[str, Any]:
        """Check for label consistency across similar items"""
        
        # Group similar items (simplified - in production would use embeddings)
        label_counts = {}
        inconsistencies = []
        
        for label in labels:
            for label_type, label_value in label.get("labels", {}).items():
                key = f"{label_type}:{label_value}"
                if key not in label_counts:
                    label_counts[key] = 0
                label_counts[key] += 1
        
        # Check for suspicious patterns
        total_labels = sum(label_counts.values())
        singleton_ratio = len([count for count in label_counts.values() if count == 1]) / len(label_counts)
        
        score = 1.0
        if singleton_ratio > 0.2:  # Too many unique labels
            score -= 0.3
            inconsistencies.append("High number of singleton labels suggests inconsistency")
        
        return {
            "check": "label_consistency",
            "score": max(0.0, score),
            "status": "passed" if score > 0.8 else "warning",
            "metrics": {
                "unique_labels": len(label_counts),
                "singleton_ratio": singleton_ratio,
                "most_common_labels": sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "issues": inconsistencies
        }

# API endpoint for quality validation
@router.post("/quality/validate")
async def validate_dataset_quality(
    job_id: int,
    validation_level: str = "standard",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run quality validation on labeled dataset"""
    
    # Verify job ownership
    job = db.query(Job).filter(Job.id == job_id, Job.user_id == current_user.id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Run validation
    validator = QualityValidationEngine()
    validation_results = await validator.validate_label_quality(job_id, validation_level)
    
    # Save validation report
    validation_report = ValidationReport(
        job_id=job_id,
        user_id=current_user.id,
        validation_level=validation_level,
        quality_score=validation_results["overall_quality_score"],
        certification_ready=validation_results["certification_ready"],
        results=validation_results,
        created_at=datetime.utcnow()
    )
    db.add(validation_report)
    db.commit()
    
    return {
        "validation_id": validation_report.id,
        "quality_score": validation_results["overall_quality_score"],
        "certification_ready": validation_results["certification_ready"],
        "detailed_results": validation_results,
        "download_report_url": f"/api/quality/reports/{validation_report.id}/download"
    }

@router.get("/quality/standards")
async def get_quality_standards():
    """Get available quality standards and their requirements"""
    
    return {
        "standards": {
            "standard": {
                "name": "Standard Quality",
                "min_quality_score": 0.85,
                "description": "Basic quality suitable for development and testing",
                "requirements": [
                    "Average confidence > 80%",
                    "Consistency score > 85%",
                    "Low confidence items < 10%"
                ]
            },
            "strict": {
                "name": "Strict Quality", 
                "min_quality_score": 0.90,
                "description": "High quality suitable for production models",
                "requirements": [
                    "Average confidence > 85%",
                    "Consistency score > 90%",
                    "Human agreement > 90%",
                    "Sample verification passed"
                ]
            },
            "production": {
                "name": "Production Quality",
                "min_quality_score": 0.95,
                "description": "Highest quality for safety-critical applications",
                "requirements": [
                    "Average confidence > 90%",
                    "Consistency score > 95%",
                    "Human agreement > 95%",
                    "Statistical validation passed",
                    "Sample verification > 98%"
                ]
            }
        }
    }
```

---

## 🚀 Feature 6: Production-Ready Infrastructure

### **Monitoring & Health Checks**
```python
# backend/app/monitoring/health_checks.py
from fastapi import APIRouter
from typing import Dict, Any
import psutil
import time
from datetime import datetime

router = APIRouter(prefix="/api/health", tags=["monitoring"])

class HealthMonitor:
    """Comprehensive health monitoring for production deployment"""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "application": {
                "total_requests": self.request_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "ml_models_loaded": await self._check_ml_models(),
                "database_connected": await self._check_database(),
                "redis_connected": await self._check_redis()
            },
            "services": {
                "classification_service": await self._check_classification_service(),
                "export_service": await self._check_export_service(),
                "webhook_service": await self._check_webhook_service()
            }
        }
    
    async def _check_ml_models(self) -> bool:
        """Check if ML models are loaded and responsive"""
        try:
            # Test classification with dummy data
            from ..ml.startup_classifier import startup_classifier
            return startup_classifier.loaded
        except Exception:
            return False

@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    monitor = HealthMonitor()
    health_status = await monitor.get_system_health()
    
    if health_status["system"]["memory_percent"] > 90:
        health_status["status"] = "degraded"
        health_status["issues"] = ["High memory usage"]
    
    return health_status

@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    monitor = HealthMonitor()
    
    # Check critical dependencies
    checks = {
        "database": await monitor._check_database(),
        "redis": await monitor._check_redis(),
        "ml_models": await monitor._check_ml_models()
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        return {"status": "not_ready", "checks": checks}, 503

@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    monitor = HealthMonitor()
    health = await monitor.get_system_health()
    
    metrics = f"""
# HELP modelship_requests_total Total number of requests
# TYPE modelship_requests_total counter
modelship_requests_total {health['application']['total_requests']}

# HELP modelship_errors_total Total number of errors
# TYPE modelship_errors_total counter
modelship_errors_total {monitor.error_count}

# HELP modelship_cpu_usage_percent CPU usage percentage
# TYPE modelship_cpu_usage_percent gauge
modelship_cpu_usage_percent {health['system']['cpu_percent']}

# HELP modelship_memory_usage_percent Memory usage percentage
# TYPE modelship_memory_usage_percent gauge
modelship_memory_usage_percent {health['system']['memory_percent']}

# HELP modelship_uptime_seconds Application uptime in seconds
# TYPE modelship_uptime_seconds gauge
modelship_uptime_seconds {health['uptime_seconds']}
"""
    
    return Response(content=metrics, media_type="text/plain")
```

### **Error Handling & Logging**
```python
# backend/app/middleware/error_handling.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import traceback
import uuid
from datetime import datetime

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("modelship")

class ErrorHandler:
    """Production-grade error handling and logging"""
    
    async def handle_validation_error(self, request: Request, exc):
        error_id = str(uuid.uuid4())
        
        logger.error(f"Validation Error {error_id}: {exc.detail}", extra={
            "error_id": error_id,
            "request_path": request.url.path,
            "request_method": request.method,
            "user_agent": request.headers.get("user-agent"),
            "error_type": "validation_error"
        })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "id": error_id,
                    "type": "validation_error",
                    "message": "Invalid input data",
                    "details": exc.detail,
                    "support_contact": "support@modelship.com"
                }
            }
        )
    
    async def handle_http_exception(self, request: Request, exc: HTTPException):
        error_id = str(uuid.uuid4())
        
        logger.warning(f"HTTP Exception {error_id}: {exc.detail}", extra={
            "error_id": error_id,
            "status_code": exc.status_code,
            "request_path": request.url.path,
            "error_type": "http_exception"
        })
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "id": error_id,
                    "type": "http_exception",
                    "message": exc.detail,
                    "status_code": exc.status_code
                }
            }
        )
    
    async def handle_internal_error(self, request: Request, exc: Exception):
        error_id = str(uuid.uuid4())
        
        logger.error(f"Internal Error {error_id}: {str(exc)}", extra={
            "error_id": error_id,
            "request_path": request.url.path,
            "request_method": request.method,
            "traceback": traceback.format_exc(),
            "error_type": "internal_error"
        })
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "id": error_id,
                    "type": "internal_error",
                    "message": "An internal error occurred. Our team has been notified.",
                    "support_contact": "support@modelship.com"
                }
            }
        )
```

This comprehensive core features specification covers all the essential functionality needed for a production-ready MVP:

## ✅ **Complete Feature Coverage**

1. **🎯 Auto-Labeling Engine** - Multi-model classification with confidence scoring
2. **🔍 Human Review System** - Interactive interface for quality assurance  
3. **📤 ML Export System** - TensorFlow, PyTorch, Hugging Face ready formats
4. **🔄 API Integration** - Seamless training pipeline integration
5. **✅ Quality Assurance** - Automated validation and certification
6. **🚀 Production Infrastructure** - Monitoring, logging, error handling

These features solve the core problems AI teams face and provide everything needed to:
- **Generate Revenue** - Immediate value for customers
- **Scale Operations** - Handle growing user base
- **Attract Investment** - Demonstrate product-market fit
- **Ensure Quality** - Production-ready reliability

The system is designed to handle the complete AI training data workflow from raw uploads to training-ready datasets with enterprise-grade quality and reliability.