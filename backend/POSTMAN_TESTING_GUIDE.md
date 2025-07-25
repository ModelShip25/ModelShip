# ModelShip API Testing Guide - Advanced Features

This guide covers testing all the advanced features of the ModelShip API including the new ML-ready export formats, advanced classification endpoints, and enterprise features.

## 🚀 Quick Start - New Features

### 1. Quick Image Classification (No Auth Required)
**Endpoint:** `POST /api/classify/image/quick`
- **Purpose:** Frictionless image classification without authentication
- **Use Case:** Testing, demos, public access

```
POST http://localhost:8000/api/classify/image/quick
Content-Type: multipart/form-data

file: [Upload your image file]
```

**Expected Response:**
```json
{
  "predicted_label": "Egyptian cat",
  "confidence": 85.42,
  "processing_time": 0.523,
  "classification_id": "uuid-here",
  "status": "success",
  "note": "Quick classification - register for advanced features"
}
```

### 2. Enhanced Image Classification (With Metadata)
**Endpoint:** `POST /api/classify/image`
- **Purpose:** Full-featured classification with detailed metadata
- **Headers:** `Authorization: Bearer YOUR_TOKEN`

```
POST http://localhost:8000/api/classify/image
Authorization: Bearer YOUR_TOKEN
Content-Type: multipart/form-data

file: [Upload your image file]
```

**Expected Response:**
```json
{
  "predicted_label": "Egyptian cat",
  "confidence": 85.42,
  "processing_time": 0.523,
  "classification_id": "uuid-here",
  "model_used": "resnet50",
  "credits_remaining": 99,
  "metadata": {
    "timestamp": "2024-01-15T10:30:00Z",
    "confidence_calibrated": true,
    "preprocessing_steps": ["resize", "normalize", "tensor_conversion"],
    "device_used": "cpu"
  },
  "quality_metrics": {
    "prediction_entropy": 2.345,
    "confidence_calibration_score": 0.92,
    "prediction_stability": "high"
  }
}
```

### 3. Model Information
**Endpoint:** `GET /api/classify/models`
- **Purpose:** Get available models and performance statistics

```
GET http://localhost:8000/api/classify/models
```

**Expected Response:**
```json
{
  "available_models": {
    "resnet50": {
      "name": "ResNet-50",
      "type": "image_classification",
      "tier": "startup",
      "categories": 1000,
      "accuracy": 0.76,
      "avg_processing_time": 0.5,
      "supported_formats": ["jpg", "jpeg", "png", "gif", "webp", "bmp"],
      "description": "General-purpose image classification with 1000 ImageNet categories",
      "status": "available"
    }
  },
  "performance_stats": {
    "total_classifications": 150,
    "average_processing_time": 0.523,
    "model_usage": {
      "resnet50": 150
    },
    "error_rate": 2.0,
    "success_rate": 98.0
  },
  "service_status": "operational"
}
```

## 🔬 Advanced Export Features

### 1. TensorFlow Dataset Export
**Endpoint:** `POST /api/export/ml/{job_id}?format_type=tensorflow`
- **Purpose:** Create TensorFlow-ready dataset with train/val/test splits
- **Headers:** `Authorization: Bearer YOUR_TOKEN`

```
POST http://localhost:8000/api/export/ml/1?format_type=tensorflow&train_split=0.7&val_split=0.2&test_split=0.1
Authorization: Bearer YOUR_TOKEN
```

**Expected Response:**
```json
{
  "message": "Tensorflow dataset created successfully",
  "download_url": "/api/export/ml/download/tensorflow_dataset_job_1_20240115_103000.zip",
  "filename": "tensorflow_dataset_job_1_20240115_103000.zip",
  "format": "tensorflow",
  "job_id": 1,
  "total_samples": 45,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 2. PyTorch Dataset Export
**Endpoint:** `POST /api/export/ml/{job_id}?format_type=pytorch`

```
POST http://localhost:8000/api/export/ml/1?format_type=pytorch
Authorization: Bearer YOUR_TOKEN
```

**Expected Response:**
```json
{
  "message": "Pytorch dataset created successfully",
  "download_url": "/api/export/ml/download/pytorch_dataset_job_1_20240115_103000.zip",
  "filename": "pytorch_dataset_job_1_20240115_103000.zip",
  "format": "pytorch",
  "job_id": 1,
  "total_samples": 45,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### 3. COCO Format Export
**Endpoint:** `POST /api/export/ml/{job_id}?format_type=coco`

```
POST http://localhost:8000/api/export/ml/1?format_type=coco
Authorization: Bearer YOUR_TOKEN
```

### 4. Available ML Export Formats
**Endpoint:** `GET /api/export/ml/formats`

```
GET http://localhost:8000/api/export/ml/formats
```

**Expected Response:**
```json
{
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
```

## 📊 Enhanced Job Management

### 1. Job Results with Filtering
**Endpoint:** `GET /api/classify/results/{job_id}`
- **Parameters:** 
  - `include_errors`: Include failed results (default: false)
  - `confidence_threshold`: Minimum confidence filter (default: 0.0)

```
GET http://localhost:8000/api/classify/results/1?include_errors=false&confidence_threshold=0.8
Authorization: Bearer YOUR_TOKEN
```

**Expected Response:**
```json
{
  "job_id": 1,
  "job_status": "completed",
  "results": [
    {
      "result_id": 1,
      "filename": "cat.jpg",
      "predicted_label": "Egyptian cat",
      "confidence": 85.42,
      "processing_time": 0.523,
      "status": "success",
      "error_message": null,
      "reviewed": false,
      "ground_truth": null
    }
  ],
  "summary": {
    "total_results": 1,
    "successful_results": 1,
    "failed_results": 0,
    "average_confidence": 85.42,
    "average_processing_time": 0.523
  },
  "filters_applied": {
    "include_errors": false,
    "confidence_threshold": 0.8
  }
}
```

### 2. User Jobs with Pagination
**Endpoint:** `GET /api/classify/jobs`
- **Parameters:**
  - `limit`: Number of jobs to return (default: 20, max: 100)
  - `offset`: Number of jobs to skip (default: 0)

```
GET http://localhost:8000/api/classify/jobs?limit=10&offset=0
Authorization: Bearer YOUR_TOKEN
```

## 🧪 Complete Testing Workflow

### Step 1: Register and Login
```
POST http://localhost:8000/api/auth/register
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "testpassword"
}
```

```
POST http://localhost:8000/api/auth/login
Content-Type: application/json

{
  "email": "test@example.com",
  "password": "testpassword"
}
```

### Step 2: Test Quick Classification (No Auth)
```
POST http://localhost:8000/api/classify/image/quick
Content-Type: multipart/form-data

file: [Upload test image]
```

### Step 3: Test Authenticated Classification
```
POST http://localhost:8000/api/classify/image
Authorization: Bearer YOUR_TOKEN
Content-Type: multipart/form-data

file: [Upload test image]
```

### Step 4: Create Batch Job
```
POST http://localhost:8000/api/classify/batch
Authorization: Bearer YOUR_TOKEN
Content-Type: multipart/form-data

files: [Upload multiple images]
job_type: image
```

### Step 5: Monitor Job Progress
```
GET http://localhost:8000/api/classify/jobs/{job_id}
Authorization: Bearer YOUR_TOKEN
```

### Step 6: Get Results
```
GET http://localhost:8000/api/classify/results/{job_id}
Authorization: Bearer YOUR_TOKEN
```

### Step 7: Export ML-Ready Dataset
```
POST http://localhost:8000/api/export/ml/{job_id}?format_type=tensorflow
Authorization: Bearer YOUR_TOKEN
```

### Step 8: Download Export
```
GET http://localhost:8000/api/export/ml/download/{filename}
Authorization: Bearer YOUR_TOKEN
```

## 🎯 Key Features to Test

### ✅ Advanced Classification Features
- [x] Confidence calibration
- [x] Processing time tracking
- [x] Detailed metadata
- [x] Quality metrics
- [x] Performance statistics
- [x] Model information

### ✅ ML-Ready Export Formats
- [x] TensorFlow datasets with train/val/test splits
- [x] PyTorch datasets with custom transforms
- [x] COCO format for object detection
- [x] Automated dataset scripts
- [x] Comprehensive metadata

### ✅ Enterprise Features
- [x] Batch processing with progress tracking
- [x] Advanced filtering and pagination
- [x] Detailed job statistics
- [x] Error handling and recovery
- [x] Performance monitoring

### ✅ User Experience
- [x] Frictionless quick classification
- [x] Enhanced authenticated experience
- [x] Comprehensive API documentation
- [x] Clear error messages
- [x] Progress tracking

## 🔧 Troubleshooting

### Common Issues:
1. **Model Loading Errors**: Check if models are downloading correctly
2. **File Upload Issues**: Ensure file types are supported (jpg, png, gif, webp)
3. **Authentication Errors**: Verify token is valid and not expired
4. **Export Failures**: Check job status is "completed" before exporting

### Performance Tips:
1. Use batch processing for multiple images
2. Set appropriate confidence thresholds
3. Monitor processing times and adjust batch sizes
4. Use quick classification for testing and demos

This comprehensive testing guide covers all the advanced features of ModelShip. The API is now ready for production use with enterprise-grade capabilities!
