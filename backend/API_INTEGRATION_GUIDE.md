# ModelShip API Integration Guide

## 🚀 **Complete API Reference for Developers**

**Version:** 1.0  
**Base URL:** `https://api.modelship.ai` (Production) | `http://localhost:8000` (Development)  
**Authentication:** JWT Bearer Token  
**Content-Type:** `application/json`

---

## 📋 **Table of Contents**

1. [Quick Start Guide](#quick-start-guide)
2. [Authentication & Security](#authentication--security)
3. [Frontend Developer Integration](#frontend-developer-integration)
4. [External ML Platform Integration](#external-ml-platform-integration)
5. [API Endpoints Reference](#api-endpoints-reference)
6. [Real-time Data Access](#real-time-data-access)
7. [Webhooks & Callbacks](#webhooks--callbacks)
8. [Error Handling](#error-handling)
9. [Rate Limits & Best Practices](#rate-limits--best-practices)
10. [SDK Examples](#sdk-examples)

---

## 🚀 **Quick Start Guide**

### **For Frontend Developers**
```javascript
// 1. Install ModelShip SDK
npm install modelship-sdk

// 2. Initialize client
import ModelShip from 'modelship-sdk';
const client = new ModelShip({
  apiKey: 'your-api-key',
  baseURL: 'https://api.modelship.ai'
});

// 3. Authenticate user
const user = await client.auth.login({
  email: 'user@example.com',
  password: 'password'
});

// 4. Create project and classify data
const project = await client.projects.create({
  name: 'Product Classification',
  type: 'image_classification'
});

const results = await client.classify.images([
  'image1.jpg', 'image2.jpg'
], { projectId: project.id });
```

### **For ML Platform Integration**
```python
# 1. Install ModelShip Python SDK
pip install modelship-python

# 2. Initialize client
from modelship import ModelShipClient

client = ModelShipClient(
    api_key="your-api-key",
    base_url="https://api.modelship.ai"
)

# 3. Stream labeled data directly to TensorFlow
import tensorflow as tf

dataset = client.datasets.stream(
    project_id="project_123",
    format="tensorflow",
    batch_size=32
)

# 4. Train model with live data
model = tf.keras.Sequential([...])
model.fit(dataset, epochs=10)
```

---

## 🔐 **Authentication & Security**

### **API Key Authentication**
```bash
# Get API key from dashboard
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.modelship.ai/api/auth/me
```

### **JWT Token Authentication**
```javascript
// Login to get JWT token
const response = await fetch('/api/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password'
  })
});

const { access_token } = await response.json();

// Use token for subsequent requests
const headers = {
  'Authorization': `Bearer ${access_token}`,
  'Content-Type': 'application/json'
};
```

### **API Key Management**
```bash
# Create API key
POST /api/auth/api-keys
{
  "name": "TensorFlow Integration",
  "permissions": ["read", "write"],
  "expires_at": "2025-12-31T23:59:59Z"
}

# List API keys
GET /api/auth/api-keys

# Revoke API key
DELETE /api/auth/api-keys/{key_id}
```

---

## 🎨 **Frontend Developer Integration**

### **React Integration Example**
```jsx
import React, { useState, useEffect } from 'react';
import { ModelShipProvider, useModelShip } from 'modelship-react';

function App() {
  return (
    <ModelShipProvider apiKey="your-api-key">
      <Dashboard />
    </ModelShipProvider>
  );
}

function Dashboard() {
  const { projects, createProject, classifyImages } = useModelShip();
  const [selectedProject, setSelectedProject] = useState(null);

  const handleImageUpload = async (files) => {
    const results = await classifyImages(files, {
      projectId: selectedProject.id,
      confidence_threshold: 0.8
    });
    
    console.log('Classification results:', results);
  };

  return (
    <div>
      <ProjectSelector 
        projects={projects}
        onSelect={setSelectedProject}
      />
      <ImageUploader onUpload={handleImageUpload} />
      <ResultsViewer projectId={selectedProject?.id} />
    </div>
  );
}
```

### **Vue.js Integration Example**
```vue
<template>
  <div id="app">
    <project-manager 
      :projects="projects"
      @project-selected="onProjectSelected"
    />
    <classification-interface 
      :project-id="selectedProjectId"
      @results-updated="onResultsUpdated"
    />
  </div>
</template>

<script>
import { ModelShipAPI } from 'modelship-vue';

export default {
  name: 'App',
  data() {
    return {
      projects: [],
      selectedProjectId: null,
      api: new ModelShipAPI({ apiKey: process.env.VUE_APP_MODELSHIP_KEY })
    };
  },
  async mounted() {
    this.projects = await this.api.projects.list();
  },
  methods: {
    onProjectSelected(projectId) {
      this.selectedProjectId = projectId;
    },
    async onResultsUpdated() {
      // Refresh project analytics
      const analytics = await this.api.analytics.getProject(this.selectedProjectId);
      this.$emit('analytics-updated', analytics);
    }
  }
};
</script>
```

### **Angular Integration Example**
```typescript
import { Injectable } from '@angular/core';
import { ModelShipService } from 'modelship-angular';

@Injectable({
  providedIn: 'root'
})
export class DataLabelingService {
  constructor(private modelShip: ModelShipService) {}

  async createLabelingProject(config: ProjectConfig) {
    const project = await this.modelShip.projects.create(config);
    
    // Set up real-time updates
    this.modelShip.realtime.subscribe(`project:${project.id}`, (update) => {
      this.handleProjectUpdate(update);
    });
    
    return project;
  }

  async processDataBatch(files: File[], projectId: string) {
    const job = await this.modelShip.classify.batch(files, {
      projectId,
      callback_url: `${window.location.origin}/api/webhooks/classification`
    });
    
    return job;
  }
}
```

---

## 🤖 **External ML Platform Integration**

### **TensorFlow Integration**
```python
import tensorflow as tf
from modelship import ModelShipClient, TensorFlowAdapter

# Initialize ModelShip client
client = ModelShipClient(api_key="your-api-key")

# Create TensorFlow dataset from ModelShip project
adapter = TensorFlowAdapter(client)
dataset = adapter.create_dataset(
    project_id="project_123",
    batch_size=32,
    image_size=(224, 224),
    num_classes=10
)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with live data from ModelShip
model.fit(
    dataset,
    epochs=10,
    validation_data=adapter.create_validation_dataset(project_id="project_123")
)

# Push trained model back to ModelShip
client.models.upload(model, project_id="project_123")
```

### **PyTorch Integration**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modelship import ModelShipClient, PyTorchDataset

# Initialize client
client = ModelShipClient(api_key="your-api-key")

# Create PyTorch dataset
dataset = PyTorchDataset(
    client=client,
    project_id="project_123",
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

model = ImageClassifier(num_classes=len(dataset.classes))

# Training loop with live data
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        # Training code here
        pass
    
    # Push metrics back to ModelShip
    client.analytics.log_training_metrics(
        project_id="project_123",
        epoch=epoch,
        metrics={"accuracy": accuracy, "loss": loss}
    )
```

### **Hugging Face Integration**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from modelship import ModelShipClient, HuggingFaceDataset

# Initialize client
client = ModelShipClient(api_key="your-api-key")

# Create Hugging Face dataset
dataset = HuggingFaceDataset(
    client=client,
    project_id="text_project_456",
    tokenizer_name="bert-base-uncased"
)

# Load pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(dataset.label_names)
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Training with Trainer API
trainer = Trainer(
    model=model,
    train_dataset=dataset.train_split,
    eval_dataset=dataset.val_split,
    tokenizer=tokenizer
)

trainer.train()

# Upload fine-tuned model to ModelShip
client.models.upload_huggingface(
    model=model,
    tokenizer=tokenizer,
    project_id="text_project_456"
)
```

### **MLflow Integration**
```python
import mlflow
from modelship import ModelShipClient

# Initialize clients
client = ModelShipClient(api_key="your-api-key")
mlflow.set_tracking_uri("https://your-mlflow-server.com")

# Track experiments with ModelShip data
with mlflow.start_run():
    # Log dataset info from ModelShip
    dataset_info = client.projects.get_dataset_info("project_123")
    mlflow.log_params({
        "dataset_size": dataset_info["total_samples"],
        "num_classes": dataset_info["num_classes"],
        "modelship_project_id": "project_123"
    })
    
    # Train model (your training code here)
    model = train_model(client.datasets.get_training_data("project_123"))
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": model_accuracy,
        "f1_score": f1_score
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Register model back to ModelShip
    client.models.register_mlflow_model(
        mlflow_run_id=mlflow.active_run().info.run_id,
        project_id="project_123"
    )
```

---

## 📡 **Real-time Data Access**

### **Streaming API for Live Training**
```python
# Stream new labeled data as it becomes available
async def stream_labeled_data(project_id: str):
    async with client.datasets.stream(project_id) as stream:
        async for batch in stream:
            # Process new labeled data immediately
            yield batch
            
            # Update model incrementally
            model.partial_fit(batch.features, batch.labels)

# Usage
async for data_batch in stream_labeled_data("project_123"):
    print(f"Received {len(data_batch)} new labeled samples")
    # Your incremental training logic here
```

### **WebSocket Integration**
```javascript
// Real-time project updates
const ws = new WebSocket('wss://api.modelship.ai/ws/projects/123');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  
  switch(update.type) {
    case 'new_labels':
      // New data labeled - fetch and retrain
      fetchLatestData(update.project_id);
      break;
    case 'quality_alert':
      // Quality threshold breached - review needed
      showQualityAlert(update.details);
      break;
    case 'job_completed':
      // Classification job finished
      updateProjectProgress(update.progress);
      break;
  }
};
```

### **Polling API for Batch Updates**
```python
import time
from modelship import ModelShipClient

def poll_for_updates(project_id: str, interval: int = 60):
    """Poll for new labeled data every minute"""
    client = ModelShipClient(api_key="your-api-key")
    last_update = None
    
    while True:
        # Check for new data since last update
        new_data = client.datasets.get_updates(
            project_id=project_id,
            since=last_update
        )
        
        if new_data:
            print(f"Found {len(new_data)} new labeled samples")
            
            # Process new data
            for sample in new_data:
                yield sample
            
            last_update = new_data[-1]['created_at']
        
        time.sleep(interval)

# Usage
for new_sample in poll_for_updates("project_123"):
    # Add to training dataset
    training_data.append(new_sample)
    
    # Trigger retraining if enough new data
    if len(training_data) % 100 == 0:
        retrain_model(training_data)
```

---

## 🔗 **Webhooks & Callbacks**

### **Setting Up Webhooks**
```bash
# Register webhook endpoint
POST /api/webhooks/register
{
  "url": "https://your-ml-platform.com/webhooks/modelship",
  "events": ["data.labeled", "project.completed", "quality.alert"],
  "secret": "your-webhook-secret"
}
```

### **Webhook Event Handling**
```python
from flask import Flask, request, jsonify
import hmac
import hashlib

app = Flask(__name__)

@app.route('/webhooks/modelship', methods=['POST'])
def handle_modelship_webhook():
    # Verify webhook signature
    signature = request.headers.get('X-ModelShip-Signature')
    payload = request.get_data()
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    if not hmac.compare_digest(signature, expected_signature):
        return jsonify({'error': 'Invalid signature'}), 401
    
    event = request.json
    
    if event['type'] == 'data.labeled':
        # New data available - trigger training pipeline
        trigger_training_pipeline(event['project_id'])
        
    elif event['type'] == 'project.completed':
        # Project finished - run final model evaluation
        run_final_evaluation(event['project_id'])
        
    elif event['type'] == 'quality.alert':
        # Quality issue detected - pause training
        pause_training_pipeline(event['project_id'])
    
    return jsonify({'status': 'received'}), 200
```

---

## 📊 **API Endpoints Reference**

### **Authentication Endpoints**
```bash
POST   /api/auth/login              # User login
POST   /api/auth/register           # User registration
POST   /api/auth/refresh            # Token refresh
GET    /api/auth/me                 # Current user info
POST   /api/auth/api-keys           # Create API key
GET    /api/auth/api-keys           # List API keys
DELETE /api/auth/api-keys/{id}      # Revoke API key
```

### **Project Management**
```bash
POST   /api/projects/              # Create project
GET    /api/projects/              # List projects
GET    /api/projects/{id}          # Get project details
PUT    /api/projects/{id}          # Update project
DELETE /api/projects/{id}          # Delete project
GET    /api/projects/{id}/dataset  # Get project dataset
GET    /api/projects/{id}/progress # Get project progress
```

### **Data Classification**
```bash
POST   /api/classify/image         # Single image classification
POST   /api/classify/image/batch   # Batch image classification
POST   /api/classify/text          # Single text classification
POST   /api/classify/text/batch    # Batch text classification
GET    /api/classify/models        # Available models
POST   /api/classify/custom        # Custom model inference
```

### **Dataset Access**
```bash
GET    /api/datasets/{project_id}           # Get dataset
GET    /api/datasets/{project_id}/stream    # Stream dataset
GET    /api/datasets/{project_id}/download  # Download dataset
GET    /api/datasets/{project_id}/stats     # Dataset statistics
POST   /api/datasets/{project_id}/export   # Export in specific format
```

### **Real-time Integration**
```bash
GET    /api/realtime/projects/{id}/updates # Get real-time updates
POST   /api/webhooks/register              # Register webhook
GET    /api/webhooks/                      # List webhooks
DELETE /api/webhooks/{id}                  # Delete webhook
```

---

## ⚡ **Rate Limits & Best Practices**

### **Rate Limits**
```
Authentication:     100 requests/minute
Classification:     1000 requests/minute  
Dataset Access:     500 requests/minute
Webhooks:          10 requests/second
Streaming:         No limit (connection-based)
```

### **Best Practices**

**For Frontend Developers:**
```javascript
// Use connection pooling
const client = new ModelShip({
  apiKey: 'your-key',
  maxConcurrentRequests: 10,
  retryConfig: {
    retries: 3,
    retryDelay: 1000
  }
});

// Implement caching
const cache = new Map();
const getCachedProject = async (id) => {
  if (cache.has(id)) return cache.get(id);
  
  const project = await client.projects.get(id);
  cache.set(id, project);
  return project;
};

// Batch operations when possible
const results = await client.classify.batch(images, {
  batchSize: 50  // Process 50 images at once
});
```

**For ML Platform Integration:**
```python
# Use connection pooling
from modelship import ModelShipClient
from concurrent.futures import ThreadPoolExecutor

client = ModelShipClient(
    api_key="your-key",
    max_workers=5,  # Concurrent connections
    timeout=30      # Request timeout
)

# Implement exponential backoff
import time
import random

def with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff with jitter
            delay = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

# Use async operations for better performance
import asyncio

async def process_dataset_async(project_id):
    tasks = []
    async with client.datasets.stream_async(project_id) as stream:
        async for batch in stream:
            task = asyncio.create_task(process_batch(batch))
            tasks.append(task)
    
    await asyncio.gather(*tasks)
```

---

## 🛠️ **SDK Examples**

### **Python SDK**
```python
# Installation
pip install modelship-python

# Full example
from modelship import ModelShipClient
import tensorflow as tf

# Initialize client
client = ModelShipClient(
    api_key="your-api-key",
    base_url="https://api.modelship.ai"
)

# Create project
project = client.projects.create({
    "name": "Product Classification",
    "type": "image_classification",
    "confidence_threshold": 0.8
})

# Upload and classify data
job = client.classify.upload_and_process(
    files=["product1.jpg", "product2.jpg"],
    project_id=project.id
)

# Wait for completion
job.wait_for_completion()

# Get results and create TensorFlow dataset
dataset = client.datasets.to_tensorflow(
    project_id=project.id,
    batch_size=32,
    image_size=(224, 224)
)

# Train model
model = tf.keras.applications.ResNet50(weights=None, classes=project.num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(dataset, epochs=10)

# Register trained model
client.models.register(model, project_id=project.id)
```

### **JavaScript SDK**
```javascript
// Installation
npm install modelship-sdk

// Full example
import ModelShip from 'modelship-sdk';

const client = new ModelShip({
  apiKey: 'your-api-key',
  baseURL: 'https://api.modelship.ai'
});

// Create project with schema
const project = await client.projects.create({
  name: 'Sentiment Analysis',
  type: 'text_classification',
  labelSchema: {
    categories: ['positive', 'negative', 'neutral'],
    isMultiLabel: false
  }
});

// Process text data
const texts = [
  'I love this product!',
  'This is terrible.',
  'It\'s okay, nothing special.'
];

const job = await client.classify.text(texts, {
  projectId: project.id,
  model: 'sentiment-bert'
});

// Monitor progress
job.onProgress((progress) => {
  console.log(`Progress: ${progress.completed}/${progress.total}`);
});

await job.completion();

// Export results
const exportUrl = await client.export.create(job.id, {
  format: 'json',
  includeConfidence: true
});

console.log(`Results available at: ${exportUrl}`);
```

### **cURL Examples**
```bash
# Authentication
curl -X POST "https://api.modelship.ai/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Create project
curl -X POST "https://api.modelship.ai/api/projects/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Image Classification Project",
    "project_type": "image_classification",
    "confidence_threshold": 0.8
  }'

# Upload and classify images
curl -X POST "https://api.modelship.ai/api/classify/image/batch" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "project_id=123"

# Get real-time dataset access
curl -X GET "https://api.modelship.ai/api/datasets/123/stream" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Accept: application/x-ndjson"

# Export in TensorFlow format
curl -X POST "https://api.modelship.ai/api/export/formats/tensorflow/456" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"include_reviewed_only": true}'
```

---

## 🚨 **Error Handling**

### **Common Error Responses**
```json
{
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Invalid API key or token",
    "details": {
      "suggestion": "Check your API key or refresh your token"
    }
  }
}

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED", 
    "message": "Too many requests",
    "details": {
      "limit": 1000,
      "window": "1 minute",
      "retry_after": 45
    }
  }
}

{
  "error": {
    "code": "INSUFFICIENT_CREDITS",
    "message": "Not enough credits for this operation",
    "details": {
      "required": 100,
      "available": 25,
      "upgrade_url": "https://modelship.ai/upgrade"
    }
  }
}
```

### **Error Handling Best Practices**
```python
from modelship import ModelShipClient, ModelShipError

client = ModelShipClient(api_key="your-key")

try:
    result = client.classify.image("image.jpg")
except ModelShipError as e:
    if e.code == "RATE_LIMIT_EXCEEDED":
        # Wait and retry
        time.sleep(e.retry_after)
        result = client.classify.image("image.jpg")
    elif e.code == "INSUFFICIENT_CREDITS":
        # Handle billing
        print(f"Upgrade needed: {e.upgrade_url}")
    else:
        # Log error and continue
        logger.error(f"Classification failed: {e.message}")
```

---

## 🎯 **Integration Examples by Use Case**

### **Continuous Learning Pipeline**
```python
# Set up continuous learning with ModelShip
class ContinuousLearningPipeline:
    def __init__(self, project_id, model):
        self.client = ModelShipClient(api_key="your-key")
        self.project_id = project_id
        self.model = model
        
    async def run(self):
        # Stream new labeled data
        async with self.client.datasets.stream(self.project_id) as stream:
            batch = []
            async for sample in stream:
                batch.append(sample)
                
                # Retrain when we have enough new data
                if len(batch) >= 100:
                    await self.retrain(batch)
                    batch = []
    
    async def retrain(self, new_data):
        # Incremental training
        self.model.partial_fit(
            [s.features for s in new_data],
            [s.label for s in new_data]
        )
        
        # Push updated model back
        await self.client.models.update(
            self.model,
            project_id=self.project_id
        )
```

### **Quality Monitoring Dashboard**
```javascript
// Real-time quality monitoring
class QualityMonitor {
  constructor(projectId) {
    this.client = new ModelShip({ apiKey: 'your-key' });
    this.projectId = projectId;
  }
  
  async startMonitoring() {
    // Set up WebSocket for real-time updates
    const ws = this.client.realtime.connect(this.projectId);
    
    ws.on('quality_metrics', (metrics) => {
      this.updateDashboard(metrics);
      
      // Alert if quality drops
      if (metrics.accuracy < 0.8) {
        this.triggerQualityAlert(metrics);
      }
    });
    
    // Poll for detailed analytics
    setInterval(async () => {
      const analytics = await this.client.analytics.getQualityMetrics(this.projectId);
      this.updateDetailedMetrics(analytics);
    }, 30000); // Every 30 seconds
  }
  
  updateDashboard(metrics) {
    document.getElementById('accuracy').textContent = `${metrics.accuracy * 100}%`;
    document.getElementById('throughput').textContent = `${metrics.throughput}/hour`;
    document.getElementById('review-rate').textContent = `${metrics.review_rate * 100}%`;
  }
}
```

This comprehensive API documentation enables both frontend developers and external ML platforms to seamlessly integrate with ModelShip, creating a powerful ecosystem for automated data labeling and continuous learning! 🚀 