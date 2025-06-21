# ModelShip Frontend API Guide

Complete frontend developer reference for integrating with ModelShip platform.

## Quick Setup

### Environment Configuration
```javascript
// .env.local
REACT_APP_API_BASE_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
REACT_APP_UPLOAD_MAX_SIZE=10485760
```

### API Client Setup
```javascript
// src/services/api.js
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('authToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default apiClient;
```

## Authentication

### User Login
```javascript
// POST /api/auth/login
const loginUser = async (credentials) => {
  try {
    const response = await apiClient.post('/api/auth/login', {
      email: credentials.email,
      password: credentials.password
    });
    
    localStorage.setItem('authToken', response.data.access_token);
    
    return {
      success: true,
      user: response.data.user,
      token: response.data.access_token
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Login failed'
    };
  }
};
```

### User Registration
```javascript
// POST /api/auth/register
const registerUser = async (userData) => {
  try {
    const response = await apiClient.post('/api/auth/register', {
      email: userData.email,
      password: userData.password,
      full_name: userData.fullName
    });
    
    return {
      success: true,
      user: response.data.user,
      token: response.data.access_token
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Registration failed'
    };
  }
};
```

## Project Management

### Create Project
```javascript
// POST /api/projects/
const createProject = async (projectData) => {
  try {
    const response = await apiClient.post('/api/projects/', {
      name: projectData.name,
      description: projectData.description,
      project_type: projectData.type,
      confidence_threshold: projectData.confidenceThreshold || 0.8
    });
    
    return {
      success: true,
      project: response.data
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Failed to create project'
    };
  }
};
```

### List Projects
```javascript
// GET /api/projects/
const getProjects = async (filters = {}) => {
  try {
    const params = new URLSearchParams();
    if (filters.status) params.append('status', filters.status);
    if (filters.type) params.append('project_type', filters.type);
    
    const response = await apiClient.get(`/api/projects/?${params}`);
    
    return {
      success: true,
      projects: response.data.projects,
      total: response.data.total
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Failed to fetch projects'
    };
  }
};
```

## File Upload

### Upload Files
```javascript
// POST /api/upload/
const uploadFiles = async (files, projectId, onProgress) => {
  try {
    const formData = new FormData();
    
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    if (projectId) {
      formData.append('project_id', projectId);
    }
    
    const response = await apiClient.post('/api/upload/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        if (onProgress) onProgress(percentCompleted);
      },
    });
    
    return {
      success: true,
      files: response.data.files,
      jobId: response.data.job_id
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Upload failed'
    };
  }
};
```

## Classification

### Classify Images
```javascript
// POST /api/classify/image/batch
const classifyImages = async (files, options = {}) => {
  try {
    const formData = new FormData();
    
    files.forEach(file => {
      formData.append('files', file);
    });
    
    if (options.projectId) {
      formData.append('project_id', options.projectId);
    }
    
    const response = await apiClient.post('/api/classify/image/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return {
      success: true,
      jobId: response.data.job_id,
      results: response.data.results
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Classification failed'
    };
  }
};
```

### Classify Text
```javascript
// POST /api/classify/text/batch
const classifyText = async (texts, options = {}) => {
  try {
    const response = await apiClient.post('/api/classify/text/batch', {
      texts: texts,
      project_id: options.projectId,
      confidence_threshold: options.confidenceThreshold || 0.5,
      model_type: options.modelType || 'sentiment'
    });
    
    return {
      success: true,
      jobId: response.data.job_id,
      results: response.data.results
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Text classification failed'
    };
  }
};
```

## Results & Review

### Get Results
```javascript
// GET /api/results/{job_id}
const getResults = async (jobId, filters = {}) => {
  try {
    const params = new URLSearchParams();
    if (filters.reviewed !== undefined) params.append('reviewed', filters.reviewed);
    if (filters.confidence_min) params.append('confidence_min', filters.confidence_min);
    
    const response = await apiClient.get(`/api/results/${jobId}?${params}`);
    
    return {
      success: true,
      results: response.data.results,
      total: response.data.total
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Failed to fetch results'
    };
  }
};
```

### Update Review
```javascript
// PUT /api/review/result/{result_id}
const updateResultReview = async (resultId, reviewData) => {
  try {
    const response = await apiClient.put(`/api/review/result/${resultId}`, {
      reviewed: reviewData.reviewed,
      ground_truth: reviewData.groundTruth,
      confidence_score: reviewData.confidenceScore
    });
    
    return {
      success: true,
      result: response.data
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Failed to update review'
    };
  }
};
```

## Analytics

### Get Project Analytics
```javascript
// GET /api/analytics/project/{project_id}
const getProjectAnalytics = async (projectId, timeRange = '7d') => {
  try {
    const response = await apiClient.get(
      `/api/analytics/project/${projectId}?time_range=${timeRange}`
    );
    
    return {
      success: true,
      analytics: response.data
    };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.detail || 'Failed to fetch analytics'
    };
  }
};
```

## Error Handling

### Global Error Handler
```javascript
export const handleApiError = (error) => {
  if (error.response) {
    const { status, data } = error.response;
    
    switch (status) {
      case 401:
        return 'Authentication required. Please log in.';
      case 403:
        return 'Access denied.';
      case 404:
        return 'Resource not found.';
      case 422:
        return data.detail || 'Invalid input data.';
      case 429:
        return 'Too many requests. Please try again later.';
      case 500:
        return 'Server error. Please try again later.';
      default:
        return data.detail || `Error: ${status}`;
    }
  } else if (error.request) {
    return 'Network error. Please check your connection.';
  } else {
    return error.message || 'An unexpected error occurred.';
  }
};
```

## React Component Examples

### Login Component
```jsx
import React, { useState } from 'react';
import { loginUser } from '../services/api';

const LoginForm = ({ onLoginSuccess }) => {
  const [credentials, setCredentials] = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const result = await loginUser(credentials);
    
    if (result.success) {
      onLoginSuccess(result.user, result.token);
    } else {
      setError(result.error);
    }
    
    setLoading(false);
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={credentials.email}
        onChange={(e) => setCredentials({...credentials, email: e.target.value})}
        placeholder="Email"
        required
      />
      <input
        type="password"
        value={credentials.password}
        onChange={(e) => setCredentials({...credentials, password: e.target.value})}
        placeholder="Password"
        required
      />
      {error && <div className="error">{error}</div>}
      <button type="submit" disabled={loading}>
        {loading ? 'Signing in...' : 'Sign In'}
      </button>
    </form>
  );
};

export default LoginForm;
```

### File Upload Component
```jsx
import React, { useState } from 'react';
import { uploadFiles } from '../services/api';

const FileUploader = ({ projectId, onUploadComplete }) => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleFileChange = async (e) => {
    const files = Array.from(e.target.files);
    setUploading(true);

    const result = await uploadFiles(
      files, 
      projectId, 
      (progressPercent) => setProgress(progressPercent)
    );

    if (result.success) {
      onUploadComplete(result.files, result.jobId);
    }

    setUploading(false);
  };

  return (
    <div>
      <input
        type="file"
        multiple
        onChange={handleFileChange}
        disabled={uploading}
        accept="image/*,text/*"
      />
      {uploading && (
        <div>
          <div>Uploading... {progress}%</div>
          <progress value={progress} max="100" />
        </div>
      )}
    </div>
  );
};

export default FileUploader;
```

This guide provides the essential API integration patterns for building a complete ModelShip frontend! 🚀 