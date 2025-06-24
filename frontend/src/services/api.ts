import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests (auth disabled for development)
apiClient.interceptors.request.use((config) => {
  // Auth temporarily disabled for development
  return config;
});

// Handle auth errors (auth disabled for development)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Auth temporarily disabled - don't redirect on 401
    return Promise.reject(error);
  }
);

// Auth API (disabled for development)
export const authAPI = {
  login: async (email: string, password: string) => {
    // Auth disabled for development
    return { token: 'dev-token', user: { email } };
  },

  register: async (userData: { email: string; password: string; full_name: string }) => {
    // Auth disabled for development
    return { message: 'Registration disabled in development mode' };
  },

  getCurrentUser: async () => {
    // Auth disabled for development
    return { id: 1, email: 'dev@example.com', name: 'Development User' };
  },

  logout: () => {
    // Auth disabled for development
  }
};

// ✅ NEW: Simple Projects API (file-based storage)
export const simpleProjectsAPI = {
  getProjects: async () => {
    const response = await apiClient.get('/api/simple-projects/');
    return response.data;
  },

  createProject: async (projectData: {
    name: string;
    description?: string;
    project_type: string;
    confidence_threshold?: number;
    auto_approve_threshold?: number;
    guidelines?: string;
  }) => {
    const response = await apiClient.post('/api/simple-projects/', projectData);
    return response.data;
  },

  createTestProject: async () => {
    const response = await apiClient.post('/api/simple-projects/test-create');
    return response.data;
  },

  getProject: async (projectId: number) => {
    const response = await apiClient.get(`/api/simple-projects/${projectId}`);
    return response.data;
  },

  updateProject: async (projectId: number, updates: any) => {
    const response = await apiClient.put(`/api/simple-projects/${projectId}`, updates);
    return response.data;
  },

  deleteProject: async (projectId: number) => {
    const response = await apiClient.delete(`/api/simple-projects/${projectId}`);
    return response.data;
  },

  getSupportedTypes: async () => {
    const response = await apiClient.get('/api/simple-projects/types/supported');
    return response.data;
  },

  getStorageStats: async () => {
    const response = await apiClient.get('/api/simple-projects/stats/storage');
    return response.data;
  }
};

// Projects API (legacy - updated to use simple projects)
export const projectsAPI = {
  getProjects: async (filters?: { status?: string; type?: string }) => {
    // Use simple projects API
    return await simpleProjectsAPI.getProjects();
  },

  createProject: async (projectData: {
    name: string;
    description?: string;
    project_type: string;
    confidence_threshold?: number;
  }) => {
    // Use simple projects API
    return await simpleProjectsAPI.createProject(projectData);
  },

  getProject: async (projectId: number) => {
    // Use simple projects API
    return await simpleProjectsAPI.getProject(projectId);
  },

  updateProject: async (projectId: number, updates: any) => {
    // Use simple projects API
    return await simpleProjectsAPI.updateProject(projectId, updates);
  },

  deleteProject: async (projectId: number) => {
    // Use simple projects API
    return await simpleProjectsAPI.deleteProject(projectId);
  }
};

// Files API
export const filesAPI = {
  uploadFiles: async (files: File[], projectId?: number, onProgress?: (progress: number) => void) => {
    const formData = new FormData();
    
    files.forEach((file) => {
      formData.append('files', file);
    });
    
    if (projectId) {
      formData.append('project_id', projectId.toString());
    }
    
    const response = await apiClient.post('/api/upload/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(percentCompleted);
        }
      },
    });
    return response.data;
  },

  getFiles: async (projectId?: number) => {
    const params = projectId ? `?project_id=${projectId}` : '';
    const response = await apiClient.get(`/api/files/${params}`);
    return response.data;
  }
};

// Classification API
export const classificationAPI = {
  classifyImages: async (files: File[], options?: {
    model?: string;
    confidence_threshold?: number;
    include_metadata?: boolean;
  }) => {
    const formData = new FormData();
    
    files.forEach(file => {
      formData.append('files', file);
    });
    
    if (options?.model) formData.append('model', options.model);
    if (options?.confidence_threshold) formData.append('confidence_threshold', options.confidence_threshold.toString());
    if (options?.include_metadata) formData.append('include_metadata', options.include_metadata.toString());
    
    const response = await apiClient.post('/api/classify/image/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  classifyText: async (texts: string[], options?: {
    classification_type?: string;
    confidence_threshold?: number;
    include_metadata?: boolean;
  }) => {
    const response = await apiClient.post('/api/classify/text/batch', {
      texts,
      ...options
    });
    return response.data;
  },

  getClassificationJobs: async (projectId?: number) => {
    const params = projectId ? `?project_id=${projectId}` : '';
    const response = await apiClient.get(`/api/jobs/${params}`);
    return response.data;
  }
};

// Review API
export const reviewAPI = {
  getReviewQueue: async (projectId?: number, filters?: {
    confidence_threshold?: number;
    reviewed?: boolean;
    limit?: number;
  }) => {
    const params = new URLSearchParams();
    if (projectId) params.append('project_id', projectId.toString());
    if (filters?.confidence_threshold) params.append('confidence_threshold', filters.confidence_threshold.toString());
    if (filters?.reviewed !== undefined) params.append('reviewed', filters.reviewed.toString());
    if (filters?.limit) params.append('limit', filters.limit.toString());
    
    const response = await apiClient.get(`/api/review/queue?${params}`);
    return response.data;
  },

  submitReview: async (resultId: number, reviewData: {
    action: 'approve' | 'correct' | 'reject';
    corrected_label?: string;
    reason?: string;
  }) => {
    const response = await apiClient.post(`/api/review/${resultId}`, reviewData);
    return response.data;
  },

  bulkReview: async (resultIds: number[], action: 'approve' | 'reject') => {
    const response = await apiClient.post('/api/review/bulk', {
      result_ids: resultIds,
      action
    });
    return response.data;
  }
};

// Export API
export const exportAPI = {
  exportResults: async (jobId: number, format: string = 'json') => {
    const response = await apiClient.post(`/api/export/create/${jobId}`, {
      export_format: format,
      include_confidence: true
    });
    return response.data;
  },

  downloadExport: async (filename: string) => {
    const response = await apiClient.get(`/api/export/download/${filename}`, {
      responseType: 'blob'
    });
    return response.data;
  }
};

export default apiClient; 