import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
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

// Handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: async (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);
    
    const response = await apiClient.post('/api/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
    });
    return response.data;
  },

  register: async (userData: { email: string; password: string; full_name: string }) => {
    const response = await apiClient.post('/api/auth/register', userData);
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await apiClient.get('/api/auth/me');
    return response.data;
  },

  logout: () => {
    localStorage.removeItem('authToken');
  }
};

// Projects API
export const projectsAPI = {
  getProjects: async (filters?: { status?: string; type?: string }) => {
    const params = new URLSearchParams();
    if (filters?.status) params.append('status', filters.status);
    if (filters?.type) params.append('project_type', filters.type);
    
    const response = await apiClient.get(`/api/projects/?${params}`);
    return response.data;
  },

  createProject: async (projectData: {
    name: string;
    description?: string;
    project_type: string;
    confidence_threshold?: number;
  }) => {
    const response = await apiClient.post('/api/projects/', projectData);
    return response.data;
  },

  getProject: async (projectId: number) => {
    const response = await apiClient.get(`/api/projects/${projectId}`);
    return response.data;
  },

  updateProject: async (projectId: number, updates: any) => {
    const response = await apiClient.put(`/api/projects/${projectId}`, updates);
    return response.data;
  },

  deleteProject: async (projectId: number) => {
    const response = await apiClient.delete(`/api/projects/${projectId}`);
    return response.data;
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

// Analytics API
export const analyticsAPI = {
  getUserAnalytics: async (timeframe?: string) => {
    const params = timeframe ? `?timeframe=${timeframe}` : '';
    const response = await apiClient.get(`/api/analytics/user${params}`);
    return response.data;
  },

  getProjectAnalytics: async (projectId: number, timeframe?: string) => {
    const params = timeframe ? `?timeframe=${timeframe}` : '';
    const response = await apiClient.get(`/api/analytics/project/${projectId}${params}`);
    return response.data;
  },

  getPlatformOverview: async () => {
    const response = await apiClient.get('/api/analytics/platform-overview');
    return response.data;
  }
};

// Export API
export const exportAPI = {
  exportProject: async (projectId: number, format: string, options?: any) => {
    const response = await apiClient.post(`/api/export/project/${projectId}`, {
      format,
      ...options
    });
    return response.data;
  },

  downloadExport: async (exportId: string) => {
    const response = await apiClient.get(`/api/export/download/${exportId}`, {
      responseType: 'blob'
    });
    return response.data;
  }
};

export default apiClient; 