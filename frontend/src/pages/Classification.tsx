import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { 
  PlayIcon, 
  PauseIcon,
  CheckIcon,
  XMarkIcon,
  PhotoIcon,
  DocumentTextIcon,
  ChartBarIcon,
  ArrowPathIcon,
  EyeIcon,
  PencilIcon
} from '@heroicons/react/24/outline';
import apiClient from '../services/api';

interface FileItem {
  id: number;
  filename: string;
  file_path: string;
  file_type: string;
  classification_result?: {
    predicted_label: string;
    confidence: number;
    reviewed: boolean;
    ground_truth?: string;
  };
}

interface Project {
  id: number;
  name: string;
  description: string;
}

const Classification: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const projectId = searchParams.get('project');
  
  const [project, setProject] = useState<Project | null>(null);
  const [files, setFiles] = useState<FileItem[]>([]);
  const [currentFileIndex, setCurrentFileIndex] = useState(0);
  const [classifying, setClassifying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoClassify, setAutoClassify] = useState(false);
  
  const [editingLabel, setEditingLabel] = useState(false);
  const [customLabel, setCustomLabel] = useState('');

  useEffect(() => {
    if (projectId) {
      fetchProjectData();
    }
  }, [projectId]);

  const fetchProjectData = async () => {
    if (!projectId) return;
    
    try {
      setLoading(true);
      
      // Fetch project details
      const projectResponse = await apiClient.get(`/api/projects/${projectId}`);
      setProject(projectResponse.data);

      // Fetch project files
      const filesResponse = await apiClient.get(`/api/projects/${projectId}/files`);
      setFiles(filesResponse.data.files || []);

      setError(null);
    } catch (err) {
      setError('Failed to load project data');
      console.error('Project fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const classifyCurrentFile = async () => {
    if (!files[currentFileIndex]) return;
    
    const file = files[currentFileIndex];
    setClassifying(true);

    try {
      const isImage = file.file_type.startsWith('image/');
      const endpoint = isImage ? '/api/classify/image' : '/api/classify/text';
      
      const formData = new FormData();
      // For actual file classification, we'd need to send the file
      // For now, simulate with file path
      formData.append('file_id', file.id.toString());
      formData.append('project_id', projectId!);

      const response = await apiClient.post(endpoint, formData, { headers: { 'Content-Type': 'multipart/form-data' } });
      const result = response.data;
      
      // Update the current file with classification result
      setFiles(prev => prev.map((f, index) => 
        index === currentFileIndex 
          ? { 
              ...f, 
              classification_result: {
                predicted_label: result.predicted_label || result.label,
                confidence: result.confidence || 0.85,
                reviewed: false
              }
            } 
          : f
      ));

      // Auto-advance if auto-classify is enabled
      if (autoClassify && currentFileIndex < files.length - 1) {
        setTimeout(() => {
          setCurrentFileIndex(prev => prev + 1);
        }, 1000);
      }
    } catch (err) {
      setError('Classification failed');
      console.error('Classification error:', err);
    } finally {
      setClassifying(false);
    }
  };

  const classifyAllFiles = async () => {
    setAutoClassify(true);
    setCurrentFileIndex(0);
    
    for (let i = 0; i < files.length; i++) {
      setCurrentFileIndex(i);
      await new Promise(resolve => setTimeout(resolve, 500)); // Small delay
      await classifyCurrentFile();
    }
    
    setAutoClassify(false);
  };

  const reviewResult = (approved: boolean) => {
    if (!files[currentFileIndex]?.classification_result) return;

    setFiles(prev => prev.map((f, index) => 
      index === currentFileIndex && f.classification_result
        ? { 
            ...f, 
            classification_result: {
              ...f.classification_result,
              reviewed: true,
              ground_truth: approved ? f.classification_result.predicted_label : customLabel || 'incorrect'
            }
          } 
        : f
    ));

    // Auto-advance to next unreviewed file
    const nextIndex = files.findIndex((f, index) => 
      index > currentFileIndex && (!f.classification_result?.reviewed)
    );
    
    if (nextIndex !== -1) {
      setCurrentFileIndex(nextIndex);
    }

    setEditingLabel(false);
    setCustomLabel('');
  };

  const saveCustomLabel = () => {
    if (!customLabel.trim()) return;
    
    setFiles(prev => prev.map((f, index) => 
      index === currentFileIndex
        ? { 
            ...f, 
            classification_result: {
              predicted_label: f.classification_result?.predicted_label || customLabel,
              confidence: f.classification_result?.confidence || 1.0,
              reviewed: true,
              ground_truth: customLabel
            }
          } 
        : f
    ));

    setEditingLabel(false);
    setCustomLabel('');
  };

  const getProgressStats = () => {
    const classified = files.filter(f => f.classification_result).length;
    const reviewed = files.filter(f => f.classification_result?.reviewed).length;
    const accuracy = reviewed > 0 ? 
      files.filter(f => f.classification_result?.reviewed && 
        f.classification_result.predicted_label === f.classification_result.ground_truth).length / reviewed 
      : 0;
    
    return { classified, reviewed, accuracy: Math.round(accuracy * 100) };
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading project...</p>
        </div>
      </div>
    );
  }

  if (!project || files.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">No files to classify</h1>
          <p className="text-gray-600 mb-6">This project doesn't have any uploaded files.</p>
          <button
            onClick={() => navigate('/upload')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
          >
            Upload Files
          </button>
        </div>
      </div>
    );
  }

  const currentFile = files[currentFileIndex];
  const stats = getProgressStats();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">{project.name}</h1>
              <p className="text-gray-600">Classification & Review</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                File {currentFileIndex + 1} of {files.length}
              </div>
              <button
                onClick={() => navigate(`/results?project=${projectId}`)}
                className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium"
              >
                View Results
              </button>
            </div>
          </div>
          
          {/* Progress Stats */}
          <div className="mt-4 grid grid-cols-4 gap-4">
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="text-sm text-blue-600">Total Files</div>
              <div className="text-lg font-bold text-blue-900">{files.length}</div>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <div className="text-sm text-green-600">Classified</div>
              <div className="text-lg font-bold text-green-900">{stats.classified}</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-3">
              <div className="text-sm text-purple-600">Reviewed</div>
              <div className="text-lg font-bold text-purple-900">{stats.reviewed}</div>
            </div>
            <div className="bg-yellow-50 rounded-lg p-3">
              <div className="text-sm text-yellow-600">Accuracy</div>
              <div className="text-lg font-bold text-yellow-900">{stats.accuracy}%</div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Classification Area */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow">
              {/* File Display */}
              <div className="p-6 border-b">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    {currentFile.filename}
                  </h3>
                  <div className="flex items-center space-x-2">
                    {currentFile.file_type.startsWith('image/') ? (
                      <PhotoIcon className="w-5 h-5 text-purple-500" />
                    ) : (
                      <DocumentTextIcon className="w-5 h-5 text-green-500" />
                    )}
                    <span className="text-sm text-gray-500">{currentFile.file_type}</span>
                  </div>
                </div>

                {/* File Preview */}
                <div className="bg-gray-50 rounded-lg p-8 text-center min-h-64 flex items-center justify-center">
                  {currentFile.file_type.startsWith('image/') ? (
                    <img
                      src={`http://localhost:8000${currentFile.file_path}`}
                      alt={currentFile.filename}
                      className="max-w-full max-h-64 object-contain"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                        (e.target as HTMLImageElement).nextElementSibling!.classList.remove('hidden');
                      }}
                    />
                  ) : (
                    <div className="text-center">
                      <DocumentTextIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-gray-600">Text file preview not available</p>
                      <p className="text-sm text-gray-500 mt-2">{currentFile.filename}</p>
                    </div>
                  )}
                  <div className="hidden">
                    <PhotoIcon className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600">Image preview not available</p>
                  </div>
                </div>
              </div>

              {/* Classification Results */}
              <div className="p-6">
                {currentFile.classification_result ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h4 className="text-lg font-semibold text-gray-900">Classification Result</h4>
                      {currentFile.classification_result.reviewed && (
                        <span className="bg-green-100 text-green-800 text-sm px-2 py-1 rounded-full">
                          Reviewed
                        </span>
                      )}
                    </div>
                    
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-gray-600">Predicted Label:</span>
                        <span className="font-medium text-gray-900">
                          {currentFile.classification_result.predicted_label}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-600">Confidence:</span>
                        <span className="font-medium text-gray-900">
                          {Math.round(currentFile.classification_result.confidence * 100)}%
                        </span>
                      </div>
                    </div>

                    {/* Review Actions */}
                    {!currentFile.classification_result.reviewed && (
                      <div className="space-y-3">
                        <p className="text-sm text-gray-600">Is this classification correct?</p>
                        <div className="flex space-x-3">
                          <button
                            onClick={() => reviewResult(true)}
                            className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium"
                          >
                            <CheckIcon className="w-4 h-4 mr-2" />
                            Correct
                          </button>
                          <button
                            onClick={() => setEditingLabel(true)}
                            className="flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium"
                          >
                            <XMarkIcon className="w-4 h-4 mr-2" />
                            Incorrect
                          </button>
                        </div>

                        {editingLabel && (
                          <div className="space-y-3 p-4 bg-yellow-50 rounded-lg">
                            <label className="block text-sm font-medium text-gray-700">
                              Enter correct label:
                            </label>
                            <input
                              type="text"
                              value={customLabel}
                              onChange={(e) => setCustomLabel(e.target.value)}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                              placeholder="e.g., cat, product, positive"
                            />
                            <div className="flex space-x-2">
                              <button
                                onClick={saveCustomLabel}
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium"
                              >
                                Save
                              </button>
                              <button
                                onClick={() => {
                                  setEditingLabel(false);
                                  setCustomLabel('');
                                }}
                                className="px-4 py-2 border border-gray-300 text-gray-700 hover:bg-gray-50 rounded-lg text-sm font-medium"
                              >
                                Cancel
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <ChartBarIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-600 mb-4">No classification result yet</p>
                    <button
                      onClick={classifyCurrentFile}
                      disabled={classifying}
                      className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium flex items-center mx-auto"
                    >
                      {classifying ? (
                        <>
                          <ArrowPathIcon className="w-5 h-5 mr-2 animate-spin" />
                          Classifying...
                        </>
                      ) : (
                        <>
                          <PlayIcon className="w-5 h-5 mr-2" />
                          Classify This File
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Side Panel */}
          <div className="space-y-6">
            {/* Navigation */}
            <div className="bg-white rounded-lg shadow p-6">
              <h4 className="font-semibold text-gray-900 mb-4">Navigation</h4>
              <div className="space-y-3">
                <button
                  onClick={() => setCurrentFileIndex(Math.max(0, currentFileIndex - 1))}
                  disabled={currentFileIndex === 0}
                  className="w-full px-4 py-2 border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400 rounded-lg font-medium"
                >
                  Previous File
                </button>
                <button
                  onClick={() => setCurrentFileIndex(Math.min(files.length - 1, currentFileIndex + 1))}
                  disabled={currentFileIndex === files.length - 1}
                  className="w-full px-4 py-2 border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:bg-gray-100 disabled:text-gray-400 rounded-lg font-medium"
                >
                  Next File
                </button>
              </div>
            </div>

            {/* Batch Actions */}
            <div className="bg-white rounded-lg shadow p-6">
              <h4 className="font-semibold text-gray-900 mb-4">Batch Actions</h4>
              <div className="space-y-3">
                <button
                  onClick={classifyAllFiles}
                  disabled={classifying || autoClassify}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium"
                >
                  {autoClassify ? 'Classifying All...' : 'Classify All Files'}
                </button>
                <button
                  onClick={() => navigate(`/results?project=${projectId}`)}
                  className="w-full bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium"
                >
                  Export Results
                </button>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-red-700 text-sm">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="mt-2 text-red-600 hover:text-red-500 text-sm font-medium"
                >
                  Dismiss
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Classification; 