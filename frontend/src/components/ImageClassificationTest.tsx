import React, { useState } from 'react';
import apiClient from '../services/api';

interface ClassificationResult {
  result_id?: number;
  filename: string;
  predicted_label: string;
  confidence: number;
  processing_time?: number;
  status: string;
  error_message?: string;
}

interface JobResult {
  job_id: number;
  job_status: string;
  results: ClassificationResult[];
  summary: {
    total_results: number;
    successful_results: number;
    failed_results: number;
    average_confidence: number;
    average_processing_time: number;
  };
}

const ImageClassificationTest: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [singleResults, setSingleResults] = useState<any>(null);
  const [batchResults, setBatchResults] = useState<JobResult | null>(null);
  const [availableModels, setAvailableModels] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('single');

  // Test single image classification
  const testSingleClassification = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError('Please select an image file first');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFiles[0]);

      const response = await apiClient.post('/api/classify/image', formData, { headers: { 'Content-Type': 'multipart/form-data' } });

      setSingleResults(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  // Test batch image classification
  const testBatchClassification = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError('Please select image files first');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      Array.from(selectedFiles).forEach(file => {
        formData.append('files', file);
      });

      const response = await apiClient.post('/api/classify/image/batch', formData, { headers: { 'Content-Type': 'multipart/form-data' } });

      const data = response.data;
      
      // If we get a job_id, fetch the results
      if (data.job_id) {
        // Wait a moment then fetch results
        setTimeout(async () => {
          try {
            const resultsResponse = await apiClient.get(`/api/classify/results/${data.job_id}`);
            setBatchResults(resultsResponse.data);
          } catch (err) {
            console.error('Failed to fetch batch results:', err);
          }
        }, 1000);
      } else {
        setBatchResults(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch classification failed');
    } finally {
      setLoading(false);
    }
  };

  // Get available models
  const getAvailableModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get('/api/classify/models');
      setAvailableModels(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get models');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFiles(e.target.files);
    setSingleResults(null);
    setBatchResults(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">Image Classification Testing</h2>
        <p className="text-gray-600 mt-2">
          Test single and batch image classification endpoints
        </p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {['single', 'batch', 'models'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab === 'single' ? 'Single Image' : tab === 'batch' ? 'Batch Images' : 'Available Models'}
            </button>
          ))}
        </nav>
      </div>

      {/* File Upload */}
      {(activeTab === 'single' || activeTab === 'batch') && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Upload Images</h3>
          <input
            type="file"
            accept="image/*"
            multiple={activeTab === 'batch'}
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          {selectedFiles && (
            <p className="mt-2 text-sm text-gray-600">
              Selected: {selectedFiles.length} file(s)
            </p>
          )}
        </div>
      )}

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Actions</h3>
        <div className="flex gap-3">
          {activeTab === 'single' && (
            <button
              onClick={testSingleClassification}
              disabled={loading || !selectedFiles}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
            >
              Classify Single Image
            </button>
          )}
          {activeTab === 'batch' && (
            <button
              onClick={testBatchClassification}
              disabled={loading || !selectedFiles}
              className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
            >
              Classify Batch
            </button>
          )}
          {activeTab === 'models' && (
            <button
              onClick={getAvailableModels}
              disabled={loading}
              className="bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white px-4 py-2 rounded-md"
            >
              Get Available Models
            </button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-gray-600">Processing...</span>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="text-red-800 font-semibold">Error</h4>
          <p className="text-red-600 mt-1">{error}</p>
        </div>
      )}

      {/* Single Classification Results */}
      {activeTab === 'single' && singleResults && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-green-800 font-semibold mb-3">Single Classification Result</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-700">Filename:</span>
              <span className="ml-2">{singleResults.filename}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Predicted Label:</span>
              <span className="ml-2 font-semibold text-green-600">{singleResults.predicted_label}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Confidence:</span>
              <span className="ml-2">{singleResults.confidence}%</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Processing Time:</span>
              <span className="ml-2">{singleResults.processing_time}s</span>
            </div>
          </div>
        </div>
      )}

      {/* Batch Classification Results */}
      {activeTab === 'batch' && batchResults && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-blue-800 font-semibold mb-3">Batch Classification Results</h4>
          
          {/* Summary */}
          <div className="mb-4 p-3 bg-white rounded border">
            <h5 className="font-medium mb-2">Summary</h5>
            <div className="grid grid-cols-4 gap-4 text-sm">
              <div>Total: {batchResults.summary.total_results}</div>
              <div>Successful: {batchResults.summary.successful_results}</div>
              <div>Failed: {batchResults.summary.failed_results}</div>
              <div>Avg Confidence: {batchResults.summary.average_confidence}%</div>
            </div>
          </div>

          {/* Individual Results */}
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {batchResults.results.map((result, index) => (
              <div key={index} className="p-3 bg-white rounded border text-sm">
                <div className="grid grid-cols-4 gap-2">
                  <div className="truncate">
                    <span className="font-medium">File:</span> {result.filename}
                  </div>
                  <div>
                    <span className="font-medium">Label:</span> {result.predicted_label}
                  </div>
                  <div>
                    <span className="font-medium">Confidence:</span> {result.confidence}%
                  </div>
                  <div>
                    <span className={`font-medium ${result.status === 'success' ? 'text-green-600' : 'text-red-600'}`}>
                      {result.status}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Available Models */}
      {activeTab === 'models' && availableModels && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h4 className="text-purple-800 font-semibold mb-3">Available Models</h4>
          <div className="space-y-3">
            {Object.entries(availableModels).map(([key, value]) => (
              <div key={key} className="p-3 bg-white rounded border">
                <h5 className="font-medium text-gray-800">{key.replace(/_/g, ' ').toUpperCase()}</h5>
                <pre className="text-xs text-gray-600 mt-1 whitespace-pre-wrap">
                  {JSON.stringify(value, null, 2)}
                </pre>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Test Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Test Instructions</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Single Classification:</strong> Upload one image to test /api/classify/image</p>
          <p><strong>Batch Classification:</strong> Upload multiple images to test /api/classify/image/batch</p>
          <p><strong>Supported Formats:</strong> JPG, PNG, GIF, WebP</p>
          <p><strong>Expected Response:</strong> JSON with filename, predicted_label, confidence, processing_time</p>
        </div>
      </div>
    </div>
  );
};

export default ImageClassificationTest; 