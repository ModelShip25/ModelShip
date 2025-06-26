import React, { useState } from 'react';
import apiClient from '../services/api';

const ObjectDetectionTest: React.FC = () => {
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('single');

  const testSingleDetection = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError('Please select an image file first');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', selectedFiles[0]);

      const response = await apiClient.post('/api/classify/image/detect', formData, { headers: { 'Content-Type': 'multipart/form-data' } });

      if (response.status !== 200) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = response.data;
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Object detection failed');
    } finally {
      setLoading(false);
    }
  };

  const testBatchDetection = async () => {
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

      const response = await apiClient.post('/api/classify/batch/detect', formData, { headers: { 'Content-Type': 'multipart/form-data' } });

      if (response.status !== 200) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = response.data;
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Batch detection failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSelectedFiles(e.target.files);
    setResults(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">Object Detection Testing</h2>
        <p className="text-gray-600 mt-2">Test YOLO/SAHI object detection endpoints</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {['single', 'batch'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab === 'single' ? 'Single Detection' : 'Batch Detection'}
            </button>
          ))}
        </nav>
      </div>

      {/* File Upload */}
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

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Actions</h3>
        <div className="flex gap-3">
          {activeTab === 'single' && (
            <button
              onClick={testSingleDetection}
              disabled={loading || !selectedFiles}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
            >
              Detect Objects (Single)
            </button>
          )}
          {activeTab === 'batch' && (
            <button
              onClick={testBatchDetection}
              disabled={loading || !selectedFiles}
              className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
            >
              Detect Objects (Batch)
            </button>
          )}
        </div>
      </div>

      {/* Loading/Error/Results */}
      {loading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-gray-600">Processing...</span>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="text-red-800 font-semibold">Error</h4>
          <p className="text-red-600 mt-1">{error}</p>
        </div>
      )}

      {results && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-green-800 font-semibold mb-3">Detection Results</h4>
          <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-auto max-h-96">
            {JSON.stringify(results, null, 2)}
          </pre>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Object Detection Info</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Single Detection:</strong> /api/classify/image/detect</p>
          <p><strong>Batch Detection:</strong> /api/classify/batch/detect</p>
          <p><strong>Expected Output:</strong> Bounding boxes, class names, confidence scores</p>
          <p><strong>Models:</strong> YOLO with SAHI enhancement for large images</p>
        </div>
      </div>
    </div>
  );
};

export default ObjectDetectionTest; 