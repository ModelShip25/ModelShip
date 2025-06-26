import React, { useState } from 'react';
import apiClient from '../services/api';

const ExportSystemTest: React.FC = () => {
  const [jobId, setJobId] = useState('');
  const [exportFormat, setExportFormat] = useState('json');
  const [includeReviewedOnly, setIncludeReviewedOnly] = useState(false);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.0);
  const [exportResults, setExportResults] = useState<any>(null);
  const [availableFormats, setAvailableFormats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('basic');

  const createBasicExport = async () => {
    if (!jobId.trim()) {
      setError('Please enter a job ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post(`/api/export/create/${jobId}`);
      setExportResults(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed');
    } finally {
      setLoading(false);
    }
  };

  const createCocoExport = async () => {
    if (!jobId.trim()) {
      setError('Please enter a job ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post(`/api/export/formats/coco/${jobId}`);
      setExportResults(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'COCO export failed');
    } finally {
      setLoading(false);
    }
  };

  const createYoloExport = async () => {
    if (!jobId.trim()) {
      setError('Please enter a job ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post(`/api/export/formats/yolo/${jobId}`);
      setExportResults(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'YOLO export failed');
    } finally {
      setLoading(false);
    }
  };

  const getAvailableFormats = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get('/api/export/formats');
      setAvailableFormats(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get formats');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">Export System Testing</h2>
        <p className="text-gray-600 mt-2">Test various export formats and options</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {['basic', 'coco', 'yolo', 'formats'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab === 'formats' ? 'Available Formats' : `${tab.toUpperCase()} Export`}
            </button>
          ))}
        </nav>
      </div>

      {/* Configuration */}
      {activeTab !== 'formats' && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Export Configuration</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Job ID</label>
              <input
                type="text"
                value={jobId}
                onChange={(e) => setJobId(e.target.value)}
                placeholder="Enter job ID to export"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>

            {activeTab === 'basic' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Export Format</label>
                <select
                  value={exportFormat}
                  onChange={(e) => setExportFormat(e.target.value)}
                  className="w-full p-2 border border-gray-300 rounded-md"
                >
                  <option value="json">JSON</option>
                  <option value="csv">CSV</option>
                  <option value="xml">XML</option>
                </select>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confidence Threshold</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                checked={includeReviewedOnly}
                onChange={(e) => setIncludeReviewedOnly(e.target.checked)}
                className="mr-2"
              />
              <label className="text-sm font-medium text-gray-700">Include reviewed items only</label>
            </div>
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Actions</h3>
        <div className="flex gap-3">
          {activeTab === 'basic' && (
            <button
              onClick={createBasicExport}
              disabled={loading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
            >
              Create Basic Export
            </button>
          )}
          {activeTab === 'coco' && (
            <button
              onClick={createCocoExport}
              disabled={loading}
              className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
            >
              Create COCO Export
            </button>
          )}
          {activeTab === 'yolo' && (
            <button
              onClick={createYoloExport}
              disabled={loading}
              className="bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white px-4 py-2 rounded-md"
            >
              Create YOLO Export
            </button>
          )}
          {activeTab === 'formats' && (
            <button
              onClick={getAvailableFormats}
              disabled={loading}
              className="bg-indigo-500 hover:bg-indigo-600 disabled:bg-indigo-300 text-white px-4 py-2 rounded-md"
            >
              Get Available Formats
            </button>
          )}
        </div>
      </div>

      {/* Loading/Error */}
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

      {/* Results Display */}
      {(exportResults || availableFormats) && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-green-800 font-semibold mb-3">
            {activeTab === 'formats' ? 'Available Export Formats' : 'Export Results'}
          </h4>
          <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-auto max-h-96">
            {JSON.stringify(exportResults || availableFormats, null, 2)}
          </pre>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Export System Info</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Basic Export:</strong> POST /api/export/create/{`{job_id}`}</p>
          <p><strong>COCO Export:</strong> POST /api/export/formats/coco/{`{job_id}`}</p>
          <p><strong>YOLO Export:</strong> POST /api/export/formats/yolo/{`{job_id}`}</p>
          <p><strong>Available Formats:</strong> GET /api/export/formats</p>
          <p><strong>Download:</strong> Use download_url from export response</p>
        </div>
      </div>
    </div>
  );
};

export default ExportSystemTest; 