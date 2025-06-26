import React, { useState, useEffect } from 'react';
import apiClient from '../services/api';

interface HealthData {
  status: string;
  service?: string;
  version?: string;
  timestamp?: string;
  message?: string;
  features?: any;
}

const HealthCheck: React.FC = () => {
  const [healthData, setHealthData] = useState<HealthData | null>(null);
  const [rootData, setRootData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async (endpoint: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get(endpoint);
      if (response.status < 200 || response.status >= 300) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = response.data;
      if (endpoint === '/health') {
        setHealthData(data);
      } else {
        setRootData(data);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Auto-check health on component mount
    checkHealth('/health');
    checkHealth('/');
  }, []);

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">System Health Check</h2>
        <p className="text-gray-600 mt-2">
          Test backend connectivity and system status
        </p>
      </div>

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Endpoints</h3>
        <div className="flex gap-3">
          <button
            onClick={() => checkHealth('/health')}
            disabled={loading}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
          >
            Test /health
          </button>
          <button
            onClick={() => checkHealth('/')}
            disabled={loading}
            className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
          >
            Test / (root)
          </button>
          <button
            onClick={() => checkHealth('/api/health')}
            disabled={loading}
            className="bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white px-4 py-2 rounded-md"
          >
            Test /api/health
          </button>
        </div>
      </div>

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-gray-600">Testing connection...</span>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="text-red-800 font-semibold">Connection Error</h4>
          <p className="text-red-600 mt-1">{error}</p>
          <div className="mt-3 text-sm text-red-700">
            <p>Common issues:</p>
            <ul className="list-disc list-inside mt-1 space-y-1">
              <li>Backend server not running (try: <code>cd backend && python main.py</code>)</li>
              <li>CORS issues (check if backend allows localhost:3000)</li>
              <li>Port mismatch (backend should be on port 8000)</li>
            </ul>
          </div>
        </div>
      )}

      {/* Health Data Display */}
      {healthData && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-green-800 font-semibold flex items-center">
            <span className="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
            Health Check (/health)
          </h4>
          <div className="mt-3 space-y-2">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Status:</span>
                <span className="ml-2 text-green-600">{healthData.status}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Service:</span>
                <span className="ml-2">{healthData.service || 'N/A'}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Version:</span>
                <span className="ml-2">{healthData.version || 'N/A'}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Timestamp:</span>
                <span className="ml-2">{healthData.timestamp || 'N/A'}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Root Data Display */}
      {rootData && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-blue-800 font-semibold flex items-center">
            <span className="w-3 h-3 bg-blue-500 rounded-full mr-2"></span>
            API Info (/)
          </h4>
          <div className="mt-3 space-y-3">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-700">Version:</span>
                <span className="ml-2">{rootData.version}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Status:</span>
                <span className="ml-2 text-blue-600">{rootData.status}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Docs:</span>
                <span className="ml-2">{rootData.docs}</span>
              </div>
              <div>
                <span className="font-medium text-gray-700">Message:</span>
                <span className="ml-2">{rootData.message}</span>
              </div>
            </div>

            {/* Features */}
            {rootData.features && (
              <div className="mt-4">
                <h5 className="font-medium text-gray-700 mb-2">Available Features:</h5>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {Object.entries(rootData.features).map(([key, value]) => (
                    <div key={key} className="flex items-center">
                      <span className={`w-2 h-2 rounded-full mr-2 ${
                        value ? 'bg-green-500' : 'bg-red-500'
                      }`}></span>
                      <span className="truncate">{key.replace(/_/g, ' ')}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Phase 1 Workflows */}
            {rootData.phase_1_workflows && (
              <div className="mt-4">
                <h5 className="font-medium text-gray-700 mb-2">Phase 1 Workflows:</h5>
                <div className="space-y-1 text-xs">
                  {Object.entries(rootData.phase_1_workflows).map(([key, value]) => (
                    <div key={key} className="flex">
                      <span className="font-medium w-32">{key.replace(/_/g, ' ')}:</span>
                      <span className="text-gray-600">{value as string}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Connection Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Backend Connection Info</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Expected Backend URL:</strong> http://localhost:8000</p>
          <p><strong>Start Backend:</strong> <code>cd backend && python main.py</code></p>
          <p><strong>API Documentation:</strong> http://localhost:8000/docs</p>
          <p><strong>Alternative Health:</strong> http://localhost:8000/api/health</p>
        </div>
      </div>
    </div>
  );
};

export default HealthCheck; 