import React, { useState } from 'react';
import apiClient from '../services/api';

const JobManagementTest: React.FC = () => {
  const [jobId, setJobId] = useState('');
  const [jobStatus, setJobStatus] = useState<any>(null);
  const [jobResults, setJobResults] = useState<any>(null);
  const [userJobs, setUserJobs] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('status');

  const getJobStatus = async () => {
    if (!jobId.trim()) {
      setError('Please enter a job ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get(`/api/classify/jobs/${jobId}`);
      setJobStatus(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get job status');
    } finally {
      setLoading(false);
    }
  };

  const getJobResults = async () => {
    if (!jobId.trim()) {
      setError('Please enter a job ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get(`/api/classify/results/${jobId}`);
      setJobResults(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get job results');
    } finally {
      setLoading(false);
    }
  };

  const getUserJobs = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get('/api/classify/jobs');
      setUserJobs(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get user jobs');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">Job Management Testing</h2>
        <p className="text-gray-600 mt-2">Test job status and results endpoints</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {['status', 'results', 'list'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab === 'status' ? 'Job Status' : tab === 'results' ? 'Job Results' : 'List Jobs'}
            </button>
          ))}
        </nav>
      </div>

      {/* Job ID Input */}
      {(activeTab === 'status' || activeTab === 'results') && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Job ID</h3>
          <input
            type="text"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
            placeholder="Enter job ID (e.g., 1, 2, 3...)"
            className="w-full p-3 border border-gray-300 rounded-md"
          />
        </div>
      )}

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Actions</h3>
        <div className="flex gap-3">
          {activeTab === 'status' && (
            <button
              onClick={getJobStatus}
              disabled={loading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
            >
              Get Job Status
            </button>
          )}
          {activeTab === 'results' && (
            <button
              onClick={getJobResults}
              disabled={loading}
              className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
            >
              Get Job Results
            </button>
          )}
          {activeTab === 'list' && (
            <button
              onClick={getUserJobs}
              disabled={loading}
              className="bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white px-4 py-2 rounded-md"
            >
              List All Jobs
            </button>
          )}
        </div>
      </div>

      {/* Loading/Error */}
      {loading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-2 text-gray-600">Loading...</span>
        </div>
      )}

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="text-red-800 font-semibold">Error</h4>
          <p className="text-red-600 mt-1">{error}</p>
        </div>
      )}

      {/* Job Status Display */}
      {activeTab === 'status' && jobStatus && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-blue-800 font-semibold mb-3">Job Status</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-700">Job ID:</span>
              <span className="ml-2">{jobStatus.job_id}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Status:</span>
              <span className={`ml-2 font-semibold ${
                jobStatus.status === 'completed' ? 'text-green-600' :
                jobStatus.status === 'failed' ? 'text-red-600' :
                'text-yellow-600'
              }`}>{jobStatus.status}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Job Type:</span>
              <span className="ml-2">{jobStatus.job_type}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Progress:</span>
              <span className="ml-2">{jobStatus.progress_percentage}%</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Total Items:</span>
              <span className="ml-2">{jobStatus.total_items}</span>
            </div>
            <div>
              <span className="font-medium text-gray-700">Completed:</span>
              <span className="ml-2">{jobStatus.completed_items}</span>
            </div>
          </div>
        </div>
      )}

      {/* Job Results Display */}
      {activeTab === 'results' && jobResults && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-green-800 font-semibold mb-3">Job Results</h4>
          <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-auto max-h-96">
            {JSON.stringify(jobResults, null, 2)}
          </pre>
        </div>
      )}

      {/* User Jobs List */}
      {activeTab === 'list' && userJobs && (
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h4 className="text-purple-800 font-semibold mb-3">User Jobs</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {userJobs.jobs && userJobs.jobs.map((job: any, index: number) => (
              <div key={index} className="p-3 bg-white rounded border text-sm">
                <div className="grid grid-cols-4 gap-2">
                  <div>
                    <span className="font-medium">ID:</span> {job.job_id}
                  </div>
                  <div>
                    <span className="font-medium">Type:</span> {job.job_type}
                  </div>
                  <div>
                    <span className="font-medium">Status:</span> 
                    <span className={`ml-1 ${
                      job.status === 'completed' ? 'text-green-600' :
                      job.status === 'failed' ? 'text-red-600' :
                      'text-yellow-600'
                    }`}>{job.status}</span>
                  </div>
                  <div>
                    <span className="font-medium">Progress:</span> {job.progress_percentage}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Job Management Info</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Job Status:</strong> GET /api/classify/jobs/{`{job_id}`}</p>
          <p><strong>Job Results:</strong> GET /api/classify/results/{`{job_id}`}</p>
          <p><strong>List Jobs:</strong> GET /api/classify/jobs</p>
          <p><strong>Note:</strong> Create jobs through image/text classification first</p>
        </div>
      </div>
    </div>
  );
};

export default JobManagementTest; 