import React, { useState } from 'react';
import apiClient from '../services/api';

const ReviewSystemTest: React.FC = () => {
  const [resultId, setResultId] = useState('');
  const [reviewData, setReviewData] = useState({
    reviewed: true,
    ground_truth: '',
    confidence_score: 0.9
  });
  const [reviewQueue, setReviewQueue] = useState<any>(null);
  const [reviewStats, setReviewStats] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('queue');

  const getReviewQueue = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get('/api/review/queue');
      setReviewQueue(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get review queue');
    } finally {
      setLoading(false);
    }
  };

  const submitReview = async () => {
    if (!resultId.trim()) {
      setError('Please enter a result ID');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.put(`/api/review/result/${resultId}`, reviewData, { headers: { 'Content-Type': 'application/json' } });
      setReviewStats(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit review');
    } finally {
      setLoading(false);
    }
  };

  const getAnalytics = async () => {
    if (!resultId.trim()) {
      setError('Please enter a job ID for analytics');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.get(`/api/review/analytics/${resultId}`);
      setReviewStats(response.data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get analytics');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">Review System Testing</h2>
        <p className="text-gray-600 mt-2">Test review queue and submission endpoints</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {['queue', 'submit', 'analytics'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-2 px-1 border-b-2 font-medium text-sm capitalize ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </div>

      {/* Review Submission Form */}
      {activeTab === 'submit' && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Submit Review</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Result ID</label>
              <input
                type="text"
                value={resultId}
                onChange={(e) => setResultId(e.target.value)}
                placeholder="Enter result ID to review"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Ground Truth Label</label>
              <input
                type="text"
                value={reviewData.ground_truth}
                onChange={(e) => setReviewData({...reviewData, ground_truth: e.target.value})}
                placeholder="Enter correct label"
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confidence Score</label>
              <input
                type="number"
                min="0"
                max="1"
                step="0.1"
                value={reviewData.confidence_score}
                onChange={(e) => setReviewData({...reviewData, confidence_score: parseFloat(e.target.value)})}
                className="w-full p-2 border border-gray-300 rounded-md"
              />
            </div>
            <div className="flex items-center">
              <input
                type="checkbox"
                checked={reviewData.reviewed}
                onChange={(e) => setReviewData({...reviewData, reviewed: e.target.checked})}
                className="mr-2"
              />
              <label className="text-sm font-medium text-gray-700">Mark as reviewed</label>
            </div>
          </div>
        </div>
      )}

      {/* Analytics Input */}
      {activeTab === 'analytics' && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Review Analytics</h3>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Job ID</label>
            <input
              type="text"
              value={resultId}
              onChange={(e) => setResultId(e.target.value)}
              placeholder="Enter job ID for analytics"
              className="w-full p-2 border border-gray-300 rounded-md"
            />
          </div>
        </div>
      )}

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Actions</h3>
        <div className="flex gap-3">
          {activeTab === 'queue' && (
            <button
              onClick={getReviewQueue}
              disabled={loading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
            >
              Get Review Queue
            </button>
          )}
          {activeTab === 'submit' && (
            <button
              onClick={submitReview}
              disabled={loading}
              className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
            >
              Submit Review
            </button>
          )}
          {activeTab === 'analytics' && (
            <button
              onClick={getAnalytics}
              disabled={loading}
              className="bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white px-4 py-2 rounded-md"
            >
              Get Analytics
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

      {/* Results Display */}
      {(reviewQueue || reviewStats) && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h4 className="text-green-800 font-semibold mb-3">
            {activeTab === 'queue' ? 'Review Queue' : 
             activeTab === 'submit' ? 'Review Submitted' : 'Review Analytics'}
          </h4>
          <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-auto max-h-96">
            {JSON.stringify(reviewQueue || reviewStats, null, 2)}
          </pre>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Review System Info</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Review Queue:</strong> GET /api/review/queue</p>
          <p><strong>Submit Review:</strong> PUT /api/review/result/{`{result_id}`}</p>
          <p><strong>Analytics:</strong> GET /api/review/analytics/{`{job_id}`}</p>
          <p><strong>Note:</strong> Need existing results from classification jobs</p>
        </div>
      </div>
    </div>
  );
};

export default ReviewSystemTest; 