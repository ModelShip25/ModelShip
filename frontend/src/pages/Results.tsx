import React, { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { 
  ArrowDownTrayIcon,
  FunnelIcon,
  EyeIcon,
  PencilIcon,
  CheckCircleIcon,
  XCircleIcon,
  PhotoIcon,
  DocumentTextIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import apiClient from '../services/api';

interface ResultItem {
  id: number;
  filename: string;
  file_type: string;
  predicted_label: string;
  confidence: number;
  reviewed: boolean;
  ground_truth?: string;
  created_at: string;
}

interface Project {
  id: number;
  name: string;
  description: string;
}

interface ExportFormat {
  key: string;
  name: string;
  description: string;
}

const Results: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const projectId = searchParams.get('project');
  
  const [project, setProject] = useState<Project | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [filteredResults, setFilteredResults] = useState<ResultItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [showReviewed, setShowReviewed] = useState(true);
  const [showUnreviewed, setShowUnreviewed] = useState(true);
  const [minConfidence, setMinConfidence] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Export
  const [exporting, setExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState('json');

  const exportFormats: ExportFormat[] = [
    { key: 'json', name: 'JSON', description: 'Standard JSON format with all metadata' },
    { key: 'csv', name: 'CSV', description: 'Comma-separated values for spreadsheet apps' },
    { key: 'coco', name: 'COCO', description: 'COCO dataset format for computer vision' },
    { key: 'yolo', name: 'YOLO', description: 'YOLO format for object detection' }
  ];

  useEffect(() => {
    if (projectId) {
      fetchProjectResults();
    }
  }, [projectId]);

  useEffect(() => {
    applyFilters();
  }, [results, showReviewed, showUnreviewed, minConfidence, searchTerm]);

  const fetchProjectResults = async () => {
    if (!projectId) return;
    
    try {
      setLoading(true);
      
      // Fetch project details
      const projectResponse = await apiClient.get(`/api/projects/${projectId}`);
      setProject(projectResponse.data);

      // Fetch results
      const resultsResponse = await apiClient.get(`/api/projects/${projectId}/results`);
      setResults(resultsResponse.data.results || []);

      setError(null);
    } catch (err) {
      setError('Failed to load results');
      console.error('Results fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = results.filter(result => {
      // Review status filter
      if (!showReviewed && result.reviewed) return false;
      if (!showUnreviewed && !result.reviewed) return false;
      
      // Confidence filter
      if (result.confidence < minConfidence / 100) return false;
      
      // Search filter
      if (searchTerm && !result.filename.toLowerCase().includes(searchTerm.toLowerCase()) &&
          !result.predicted_label.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false;
      }
      
      return true;
    });
    
    setFilteredResults(filtered);
  };

  const handleExport = async () => {
    if (!projectId) return;
    
    setExporting(true);
    try {
      const response = await apiClient.post(`/api/export/${projectId}`, {
        format: exportFormat,
        include_reviewed_only: false,
        include_confidence: true
      }, { responseType: 'blob' });

      if (response.data) {
        const blob = new Blob([response.data]);
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `${project?.name || 'results'}_${exportFormat}.${exportFormat === 'json' ? 'json' : 'csv'}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        throw new Error('Export failed');
      }
    } catch (err) {
      setError('Export failed');
      console.error('Export error:', err);
    } finally {
      setExporting(false);
    }
  };

  const getAccuracyStats = () => {
    const reviewed = results.filter(r => r.reviewed);
    const correct = reviewed.filter(r => r.predicted_label === r.ground_truth);
    const accuracy = reviewed.length > 0 ? (correct.length / reviewed.length) * 100 : 0;
    
    return {
      total: results.length,
      classified: results.filter(r => r.predicted_label).length,
      reviewed: reviewed.length,
      accuracy: Math.round(accuracy)
    };
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  if (!project) {
    return (
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-2xl font-bold text-gray-900 mb-4">Project not found</h1>
          <button
            onClick={() => navigate('/dashboard')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  const stats = getAccuracyStats();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">{project.name}</h1>
              <p className="text-gray-600 mt-1">Classification Results & Export</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <button
                onClick={() => navigate(`/classification?project=${projectId}`)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium"
              >
                Continue Classification
              </button>
            </div>
          </div>
          
          {/* Stats */}
          <div className="mt-6 grid grid-cols-4 gap-6">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex items-center">
                <ChartBarIcon className="w-8 h-8 text-blue-500" />
                <div className="ml-3">
                  <p className="text-sm text-blue-600">Total Files</p>
                  <p className="text-2xl font-bold text-blue-900">{stats.total}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center">
                <CheckCircleIcon className="w-8 h-8 text-green-500" />
                <div className="ml-3">
                  <p className="text-sm text-green-600">Classified</p>
                  <p className="text-2xl font-bold text-green-900">{stats.classified}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="flex items-center">
                <EyeIcon className="w-8 h-8 text-purple-500" />
                <div className="ml-3">
                  <p className="text-sm text-purple-600">Reviewed</p>
                  <p className="text-2xl font-bold text-purple-900">{stats.reviewed}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-yellow-50 rounded-lg p-4">
              <div className="flex items-center">
                <div className="w-8 h-8 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-bold">%</span>
                </div>
                <div className="ml-3">
                  <p className="text-sm text-yellow-600">Accuracy</p>
                  <p className="text-2xl font-bold text-yellow-900">{stats.accuracy}%</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Filters Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6 space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <FunnelIcon className="w-5 h-5 mr-2" />
                  Filters
                </h3>
                
                {/* Search */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Search
                  </label>
                  <input
                    type="text"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="Filename or label..."
                  />
                </div>

                {/* Review Status */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Review Status
                  </label>
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={showReviewed}
                        onChange={(e) => setShowReviewed(e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">Reviewed</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="checkbox"
                        checked={showUnreviewed}
                        onChange={(e) => setShowUnreviewed(e.target.checked)}
                        className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                      />
                      <span className="ml-2 text-sm text-gray-700">Unreviewed</span>
                    </label>
                  </div>
                </div>

                {/* Confidence */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Min Confidence: {minConfidence}%
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={minConfidence}
                    onChange={(e) => setMinConfidence(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
              </div>

              {/* Export Section */}
              <div className="border-t pt-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Export Results</h4>
                
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Format
                  </label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    {exportFormats.map(format => (
                      <option key={format.key} value={format.key}>
                        {format.name}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {exportFormats.find(f => f.key === exportFormat)?.description}
                  </p>
                </div>

                <button
                  onClick={handleExport}
                  disabled={exporting || filteredResults.length === 0}
                  className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-lg font-medium flex items-center justify-center"
                >
                  {exporting ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Exporting...
                    </>
                  ) : (
                    <>
                      <ArrowDownTrayIcon className="w-4 h-4 mr-2" />
                      Export ({filteredResults.length} items)
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Results Table */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b">
                <h3 className="text-lg font-semibold text-gray-900">
                  Results ({filteredResults.length} of {results.length})
                </h3>
              </div>

              {filteredResults.length === 0 ? (
                <div className="text-center py-12">
                  <ChartBarIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600">No results found matching your filters</p>
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          File
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Predicted Label
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Confidence
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Status
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Ground Truth
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {filteredResults.map((result) => (
                        <tr key={result.id} className="hover:bg-gray-50">
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              {result.file_type.startsWith('image/') ? (
                                <PhotoIcon className="w-5 h-5 text-purple-500 mr-2" />
                              ) : (
                                <DocumentTextIcon className="w-5 h-5 text-green-500 mr-2" />
                              )}
                              <div>
                                <div className="text-sm font-medium text-gray-900 truncate max-w-48">
                                  {result.filename}
                                </div>
                                <div className="text-sm text-gray-500">
                                  {result.file_type}
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className="text-sm text-gray-900">{result.predicted_label}</span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    result.confidence >= 0.8 ? 'bg-green-500' :
                                    result.confidence >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${result.confidence * 100}%` }}
                                ></div>
                              </div>
                              <span className="text-sm text-gray-900">
                                {Math.round(result.confidence * 100)}%
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            {result.reviewed ? (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                <CheckCircleIcon className="w-3 h-3 mr-1" />
                                Reviewed
                              </span>
                            ) : (
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                                <XCircleIcon className="w-3 h-3 mr-1" />
                                Pending
                              </span>
                            )}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className="text-sm text-gray-900">
                              {result.ground_truth || '-'}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                            <button
                              onClick={() => navigate(`/classification?project=${projectId}&file=${result.id}`)}
                              className="text-blue-600 hover:text-blue-900 mr-3"
                            >
                              <EyeIcon className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-700">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-red-600 hover:text-red-500 font-medium"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Results; 