import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowLeftIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  PhotoIcon,
  DocumentTextIcon,
  ChartBarIcon,
  ClockIcon,
  StarIcon,
  EyeIcon
} from '@heroicons/react/24/outline';

interface MockResult {
  id: string;
  type: 'image' | 'text' | 'batch';
  fileName: string;
  label: string;
  confidence: number;
  processingTime: number;
  status: 'completed' | 'processing' | 'error';
  description?: string;
  metadata?: any;
}

const TestResults: React.FC = () => {
  const [filter, setFilter] = useState<'all' | 'image' | 'text' | 'batch'>('all');

  // Mock results data
  const mockResults: MockResult[] = [
    {
      id: '1',
      type: 'image',
      fileName: 'golden_retriever.jpg',
      label: 'Golden Retriever',
      confidence: 94.2,
      processingTime: 0.8,
      status: 'completed',
      description: 'A golden retriever dog breed with characteristic golden coat',
      metadata: { width: 1920, height: 1080, size: '2.3MB' }
    },
    {
      id: '2',
      type: 'text',
      fileName: 'customer_review.txt',
      label: 'Positive Sentiment',
      confidence: 87.6,
      processingTime: 0.3,
      status: 'completed',
      description: 'Customer review expressing satisfaction with product quality',
      metadata: { wordCount: 45, language: 'English' }
    },
    {
      id: '3',
      type: 'image',
      fileName: 'sports_car.jpg',
      label: 'Sports Car',
      confidence: 91.8,
      processingTime: 0.9,
      status: 'completed',
      description: 'Red sports car, likely a Ferrari or similar luxury vehicle',
      metadata: { width: 1600, height: 900, size: '1.8MB' }
    },
    {
      id: '4',
      type: 'text',
      fileName: 'spam_email.txt',
      label: 'Spam',
      confidence: 96.4,
      processingTime: 0.2,
      status: 'completed',
      description: 'Email classified as spam due to promotional language patterns',
      metadata: { wordCount: 127, language: 'English' }
    },
    {
      id: '5',
      type: 'batch',
      fileName: 'product_images_batch.zip',
      label: 'Mixed Categories',
      confidence: 89.3,
      processingTime: 12.4,
      status: 'completed',
      description: 'Batch processing of 25 product images with various categories',
      metadata: { totalFiles: 25, successfulClassifications: 24, averageConfidence: 89.3 }
    },
    {
      id: '6',
      type: 'image',
      fileName: 'cat_sleeping.jpg',
      label: 'Egyptian Cat',
      confidence: 82.1,
      processingTime: 0.7,
      status: 'completed',
      description: 'Domestic cat in sleeping position, possibly Egyptian Mau breed',
      metadata: { width: 1024, height: 768, size: '1.2MB' }
    }
  ];

  const filteredResults = filter === 'all' 
    ? mockResults 
    : mockResults.filter(result => result.type === filter);

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'image': return <PhotoIcon className="h-5 w-5" />;
      case 'text': return <DocumentTextIcon className="h-5 w-5" />;
      case 'batch': return <ChartBarIcon className="h-5 w-5" />;
      default: return <EyeIcon className="h-5 w-5" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600 bg-green-100';
    if (confidence >= 80) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'processing': return <ClockIcon className="h-5 w-5 text-yellow-500 animate-spin" />;
      case 'error': return <ExclamationCircleIcon className="h-5 w-5 text-red-500" />;
      default: return <ClockIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Link 
                to="/test" 
                className="mr-4 p-2 rounded-lg hover:bg-gray-100 transition-colors"
              >
                <ArrowLeftIcon className="h-6 w-6 text-gray-600" />
              </Link>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Classification Results</h1>
                <p className="text-gray-600">View and analyze your AI classification results</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <StarIcon className="h-5 w-5 text-yellow-500" />
                <span className="text-sm text-gray-600">Average Confidence: 89.2%</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Filters */}
        <div className="mb-6">
          <div className="flex space-x-2">
            {['all', 'image', 'text', 'batch'].map((filterType) => (
              <button
                key={filterType}
                onClick={() => setFilter(filterType as any)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  filter === filterType
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300'
                }`}
              >
                {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
                {filterType !== 'all' && (
                  <span className="ml-2 text-xs bg-gray-200 text-gray-700 px-2 py-0.5 rounded-full">
                    {mockResults.filter(r => r.type === filterType).length}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Results Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredResults.map((result) => (
            <div key={result.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
              {/* Header */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                  <div className={`p-2 rounded-lg ${
                    result.type === 'image' ? 'bg-blue-100 text-blue-600' :
                    result.type === 'text' ? 'bg-green-100 text-green-600' :
                    'bg-purple-100 text-purple-600'
                  }`}>
                    {getTypeIcon(result.type)}
                  </div>
                  <span className="text-sm font-medium text-gray-900 capitalize">{result.type}</span>
                </div>
                {getStatusIcon(result.status)}
              </div>

              {/* File Name */}
              <h3 className="text-lg font-semibold text-gray-900 mb-2 truncate" title={result.fileName}>
                {result.fileName}
              </h3>

              {/* Classification Result */}
              <div className="mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-gray-600">Predicted Label</span>
                  <span className={`text-xs px-2 py-1 rounded-full font-medium ${getConfidenceColor(result.confidence)}`}>
                    {result.confidence}%
                  </span>
                </div>
                <p className="text-xl font-bold text-gray-900">{result.label}</p>
                {result.description && (
                  <p className="text-sm text-gray-600 mt-2">{result.description}</p>
                )}
              </div>

              {/* Metadata */}
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Processing Time</span>
                  <span className="font-medium">{result.processingTime}s</span>
                </div>
                
                {result.metadata && (
                  <>
                    {result.type === 'image' && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Dimensions</span>
                          <span className="font-medium">{result.metadata.width}×{result.metadata.height}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">File Size</span>
                          <span className="font-medium">{result.metadata.size}</span>
                        </div>
                      </>
                    )}
                    
                    {result.type === 'text' && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Word Count</span>
                          <span className="font-medium">{result.metadata.wordCount}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Language</span>
                          <span className="font-medium">{result.metadata.language}</span>
                        </div>
                      </>
                    )}
                    
                    {result.type === 'batch' && (
                      <>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Total Files</span>
                          <span className="font-medium">{result.metadata.totalFiles}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-600">Successful</span>
                          <span className="font-medium">{result.metadata.successfulClassifications}</span>
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>

              {/* Actions */}
              <div className="mt-4 pt-4 border-t border-gray-200">
                <div className="flex space-x-2">
                  <button className="flex-1 bg-blue-600 text-white py-2 px-3 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
                    View Details
                  </button>
                  <button className="px-3 py-2 border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors">
                    Export
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Empty State */}
        {filteredResults.length === 0 && (
          <div className="text-center py-12">
            <EyeIcon className="h-16 w-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
            <p className="text-gray-600 mb-6">
              No classification results match your current filter.
            </p>
            <Link
              to="/test"
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Start New Classification
            </Link>
          </div>
        )}

        {/* Summary Stats */}
        {filteredResults.length > 0 && (
          <div className="mt-12 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Summary Statistics</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{filteredResults.length}</div>
                <div className="text-sm text-gray-600">Total Classifications</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {(filteredResults.reduce((acc, r) => acc + r.confidence, 0) / filteredResults.length).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600">Average Confidence</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {(filteredResults.reduce((acc, r) => acc + r.processingTime, 0) / filteredResults.length).toFixed(1)}s
                </div>
                <div className="text-sm text-gray-600">Average Time</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-600">
                  {filteredResults.filter(r => r.status === 'completed').length}
                </div>
                <div className="text-sm text-gray-600">Completed</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TestResults; 