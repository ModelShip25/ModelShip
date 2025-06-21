import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  CloudArrowUpIcon, 
  PhotoIcon, 
  DocumentTextIcon,
  BeakerIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ClockIcon,
  EyeIcon
} from '@heroicons/react/24/outline';

interface ClassificationResult {
  predicted_label: string;
  confidence: number;
  processing_time: number;
  classification_id: string;
  status: string;
  description?: string;
  metadata?: any;
}

const TestClassification: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'image' | 'text' | 'batch'>('image');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Image classification state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Text classification state
  const [textInput, setTextInput] = useState('');
  const [textType, setTextType] = useState('general');

  // Batch processing state
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const [batchResults, setBatchResults] = useState<any[]>([]);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      setResult(null);
      setError(null);
    }
  };

  const classifyImage = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch('http://localhost:8000/api/classify/image/quick', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Classification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const classifyText = async () => {
    if (!textInput.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('text', textInput);
      formData.append('classification_type', textType);

      const response = await fetch('http://localhost:8000/api/classify/text/quick', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Text classification failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    setBatchFiles(files);
    setBatchResults([]);
    setError(null);
  };

  const processBatch = async () => {
    if (batchFiles.length === 0) return;

    setIsLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      batchFiles.forEach(file => {
        formData.append('files', file);
      });

      const response = await fetch('http://localhost:8000/api/classify/batch/quick', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setBatchResults([data]);
    } catch (err: any) {
      setError(err.message || 'Batch processing failed');
    } finally {
      setIsLoading(false);
    }
  };

  const tabs = [
    { id: 'image', name: 'Image Classification', icon: PhotoIcon },
    { id: 'text', name: 'Text Classification', icon: DocumentTextIcon },
    { id: 'batch', name: 'Batch Processing', icon: ChartBarIcon },
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center items-center mb-4">
            <BeakerIcon className="h-12 w-12 text-blue-600 mr-3" />
            <h1 className="text-4xl font-bold text-gray-900">ModelShip Test Lab</h1>
          </div>
          <p className="text-xl text-gray-600">
            Test our AI-powered auto-labeling features - no authentication required
          </p>
        </div>

        {/* Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex justify-center space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  } whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm flex items-center`}
                >
                  <tab.icon className="h-5 w-5 mr-2" />
                  {tab.name}
                </button>
              ))}
            </nav>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              {activeTab === 'image' && 'Upload Image'}
              {activeTab === 'text' && 'Enter Text'}
              {activeTab === 'batch' && 'Batch Upload'}
            </h2>

            {/* Image Classification */}
            {activeTab === 'image' && (
              <div className="space-y-6">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageSelect}
                    className="hidden"
                    id="image-upload"
                  />
                  <label htmlFor="image-upload" className="cursor-pointer">
                    {previewUrl ? (
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="max-h-64 mx-auto rounded-lg"
                      />
                    ) : (
                      <div>
                        <CloudArrowUpIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                        <p className="text-lg text-gray-600">Click to upload an image</p>
                        <p className="text-sm text-gray-500">JPG, PNG, GIF up to 10MB</p>
                      </div>
                    )}
                  </label>
                </div>

                {selectedFile && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">
                      <strong>File:</strong> {selectedFile.name}
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Size:</strong> {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                )}

                <button
                  onClick={classifyImage}
                  disabled={!selectedFile || isLoading}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <ClockIcon className="h-5 w-5 mr-2 animate-spin" />
                      Classifying...
                    </>
                  ) : (
                    <>
                      <PhotoIcon className="h-5 w-5 mr-2" />
                      Classify Image
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Text Classification */}
            {activeTab === 'text' && (
              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Classification Type
                  </label>
                  <select
                    value={textType}
                    onChange={(e) => setTextType(e.target.value)}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="general">General Classification</option>
                    <option value="sentiment">Sentiment Analysis</option>
                    <option value="topic">Topic Detection</option>
                    <option value="spam">Spam Detection</option>
                    <option value="language">Language Detection</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Text to Classify
                  </label>
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="Enter text to classify..."
                    rows={6}
                    className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <button
                  onClick={classifyText}
                  disabled={!textInput.trim() || isLoading}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <ClockIcon className="h-5 w-5 mr-2 animate-spin" />
                      Classifying...
                    </>
                  ) : (
                    <>
                      <DocumentTextIcon className="h-5 w-5 mr-2" />
                      Classify Text
                    </>
                  )}
                </button>
              </div>
            )}

            {/* Batch Processing */}
            {activeTab === 'batch' && (
              <div className="space-y-6">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors">
                  <input
                    type="file"
                    accept="image/*"
                    multiple
                    onChange={handleBatchUpload}
                    className="hidden"
                    id="batch-upload"
                  />
                  <label htmlFor="batch-upload" className="cursor-pointer">
                    <CloudArrowUpIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                    <p className="text-lg text-gray-600">Upload multiple images</p>
                    <p className="text-sm text-gray-500">Select up to 10 images for batch processing</p>
                  </label>
                </div>

                {batchFiles.length > 0 && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600 mb-2">
                      <strong>Selected Files:</strong> {batchFiles.length}
                    </p>
                    <div className="max-h-32 overflow-y-auto">
                      {batchFiles.map((file, index) => (
                        <p key={index} className="text-xs text-gray-500">
                          {file.name}
                        </p>
                      ))}
                    </div>
                  </div>
                )}

                <button
                  onClick={processBatch}
                  disabled={batchFiles.length === 0 || isLoading}
                  className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isLoading ? (
                    <>
                      <ClockIcon className="h-5 w-5 mr-2 animate-spin" />
                      Processing Batch...
                    </>
                  ) : (
                    <>
                      <ChartBarIcon className="h-5 w-5 mr-2" />
                      Process Batch
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Results</h2>

            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
                <div className="flex items-center">
                  <ExclamationCircleIcon className="h-5 w-5 text-red-400 mr-2" />
                  <p className="text-red-800">{error}</p>
                </div>
              </div>
            )}

            {result && (
              <div className="space-y-4">
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center mb-2">
                    <CheckCircleIcon className="h-5 w-5 text-green-400 mr-2" />
                    <p className="text-green-800 font-medium">Classification Complete</p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Predicted Label</p>
                    <p className="text-lg font-bold text-gray-900">{result.predicted_label}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Confidence</p>
                    <p className="text-lg font-bold text-gray-900">{result.confidence}%</p>
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Processing Time</p>
                  <p className="text-lg font-bold text-gray-900">{result.processing_time}s</p>
                </div>

                {result.description && (
                  <div className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Description</p>
                    <p className="text-gray-900">{result.description}</p>
                  </div>
                )}

                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-sm text-gray-600">Classification ID</p>
                  <p className="text-xs font-mono text-gray-700">{result.classification_id}</p>
                </div>
              </div>
            )}

            {batchResults.length > 0 && (
              <div className="space-y-4">
                <h3 className="text-lg font-bold text-gray-900">Batch Results</h3>
                {batchResults.map((batchResult, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-4">
                    <p className="text-sm text-gray-600">Job ID: {batchResult.job_id}</p>
                    <p className="text-sm text-gray-600">Status: {batchResult.status}</p>
                    <p className="text-sm text-gray-600">Files: {batchResult.total_files}</p>
                  </div>
                ))}
              </div>
            )}

            {!result && !error && !isLoading && (
              <div className="text-center py-12">
                <BeakerIcon className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">
                  {activeTab === 'image' && 'Upload an image to see classification results'}
                  {activeTab === 'text' && 'Enter text to see classification results'}
                  {activeTab === 'batch' && 'Upload multiple files for batch processing'}
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Feature Info */}
        <div className="mt-12 bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-bold text-blue-900 mb-4">🚀 ModelShip Core Features</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <h4 className="font-medium text-blue-800">Image Classification</h4>
              <p className="text-blue-700">ResNet-50 model with 1000+ categories</p>
            </div>
            <div>
              <h4 className="font-medium text-blue-800">Text Analysis</h4>
              <p className="text-blue-700">Sentiment, topic, and spam detection</p>
            </div>
            <div>
              <h4 className="font-medium text-blue-800">Batch Processing</h4>
              <p className="text-blue-700">Process multiple files efficiently</p>
            </div>
          </div>
          <div className="mt-6 text-center">
            <Link
              to="/test-results"
              className="inline-flex items-center bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              <EyeIcon className="h-5 w-5 mr-2" />
              View Sample Results
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TestClassification; 