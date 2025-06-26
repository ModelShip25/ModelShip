import React, { useState } from 'react';
import apiClient from '../services/api';

interface TextClassificationResult {
  text: string;
  predicted_label: string;
  confidence: number;
  processing_time?: number;
  status: string;
  error_message?: string;
}

const TextClassificationTest: React.FC = () => {
  const [singleText, setSingleText] = useState('');
  const [batchTexts, setBatchTexts] = useState('');
  const [modelType, setModelType] = useState('sentiment');
  const [results, setResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('single');

  const testSingleClassification = async () => {
    if (!singleText.trim()) {
      setError('Please enter text to classify');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post('/api/classify/text', {
        text: singleText,
        classification_type: modelType
      });
      setResults(response.data);
    } catch (err: any) {
      setError(err?.message || 'Classification failed');
    } finally {
      setLoading(false);
    }
  };

  const testBatchClassification = async () => {
    const texts = batchTexts.split('\n').filter(text => text.trim());
    if (texts.length === 0) {
      setError('Please enter texts to classify (one per line)');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post('/api/classify/text/batch', {
        texts: texts,
        classification_type: modelType
      });
      setResults(response.data);
    } catch (err: any) {
      setError(err?.message || 'Batch classification failed');
    } finally {
      setLoading(false);
    }
  };

  const testQuickClassification = async () => {
    if (!singleText.trim()) {
      setError('Please enter text to classify');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      const response = await apiClient.post('/api/classify/text/quick', {
        text: singleText,
        classification_type: modelType
      });
      setResults(response.data);
    } catch (err: any) {
      setError(err?.message || 'Quick classification failed');
    } finally {
      setLoading(false);
    }
  };

  const sampleTexts = {
    sentiment: [
      'I love this product! It works amazingly well.',
      'This is terrible quality and I hate it.',
      'The service was okay, nothing special but not bad either.'
    ],
    emotion: [
      'I am so excited about this opportunity!',
      'I feel really sad about what happened.',
      'This makes me angry and frustrated.'
    ],
    spam: [
      'BUY NOW! Limited time offer! Click here for amazing deals!',
      'Hi, would you like to schedule a meeting next week?',
      'FREE MONEY! WIN $1000000 NOW! CLICK HERE!!!'
    ],
    toxicity: [
      'You are an idiot and I hate you!',
      'Thanks for your help, I appreciate it.',
      'Get lost, nobody wants you here!'
    ]
  };

  const loadSampleText = (category: string) => {
    const samples = sampleTexts[category as keyof typeof sampleTexts];
    if (samples) {
      setSingleText(samples[0]);
      setBatchTexts(samples.join('\n'));
    }
  };

  return (
    <div className="space-y-6">
      <div className="border-b border-gray-200 pb-4">
        <h2 className="text-2xl font-bold text-gray-900">Text Classification Testing</h2>
        <p className="text-gray-600 mt-2">Test text classification endpoints</p>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {['single', 'batch', 'quick'].map((tab) => (
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

      {/* Configuration */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Configuration</h3>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model Type</label>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-md"
            >
              <option value="sentiment">Sentiment Analysis</option>
              <option value="emotion">Emotion Detection</option>
              <option value="topic">Topic Classification</option>
              <option value="spam">Spam Detection</option>
              <option value="toxicity">Toxicity Detection</option>
              <option value="language">Language Detection</option>
            </select>
          </div>
          
          {/* Sample Text Buttons */}
          <div>
            <span className="text-sm font-medium text-gray-700 mr-2">Load Sample:</span>
            {Object.keys(sampleTexts).map((category) => (
              <button
                key={category}
                onClick={() => loadSampleText(category)}
                className="mr-2 mb-1 px-3 py-1 bg-gray-200 hover:bg-gray-300 rounded-md text-xs"
              >
                {category}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Text Input */}
      {activeTab === 'single' && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Single Text Input</h3>
          <textarea
            value={singleText}
            onChange={(e) => setSingleText(e.target.value)}
            placeholder="Enter text to classify..."
            className="w-full p-3 border border-gray-300 rounded-md h-24"
          />
        </div>
      )}

      {activeTab === 'batch' && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Batch Text Input</h3>
          <textarea
            value={batchTexts}
            onChange={(e) => setBatchTexts(e.target.value)}
            placeholder="Enter texts to classify (one per line)..."
            className="w-full p-3 border border-gray-300 rounded-md h-32"
          />
          <p className="text-sm text-gray-600 mt-1">
            Lines: {batchTexts.split('\n').filter(text => text.trim()).length}
          </p>
        </div>
      )}

      {activeTab === 'quick' && (
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold mb-3">Quick Classification (No Auth)</h3>
          <textarea
            value={singleText}
            onChange={(e) => setSingleText(e.target.value)}
            placeholder="Enter text for quick classification..."
            className="w-full p-3 border border-gray-300 rounded-md h-24"
          />
        </div>
      )}

      {/* Control Panel */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Test Actions</h3>
        <div className="flex gap-3">
          {activeTab === 'single' && (
            <button
              onClick={testSingleClassification}
              disabled={loading}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-4 py-2 rounded-md"
            >
              Classify Single Text
            </button>
          )}
          {activeTab === 'batch' && (
            <button
              onClick={testBatchClassification}
              disabled={loading}
              className="bg-green-500 hover:bg-green-600 disabled:bg-green-300 text-white px-4 py-2 rounded-md"
            >
              Classify Batch
            </button>
          )}
          {activeTab === 'quick' && (
            <button
              onClick={testQuickClassification}
              disabled={loading}
              className="bg-purple-500 hover:bg-purple-600 disabled:bg-purple-300 text-white px-4 py-2 rounded-md"
            >
              Quick Classify
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
          <h4 className="text-green-800 font-semibold mb-3">Classification Results</h4>
          <pre className="text-sm text-gray-700 whitespace-pre-wrap overflow-auto max-h-64">
            {JSON.stringify(results, null, 2)}
          </pre>
        </div>
      )}

      {/* Test Instructions */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-2">Test Instructions</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Single:</strong> Classify one text with authentication</p>
          <p><strong>Batch:</strong> Classify multiple texts at once</p>
          <p><strong>Quick:</strong> Fast classification without authentication</p>
          <p><strong>Models:</strong> sentiment, emotion, topic, spam, toxicity, language</p>
        </div>
      </div>
    </div>
  );
};

export default TextClassificationTest; 