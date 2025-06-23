import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/api';

interface TestResult {
  id: string;
  type: string;
  status: 'success' | 'error' | 'loading';
  data?: any;
  error?: string;
  timestamp: string;
}

interface Schema {
  id: string;
  name: string;
  description: string;
  label_type: string;
  categories_count: number;
  is_built_in: boolean;
  auto_approval_threshold: number;
}

const Phase1Testing: React.FC = () => {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('text-classification');
  const [schemas, setSchemas] = useState<Schema[]>([]);

  // Text Classification States
  const [textInput, setTextInput] = useState('John Smith works at Microsoft in Seattle and loves Apple products.');
  const [textClassificationType, setTextClassificationType] = useState('ner');
  const [customCategories, setCustomCategories] = useState('');
  const [includeMetadata, setIncludeMetadata] = useState(true);

  // Batch Text States
  const [batchTexts, setBatchTexts] = useState(`John Smith works at Microsoft in Seattle.
Apple Inc. is located in Cupertino, California.
Google was founded by Larry Page and Sergey Brin.
Amazon CEO Jeff Bezos announced new initiatives.
Tesla's Elon Musk tweeted about the latest updates.`);

  // Image Classification States
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string>('');
  const [projectId, setProjectId] = useState(1);
  const [modelName, setModelName] = useState('yolo8n');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);

  // Schema Management States
  const [newSchemaName, setNewSchemaName] = useState('');
  const [newSchemaDescription, setNewSchemaDescription] = useState('');
  const [newSchemaType, setNewSchemaType] = useState('classification');
  const [schemaCategories, setSchemaCategories] = useState([
    { id: 'category1', name: 'Category 1', description: 'First category', color: '#FF6B6B' }
  ]);

  // Job Monitoring States
  const [jobId, setJobId] = useState('');
  const [jobResults, setJobResults] = useState<any>(null);

  // Backend connection status
  const [backendStatus, setBackendStatus] = useState<'connected' | 'disconnected' | 'checking'>('checking');

  useEffect(() => {
    loadSchemas();
    checkBackendConnection();
  }, []);

  const addTestResult = (result: Omit<TestResult, 'id' | 'timestamp'>) => {
    const newResult: TestResult = {
      ...result,
      id: Date.now().toString(),
      timestamp: new Date().toLocaleTimeString()
    };
    setTestResults(prev => [newResult, ...prev]);
  };

  const loadSchemas = async () => {
    try {
      const response = await apiClient.get('/api/schemas');
      setSchemas(response.data.schemas || []);
    } catch (error) {
      console.error('Failed to load schemas:', error);
    }
  };

  const checkBackendConnection = async () => {
    try {
      const response = await apiClient.get('/api/health');
      setBackendStatus('connected');
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  // Text Classification Tests
  const testSingleTextClassification = async () => {
    setLoading(true);
    addTestResult({ type: 'Single Text Classification', status: 'loading' });

    try {
      const formData = new FormData();
      formData.append('text', textInput);
      formData.append('classification_type', textClassificationType);
      formData.append('include_metadata', includeMetadata.toString());
      
      if (customCategories && textClassificationType === 'topic') {
        const categories = customCategories.split(',').map(c => c.trim());
        categories.forEach(cat => formData.append('custom_categories', cat));
      }

      const response = await apiClient.post('/api/classify/text', formData);
      
      addTestResult({
        type: 'Single Text Classification',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Single Text Classification',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const testBatchTextClassification = async () => {
    setLoading(true);
    addTestResult({ type: 'Batch Text Classification', status: 'loading' });

    try {
      const texts = batchTexts.split('\n').filter(t => t.trim());
      const formData = new FormData();
      texts.forEach(text => formData.append('texts', text.trim()));
      formData.append('classification_type', textClassificationType);
      formData.append('include_metadata', includeMetadata.toString());

      const response = await apiClient.post('/api/classify/text/batch', formData);
      
      addTestResult({
        type: 'Batch Text Classification',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Batch Text Classification',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const testGetTextModels = async () => {
    setLoading(true);
    addTestResult({ type: 'Get Text Models', status: 'loading' });

    try {
      const response = await apiClient.get('/api/classify/text/models');
      
      addTestResult({
        type: 'Get Text Models',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Get Text Models',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  // Image Classification Tests
  const testSingleImageClassification = async () => {
    if (!selectedImage) {
      alert('Please select an image first');
      return;
    }

    setLoading(true);
    addTestResult({ type: 'Single Image Classification', status: 'loading' });

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);

      const response = await apiClient.post('/api/classify/image', formData);
      
      addTestResult({
        type: 'Single Image Classification',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Single Image Classification',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const testObjectDetection = async () => {
    if (!selectedImage) {
      alert('Please select an image first');
      return;
    }

    setLoading(true);
    addTestResult({ type: 'Object Detection', status: 'loading' });

    try {
      const formData = new FormData();
      formData.append('file', selectedImage);
      formData.append('project_id', projectId.toString());
      formData.append('model_name', modelName);
      formData.append('confidence_threshold', confidenceThreshold.toString());

      const response = await apiClient.post('/api/classify/image/detect', formData);
      
      addTestResult({
        type: 'Object Detection',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Object Detection',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  // Schema Management Tests
  const testCreateSchema = async () => {
    setLoading(true);
    addTestResult({ type: 'Create Schema', status: 'loading' });

    try {
      const schemaData = {
        name: newSchemaName,
        description: newSchemaDescription,
        label_type: newSchemaType,
        categories: schemaCategories.map(cat => ({
          id: cat.id,
          name: cat.name,
          description: cat.description,
          color: cat.color
        })),
        auto_approval_enabled: true,
        auto_approval_threshold: 0.80
      };

      const response = await apiClient.post('/api/schemas', schemaData);
      
      addTestResult({
        type: 'Create Schema',
        status: 'success',
        data: response.data
      });
      
      loadSchemas(); // Refresh schemas list
    } catch (error: any) {
      addTestResult({
        type: 'Create Schema',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const testGetBuiltInSchemas = async () => {
    setLoading(true);
    addTestResult({ type: 'Get Built-in Schemas', status: 'loading' });

    try {
      const response = await apiClient.get('/api/schemas/templates/built-in');
      
      addTestResult({
        type: 'Get Built-in Schemas',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Get Built-in Schemas',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const testValidateSchema = async (schemaId: string) => {
    setLoading(true);
    addTestResult({ type: 'Validate Schema', status: 'loading' });

    try {
      const response = await apiClient.post(`/api/schemas/${schemaId}/validate`);
      
      addTestResult({
        type: 'Validate Schema',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Validate Schema',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  // Job Monitoring Tests
  const testGetJobStatus = async () => {
    if (!jobId) {
      alert('Please enter a job ID');
      return;
    }

    setLoading(true);
    addTestResult({ type: 'Get Job Status', status: 'loading' });

    try {
      const response = await apiClient.get(`/api/classify/jobs/${jobId}`);
      
      addTestResult({
        type: 'Get Job Status',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Get Job Status',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const testGetJobResults = async () => {
    if (!jobId) {
      alert('Please enter a job ID');
      return;
    }

    setLoading(true);
    addTestResult({ type: 'Get Job Results', status: 'loading' });

    try {
      const response = await apiClient.get(`/api/classify/results/${jobId}`);
      setJobResults(response.data);
      
      addTestResult({
        type: 'Get Job Results',
        status: 'success',
        data: response.data
      });
    } catch (error: any) {
      addTestResult({
        type: 'Get Job Results',
        status: 'error',
        error: error.response?.data?.detail || error.message
      });
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const addSchemaCategory = () => {
    const newCategory = {
      id: `category${schemaCategories.length + 1}`,
      name: `Category ${schemaCategories.length + 1}`,
      description: '',
      color: '#' + Math.floor(Math.random()*16777215).toString(16)
    };
    setSchemaCategories([...schemaCategories, newCategory]);
  };

  const updateSchemaCategory = (index: number, field: string, value: string) => {
    const updated = [...schemaCategories];
    updated[index] = { ...updated[index], [field]: value };
    setSchemaCategories(updated);
  };

  const removeSchemaCategory = (index: number) => {
    setSchemaCategories(schemaCategories.filter((_, i) => i !== index));
  };

  const clearResults = () => {
    setTestResults([]);
  };

  const runAllTests = async () => {
    setLoading(true);
    addTestResult({ type: 'Running All Tests', status: 'loading' });

    try {
      // Test text classification
      await testSingleTextClassification();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await testGetTextModels();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Test schema management
      await testGetBuiltInSchemas();
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Test batch text if we have multiple texts
      if (batchTexts.split('\n').filter(t => t.trim()).length > 1) {
        await testBatchTextClassification();
      }
      
      addTestResult({
        type: 'All Tests Complete',
        status: 'success',
        data: { message: 'All available tests completed successfully' }
      });
    } catch (error) {
      addTestResult({
        type: 'All Tests',
        status: 'error',
        error: 'Some tests failed during execution'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            🚀 ModelShip Phase 1 Testing Suite
          </h1>
          <p className="text-gray-600 mb-6">
            Comprehensive testing interface for all Phase 1 features: NER Text Classification, 
            Auto-Approval Workflow, Label Schema Management, Image Classification, and Object Detection.
          </p>
          
          {/* Backend Status Indicator */}
          <div className="flex items-center space-x-4 mb-6">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
              backendStatus === 'connected' 
                ? 'bg-green-100 text-green-800' 
                : backendStatus === 'disconnected'
                ? 'bg-red-100 text-red-800'
                : 'bg-yellow-100 text-yellow-800'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                backendStatus === 'connected' 
                  ? 'bg-green-500' 
                  : backendStatus === 'disconnected'
                  ? 'bg-red-500'
                  : 'bg-yellow-500'
              }`}></div>
              <span>
                Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}
              </span>
            </div>
            
            {backendStatus === 'disconnected' && (
              <button
                onClick={checkBackendConnection}
                className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
              >
                Retry Connection
              </button>
            )}
          </div>
          
          <div className="flex space-x-4 mb-6">
            <button
              onClick={runAllTests}
              disabled={loading}
              className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Running...' : '🎯 Run All Tests'}
            </button>
            <button
              onClick={clearResults}
              className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700"
            >
              🗑️ Clear Results
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Testing Controls */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-lg p-6">
              {/* Tab Navigation */}
              <div className="flex flex-wrap gap-2 mb-6 border-b">
                {[
                  { id: 'text-classification', label: '📝 Text Classification', icon: '🔤' },
                  { id: 'image-classification', label: '🖼️ Image Classification', icon: '🎨' },
                  { id: 'schema-management', label: '🏷️ Schema Management', icon: '⚙️' },
                  { id: 'job-monitoring', label: '📊 Job Monitoring', icon: '📈' }
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`px-4 py-2 rounded-t-lg font-medium ${
                      activeTab === tab.id
                        ? 'bg-blue-500 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {tab.icon} {tab.label}
                  </button>
                ))}
              </div>

              {/* Text Classification Tab */}
              {activeTab === 'text-classification' && (
                <div className="space-y-6">
                  <h3 className="text-xl font-semibold text-gray-900">📝 Text Classification & NER Testing</h3>
                  
                  {/* Single Text Classification */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Single Text Classification</h4>
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Text Input</label>
                        <textarea
                          value={textInput}
                          onChange={(e) => setTextInput(e.target.value)}
                          className="w-full p-3 border border-gray-300 rounded-lg"
                          rows={3}
                          placeholder="Enter text to classify..."
                        />
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Classification Type</label>
                          <select
                            value={textClassificationType}
                            onChange={(e) => setTextClassificationType(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                          >
                            <option value="sentiment">Sentiment Analysis</option>
                            <option value="emotion">Emotion Detection</option>
                            <option value="topic">Topic Classification</option>
                            <option value="spam">Spam Detection</option>
                            <option value="toxicity">Toxicity Detection</option>
                            <option value="language">Language Detection</option>
                            <option value="ner">Named Entity Recognition</option>
                            <option value="named_entity">Advanced NER</option>
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">
                            <input
                              type="checkbox"
                              checked={includeMetadata}
                              onChange={(e) => setIncludeMetadata(e.target.checked)}
                              className="mr-2"
                            />
                            Include Metadata
                          </label>
                        </div>
                      </div>
                      
                      {textClassificationType === 'topic' && (
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Custom Categories (comma-separated)</label>
                          <input
                            type="text"
                            value={customCategories}
                            onChange={(e) => setCustomCategories(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                            placeholder="technology, business, sports, politics"
                          />
                        </div>
                      )}
                      
                      <button
                        onClick={testSingleTextClassification}
                        disabled={loading}
                        className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                      >
                        {loading ? 'Processing...' : '🔍 Classify Text'}
                      </button>
                    </div>
                  </div>

                  {/* Batch Text Classification */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Batch Text Classification</h4>
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                          Batch Texts (one per line)
                        </label>
                        <textarea
                          value={batchTexts}
                          onChange={(e) => setBatchTexts(e.target.value)}
                          className="w-full p-3 border border-gray-300 rounded-lg"
                          rows={5}
                          placeholder="Enter multiple texts, one per line..."
                        />
                      </div>
                      
                      <button
                        onClick={testBatchTextClassification}
                        disabled={loading}
                        className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 disabled:opacity-50"
                      >
                        {loading ? 'Processing...' : '📚 Batch Classify'}
                      </button>
                    </div>
                  </div>

                  {/* Get Models */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Model Information</h4>
                    <button
                      onClick={testGetTextModels}
                      disabled={loading}
                      className="bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
                    >
                      {loading ? 'Loading...' : '🤖 Get Available Models'}
                    </button>
                  </div>
                </div>
              )}

              {/* Image Classification Tab */}
              {activeTab === 'image-classification' && (
                <div className="space-y-6">
                  <h3 className="text-xl font-semibold text-gray-900">🖼️ Image Classification & Object Detection</h3>
                  
                  {/* Image Upload */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Image Upload</h4>
                    <div className="space-y-3">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="w-full p-2 border border-gray-300 rounded-lg"
                      />
                      
                      {imagePreview && (
                        <div className="mt-3">
                          <img
                            src={imagePreview}
                            alt="Preview"
                            className="max-w-xs max-h-48 object-contain border rounded-lg"
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Image Classification */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Image Classification</h4>
                    <button
                      onClick={testSingleImageClassification}
                      disabled={loading || !selectedImage}
                      className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50"
                    >
                      {loading ? 'Classifying...' : '🎯 Classify Image'}
                    </button>
                  </div>

                  {/* Object Detection */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Object Detection</h4>
                    <div className="space-y-3">
                      <div className="grid grid-cols-3 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Project ID</label>
                          <input
                            type="number"
                            value={projectId}
                            onChange={(e) => setProjectId(parseInt(e.target.value))}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
                          <select
                            value={modelName}
                            onChange={(e) => setModelName(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                          >
                            <option value="yolo8n">YOLO8 Nano (Fast)</option>
                            <option value="yolo8s">YOLO8 Small (Accurate)</option>
                          </select>
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Confidence</label>
                          <input
                            type="number"
                            step="0.05"
                            min="0"
                            max="1"
                            value={confidenceThreshold}
                            onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                          />
                        </div>
                      </div>
                      
                      <button
                        onClick={testObjectDetection}
                        disabled={loading || !selectedImage}
                        className="bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 disabled:opacity-50"
                      >
                        {loading ? 'Detecting...' : '🔍 Detect Objects'}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Schema Management Tab */}
              {activeTab === 'schema-management' && (
                <div className="space-y-6">
                  <h3 className="text-xl font-semibold text-gray-900">🏷️ Label Schema Management</h3>
                  
                  {/* Built-in Schemas */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Built-in Schemas</h4>
                    <button
                      onClick={testGetBuiltInSchemas}
                      disabled={loading}
                      className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                    >
                      {loading ? 'Loading...' : '📋 Get Built-in Schemas'}
                    </button>
                  </div>

                  {/* Current Schemas */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Current Schemas</h4>
                    <div className="space-y-2 mb-4">
                      {schemas.map((schema) => (
                        <div key={schema.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div>
                            <span className="font-medium">{schema.name}</span>
                            <span className="text-sm text-gray-500 ml-2">
                              ({schema.label_type}, {schema.categories_count} categories)
                            </span>
                          </div>
                          <button
                            onClick={() => testValidateSchema(schema.id)}
                            disabled={loading}
                            className="bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 disabled:opacity-50 text-sm"
                          >
                            Validate
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Create New Schema */}
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Create New Schema</h4>
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Schema Name</label>
                          <input
                            type="text"
                            value={newSchemaName}
                            onChange={(e) => setNewSchemaName(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                            placeholder="My Custom Schema"
                          />
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-700 mb-1">Schema Type</label>
                          <select
                            value={newSchemaType}
                            onChange={(e) => setNewSchemaType(e.target.value)}
                            className="w-full p-2 border border-gray-300 rounded-lg"
                          >
                            <option value="classification">Classification</option>
                            <option value="object_detection">Object Detection</option>
                            <option value="text_classification">Text Classification</option>
                            <option value="named_entity">Named Entity</option>
                            <option value="sentiment">Sentiment</option>
                          </select>
                        </div>
                      </div>
                      
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                        <textarea
                          value={newSchemaDescription}
                          onChange={(e) => setNewSchemaDescription(e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-lg"
                          rows={2}
                          placeholder="Schema description..."
                        />
                      </div>
                      
                      {/* Categories */}
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <label className="block text-sm font-medium text-gray-700">Categories</label>
                          <button
                            onClick={addSchemaCategory}
                            className="bg-green-600 text-white px-3 py-1 rounded text-sm hover:bg-green-700"
                          >
                            + Add Category
                          </button>
                        </div>
                        
                        <div className="space-y-2">
                          {schemaCategories.map((category, index) => (
                            <div key={index} className="flex items-center space-x-2 p-2 bg-gray-50 rounded">
                              <input
                                type="text"
                                value={category.name}
                                onChange={(e) => updateSchemaCategory(index, 'name', e.target.value)}
                                className="flex-1 p-1 border rounded"
                                placeholder="Category name"
                              />
                              <input
                                type="text"
                                value={category.description}
                                onChange={(e) => updateSchemaCategory(index, 'description', e.target.value)}
                                className="flex-1 p-1 border rounded"
                                placeholder="Description"
                              />
                              <input
                                type="color"
                                value={category.color}
                                onChange={(e) => updateSchemaCategory(index, 'color', e.target.value)}
                                className="w-12 h-8 border rounded"
                              />
                              <button
                                onClick={() => removeSchemaCategory(index)}
                                className="bg-red-600 text-white px-2 py-1 rounded text-sm hover:bg-red-700"
                              >
                                ×
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <button
                        onClick={testCreateSchema}
                        disabled={loading || !newSchemaName || schemaCategories.length === 0}
                        className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 disabled:opacity-50"
                      >
                        {loading ? 'Creating...' : '🏗️ Create Schema'}
                      </button>
                    </div>
                  </div>
                </div>
              )}

              {/* Job Monitoring Tab */}
              {activeTab === 'job-monitoring' && (
                <div className="space-y-6">
                  <h3 className="text-xl font-semibold text-gray-900">📊 Job Monitoring & Results</h3>
                  
                  <div className="border rounded-lg p-4">
                    <h4 className="font-medium text-gray-900 mb-3">Job Status & Results</h4>
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Job ID</label>
                        <input
                          type="text"
                          value={jobId}
                          onChange={(e) => setJobId(e.target.value)}
                          className="w-full p-2 border border-gray-300 rounded-lg"
                          placeholder="Enter job ID from batch operations..."
                        />
                      </div>
                      
                      <div className="flex space-x-4">
                        <button
                          onClick={testGetJobStatus}
                          disabled={loading || !jobId}
                          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50"
                        >
                          {loading ? 'Loading...' : '📈 Get Job Status'}
                        </button>
                        
                        <button
                          onClick={testGetJobResults}
                          disabled={loading || !jobId}
                          className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:opacity-50"
                        >
                          {loading ? 'Loading...' : '📊 Get Job Results'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Job Results Display */}
                  {jobResults && (
                    <div className="border rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 mb-3">Job Results Preview</h4>
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <pre className="text-sm overflow-x-auto">
                          {JSON.stringify(jobResults, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-gray-900">📋 Test Results</h3>
                <span className="text-sm text-gray-500">
                  {testResults.length} results
                </span>
              </div>
              
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {testResults.length === 0 ? (
                  <p className="text-gray-500 text-center py-8">
                    No test results yet. Run some tests to see results here.
                  </p>
                ) : (
                  testResults.map((result) => (
                    <div
                      key={result.id}
                      className={`p-3 rounded-lg border-l-4 ${
                        result.status === 'success'
                          ? 'bg-green-50 border-green-400'
                          : result.status === 'error'
                          ? 'bg-red-50 border-red-400'
                          : 'bg-yellow-50 border-yellow-400'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-sm">
                          {result.status === 'success' ? '✅' : result.status === 'error' ? '❌' : '⏳'} 
                          {result.type}
                        </span>
                        <span className="text-xs text-gray-500">{result.timestamp}</span>
                      </div>
                      
                      {result.status === 'error' && result.error && (
                        <p className="text-red-600 text-sm">{result.error}</p>
                      )}
                      
                      {result.status === 'success' && result.data && (
                        <div className="mt-2">
                          <details className="text-sm">
                            <summary className="cursor-pointer text-blue-600 hover:text-blue-800">
                              View Details
                            </summary>
                            <pre className="mt-2 text-xs bg-white p-2 rounded border overflow-x-auto">
                              {JSON.stringify(result.data, null, 2)}
                            </pre>
                          </details>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Phase1Testing; 