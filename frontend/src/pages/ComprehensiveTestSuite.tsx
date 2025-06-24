import React, { useState, useEffect } from 'react';
import api from '../services/api';

interface TestResult {
  success: boolean;
  message: string;
  data?: any;
  timestamp: string;
}

interface TestSection {
  name: string;
  tests: TestCase[];
  expanded: boolean;
}

interface TestCase {
  name: string;
  description: string;
  endpoint: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  payload?: any;
  result?: TestResult;
  running: boolean;
}

const ComprehensiveTestSuite: React.FC = () => {
  const [testSections, setTestSections] = useState<TestSection[]>([]);
  const [globalResults, setGlobalResults] = useState<{ [key: string]: TestResult }>({});
  const [runningAll, setRunningAll] = useState(false);

  useEffect(() => {
    initializeTestSections();
  }, []);

  const initializeTestSections = () => {
    const sections: TestSection[] = [
      {
        name: "🚀 Phase 1: Core Auto-Labeling Platform",
        expanded: true,
        tests: [
          {
            name: "Project Management",
            description: "Create and manage projects",
            endpoint: "/api/projects",
            method: "POST",
            payload: {
              name: "Test Project",
              description: "Comprehensive test project",
              project_type: "classification"
            },
            running: false
          },
          {
            name: "File Upload",
            description: "Upload files for processing",
            endpoint: "/api/upload",
            method: "POST",
            payload: {
              files: ["test_image.jpg", "test_document.pdf"],
              project_id: 1
            },
            running: false
          },
          {
            name: "Image Classification",
            description: "Classify uploaded images",
            endpoint: "/api/classify/image",
            method: "POST",
            payload: {
              image_path: "uploads/test_image.jpg",
              confidence_threshold: 0.8
            },
            running: false
          },
          {
            name: "Text Classification",
            description: "Classify text content",
            endpoint: "/api/classify/text",
            method: "POST",
            payload: {
              text: "This is a sample text for classification testing",
              project_id: 1
            },
            running: false
          },
          {
            name: "Object Detection",
            description: "Detect objects in images",
            endpoint: "/api/object-detection/detect",
            method: "POST",
            payload: {
              image_path: "uploads/test_image.jpg",
              confidence_threshold: 0.5
            },
            running: false
          },
          {
            name: "Review System",
            description: "Human review and approval",
            endpoint: "/api/review/submit",
            method: "POST",
            payload: {
              result_id: 1,
              action: "approve",
              confidence: 0.95,
              notes: "Looks correct"
            },
            running: false
          },
          {
            name: "Export Results",
            description: "Export in various formats",
            endpoint: "/api/export/coco",
            method: "POST",
            payload: {
              project_id: 1,
              include_annotations: true
            },
            running: false
          }
        ]
      },
      {
        name: "🛠 Phase 2: Developer Experience & Advanced QA",
        expanded: true,
        tests: [
          {
            name: "Advanced Export Formats",
            description: "Export in YOLO, COCO, JSON formats",
            endpoint: "/api/advanced-export/export",
            method: "POST",
            payload: {
              project_id: 1,
              format: "yolo",
              include_metadata: true
            },
            running: false
          },
          {
            name: "Quality Dashboard",
            description: "Get annotation quality metrics",
            endpoint: "/api/annotation-quality/dashboard/1",
            method: "GET",
            running: false
          },
          {
            name: "Gold Standard Testing",
            description: "Test with gold standard samples",
            endpoint: "/api/gold-standard/test",
            method: "POST",
            payload: {
              project_id: 1,
              sample_size: 10
            },
            running: false
          },
          {
            name: "Data Versioning",
            description: "Create and manage data versions",
            endpoint: "/api/data-versioning/create-version",
            method: "POST",
            payload: {
              project_id: 1,
              version_name: "v1.0.0",
              description: "Initial version"
            },
            running: false
          },
          {
            name: "Label Schema Management",
            description: "Manage label schemas",
            endpoint: "/api/label-schema/create",
            method: "POST",
            payload: {
              project_id: 1,
              schema: {
                categories: ["positive", "negative", "neutral"],
                hierarchical: false
              }
            },
            running: false
          },
          {
            name: "Analytics Dashboard",
            description: "Get project analytics",
            endpoint: "/api/analytics/dashboard/1",
            method: "GET",
            running: false
          }
        ]
      },
      {
        name: "🌐 Phase 3: Industry Verticals & Advanced Features",
        expanded: true,
        tests: [
          {
            name: "Vertical Templates",
            description: "Create project with industry template",
            endpoint: "/api/vertical-templates/create-project",
            method: "POST",
            payload: {
              template_id: "healthcare_radiology",
              project_name: "Medical Image Analysis",
              customizations: {
                confidence_threshold: 0.95,
                require_expert_review: true
              }
            },
            running: false
          },
          {
            name: "Expert-In-Loop Request",
            description: "Request expert review",
            endpoint: "/api/expert-in-loop/request-review",
            method: "POST",
            payload: {
              project_id: 1,
              item_id: "complex_case_1",
              expertise_required: ["radiology"],
              urgency: "high"
            },
            running: false
          },
          {
            name: "ML Pre-labeling Model",
            description: "Create ML pre-labeling model",
            endpoint: "/api/ml-prelabeling/create-model",
            method: "POST",
            payload: {
              project_id: 1,
              model_type: "random_forest",
              training_data: [
                {"text": "Great product!", "label": "positive"},
                {"text": "Poor quality", "label": "negative"},
                {"text": "Amazing service", "label": "positive"},
                {"text": "Terrible experience", "label": "negative"}
              ]
            },
            running: false
          },
          {
            name: "Generate Pre-labels",
            description: "Generate ML pre-labels",
            endpoint: "/api/ml-prelabeling/generate-prelabels",
            method: "POST",
            payload: {
              project_id: 1,
              items: [
                {"id": "item1", "text": "Excellent quality and service"},
                {"id": "item2", "text": "Disappointing product quality"}
              ]
            },
            running: false
          },
          {
            name: "Consensus Task",
            description: "Create consensus annotation task",
            endpoint: "/api/consensus/create-task",
            method: "POST",
            payload: {
              project_id: 1,
              item_data: {
                "id": "consensus_item_1",
                "text": "Ambiguous review requiring multiple annotators",
                "type": "text"
              },
              required_annotators: 3,
              consensus_method: "majority_vote"
            },
            running: false
          },
          {
            name: "Bias & Fairness Report",
            description: "Generate bias analysis report",
            endpoint: "/api/bias-fairness/generate-report",
            method: "POST",
            payload: {
              project_id: 1,
              report_type: "comprehensive",
              protected_attributes: ["gender", "age_group"]
            },
            running: false
          },
          {
            name: "Security Compliance Assessment",
            description: "Run compliance assessment",
            endpoint: "/api/security-compliance/assess",
            method: "POST",
            payload: {
              project_id: 1,
              standards: ["HIPAA", "GDPR", "SOC2"]
            },
            running: false
          }
        ]
      },
      {
        name: "🎯 Advanced Features: Golden Dataset & Edge Cases",
        expanded: true,
        tests: [
          {
            name: "Golden Dataset Performance",
            description: "Track annotator performance against gold standard",
            endpoint: "/api/gold-standard/performance/1",
            method: "GET",
            running: false
          },
          {
            name: "Active Edge-Case Detection",
            description: "Identify and surface edge cases",
            endpoint: "/api/active-learning/edge-cases",
            method: "POST",
            payload: {
              project_id: 1,
              uncertainty_threshold: 0.3,
              sample_size: 20
            },
            running: false
          },
          {
            name: "Feedback Loop Integration",
            description: "Submit feedback for model improvement",
            endpoint: "/api/feedback/submit",
            method: "POST",
            payload: {
              project_id: 1,
              item_id: "feedback_item_1",
              issue_type: "incorrect_label",
              correct_label: "positive",
              confidence: 0.9
            },
            running: false
          },
          {
            name: "Taxonomy Management",
            description: "Visual taxonomy editor operations",
            endpoint: "/api/taxonomy/update",
            method: "POST",
            payload: {
              project_id: 1,
              taxonomy: {
                "root": {
                  "sentiment": ["positive", "negative", "neutral"],
                  "topics": ["technology", "healthcare", "finance"]
                }
              }
            },
            running: false
          }
        ]
      }
    ];

    setTestSections(sections);
  };

  const runTest = async (sectionIndex: number, testIndex: number) => {
    const newSections = [...testSections];
    newSections[sectionIndex].tests[testIndex].running = true;
    setTestSections(newSections);

    const test = newSections[sectionIndex].tests[testIndex];
    const testKey = `${sectionIndex}-${testIndex}`;

    try {
      let response;
      const endpoint = test.endpoint;

      if (test.method === 'GET') {
        response = await fetch(`http://localhost:8000${endpoint}`);
      } else {
        response = await fetch(`http://localhost:8000${endpoint}`, {
          method: test.method,
          headers: {
            'Content-Type': 'application/json',
          },
          body: test.payload ? JSON.stringify(test.payload) : undefined,
        });
      }

      const data = await response.json();
      
      const result: TestResult = {
        success: response.ok,
        message: response.ok ? 'Test passed successfully' : data.detail || 'Test failed',
        data: data,
        timestamp: new Date().toISOString()
      };

      setGlobalResults(prev => ({
        ...prev,
        [testKey]: result
      }));

    } catch (error) {
      const result: TestResult = {
        success: false,
        message: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString()
      };

      setGlobalResults(prev => ({
        ...prev,
        [testKey]: result
      }));
    }

    newSections[sectionIndex].tests[testIndex].running = false;
    setTestSections(newSections);
  };

  const runAllTests = async () => {
    setRunningAll(true);
    
    for (let sectionIndex = 0; sectionIndex < testSections.length; sectionIndex++) {
      for (let testIndex = 0; testIndex < testSections[sectionIndex].tests.length; testIndex++) {
        await runTest(sectionIndex, testIndex);
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 500));
      }
    }
    
    setRunningAll(false);
  };

  const toggleSection = (sectionIndex: number) => {
    const newSections = [...testSections];
    newSections[sectionIndex].expanded = !newSections[sectionIndex].expanded;
    setTestSections(newSections);
  };

  const getTestResult = (sectionIndex: number, testIndex: number) => {
    return globalResults[`${sectionIndex}-${testIndex}`];
  };

  const getOverallStats = () => {
    const totalTests = testSections.reduce((acc, section) => acc + section.tests.length, 0);
    const completedTests = Object.keys(globalResults).length;
    const passedTests = Object.values(globalResults).filter(result => result.success).length;
    const failedTests = completedTests - passedTests;

    return { totalTests, completedTests, passedTests, failedTests };
  };

  const stats = getOverallStats();

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                🚀 ModelShip Comprehensive Test Suite
              </h1>
              <p className="text-gray-600 mt-2">
                Complete testing interface for all Phase 1-3 features
              </p>
            </div>
            <button
              onClick={runAllTests}
              disabled={runningAll}
              className={`px-6 py-3 rounded-lg font-medium ${
                runningAll
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700'
              } text-white transition-colors`}
            >
              {runningAll ? '🔄 Running All Tests...' : '▶️ Run All Tests'}
            </button>
          </div>

          {/* Stats Dashboard */}
          <div className="grid grid-cols-4 gap-4 mt-6">
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{stats.totalTests}</div>
              <div className="text-sm text-blue-700">Total Tests</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{stats.passedTests}</div>
              <div className="text-sm text-green-700">Passed</div>
            </div>
            <div className="bg-red-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-red-600">{stats.failedTests}</div>
              <div className="text-sm text-red-700">Failed</div>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-gray-600">{stats.completedTests}</div>
              <div className="text-sm text-gray-700">Completed</div>
            </div>
          </div>
        </div>

        {/* Test Sections */}
        {testSections.map((section, sectionIndex) => (
          <div key={sectionIndex} className="bg-white rounded-lg shadow-sm mb-6">
            {/* Section Header */}
            <div
              className="p-6 border-b border-gray-200 cursor-pointer hover:bg-gray-50"
              onClick={() => toggleSection(sectionIndex)}
            >
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-gray-900">
                  {section.name}
                </h2>
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-500">
                    {section.tests.length} tests
                  </span>
                  <span className="text-lg">
                    {section.expanded ? '▼' : '▶️'}
                  </span>
                </div>
              </div>
            </div>

            {/* Test Cases */}
            {section.expanded && (
              <div className="p-6">
                <div className="grid gap-4">
                  {section.tests.map((test, testIndex) => {
                    const result = getTestResult(sectionIndex, testIndex);
                    
                    return (
                      <div
                        key={testIndex}
                        className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-3">
                              <h3 className="font-medium text-gray-900">
                                {test.name}
                              </h3>
                              <span className={`px-2 py-1 text-xs rounded-full ${
                                test.method === 'GET' ? 'bg-green-100 text-green-800' :
                                test.method === 'POST' ? 'bg-blue-100 text-blue-800' :
                                test.method === 'PUT' ? 'bg-yellow-100 text-yellow-800' :
                                'bg-red-100 text-red-800'
                              }`}>
                                {test.method}
                              </span>
                              {result && (
                                <span className={`px-2 py-1 text-xs rounded-full ${
                                  result.success ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                                }`}>
                                  {result.success ? '✅ PASS' : '❌ FAIL'}
                                </span>
                              )}
                            </div>
                            <p className="text-sm text-gray-600 mt-1">
                              {test.description}
                            </p>
                            <code className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded mt-2 inline-block">
                              {test.endpoint}
                            </code>
                          </div>
                          
                          <button
                            onClick={() => runTest(sectionIndex, testIndex)}
                            disabled={test.running}
                            className={`px-4 py-2 rounded-lg text-sm font-medium ${
                              test.running
                                ? 'bg-gray-400 cursor-not-allowed'
                                : result?.success
                                ? 'bg-green-600 hover:bg-green-700'
                                : result
                                ? 'bg-red-600 hover:bg-red-700'
                                : 'bg-blue-600 hover:bg-blue-700'
                            } text-white transition-colors`}
                          >
                            {test.running ? '🔄 Running...' : '▶️ Run Test'}
                          </button>
                        </div>

                        {/* Test Payload */}
                        {test.payload && (
                          <div className="mt-4">
                            <details className="group">
                              <summary className="cursor-pointer text-sm font-medium text-gray-700 hover:text-gray-900">
                                📝 Request Payload
                              </summary>
                              <pre className="mt-2 text-xs bg-gray-100 p-3 rounded overflow-x-auto">
                                {JSON.stringify(test.payload, null, 2)}
                              </pre>
                            </details>
                          </div>
                        )}

                        {/* Test Result */}
                        {result && (
                          <div className="mt-4">
                            <div className={`p-3 rounded-lg ${
                              result.success ? 'bg-green-50 border border-green-200' : 'bg-red-50 border border-red-200'
                            }`}>
                              <div className="flex items-center justify-between">
                                <span className={`font-medium ${
                                  result.success ? 'text-green-800' : 'text-red-800'
                                }`}>
                                  {result.message}
                                </span>
                                <span className="text-xs text-gray-500">
                                  {new Date(result.timestamp).toLocaleTimeString()}
                                </span>
                              </div>
                              
                              {result.data && (
                                <details className="mt-2">
                                  <summary className="cursor-pointer text-sm font-medium text-gray-700">
                                    📊 Response Data
                                  </summary>
                                  <pre className="mt-2 text-xs bg-white p-3 rounded border overflow-x-auto max-h-40 overflow-y-auto">
                                    {JSON.stringify(result.data, null, 2)}
                                  </pre>
                                </details>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Footer */}
        <div className="bg-white rounded-lg shadow-sm p-6 mt-8">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              🎯 ModelShip Test Suite Complete
            </h3>
            <p className="text-gray-600">
              This comprehensive test suite covers all features from Phase 1 (Core Auto-Labeling) 
              through Phase 3 (Industry Verticals & Advanced Features). Use this interface to 
              validate the complete functionality of the ModelShip platform.
            </p>
            <div className="mt-4 flex justify-center space-x-4">
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800">
                Phase 1: Core Platform ✅
              </span>
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-green-100 text-green-800">
                Phase 2: Advanced QA ✅
              </span>
              <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-purple-100 text-purple-800">
                Phase 3: Enterprise Features ✅
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComprehensiveTestSuite; 