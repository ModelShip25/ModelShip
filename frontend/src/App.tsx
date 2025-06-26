import React, { useState } from 'react';
import './App.css';

// Import test components
import ImageClassificationTest from './components/ImageClassificationTest';
import TextClassificationTest from './components/TextClassificationTest';
import ObjectDetectionTest from './components/ObjectDetectionTest';
import JobManagementTest from './components/JobManagementTest';
import ReviewSystemTest from './components/ReviewSystemTest';
import ExportSystemTest from './components/ExportSystemTest';
import HealthCheck from './components/HealthCheck';

function App() {
  const [activeTab, setActiveTab] = useState('health');

  const tabs = [
    { id: 'health', name: 'Health Check', component: HealthCheck },
    { id: 'image-classification', name: 'Image Classification', component: ImageClassificationTest },
    { id: 'text-classification', name: 'Text Classification', component: TextClassificationTest },
    { id: 'object-detection', name: 'Object Detection', component: ObjectDetectionTest },
    { id: 'job-management', name: 'Job Management', component: JobManagementTest },
    { id: 'review-system', name: 'Review System', component: ReviewSystemTest },
    { id: 'export-system', name: 'Export System', component: ExportSystemTest }
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || HealthCheck;

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <h1 className="text-3xl font-bold text-gray-900">
              ModelShip API Testing Dashboard
            </h1>
            <div className="text-sm text-gray-500">
              Backend Feature Testing Interface
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8 overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.name}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-md p-6">
          <ActiveComponent />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            ModelShip Backend Testing Interface - Testing all {tabs.length} core feature categories
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
