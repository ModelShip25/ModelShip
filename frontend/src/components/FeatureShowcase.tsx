import React, { useState } from 'react';

interface Feature {
  icon: string;
  title: string;
  description: string;
  phase: 1 | 2 | 3;
  category: string;
  endpoint: string;
  status: 'implemented' | 'testing' | 'demo';
}

const FeatureShowcase: React.FC = () => {
  const [selectedPhase, setSelectedPhase] = useState<number | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);

  const features: Feature[] = [
    // Phase 1 Features
    {
      icon: "📁",
      title: "Project Management",
      description: "Create, organize, and manage annotation projects with team collaboration",
      phase: 1,
      category: "Core Platform",
      endpoint: "/api/projects",
      status: "implemented"
    },
    {
      icon: "📤",
      title: "File Upload & Management",
      description: "Drag-and-drop upload for images, text, and documents with batch processing",
      phase: 1,
      category: "Core Platform",
      endpoint: "/api/upload",
      status: "implemented"
    },
    {
      icon: "🖼️",
      title: "Image Classification",
      description: "AI-powered image classification with confidence scoring and batch processing",
      phase: 1,
      category: "ML Services",
      endpoint: "/api/classify/image",
      status: "implemented"
    },
    {
      icon: "📝",
      title: "Text Classification",
      description: "Natural language processing for sentiment analysis and topic classification",
      phase: 1,
      category: "ML Services",
      endpoint: "/api/classify/text",
      status: "implemented"
    },
    {
      icon: "🎯",
      title: "Object Detection",
      description: "YOLO-based object detection with bounding box annotations",
      phase: 1,
      category: "ML Services",
      endpoint: "/api/object-detection/detect",
      status: "implemented"
    },
    {
      icon: "👥",
      title: "Human Review System",
      description: "Collaborative review interface with approve/reject/modify workflows",
      phase: 1,
      category: "Quality Control",
      endpoint: "/api/review/submit",
      status: "implemented"
    },
    {
      icon: "📊",
      title: "Multi-Format Export",
      description: "Export annotations in COCO, YOLO, JSON, CSV formats",
      phase: 1,
      category: "Export",
      endpoint: "/api/export/coco",
      status: "implemented"
    },

    // Phase 2 Features
    {
      icon: "🔧",
      title: "Advanced Export Formats",
      description: "Custom export templates and advanced format options",
      phase: 2,
      category: "Export",
      endpoint: "/api/advanced-export/export",
      status: "implemented"
    },
    {
      icon: "📈",
      title: "Quality Control Dashboard",
      description: "Real-time annotation metrics and quality monitoring",
      phase: 2,
      category: "Analytics",
      endpoint: "/api/annotation-quality/dashboard",
      status: "implemented"
    },
    {
      icon: "🏆",
      title: "Gold Standard Testing",
      description: "Benchmark annotator performance against verified datasets",
      phase: 2,
      category: "Quality Control",
      endpoint: "/api/gold-standard/test",
      status: "implemented"
    },
    {
      icon: "🔄",
      title: "Data Versioning",
      description: "Version control for datasets with rollback capabilities",
      phase: 2,
      category: "Data Management",
      endpoint: "/api/data-versioning/create-version",
      status: "implemented"
    },
    {
      icon: "🏷️",
      title: "Label Schema Management",
      description: "Dynamic label schema creation and management",
      phase: 2,
      category: "Data Management",
      endpoint: "/api/label-schema/create",
      status: "implemented"
    },
    {
      icon: "📊",
      title: "Analytics Dashboard",
      description: "Comprehensive project analytics and performance insights",
      phase: 2,
      category: "Analytics",
      endpoint: "/api/analytics/dashboard",
      status: "implemented"
    },

    // Phase 3 Features
    {
      icon: "🏥",
      title: "Industry Vertical Templates",
      description: "Pre-built workflows for Healthcare, Legal, Retail, Industrial sectors",
      phase: 3,
      category: "Enterprise",
      endpoint: "/api/vertical-templates/create-project",
      status: "implemented"
    },
    {
      icon: "👨‍⚕️",
      title: "Expert-In-Loop Service",
      description: "Premium expert reviewer pool with domain-specific expertise",
      phase: 3,
      category: "Expert Services",
      endpoint: "/api/expert-in-loop/request-review",
      status: "implemented"
    },
    {
      icon: "🤖",
      title: "ML-Assisted Pre-labeling",
      description: "Lightweight models that reduce annotation time by 60-80%",
      phase: 3,
      category: "ML Services",
      endpoint: "/api/ml-prelabeling/create-model",
      status: "implemented"
    },
    {
      icon: "🤝",
      title: "Task Consensus Controls",
      description: "Configurable multi-annotator consensus with quality metrics",
      phase: 3,
      category: "Quality Control",
      endpoint: "/api/consensus/create-task",
      status: "implemented"
    },
    {
      icon: "⚖️",
      title: "Bias & Fairness Reports",
      description: "Automated bias detection and fairness monitoring",
      phase: 3,
      category: "Analytics",
      endpoint: "/api/bias-fairness/generate-report",
      status: "implemented"
    },
    {
      icon: "🔒",
      title: "Security & Compliance",
      description: "HIPAA, GDPR, SOC2 compliance with enterprise security",
      phase: 3,
      category: "Enterprise",
      endpoint: "/api/security-compliance/assess",
      status: "implemented"
    },
    {
      icon: "🎯",
      title: "Active Edge-Case Detection",
      description: "Smart identification and routing of challenging annotation cases",
      phase: 3,
      category: "ML Services",
      endpoint: "/api/active-learning/edge-cases",
      status: "implemented"
    },
    {
      icon: "🔄",
      title: "Feedback Loop Integration",
      description: "Continuous model improvement through user feedback",
      phase: 3,
      category: "ML Services",
      endpoint: "/api/feedback/submit",
      status: "implemented"
    },
    {
      icon: "🏗️",
      title: "Taxonomy Management",
      description: "Visual taxonomy editor with hierarchical category support",
      phase: 3,
      category: "Data Management",
      endpoint: "/api/taxonomy/update",
      status: "implemented"
    }
  ];

  const categories = Array.from(new Set(features.map(f => f.category)));
  
  const filteredFeatures = features.filter(feature => {
    const phaseMatch = selectedPhase === null || feature.phase === selectedPhase;
    const categoryMatch = selectedCategory === null || feature.category === selectedCategory;
    return phaseMatch && categoryMatch;
  });

  const getPhaseColor = (phase: number) => {
    switch (phase) {
      case 1: return 'bg-green-100 text-green-800 border-green-200';
      case 2: return 'bg-blue-100 text-blue-800 border-blue-200';
      case 3: return 'bg-purple-100 text-purple-800 border-purple-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getCategoryColor = (category: string) => {
    const colors: { [key: string]: string } = {
      'Core Platform': 'bg-green-50 border-green-200',
      'ML Services': 'bg-blue-50 border-blue-200',
      'Quality Control': 'bg-yellow-50 border-yellow-200',
      'Export': 'bg-purple-50 border-purple-200',
      'Analytics': 'bg-indigo-50 border-indigo-200',
      'Data Management': 'bg-teal-50 border-teal-200',
      'Enterprise': 'bg-red-50 border-red-200',
      'Expert Services': 'bg-orange-50 border-orange-200'
    };
    return colors[category] || 'bg-gray-50 border-gray-200';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">
          🚀 ModelShip Feature Showcase
        </h2>
        <p className="text-gray-600 mb-6">
          Comprehensive overview of all implemented features across three development phases
        </p>

        {/* Filters */}
        <div className="flex flex-wrap gap-4">
          {/* Phase Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Phase:</label>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedPhase(null)}
                className={`px-3 py-1 rounded-full text-sm font-medium border ${
                  selectedPhase === null ? 'bg-gray-800 text-white border-gray-800' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                All Phases
              </button>
              {[1, 2, 3].map(phase => (
                <button
                  key={phase}
                  onClick={() => setSelectedPhase(phase)}
                  className={`px-3 py-1 rounded-full text-sm font-medium border ${
                    selectedPhase === phase ? getPhaseColor(phase) : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  Phase {phase}
                </button>
              ))}
            </div>
          </div>

          {/* Category Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Category:</label>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setSelectedCategory(null)}
                className={`px-3 py-1 rounded-full text-sm font-medium border ${
                  selectedCategory === null ? 'bg-gray-800 text-white border-gray-800' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                }`}
              >
                All Categories
              </button>
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-3 py-1 rounded-full text-sm font-medium border ${
                    selectedCategory === category ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
                  }`}
                >
                  {category}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Feature Grid */}
      <div className="p-6">
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredFeatures.map((feature, index) => (
            <div
              key={index}
              className={`rounded-lg border-2 p-6 hover:shadow-md transition-shadow ${getCategoryColor(feature.category)}`}
            >
              {/* Feature Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="text-3xl">{feature.icon}</div>
                <div className="flex space-x-2">
                  <span className={`px-2 py-1 text-xs rounded-full border ${getPhaseColor(feature.phase)}`}>
                    Phase {feature.phase}
                  </span>
                  <span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800 border border-green-200">
                    ✅ Ready
                  </span>
                </div>
              </div>

              {/* Feature Content */}
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-gray-600 mb-4">
                {feature.description}
              </p>

              {/* Feature Details */}
              <div className="space-y-2">
                <div className="flex items-center text-xs text-gray-500">
                  <span className="font-medium">Category:</span>
                  <span className="ml-1">{feature.category}</span>
                </div>
                <div className="flex items-center text-xs text-gray-500">
                  <span className="font-medium">Endpoint:</span>
                  <code className="ml-1 bg-gray-100 px-1 py-0.5 rounded">{feature.endpoint}</code>
                </div>
              </div>

              {/* Test Button */}
              <button className="mt-4 w-full bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
                🧪 Test Feature
              </button>
            </div>
          ))}
        </div>

        {/* Summary Stats */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200">
            <div className="text-2xl font-bold text-green-600">
              {features.filter(f => f.phase === 1).length}
            </div>
            <div className="text-sm text-green-700">Phase 1 Features</div>
          </div>
          <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200">
            <div className="text-2xl font-bold text-blue-600">
              {features.filter(f => f.phase === 2).length}
            </div>
            <div className="text-sm text-blue-700">Phase 2 Features</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200">
            <div className="text-2xl font-bold text-purple-600">
              {features.filter(f => f.phase === 3).length}
            </div>
            <div className="text-sm text-purple-700">Phase 3 Features</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg border border-gray-200">
            <div className="text-2xl font-bold text-gray-600">
              {features.length}
            </div>
            <div className="text-sm text-gray-700">Total Features</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeatureShowcase; 