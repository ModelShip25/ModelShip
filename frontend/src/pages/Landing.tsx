import React from 'react';
import { Link } from 'react-router-dom';
import { 
  CloudArrowUpIcon,
  PhotoIcon, 
  DocumentTextIcon, 
  ChartBarIcon,
  ArrowRightIcon,
  SparklesIcon,
  ClockIcon,
  ShieldCheckIcon,
  CpuChipIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-700 text-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-5xl font-bold mb-6">
              AI-Powered Auto-Labeling Platform
            </h1>
            <p className="text-xl mb-8 max-w-3xl mx-auto">
              Upload your images or text files and get professional-quality labels with AI classification and human review.
            </p>
            <div className="flex justify-center space-x-4">
              <Link
                to="/dashboard"
                className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Get Started Free
              </Link>
              <Link
                to="/login"
                className="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition-colors"
              >
                Sign In
              </Link>
            </div>
            <p className="text-sm text-blue-100 mt-4">
              No credit card required • Start labeling in minutes
            </p>
          </div>
        </div>
      </div>

      {/* How It Works */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-600">
              From upload to export in 3 simple steps
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="text-center">
              <div className="bg-blue-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <CloudArrowUpIcon className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                1. Upload Your Files
              </h3>
              <p className="text-gray-600">
                Drag and drop your images or text files. We support JPG, PNG, GIF, TXT, and CSV formats.
              </p>
            </div>

            {/* Step 2 */}
            <div className="text-center">
              <div className="bg-green-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <CpuChipIcon className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                2. AI Classification
              </h3>
              <p className="text-gray-600">
                Our advanced AI models automatically classify your content with high accuracy and confidence scores.
              </p>
            </div>

            {/* Step 3 */}
            <div className="text-center">
              <div className="bg-purple-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <ArrowRightIcon className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                3. Review & Export
              </h3>
              <p className="text-gray-600">
                Review results, make corrections, and export in your preferred format (JSON, CSV, COCO).
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Powerful Features
            </h2>
            <p className="text-lg text-gray-600">
              Everything you need for professional data labeling
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Image Classification */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
              <PhotoIcon className="w-8 h-8 text-purple-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Image Classification</h3>
              <p className="text-gray-600 text-sm">
                Advanced computer vision models for object detection, scene classification, and content moderation.
              </p>
            </div>

            {/* Text Classification */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
              <DocumentTextIcon className="w-8 h-8 text-green-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Text Classification</h3>
              <p className="text-gray-600 text-sm">
                Sentiment analysis, topic classification, spam detection, and custom text categorization.
              </p>
            </div>

            {/* Human Review */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
              <CheckCircleIcon className="w-8 h-8 text-blue-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Human Review</h3>
              <p className="text-gray-600 text-sm">
                Built-in review system to validate AI predictions and ensure the highest quality labels.
              </p>
            </div>

            {/* Batch Processing */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
              <ClockIcon className="w-8 h-8 text-yellow-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Batch Processing</h3>
              <p className="text-gray-600 text-sm">
                Process hundreds of files at once with automated workflows and progress tracking.
              </p>
            </div>

            {/* Multiple Formats */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
              <ChartBarIcon className="w-8 h-8 text-indigo-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Export Flexibility</h3>
              <p className="text-gray-600 text-sm">
                Export in JSON, CSV, COCO, YOLO, or custom formats compatible with your ML pipeline.
              </p>
            </div>

            {/* Security */}
            <div className="bg-white rounded-lg shadow-sm p-6 border">
              <ShieldCheckIcon className="w-8 h-8 text-red-500 mb-4" />
              <h3 className="text-lg font-semibold mb-2">Secure & Private</h3>
              <p className="text-gray-600 text-sm">
                Enterprise-grade security with encrypted storage and complete data privacy protection.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Pricing Preview */}
      <div className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Simple Pricing
            </h2>
            <p className="text-lg text-gray-600">
              Start free, scale as you grow
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {/* Free Tier */}
            <div className="bg-gray-50 rounded-lg p-6 border">
              <div className="text-center">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Free</h3>
                <div className="text-3xl font-bold text-gray-900 mb-4">$0</div>
                <ul className="text-sm text-gray-600 space-y-2 mb-6">
                  <li>• 100 classifications/month</li>
                  <li>• Basic AI models</li>
                  <li>• Standard export formats</li>
                  <li>• Community support</li>
                </ul>
                <Link
                  to="/dashboard"
                  className="block w-full bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium text-center"
                >
                  Get Started
                </Link>
              </div>
            </div>

            {/* Pro Tier */}
            <div className="bg-blue-50 rounded-lg p-6 border-2 border-blue-200 relative">
              <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                <span className="bg-blue-600 text-white px-3 py-1 rounded-full text-xs font-medium">
                  Most Popular
                </span>
              </div>
              <div className="text-center">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Pro</h3>
                <div className="text-3xl font-bold text-gray-900 mb-4">$29<span className="text-lg text-gray-600">/mo</span></div>
                <ul className="text-sm text-gray-600 space-y-2 mb-6">
                  <li>• 10,000 classifications/month</li>
                  <li>• Advanced AI models</li>
                  <li>• All export formats</li>
                  <li>• Priority support</li>
                  <li>• API access</li>
                </ul>
                <Link
                  to="/login"
                  className="block w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium text-center"
                >
                  Start Pro Trial
                </Link>
              </div>
            </div>

            {/* Enterprise Tier */}
            <div className="bg-gray-50 rounded-lg p-6 border">
              <div className="text-center">
                <h3 className="text-xl font-semibold text-gray-900 mb-2">Enterprise</h3>
                <div className="text-3xl font-bold text-gray-900 mb-4">Custom</div>
                <ul className="text-sm text-gray-600 space-y-2 mb-6">
                  <li>• Unlimited classifications</li>
                  <li>• Custom AI models</li>
                  <li>• White-label solution</li>
                  <li>• Dedicated support</li>
                  <li>• On-premise deployment</li>
                </ul>
                <Link
                  to="/login"
                  className="block w-full bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg font-medium text-center"
                >
                  Contact Sales
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-16 bg-blue-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Start Labeling?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Join thousands of teams using ModelShip to accelerate their AI projects
          </p>
          <div className="flex justify-center space-x-4">
            <Link
              to="/dashboard"
              className="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold hover:bg-gray-100 transition-colors inline-flex items-center"
            >
              <SparklesIcon className="w-5 h-5 mr-2" />
              Start Free Today
            </Link>
            <Link
              to="/login"
              className="border-2 border-white text-white px-8 py-3 rounded-lg font-semibold hover:bg-white hover:text-blue-600 transition-colors"
            >
              View Demo
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing; 