import React from 'react';
import { Link } from 'react-router-dom';
import { 
  BeakerIcon, 
  PhotoIcon, 
  DocumentTextIcon, 
  ChartBarIcon,
  ArrowRightIcon,
  SparklesIcon,
  ClockIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <SparklesIcon className="h-8 w-8 text-blue-600 mr-2" />
              <h1 className="text-2xl font-bold text-gray-900">ModelShip</h1>
            </div>
            <div className="flex space-x-4">
              <Link
                to="/test"
                className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center"
              >
                <BeakerIcon className="h-5 w-5 mr-2" />
                Try Demo
              </Link>
              <Link
                to="/login"
                className="border border-blue-600 text-blue-600 px-4 py-2 rounded-lg hover:bg-blue-50 transition-colors"
              >
                Login
              </Link>
            </div>
          </div>
        </div>
      </header>

      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            AI-Powered Auto-Labeling Platform
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Accelerate your machine learning projects with intelligent data labeling. 
            Reduce manual work by 90% while maintaining high accuracy.
          </p>
          <div className="flex justify-center space-x-4">
            <Link
              to="/test"
              className="bg-blue-600 text-white px-8 py-4 rounded-lg text-lg font-medium hover:bg-blue-700 transition-colors flex items-center"
            >
              <BeakerIcon className="h-6 w-6 mr-2" />
              Test Our AI Models
              <ArrowRightIcon className="h-5 w-5 ml-2" />
            </Link>
            <button className="border border-gray-300 text-gray-700 px-8 py-4 rounded-lg text-lg font-medium hover:bg-gray-50 transition-colors">
              Watch Demo
            </button>
          </div>
        </div>
      </section>

      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Powerful Auto-Labeling Features
            </h2>
            <p className="text-lg text-gray-600">
              Everything you need to label your data efficiently and accurately
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="bg-blue-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <PhotoIcon className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Image Classification</h3>
              <p className="text-gray-600 mb-4">
                ResNet-50 model with 1000+ categories. Perfect for product catalogs, 
                medical imaging, and content moderation.
              </p>
              <ul className="text-sm text-gray-500 space-y-1">
                <li>• 76%+ accuracy</li>
                <li>• Sub-second processing</li>
                <li>• Confidence scoring</li>
              </ul>
            </div>

            <div className="text-center p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="bg-green-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <DocumentTextIcon className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Text Analysis</h3>
              <p className="text-gray-600 mb-4">
                Sentiment analysis, topic detection, spam filtering, and language 
                identification for all your text data.
              </p>
              <ul className="text-sm text-gray-500 space-y-1">
                <li>• Multi-language support</li>
                <li>• Custom categories</li>
                <li>• Real-time processing</li>
              </ul>
            </div>

            <div className="text-center p-6 rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
              <div className="bg-purple-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <ChartBarIcon className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Batch Processing</h3>
              <p className="text-gray-600 mb-4">
                Process thousands of files efficiently with our optimized batch 
                processing system and progress tracking.
              </p>
              <ul className="text-sm text-gray-500 space-y-1">
                <li>• Parallel processing</li>
                <li>• Progress monitoring</li>
                <li>• Error handling</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Why Choose ModelShip?
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-blue-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <ClockIcon className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">90% Faster</h3>
              <p className="text-gray-600">
                Reduce manual labeling time from days to hours with our intelligent auto-labeling
              </p>
            </div>

            <div className="text-center">
              <div className="bg-green-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <ShieldCheckIcon className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">High Accuracy</h3>
              <p className="text-gray-600">
                State-of-the-art models with confidence calibration for reliable results
              </p>
            </div>

            <div className="text-center">
              <div className="bg-purple-100 rounded-full p-3 w-16 h-16 mx-auto mb-4 flex items-center justify-center">
                <SparklesIcon className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Easy to Use</h3>
              <p className="text-gray-600">
                Simple drag-and-drop interface. No ML expertise required to get started
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-blue-600">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Try ModelShip?
          </h2>
          <p className="text-xl text-blue-100 mb-8">
            Test our AI models with your own data - no signup required
          </p>
          <Link
            to="/test"
            className="bg-white text-blue-600 px-8 py-4 rounded-lg text-lg font-medium hover:bg-gray-100 transition-colors inline-flex items-center"
          >
            <BeakerIcon className="h-6 w-6 mr-2" />
            Start Testing Now
            <ArrowRightIcon className="h-5 w-5 ml-2" />
          </Link>
        </div>
      </section>

      <footer className="bg-gray-900 text-white py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center mb-4">
                <SparklesIcon className="h-8 w-8 text-blue-400 mr-2" />
                <h3 className="text-xl font-bold">ModelShip</h3>
              </div>
              <p className="text-gray-400">
                AI-powered auto-labeling platform for accelerating machine learning projects.
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Product</h4>
              <ul className="space-y-2 text-gray-400">
                <li><Link to="/test" className="hover:text-white">Try Demo</Link></li>
                <li><button className="hover:text-white text-left">Features</button></li>
                <li><button className="hover:text-white text-left">Pricing</button></li>
                <li><button className="hover:text-white text-left">API Docs</button></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Company</h4>
              <ul className="space-y-2 text-gray-400">
                <li><button className="hover:text-white text-left">About</button></li>
                <li><button className="hover:text-white text-left">Blog</button></li>
                <li><button className="hover:text-white text-left">Careers</button></li>
                <li><button className="hover:text-white text-left">Contact</button></li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4">Support</h4>
              <ul className="space-y-2 text-gray-400">
                <li><button className="hover:text-white text-left">Help Center</button></li>
                <li><button className="hover:text-white text-left">Community</button></li>
                <li><button className="hover:text-white text-left">Status</button></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 mt-8 pt-8 text-center text-gray-400">
            <p>&copy; 2024 ModelShip. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing; 