import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  PlusIcon, 
  FolderIcon, 
  DocumentTextIcon, 
  PhotoIcon,
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import { simpleProjectsAPI } from '../services/api';

interface Project {
  project_id: number;
  name: string;
  description: string;
  project_type: string;
  status: string;
  created_at: string;
  total_items: number;
  labeled_items: number;
  reviewed_items: number;
  approved_items: number;
  default_labels: string[];
}

interface DashboardStats {
  total_projects: number;
  total_files: number;
  total_classified: number;
  total_reviewed: number;
  accuracy_score: number;
}

const Dashboard: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      
      // Fetch projects using new simple projects API
      const projectsResponse = await simpleProjectsAPI.getProjects();
      console.log('Projects response:', projectsResponse);
      
      if (projectsResponse.success) {
        const projectsData = projectsResponse.projects || [];
        setProjects(projectsData);
        
        // Calculate dashboard stats from projects
        const totalProjects = projectsData.length;
        const totalFiles = projectsData.reduce((sum: number, p: Project) => sum + (p.total_items || 0), 0);
        const totalClassified = projectsData.reduce((sum: number, p: Project) => sum + (p.labeled_items || 0), 0);
        const totalReviewed = projectsData.reduce((sum: number, p: Project) => sum + (p.reviewed_items || 0), 0);
        
        setStats({
          total_projects: totalProjects,
          total_files: totalFiles,
          total_classified: totalClassified,
          total_reviewed: totalReviewed,
          accuracy_score: totalClassified > 0 ? Math.round((totalReviewed / totalClassified) * 100) : 0
        });
      } else {
        throw new Error('Failed to fetch projects');
      }

      setError(null);
    } catch (err: any) {
      setError(`Failed to load dashboard data: ${err.message}`);
      console.error('Dashboard fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateTestProject = async () => {
    try {
      setLoading(true);
      const response = await simpleProjectsAPI.createTestProject();
      
      if (response.success) {
        // Refresh the dashboard
        await fetchDashboardData();
      } else {
        setError('Failed to create test project');
      }
    } catch (err: any) {
      setError(`Failed to create test project: ${err.message}`);
      console.error('Test project creation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'paused':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      default:
        return <ClockIcon className="w-5 h-5 text-blue-500" />;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'image_classification':
      case 'object_detection':
        return <PhotoIcon className="w-5 h-5 text-purple-500" />;
      case 'text_classification':
        return <DocumentTextIcon className="w-5 h-5 text-green-500" />;
      default:
        return <FolderIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
              <p className="text-gray-600 mt-1">Manage your AI labeling projects</p>
            </div>
                        <div className="flex space-x-3">
              <button
                onClick={handleCreateTestProject}
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg font-medium"
                disabled={loading}
              >
                Quick Test Project
              </button>
              <Link
                to="/upload"
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium flex items-center"
              >
                <PlusIcon className="w-5 h-5 mr-2" />
                New Project
              </Link>
            </div>
          </div>
        </div>

        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <FolderIcon className="w-8 h-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Projects</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_projects}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <DocumentTextIcon className="w-8 h-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Files</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_files}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <ChartBarIcon className="w-8 h-8 text-purple-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Classified</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_classified}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <CheckCircleIcon className="w-8 h-8 text-emerald-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Reviewed</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.total_reviewed}</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center">
                <div className="w-8 h-8 bg-gradient-to-r from-green-400 to-blue-500 rounded-full flex items-center justify-center">
                  <span className="text-white text-sm font-bold">%</span>
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Accuracy</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.accuracy_score}%</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-8">
            <p className="text-red-700">{error}</p>
            <button
              onClick={fetchDashboardData}
              className="mt-2 text-red-600 hover:text-red-500 font-medium"
            >
              Try again
            </button>
          </div>
        )}

        {/* Projects Grid */}
        <div>
          <h2 className="text-xl font-semibold text-gray-900 mb-6">Your Projects</h2>
          
          {projects.length === 0 ? (
            <div className="bg-white rounded-lg shadow text-center py-12">
              <FolderIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No projects yet</h3>
              <p className="text-gray-600 mb-6">Create your first AI labeling project to get started</p>
              <div className="flex justify-center space-x-3">
                <button
                  onClick={handleCreateTestProject}
                  className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg font-medium"
                  disabled={loading}
                >
                  Create Test Project
                </button>
                <Link
                  to="/upload"
                  className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium inline-flex items-center"
                >
                  <PlusIcon className="w-5 h-5 mr-2" />
                  Create New Project
                </Link>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {projects.map((project) => (
                <div key={project.project_id} className="bg-white rounded-lg shadow hover:shadow-md transition-shadow">
                  <div className="p-6">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center">
                        {getTypeIcon(project.project_type)}
                        <h3 className="ml-2 text-lg font-semibold text-gray-900 truncate">
                          {project.name}
                        </h3>
                      </div>
                      {getStatusIcon(project.status)}
                    </div>
                    
                    <p className="text-gray-600 text-sm mb-4 line-clamp-2">
                      {project.description || 'No description'}
                    </p>
                    
                    <div className="space-y-2 mb-4">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Total Items:</span>
                        <span className="font-medium">{project.total_items}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Labeled:</span>
                        <span className="font-medium">{project.labeled_items}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Reviewed:</span>
                        <span className="font-medium">{project.reviewed_items}</span>
                      </div>
                    </div>
                    
                    {/* Progress Bar */}
                    <div className="mb-4">
                      <div className="flex justify-between text-sm text-gray-600 mb-1">
                        <span>Progress</span>
                        <span>
                          {project.total_items > 0 
                            ? Math.round((project.labeled_items / project.total_items) * 100)
                            : 0}%
                        </span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full" 
                          style={{ 
                            width: `${project.total_items > 0 
                              ? (project.labeled_items / project.total_items) * 100 
                              : 0}%` 
                          }}
                        ></div>
                      </div>
                    </div>
                    
                    {/* Default Labels */}
                    {project.default_labels && project.default_labels.length > 0 && (
                      <div className="mb-4">
                        <span className="text-sm text-gray-500">Labels: </span>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {project.default_labels.slice(0, 3).map((label, index) => (
                            <span 
                              key={index}
                              className="inline-block bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded"
                            >
                              {label}
                            </span>
                          ))}
                          {project.default_labels.length > 3 && (
                            <span className="text-xs text-gray-500">
                              +{project.default_labels.length - 3} more
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                    
                    <div className="flex space-x-2">
                      <Link
                        to={`/classification?project=${project.project_id}`}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm text-center font-medium"
                      >
                        Classify
                      </Link>
                      <Link
                        to={`/results?project=${project.project_id}`}
                        className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm text-center font-medium"
                      >
                        Results
                      </Link>
                    </div>
                    
                    <p className="text-xs text-gray-500 mt-3">
                      Created {new Date(project.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 