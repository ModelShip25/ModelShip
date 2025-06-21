import React, { useState, useEffect } from 'react';
import {
  ChartBarIcon,
  FolderIcon,
  DocumentDuplicateIcon,
  EyeIcon,
  CloudArrowUpIcon,
  ArrowUpIcon,
  ArrowDownIcon,
} from '@heroicons/react/24/outline';
import { analyticsAPI, projectsAPI } from '../services/api';
import { useAuth } from '../contexts/AuthContext';

interface StatCard {
  name: string;
  value: string;
  change: string;
  changeType: 'increase' | 'decrease';
  icon: React.ComponentType<any>;
}

const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [stats, setStats] = useState<StatCard[]>([]);
  const [recentProjects, setRecentProjects] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load user analytics
      const analyticsData = await analyticsAPI.getUserAnalytics('7d');
      
      // Load recent projects
      const projectsData = await projectsAPI.getProjects({ status: 'active' });
      setRecentProjects(projectsData.projects?.slice(0, 5) || []);
      
      // Prepare stats
      const statsData: StatCard[] = [
        {
          name: 'Total Projects',
          value: analyticsData.total_projects?.toString() || '0',
          change: '+12%',
          changeType: 'increase',
          icon: FolderIcon,
        },
        {
          name: 'Labels Generated',
          value: analyticsData.total_labels?.toLocaleString() || '0',
          change: '+19%',
          changeType: 'increase',
          icon: DocumentDuplicateIcon,
        },
        {
          name: 'Review Queue',
          value: analyticsData.pending_reviews?.toString() || '0',
          change: '-8%',
          changeType: 'decrease',
          icon: EyeIcon,
        },
        {
          name: 'Automation Rate',
          value: `${analyticsData.automation_rate || 0}%`,
          change: '+5%',
          changeType: 'increase',
          icon: ChartBarIcon,
        },
      ];
      
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      // Set default stats if API fails
      setStats([
        {
          name: 'Total Projects',
          value: '0',
          change: '+0%',
          changeType: 'increase',
          icon: FolderIcon,
        },
        {
          name: 'Labels Generated',
          value: '0',
          change: '+0%',
          changeType: 'increase',
          icon: DocumentDuplicateIcon,
        },
        {
          name: 'Review Queue',
          value: '0',
          change: '+0%',
          changeType: 'increase',
          icon: EyeIcon,
        },
        {
          name: 'Automation Rate',
          value: '0%',
          change: '+0%',
          changeType: 'increase',
          icon: ChartBarIcon,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const quickActions = [
    {
      name: 'Upload Data',
      description: 'Upload images or text files for labeling',
      href: '/upload',
      icon: CloudArrowUpIcon,
      color: 'bg-blue-500',
    },
    {
      name: 'Create Project',
      description: 'Start a new labeling project',
      href: '/projects/new',
      icon: FolderIcon,
      color: 'bg-green-500',
    },
    {
      name: 'Review Labels',
      description: 'Review and approve auto-generated labels',
      href: '/review',
      icon: EyeIcon,
      color: 'bg-purple-500',
    },
    {
      name: 'View Analytics',
      description: 'Analyze your labeling performance',
      href: '/analytics',
      icon: ChartBarIcon,
      color: 'bg-orange-500',
    },
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Welcome back, {user?.full_name}!
              </h1>
              <p className="mt-1 text-sm text-gray-500">
                Here's what's happening with your data labeling projects today.
              </p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-500">Credits Remaining</p>
              <p className="text-2xl font-bold text-primary-600">
                {user?.credits_remaining?.toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((item) => (
          <div
            key={item.name}
            className="bg-white overflow-hidden shadow rounded-lg"
          >
            <div className="p-5">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <item.icon className="h-6 w-6 text-gray-400" aria-hidden="true" />
                </div>
                <div className="ml-5 w-0 flex-1">
                  <dl>
                    <dt className="text-sm font-medium text-gray-500 truncate">
                      {item.name}
                    </dt>
                    <dd>
                      <div className="text-lg font-medium text-gray-900">
                        {item.value}
                      </div>
                    </dd>
                  </dl>
                </div>
              </div>
            </div>
            <div className="bg-gray-50 px-5 py-3">
              <div className="text-sm">
                <div className="flex items-center">
                  {item.changeType === 'increase' ? (
                    <ArrowUpIcon className="h-4 w-4 text-green-500 mr-1" />
                  ) : (
                    <ArrowDownIcon className="h-4 w-4 text-red-500 mr-1" />
                  )}
                  <span
                    className={
                      item.changeType === 'increase'
                        ? 'text-green-600'
                        : 'text-red-600'
                    }
                  >
                    {item.change}
                  </span>
                  <span className="text-gray-500 ml-1">from last week</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
            Quick Actions
          </h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {quickActions.map((action) => (
              <a
                key={action.name}
                href={action.href}
                className="relative group bg-white p-6 focus-within:ring-2 focus-within:ring-inset focus-within:ring-primary-500 rounded-lg border border-gray-200 hover:border-gray-300 transition-colors"
              >
                <div>
                  <span
                    className={`${action.color} rounded-lg inline-flex p-3 ring-4 ring-white`}
                  >
                    <action.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </span>
                </div>
                <div className="mt-4">
                  <h3 className="text-lg font-medium text-gray-900">
                    <span className="absolute inset-0" aria-hidden="true" />
                    {action.name}
                  </h3>
                  <p className="mt-2 text-sm text-gray-500">
                    {action.description}
                  </p>
                </div>
                <span
                  className="pointer-events-none absolute top-6 right-6 text-gray-300 group-hover:text-gray-400"
                  aria-hidden="true"
                >
                  <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 24 24">
                    <path d="m11.293 17.293 1.414 1.414L19.414 12l-6.707-6.707-1.414 1.414L15.586 11H5v2h10.586l-4.293 4.293z" />
                  </svg>
                </span>
              </a>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Projects */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg leading-6 font-medium text-gray-900">
              Recent Projects
            </h3>
            <a
              href="/projects"
              className="text-sm font-medium text-primary-600 hover:text-primary-500"
            >
              View all
            </a>
          </div>
          
          {recentProjects.length > 0 ? (
            <div className="space-y-3">
              {recentProjects.map((project: any) => (
                <div
                  key={project.id}
                  className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
                >
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
                      <FolderIcon className="w-5 h-5 text-primary-600" />
                    </div>
                    <div className="ml-3">
                      <p className="text-sm font-medium text-gray-900">
                        {project.name}
                      </p>
                      <p className="text-xs text-gray-500">
                        {project.project_type} • {project.status}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-900">
                      {project.labeled_items || 0} / {project.total_items || 0}
                    </p>
                    <p className="text-xs text-gray-500">labeled</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-6">
              <FolderIcon className="mx-auto h-12 w-12 text-gray-400" />
              <h3 className="mt-2 text-sm font-medium text-gray-900">No projects</h3>
              <p className="mt-1 text-sm text-gray-500">
                Get started by creating a new project.
              </p>
              <div className="mt-6">
                <a
                  href="/projects/new"
                  className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
                >
                  Create Project
                </a>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 