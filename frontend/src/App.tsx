import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Layout from './components/Layout/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import TestClassification from './pages/TestClassification';
import TestResults from './pages/TestResults';
import Landing from './pages/Landing';
import './App.css';

// Create a query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return user ? <>{children}</> : <Navigate to="/login" replace />;
};

// Placeholder components for routes
const Projects = () => <div className="p-6"><h1 className="text-2xl font-bold">Projects</h1><p>Project management coming soon...</p></div>;
const Upload = () => <div className="p-6"><h1 className="text-2xl font-bold">Upload</h1><p>File upload interface coming soon...</p></div>;
const Classify = () => <div className="p-6"><h1 className="text-2xl font-bold">Classify</h1><p>Classification interface coming soon...</p></div>;
const Review = () => <div className="p-6"><h1 className="text-2xl font-bold">Review</h1><p>Review system coming soon...</p></div>;
const Analytics = () => <div className="p-6"><h1 className="text-2xl font-bold">Analytics</h1><p>Analytics dashboard coming soon...</p></div>;
const Export = () => <div className="p-6"><h1 className="text-2xl font-bold">Export</h1><p>Export functionality coming soon...</p></div>;
const Team = () => <div className="p-6"><h1 className="text-2xl font-bold">Team</h1><p>Team management coming soon...</p></div>;
const Settings = () => <div className="p-6"><h1 className="text-2xl font-bold">Settings</h1><p>Settings panel coming soon...</p></div>;

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <Router>
          <div className="App">
            <Routes>
              {/* Public routes */}
              <Route path="/login" element={<Login />} />
              <Route path="/test" element={<TestClassification />} />
              <Route path="/landing" element={<Landing />} />
              
              {/* Protected routes */}
              <Route path="/" element={
                <ProtectedRoute>
                  <Layout>
                    <Dashboard />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/projects" element={
                <ProtectedRoute>
                  <Layout>
                    <Projects />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/upload" element={
                <ProtectedRoute>
                  <Layout>
                    <Upload />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/classify" element={
                <ProtectedRoute>
                  <Layout>
                    <Classify />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/review" element={
                <ProtectedRoute>
                  <Layout>
                    <Review />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/analytics" element={
                <ProtectedRoute>
                  <Layout>
                    <Analytics />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/export" element={
                <ProtectedRoute>
                  <Layout>
                    <Export />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/team" element={
                <ProtectedRoute>
                  <Layout>
                    <Team />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/settings" element={
                <ProtectedRoute>
                  <Layout>
                    <Settings />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/test-results" element={
                <ProtectedRoute>
                  <Layout>
                    <TestResults />
                  </Layout>
                </ProtectedRoute>
              } />
              
              {/* Redirect to landing by default */}
              <Route path="*" element={<Navigate to="/landing" replace />} />
              <Route index element={<Navigate to="/landing" replace />} />
            </Routes>
            
            {/* Toast notifications */}
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#363636',
                  color: '#fff',
                },
                success: {
                  duration: 3000,
                  style: {
                    background: '#10b981',
                  },
                },
                error: {
                  duration: 5000,
                  style: {
                    background: '#ef4444',
                  },
                },
              }}
            />
          </div>
        </Router>
      </AuthProvider>
    </QueryClientProvider>
  );
}

export default App;
