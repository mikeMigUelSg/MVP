import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from './LoadingSpinner';

const RequireAuth = ({ children }) => {
  const { isAuthenticated, isInitialized, status } = useAuth();
  const location = useLocation();

  const isLoading = !isInitialized || status === 'loading';

  if (isLoading) {
    return (
      <div className="auth-loading-state">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
};

export default RequireAuth;
