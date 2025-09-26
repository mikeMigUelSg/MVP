// src/components/ErrorBoundary/ErrorBoundary.jsx
import React from 'react';
import { Card, Button } from '../index';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <Card glass padding="xl" className="error-card">
            <div className="error-content">
              <div className="error-icon">💥</div>
              <h2>Algo correu mal</h2>
              <p>
                {this.props.error?.message || 
                 this.state.error?.message || 
                 'Ocorreu um erro inesperado'}
              </p>
              
              <div className="error-actions">
                <Button
                  variant="primary"
                  onClick={() => {
                    this.setState({ hasError: false, error: null });
                    if (this.props.onRetry) {
                      this.props.onRetry();
                    }
                  }}
                >
                  Tentar Novamente
                </Button>
                
                <Button
                  variant="ghost"
                  onClick={() => window.location.reload()}
                >
                  Recarregar Página
                </Button>
              </div>
            </div>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;