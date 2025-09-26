import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { SimulationProvider } from './contexts/SimulationContext';
import { ErrorBoundary } from './components';

// Pages
import Dashboard from './pages/Dashboard';
import Simulation from './pages/Simulation';
import Results from './pages/Results';
import Settings from './pages/Settings';


function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <AuthProvider>
          <SimulationProvider>
            <Router>
              <div className="app">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/simulation" element={<Simulation />} />
                  <Route path="/results/:id" element={<Results />} />
                  <Route path="/settings" element={<Settings />} />
                </Routes>
              </div>
            </Router>
          </SimulationProvider>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;