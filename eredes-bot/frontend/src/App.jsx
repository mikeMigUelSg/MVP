import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { SimulationProvider } from './contexts/SimulationContext';
import { ErrorBoundary, RequireAuth } from './components';

// Pages
import Dashboard from './pages/Dashboard';
import Simulation from './pages/Simulation';
import Results from './pages/Results';
import Settings from './pages/Settings';
import Login from './pages/Login';


function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <AuthProvider>
          <SimulationProvider>
            <Router>
              <div className="app">
                <Routes>
                  <Route path="/login" element={<Login />} />
                  <Route
                    path="/"
                    element={(
                      <RequireAuth>
                        <Dashboard />
                      </RequireAuth>
                    )}
                  />
                  <Route
                    path="/simulation"
                    element={(
                      <RequireAuth>
                        <Simulation />
                      </RequireAuth>
                    )}
                  />
                  <Route
                    path="/results/:id"
                    element={(
                      <RequireAuth>
                        <Results />
                      </RequireAuth>
                    )}
                  />
                  <Route
                    path="/settings"
                    element={(
                      <RequireAuth>
                        <Settings />
                      </RequireAuth>
                    )}
                  />
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