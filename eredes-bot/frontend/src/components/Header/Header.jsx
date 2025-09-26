// src/components/Header/Header.jsx
import React from 'react';
import { useTheme } from '../../contexts/ThemeContext';
import { useAuth } from '../../contexts/AuthContext';
import Button from '../Button';
import './Header.styles.css';

const Header = () => {
  const { isDark, toggleTheme } = useTheme();
  const { user, logout, isAuthenticated } = useAuth();

  return (
    <header className="header glass">
      <div className="header-container">
        <div className="header-brand">
          <div className="logo-container">
            <div className="logo-icon">
              <span className="logo-symbol">⚡</span>
            </div>
            <div className="logo-text">
              <h1 className="gradient-text">E-REDES Simulator</h1>
              <p className="tagline">Advanced Energy Analytics</p>
            </div>
          </div>
        </div>
        
        <nav className="header-nav">
          <div className="nav-actions">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleTheme}
              className="theme-toggle"
            >
              {isDark ? '☀️' : '🌙'}
            </Button>
            
            {isAuthenticated && user && (
              <div className="user-menu">
                <div className="user-avatar">
                  <span>{user.name?.charAt(0) || 'U'}</span>
                </div>
                <div className="user-info">
                  <span className="user-name">{user.name}</span>
                  <span className="user-nif">NIF: {user.nif}</span>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={logout}
                  className="logout-btn"
                >
                  Sair
                </Button>
              </div>
            )}
          </div>
        </nav>
      </div>
    </header>
  );
};

export default Header;