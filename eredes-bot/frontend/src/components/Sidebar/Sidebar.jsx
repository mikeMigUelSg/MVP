import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Sidebar.styles.css';

const Sidebar = ({ isOpen, onToggle }) => {
  const location = useLocation();

  const menuItems = [
    {
      path: '/',
      icon: '<à',
      label: 'Dashboard',
      exact: true
    },
    {
      path: '/simulation',
      icon: '™',
      label: 'Simulação'
    },
    {
      path: '/results',
      icon: '=Ê',
      label: 'Resultados'
    },
    {
      path: '/settings',
      icon: '™',
      label: 'Definições'
    }
  ];

  const isActive = (item) => {
    if (item.exact) {
      return location.pathname === item.path;
    }
    return location.pathname.startsWith(item.path);
  };

  return (
    <aside className={`sidebar ${isOpen ? 'sidebar-open' : 'sidebar-closed'}`}>
      <div className="sidebar-header">
        <div className="sidebar-logo">
          <span className="logo-icon">¡</span>
          {isOpen && <span className="logo-text">E-REDES Bot</span>}
        </div>
      </div>

      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={`sidebar-item ${isActive(item) ? 'active' : ''}`}
            title={!isOpen ? item.label : undefined}
          >
            <span className="sidebar-icon">{item.icon}</span>
            {isOpen && <span className="sidebar-label">{item.label}</span>}
          </Link>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button
          className="sidebar-toggle"
          onClick={onToggle}
          title={isOpen ? 'Fechar menu' : 'Abrir menu'}
        >
          {isOpen ? 'À' : '¶'}
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;