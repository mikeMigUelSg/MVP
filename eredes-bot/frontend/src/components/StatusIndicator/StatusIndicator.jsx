import React from 'react';
import './StatusIndicator.styles.css';

const StatusIndicator = ({
  status = 'pending',
  message,
  size = 'md',
  showLabel = true
}) => {
  const getStatusConfig = (status) => {
    const configs = {
      success: {
        icon: '',
        label: 'Sucesso',
        color: 'var(--color-success)'
      },
      error: {
        icon: 'L',
        label: 'Erro',
        color: 'var(--color-error)'
      },
      warning: {
        icon: ' ',
        label: 'Aviso',
        color: 'var(--color-warning)'
      },
      pending: {
        icon: 'ó',
        label: 'Pendente',
        color: 'var(--color-info)'
      },
      loading: {
        icon: '=',
        label: 'Carregando',
        color: 'var(--color-primary)'
      }
    };
    return configs[status] || configs.pending;
  };

  const config = getStatusConfig(status);

  return (
    <div className={`status-indicator status-${status} size-${size}`}>
      <span
        className="status-icon"
        style={{ color: config.color }}
        title={message || config.label}
      >
        {config.icon}
      </span>
      {showLabel && (
        <span className="status-text">
          {message || config.label}
        </span>
      )}
    </div>
  );
};

export default StatusIndicator;