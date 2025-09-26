// src/components/MetricCard/MetricCard.jsx
import React from 'react';
import Card from '../Card';
import './MetricCard.styles.css';

const MetricCard = ({
  title,
  value,
  unit,
  icon,
  trend,
  trendValue,
  color = 'primary',
  size = 'md',
  animated = true,
  className = '',
  ...props
}) => {
  const sizeClass = `metric-card--${size}`;
  const colorClass = `metric-card--${color}`;
  const animatedClass = animated ? 'metric-card--animated' : '';
  
  const classes = [
    'metric-card',
    sizeClass,
    colorClass,
    animatedClass,
    className
  ].filter(Boolean).join(' ');

  return (
    <Card
      glass
      hover
      padding="lg"
      className={classes}
      {...props}
    >
      <div className="metric-header">
        {icon && <div className="metric-icon">{icon}</div>}
        <div className="metric-info">
          <h3 className="metric-title">{title}</h3>
          {trend && (
            <div className={`metric-trend metric-trend--${trend.direction}`}>
              <span className="trend-icon">
                {trend.direction === 'up' ? '↗️' : trend.direction === 'down' ? '↘️' : '➡️'}
              </span>
              <span className="trend-value">{trendValue}</span>
            </div>
          )}
        </div>
      </div>
      
      <div className="metric-value-container">
        <span className="metric-value">{value}</span>
        {unit && <span className="metric-unit">{unit}</span>}
      </div>
    </Card>
  );
};

export default MetricCard;