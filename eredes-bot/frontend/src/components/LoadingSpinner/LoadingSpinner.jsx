import React from 'react';
import './LoadingSpinner.styles.css';

const LoadingSpinner = ({
  size = 'md',
  color = 'primary',
  className = '',
  ...props
}) => {
  const sizeClass = `spinner--${size}`;
  const colorClass = `spinner--${color}`;
  const classes = ['spinner', sizeClass, colorClass, className].filter(Boolean).join(' ');

  return (
    <div className={classes} {...props}>
      <div className="spinner-inner">
        <div className="spinner-circle"></div>
        <div className="spinner-circle"></div>
        <div className="spinner-circle"></div>
      </div>
    </div>
  );
};

export default LoadingSpinner;