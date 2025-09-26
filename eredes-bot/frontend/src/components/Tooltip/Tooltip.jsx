import React, { useState } from 'react';
import './Tooltip.styles.css';

const Tooltip = ({
  children,
  content,
  position = 'top',
  delay = 0,
  className = '',
  disabled = false
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [timer, setTimer] = useState(null);

  const showTooltip = () => {
    if (disabled || !content) return;

    if (delay > 0) {
      const newTimer = setTimeout(() => {
        setIsVisible(true);
      }, delay);
      setTimer(newTimer);
    } else {
      setIsVisible(true);
    }
  };

  const hideTooltip = () => {
    if (timer) {
      clearTimeout(timer);
      setTimer(null);
    }
    setIsVisible(false);
  };

  if (!content || disabled) {
    return children;
  }

  return (
    <div
      className={`tooltip-wrapper ${className}`}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
    >
      {children}

      {isVisible && (
        <div className={`tooltip tooltip-${position}`}>
          <div className="tooltip-content">
            {content}
          </div>
          <div className="tooltip-arrow" />
        </div>
      )}
    </div>
  );
};

export default Tooltip;