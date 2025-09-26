// src/components/Input/Input.jsx
import React, { forwardRef, useState } from 'react';
import './Input.styles.css';

const Input = forwardRef(({
  type = 'text',
  label,
  error,
  success,
  disabled = false,
  required = false,
  size = 'md',
  variant = 'default',
  leftIcon,
  rightIcon,
  className = '',
  ...props
}, ref) => {
  const [focused, setFocused] = useState(false);
  
  const baseClass = 'input-wrapper';
  const sizeClass = `input--${size}`;
  const variantClass = `input--${variant}`;
  const stateClass = error ? 'input--error' : success ? 'input--success' : '';
  const focusClass = focused ? 'input--focused' : '';
  const disabledClass = disabled ? 'input--disabled' : '';
  
  const wrapperClasses = [
    baseClass,
    sizeClass,
    variantClass,
    stateClass,
    focusClass,
    disabledClass,
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={wrapperClasses}>
      {label && (
        <label className="input-label">
          {label}
          {required && <span className="input-required">*</span>}
        </label>
      )}
      
      <div className="input-container">
        {leftIcon && <div className="input-icon input-icon--left">{leftIcon}</div>}
        
        <input
          ref={ref}
          type={type}
          disabled={disabled}
          className="input-field"
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          {...props}
        />
        
        {rightIcon && <div className="input-icon input-icon--right">{rightIcon}</div>}
      </div>
      
      {error && <div className="input-message input-message--error">{error}</div>}
      {success && <div className="input-message input-message--success">{success}</div>}
    </div>
  );
});

Input.displayName = 'Input';

export default Input;