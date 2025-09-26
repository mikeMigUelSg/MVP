import React, { forwardRef } from 'react';
import './Button.styles.css';

const Button = forwardRef(({
  children,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  leftIcon,
  rightIcon,
  className = '',
  ...props
}, ref) => {
  const baseClass = 'btn';
  const variantClass = `btn--${variant}`;
  const sizeClass = `btn--${size}`;
  const stateClass = disabled || loading ? 'btn--disabled' : '';
  
  const classes = [baseClass, variantClass, sizeClass, stateClass, className]
    .filter(Boolean)
    .join(' ');

  return (
    <button
      ref={ref}
      className={classes}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <span className="btn__spinner"></span>}
      {leftIcon && !loading && <span className="btn__icon btn__icon--left">{leftIcon}</span>}
      <span className="btn__content">{children}</span>
      {rightIcon && !loading && <span className="btn__icon btn__icon--right">{rightIcon}</span>}
    </button>
  );
});

Button.displayName = 'Button';

export default Button;