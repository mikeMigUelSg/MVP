import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import Button from '../Button';
import './Modal.styles.css';

const Modal = ({
  isOpen,
  onClose,
  title,
  children,
  size = 'md',
  closable = true,
  footer,
  className = '',
  ...props
}) => {
  const modalRef = useRef(null);
  
  useEffect(() => {
    const handleEscape = (e) => {
      if (e.key === 'Escape' && closable) {
        onClose();
      }
    };
    
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      document.body.style.overflow = 'hidden';
    }
    
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = '';
    };
  }, [isOpen, closable, onClose]);

  const handleBackdropClick = (e) => {
    if (e.target === e.currentTarget && closable) {
      onClose();
    }
  };

  if (!isOpen) return null;

  const sizeClass = `modal--${size}`;
  const classes = ['modal-content', sizeClass, className].filter(Boolean).join(' ');

  return createPortal(
    <div className="modal-overlay animate-fade-in" onClick={handleBackdropClick}>
      <div ref={modalRef} className={classes} {...props}>
        {(title || closable) && (
          <div className="modal-header">
            {title && <h2 className="modal-title">{title}</h2>}
            {closable && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="modal-close"
                aria-label="Fechar modal"
              >
                ✕
              </Button>
            )}
          </div>
        )}
        
        <div className="modal-body">
          {children}
        </div>
        
        {footer && (
          <div className="modal-footer">
            {footer}
          </div>
        )}
      </div>
    </div>,
    document.body
  );
};

export default Modal;