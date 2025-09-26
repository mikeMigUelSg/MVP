import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import './Login.styles.css';

const initialForm = {
  nif: '',
  password: '',
};

const Login = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, status, error, isAuthenticated, isInitialized, resetStatus } = useAuth();
  const [formValues, setFormValues] = useState(initialForm);
  const [formErrors, setFormErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    if (isInitialized && isAuthenticated) {
      const redirectTo = location.state?.from?.pathname || '/';
      navigate(redirectTo, { replace: true });
    }
  }, [isAuthenticated, isInitialized, location.state, navigate]);

  useEffect(() => {
    if (status === 'success' || status === 'error') {
      const timeout = setTimeout(() => resetStatus(), 800);
      return () => clearTimeout(timeout);
    }
    return () => {};
  }, [status, resetStatus]);

  const validate = () => {
    const nextErrors = {};
    const nif = formValues.nif.trim();

    if (!nif) {
      nextErrors.nif = 'NIF é obrigatório';
    } else if (!/^\d{9}$/u.test(nif)) {
      nextErrors.nif = 'Introduza um NIF válido com 9 dígitos';
    }

    if (!formValues.password) {
      nextErrors.password = 'Password é obrigatória';
    }

    setFormErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleChange = (field) => (event) => {
    setFormValues((prev) => ({ ...prev, [field]: event.target.value }));
    if (formErrors[field]) {
      setFormErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!validate()) {
      return;
    }

    try {
      await login({ nif: formValues.nif, password: formValues.password });
    } catch (_) {
      // Error is handled in context state
    }
  };

  const isLoading = status === 'loading';

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-card">
          <div className="login-heading">
            <h6>
              Cliente <strong>particular</strong>
            </h6>
          </div>
          <form className="login-form" onSubmit={handleSubmit} noValidate>
            <div className={`login-field ${formValues.nif ? 'has-value' : ''} ${formErrors.nif ? 'has-error' : ''}`}>
              <input
                id="username"
                name="username"
                type="text"
                autoComplete="username"
                value={formValues.nif}
                onChange={handleChange('nif')}
                disabled={isLoading}
                placeholder=" "
              />
              <label htmlFor="username">NIF</label>
              {formErrors.nif && <p className="field-error">{formErrors.nif}</p>}
            </div>

            <div className="forgot-password">
              <a href="https://balcaodigital.e-redes.pt/login/recover/residential" target="_blank" rel="noreferrer">
                Esqueceu-se da password?
              </a>
            </div>

            <div className={`login-field password-field ${formValues.password ? 'has-value' : ''} ${formErrors.password ? 'has-error' : ''}`}>
              <div className="password-wrapper">
                <input
                  id="labelPassword"
                  name="labelPassword"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  value={formValues.password}
                  onChange={handleChange('password')}
                  disabled={isLoading}
                  placeholder=" "
                />
                <label htmlFor="labelPassword">Password</label>
                <button
                  type="button"
                  className="toggle-password"
                  onClick={() => setShowPassword((prev) => !prev)}
                  aria-label={showPassword ? 'Esconder password' : 'Mostrar password'}
                  disabled={isLoading}
                >
                  <span className="icon-eye" aria-hidden="true">👁️</span>
                </button>
              </div>
              {formErrors.password && <p className="field-error">{formErrors.password}</p>}
            </div>

            {error && (
              <div className="login-error">
                <p>{error}</p>
              </div>
            )}

            <div className="login-actions">
              <button type="submit" className="submit-button" disabled={isLoading}>
                {isLoading ? 'A autenticar…' : 'Entrar'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;
