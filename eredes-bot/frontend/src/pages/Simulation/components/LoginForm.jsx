import React, { useState } from 'react';
import { useAuth } from '../../../contexts/AuthContext';
import { Card, Button, Input, StatusIndicator } from '../../../components';

const LoginForm = ({ onSuccess, onCancel }) => {
  const { login, status, error } = useAuth();

  const [credentials, setCredentials] = useState({
    username: '',
    password: ''
  });

  const [validationErrors, setValidationErrors] = useState({});
  const [showPassword, setShowPassword] = useState(false);

  const validateForm = () => {
    const errors = {};

    if (!credentials.username.trim()) {
      errors.username = 'NIF é obrigatório';
    } else if (!/^\d{9}$/.test(credentials.username.trim())) {
      errors.username = 'NIF deve ter 9 dígitos';
    }

    if (!credentials.password) {
      errors.password = 'Password é obrigatória';
    } else if (credentials.password.length < 4) {
      errors.password = 'Password deve ter pelo menos 4 caracteres';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      await login(credentials);
      onSuccess?.();
    } catch (err) {
      console.error('Login failed:', err);
    }
  };

  const handleInputChange = (field, value) => {
    setCredentials(prev => ({
      ...prev,
      [field]: value
    }));

    if (validationErrors[field]) {
      setValidationErrors(prev => ({
        ...prev,
        [field]: ''
      }));
    }
  };

  const isLoading = status === 'loading';

  return (
    <div className="login-form">
      <div className="login-header">
        <div className="login-icon">🔐</div>
        <h2>Acesso ao Balcão Digital E-REDES</h2>
        <p>
          Introduza as suas credenciais para aceder aos dados reais de consumo
        </p>
      </div>

      <Card glass padding="lg" className="login-card">
        <form onSubmit={handleSubmit} className="login-form-content">
          <div className="form-group">
            <label htmlFor="nif" className="form-label">
              NIF (Número de Identificação Fiscal)
            </label>
            <Input
              id="nif"
              type="text"
              placeholder="123456789"
              value={credentials.username}
              onChange={(e) => handleInputChange('username', e.target.value)}
              error={validationErrors.username}
              maxLength={9}
              disabled={isLoading}
              autoComplete="username"
              required
            />
            <small className="form-hint">
              O mesmo NIF que utiliza no site oficial E-REDES
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="password" className="form-label">
              Password
            </label>
            <div className="password-input-wrapper">
              <Input
                id="password"
                type={showPassword ? 'text' : 'password'}
                placeholder="Introduza a sua password"
                value={credentials.password}
                onChange={(e) => handleInputChange('password', e.target.value)}
                error={validationErrors.password}
                disabled={isLoading}
                autoComplete="current-password"
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                disabled={isLoading}
                tabIndex={-1}
              >
                {showPassword ? '👁️' : '👁️‍🗨️'}
              </button>
            </div>
            <small className="form-hint">
              A mesma password que utiliza no Balcão Digital
            </small>
          </div>

          {error && (
            <div className="form-error">
              <StatusIndicator status="error" message={error} size="sm" />
            </div>
          )}

          <div className="security-notice">
            <div className="security-icon">🔒</div>
            <div className="security-text">
              <strong>Segurança Garantida</strong>
              <p>
                As suas credenciais são utilizadas apenas para autenticação direta
                com o E-REDES e nunca são armazenadas nos nossos servidores.
              </p>
            </div>
          </div>

          <div className="form-actions">
            <Button
              type="submit"
              variant="primary"
              size="lg"
              disabled={isLoading}
              loading={isLoading}
              fullWidth
            >
              {isLoading ? 'A autenticar...' : 'Fazer Login'}
            </Button>

            <Button
              type="button"
              variant="ghost"
              size="md"
              onClick={onCancel}
              disabled={isLoading}
              fullWidth
            >
              Cancelar
            </Button>
          </div>
        </form>
      </Card>

      <div className="login-footer">
        <div className="login-help">
          <h4>Como funciona?</h4>
          <ul>
            <li>
              <span className="step-number">1</span>
              Autenticamos com as suas credenciais no site oficial E-REDES
            </li>
            <li>
              <span className="step-number">2</span>
              Obtemos os dados reais de consumo da sua instalação
            </li>
            <li>
              <span className="step-number">3</span>
              Processamos e analisamos os dados localmente
            </li>
            <li>
              <span className="step-number">4</span>
              Apresentamos relatórios detalhados e insights
            </li>
          </ul>
        </div>

        <div className="login-privacy">
          <div className="privacy-icon">🛡️</div>
          <div className="privacy-text">
            <strong>Privacidade Total</strong>
            <p>
              Zero dados armazenados. Todas as operações são realizadas em tempo real
              e os dados são processados localmente.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginForm;
