import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import {
  Card,
  Button,
  Input,
  StatusIndicator,
  Modal
} from '../../components';
import './Settings.styles.css';

const Settings = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const { theme, toggleTheme } = useTheme();

  const [settings, setSettings] = useState({
    notifications: {
      email: true,
      push: false,
      sms: false
    },
    privacy: {
      dataRetention: '30days',
      shareAnalytics: false,
      saveHistory: true
    },
    export: {
      defaultFormat: 'xlsx',
      includeCharts: true,
      compression: 'medium'
    },
    advanced: {
      apiTimeout: 120,
      retryAttempts: 3,
      debugMode: false
    }
  });

  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const [showClearData, setShowClearData] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState('');

  useEffect(() => {
    // Load settings from localStorage
    const savedSettings = localStorage.getItem('app_settings');
    if (savedSettings) {
      setSettings({ ...settings, ...JSON.parse(savedSettings) });
    }
  }, []);

  const handleSettingChange = (category, key, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [key]: value
      }
    }));
  };

  const saveSettings = async () => {
    setIsSaving(true);
    try {
      localStorage.setItem('app_settings', JSON.stringify(settings));
      setSaveMessage('Definições guardadas com sucesso!');
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      setSaveMessage('Erro ao guardar definições');
    } finally {
      setIsSaving(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      setShowLogoutConfirm(false);
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const clearAllData = async () => {
    try {
      localStorage.clear();
      sessionStorage.clear();

      // Reset settings to defaults
      setSettings({
        notifications: {
          email: true,
          push: false,
          sms: false
        },
        privacy: {
          dataRetention: '30days',
          shareAnalytics: false,
          saveHistory: true
        },
        export: {
          defaultFormat: 'xlsx',
          includeCharts: true,
          compression: 'medium'
        },
        advanced: {
          apiTimeout: 120,
          retryAttempts: 3,
          debugMode: false
        }
      });

      setShowClearData(false);
      setSaveMessage('Dados limpos com sucesso!');
      setTimeout(() => setSaveMessage(''), 3000);
    } catch (error) {
      setSaveMessage('Erro ao limpar dados');
    }
  };

  return (
    <div className="settings-page">
      <div className="page-header">
        <div className="header-content">
          <h1 className="page-title gradient-text">
            ⚙️ Definições
          </h1>
          <p className="page-subtitle">
            Configure a aplicação de acordo com as suas preferências
          </p>
        </div>

        <div className="header-actions">
          <Button
            onClick={saveSettings}
            variant="primary"
            loading={isSaving}
            disabled={isSaving}
          >
            💾 Guardar Definições
          </Button>
        </div>
      </div>

      {saveMessage && (
        <div className={`save-message ${saveMessage.includes('Erro') ? 'error' : 'success'}`}>
          {saveMessage}
        </div>
      )}

      <div className="settings-content">
        {/* Account Section */}
        {isAuthenticated && (
          <Card glass padding="lg" className="settings-section">
            <div className="section-header">
              <h2>👤 Conta</h2>
              <p>Informações da sua conta E-REDES</p>
            </div>

            <div className="account-info">
              <div className="account-status">
                <StatusIndicator
                  status="success"
                  message="Conta conectada"
                />
              </div>

              {user && (
                <div className="user-details">
                  <div className="user-field">
                    <span className="field-label">Nome:</span>
                    <span className="field-value">{user.name || 'N/A'}</span>
                  </div>
                  <div className="user-field">
                    <span className="field-label">NIF:</span>
                    <span className="field-value">{user.nif || 'N/A'}</span>
                  </div>
                  <div className="user-field">
                    <span className="field-label">Último acesso:</span>
                    <span className="field-value">
                      {new Date().toLocaleString('pt-PT')}
                    </span>
                  </div>
                </div>
              )}

              <div className="account-actions">
                <Button
                  onClick={() => setShowLogoutConfirm(true)}
                  variant="ghost"
                  size="sm"
                >
                  🚪 Terminar Sessão
                </Button>
              </div>
            </div>
          </Card>
        )}

        {/* Appearance Section */}
        <Card glass padding="lg" className="settings-section">
          <div className="section-header">
            <h2>🎨 Aparência</h2>
            <p>Personalize o tema e layout da aplicação</p>
          </div>

          <div className="settings-group">
            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Tema</span>
                <span className="setting-description">
                  Escolha entre tema claro ou escuro
                </span>
              </div>
              <div className="setting-control">
                <Button
                  onClick={toggleTheme}
                  variant="ghost"
                  size="sm"
                >
                  {theme === 'dark' ? '☀️ Claro' : '🌙 Escuro'}
                </Button>
              </div>
            </div>
          </div>
        </Card>

        {/* Notifications Section */}
        <Card glass padding="lg" className="settings-section">
          <div className="section-header">
            <h2>🔔 Notificações</h2>
            <p>Configure como pretende receber notificações</p>
          </div>

          <div className="settings-group">
            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Email</span>
                <span className="setting-description">
                  Receber notificações por email
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.notifications.email}
                  onChange={(e) => handleSettingChange('notifications', 'email', e.target.checked)}
                />
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Push</span>
                <span className="setting-description">
                  Notificações push no navegador
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.notifications.push}
                  onChange={(e) => handleSettingChange('notifications', 'push', e.target.checked)}
                />
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">SMS</span>
                <span className="setting-description">
                  Notificações por SMS (funcionalidade futura)
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.notifications.sms}
                  onChange={(e) => handleSettingChange('notifications', 'sms', e.target.checked)}
                  disabled
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Privacy Section */}
        <Card glass padding="lg" className="settings-section">
          <div className="section-header">
            <h2>🔒 Privacidade</h2>
            <p>Controle como os seus dados são utilizados</p>
          </div>

          <div className="settings-group">
            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Retenção de Dados</span>
                <span className="setting-description">
                  Tempo que os dados ficam armazenados localmente
                </span>
              </div>
              <div className="setting-control">
                <select
                  value={settings.privacy.dataRetention}
                  onChange={(e) => handleSettingChange('privacy', 'dataRetention', e.target.value)}
                  className="form-select"
                >
                  <option value="7days">7 dias</option>
                  <option value="30days">30 dias</option>
                  <option value="90days">90 days</option>
                  <option value="never">Nunca limpar</option>
                </select>
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Partilhar Análises</span>
                <span className="setting-description">
                  Ajudar a melhorar a aplicação partilhando dados anónimos
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.privacy.shareAnalytics}
                  onChange={(e) => handleSettingChange('privacy', 'shareAnalytics', e.target.checked)}
                />
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Guardar Histórico</span>
                <span className="setting-description">
                  Manter histórico de simulações localmente
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.privacy.saveHistory}
                  onChange={(e) => handleSettingChange('privacy', 'saveHistory', e.target.checked)}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Export Section */}
        <Card glass padding="lg" className="settings-section">
          <div className="section-header">
            <h2>📤 Exportação</h2>
            <p>Configurações padrão para exportação de dados</p>
          </div>

          <div className="settings-group">
            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Formato Padrão</span>
                <span className="setting-description">
                  Formato preferido para exportações
                </span>
              </div>
              <div className="setting-control">
                <select
                  value={settings.export.defaultFormat}
                  onChange={(e) => handleSettingChange('export', 'defaultFormat', e.target.value)}
                  className="form-select"
                >
                  <option value="xlsx">Excel (.xlsx)</option>
                  <option value="csv">CSV (.csv)</option>
                  <option value="json">JSON (.json)</option>
                  <option value="pdf">PDF (.pdf)</option>
                </select>
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Incluir Gráficos</span>
                <span className="setting-description">
                  Incluir gráficos nas exportações (quando suportado)
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.export.includeCharts}
                  onChange={(e) => handleSettingChange('export', 'includeCharts', e.target.checked)}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Advanced Section */}
        <Card glass padding="lg" className="settings-section">
          <div className="section-header">
            <h2>🔧 Avançado</h2>
            <p>Configurações técnicas para utilizadores experientes</p>
          </div>

          <div className="settings-group">
            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Timeout da API</span>
                <span className="setting-description">
                  Tempo limite para chamadas à API (segundos)
                </span>
              </div>
              <div className="setting-control">
                <Input
                  type="number"
                  value={settings.advanced.apiTimeout}
                  onChange={(e) => handleSettingChange('advanced', 'apiTimeout', parseInt(e.target.value))}
                  min="30"
                  max="300"
                  size="sm"
                />
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Tentativas de Retry</span>
                <span className="setting-description">
                  Número de tentativas em caso de falha
                </span>
              </div>
              <div className="setting-control">
                <Input
                  type="number"
                  value={settings.advanced.retryAttempts}
                  onChange={(e) => handleSettingChange('advanced', 'retryAttempts', parseInt(e.target.value))}
                  min="1"
                  max="10"
                  size="sm"
                />
              </div>
            </div>

            <div className="setting-item">
              <div className="setting-label">
                <span className="setting-name">Modo Debug</span>
                <span className="setting-description">
                  Mostrar logs detalhados na consola
                </span>
              </div>
              <div className="setting-control">
                <input
                  type="checkbox"
                  checked={settings.advanced.debugMode}
                  onChange={(e) => handleSettingChange('advanced', 'debugMode', e.target.checked)}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* Danger Zone */}
        <Card glass padding="lg" className="settings-section danger-zone">
          <div className="section-header">
            <h2>⚠️ Zona Perigosa</h2>
            <p>Acções que podem afectar permanentemente os seus dados</p>
          </div>

          <div className="danger-actions">
            <div className="danger-item">
              <div className="danger-info">
                <h4>Limpar Todos os Dados</h4>
                <p>Remove todas as definições e dados armazenados localmente</p>
              </div>
              <Button
                onClick={() => setShowClearData(true)}
                variant="ghost"
                size="sm"
                className="danger-button"
              >
                🗑️ Limpar Dados
              </Button>
            </div>

            {isAuthenticated && (
              <div className="danger-item">
                <div className="danger-info">
                  <h4>Terminar Sessão</h4>
                  <p>Desconecta da conta E-REDES e remove tokens de autenticação</p>
                </div>
                <Button
                  onClick={() => setShowLogoutConfirm(true)}
                  variant="ghost"
                  size="sm"
                  className="danger-button"
                >
                  🚪 Logout
                </Button>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* Logout Confirmation Modal */}
      <Modal
        isOpen={showLogoutConfirm}
        onClose={() => setShowLogoutConfirm(false)}
        title="Confirmar Logout"
        size="sm"
      >
        <div className="modal-content">
          <p>Tem a certeza que pretende terminar a sessão?</p>
          <p>Terá que fazer login novamente para aceder aos dados do E-REDES.</p>

          <div className="modal-actions">
            <Button
              onClick={() => setShowLogoutConfirm(false)}
              variant="ghost"
            >
              Cancelar
            </Button>
            <Button
              onClick={handleLogout}
              variant="primary"
            >
              Terminar Sessão
            </Button>
          </div>
        </div>
      </Modal>

      {/* Clear Data Confirmation Modal */}
      <Modal
        isOpen={showClearData}
        onClose={() => setShowClearData(false)}
        title="Limpar Todos os Dados"
        size="sm"
      >
        <div className="modal-content">
          <p><strong>Atenção:</strong> Esta acção não pode ser desfeita.</p>
          <p>Todos os dados armazenados localmente serão eliminados, incluindo:</p>
          <ul>
            <li>Definições personalizadas</li>
            <li>Histórico de simulações</li>
            <li>Tokens de autenticação</li>
            <li>Cache de dados</li>
          </ul>

          <div className="modal-actions">
            <Button
              onClick={() => setShowClearData(false)}
              variant="ghost"
            >
              Cancelar
            </Button>
            <Button
              onClick={clearAllData}
              variant="primary"
              className="danger-button"
            >
              Limpar Dados
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Settings;