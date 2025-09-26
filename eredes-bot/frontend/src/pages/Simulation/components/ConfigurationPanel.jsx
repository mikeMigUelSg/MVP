import React from 'react';
import { Card, Button, Input } from '../../../components';

const ConfigurationPanel = ({
  config,
  onChange,
  onStart,
  isAuthenticated,
  loading
}) => {

  const handleInputChange = (field) => (e) => {
    onChange(field, e.target.value);
  };

  const handleDateChange = (field) => (e) => {
    onChange(field, e.target.value);
  };

  const validateCPE = (cpe) => {
    if (!cpe) return '';
    const cleanCPE = cpe.replace(/\s/g, '').toUpperCase();
    return /^PT\d{22}[A-Z]{2}$/.test(cleanCPE) ? '' : 'Formato CPE inválido';
  };

  const validateDateRange = () => {
    if (!config.startDate || !config.endDate) return '';

    const start = new Date(config.startDate);
    const end = new Date(config.endDate);
    const now = new Date();

    if (start >= end) return 'Data de início deve ser anterior à data de fim';
    if (end > now) return 'Data de fim não pode ser no futuro';

    const daysDiff = (end - start) / (1000 * 60 * 60 * 24);
    if (daysDiff > 365) return 'Período máximo permitido é de 365 dias';
    if (daysDiff < 1) return 'Período mínimo é de 1 dia';

    return '';
  };

  const cpeError = validateCPE(config.cpe);
  const dateError = validateDateRange();
  const hasErrors = cpeError || dateError;

  return (
    <Card glass padding="lg" className="configuration-panel">
      <div className="panel-header">
        <h2 className="panel-title">⚙️ Configuração da Simulação</h2>
        <p className="panel-subtitle">
          Configure os parâmetros para análise dos seus dados de consumo
        </p>
      </div>

      <div className="configuration-form">
        {/* CPE Input */}
        <div className="form-group">
          <label htmlFor="cpe" className="form-label">
            <span className="label-text">CPE (Código do Ponto de Entrega)</span>
            <span className="label-required">*</span>
          </label>
          <Input
            id="cpe"
            type="text"
            placeholder="PT0000000000000000000000AB"
            value={config.cpe}
            onChange={handleInputChange('cpe')}
            error={cpeError}
            disabled={loading}
            maxLength={26}
          />
          <small className="form-hint">
            Encontre o seu CPE na fatura de eletricidade ou no Balcão Digital
          </small>
          {cpeError && (
            <small className="form-error-text">{cpeError}</small>
          )}
        </div>

        {/* Date Range */}
        <div className="date-range-group">
          <div className="form-group">
            <label htmlFor="startDate" className="form-label">
              <span className="label-text">Data de Início</span>
              <span className="label-required">*</span>
            </label>
            <Input
              id="startDate"
              type="datetime-local"
              value={config.startDate}
              onChange={handleDateChange('startDate')}
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="endDate" className="form-label">
              <span className="label-text">Data de Fim</span>
              <span className="label-required">*</span>
            </label>
            <Input
              id="endDate"
              type="datetime-local"
              value={config.endDate}
              onChange={handleDateChange('endDate')}
              disabled={loading}
            />
          </div>
        </div>

        {dateError && (
          <div className="date-error">
            <small className="form-error-text">{dateError}</small>
          </div>
        )}

        {/* Advanced Options */}
        <div className="advanced-options">
          <div className="options-header">
            <h3>Opções Avançadas</h3>
          </div>

          <div className="form-group">
            <label htmlFor="timezone" className="form-label">
              Fuso Horário
            </label>
            <select
              id="timezone"
              value={config.timezone}
              onChange={handleInputChange('timezone')}
              disabled={loading}
              className="form-select"
            >
              <option value="local">Local (Automático)</option>
              <option value="utc">UTC</option>
              <option value="europe/lisbon">Europa/Lisboa</option>
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="outputFormat" className="form-label">
              Formato de Saída
            </label>
            <select
              id="outputFormat"
              value={config.outputFormat}
              onChange={handleInputChange('outputFormat')}
              disabled={loading}
              className="form-select"
            >
              <option value="xlsx">Excel (.xlsx) - Recomendado</option>
              <option value="csv">CSV (.csv) - Dados tabulares</option>
              <option value="json">JSON (.json) - Para integrações</option>
              <option value="pdf">PDF (.pdf) - Relatório visual</option>
            </select>
          </div>
        </div>

        {/* Configuration Summary */}
        {config.cpe && config.startDate && config.endDate && !hasErrors && (
          <div className="config-summary">
            <h4>Resumo da Configuração</h4>
            <div className="summary-items">
              <div className="summary-item">
                <span className="summary-label">CPE:</span>
                <span className="summary-value">{config.cpe}</span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Período:</span>
                <span className="summary-value">
                  {new Date(config.startDate).toLocaleDateString('pt-PT')} -
                  {new Date(config.endDate).toLocaleDateString('pt-PT')}
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Duração:</span>
                <span className="summary-value">
                  {Math.ceil((new Date(config.endDate) - new Date(config.startDate)) / (1000 * 60 * 60 * 24))} dias
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Formato:</span>
                <span className="summary-value">{config.outputFormat.toUpperCase()}</span>
              </div>
            </div>
          </div>
        )}

        {/* Action Button */}
        <div className="panel-actions">
          <Button
            variant="primary"
            size="lg"
            onClick={onStart}
            disabled={loading || hasErrors || !config.cpe || !config.startDate || !config.endDate}
            loading={loading}
            fullWidth
          >
            {loading ? (
              'A processar...'
            ) : !isAuthenticated ? (
              '🔐 Fazer Login e Iniciar'
            ) : (
              '🚀 Iniciar Simulação'
            )}
          </Button>

          {!isAuthenticated && (
            <div className="auth-notice">
              <p>
                <span className="notice-icon">ℹ️</span>
                Será solicitado o login no Balcão Digital E-REDES antes de iniciar
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Quick Presets */}
      <div className="quick-presets">
        <h4>Períodos Rápidos</h4>
        <div className="preset-buttons">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              const now = new Date();
              const lastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
              const endLastMonth = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59);

              onChange('startDate', lastMonth.toISOString().slice(0, 16));
              onChange('endDate', endLastMonth.toISOString().slice(0, 16));
            }}
            disabled={loading}
          >
            Último Mês
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              const now = new Date();
              const last30Days = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

              onChange('startDate', last30Days.toISOString().slice(0, 16));
              onChange('endDate', now.toISOString().slice(0, 16));
            }}
            disabled={loading}
          >
            Últimos 30 Dias
          </Button>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => {
              const now = new Date();
              const last7Days = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);

              onChange('startDate', last7Days.toISOString().slice(0, 16));
              onChange('endDate', now.toISOString().slice(0, 16));
            }}
            disabled={loading}
          >
            Últimos 7 Dias
          </Button>
        </div>
      </div>

      {/* Help Section */}
      <div className="help-section">
        <h4>Ajuda</h4>
        <div className="help-items">
          <div className="help-item">
            <span className="help-icon">💡</span>
            <div className="help-content">
              <strong>CPE não encontra?</strong>
              <p>Consulte a sua fatura de eletricidade ou aceda ao Balcão Digital E-REDES</p>
            </div>
          </div>

          <div className="help-item">
            <span className="help-icon">📅</span>
            <div className="help-content">
              <strong>Período recomendado</strong>
              <p>Para análises detalhadas, recomendamos pelo menos 30 dias de dados</p>
            </div>
          </div>

          <div className="help-item">
            <span className="help-icon">⚡</span>
            <div className="help-content">
              <strong>Dados em tempo real</strong>
              <p>Os dados são obtidos diretamente do E-REDES, não são simulados</p>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default ConfigurationPanel;