import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import LoginForm from './components/LoginForm';

const Simulation = () => {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(1);
  const { isAuthenticated } = useAuth();
  const [formData, setFormData] = useState({
    startDate: '',
    endDate: '',
    tariffType: 'normal',
    contractedPower: '6.9',
    includeTVAC: true,
    // Advanced Options
    timeHorizon: '12',
    analysisType: 'detailed'
  });
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});

  const steps = [
    { id: 1, title: 'Autenticação E-REDES', icon: '🔐' },
    { id: 2, title: 'Configuração da Simulação', icon: '⚙️' },
    { id: 3, title: 'Confirmar e Executar', icon: '🚀' }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  const handleSimulationConfig = () => {
    const newErrors = {};
    if (!formData.startDate) newErrors.startDate = 'Data de início é obrigatória';
    if (!formData.endDate) newErrors.endDate = 'Data de fim é obrigatória';

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setCurrentStep(3);
  };

  const handleRunSimulation = async () => {
    setLoading(true);

    // Simulate API call
    setTimeout(() => {
      const simulationId = Math.random().toString(36).substr(2, 9);
      setLoading(false);
      navigate(`/results/${simulationId}`);
    }, 3000);
  };

  useEffect(() => {
    if (isAuthenticated) {
      setCurrentStep((prev) => (prev < 2 ? 2 : prev));
    }
  }, [isAuthenticated]);

  const renderLoginForm = () => (
    <LoginForm
      onSuccess={() => setCurrentStep(2)}
      onCancel={() => navigate('/')}
    />
  );

  const renderConfigForm = () => (
    <div style={{
      padding: '30px',
      backgroundColor: 'white',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
      maxWidth: '600px',
      margin: '0 auto'
    }}>
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h2 style={{ color: '#2563eb', margin: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
          ⚙️ Configuração da Simulação
        </h2>
        <p style={{ color: '#6b7280', margin: '10px 0 0' }}>
          Configure os parâmetros da sua simulação
        </p>
      </div>

      <div style={{ display: 'grid', gap: '20px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#374151' }}>
              Data de Início
            </label>
            <input
              type="date"
              value={formData.startDate}
              onChange={(e) => handleInputChange('startDate', e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                border: `2px solid ${errors.startDate ? '#ef4444' : '#d1d5db'}`,
                borderRadius: '8px',
                fontSize: '16px',
                outline: 'none',
                transition: 'border-color 0.2s',
                boxSizing: 'border-box'
              }}
            />
            {errors.startDate && <p style={{ color: '#ef4444', fontSize: '14px', margin: '5px 0 0' }}>{errors.startDate}</p>}
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#374151' }}>
              Data de Fim
            </label>
            <input
              type="date"
              value={formData.endDate}
              onChange={(e) => handleInputChange('endDate', e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                border: `2px solid ${errors.endDate ? '#ef4444' : '#d1d5db'}`,
                borderRadius: '8px',
                fontSize: '16px',
                outline: 'none',
                transition: 'border-color 0.2s',
                boxSizing: 'border-box'
              }}
            />
            {errors.endDate && <p style={{ color: '#ef4444', fontSize: '14px', margin: '5px 0 0' }}>{errors.endDate}</p>}
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#374151' }}>
              Tipo de Tarifa
            </label>
            <select
              value={formData.tariffType}
              onChange={(e) => handleInputChange('tariffType', e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                border: '2px solid #d1d5db',
                borderRadius: '8px',
                fontSize: '16px',
                outline: 'none',
                boxSizing: 'border-box'
              }}
            >
              <option value="normal">Tarifa Normal</option>
              <option value="bi-hourly">Tarifa Bi-horária</option>
              <option value="tri-hourly">Tarifa Tri-horária</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#374151' }}>
              Potência Contratada (kVA)
            </label>
            <select
              value={formData.contractedPower}
              onChange={(e) => handleInputChange('contractedPower', e.target.value)}
              style={{
                width: '100%',
                padding: '12px',
                border: '2px solid #d1d5db',
                borderRadius: '8px',
                fontSize: '16px',
                outline: 'none',
                boxSizing: 'border-box'
              }}
            >
              <option value="3.45">3.45 kVA</option>
              <option value="6.9">6.9 kVA</option>
              <option value="10.35">10.35 kVA</option>
              <option value="13.8">13.8 kVA</option>
              <option value="20.7">20.7 kVA</option>
            </select>
          </div>
        </div>

        <div style={{
          padding: '20px',
          backgroundColor: '#f9fafb',
          borderRadius: '8px',
          border: '1px solid #e5e7eb'
        }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '10px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={formData.includeTVAC}
              onChange={(e) => handleInputChange('includeTVAC', e.target.checked)}
              style={{ transform: 'scale(1.2)' }}
            />
            <span style={{ color: '#374151', fontWeight: '500' }}>
              💰 Incluir TVAC (Taxa de IVA) nos cálculos
            </span>
          </label>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '15px', marginTop: '30px' }}>
        <button
          onClick={() => setCurrentStep(1)}
          style={{
            flex: 1,
            padding: '14px',
            backgroundColor: '#6b7280',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: '500',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          ← Voltar
        </button>
        <button
          onClick={handleSimulationConfig}
          style={{
            flex: 2,
            padding: '14px',
            backgroundColor: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: '500',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          Continuar →
        </button>
      </div>
    </div>
  );

  const renderConfirmation = () => (
    <div style={{
      padding: '30px',
      backgroundColor: 'white',
      borderRadius: '12px',
      boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
      maxWidth: '600px',
      margin: '0 auto'
    }}>
      <div style={{ textAlign: 'center', marginBottom: '30px' }}>
        <h2 style={{ color: '#2563eb', margin: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
          🚀 Confirmar e Executar
        </h2>
        <p style={{ color: '#6b7280', margin: '10px 0 0' }}>
          Revise os parâmetros antes de executar a simulação
        </p>
      </div>

      <div style={{
        padding: '20px',
        backgroundColor: '#f9fafb',
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3 style={{ margin: '0 0 15px', color: '#374151' }}>📋 Resumo da Configuração</h3>
        <div style={{ display: 'grid', gap: '10px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#6b7280' }}>Período:</span>
            <span style={{ fontWeight: '500' }}>{formData.startDate} até {formData.endDate}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#6b7280' }}>Tipo de Tarifa:</span>
            <span style={{ fontWeight: '500' }}>{formData.tariffType}</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#6b7280' }}>Potência Contratada:</span>
            <span style={{ fontWeight: '500' }}>{formData.contractedPower} kVA</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#6b7280' }}>Incluir TVAC:</span>
            <span style={{ fontWeight: '500' }}>{formData.includeTVAC ? 'Sim' : 'Não'}</span>
          </div>
        </div>
      </div>

      {loading && (
        <div style={{
          padding: '20px',
          backgroundColor: '#dbeafe',
          borderRadius: '8px',
          marginBottom: '20px',
          textAlign: 'center'
        }}>
          <p style={{ margin: 0, color: '#1e40af', fontWeight: '500' }}>
            🔄 A executar simulação... Isto pode demorar alguns momentos.
          </p>
        </div>
      )}

      <div style={{ display: 'flex', gap: '15px' }}>
        <button
          onClick={() => setCurrentStep(2)}
          disabled={loading}
          style={{
            flex: 1,
            padding: '14px',
            backgroundColor: loading ? '#9ca3af' : '#6b7280',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: '500',
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          ← Voltar
        </button>
        <button
          onClick={handleRunSimulation}
          disabled={loading}
          style={{
            flex: 2,
            padding: '14px',
            backgroundColor: loading ? '#9ca3af' : '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: '500',
            cursor: loading ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
        >
          {loading ? '🔄 Executando...' : '🚀 Executar Simulação'}
        </button>
      </div>
    </div>
  );

  return (
    <div style={{
      padding: '20px',
      fontFamily: 'Arial, sans-serif',
      backgroundColor: '#f5f5f5',
      minHeight: '100vh'
    }}>
      {/* Header */}
      <div style={{
        marginBottom: '30px',
        textAlign: 'center'
      }}>
        <h1 style={{
          margin: 0,
          color: '#2563eb',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '10px'
        }}>
          <span>⚙️</span>
          Simulação E-REDES
        </h1>
        <p style={{ margin: '10px 0 0', color: '#6b7280' }}>
          Configure e execute a sua simulação de faturação
        </p>
      </div>

      {/* Steps Indicator */}
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        marginBottom: '40px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '20px',
          padding: '20px',
          backgroundColor: 'white',
          borderRadius: '12px',
          boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
        }}>
          {steps.map((step, index) => (
            <div key={step.id} style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '5px'
              }}>
                <div style={{
                  width: '50px',
                  height: '50px',
                  borderRadius: '50%',
                  backgroundColor: currentStep >= step.id ? '#2563eb' : '#e5e7eb',
                  color: currentStep >= step.id ? 'white' : '#6b7280',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '20px',
                  fontWeight: 'bold',
                  transition: 'all 0.3s'
                }}>
                  {currentStep > step.id ? '✓' : step.icon}
                </div>
                <span style={{
                  fontSize: '14px',
                  color: currentStep >= step.id ? '#2563eb' : '#6b7280',
                  fontWeight: '500',
                  textAlign: 'center',
                  maxWidth: '100px'
                }}>
                  {step.title}
                </span>
              </div>
              {index < steps.length - 1 && (
                <div style={{
                  width: '30px',
                  height: '2px',
                  backgroundColor: currentStep > step.id ? '#2563eb' : '#e5e7eb',
                  transition: 'background-color 0.3s'
                }} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Current Step Content */}
      {currentStep === 1 && renderLoginForm()}
      {currentStep === 2 && renderConfigForm()}
      {currentStep === 3 && renderConfirmation()}
    </div>
  );
};

export default Simulation;