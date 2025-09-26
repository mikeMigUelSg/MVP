import React from 'react';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
  const navigate = useNavigate();

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
        padding: '20px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{
          margin: 0,
          color: '#2563eb',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <span>⚡</span>
          E-REDES Bot Dashboard
        </h1>
        <p style={{ margin: '8px 0 0', color: '#6b7280' }}>
          Bem-vindo ao seu painel de simulações de eletricidade
        </p>
      </div>

      {/* Welcome Message */}
      <div style={{
        padding: '30px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        textAlign: 'center',
        marginBottom: '30px'
      }}>
        <div style={{ fontSize: '64px', marginBottom: '20px' }}>🚀</div>
        <h2 style={{ margin: '0 0 15px', color: '#374151' }}>
          Pronto para começar?
        </h2>
        <p style={{ margin: '0 0 25px', color: '#6b7280', fontSize: '18px' }}>
          Configure a sua primeira simulação de faturação E-REDES
        </p>
        <button
          onClick={() => navigate('/simulation')}
          style={{
            padding: '15px 30px',
            backgroundColor: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: 'pointer',
            fontSize: '18px',
            fontWeight: '500',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '10px',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={e => e.target.style.backgroundColor = '#1d4ed8'}
          onMouseOut={e => e.target.style.backgroundColor = '#2563eb'}
        >
          ⚙️ Iniciar Simulação
        </button>
      </div>

      {/* Quick Actions */}
      <div style={{
        padding: '20px',
        backgroundColor: 'white',
        borderRadius: '12px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        marginBottom: '30px'
      }}>
        <h2 style={{ margin: '0 0 20px', color: '#374151' }}>
          <span style={{ marginRight: '10px' }}>🔧</span>
          Ferramentas
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '15px' }}>
          <button
            onClick={() => navigate('/simulation')}
            style={{
              padding: '20px',
              backgroundColor: '#f8fafc',
              border: '2px solid #e2e8f0',
              borderRadius: '8px',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s'
            }}
            onMouseOver={e => {
              e.target.style.borderColor = '#2563eb';
              e.target.style.backgroundColor = '#f1f5f9';
            }}
            onMouseOut={e => {
              e.target.style.borderColor = '#e2e8f0';
              e.target.style.backgroundColor = '#f8fafc';
            }}
          >
            <div style={{ fontSize: '24px', marginBottom: '10px' }}>⚙️</div>
            <h3 style={{ margin: '0 0 5px', color: '#374151', fontSize: '16px' }}>Nova Simulação</h3>
            <p style={{ margin: 0, color: '#6b7280', fontSize: '14px' }}>
              Configure e execute uma nova análise
            </p>
          </button>

          <button
            onClick={() => navigate('/settings')}
            style={{
              padding: '20px',
              backgroundColor: '#f8fafc',
              border: '2px solid #e2e8f0',
              borderRadius: '8px',
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'all 0.2s'
            }}
            onMouseOver={e => {
              e.target.style.borderColor = '#2563eb';
              e.target.style.backgroundColor = '#f1f5f9';
            }}
            onMouseOut={e => {
              e.target.style.borderColor = '#e2e8f0';
              e.target.style.backgroundColor = '#f8fafc';
            }}
          >
            <div style={{ fontSize: '24px', marginBottom: '10px' }}>⚙️</div>
            <h3 style={{ margin: '0 0 5px', color: '#374151', fontSize: '16px' }}>Configurações</h3>
            <p style={{ margin: 0, color: '#6b7280', fontSize: '14px' }}>
              Personalize as suas preferências
            </p>
          </button>

          <div style={{
            padding: '20px',
            backgroundColor: '#f8fafc',
            border: '2px solid #e2e8f0',
            borderRadius: '8px',
            textAlign: 'left',
            opacity: 0.6
          }}>
            <div style={{ fontSize: '24px', marginBottom: '10px' }}>📊</div>
            <h3 style={{ margin: '0 0 5px', color: '#374151', fontSize: '16px' }}>Resultados</h3>
            <p style={{ margin: 0, color: '#6b7280', fontSize: '14px' }}>
              Execute uma simulação primeiro
            </p>
          </div>
        </div>
      </div>

      {/* Info Section */}
      <div style={{
        padding: '20px',
        backgroundColor: '#dbeafe',
        borderRadius: '12px',
        border: '1px solid #bfdbfe'
      }}>
        <div style={{ display: 'flex', alignItems: 'start', gap: '15px' }}>
          <span style={{ fontSize: '24px' }}>💡</span>
          <div>
            <h3 style={{ margin: '0 0 10px', color: '#1e40af' }}>Como funciona?</h3>
            <p style={{ margin: '0 0 10px', color: '#1e40af' }}>
              1. <strong>Autentique-se</strong> com as suas credenciais E-REDES<br />
              2. <strong>Configure</strong> o período e parâmetros da simulação<br />
              3. <strong>Execute</strong> a análise e obtenha resultados detalhados
            </p>
            <p style={{ margin: 0, color: '#3730a3', fontSize: '14px' }}>
              ⚠️ Necessita de conta E-REDES válida para aceder aos dados de consumo
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;