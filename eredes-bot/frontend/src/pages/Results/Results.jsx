import React, { useState, useEffect, useMemo } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import { 
  Card, 
  Button, 
  MetricCard, 
  Chart, 
  StatusIndicator,
  Modal 
} from '../../components';
import './Results.styles.css';

const Results = () => {
  const { id } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [chartView, setChartView] = useState('consumption'); // consumption, power, cost
  const [timeRange, setTimeRange] = useState('24h'); // 24h, 7d, 30d
  const [showExportModal, setShowExportModal] = useState(false);
  const [exportFormat, setExportFormat] = useState('xlsx');
  const [selectedMetric, setSelectedMetric] = useState(null);

  // Mock data generation
  const generateMockData = () => {
    const config = location.state?.simulationConfig || {};
    const startDate = new Date(config.startDate || '2024-01-01');
    const endDate = new Date(config.endDate || '2024-01-31');
    const days = Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24));
    
    // Generate hourly data
    const hourlyData = [];
    const dailyData = [];
    const weeklyData = [];
    
    for (let i = 0; i < days * 24; i++) {
      const date = new Date(startDate.getTime() + i * 60 * 60 * 1000);
      const hour = date.getHours();
      
      // Realistic consumption pattern
      let baseConsumption = 0.8; // Base load
      
      // Daily pattern
      if (hour >= 7 && hour <= 9) baseConsumption += 1.2; // Morning peak
      if (hour >= 19 && hour <= 22) baseConsumption += 1.8; // Evening peak
      if (hour >= 0 && hour <= 6) baseConsumption -= 0.3; // Night reduction
      
      // Add some randomness
      const consumption = baseConsumption + (Math.random() - 0.5) * 0.4;
      const power = Math.max(0.1, consumption + Math.sin(i / 6) * 0.3);
      
      hourlyData.push({
        timestamp: date,
        consumption: Math.max(0.1, consumption),
        power: power,
        cost: consumption * 0.15, // €0.15 per kWh
        label: `${date.getDate()}/${date.getMonth() + 1} ${hour}:00`
      });
    }
    
    // Aggregate daily data
    for (let day = 0; day < days; day++) {
      const dayData = hourlyData.slice(day * 24, (day + 1) * 24);
      const date = new Date(startDate.getTime() + day * 24 * 60 * 60 * 1000);
      
      dailyData.push({
        date,
        consumption: dayData.reduce((sum, h) => sum + h.consumption, 0),
        avgPower: dayData.reduce((sum, h) => sum + h.power, 0) / 24,
        cost: dayData.reduce((sum, h) => sum + h.cost, 0),
        peak: Math.max(...dayData.map(h => h.power)),
        label: `${date.getDate()}/${date.getMonth() + 1}`
      });
    }
    
    // Calculate totals and averages
    const totalConsumption = dailyData.reduce((sum, d) => sum + d.consumption, 0);
    const totalCost = dailyData.reduce((sum, d) => sum + d.cost, 0);
    const avgPower = dailyData.reduce((sum, d) => sum + d.avgPower, 0) / days;
    const peakPower = Math.max(...dailyData.map(d => d.peak));
    const minPower = Math.min(...hourlyData.map(h => h.power));
    
    return {
      id,
      config,
      period: {
        start: startDate,
        end: endDate,
        days
      },
      totals: {
        consumption: totalConsumption,
        cost: totalCost,
        avgPower,
        peakPower,
        minPower,
        dataPoints: hourlyData.length,
        carbonFootprint: totalConsumption * 0.35 // kg CO2
      },
      breakdown: {
        peakHours: totalConsumption * 0.4,
        offPeakHours: totalConsumption * 0.6,
        weekdays: totalConsumption * 0.71,
        weekends: totalConsumption * 0.29
      },
      trends: {
        consumption: Math.random() > 0.5 ? 'up' : 'down',
        cost: Math.random() > 0.5 ? 'up' : 'down',
        efficiency: Math.random() > 0.5 ? 'up' : 'down'
      },
      data: {
        hourly: hourlyData,
        daily: dailyData,
        weekly: weeklyData
      },
      insights: [
        {
          type: 'tip',
          icon: '💡',
          title: 'Pico de Consumo',
          message: `O seu maior consumo ocorre entre as 19h-22h (${peakPower.toFixed(1)} kW)`
        },
        {
          type: 'warning',
          icon: '⚠️',
          title: 'Consumo Elevado',
          message: 'Consumo 15% acima da média nacional para habitações similares'
        },
        {
          type: 'success',
          icon: '🌱',
          title: 'Pegada de Carbono',
          message: `${(totalConsumption * 0.35).toFixed(1)} kg CO2 gerados no período`
        }
      ]
    };
  };

  useEffect(() => {
    // Simulate data loading
    const loadResults = async () => {
      setLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockResults = generateMockData();
      setResults(mockResults);
      setLoading(false);
    };
    
    loadResults();
  }, [id, location.state]);

  const chartData = useMemo(() => {
    if (!results) return [];
    
    const sourceData = timeRange === '24h' ? results.data.hourly.slice(-24) :
                      timeRange === '7d' ? results.data.daily.slice(-7) :
                      results.data.daily;
    
    return sourceData.map(item => ({
      label: item.label || `${item.date?.getDate()}/${item.date?.getMonth() + 1}`,
      value: chartView === 'consumption' ? item.consumption :
             chartView === 'power' ? (item.power || item.avgPower) :
             item.cost,
      timestamp: item.timestamp || item.date
    }));
  }, [results, chartView, timeRange]);

  const handleExport = async () => {
    setShowExportModal(false);
    
    // Simulate export
    const filename = `e-redes-results-${id}.${exportFormat}`;
    
    // In real implementation, this would call the backend
    await new Promise(resolve => setTimeout(resolve, 1000));
    alert(`Ficheiro ${filename} gerado com sucesso!`);
  };

  const getChartConfig = () => {
    switch (chartView) {
      case 'consumption':
        return {
          title: '📊 Consumo Energético',
          color: '#2196f3',
          yLabel: 'kWh'
        };
      case 'power':
        return {
          title: '⚡ Potência Instantânea',
          color: '#ff9800',
          yLabel: 'kW'
        };
      case 'cost':
        return {
          title: '💰 Custo Acumulado',
          color: '#4caf50',
          yLabel: '€'
        };
      default:
        return { title: 'Dados', color: '#2196f3', yLabel: '' };
    }
  };

  if (loading) {
    return (
      <div className="results-loading">
        <Card glass padding="xl" className="loading-card">
          <div className="loading-content">
            <div className="loading-spinner">
              <div className="spinner-ring"></div>
              <div className="spinner-ring"></div>
              <div className="spinner-ring"></div>
            </div>
            <h2>Processando Resultados</h2>
            <p>A analisar os seus dados de consumo...</p>
          </div>
        </Card>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="results-error">
        <Card glass padding="xl" className="error-card">
          <div className="error-content">
            <div className="error-icon">❌</div>
            <h2>Resultados Não Encontrados</h2>
            <p>Não foi possível carregar os resultados da simulação.</p>
            <Button onClick={() => navigate('/simulation')} variant="primary">
              Nova Simulação
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  const chartConfig = getChartConfig();

  return (
    <div className="results-page">
      {/* Header Section */}
      <div className="results-header">
        <div className="header-content">
          <div className="header-main">
            <h1 className="page-title gradient-text">
              📈 Análise de Consumo
            </h1>
            <p className="page-subtitle">
              Período: {results.period.start.toLocaleDateString()} - {results.period.end.toLocaleDateString()}
            </p>
            <div className="header-meta">
              <span className="meta-item">
                <strong>{results.period.days}</strong> dias analisados
              </span>
              <span className="meta-item">
                <strong>{results.totals.dataPoints}</strong> pontos de dados
              </span>
              <span className="meta-item">
                CPE: <strong>{results.config.cpe || 'N/A'}</strong>
              </span>
            </div>
          </div>
          
          <div className="header-actions">
            <Button
              variant="outline"
              leftIcon="📊"
              onClick={() => setShowExportModal(true)}
            >
              Exportar Dados
            </Button>
            <Button
              variant="secondary"
              leftIcon="🔄"
              onClick={() => navigate('/simulation')}
            >
              Nova Simulação
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="results-content">
        {/* Key Metrics Grid */}
        <div className="metrics-grid">
          <MetricCard
            title="Consumo Total"
            value={results.totals.consumption.toFixed(1)}
            unit="kWh"
            icon="⚡"
            trend={results.trends.consumption === 'up' ? { direction: 'up' } : { direction: 'down' }}
            trendValue="vs período anterior"
            color="primary"
            animated
            onClick={() => setSelectedMetric('consumption')}
          />
          
          <MetricCard
            title="Custo Estimado"
            value={results.totals.cost.toFixed(2)}
            unit="€"
            icon="💰"
            trend={results.trends.cost === 'up' ? { direction: 'up' } : { direction: 'down' }}
            trendValue="vs período anterior"
            color="success"
            animated
            onClick={() => setSelectedMetric('cost')}
          />
          
          <MetricCard
            title="Potência Média"
            value={results.totals.avgPower.toFixed(2)}
            unit="kW"
            icon="📊"
            trend={{ direction: 'neutral' }}
            trendValue="estável"
            color="warning"
            animated
            onClick={() => setSelectedMetric('power')}
          />
          
          <MetricCard
            title="Pico Máximo"
            value={results.totals.peakPower.toFixed(2)}
            unit="kW"
            icon="🔥"
            trend={results.trends.efficiency === 'up' ? { direction: 'down' } : { direction: 'up' }}
            trendValue="eficiência"
            color="error"
            animated
            onClick={() => setSelectedMetric('peak')}
          />
        </div>

        {/* Interactive Chart Section */}
        <Card glass padding="xl" className="chart-section">
          <div className="chart-controls">
            <div className="chart-tabs">
              <button 
                className={`chart-tab ${chartView === 'consumption' ? 'active' : ''}`}
                onClick={() => setChartView('consumption')}
              >
                📊 Consumo
              </button>
              <button 
                className={`chart-tab ${chartView === 'power' ? 'active' : ''}`}
                onClick={() => setChartView('power')}
              >
                ⚡ Potência
              </button>
              <button 
                className={`chart-tab ${chartView === 'cost' ? 'active' : ''}`}
                onClick={() => setChartView('cost')}
              >
                💰 Custo
              </button>
            </div>
            
            <div className="time-range-selector">
              <button 
                className={`range-btn ${timeRange === '24h' ? 'active' : ''}`}
                onClick={() => setTimeRange('24h')}
              >
                24h
              </button>
              <button 
                className={`range-btn ${timeRange === '7d' ? 'active' : ''}`}
                onClick={() => setTimeRange('7d')}
              >
                7d
              </button>
              <button 
                className={`range-btn ${timeRange === '30d' ? 'active' : ''}`}
                onClick={() => setTimeRange('30d')}
              >
                30d
              </button>
            </div>
          </div>
          
          <Chart
            data={chartData}
            title={chartConfig.title}
            type="line"
            color={chartConfig.color}
            yLabel={chartConfig.yLabel}
            height={400}
            gradient
            animated
          />
        </Card>

        {/* Detailed Analysis Grid */}
        <div className="analysis-grid">
          {/* Breakdown Card */}
          <Card glass padding="lg" className="breakdown-card">
            <div className="card-header">
              <h3 className="card-title">⏰ Distribuição por Período</h3>
            </div>
            
            <div className="breakdown-list">
              <div className="breakdown-item">
                <div className="breakdown-label">
                  <span className="breakdown-icon">🌅</span>
                  Horas de Ponta
                </div>
                <div className="breakdown-value">
                  <span className="value">{results.breakdown.peakHours.toFixed(1)}</span>
                  <span className="unit">kWh</span>
                  <div className="breakdown-bar">
                    <div 
                      className="breakdown-fill peak"
                      style={{ width: `${(results.breakdown.peakHours / results.totals.consumption) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="breakdown-item">
                <div className="breakdown-label">
                  <span className="breakdown-icon">🌙</span>
                  Horas de Vazio
                </div>
                <div className="breakdown-value">
                  <span className="value">{results.breakdown.offPeakHours.toFixed(1)}</span>
                  <span className="unit">kWh</span>
                  <div className="breakdown-bar">
                    <div 
                      className="breakdown-fill off-peak"
                      style={{ width: `${(results.breakdown.offPeakHours / results.totals.consumption) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="breakdown-item">
                <div className="breakdown-label">
                  <span className="breakdown-icon">📊</span>
                  Dias Úteis
                </div>
                <div className="breakdown-value">
                  <span className="value">{results.breakdown.weekdays.toFixed(1)}</span>
                  <span className="unit">kWh</span>
                  <div className="breakdown-bar">
                    <div 
                      className="breakdown-fill weekday"
                      style={{ width: `${(results.breakdown.weekdays / results.totals.consumption) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
              
              <div className="breakdown-item">
                <div className="breakdown-label">
                  <span className="breakdown-icon">🏖️</span>
                  Fins de Semana
                </div>
                <div className="breakdown-value">
                  <span className="value">{results.breakdown.weekends.toFixed(1)}</span>
                  <span className="unit">kWh</span>
                  <div className="breakdown-bar">
                    <div 
                      className="breakdown-fill weekend"
                      style={{ width: `${(results.breakdown.weekends / results.totals.consumption) * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Environmental Impact */}
          <Card glass padding="lg" className="environmental-card">
            <div className="card-header">
              <h3 className="card-title">🌱 Impacto Ambiental</h3>
            </div>
            
            <div className="environmental-content">
              <div className="carbon-footprint">
                <div className="carbon-icon">🌍</div>
                <div className="carbon-info">
                  <span className="carbon-value">{results.totals.carbonFootprint.toFixed(1)}</span>
                  <span className="carbon-unit">kg CO₂</span>
                  <span className="carbon-label">emitidos</span>
                </div>
              </div>
              
              <div className="environmental-comparison">
                <div className="comparison-item">
                  <span className="comparison-icon">🚗</span>
                  <span className="comparison-text">
                    Equivalente a {(results.totals.carbonFootprint / 2.3).toFixed(0)} km de carro
                  </span>
                </div>
                <div className="comparison-item">
                  <span className="comparison-icon">🌳</span>
                  <span className="comparison-text">
                    {(results.totals.carbonFootprint / 22).toFixed(1)} árvores necessárias para compensar
                  </span>
                </div>
              </div>
            </div>
          </Card>

          {/* AI Insights */}
          <Card glass padding="lg" className="insights-card full-width">
            <div className="card-header">
              <h3 className="card-title">🤖 Insights Inteligentes</h3>
            </div>
            
            <div className="insights-grid">
              {results.insights.map((insight, index) => (
                <div key={index} className={`insight-item insight--${insight.type}`}>
                  <div className="insight-icon">{insight.icon}</div>
                  <div className="insight-content">
                    <h4 className="insight-title">{insight.title}</h4>
                    <p className="insight-message">{insight.message}</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Export Modal */}
      <Modal
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        title="📊 Exportar Resultados"
        size="md"
      >
        <div className="export-modal-content">
          <div className="export-options">
            <h4>Formato de Exportação</h4>
            <div className="format-options">
              <label className="format-option">
                <input
                  type="radio"
                  name="format"
                  value="xlsx"
                  checked={exportFormat === 'xlsx'}
                  onChange={(e) => setExportFormat(e.target.value)}
                />
                <span className="format-label">
                  <span className="format-icon">📊</span>
                  <span className="format-text">Excel (.xlsx)</span>
                  <span className="format-desc">Completo com gráficos</span>
                </span>
              </label>
              
              <label className="format-option">
                <input
                  type="radio"
                  name="format"
                  value="csv"
                  checked={exportFormat === 'csv'}
                  onChange={(e) => setExportFormat(e.target.value)}
                />
                <span className="format-label">
                  <span className="format-icon">📄</span>
                  <span className="format-text">CSV (.csv)</span>
                  <span className="format-desc">Dados tabulares apenas</span>
                </span>
              </label>
              
              <label className="format-option">
                <input
                  type="radio"
                  name="format"
                  value="pdf"
                  checked={exportFormat === 'pdf'}
                  onChange={(e) => setExportFormat(e.target.value)}
                />
                <span className="format-label">
                  <span className="format-icon">📋</span>
                  <span className="format-text">PDF (.pdf)</span>
                  <span className="format-desc">Relatório visual</span>
                </span>
              </label>
            </div>
          </div>
          
          <div className="export-preview">
            <h4>O que será incluído:</h4>
            <ul className="export-includes">
              <li>✅ Dados detalhados de consumo</li>
              <li>✅ Métricas e KPIs principais</li>
              <li>✅ Gráficos de tendência</li>
              <li>✅ Análise de custos</li>
              <li>✅ Insights e recomendações</li>
            </ul>
          </div>
          
          <div className="modal-actions">
            <Button
              variant="ghost"
              onClick={() => setShowExportModal(false)}
            >
              Cancelar
            </Button>
            <Button
              variant="primary"
              onClick={handleExport}
              leftIcon="📥"
            >
              Exportar {exportFormat.toUpperCase()}
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Results;