import { api } from './api';

class ERedesService {

  async getRecentSimulations(limit = 10) {
    try {
      const response = await api.get(`/simulations/recent?limit=${limit}`);
      return response;
    } catch (error) {
      console.error('Error fetching recent simulations:', error);
      return { simulations: [] };
    }
  }

  async getDashboardStats() {
    try {
      const response = await api.get('/dashboard/stats');
      return response;
    } catch (error) {
      console.error('Error fetching dashboard stats:', error);
      return { stats: {} };
    }
  }

  async getRecentActivity(limit = 10) {
    try {
      const response = await api.get(`/activity/recent?limit=${limit}`);
      return response;
    } catch (error) {
      console.error('Error fetching recent activity:', error);
      return { activities: [] };
    }
  }

  async getSimulationData(simulationId) {
    try {
      const response = await api.get(`/simulations/${simulationId}`);
      return response;
    } catch (error) {
      console.error('Error fetching simulation data:', error);
      throw error;
    }
  }

  async getConsumptionData(simulationId) {
    try {
      const response = await api.get(`/simulations/${simulationId}/consumption`);
      return response;
    } catch (error) {
      console.error('Error fetching consumption data:', error);
      throw error;
    }
  }

  async getCostAnalysis(simulationId) {
    try {
      const response = await api.get(`/simulations/${simulationId}/cost-analysis`);
      return response;
    } catch (error) {
      console.error('Error fetching cost analysis:', error);
      throw error;
    }
  }

  async getInsights(simulationId) {
    try {
      const response = await api.get(`/simulations/${simulationId}/insights`);
      return response;
    } catch (error) {
      console.error('Error fetching insights:', error);
      return { insights: [] };
    }
  }

  async exportSimulation(simulationId, format = 'xlsx') {
    try {
      const response = await api.get(`/simulations/${simulationId}/export?format=${format}`, {
        responseType: 'blob'
      });

      // Create download link
      const blob = new Blob([response.data], {
        type: this.getContentType(format)
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `simulation_${simulationId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);

      return { success: true };
    } catch (error) {
      console.error('Error exporting simulation:', error);
      throw error;
    }
  }

  async deleteSimulation(simulationId) {
    try {
      const response = await api.delete(`/simulations/${simulationId}`);
      return response;
    } catch (error) {
      console.error('Error deleting simulation:', error);
      throw error;
    }
  }

  // Helper method to get content type for different formats
  getContentType(format) {
    switch (format) {
      case 'xlsx':
        return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
      case 'csv':
        return 'text/csv';
      case 'json':
        return 'application/json';
      case 'pdf':
        return 'application/pdf';
      default:
        return 'application/octet-stream';
    }
  }

  // Utility methods for data processing
  processConsumptionData(rawData) {
    if (!rawData || !Array.isArray(rawData)) {
      return [];
    }

    return rawData.map(entry => ({
      timestamp: entry.timestamp,
      value: parseFloat(entry.consumption_kwh) || 0,
      label: new Date(entry.timestamp).toLocaleDateString('pt-PT'),
      hour: new Date(entry.timestamp).getHours()
    }));
  }

  calculateDailyAverages(consumptionData) {
    const dailyTotals = {};

    consumptionData.forEach(entry => {
      const date = entry.timestamp.split('T')[0];
      if (!dailyTotals[date]) {
        dailyTotals[date] = 0;
      }
      dailyTotals[date] += entry.value;
    });

    return Object.entries(dailyTotals).map(([date, total]) => ({
      label: new Date(date).toLocaleDateString('pt-PT'),
      value: total,
      timestamp: date
    }));
  }

  calculateHourlyPattern(consumptionData) {
    const hourlyTotals = new Array(24).fill(0);
    const hourlyCounts = new Array(24).fill(0);

    consumptionData.forEach(entry => {
      const hour = new Date(entry.timestamp).getHours();
      hourlyTotals[hour] += entry.value;
      hourlyCounts[hour]++;
    });

    return hourlyTotals.map((total, hour) => ({
      label: `${hour}:00`,
      value: hourlyCounts[hour] > 0 ? total / hourlyCounts[hour] : 0,
      hour
    }));
  }

  categorizeByTariff(consumptionData) {
    const categories = {
      ponta: { total: 0, hours: [18, 19, 20, 21] }, // Peak hours
      cheia: { total: 0, hours: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] }, // Standard hours
      vazio: { total: 0, hours: [22, 23, 0, 1, 2, 3, 4, 5, 6] }, // Off-peak hours
    };

    consumptionData.forEach(entry => {
      const hour = new Date(entry.timestamp).getHours();

      if (categories.ponta.hours.includes(hour)) {
        categories.ponta.total += entry.value;
      } else if (categories.cheia.hours.includes(hour)) {
        categories.cheia.total += entry.value;
      } else {
        categories.vazio.total += entry.value;
      }
    });

    return [
      { label: 'Vazio', value: categories.vazio.total, color: '#4caf50' },
      { label: 'Cheia', value: categories.cheia.total, color: '#ff9800' },
      { label: 'Ponta', value: categories.ponta.total, color: '#f44336' }
    ];
  }

  generateInsights(consumptionData, costData) {
    const insights = [];

    if (!consumptionData || consumptionData.length === 0) {
      return insights;
    }

    // Calculate basic statistics
    const totalConsumption = consumptionData.reduce((sum, entry) => sum + entry.value, 0);
    const averageDaily = totalConsumption / (consumptionData.length / 96); // Assuming 15-min intervals
    const maxConsumption = Math.max(...consumptionData.map(entry => entry.value));

    // Peak usage insight
    const peakEntry = consumptionData.find(entry => entry.value === maxConsumption);
    if (peakEntry) {
      const peakTime = new Date(peakEntry.timestamp).toLocaleString('pt-PT');
      insights.push({
        type: 'peak',
        icon: '¡',
        title: 'Pico de Consumo',
        description: `O seu maior consumo foi de ${maxConsumption.toFixed(2)} kWh em ${peakTime}`,
        severity: 'info'
      });
    }

    // Efficiency insight
    if (averageDaily < 10) {
      insights.push({
        type: 'efficiency',
        icon: '<1',
        title: 'Consumo Eficiente',
        description: `Com ${averageDaily.toFixed(1)} kWh/dia, está abaixo da média nacional (12 kWh/dia)`,
        severity: 'success'
      });
    } else if (averageDaily > 20) {
      insights.push({
        type: 'efficiency',
        icon: ' ',
        title: 'Alto Consumo',
        description: `${averageDaily.toFixed(1)} kWh/dia é superior à média. Consider medidas de poupança`,
        severity: 'warning'
      });
    }

    // Tariff optimization
    const tariffData = this.categorizeByTariff(consumptionData);
    const pontaPercentage = (tariffData[2].value / totalConsumption) * 100;

    if (pontaPercentage > 30) {
      insights.push({
        type: 'tariff',
        icon: '=°',
        title: 'Otimização de Tarifa',
        description: `${pontaPercentage.toFixed(1)}% do consumo em horas de ponta. Considere alterar hábitos`,
        severity: 'warning'
      });
    }

    // Cost saving opportunities
    if (costData && costData.estimatedSavings > 0) {
      insights.push({
        type: 'savings',
        icon: '=¡',
        title: 'Oportunidade de Poupança',
        description: `Pode poupar até ${costData.estimatedSavings.toFixed(2)}¬ otimizando o consumo`,
        severity: 'info'
      });
    }

    return insights;
  }
}

export const eredesService = new ERedesService();