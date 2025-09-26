import { api } from './api';

class SimulationService {

  async createSimulation(simulationRequest) {
    try {
      const response = await api.post('/simulations/create', simulationRequest);
      return response;
    } catch (error) {
      console.error('Error creating simulation:', error);
      throw error;
    }
  }

  async getSimulationStatus(simulationId) {
    try {
      const response = await api.get(`/simulations/${simulationId}/status`);
      return response;
    } catch (error) {
      console.error('Error fetching simulation status:', error);
      throw error;
    }
  }

  async getAllSimulations(page = 1, limit = 20) {
    try {
      const response = await api.get(`/simulations?page=${page}&limit=${limit}`);
      return response;
    } catch (error) {
      console.error('Error fetching simulations:', error);
      throw error;
    }
  }

  async getSimulationResults(simulationId) {
    try {
      const response = await api.get(`/simulations/${simulationId}/results`);
      return response;
    } catch (error) {
      console.error('Error fetching simulation results:', error);
      throw error;
    }
  }

  async cancelSimulation(simulationId) {
    try {
      const response = await api.post(`/simulations/${simulationId}/cancel`);
      return response;
    } catch (error) {
      console.error('Error cancelling simulation:', error);
      throw error;
    }
  }

  async retrySimulation(simulationId) {
    try {
      const response = await api.post(`/simulations/${simulationId}/retry`);
      return response;
    } catch (error) {
      console.error('Error retrying simulation:', error);
      throw error;
    }
  }

  // Utility methods for validation
  validateCPE(cpe) {
    if (!cpe) return 'CPE é obrigatório';

    // Remove any spaces or special characters
    const cleanCPE = cpe.replace(/\s/g, '').toUpperCase();

    // Check format: PT + 22 digits + 2 letters
    const cpeRegex = /^PT\d{22}[A-Z]{2}$/;

    if (!cpeRegex.test(cleanCPE)) {
      return 'Formato CPE inválido. Deve ser PT seguido de 22 dígitos e 2 letras (ex: PT0000000000000000000000AB)';
    }

    return null;
  }

  validateDateRange(startDate, endDate) {
    if (!startDate) return 'Data de início é obrigatória';
    if (!endDate) return 'Data de fim é obrigatória';

    const start = new Date(startDate);
    const end = new Date(endDate);
    const now = new Date();

    if (start >= end) {
      return 'Data de início deve ser anterior à data de fim';
    }

    if (start > now) {
      return 'Data de início não pode ser no futuro';
    }

    if (end > now) {
      return 'Data de fim não pode ser no futuro';
    }

    // Check if date range is not too large (performance limit)
    const daysDiff = (end - start) / (1000 * 60 * 60 * 24);
    if (daysDiff > 365) {
      return 'Período máximo permitido é de 365 dias';
    }

    if (daysDiff < 1) {
      return 'Período mínimo é de 1 dia';
    }

    return null;
  }

  validateSimulationConfig(config) {
    const errors = {};

    const cpeError = this.validateCPE(config.cpe);
    if (cpeError) errors.cpe = cpeError;

    const dateError = this.validateDateRange(config.startDate, config.endDate);
    if (dateError) errors.dateRange = dateError;

    if (!config.outputFormat) {
      errors.outputFormat = 'Formato de saída é obrigatório';
    } else if (!['xlsx', 'csv', 'json', 'pdf'].includes(config.outputFormat)) {
      errors.outputFormat = 'Formato de saída inválido';
    }

    return Object.keys(errors).length > 0 ? errors : null;
  }

  // Helper methods for data processing
  formatSimulationForDisplay(simulation) {
    return {
      ...simulation,
      displayName: this.generateDisplayName(simulation),
      formattedStartDate: new Date(simulation.startDate).toLocaleDateString('pt-PT'),
      formattedEndDate: new Date(simulation.endDate).toLocaleDateString('pt-PT'),
      formattedCreatedAt: new Date(simulation.createdAt).toLocaleString('pt-PT'),
      duration: this.calculateDuration(simulation.startDate, simulation.endDate),
      statusLabel: this.getStatusLabel(simulation.status),
      statusColor: this.getStatusColor(simulation.status)
    };
  }

  generateDisplayName(simulation) {
    const cpe = simulation.cpe ? simulation.cpe.slice(-6) : 'Unknown';
    const date = new Date(simulation.createdAt).toLocaleDateString('pt-PT', {
      month: 'short',
      day: 'numeric'
    });
    return `Simulação ${cpe} - ${date}`;
  }

  calculateDuration(startDate, endDate) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const days = Math.ceil((end - start) / (1000 * 60 * 60 * 24));

    if (days === 1) return '1 dia';
    if (days < 30) return `${days} dias`;
    if (days < 365) return `${Math.ceil(days / 30)} meses`;
    return `${Math.ceil(days / 365)} anos`;
  }

  getStatusLabel(status) {
    switch (status) {
      case 'pending': return 'Pendente';
      case 'running': return 'Em execução';
      case 'completed': return 'Concluída';
      case 'failed': return 'Falhou';
      case 'cancelled': return 'Cancelada';
      default: return 'Desconhecido';
    }
  }

  getStatusColor(status) {
    switch (status) {
      case 'pending': return '#757575';
      case 'running': return '#ff9800';
      case 'completed': return '#4caf50';
      case 'failed': return '#f44336';
      case 'cancelled': return '#9e9e9e';
      default: return '#757575';
    }
  }

  // Progress tracking utilities
  estimateProgress(simulation) {
    if (!simulation) return 0;

    switch (simulation.status) {
      case 'pending': return 0;
      case 'running':
        // Estimate based on time elapsed
        const createdAt = new Date(simulation.createdAt);
        const now = new Date();
        const elapsed = (now - createdAt) / 1000; // seconds
        const estimatedTotal = 120; // 2 minutes average
        return Math.min(Math.floor((elapsed / estimatedTotal) * 100), 95);
      case 'completed': return 100;
      case 'failed':
      case 'cancelled': return 0;
      default: return 0;
    }
  }

  getProgressMessage(simulation) {
    if (!simulation) return 'Inicializando...';

    switch (simulation.status) {
      case 'pending': return 'Simulação na fila de espera...';
      case 'running':
        const progress = this.estimateProgress(simulation);
        if (progress < 20) return 'Conectando ao Balcão Digital E-REDES...';
        if (progress < 50) return 'Autenticando e obtendo dados...';
        if (progress < 80) return 'Processando dados de consumo...';
        return 'Finalizando análise...';
      case 'completed': return 'Simulação concluída com sucesso!';
      case 'failed': return simulation.error || 'Simulação falhou';
      case 'cancelled': return 'Simulação cancelada pelo utilizador';
      default: return 'Estado desconhecido';
    }
  }
}

export const simulationService = new SimulationService();