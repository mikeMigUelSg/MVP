import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { simulationService } from '../services/simulationService';

const SimulationContext = createContext();

const initialState = {
  simulations: [],
  currentSimulation: null,
  status: 'idle', // idle, creating, running, completed, failed
  error: null,
  results: null,
  statistics: {
    totalSimulations: 0,
    completedSimulations: 0,
    failedSimulations: 0,
    totalDataProcessed: 0
  }
};

const simulationReducer = (state, action) => {
  switch (action.type) {
    case 'SET_SIMULATIONS':
      return {
        ...state,
        simulations: action.payload,
        statistics: {
          ...state.statistics,
          totalSimulations: action.payload.length,
          completedSimulations: action.payload.filter(s => s.status === 'completed').length,
          failedSimulations: action.payload.filter(s => s.status === 'failed').length
        }
      };

    case 'ADD_SIMULATION':
      return {
        ...state,
        simulations: [action.payload, ...state.simulations],
        currentSimulation: action.payload,
        statistics: {
          ...state.statistics,
          totalSimulations: state.statistics.totalSimulations + 1
        }
      };

    case 'UPDATE_SIMULATION':
      return {
        ...state,
        simulations: state.simulations.map(sim =>
          sim.id === action.payload.id ? { ...sim, ...action.payload } : sim
        ),
        currentSimulation: state.currentSimulation?.id === action.payload.id
          ? { ...state.currentSimulation, ...action.payload }
          : state.currentSimulation
      };

    case 'SET_CURRENT_SIMULATION':
      return {
        ...state,
        currentSimulation: action.payload
      };

    case 'SET_STATUS':
      return {
        ...state,
        status: action.payload
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        status: 'failed'
      };

    case 'SET_RESULTS':
      return {
        ...state,
        results: action.payload,
        status: 'completed'
      };

    case 'CLEAR_ERROR':
      return {
        ...state,
        error: null
      };

    case 'RESET_STATE':
      return initialState;

    default:
      return state;
  }
};

export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};

export const SimulationProvider = ({ children }) => {
  const [state, dispatch] = useReducer(simulationReducer, initialState);

  // Load all simulations
  const loadSimulations = useCallback(async () => {
    try {
      dispatch({ type: 'SET_STATUS', payload: 'loading' });
      const response = await simulationService.getAllSimulations();
      dispatch({ type: 'SET_SIMULATIONS', payload: response.simulations || [] });
      dispatch({ type: 'SET_STATUS', payload: 'idle' });
    } catch (error) {
      console.error('Failed to load simulations:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  }, []);

  // Create new simulation
  const createSimulation = useCallback(async (config) => {
    try {
      dispatch({ type: 'SET_STATUS', payload: 'creating' });
      dispatch({ type: 'CLEAR_ERROR' });

      const response = await simulationService.createSimulation(config);

      if (!response.success) {
        throw new Error(response.error || 'Failed to create simulation');
      }

      const newSimulation = {
        id: response.simulationId,
        ...config,
        status: 'pending',
        createdAt: new Date().toISOString(),
        progress: 0
      };

      dispatch({ type: 'ADD_SIMULATION', payload: newSimulation });
      dispatch({ type: 'SET_STATUS', payload: 'running' });

      return newSimulation;
    } catch (error) {
      console.error('Failed to create simulation:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
      throw error;
    }
  }, []);

  // Monitor simulation progress
  const monitorSimulation = useCallback(async (simulationId) => {
    try {
      const checkStatus = async () => {
        try {
          const response = await simulationService.getSimulationStatus(simulationId);

          const updatedSimulation = {
            id: simulationId,
            status: response.status,
            progress: response.progress || 0,
            error: response.error,
            results: response.results,
            updatedAt: new Date().toISOString()
          };

          dispatch({ type: 'UPDATE_SIMULATION', payload: updatedSimulation });

          if (response.status === 'completed') {
            dispatch({ type: 'SET_RESULTS', payload: response.results });
            return true; // Stop monitoring
          } else if (response.status === 'failed') {
            dispatch({ type: 'SET_ERROR', payload: response.error || 'Simulation failed' });
            return true; // Stop monitoring
          }

          return false; // Continue monitoring
        } catch (error) {
          console.error('Status check failed:', error);
          return false; // Continue monitoring despite error
        }
      };

      // Initial check
      const shouldStop = await checkStatus();
      if (shouldStop) return;

      // Set up polling
      const interval = setInterval(async () => {
        const shouldStop = await checkStatus();
        if (shouldStop) {
          clearInterval(interval);
        }
      }, 5000); // Check every 5 seconds

      // Cleanup after 10 minutes
      setTimeout(() => {
        clearInterval(interval);
        dispatch({ type: 'SET_ERROR', payload: 'Simulation timeout' });
      }, 600000);

    } catch (error) {
      console.error('Failed to monitor simulation:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
    }
  }, []);

  // Get simulation results
  const getResults = useCallback(async (simulationId) => {
    try {
      dispatch({ type: 'SET_STATUS', payload: 'loading' });
      const response = await simulationService.getSimulationResults(simulationId);
      dispatch({ type: 'SET_RESULTS', payload: response });
      dispatch({ type: 'SET_STATUS', payload: 'completed' });
      return response;
    } catch (error) {
      console.error('Failed to get results:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
      throw error;
    }
  }, []);

  // Cancel simulation
  const cancelSimulation = useCallback(async (simulationId) => {
    try {
      await simulationService.cancelSimulation(simulationId);

      const updatedSimulation = {
        id: simulationId,
        status: 'cancelled',
        updatedAt: new Date().toISOString()
      };

      dispatch({ type: 'UPDATE_SIMULATION', payload: updatedSimulation });
    } catch (error) {
      console.error('Failed to cancel simulation:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
      throw error;
    }
  }, []);

  // Retry simulation
  const retrySimulation = useCallback(async (simulationId) => {
    try {
      dispatch({ type: 'SET_STATUS', payload: 'creating' });
      dispatch({ type: 'CLEAR_ERROR' });

      const response = await simulationService.retrySimulation(simulationId);

      const updatedSimulation = {
        id: simulationId,
        status: 'pending',
        progress: 0,
        error: null,
        updatedAt: new Date().toISOString()
      };

      dispatch({ type: 'UPDATE_SIMULATION', payload: updatedSimulation });
      dispatch({ type: 'SET_STATUS', payload: 'running' });

      // Start monitoring the retried simulation
      await monitorSimulation(simulationId);

      return updatedSimulation;
    } catch (error) {
      console.error('Failed to retry simulation:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
      throw error;
    }
  }, [monitorSimulation]);

  // Delete simulation
  const deleteSimulation = useCallback(async (simulationId) => {
    try {
      await simulationService.deleteSimulation(simulationId);

      dispatch({
        type: 'SET_SIMULATIONS',
        payload: state.simulations.filter(sim => sim.id !== simulationId)
      });

      // Clear current simulation if it was deleted
      if (state.currentSimulation?.id === simulationId) {
        dispatch({ type: 'SET_CURRENT_SIMULATION', payload: null });
      }
    } catch (error) {
      console.error('Failed to delete simulation:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
      throw error;
    }
  }, [state.simulations, state.currentSimulation]);

  // Set current simulation
  const setCurrentSimulation = useCallback((simulation) => {
    dispatch({ type: 'SET_CURRENT_SIMULATION', payload: simulation });
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    dispatch({ type: 'CLEAR_ERROR' });
  }, []);

  // Reset context state
  const resetState = useCallback(() => {
    dispatch({ type: 'RESET_STATE' });
  }, []);

  // Get simulation by ID
  const getSimulation = useCallback((simulationId) => {
    return state.simulations.find(sim => sim.id === simulationId);
  }, [state.simulations]);

  // Get recent simulations
  const getRecentSimulations = useCallback((limit = 5) => {
    return state.simulations
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
      .slice(0, limit);
  }, [state.simulations]);

  // Get simulation statistics
  const getStatistics = useCallback(() => {
    const totalDataProcessed = state.simulations
      .filter(sim => sim.results?.totalConsumption)
      .reduce((sum, sim) => sum + (sim.results.totalConsumption || 0), 0);

    return {
      ...state.statistics,
      totalDataProcessed: totalDataProcessed.toFixed(2)
    };
  }, [state.simulations, state.statistics]);

  const value = {
    // State
    ...state,

    // Actions
    loadSimulations,
    createSimulation,
    monitorSimulation,
    getResults,
    cancelSimulation,
    retrySimulation,
    deleteSimulation,
    setCurrentSimulation,
    clearError,
    resetState,

    // Getters
    getSimulation,
    getRecentSimulations,
    getStatistics
  };

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
};