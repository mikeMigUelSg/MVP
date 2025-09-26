import { useState, useCallback, useRef, useEffect } from 'react';
import { api, ApiError } from '../services/api';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const abortControllerRef = useRef(null);

  // Cleanup function to abort pending requests
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const request = useCallback(async (apiCall, options = {}) => {
    try {
      // Abort any pending request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // Create new AbortController for this request
      abortControllerRef.current = new AbortController();

      setLoading(true);
      setError(null);

      const result = await apiCall({
        ...options,
        signal: abortControllerRef.current.signal
      });

      setData(result);
      setLoading(false);

      return result;
    } catch (err) {
      // Don't set error state if request was aborted
      if (err.name === 'AbortError') {
        return;
      }

      const errorMessage = err instanceof ApiError
        ? err.message
        : err.message || 'An unknown error occurred';

      setError(errorMessage);
      setLoading(false);
      throw err;
    }
  }, []);

  const get = useCallback((endpoint, options = {}) => {
    return request(() => api.get(endpoint, options));
  }, [request]);

  const post = useCallback((endpoint, data = null, options = {}) => {
    return request(() => api.post(endpoint, data, options));
  }, [request]);

  const put = useCallback((endpoint, data = null, options = {}) => {
    return request(() => api.put(endpoint, data, options));
  }, [request]);

  const del = useCallback((endpoint, options = {}) => {
    return request(() => api.delete(endpoint, options));
  }, [request]);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  const reset = useCallback(() => {
    setLoading(false);
    setError(null);
    setData(null);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
  }, []);

  return {
    loading,
    error,
    data,
    request,
    get,
    post,
    put,
    delete: del,
    clearError,
    reset
  };
};

// Specialized hook for specific API endpoints
export const useAuthApi = () => {
  const { request, ...rest } = useApi();

  const login = useCallback((credentials) => {
    return request(() => api.post('/auth/login', credentials));
  }, [request]);

  const logout = useCallback(() => {
    return request(() => api.post('/auth/logout'));
  }, [request]);

  const refreshToken = useCallback(() => {
    return request(() => api.post('/auth/refresh'));
  }, [request]);

  return {
    ...rest,
    login,
    logout,
    refreshToken
  };
};

export const useSimulationApi = () => {
  const { request, ...rest } = useApi();

  const createSimulation = useCallback((config) => {
    return request(() => api.post('/simulations/create', config));
  }, [request]);

  const getSimulation = useCallback((id) => {
    return request(() => api.get(`/simulations/${id}`));
  }, [request]);

  const getSimulationStatus = useCallback((id) => {
    return request(() => api.get(`/simulations/${id}/status`));
  }, [request]);

  const getSimulationResults = useCallback((id) => {
    return request(() => api.get(`/simulations/${id}/results`));
  }, [request]);

  const getAllSimulations = useCallback(() => {
    return request(() => api.get('/simulations'));
  }, [request]);

  const deleteSimulation = useCallback((id) => {
    return request(() => api.delete(`/simulations/${id}`));
  }, [request]);

  const exportSimulation = useCallback((id, format) => {
    return request(() => api.get(`/simulations/${id}/export?format=${format}`, {
      responseType: 'blob'
    }));
  }, [request]);

  return {
    ...rest,
    createSimulation,
    getSimulation,
    getSimulationStatus,
    getSimulationResults,
    getAllSimulations,
    deleteSimulation,
    exportSimulation
  };
};

export default useApi;