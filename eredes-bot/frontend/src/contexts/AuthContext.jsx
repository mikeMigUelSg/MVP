import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import { authService } from '../services/authService';

const AuthContext = createContext();

const initialState = {
  isAuthenticated: false,
  authToken: null,
  user: null,
  status: 'idle', // idle, loading, success, error
  error: null,
};

const authReducer = (state, action) => {
  switch (action.type) {
    case 'LOGIN_START':
      return { ...state, status: 'loading', error: null };
    case 'LOGIN_SUCCESS':
      return {
        ...state,
        isAuthenticated: true,
        authToken: action.payload.authToken,
        user: action.payload.user,
        status: 'success',
        error: null,
      };
    case 'LOGIN_ERROR':
      return {
        ...state,
        isAuthenticated: false,
        authToken: null,
        user: null,
        status: 'error',
        error: action.payload,
      };
    case 'LOGOUT':
      return { ...initialState };
    case 'RESET_STATUS':
      return { ...state, status: 'idle', error: null };
    case 'TOKEN_REFRESHED':
      return { ...state, authToken: action.payload };
    default:
      return state;
  }
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  // Initialize auth state on app start
  useEffect(() => {
    const token = authService.getToken();
    const user = authService.getUser();
    
    if (token && user) {
      dispatch({ 
        type: 'LOGIN_SUCCESS', 
        payload: { 
          authToken: token, 
          user 
        } 
      });
    }
  }, []);

  const login = useCallback(async (credentials) => {
    dispatch({ type: 'LOGIN_START' });
    
    try {
      const result = await authService.login(credentials);
      
      dispatch({ 
        type: 'LOGIN_SUCCESS', 
        payload: {
          authToken: result.authToken,
          user: result.user,
        }
      });
      
      return result;
    } catch (error) {
      dispatch({ type: 'LOGIN_ERROR', payload: error.message });
      throw error;
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await authService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      dispatch({ type: 'LOGOUT' });
    }
  }, []);

  const refreshToken = useCallback(async () => {
    try {
      const newToken = await authService.refreshToken();
      dispatch({ type: 'TOKEN_REFRESHED', payload: newToken });
      return newToken;
    } catch (error) {
      dispatch({ type: 'LOGOUT' });
      throw error;
    }
  }, []);

  const resetStatus = useCallback(() => {
    dispatch({ type: 'RESET_STATUS' });
  }, []);

  const value = {
    ...state,
    login,
    logout,
    refreshToken,
    resetStatus,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};