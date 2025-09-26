import React, { createContext, useContext, useReducer, useCallback, useEffect } from 'react';
import { authService } from '../services/authService';

const AuthContext = createContext();

const initialState = {
  isAuthenticated: false,
  authToken: null,
  user: null,
  status: 'idle',
  error: null,
  isInitialized: false,
};

const authReducer = (state, action) => {
  switch (action.type) {
    case 'SESSION_CHECK_START':
      return { ...state, status: 'loading', error: null };
    case 'SESSION_READY':
      return { ...state, status: 'idle', error: null, isInitialized: true };
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
        isInitialized: true,
      };
    case 'LOGIN_ERROR':
      return {
        ...state,
        isAuthenticated: false,
        authToken: null,
        user: null,
        status: 'error',
        error: action.payload,
        isInitialized: true,
      };
    case 'LOGOUT':
      return { ...initialState, isInitialized: true };
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

  useEffect(() => {
    let isMounted = true;

    const initialize = async () => {
      dispatch({ type: 'SESSION_CHECK_START' });

      try {
        const session = await authService.getSession();
        if (!isMounted) return;

        if (session?.isAuthenticated) {
          dispatch({
            type: 'LOGIN_SUCCESS',
            payload: {
              authToken: session.authToken || null,
              user: session.user || null,
            },
          });
        } else {
          dispatch({ type: 'LOGOUT' });
        }
      } catch (error) {
        if (!isMounted) return;
        dispatch({ type: 'LOGIN_ERROR', payload: error.message || 'Falha ao verificar sessão' });
      } finally {
        if (isMounted) {
          dispatch({ type: 'SESSION_READY' });
        }
      }
    };

    initialize();

    return () => {
      isMounted = false;
    };
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
        },
      });

      dispatch({ type: 'RESET_STATUS' });
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
