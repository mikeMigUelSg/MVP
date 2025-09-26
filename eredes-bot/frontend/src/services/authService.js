import { api, ApiError } from './api';

class AuthService {
  constructor() {
    this.sessionCache = null;
  }

  async login(credentials) {
    const payload = {
      nif: (credentials.nif || credentials.username || '').trim(),
      password: credentials.password || '',
    };

    const response = await api.post('/auth/login', payload);
    this.sessionCache = {
      ...response,
      isAuthenticated: true,
    };

    return response;
  }

  async logout() {
    await api.post('/auth/logout');
    this.sessionCache = null;
  }

  async getSession() {
    if (this.sessionCache?.isAuthenticated) {
      return this.sessionCache;
    }

    try {
      const session = await api.get('/auth/session');
      this.sessionCache = session?.isAuthenticated ? { ...session } : null;
      return session;
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        this.sessionCache = null;
        return { isAuthenticated: false };
      }
      throw error;
    }
  }

  async getToken() {
    try {
      const tokenResponse = await api.get('/auth/token');
      return tokenResponse?.authToken || null;
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        return null;
      }
      throw error;
    }
  }

  async refreshToken() {
    const token = await this.getToken();
    if (!token) {
      throw new ApiError('Sessão expirada.', 401);
    }
    return token;
  }
}

export const authService = new AuthService();
