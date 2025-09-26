// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  TIMEOUT: 120000, // 2 minutes
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
};

// Application Routes
export const ROUTES = {
  DASHBOARD: '/',
  SIMULATION: '/simulation',
  RESULTS: '/results',
  SETTINGS: '/settings'
};

// Simulation Status
export const SIMULATION_STATUS = {
  IDLE: 'idle',
  PENDING: 'pending',
  RUNNING: 'running',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled'
};

// Export Formats
export const EXPORT_FORMATS = {
  XLSX: 'xlsx',
  CSV: 'csv',
  JSON: 'json',
  PDF: 'pdf'
};

// Tariff Periods (Portugal)
export const TARIFF_PERIODS = {
  PEAK: {
    name: 'Ponta',
    hours: [18, 19, 20, 21],
    color: '#f44336'
  },
  STANDARD: {
    name: 'Cheia',
    hours: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    color: '#ff9800'
  },
  OFF_PEAK: {
    name: 'Vazio',
    hours: [22, 23, 0, 1, 2, 3, 4, 5, 6],
    color: '#4caf50'
  }
};

// Energy Units
export const ENERGY_UNITS = {
  KWH: 'kWh',
  MWH: 'MWh',
  GWH: 'GWh'
};

// Power Units
export const POWER_UNITS = {
  KW: 'kW',
  MW: 'MW',
  GW: 'GW'
};

// Currency
export const CURRENCY = {
  SYMBOL: '¬',
  CODE: 'EUR',
  LOCALE: 'pt-PT'
};

// Date Formats
export const DATE_FORMATS = {
  SHORT: 'dd/MM/yyyy',
  LONG: 'dd MMMM yyyy',
  WITH_TIME: 'dd/MM/yyyy HH:mm',
  ISO: "yyyy-MM-dd'T'HH:mm:ss"
};

// Theme Options
export const THEMES = {
  LIGHT: 'light',
  DARK: 'dark',
  SYSTEM: 'system'
};

// Local Storage Keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'eredes_auth_token',
  USER_DATA: 'eredes_user',
  APP_SETTINGS: 'app_settings',
  THEME: 'app_theme',
  SIMULATION_HISTORY: 'simulation_history'
};

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Erro de ligação. Verifique a sua conexão à internet.',
  AUTHENTICATION_FAILED: 'Falha na autenticação. Verifique as suas credenciais.',
  SIMULATION_FAILED: 'A simulação falhou. Tente novamente.',
  EXPORT_FAILED: 'Erro na exportação. Tente novamente.',
  VALIDATION_ERROR: 'Dados inválidos. Verifique os campos.',
  TIMEOUT_ERROR: 'Operação expirou. Tente novamente.',
  GENERIC_ERROR: 'Ocorreu um erro inesperado.'
};

// Success Messages
export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: 'Login efetuado com sucesso!',
  LOGOUT_SUCCESS: 'Logout efetuado com sucesso!',
  SIMULATION_CREATED: 'Simulação criada com sucesso!',
  SIMULATION_COMPLETED: 'Simulação concluída!',
  EXPORT_SUCCESS: 'Dados exportados com sucesso!',
  SETTINGS_SAVED: 'Definições guardadas com sucesso!'
};

// Validation Rules
export const VALIDATION_RULES = {
  CPE: {
    PATTERN: /^PT\d{22}[A-Z]{2}$/,
    MESSAGE: 'CPE deve ter o formato PT seguido de 22 dígitos e 2 letras'
  },
  NIF: {
    PATTERN: /^\d{9}$/,
    MESSAGE: 'NIF deve ter exatamente 9 dígitos'
  },
  PASSWORD: {
    MIN_LENGTH: 4,
    MESSAGE: 'Password deve ter pelo menos 4 caracteres'
  },
  DATE_RANGE: {
    MAX_DAYS: 365,
    MIN_DAYS: 1,
    MESSAGE: 'Período deve estar entre 1 e 365 dias'
  }
};

// Chart Colors
export const CHART_COLORS = {
  PRIMARY: '#2196f3',
  SUCCESS: '#4caf50',
  WARNING: '#ff9800',
  ERROR: '#f44336',
  INFO: '#00bcd4',
  NEUTRAL: '#9e9e9e'
};

// Animation Durations (milliseconds)
export const ANIMATION_DURATIONS = {
  FAST: 150,
  NORMAL: 300,
  SLOW: 500,
  CHART: 1500
};

// File Size Limits (bytes)
export const FILE_SIZE_LIMITS = {
  EXPORT_MAX: 100 * 1024 * 1024, // 100MB
  UPLOAD_MAX: 10 * 1024 * 1024 // 10MB
};

// Pagination
export const PAGINATION = {
  DEFAULT_PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100
};

// Environment Detection
export const ENVIRONMENT = {
  IS_DEVELOPMENT: process.env.NODE_ENV === 'development',
  IS_PRODUCTION: process.env.NODE_ENV === 'production',
  IS_TEST: process.env.NODE_ENV === 'test'
};

// Carbon Emissions Factor (kg CO2 per kWh) - Portugal 2024
export const CARBON_EMISSION_FACTOR = 0.35;

// Energy Conversion Factors
export const ENERGY_CONVERSIONS = {
  KWH_TO_MWH: 0.001,
  MWH_TO_GWH: 0.001,
  KW_TO_MW: 0.001,
  MW_TO_GW: 0.001
};

export default {
  API_CONFIG,
  ROUTES,
  SIMULATION_STATUS,
  EXPORT_FORMATS,
  TARIFF_PERIODS,
  ENERGY_UNITS,
  POWER_UNITS,
  CURRENCY,
  DATE_FORMATS,
  THEMES,
  STORAGE_KEYS,
  ERROR_MESSAGES,
  SUCCESS_MESSAGES,
  VALIDATION_RULES,
  CHART_COLORS,
  ANIMATION_DURATIONS,
  FILE_SIZE_LIMITS,
  PAGINATION,
  ENVIRONMENT,
  CARBON_EMISSION_FACTOR,
  ENERGY_CONVERSIONS
};