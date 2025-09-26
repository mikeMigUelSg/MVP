// Currency formatting
export const formatCurrency = (value, currency = 'EUR', locale = 'pt-PT') => {
  if (value === null || value === undefined) return '¬0,00';

  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(Number(value));
};

// Number formatting
export const formatNumber = (value, options = {}) => {
  if (value === null || value === undefined) return '0';

  const {
    locale = 'pt-PT',
    minimumFractionDigits = 0,
    maximumFractionDigits = 2,
    useGrouping = true
  } = options;

  return new Intl.NumberFormat(locale, {
    minimumFractionDigits,
    maximumFractionDigits,
    useGrouping
  }).format(Number(value));
};

// Percentage formatting
export const formatPercentage = (value, options = {}) => {
  if (value === null || value === undefined) return '0%';

  const {
    locale = 'pt-PT',
    minimumFractionDigits = 1,
    maximumFractionDigits = 1
  } = options;

  return new Intl.NumberFormat(locale, {
    style: 'percent',
    minimumFractionDigits,
    maximumFractionDigits
  }).format(Number(value) / 100);
};

// Energy formatting (kWh, MWh, etc.)
export const formatEnergy = (value, unit = 'kWh', options = {}) => {
  if (value === null || value === undefined) return `0 ${unit}`;

  const {
    decimals = 2,
    autoScale = false
  } = options;

  let formattedValue = Number(value);
  let displayUnit = unit;

  // Auto-scale for better readability
  if (autoScale && unit === 'kWh') {
    if (formattedValue >= 1000000) {
      formattedValue /= 1000000;
      displayUnit = 'GWh';
    } else if (formattedValue >= 1000) {
      formattedValue /= 1000;
      displayUnit = 'MWh';
    }
  }

  const formattedNumber = formatNumber(formattedValue, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });

  return `${formattedNumber} ${displayUnit}`;
};

// Power formatting (kW, MW, etc.)
export const formatPower = (value, unit = 'kW', options = {}) => {
  if (value === null || value === undefined) return `0 ${unit}`;

  const {
    decimals = 2,
    autoScale = false
  } = options;

  let formattedValue = Number(value);
  let displayUnit = unit;

  // Auto-scale for better readability
  if (autoScale && unit === 'kW') {
    if (formattedValue >= 1000000) {
      formattedValue /= 1000000;
      displayUnit = 'GW';
    } else if (formattedValue >= 1000) {
      formattedValue /= 1000;
      displayUnit = 'MW';
    }
  }

  const formattedNumber = formatNumber(formattedValue, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });

  return `${formattedNumber} ${displayUnit}`;
};

// Date formatting
export const formatDate = (date, format = 'short', locale = 'pt-PT') => {
  if (!date) return '';

  const dateObj = new Date(date);
  if (isNaN(dateObj.getTime())) return '';

  const formatOptions = {
    short: {
      day: 'numeric',
      month: 'numeric',
      year: 'numeric'
    },
    long: {
      day: 'numeric',
      month: 'long',
      year: 'numeric'
    },
    medium: {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    },
    time: {
      hour: '2-digit',
      minute: '2-digit'
    },
    datetime: {
      day: 'numeric',
      month: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    },
    relative: null // Special case handled separately
  };

  if (format === 'relative') {
    return formatRelativeTime(dateObj);
  }

  const options = formatOptions[format] || formatOptions.short;

  return new Intl.DateTimeFormat(locale, options).format(dateObj);
};

// Relative time formatting (e.g., "2 hours ago")
export const formatRelativeTime = (date, locale = 'pt-PT') => {
  if (!date) return '';

  const dateObj = new Date(date);
  if (isNaN(dateObj.getTime())) return '';

  const now = new Date();
  const diffInSeconds = Math.floor((now - dateObj) / 1000);

  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

  if (diffInSeconds < 60) {
    return rtf.format(-diffInSeconds, 'second');
  } else if (diffInSeconds < 3600) {
    return rtf.format(-Math.floor(diffInSeconds / 60), 'minute');
  } else if (diffInSeconds < 86400) {
    return rtf.format(-Math.floor(diffInSeconds / 3600), 'hour');
  } else if (diffInSeconds < 2592000) {
    return rtf.format(-Math.floor(diffInSeconds / 86400), 'day');
  } else if (diffInSeconds < 31536000) {
    return rtf.format(-Math.floor(diffInSeconds / 2592000), 'month');
  } else {
    return rtf.format(-Math.floor(diffInSeconds / 31536000), 'year');
  }
};

// Duration formatting (e.g., "2h 30m")
export const formatDuration = (milliseconds, format = 'short') => {
  if (!milliseconds || milliseconds < 0) return '0s';

  const totalSeconds = Math.floor(milliseconds / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  if (format === 'long') {
    const parts = [];
    if (hours > 0) parts.push(`${hours} hora${hours !== 1 ? 's' : ''}`);
    if (minutes > 0) parts.push(`${minutes} minuto${minutes !== 1 ? 's' : ''}`);
    if (seconds > 0) parts.push(`${seconds} segundo${seconds !== 1 ? 's' : ''}`);
    return parts.join(' e ') || '0 segundos';
  }

  // Short format
  const parts = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds}s`);
  return parts.join(' ');
};

// File size formatting
export const formatFileSize = (bytes, locale = 'pt-PT') => {
  if (!bytes || bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const base = 1024;

  let size = Math.abs(bytes);
  let unitIndex = 0;

  while (size >= base && unitIndex < units.length - 1) {
    size /= base;
    unitIndex++;
  }

  const formattedSize = formatNumber(size, {
    locale,
    minimumFractionDigits: unitIndex === 0 ? 0 : 1,
    maximumFractionDigits: unitIndex === 0 ? 0 : 1
  });

  return `${formattedSize} ${units[unitIndex]}`;
};

// CPE formatting (add spaces for readability)
export const formatCPE = (cpe) => {
  if (!cpe || typeof cpe !== 'string') return '';

  const cleaned = cpe.replace(/\s/g, '').toUpperCase();

  // Validate CPE format
  if (!/^PT\d{22}[A-Z]{2}$/.test(cleaned)) {
    return cpe; // Return as-is if invalid format
  }

  // Format: PT XXXX XXXX XXXX XXXX XXXX XX XX
  return cleaned.replace(/^(PT)(\d{4})(\d{4})(\d{4})(\d{4})(\d{6})([A-Z]{2})$/, '$1 $2 $3 $4 $5 $6 $7');
};

// Phone number formatting (Portuguese format)
export const formatPhoneNumber = (phone) => {
  if (!phone || typeof phone !== 'string') return '';

  const cleaned = phone.replace(/\D/g, '');

  // Portuguese mobile: 9XX XXX XXX
  if (cleaned.length === 9 && cleaned.startsWith('9')) {
    return cleaned.replace(/^(\d{3})(\d{3})(\d{3})$/, '$1 $2 $3');
  }

  // Portuguese landline: 2XX XXX XXX
  if (cleaned.length === 9 && (cleaned.startsWith('2') || cleaned.startsWith('3'))) {
    return cleaned.replace(/^(\d{3})(\d{3})(\d{3})$/, '$1 $2 $3');
  }

  return phone; // Return as-is if doesn't match expected formats
};

// Status formatting with colors
export const formatStatus = (status) => {
  const statusMap = {
    idle: { text: 'Inativo', color: '#9e9e9e' },
    pending: { text: 'Pendente', color: '#ff9800' },
    running: { text: 'Em execução', color: '#2196f3' },
    completed: { text: 'Concluído', color: '#4caf50' },
    failed: { text: 'Falhou', color: '#f44336' },
    cancelled: { text: 'Cancelado', color: '#9e9e9e' }
  };

  return statusMap[status] || { text: status, color: '#9e9e9e' };
};

// Truncate text with ellipsis
export const truncateText = (text, maxLength = 50, suffix = '...') => {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength - suffix.length) + suffix;
};

export default {
  formatCurrency,
  formatNumber,
  formatPercentage,
  formatEnergy,
  formatPower,
  formatDate,
  formatRelativeTime,
  formatDuration,
  formatFileSize,
  formatCPE,
  formatPhoneNumber,
  formatStatus,
  truncateText
};