// Date utility functions for the E-REDES application

// Get current date in ISO format
export const getCurrentDate = () => {
  return new Date().toISOString();
};

// Get current date formatted for datetime-local input
export const getCurrentDateTimeLocal = () => {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');

  return `${year}-${month}-${day}T${hours}:${minutes}`;
};

// Convert ISO string to datetime-local format
export const toDateTimeLocal = (isoString) => {
  if (!isoString) return '';

  const date = new Date(isoString);
  if (isNaN(date.getTime())) return '';

  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  const hours = String(date.getHours()).padStart(2, '0');
  const minutes = String(date.getMinutes()).padStart(2, '0');

  return `${year}-${month}-${day}T${hours}:${minutes}`;
};

// Get start of day
export const getStartOfDay = (date = new Date()) => {
  const startOfDay = new Date(date);
  startOfDay.setHours(0, 0, 0, 0);
  return startOfDay;
};

// Get end of day
export const getEndOfDay = (date = new Date()) => {
  const endOfDay = new Date(date);
  endOfDay.setHours(23, 59, 59, 999);
  return endOfDay;
};

// Get start of month
export const getStartOfMonth = (date = new Date()) => {
  const startOfMonth = new Date(date.getFullYear(), date.getMonth(), 1);
  return startOfMonth;
};

// Get end of month
export const getEndOfMonth = (date = new Date()) => {
  const endOfMonth = new Date(date.getFullYear(), date.getMonth() + 1, 0, 23, 59, 59, 999);
  return endOfMonth;
};

// Get last month date range
export const getLastMonthRange = () => {
  const now = new Date();
  const startOfLastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
  const endOfLastMonth = new Date(now.getFullYear(), now.getMonth(), 0, 23, 59, 59, 999);

  return {
    start: startOfLastMonth,
    end: endOfLastMonth
  };
};

// Get last N days range
export const getLastNDaysRange = (days = 30) => {
  const now = new Date();
  const pastDate = new Date(now.getTime() - (days * 24 * 60 * 60 * 1000));

  return {
    start: getStartOfDay(pastDate),
    end: getEndOfDay(now)
  };
};

// Calculate days between dates
export const daysBetween = (startDate, endDate) => {
  const start = new Date(startDate);
  const end = new Date(endDate);

  if (isNaN(start.getTime()) || isNaN(end.getTime())) return 0;

  const timeDifference = end.getTime() - start.getTime();
  return Math.ceil(timeDifference / (1000 * 60 * 60 * 24));
};

// Calculate hours between dates
export const hoursBetween = (startDate, endDate) => {
  const start = new Date(startDate);
  const end = new Date(endDate);

  if (isNaN(start.getTime()) || isNaN(end.getTime())) return 0;

  const timeDifference = end.getTime() - start.getTime();
  return Math.abs(Math.floor(timeDifference / (1000 * 60 * 60)));
};

// Check if date is today
export const isToday = (date) => {
  const today = new Date();
  const checkDate = new Date(date);

  return checkDate.getDate() === today.getDate() &&
         checkDate.getMonth() === today.getMonth() &&
         checkDate.getFullYear() === today.getFullYear();
};

// Check if date is yesterday
export const isYesterday = (date) => {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const checkDate = new Date(date);

  return checkDate.getDate() === yesterday.getDate() &&
         checkDate.getMonth() === yesterday.getMonth() &&
         checkDate.getFullYear() === yesterday.getFullYear();
};

// Check if date is within last week
export const isWithinLastWeek = (date) => {
  const weekAgo = new Date();
  weekAgo.setDate(weekAgo.getDate() - 7);
  const checkDate = new Date(date);

  return checkDate >= weekAgo && checkDate <= new Date();
};

// Check if date is weekend
export const isWeekend = (date) => {
  const checkDate = new Date(date);
  const dayOfWeek = checkDate.getDay();
  return dayOfWeek === 0 || dayOfWeek === 6; // Sunday = 0, Saturday = 6
};

// Get tariff period for given hour (Portuguese electricity tariffs)
export const getTariffPeriod = (hour) => {
  if (hour >= 18 && hour <= 21) {
    return 'peak'; // Ponta
  } else if ((hour >= 7 && hour <= 17) || hour === 22) {
    return 'standard'; // Cheia
  } else {
    return 'off-peak'; // Vazio
  }
};

// Get tariff period for given datetime
export const getTariffPeriodForDate = (date) => {
  const checkDate = new Date(date);
  return getTariffPeriod(checkDate.getHours());
};

// Check if time is peak hours
export const isPeakHours = (date) => {
  return getTariffPeriodForDate(date) === 'peak';
};

// Add days to date
export const addDays = (date, days) => {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
};

// Add months to date
export const addMonths = (date, months) => {
  const result = new Date(date);
  result.setMonth(result.getMonth() + months);
  return result;
};

// Format date range for display
export const formatDateRange = (startDate, endDate, locale = 'pt-PT') => {
  const start = new Date(startDate);
  const end = new Date(endDate);

  if (isNaN(start.getTime()) || isNaN(end.getTime())) {
    return '';
  }

  const options = {
    day: 'numeric',
    month: 'short',
    year: 'numeric'
  };

  const startFormatted = start.toLocaleDateString(locale, options);
  const endFormatted = end.toLocaleDateString(locale, options);

  // If same day, show only one date
  if (start.toDateString() === end.toDateString()) {
    return startFormatted;
  }

  // If same month and year, optimize display
  if (start.getMonth() === end.getMonth() && start.getFullYear() === end.getFullYear()) {
    const startDay = start.getDate();
    const endFormatted = end.toLocaleDateString(locale, options);
    return `${startDay} - ${endFormatted}`;
  }

  return `${startFormatted} - ${endFormatted}`;
};

// Get Portuguese month names
export const getPortugueseMonthNames = () => {
  return [
    'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
    'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
  ];
};

// Get Portuguese day names
export const getPortugueseDayNames = () => {
  return [
    'Domingo', 'Segunda-feira', 'Terça-feira', 'Quarta-feira',
    'Quinta-feira', 'Sexta-feira', 'Sábado'
  ];
};

// Validate date range for simulations
export const validateDateRange = (startDate, endDate) => {
  const errors = {};

  if (!startDate) {
    errors.startDate = 'Data de início é obrigatória';
  }

  if (!endDate) {
    errors.endDate = 'Data de fim é obrigatória';
  }

  if (startDate && endDate) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const now = new Date();

    if (isNaN(start.getTime())) {
      errors.startDate = 'Data de início inválida';
    }

    if (isNaN(end.getTime())) {
      errors.endDate = 'Data de fim inválida';
    }

    if (start >= end) {
      errors.dateRange = 'Data de início deve ser anterior à data de fim';
    }

    if (end > now) {
      errors.dateRange = 'Data de fim não pode ser no futuro';
    }

    const days = daysBetween(start, end);
    if (days > 365) {
      errors.dateRange = 'Período máximo permitido é de 365 dias';
    }

    if (days < 1) {
      errors.dateRange = 'Período mínimo é de 1 dia';
    }
  }

  return Object.keys(errors).length > 0 ? errors : null;
};

// Create date presets for quick selection
export const getDatePresets = () => {
  const now = new Date();

  return {
    today: {
      label: 'Hoje',
      start: getStartOfDay(now),
      end: getEndOfDay(now)
    },
    yesterday: {
      label: 'Ontem',
      start: getStartOfDay(addDays(now, -1)),
      end: getEndOfDay(addDays(now, -1))
    },
    last7Days: {
      label: 'Últimos 7 dias',
      ...getLastNDaysRange(7)
    },
    last30Days: {
      label: 'Últimos 30 dias',
      ...getLastNDaysRange(30)
    },
    lastMonth: {
      label: 'Último mês',
      ...getLastMonthRange()
    },
    thisMonth: {
      label: 'Este mês',
      start: getStartOfMonth(now),
      end: getEndOfMonth(now)
    }
  };
};

export default {
  getCurrentDate,
  getCurrentDateTimeLocal,
  toDateTimeLocal,
  getStartOfDay,
  getEndOfDay,
  getStartOfMonth,
  getEndOfMonth,
  getLastMonthRange,
  getLastNDaysRange,
  daysBetween,
  hoursBetween,
  isToday,
  isYesterday,
  isWithinLastWeek,
  isWeekend,
  getTariffPeriod,
  getTariffPeriodForDate,
  isPeakHours,
  addDays,
  addMonths,
  formatDateRange,
  getPortugueseMonthNames,
  getPortugueseDayNames,
  validateDateRange,
  getDatePresets
};