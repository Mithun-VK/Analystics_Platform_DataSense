// src/utils/constants.ts
/**
 * Application Constants
 * Centralized configuration for API URLs, chart types, file size limits, and other constants
 */
const getEnvVariable = (key: string, defaultValue: string): string => {
  return process.env[key] || defaultValue;
};

// ============================================================================
// API Configuration
// ============================================================================

export const API_CONFIG = {
  // Base URL
  BASE_URL: getEnvVariable('VITE_API_URL', 'http://localhost:8000/api/v1'),
  
  // Endpoints
  ENDPOINTS: {
    // Auth
    AUTH_LOGIN: '/auth/login',
    AUTH_REGISTER: '/auth/register',
    AUTH_LOGOUT: '/auth/logout',
    AUTH_REFRESH_TOKEN: '/auth/refresh',
    AUTH_VERIFY_EMAIL: '/auth/verify-email',
    AUTH_RESET_PASSWORD: '/auth/reset-password',
    AUTH_CHANGE_PASSWORD: '/auth/change-password',
    AUTH_SETUP_2FA: '/auth/setup-2fa',
    AUTH_VERIFY_2FA: '/auth/verify-2fa',
    AUTH_DISABLE_2FA: '/auth/disable-2fa',

    // Users
    USERS_PROFILE: '/users/profile',
    USERS_UPDATE_PROFILE: '/users/profile',
    USERS_UPLOAD_AVATAR: '/users/avatar',
    USERS_DELETE_AVATAR: '/users/avatar',
    USERS_PREFERENCES: '/users/preferences',
    USERS_SETTINGS: '/users/settings',
    USERS_NOTIFICATIONS: '/users/notifications',
    USERS_PRIVACY: '/users/privacy',

    // Datasets
    DATASETS_LIST: '/datasets',
    DATASETS_CREATE: '/datasets',
    DATASETS_GET: '/datasets/:id',
    DATASETS_UPDATE: '/datasets/:id',
    DATASETS_DELETE: '/datasets/:id',
    DATASETS_UPLOAD: '/datasets/upload',
    DATASETS_PREVIEW: '/datasets/:id/preview',
    DATASETS_EXPORT: '/datasets/:id/export',
    DATASETS_DUPLICATE: '/datasets/:id/duplicate',
    DATASETS_SEARCH: '/datasets/search',

    // EDA
    EDA_ANALYZE: '/eda/analyze',
    EDA_RESULTS: '/eda/:jobId',
    EDA_HISTORY: '/eda/:datasetId/history',
    EDA_COMPARE: '/eda/compare',
    EDA_EXPORT: '/eda/:jobId/export',

    // Visualizations
    VIS_GENERATE: '/visualizations/generate',
    VIS_LIST: '/visualizations',
    VIS_GET: '/visualizations/:id',
    VIS_UPDATE: '/visualizations/:id',
    VIS_DELETE: '/visualizations/:id',
    VIS_EXPORT: '/visualizations/:id/export',
    VIS_TEMPLATES: '/visualizations/templates',
    VIS_DASHBOARDS: '/visualizations/dashboards',

    // Insights
    INSIGHTS_GENERATE: '/insights/generate',
    INSIGHTS_LIST: '/insights',
    INSIGHTS_GET: '/insights/:id',
    INSIGHTS_DELETE: '/insights/:id',

    // Cleaning
    CLEANING_DETECT_ISSUES: '/cleaning/detect-issues',
    CLEANING_HANDLE_MISSING: '/cleaning/handle-missing',
    CLEANING_REMOVE_DUPLICATES: '/cleaning/remove-duplicates',
    CLEANING_OPERATIONS: '/cleaning/operations/:datasetId',
    CLEANING_APPLY: '/cleaning/apply',
    CLEANING_ROLLBACK: '/cleaning/rollback/:operationId',
  },

  // Request timeout
  TIMEOUT: 30000, // 30 seconds

  // Retry configuration
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second

  // Request headers
  HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
};

// ============================================================================
// Chart Configuration
// ============================================================================

export const CHART_TYPES = [
  'bar',
  'line',
  'pie',
  'donut',
  'scatter',
  'bubble',
  'area',
  'areaStacked',
  'histogram',
  'boxplot',
  'violin',
  'heatmap',
  'treemap',
  'sunburst',
  'sankey',
  'gauge',
  'indicator',
  'waterfall',
  'funnel',
  'timeline',
  'network',
  'map',
  'candlestick',
  'qqplot',
  'hexbin',
  'contour',
  'parallel',
  'radial',
] as const;

export const CHART_CATEGORIES = {
  comparison: ['bar', 'line', 'scatter'],
  composition: ['pie', 'donut', 'treemap', 'sunburst'],
  distribution: ['histogram', 'boxplot', 'violin', 'qqplot'],
  relationship: ['scatter', 'bubble', 'heatmap', 'network'],
  trend: ['line', 'area', 'areaStacked', 'candlestick'],
  progression: ['waterfall', 'funnel', 'gauge', 'timeline'],
  statistical: ['boxplot', 'violin', 'qqplot'],
  geographic: ['map'],
};

export const CHART_COLORS = {
  default: [
    '#3B82F6', // Blue
    '#EF4444', // Red
    '#10B981', // Green
    '#F59E0B', // Amber
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#14B8A6', // Teal
    '#F97316', // Orange
  ],
  pastel: [
    '#FCA5A5', // Light Red
    '#FBCFE8', // Light Pink
    '#E9D5FF', // Light Purple
    '#DBEAFE', // Light Blue
    '#D1FAE5', // Light Green
    '#FEF08A', // Light Yellow
  ],
  dark: [
    '#1E3A8A', // Dark Blue
    '#7F1D1D', // Dark Red
    '#065F46', // Dark Green
    '#78350F', // Dark Amber
    '#4C1D95', // Dark Purple
  ],
};

export const EXPORT_FORMATS = ['png', 'jpg', 'pdf', 'svg', 'json', 'csv', 'html'] as const;

// ============================================================================
// File Size Limits
// ============================================================================

export const FILE_SIZE_LIMITS = {
  AVATAR: 5 * 1024 * 1024, // 5 MB
  DATASET: 500 * 1024 * 1024, // 500 MB
  DOCUMENT: 50 * 1024 * 1024, // 50 MB
  IMAGE: 10 * 1024 * 1024, // 10 MB
  VIDEO: 1024 * 1024 * 1024, // 1 GB
} as const;

export const ALLOWED_FILE_TYPES = {
  DATASET: ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'application/json', 'application/octet-stream'],
  IMAGE: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/svg+xml'],
  DOCUMENT: ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'],
} as const;

export const ALLOWED_FILE_EXTENSIONS = {
  DATASET: ['.csv', '.xls', '.xlsx', '.json', '.parquet', '.sql'],
  IMAGE: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'],
  DOCUMENT: ['.pdf', '.doc', '.docx', '.txt'],
} as const;

// ============================================================================
// Date & Time Constants
// ============================================================================

export const DATE_FORMATS = {
  SHORT: 'MMM DD, YYYY',
  LONG: 'MMMM DD, YYYY',
  FULL: 'dddd, MMMM DD, YYYY',
  ISO: 'YYYY-MM-DD',
  TIME: 'HH:mm',
  TIME_SECONDS: 'HH:mm:ss',
  DATETIME: 'MMM DD, YYYY HH:mm',
} as const;

export const TIME_ZONES = [
  'UTC',
  'EST',
  'CST',
  'MST',
  'PST',
  'GMT',
  'CET',
  'IST',
  'JST',
  'AEST',
] as const;

// ============================================================================
// Pagination Constants
// ============================================================================

export const PAGINATION = {
  DEFAULT_PAGE: 1,
  DEFAULT_LIMIT: 10,
  DEFAULT_LIMIT_LARGE: 50,
  MAX_LIMIT: 100,
  PAGE_SIZE_OPTIONS: [10, 20, 50, 100],
} as const;

// ============================================================================
// Validation Constants
// ============================================================================

export const VALIDATION = {
  PASSWORD: {
    MIN_LENGTH: 8,
    MAX_LENGTH: 128,
    REQUIRE_UPPERCASE: true,
    REQUIRE_LOWERCASE: true,
    REQUIRE_NUMBERS: true,
    REQUIRE_SPECIAL_CHARS: false,
  },
  USERNAME: {
    MIN_LENGTH: 3,
    MAX_LENGTH: 20,
  },
  EMAIL: {
    MAX_LENGTH: 254,
  },
  PHONE: {
    MIN_LENGTH: 10,
    MAX_LENGTH: 15,
  },
  TEXT: {
    MIN_LENGTH: 1,
    MAX_LENGTH: 5000,
  },
  DATASET_NAME: {
    MIN_LENGTH: 1,
    MAX_LENGTH: 255,
  },
} as const;

// ============================================================================
// UI & Theme Constants
// ============================================================================

export const THEMES = {
  LIGHT: 'light',
  DARK: 'dark',
  AUTO: 'auto',
} as const;

export const FONT_SIZES = {
  EXTRA_SMALL: 'xs',
  SMALL: 'sm',
  BASE: 'base',
  LARGE: 'lg',
  EXTRA_LARGE: 'xl',
} as const;

export const BREAKPOINTS = {
  XS: 320,
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

// ============================================================================
// Status & Severity Constants
// ============================================================================

export const STATUS = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
} as const;

export const SEVERITY = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

export const NOTIFICATION_TYPE = {
  SUCCESS: 'success',
  ERROR: 'error',
  INFO: 'info',
  WARNING: 'warning',
} as const;

// ============================================================================
// Permission Constants
// ============================================================================

export const PERMISSIONS = {
  READ: 'read',
  WRITE: 'write',
  DELETE: 'delete',
  SHARE: 'share',
  ADMIN: 'admin',
} as const;

export const ROLES = {
  ADMIN: 'admin',
  MANAGER: 'manager',
  USER: 'user',
  VIEWER: 'viewer',
  GUEST: 'guest',
} as const;

// ============================================================================
// Feature Flags
// ============================================================================

export const FEATURES = {
  ENABLE_MFA: true,
  ENABLE_SOCIAL_AUTH: true,
  ENABLE_COLLABORATION: true,
  ENABLE_INSIGHTS: true,
  ENABLE_CLEANING: true,
  ENABLE_EXPORT: true,
  ENABLE_SHARING: true,
} as const;

// ============================================================================
// Cache Duration Constants (in milliseconds)
// ============================================================================

export const CACHE_DURATION = {
  SHORT: 5 * 60 * 1000, // 5 minutes
  MEDIUM: 15 * 60 * 1000, // 15 minutes
  LONG: 60 * 60 * 1000, // 1 hour
  VERY_LONG: 24 * 60 * 60 * 1000, // 24 hours
} as const;

// ============================================================================
// Debounce & Throttle Constants (in milliseconds)
// ============================================================================

export const DEBOUNCE_DELAY = {
  SHORT: 300,
  MEDIUM: 500,
  LONG: 1000,
} as const;

export const THROTTLE_DELAY = {
  SHORT: 100,
  MEDIUM: 300,
  LONG: 500,
} as const;

// ============================================================================
// Error Messages
// ============================================================================

export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your connection.',
  SERVER_ERROR: 'Server error. Please try again later.',
  UNAUTHORIZED: 'You are not authorized to perform this action.',
  FORBIDDEN: 'Access denied.',
  NOT_FOUND: 'Resource not found.',
  VALIDATION_ERROR: 'Please check your input and try again.',
  UNKNOWN_ERROR: 'An unknown error occurred.',
} as const;

// ============================================================================
// Success Messages
// ============================================================================

export const SUCCESS_MESSAGES = {
  LOGIN_SUCCESS: 'Login successful!',
  LOGOUT_SUCCESS: 'Logout successful!',
  REGISTRATION_SUCCESS: 'Registration successful! Please verify your email.',
  PROFILE_UPDATED: 'Profile updated successfully!',
  PASSWORD_CHANGED: 'Password changed successfully!',
  FILE_UPLOADED: 'File uploaded successfully!',
  ANALYSIS_COMPLETED: 'Analysis completed successfully!',
  CHART_GENERATED: 'Chart generated successfully!',
  DATA_EXPORTED: 'Data exported successfully!',
  OPERATION_SUCCESS: 'Operation completed successfully!',
} as const;

// ============================================================================
// Analytics Events
// ============================================================================

export const ANALYTICS_EVENTS = {
  USER_LOGIN: 'user_login',
  USER_LOGOUT: 'user_logout',
  USER_SIGNUP: 'user_signup',
  DATASET_UPLOADED: 'dataset_uploaded',
  ANALYSIS_STARTED: 'analysis_started',
  ANALYSIS_COMPLETED: 'analysis_completed',
  CHART_GENERATED: 'chart_generated',
  DATA_EXPORTED: 'data_exported',
  SHARING_ENABLED: 'sharing_enabled',
} as const;

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULTS = {
  SESSION_TIMEOUT: 30 * 60 * 1000, // 30 minutes
  AUTO_LOGOUT: true,
  REMEMBER_ME_DURATION: 7 * 24 * 60 * 60 * 1000, // 7 days
  CHART_HEIGHT: 400,
  CHART_WIDTH: 800,
  ITEMS_PER_PAGE: 10,
  ANALYSIS_TIMEOUT: 5 * 60 * 1000, // 5 minutes
} as const;
