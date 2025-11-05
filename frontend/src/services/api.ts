// src/services/api.ts

/**
 * API Service Layer
 * ✅ PRODUCTION READY: Comprehensive API client with retry logic, auth, and error handling
 * ✅ TESTING MODE: Authentication only (via environment variable)
 */

import axios from 'axios';
import type {
  AxiosInstance,
  AxiosRequestConfig,
  AxiosError,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from 'axios';
import { QueryClient } from '@tanstack/react-query';
import { authStore } from '@/store/authStore';
import { uiStore } from '@/store/uiStore';

// ============================================================================
// Type Definitions
// ============================================================================

interface RequestLog {
  method: string;
  url: string;
  timestamp: number;
  duration: number;
  status?: number;
  error?: string;
}

interface ApiError extends Error {
  status?: number;
  code?: string;
  originalError?: AxiosError;
}

interface ApiConfig extends AxiosRequestConfig {
  raw?: boolean;
  silent?: boolean;
  skipRetry?: boolean;
}

interface ExtendedInternalConfig extends InternalAxiosRequestConfig {
  requestStartTime?: number;
  retryCount?: number;
  raw?: boolean;
  silent?: boolean;
  skipRetry?: boolean;
}

interface ApiResponseData {
  message?: string;
  error?: string;
  [key: string]: unknown;
}

// ============================================================================
// QueryClient Setup
// ============================================================================

/**
 * ✅ Create and export QueryClient for React Query
 * This is the main client for data fetching and caching
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      gcTime: 1000 * 60 * 10, // 10 minutes (garbage collection)
      retry: 1,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

export default queryClient;

// ============================================================================
// Environment Configuration
// ============================================================================

/**
 * ✅ Get API base URL from environment
 */
const getApiBaseUrl = (): string => {
  // Production API URL
  const apiUrl = import.meta.env.VITE_API_URL;
  if (apiUrl && typeof apiUrl === 'string') {
    return apiUrl;
  }

  // Development fallback
  return 'http://localhost:8000/api/v1';
};

/**
 * ✅ Check if in development mode
 */
const isDevelopment = (): boolean => {
  return import.meta.env.DEV;
};

/**
 * ✅ Get authentication mode (testing or production)
 */
const getAuthMode = (): 'testing' | 'production' => {
  const mode = import.meta.env.VITE_AUTH_MODE;
  return mode === 'testing' ? 'testing' : 'production';
};

// ============================================================================
// Constants
// ============================================================================

const API_BASE_URL = getApiBaseUrl();
const AUTH_MODE = getAuthMode();
const REQUEST_TIMEOUT = 30000; // 30 seconds
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second
const BACKOFF_MULTIPLIER = 2;
const MAX_REQUEST_LOGS = 50;

// Retry-able status codes
const RETRYABLE_STATUS_CODES: number[] = [408, 429, 500, 502, 503, 504];

// Request logs for debugging
const requestLogs: RequestLog[] = [];

if (isDevelopment()) {
  console.debug('[API] Configuration:', {
    baseUrl: API_BASE_URL,
    authMode: AUTH_MODE,
    timeout: REQUEST_TIMEOUT,
  });
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * ✅ Calculate exponential backoff delay with jitter
 */
const calculateRetryDelay = (
  retryCount: number,
  baseDelay: number = RETRY_DELAY,
  multiplier: number = BACKOFF_MULTIPLIER
): number => {
  const exponentialDelay = baseDelay * Math.pow(multiplier, retryCount);
  const jitter = Math.random() * 1000; // Random jitter (0-1s)
  return exponentialDelay + jitter;
};

/**
 * ✅ Log request/response for debugging
 */
const logRequest = (log: RequestLog): void => {
  requestLogs.push(log);

  // Keep only last MAX_REQUEST_LOGS entries
  if (requestLogs.length > MAX_REQUEST_LOGS) {
    requestLogs.shift();
  }

  if (isDevelopment()) {
    console.debug('[API Request Log]', log);
  }
};

/**
 * ✅ Check if error is retryable
 */
const isRetryableError = (error: unknown): boolean => {
  if (!axios.isAxiosError(error)) {
    // Network error - retryable
    return true;
  }

  if (!error.response) {
    // Network error - retryable
    return true;
  }

  return RETRYABLE_STATUS_CODES.includes(error.response.status);
};

/**
 * ✅ Format error message from various error types
 */
const getErrorMessage = (error: unknown): string => {
  if (!axios.isAxiosError(error)) {
    if (error instanceof Error) {
      return error.message;
    }
    return 'An error occurred';
  }

  // Check response data for error message
  if (error.response?.data && typeof error.response.data === 'object') {
    const data = error.response.data as ApiResponseData;
    const message = data['message'] ?? data['error'] ?? 'An error occurred';

    if (typeof message === 'string') {
      return message;
    }

    return String(message);
  }

  // Network error without response
  if (error.request && !error.response) {
    return 'Network error. Please check your connection.';
  }

  return error.message || 'An error occurred';
};

/**
 * ✅ Check if value is a function
 */
const isCallable = (value: unknown): value is (...args: unknown[]) => unknown => {
  return typeof value === 'function';
};

/**
 * ✅ Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

/**
 * ✅ Generate request ID for tracking
 */
const generateRequestId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// ============================================================================
// API Instance Creation
// ============================================================================

/**
 * ✅ Create and configure Axios instance
 */
const createApiInstance = (): AxiosInstance => {
  return axios.create({
    baseURL: API_BASE_URL,
    timeout: REQUEST_TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
    },
  });
};

export const apiClient = createApiInstance();

// ============================================================================
// Request Interceptor
// ============================================================================

/**
 * ✅ Request interceptor - Add auth token and logging
 */
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig): InternalAxiosRequestConfig => {
    try {
      // Add authentication token
      const authState = authStore.getState();
      const token = authState?.token;

      if (token && typeof token === 'string') {
        config.headers.Authorization = `Bearer ${token}`;
      }

      // Add request tracking ID
      config.headers['X-Request-ID'] = generateRequestId();

      // Store request start time for duration calculation
      const extConfig = config as ExtendedInternalConfig;
      extConfig.requestStartTime = Date.now();

      // Debug logging
      if (isDevelopment()) {
        console.debug(
          `[API] ${(config.method || 'GET').toUpperCase()} ${config.url}`,
          config.data ? { data: config.data } : ''
        );
      }
    } catch (error) {
      console.warn('[API] Request interceptor error:', error);
    }

    return config;
  },
  (error: unknown): Promise<never> => {
    console.error('[API] Request interceptor failed:', error);
    return Promise.reject(error);
  }
);

// ============================================================================
// Response Interceptor
// ============================================================================

/**
 * ✅ Response interceptor - Handle success and errors with retry logic
 */
apiClient.interceptors.response.use(
  (response: AxiosResponse): AxiosResponse => {
    // Calculate request duration
    const extConfig = response.config as ExtendedInternalConfig;
    const startTime = extConfig.requestStartTime ?? Date.now();
    const duration = Date.now() - startTime;

    // Log successful response
    logRequest({
      method: response.config.method || 'GET',
      url: response.config.url || '',
      timestamp: startTime,
      duration,
      status: response.status,
    });

    if (isDevelopment()) {
      console.debug(`[API] ✅ Response (${duration}ms):`, {
        status: response.status,
        data: response.data,
      });
    }

    return response;
  },

  async (error: unknown): Promise<AxiosResponse> => {
    // Handle non-Axios errors
    if (!axios.isAxiosError(error)) {
      console.error('[API] Non-Axios error:', error);
      return Promise.reject(error);
    }

    const config = error.config as ExtendedInternalConfig | undefined;

    // Calculate request duration
    const startTime = config?.requestStartTime ?? Date.now();
    const duration = Date.now() - startTime;

    // Log failed request
    logRequest({
      method: config?.method || 'GET',
      url: config?.url || '',
      timestamp: startTime,
      duration,
      status: error.response?.status,
      error: getErrorMessage(error),
    });

    // ========================================================================
    // Handle 401 Unauthorized - Token Refresh
    // ========================================================================

    if (error.response?.status === 401 && !config?.skipRetry) {
      const isRefreshAttempt = config?.url?.includes('/auth/refresh');

      if (!isRefreshAttempt) {
        try {
          if (isDevelopment()) {
            console.debug('[API] Token expired, attempting refresh...');
          }

          // Refresh token
          const authState = authStore.getState();
          const refreshTokenFn = authState?.refreshToken;

          if (refreshTokenFn && isCallable(refreshTokenFn)) {
            await refreshTokenFn();
          }

          // Get new token and retry original request
          const newAuthState = authStore.getState();
          const newToken = newAuthState?.token;

          if (newToken && typeof newToken === 'string' && config) {
            config.headers.Authorization = `Bearer ${newToken}`;
            config.skipRetry = true;

            if (isDevelopment()) {
              console.debug('[API] Token refreshed, retrying original request');
            }

            return apiClient(config);
          }
        } catch (refreshError) {
          console.error('[API] Token refresh failed:', refreshError);

          // Clear auth and redirect to login
          try {
            const authState = authStore.getState();
            const logoutFn = authState?.logout;

            if (logoutFn && isCallable(logoutFn)) {
              await logoutFn();
            }
          } catch (logoutError) {
            console.error('[API] Logout failed:', logoutError);
          }

          window.location.href = '/login?reason=session-expired';
          return Promise.reject(refreshError);
        }
      } else {
        // Refresh token request itself failed - redirect to login
        try {
          const authState = authStore.getState();
          const logoutFn = authState?.logout;

          if (logoutFn && isCallable(logoutFn)) {
            await logoutFn();
          }
        } catch (logoutError) {
          console.error('[API] Logout failed:', logoutError);
        }

        window.location.href = '/login?reason=unauthorized';
        return Promise.reject(error);
      }
    }

    // ========================================================================
    // Handle Retryable Errors
    // ========================================================================

    const retryCount = config?.retryCount ?? 0;

    if (
      isRetryableError(error) &&
      retryCount < MAX_RETRIES &&
      !config?.skipRetry
    ) {
      const retryDelay = calculateRetryDelay(retryCount);

      console.warn(
        `[API] Request failed (${error.response?.status || error.code}), ` +
        `retrying in ${Math.round(retryDelay)}ms (attempt ${retryCount + 1}/${MAX_RETRIES})`
      );

      if (!config) {
        return Promise.reject(error);
      }

      config.retryCount = retryCount + 1;

      await sleep(retryDelay);
      return apiClient(config);
    }

    // ========================================================================
    // Handle Final Error
    // ========================================================================

    const errorMessage = getErrorMessage(error);
    const apiError = new Error(errorMessage) as ApiError;
    apiError.status = error.response?.status;
    apiError.code = error.code;
    apiError.originalError = error;

    console.error('[API Error]', {
      status: error.response?.status,
      message: errorMessage,
      code: error.code,
      url: config?.url,
    });

    // ========================================================================
    // Show Error Notification
    // ========================================================================

    if (!config?.silent) {
      try {
        const uiState = uiStore.getState();
        const addNotificationFn = uiState?.addNotification;

        if (addNotificationFn && isCallable(addNotificationFn)) {
          let notificationMessage = errorMessage;

          // Provide user-friendly error messages
          switch (error.response?.status) {
            case 404:
              notificationMessage = 'Resource not found';
              break;
            case 403:
              notificationMessage = 'Access denied';
              break;
            case 429:
              notificationMessage = 'Too many requests. Please try again later';
              break;
            case 503:
              notificationMessage = 'Service unavailable. Please try again later';
              break;
          }

          addNotificationFn({
            type: 'error',
            message: notificationMessage,
            duration: 5000,
          });
        }
      } catch (notificationError) {
        console.error('[API] Notification failed:', notificationError);
      }
    }

    return Promise.reject(apiError);
  }
);

// ============================================================================
// API Request Helpers
// ============================================================================

/**
 * ✅ Generic API request wrapper
 */
export const apiRequest = async <T = unknown>(
  config: ApiConfig
): Promise<T> => {
  try {
    const response = await apiClient(config);
    return response.data as T;
  } catch (error) {
    throw error;
  }
};

/**
 * ✅ GET request helper
 */
export const apiGet = async <T = unknown>(
  url: string,
  config?: Omit<ApiConfig, 'method' | 'url'>
): Promise<T> => {
  return apiRequest({ ...config, method: 'GET', url });
};

/**
 * ✅ POST request helper
 */
export const apiPost = async <T = unknown>(
  url: string,
  data?: unknown,
  config?: Omit<ApiConfig, 'method' | 'url' | 'data'>
): Promise<T> => {
  return apiRequest({ ...config, method: 'POST', url, data });
};

/**
 * ✅ PUT request helper
 */
export const apiPut = async <T = unknown>(
  url: string,
  data?: unknown,
  config?: Omit<ApiConfig, 'method' | 'url' | 'data'>
): Promise<T> => {
  return apiRequest({ ...config, method: 'PUT', url, data });
};

/**
 * ✅ PATCH request helper
 */
export const apiPatch = async <T = unknown>(
  url: string,
  data?: unknown,
  config?: Omit<ApiConfig, 'method' | 'url' | 'data'>
): Promise<T> => {
  return apiRequest({ ...config, method: 'PATCH', url, data });
};

/**
 * ✅ DELETE request helper
 */
export const apiDelete = async <T = unknown>(
  url: string,
  config?: Omit<ApiConfig, 'method' | 'url'>
): Promise<T> => {
  return apiRequest({ ...config, method: 'DELETE', url });
};

// ============================================================================
// File Operations
// ============================================================================

/**
 * ✅ Upload file with progress tracking
 */
export const apiUploadFile = async <T = unknown>(
  url: string,
  file: File,
  metadata?: Record<string, unknown>,
  onProgress?: (progress: number) => void
): Promise<T> => {
  const formData = new FormData();
  formData.append('file', file);

  // Add metadata fields
  if (metadata) {
    Object.entries(metadata).forEach(([key, value]) => {
      if (typeof value === 'string' || typeof value === 'number') {
        formData.append(key, String(value));
      } else {
        formData.append(key, JSON.stringify(value));
      }
    });
  }

  return apiRequest({
    method: 'POST',
    url,
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (progressEvent.total && progressEvent.total > 0) {
        const progress = Math.round(
          (progressEvent.loaded / progressEvent.total) * 100
        );
        onProgress?.(progress);
      }
    },
  });
};

/**
 * ✅ Download file with progress tracking
 */
export const apiDownloadFile = async (
  url: string,
  filename: string,
  onProgress?: (progress: number) => void
): Promise<void> => {
  try {
    const response = await apiClient.get(url, {
      responseType: 'blob',
      onDownloadProgress: (progressEvent) => {
        if (progressEvent.total && progressEvent.total > 0) {
          const progress = Math.round(
            (progressEvent.loaded / progressEvent.total) * 100
          );
          onProgress?.(progress);
        }
      },
    });

    // Create blob URL and trigger download
    const blobUrl = URL.createObjectURL(response.data as Blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = filename;
    link.click();

    // Cleanup
    URL.revokeObjectURL(blobUrl);

    if (isDevelopment()) {
      console.debug('[API] File downloaded:', filename);
    }
  } catch (error) {
    console.error('[API] Download failed:', error);
    throw error;
  }
};

// ============================================================================
// Logging & Debugging
// ============================================================================

/**
 * ✅ Get all request logs
 */
export const getApiLogs = (): RequestLog[] => {
  return [...requestLogs];
};

/**
 * ✅ Clear request logs
 */
export const clearApiLogs = (): void => {
  requestLogs.length = 0;
  if (isDevelopment()) {
    console.debug('[API] Logs cleared');
  }
};

/**
 * ✅ Get API instance for direct access
 */
export const getApiInstance = (): AxiosInstance => {
  return apiClient;
};

/**
 * ✅ Create cancel token for request cancellation
 */
export const createCancelToken = () => {
  const source = axios.CancelToken.source();
  return {
    token: source.token,
    cancel: source.cancel,
  };
};

/**
 * ✅ Health check endpoint
 */
export const checkApiHealth = async (): Promise<boolean> => {
  try {
    await apiGet('/health', { silent: true, timeout: 5000 });
    return true;
  } catch {
    return false;
  }
};

/**
 * ✅ Get API configuration
 */
export const getApiConfig = () => {
  return {
    baseUrl: API_BASE_URL,
    authMode: AUTH_MODE,
    isDevelopment: isDevelopment(),
    timeout: REQUEST_TIMEOUT,
    maxRetries: MAX_RETRIES,
  };
};

// ============================================================================
// Export Default (QueryClient for App.tsx)
// ============================================================================