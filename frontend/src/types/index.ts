// src/types/index.ts

/**
 * Barrel export file aggregating all type definitions
 * Central point for importing all types across the application
 */

// ============================================================================
// Authentication Types
// ============================================================================
export type {
  User,
  UserProfile,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  RegisterResponse,
  RefreshTokenResponse,
  PasswordResetRequest,
  PasswordResetResponse,
  EmailVerificationResponse,
  SocialLoginRequest,
  SocialLoginResponse,
  UserPreferences,
  UserSettings,
  NotificationSettings,
  PrivacySettings,
  UserActivity,
} from './auth.types';

// ============================================================================
// Dataset Types
// ============================================================================
export type {
  Dataset,
  DatasetListResponse,
  DatasetUploadResponse,
  DatasetPreview,
  DatasetMetadata,
  DatasetStatistics,
  DatasetColumn,
  DatasetDimensions,
  DatasetProcessingStatus,
} from './dataset.types';

// ✅ FIXED: Avoid duplicate exports by not re-exporting from dataset.types
// (these are already exported directly from dataset.types)

// ============================================================================
// EDA (Exploratory Data Analysis) Types
// ============================================================================
export type {
  EDAResult,
  EDAStatus,
  DescriptiveStatistics,
  CorrelationMatrix,
  MissingValues,
  OutlierAnalysis,
  DistributionAnalysis,
  DataQualityReport,
  AnomalyDetection,
} from './eda.types';

// ============================================================================
// Visualization & Chart Types
// ============================================================================
export type {
  ChartConfig,
  ChartData,
  ChartType,
  ChartStyle,
  ExportFormat,
  ChartAxis,
  ChartLegend,
  ChartTooltip,
  ChartColorScheme,
} from './visualization.types';

// ============================================================================
// Common/Utility Types
// ============================================================================

/**
 * API Response wrapper
 */
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  timestamp: number;
  requestId: string;
}

/**
 * Pagination metadata
 */
export interface PaginationMeta {
  page: number;
  limit: number;
  totalItems: number;
  totalPages: number;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
}

/**
 * Sorting configuration
 */
export interface SortConfig<T = any> {
  field: keyof T;
  order: 'asc' | 'desc';
}

/**
 * Filter configuration
 */
export interface FilterConfig {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'contains' | 'startsWith' | 'endsWith';
  value: any;
}

/**
 * File upload configuration
 */
export interface FileUploadConfig {
  maxSize: number; // bytes
  allowedTypes: string[];
  allowedExtensions: string[];
}

/**
 * Date range
 */
export interface DateRange {
  startDate: Date;
  endDate: Date;
}

/**
 * Generic async state
 */
export interface AsyncState<T = any> {
  data: T | null;
  loading: boolean;
  error: Error | null;
}

/**
 * Form field error
 */
export interface FormFieldError {
  field: string;
  message: string;
  code?: string;
}

/**
 * Form submission result
 */
export interface FormSubmitResult {
  success: boolean;
  errors?: FormFieldError[];
  message?: string;
}

// ============================================================================
// Component Props Types
// ============================================================================

/**
 * Button component props
 */
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  className?: string;
  onClick?: () => void;
  children: React.ReactNode;
}

/**
 * Input component props
 */
export interface InputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'date' | 'search';
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  error?: string;
  disabled?: boolean;
  required?: boolean;
  className?: string;
}

/**
 * Modal component props
 */
export interface ModalProps {
  isOpen: boolean;
  title?: string;
  onClose: () => void;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  children: React.ReactNode;
}

/**
 * Card component props
 */
export interface CardProps {
  title?: string;
  subtitle?: string;
  children: React.ReactNode;
  variant?: 'default' | 'elevated' | 'outlined';
  className?: string;
}

/**
 * Table column definition
 */
export interface TableColumn<T = any> {
  key: string;
  label: string;
  width?: string;
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, row: T) => React.ReactNode;
}

/**
 * Table props
 */
export interface TableProps<T = any> {
  columns: TableColumn<T>[];
  data: T[];
  loading?: boolean;
  pagination?: PaginationMeta;
  onSort?: (field: string, order: 'asc' | 'desc') => void;
  onFilter?: (filters: FilterConfig[]) => void;
  onPageChange?: (page: number) => void;
}

// ============================================================================
// Hook Return Types
// ============================================================================

/**
 * useAsync hook return type
 */
export interface UseAsyncReturn<T, E = Error> {
  data: T | null;
  loading: boolean;
  error: E | null;
  retry: () => void;
}

/**
 * useForm hook return type
 */
export interface UseFormReturn<T extends Record<string, any> = {}> {
  values: T;
  errors: Partial<Record<keyof T, string>>;
  touched: Partial<Record<keyof T, boolean>>;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onBlur: (e: React.FocusEvent<HTMLInputElement>) => void;
  setValue: (field: keyof T, value: any) => void;
  setError: (field: keyof T, error: string) => void;
  reset: () => void;
  submit: (callback: (values: T) => void | Promise<void>) => Promise<void>;
  isSubmitting: boolean;
}

/**
 * useFetch hook return type
 */
export interface UseFetchReturn<T = any> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

/**
 * usePagination hook return type
 */
export interface UsePaginationReturn<T = any> {
  items: T[];
  currentPage: number;
  pageSize: number;
  totalPages: number;
  goToPage: (page: number) => void;
  nextPage: () => void;
  previousPage: () => void;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * DeepPartial - Make all properties optional recursively
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * DeepReadonly - Make all properties readonly recursively
 */
export type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

/**
 * Nullable - Make a type nullable
 */
export type Nullable<T> = T | null;

/**
 * Optional - Make a type optional
 */
export type Optional<T> = T | undefined;

/**
 * Result - Either success or error
 */
export type Result<T, E = Error> = { success: true; data: T } | { success: false; error: E };

/**
 * Async result
 */
export type AsyncResult<T, E = Error> = Promise<Result<T, E>>;

/**
 * Dictionary type
 */
export type Dictionary<T = any> = Record<string, T>;

/**
 * ValueOf - Get type of values in an object
 */
export type ValueOf<T> = T[keyof T];

/**
 * Class constructor type
 */
export type Constructor<T = {}> = new (...args: any[]) => T;

/**
 * Omit type helper
 */
export type Omit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

/**
 * Extract keys of specific type from object
 */
export type KeysOfType<T, U> = {
  [K in keyof T]: T[K] extends U ? K : never;
}[keyof T];

// ============================================================================
// Validation Types
// ============================================================================

/**
 * Validator function type
 */
export type Validator<T = any> = (value: T) => string | null;

/**
 * Validators object
 */
export type Validators<T extends Record<string, any>> = {
  [K in keyof T]?: Validator<T[K]>;
};

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  errors: FormFieldError[];
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * Custom event detail
 */
export interface CustomEventDetail<T = any> {
  action: string;
  data?: T;
  timestamp: number;
}

/**
 * Analytics event
 */
export interface AnalyticsEvent {
  name: string;
  category: string;
  properties?: Record<string, any>;
  timestamp: number;
}

// ============================================================================
// Error Types
// ============================================================================

/**
 * Application error
 */
export class AppError extends Error {
  constructor(
    public override message: string, // ✅ FIXED: Added override modifier
    public code: string = 'APP_ERROR',
    public statusCode: number = 500,
    public details?: Record<string, any>
  ) {
    super(message);
    this.name = 'AppError';
    Object.setPrototypeOf(this, AppError.prototype);
  }
}

/**
 * Validation error
 */
export class ValidationError extends AppError {
  constructor(
    public override message: string, // ✅ FIXED: Added override modifier
    public errors: FormFieldError[],
    messageOverride: string = 'Validation failed'
  ) {
    super(messageOverride, 'VALIDATION_ERROR', 400, { errors });
    this.name = 'ValidationError';
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}

/**
 * API error
 */
export class ApiError extends AppError {
  constructor(
    public override message: string, // ✅ FIXED: Added override modifier
    statusCode: number = 500,
    code: string = 'API_ERROR',
    details?: Record<string, any>
  ) {
    super(message, code, statusCode, details);
    this.name = 'ApiError';
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Authentication error
 */
export class AuthenticationError extends AppError {
  constructor(
    public override message: string = 'Authentication failed', // ✅ FIXED: Added override modifier
    details?: Record<string, any>
  ) {
    super(message, 'AUTH_ERROR', 401, details);
    this.name = 'AuthenticationError';
    Object.setPrototypeOf(this, AuthenticationError.prototype);
  }
}

/**
 * Authorization error
 */
export class AuthorizationError extends AppError {
  constructor(
    public override message: string = 'Access denied', // ✅ FIXED: Added override modifier
    details?: Record<string, any>
  ) {
    super(message, 'AUTHZ_ERROR', 403, details);
    this.name = 'AuthorizationError';
    Object.setPrototypeOf(this, AuthorizationError.prototype);
  }
}

/**
 * Not found error
 */
export class NotFoundError extends AppError {
  constructor(
    public override message: string = 'Resource not found', // ✅ FIXED: Added override modifier
    details?: Record<string, any>
  ) {
    super(message, 'NOT_FOUND', 404, details);
    this.name = 'NotFoundError';
    Object.setPrototypeOf(this, NotFoundError.prototype);
  }
}

// ============================================================================
// Enum Types
// ============================================================================

/**
 * User roles
 */
export enum UserRole {
  Admin = 'admin',
  Manager = 'manager',
  User = 'user',
  Viewer = 'viewer',
  Guest = 'guest',
}

/**
 * Dataset status
 */
export enum DatasetStatus {
  Pending = 'pending',
  Processing = 'processing',
  Completed = 'completed',
  Failed = 'failed',
  Archived = 'archived',
}

/**
 * Analysis status
 */
export enum AnalysisStatus {
  Pending = 'pending',
  Running = 'running',
  Completed = 'completed',
  Failed = 'failed',
  Cancelled = 'cancelled',
}

/**
 * Chart types
 */
export enum ChartTypeEnum {
  Bar = 'bar',
  Line = 'line',
  Pie = 'pie',
  Scatter = 'scatter',
  Histogram = 'histogram',
  BoxPlot = 'boxplot',
  Heatmap = 'heatmap',
  AreaChart = 'area',
}

/**
 * Notification types
 */
export enum NotificationType {
  Success = 'success',
  Error = 'error',
  Info = 'info',
  Warning = 'warning',
}

/**
 * HTTP methods
 */
export enum HttpMethod {
  Get = 'GET',
  Post = 'POST',
  Put = 'PUT',
  Patch = 'PATCH',
  Delete = 'DELETE',
  Head = 'HEAD',
  Options = 'OPTIONS',
}

/**
 * Sort orders
 */
export enum SortOrder {
  Ascending = 'asc',
  Descending = 'desc',
}

// ============================================================================
// Configuration Types
// ============================================================================

/**
 * Application configuration
 */
export interface AppConfig {
  apiBaseUrl: string;
  apiTimeout: number;
  requestRetries: number;
  cacheDuration: number;
  sessionTimeout: number;
  environment: 'development' | 'staging' | 'production';
  enableLogging: boolean;
  enableAnalytics: boolean;
  features: {
    enableMFA: boolean;
    enableSocialAuth: boolean;
    enableDataCleaning: boolean;
    enableInsights: boolean;
    enableCollaborations: boolean;
  };
}

/**
 * Feature flags
 */
export interface FeatureFlags {
  [key: string]: boolean;
}

/**
 * Locale configuration
 */
export interface LocaleConfig {
  language: string;
  region: string;
  dateFormat: string;
  timeFormat: string;
  currency: string;
}

// ============================================================================
// Re-exports for convenience
// ============================================================================

// ✅ FIXED: Only re-export from specific type files (not all their exports)
// Import specific types from each file to avoid duplicate exports
export * from './auth.types';
export * from './visualization.types';

// ✅ FIXED: Selectively export from dataset.types to avoid conflicts
// If there are duplicates with eda.types, import specific types instead
