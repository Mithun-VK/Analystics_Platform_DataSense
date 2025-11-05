// src/types/dataset.types.ts

/**
 * Dataset Management Types
 * Comprehensive type definitions for all dataset-related operations
 * Production-ready with full type safety and complete coverage
 */

// ============================================================================
// Core Dataset Types
// ============================================================================

/**
 * Dataset column definition
 */
export interface DatasetColumn {
  name: string;
  type:
    | 'string'
    | 'number'
    | 'integer'
    | 'float'
    | 'boolean'
    | 'date'
    | 'datetime'
    | 'timestamp'
    | 'categorical'
    | 'mixed';
  displayName?: string;
  description?: string;
  nullable: boolean;
  unique: boolean;
  primaryKey: boolean;
  foreignKey?: boolean;
  foreignKeyReference?: {
    table: string;
    column: string;
  };
  index: boolean;
  defaultValue?: any;
  validation?: ColumnValidation;
  statistics?: ColumnStatistics;
  metadata?: Record<string, any>;
}

/**
 * Column validation rules
 */
export interface ColumnValidation {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  minValue?: number;
  maxValue?: number;
  pattern?: string;
  enum?: any[];
  custom?: string;
  allowNull?: boolean;
  unique?: boolean;
}

/**
 * Column statistics
 */
export interface ColumnStatistics {
  count: number;
  nullCount: number;
  uniqueCount: number;
  duplicateCount?: number;
  min?: number | string;
  max?: number | string;
  mean?: number;
  median?: number;
  mode?: any;
  stdDev?: number;
  variance?: number;
  skewness?: number;
  kurtosis?: number;
  percentiles?: Record<number, number>;
}

/**
 * Dataset dimensions
 */
export interface DatasetDimensions {
  rows: number;
  columns: number;
  estimatedSize: number;
  actualSize: number;
}

/**
 * Core dataset object
 */
export interface Dataset {
  // Identifiers
  id: string;
  userId: string;
  name: string;
  description?: string;
  slug?: string;

  // File Information
  fileName: string;
  fileType: 'csv' | 'excel' | 'json' | 'parquet' | 'sql' | 'custom';
  fileSize: number;
  mimeType: string;
  encoding?: string;

  // Metadata
  columns: DatasetColumn[];
  dimensions: DatasetDimensions;
  sampleData?: Record<string, any>[];
  sampleSize?: number;
  totalRows: number;
  totalColumns: number;

  // ✅ Convenience property aliases
  columnCount?: number; // Alias for totalColumns
  size?: number; // Alias for fileSize
  rowCount?: number; // Alias for totalRows

  // Status & Processing
  status:
    | 'pending'
    | 'processing'
    | 'completed'
    | 'failed'
    | 'archived'
    | 'deleted';
  processingProgress?: number;
  processingStartedAt?: string;
  processingCompletedAt?: string;
  processingError?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };

  // Versioning
  version: number;
  versions?: DatasetVersion[];
  lastModifiedBy?: string;

  // Organization
  tags: string[];
  category?: string;
  collections?: string[];
  isFavorite?: boolean;
  isPublic: boolean;
  visibility: 'private' | 'shared' | 'public';

  // Sharing & Permissions
  owner: {
    id: string;
    name: string;
    email: string;
  };
  sharedWith?: DatasetShare[];
  permissions?: DatasetPermission[];

  // Statistics & Quality
  dataQuality?: DataQualityMetrics;
  statistics?: DatasetStatistics;
  anomalies?: AnomalyInfo[];

  // Processing History
  uploads?: UploadHistory[];
  cleaningOperations?: CleaningOperation[];
  analyses?: AnalysisReference[];

  // Metadata & Timestamps
  createdAt: string;
  updatedAt: string;
  deletedAt?: string;
  lastAccessedAt?: string;
  accessCount?: number;

  // Custom Fields
  customMetadata?: Record<string, any>;
  labels?: Record<string, string>;
}

/**
 * Dataset version
 */
export interface DatasetVersion {
  id: string;
  versionNumber: number;
  name: string;
  description?: string;
  changes?: string;
  createdBy: string;
  createdAt: string;
  size: number;
}

/**
 * Dataset share configuration
 */
export interface DatasetShare {
  id: string;
  userId: string;
  email: string;
  name: string;
  permission: 'view' | 'edit' | 'manage' | 'owner';
  sharedAt: string;
  sharedBy: string;
  expiresAt?: string;
}

/**
 * Dataset permission
 */
export interface DatasetPermission {
  id: string;
  principalType: 'user' | 'role' | 'group' | 'public';
  principalId: string;
  action: 'read' | 'write' | 'delete' | 'share' | 'admin';
  effect: 'allow' | 'deny';
  createdAt: string;
}

/**
 * Data quality metrics
 */
export interface DataQualityMetrics {
  overallScore: number; // 0-100
  completeness: number; // % of non-null values
  uniqueness: number; // % of unique values
  consistency: number; // % of consistent data
  validity: number; // % of valid data format
  accuracy?: number; // % of accurate data
  timeliness?: number; // % of timely data
  issues: DataQualityIssue[];
}

/**
 * Data quality issue
 */
export interface DataQualityIssue {
  id: string;
  column: string;
  type:
    | 'missing'
    | 'duplicate'
    | 'outlier'
    | 'inconsistent'
    | 'invalid'
    | 'other';
  severity: 'low' | 'medium' | 'high' | 'critical';
  count: number;
  percentage: number;
  description: string;
  suggestion?: string;
}

/**
 * Anomaly information
 */
export interface AnomalyInfo {
  rowIndex: number;
  columns: string[];
  anomalyScore: number;
  anomalyType: string;
  details?: Record<string, any>;
}

/**
 * Dataset statistics
 */
export interface DatasetStatistics {
  totalSize: number;
  rowCount: number;
  columnCount: number;
  memoryUsage: number;
  duplicateRows: number;
  missingValues: Record<string, number>;
  outliersDetected: number;
  lastUpdated: string;
}

/**
 * Upload history
 */
export interface UploadHistory {
  id: string;
  fileName: string;
  fileSize: number;
  uploadedAt: string;
  uploadedBy: string;
  status: 'pending' | 'completed' | 'failed';
  errorMessage?: string;
}

/**
 * Cleaning operation
 */
export interface CleaningOperation {
  id: string;
  type: string;
  status: 'pending' | 'completed' | 'failed';
  appliedAt: string;
  appliedBy: string;
  recordsAffected: number;
}

/**
 * Analysis reference
 */
export interface AnalysisReference {
  id: string;
  type: string;
  status: 'pending' | 'completed' | 'failed';
  performedAt: string;
}

// ============================================================================
// Dataset Request/Response Types
// ============================================================================

/**
 * Dataset upload request
 */
export interface DatasetUploadRequest {
  file: File;
  name?: string;
  description?: string;
  tags?: string[];
  isPublic?: boolean;
  metadata?: Record<string, any>;
}

/**
 * Dataset upload response
 */
export interface DatasetUploadResponse {
  id: string;
  name: string;
  fileName: string;
  fileSize: number;
  status: string;
  uploadedAt: string;
  processingStartedAt?: string;
  message: string;
}

/**
 * Bulk upload request
 */
export interface BulkUploadRequest {
  files: File[];
  folderName?: string;
  tags?: string[];
  isPublic?: boolean;
}

/**
 * Bulk upload response
 */
export interface BulkUploadResponse {
  successful: DatasetUploadResponse[];
  failed: {
    file: string;
    error: string;
  }[];
  totalFiles: number;
  successCount: number;
  failureCount: number;
}

/**
 * ✅ FIXED: Dataset list response
 * Changed: data → datasets, currentPage → page, pageSize → limit
 */
export interface DatasetListResponse {
  datasets: Dataset[];
  pagination: {
    page: number;
    limit: number;
    totalItems: number;
    totalPages: number;
  };
  filters?: {
    search?: string;
    status?: string;
    tags?: string[];
  };
}

/**
 * Dataset preview response
 */
export interface DatasetPreview {
  id: string;
  columns: DatasetColumn[];
  rows: Record<string, any>[];
  rowCount: number;
  columnCount: number;
  sample: number;
}

/**
 * Dataset metadata response
 */
export interface DatasetMetadata {
  id: string;
  name: string;
  description?: string;
  fileType: string;
  fileSize: number;
  dimensions: DatasetDimensions;
  columns: DatasetColumn[];
  createdAt: string;
  updatedAt: string;
  owner: {
    id: string;
    name: string;
    email: string;
  };
  statistics?: DatasetStatistics;
}

/**
 * Dataset processing status
 */
export interface DatasetProcessingStatus {
  datasetId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  startedAt: string;
  completedAt?: string;
  message?: string;
  error?: {
    code: string;
    message: string;
  };
}

/**
 * Dataset export request
 */
export interface DatasetExportRequest {
  format: 'csv' | 'excel' | 'json' | 'parquet' | 'sql';
  columns?: string[];
  rows?: {
    start?: number;
    end?: number;
  };
  filters?: Record<string, any>;
  compression?: 'gzip' | 'bzip2' | 'none';
}

/**
 * Dataset export response
 */
export interface DatasetExportResponse {
  exportId: string;
  fileName: string;
  format: string;
  size: number;
  downloadUrl: string;
  expiresAt: string;
  createdAt: string;
}

/**
 * Dataset duplicate request
 */
export interface DatasetDuplicateRequest {
  name?: string;
  description?: string;
  copyData?: boolean;
  copyPermissions?: boolean;
}

/**
 * Dataset duplicate response
 */
export interface DatasetDuplicateResponse {
  id: string;
  name: string;
  message: string;
}

// ============================================================================
// Dataset Operation Types
// ============================================================================

/**
 * Dataset update request
 */
export interface DatasetUpdateRequest {
  name?: string;
  description?: string;
  tags?: string[];
  isPublic?: boolean;
  visibility?: 'private' | 'shared' | 'public';
  customMetadata?: Record<string, any>;
}

/**
 * Dataset deletion request
 */
export interface DatasetDeletionRequest {
  permanent?: boolean;
  reason?: string;
}

/**
 * Batch operation request
 */
export interface BatchOperationRequest {
  datasetIds: string[];
  operation:
    | 'delete'
    | 'archive'
    | 'publish'
    | 'unpublish'
    | 'tag'
    | 'share';
  parameters?: Record<string, any>;
}

/**
 * Batch operation response
 */
export interface BatchOperationResponse {
  successful: string[];
  failed: {
    datasetId: string;
    error: string;
  }[];
  operationId: string;
  completedAt: string;
}

// ============================================================================
// Dataset Query/Filter Types
// ============================================================================

/**
 * ✅ FIXED: Dataset filter configuration
 * Changed: dateFrom → createdAfter, dateTo → createdBefore
 */
export interface DatasetFilterConfig {
  search?: string;
  status?: string[];
  tags?: string[];
  owner?: string;
  visibility?: string;
  createdAfter?: string;
  createdBefore?: string;
  minSize?: number;
  maxSize?: number;
  fileType?: string[];
  isPublic?: boolean;
  isFavorite?: boolean;
  customFilters?: Record<string, any>;
}

/**
 * ✅ FIXED: Dataset sort configuration
 * Changed: 'size' → 'fileSize', 'rows' → 'totalRows'
 * These properties actually exist on the Dataset type
 */
export interface DatasetSortConfig {
  field:
    | 'name'
    | 'createdAt'
    | 'updatedAt'
    | 'fileSize'
    | 'totalRows'
    | 'accessCount';
  order: 'asc' | 'desc';
}

/**
 * Dataset query options
 */
export interface DatasetQueryOptions {
  filters?: DatasetFilterConfig;
  sort?: DatasetSortConfig;
  pagination?: {
    page: number;
    limit: number;
  };
  includeStats?: boolean;
  includePreview?: boolean;
  previewRows?: number;
}

// ✅ Export aliases for backward compatibility
export type DatasetFilterOptions = DatasetFilterConfig;
export type DatasetSortOptions = DatasetSortConfig;

// ============================================================================
// File Upload & Processing Types
// ============================================================================

/**
 * Upload progress event
 */
export interface UploadProgressEvent {
  datasetId?: string;
  uploadId: string;
  progress: number;
  uploadedBytes: number;
  totalBytes: number;
  speed: number;
  estimatedTimeRemaining: number;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'failed';
}

/**
 * Upload error
 */
export interface UploadError {
  code: string;
  message: string;
  fileName: string;
  details?: Record<string, any>;
}

/**
 * File chunk information
 */
export interface FileChunk {
  chunkIndex: number;
  totalChunks: number;
  size: number;
  checksum: string;
}

/**
 * File upload configuration
 */
export interface FileUploadConfig {
  maxFileSize: number;
  maxTotalSize: number;
  allowedTypes: string[];
  allowedExtensions: string[];
  chunkSize: number;
  maxConcurrentChunks: number;
  autoRetry: boolean;
  maxRetries: number;
  timeout: number;
}

// ============================================================================
// Collection Types
// ============================================================================

/**
 * Dataset collection
 */
export interface DatasetCollection {
  id: string;
  name: string;
  description?: string;
  userId: string;
  datasetIds: string[];
  color?: string;
  icon?: string;
  isPublic: boolean;
  createdAt: string;
  updatedAt: string;
}

/**
 * Collection creation request
 */
export interface CollectionCreateRequest {
  name: string;
  description?: string;
  datasetIds?: string[];
  isPublic?: boolean;
}

/**
 * Collection update request
 */
export interface CollectionUpdateRequest {
  name?: string;
  description?: string;
  datasetIds?: string[];
  isPublic?: boolean;
  color?: string;
  icon?: string;
}

// ============================================================================
// Enums
// ============================================================================

/**
 * Dataset status enumeration
 */
export enum DatasetStatusEnum {
  Pending = 'pending',
  Processing = 'processing',
  Completed = 'completed',
  Failed = 'failed',
  Archived = 'archived',
  Deleted = 'deleted',
}

/**
 * File type enumeration
 */
export enum FileTypeEnum {
  CSV = 'csv',
  Excel = 'excel',
  JSON = 'json',
  Parquet = 'parquet',
  SQL = 'sql',
  Custom = 'custom',
}

/**
 * Column type enumeration
 */
export enum ColumnTypeEnum {
  String = 'string',
  Number = 'number',
  Integer = 'integer',
  Float = 'float',
  Boolean = 'boolean',
  Date = 'date',
  DateTime = 'datetime',
  Timestamp = 'timestamp',
  Categorical = 'categorical',
  Mixed = 'mixed',
}

/**
 * Permission level enumeration
 */
export enum PermissionLevelEnum {
  View = 'view',
  Edit = 'edit',
  Manage = 'manage',
  Owner = 'owner',
}

/**
 * Visibility enumeration
 */
export enum VisibilityEnum {
  Private = 'private',
  Shared = 'shared',
  Public = 'public',
}

/**
 * Data quality issue type enumeration
 */
export enum DataQualityIssueTypeEnum {
  Missing = 'missing',
  Duplicate = 'duplicate',
  Outlier = 'outlier',
  Inconsistent = 'inconsistent',
  Invalid = 'invalid',
  Other = 'other',
}

/**
 * Severity enumeration
 */
export enum SeverityEnum {
  Low = 'low',
  Medium = 'medium',
  High = 'high',
  Critical = 'critical',
}

// ============================================================================
// Form Types
// ============================================================================

/**
 * Dataset creation form values
 */
export interface DatasetFormValues {
  name: string;
  description?: string;
  file?: File;
  tags?: string[];
  isPublic: boolean;
  category?: string;
}

/**
 * Dataset filter form values
 */
export interface DatasetFilterFormValues {
  search: string;
  status: string[];
  tags: string[];
  fileType: string[];
  dateRange: {
    start: string;
    end: string;
  };
  minSize: number;
  maxSize: number;
}

// ============================================================================
// API Integration Types
// ============================================================================

/**
 * Dataset context value
 */
export interface DatasetContextValue {
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  isLoading: boolean;
  error: string | null;
  totalCount: number;
  currentPage: number;
  loadDatasets: (options?: DatasetQueryOptions) => Promise<void>;
  getDataset: (id: string) => Promise<Dataset>;
  createDataset: (data: DatasetUploadRequest) => Promise<Dataset>;
  updateDataset: (id: string, data: DatasetUpdateRequest) => Promise<Dataset>;
  deleteDataset: (id: string) => Promise<void>;
  duplicateDataset: (id: string, name?: string) => Promise<Dataset>;
}

// ============================================================================
// Type Guards & Predicates
// ============================================================================

/**
 * Type guard for Dataset
 */
export function isDataset(obj: any): obj is Dataset {
  return (
    obj &&
    typeof obj === 'object' &&
    typeof obj.id === 'string' &&
    typeof obj.userId === 'string' &&
    typeof obj.name === 'string' &&
    typeof obj.fileName === 'string'
  );
}

/**
 * Type guard for DatasetColumn
 */
export function isDatasetColumn(obj: any): obj is DatasetColumn {
  return (
    obj &&
    typeof obj === 'object' &&
    typeof obj.name === 'string' &&
    typeof obj.type === 'string' &&
    typeof obj.nullable === 'boolean'
  );
}

/**
 * Check if dataset is ready for analysis
 */
export function isDatasetReady(dataset: Dataset): boolean {
  return (
    dataset.status === 'completed' &&
    dataset.dimensions.rows > 0 &&
    dataset.dimensions.columns > 0
  );
}

/**
 * Check if user has permission
 */
export function hasDatasetPermission(
  dataset: Dataset,
  userId: string,
  action: 'read' | 'write' | 'delete' | 'share' | 'admin'
): boolean {
  if (dataset.owner.id === userId) {
    return true;
  }

  const permission = dataset.permissions?.find(
    (p) => p.principalId === userId && p.action === action
  );

  return permission?.effect === 'allow';
}

/**
 * Get dataset quality status
 */
export function getQualityStatus(
  metrics: DataQualityMetrics | undefined
): 'excellent' | 'good' | 'fair' | 'poor' | 'unknown' {
  if (!metrics) return 'unknown';

  const score = metrics.overallScore;
  if (score >= 80) return 'excellent';
  if (score >= 60) return 'good';
  if (score >= 40) return 'fair';
  return 'poor';
}
