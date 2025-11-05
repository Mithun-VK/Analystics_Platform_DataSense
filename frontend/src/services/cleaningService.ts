// src/services/cleaningService.ts
import { apiPost, apiGet } from './api';

/**
 * Data Cleaning Service - Handles all data cleaning and preprocessing operations
 * Provides comprehensive tools for data quality improvement and transformation
 */

const CLEANING_ENDPOINTS = {
  DETECT_ISSUES: '/cleaning/detect-issues',
  HANDLE_MISSING_VALUES: '/cleaning/handle-missing',
  REMOVE_DUPLICATES: '/cleaning/remove-duplicates',
  STANDARDIZE_FORMAT: '/cleaning/standardize-format',
  HANDLE_OUTLIERS: '/cleaning/handle-outliers',
  NORMALIZE_DATA: '/cleaning/normalize',
  ENCODE_CATEGORICAL: '/cleaning/encode-categorical',
  SCALE_FEATURES: '/cleaning/scale-features',
  TEXT_CLEANING: '/cleaning/text-cleaning',
  DATE_PARSING: '/cleaning/parse-dates',
  MERGE_DATASETS: '/cleaning/merge',
  SPLIT_COLUMNS: '/cleaning/split-columns',
  COMBINE_COLUMNS: '/cleaning/combine-columns',
  FILTER_ROWS: '/cleaning/filter-rows',
  RENAME_COLUMNS: '/cleaning/rename-columns',
  REORDER_COLUMNS: '/cleaning/reorder-columns',
  TYPE_CONVERSION: '/cleaning/convert-types',
  CREATE_FEATURES: '/cleaning/create-features',
  GET_OPERATIONS: '/cleaning/operations/:datasetId',
  PREVIEW_OPERATION: '/cleaning/preview',
  APPLY_OPERATION: '/cleaning/apply',
  ROLLBACK_OPERATION: '/cleaning/rollback/:operationId',
  GET_HISTORY: '/cleaning/history/:datasetId',
  BATCH_OPERATIONS: '/cleaning/batch',
  EXPORT_CLEANED_DATA: '/cleaning/:datasetId/export/:format',
  VALIDATE_CLEANED_DATA: '/cleaning/validate',
  GENERATE_CLEANING_REPORT: '/cleaning/:datasetId/report',
};

interface MissingValueConfig {
  strategy: 'drop' | 'mean' | 'median' | 'mode' | 'forward_fill' | 'backward_fill' | 'interpolate' | 'custom';
  customValue?: any;
  threshold?: number; // percentage
  columns?: string[];
}

interface DuplicateConfig {
  subset?: string[];
  keepFirst?: boolean;
  keepLast?: boolean;
  dropAll?: boolean;
}

interface OutlierConfig {
  method: 'iqr' | 'zscore' | 'isolation_forest' | 'mahalanobis';
  threshold?: number;
  action: 'remove' | 'cap' | 'flag';
  columns?: string[];
}

interface NormalizationConfig {
  columns: string[];
  method: 'minmax' | 'zscore' | 'robust' | 'log' | 'sqrt';
  featureRange?: [number, number];
}

interface EncodingConfig {
  columns: string[];
  method: 'onehot' | 'label' | 'ordinal' | 'target';
  handleUnknown?: 'error' | 'use_encoded_value';
  sparse?: boolean;
}

interface ScalingConfig {
  columns: string[];
  method: 'standard' | 'minmax' | 'robust' | 'quantile';
}

interface TextCleaningConfig {
  columns: string[];
  lowercase?: boolean;
  removeSpecialChars?: boolean;
  removePunctuation?: boolean;
  removeStopwords?: boolean;
  stemming?: boolean;
  lemmatization?: boolean;
  trimWhitespace?: boolean;
  language?: string;
}

interface DateParsingConfig {
  columns: string[];
  format?: string;
  infer?: boolean;
  utc?: boolean;
  unit?: string;
}

interface MergeConfig {
  datasetIds: string[];
  on?: string | string[];
  how: 'inner' | 'outer' | 'left' | 'right';
}

interface FeatureCreationConfig {
  features: Array<{
    name: string;
    type: 'arithmetic' | 'polynomial' | 'interaction' | 'aggregation' | 'temporal' | 'custom';
    formula?: string;
    columns?: string[];
    parameters?: Record<string, any>;
  }>;
}

interface CleaningOperation {
  id: string;
  datasetId: string;
  type: string;
  config: any;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  timestamp: string;
  recordsAffected: number;
  preview?: any[];
}

interface CleaningReport {
  datasetId: string;
  timestamp: string;
  operations: Array<{
    type: string;
    recordsAffected: number;
    description: string;
  }>;
  dataQualityBefore: {
    completeness: number;
    uniqueness: number;
    consistency: number;
  };
  dataQualityAfter: {
    completeness: number;
    uniqueness: number;
    consistency: number;
  };
  summary: string;
}

/**
 * Detect data quality issues
 */
export const detectDataIssues = async (
  datasetId: string
): Promise<{
  missingValues: Record<string, number>;
  duplicates: number;
  outliers: Record<string, number>;
  inconsistencies: string[];
  typeIssues: string[];
}> => {
  try {
    const response = await apiPost<{
      missingValues: Record<string, number>;
      duplicates: number;
      outliers: Record<string, number>;
      inconsistencies: string[];
      typeIssues: string[];
    }>(CLEANING_ENDPOINTS.DETECT_ISSUES, { datasetId });

    console.debug('[CleaningService] Data issues detected', {
      datasetId,
      missingCount: Object.keys(response.missingValues).length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to detect data issues', error);
    throw error;
  }
};

/**
 * Handle missing values
 */
export const handleMissingValues = async (
  datasetId: string,
  config: MissingValueConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.HANDLE_MISSING_VALUES,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Missing values handled', {
      datasetId,
      strategy: config.strategy,
      recordsAffected: response.recordsAffected,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to handle missing values', error);
    throw error;
  }
};

/**
 * Remove duplicate rows
 */
export const removeDuplicates = async (
  datasetId: string,
  config: DuplicateConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.REMOVE_DUPLICATES,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Duplicates removed', {
      datasetId,
      recordsRemoved: response.recordsAffected,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to remove duplicates', error);
    throw error;
  }
};

/**
 * Standardize data format
 */
export const standardizeFormat = async (
  datasetId: string,
  config: {
    columns: string[];
    format: string;
    locale?: string;
  },
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.STANDARDIZE_FORMAT,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Format standardized', datasetId);

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to standardize format', error);
    throw error;
  }
};

/**
 * Handle outliers
 */
export const handleOutliers = async (
  datasetId: string,
  config: OutlierConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.HANDLE_OUTLIERS,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Outliers handled', {
      datasetId,
      method: config.method,
      recordsAffected: response.recordsAffected,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to handle outliers', error);
    throw error;
  }
};

/**
 * Normalize data
 */
export const normalizeData = async (
  datasetId: string,
  config: NormalizationConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.NORMALIZE_DATA,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Data normalized', {
      datasetId,
      method: config.method,
      columnsAffected: config.columns.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to normalize data', error);
    throw error;
  }
};

/**
 * Encode categorical variables
 */
export const encodeCategorical = async (
  datasetId: string,
  config: EncodingConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.ENCODE_CATEGORICAL,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Categorical variables encoded', {
      datasetId,
      method: config.method,
      columnsAffected: config.columns.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to encode categorical variables', error);
    throw error;
  }
};

/**
 * Scale features
 */
export const scaleFeatures = async (
  datasetId: string,
  config: ScalingConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.SCALE_FEATURES,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Features scaled', {
      datasetId,
      method: config.method,
      columnsAffected: config.columns.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to scale features', error);
    throw error;
  }
};

/**
 * Clean text data
 */
export const cleanText = async (
  datasetId: string,
  config: TextCleaningConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.TEXT_CLEANING,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Text cleaned', {
      datasetId,
      columnsAffected: config.columns.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to clean text', error);
    throw error;
  }
};

/**
 * Parse and standardize dates
 */
export const parseDates = async (
  datasetId: string,
  config: DateParsingConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.DATE_PARSING,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Dates parsed', {
      datasetId,
      columnsAffected: config.columns.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to parse dates', error);
    throw error;
  }
};

/**
 * Merge multiple datasets
 */
export const mergeDatasets = async (
  config: MergeConfig,
  preview: boolean = true
): Promise<{
  datasetId: string;
  rowsCreated: number;
  columnsCreated: number;
  preview?: any[];
}> => {
  try {
    const response = await apiPost<{
      datasetId: string;
      rowsCreated: number;
      columnsCreated: number;
      preview?: any[];
    }>(CLEANING_ENDPOINTS.MERGE_DATASETS, { config, preview });

    console.debug('[CleaningService] Datasets merged', {
      datasetCount: config.datasetIds.length,
      rowsCreated: response.rowsCreated,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to merge datasets', error);
    throw error;
  }
};

/**
 * Split column into multiple columns
 */
export const splitColumn = async (
  datasetId: string,
  config: {
    column: string;
    separator: string;
    newColumnNames: string[];
    maxSplit?: number;
  },
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.SPLIT_COLUMNS,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Column split', {
      datasetId,
      originalColumn: config.column,
      newColumns: config.newColumnNames.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to split column', error);
    throw error;
  }
};

/**
 * Combine multiple columns
 */
export const combineColumns = async (
  datasetId: string,
  config: {
    columns: string[];
    newColumnName: string;
    separator: string;
    removeOriginal?: boolean;
  },
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.COMBINE_COLUMNS,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Columns combined', {
      datasetId,
      columnsCount: config.columns.length,
      newColumnName: config.newColumnName,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to combine columns', error);
    throw error;
  }
};

/**
 * Filter rows based on conditions
 */
export const filterRows = async (
  datasetId: string,
  config: {
    conditions: Array<{
      column: string;
      operator: '=' | '!=' | '>' | '<' | '>=' | '<=' | 'in' | 'contains' | 'startswith' | 'endswith';
      value: any;
    }>;
    logic?: 'and' | 'or';
    keepMatchingRows?: boolean;
  },
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.FILTER_ROWS,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Rows filtered', {
      datasetId,
      conditionCount: config.conditions.length,
      recordsAffected: response.recordsAffected,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to filter rows', error);
    throw error;
  }
};

/**
 * Rename columns
 */
export const renameColumns = async (
  datasetId: string,
  mapping: Record<string, string>,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.RENAME_COLUMNS,
      { datasetId, mapping, preview }
    );

    console.debug('[CleaningService] Columns renamed', {
      datasetId,
      renameCount: Object.keys(mapping).length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to rename columns', error);
    throw error;
  }
};

/**
 * Reorder columns
 */
export const reorderColumns = async (
  datasetId: string,
  columnOrder: string[],
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.REORDER_COLUMNS,
      { datasetId, columnOrder, preview }
    );

    console.debug('[CleaningService] Columns reordered', datasetId);

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to reorder columns', error);
    throw error;
  }
};

/**
 * Convert data types
 */
export const convertDataTypes = async (
  datasetId: string,
  config: {
    columns: string[];
    targetType: 'int' | 'float' | 'string' | 'bool' | 'datetime' | 'category';
    errorHandling?: 'coerce' | 'ignore' | 'raise';
  },
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.TYPE_CONVERSION,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Data types converted', {
      datasetId,
      columnsAffected: config.columns.length,
      targetType: config.targetType,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to convert data types', error);
    throw error;
  }
};

/**
 * Create new features
 */
export const createFeatures = async (
  datasetId: string,
  config: FeatureCreationConfig,
  preview: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.CREATE_FEATURES,
      { datasetId, config, preview }
    );

    console.debug('[CleaningService] Features created', {
      datasetId,
      featuresCount: config.features.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to create features', error);
    throw error;
  }
};

/**
 * Get cleaning operations for dataset
 */
export const getCleaningOperations = async (
  datasetId: string,
  page: number = 1,
  limit: number = 20
): Promise<{
  operations: CleaningOperation[];
  pagination: any;
}> => {
  try {
    const url = CLEANING_ENDPOINTS.GET_OPERATIONS
      .replace(':datasetId', datasetId)
      .concat(`?page=${page}&limit=${limit}`);

    const response = await apiGet<{
      operations: CleaningOperation[];
      pagination: any;
    }>(url);

    console.debug('[CleaningService] Cleaning operations fetched', {
      datasetId,
      operationsCount: response.operations.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to fetch cleaning operations', error);
    throw error;
  }
};

/**
 * Preview cleaning operation
 */
export const previewCleaningOperation = async (
  datasetId: string,
  operationType: string,
  config: any,
  sampleSize: number = 100
): Promise<{ preview: any[]; recordsAffected: number }> => {
  try {
    const response = await apiPost<{ preview: any[]; recordsAffected: number }>(
      CLEANING_ENDPOINTS.PREVIEW_OPERATION,
      { datasetId, operationType, config, sampleSize }
    );

    console.debug('[CleaningService] Operation preview generated', {
      datasetId,
      operationType,
      previewRows: response.preview.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to preview operation', error);
    throw error;
  }
};

/**
 * Apply cleaning operation
 */
export const applyCleaningOperation = async (
  datasetId: string,
  operationType: string,
  config: any,
  createBackup: boolean = true
): Promise<CleaningOperation> => {
  try {
    const response = await apiPost<CleaningOperation>(
      CLEANING_ENDPOINTS.APPLY_OPERATION,
      { datasetId, operationType, config, createBackup }
    );

    console.debug('[CleaningService] Cleaning operation applied', {
      datasetId,
      operationType,
      recordsAffected: response.recordsAffected,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to apply cleaning operation', error);
    throw error;
  }
};

/**
 * Rollback cleaning operation
 */
export const rollbackOperation = async (operationId: string): Promise<void> => {
  try {
    const url = CLEANING_ENDPOINTS.ROLLBACK_OPERATION.replace(':operationId', operationId);
    await apiPost(url, {});

    console.debug('[CleaningService] Operation rolled back', operationId);
  } catch (error) {
    console.error('[CleaningService] Failed to rollback operation', error);
    throw error;
  }
};

/**
 * Get cleaning history
 */
export const getCleaningHistory = async (
  datasetId: string,
  page: number = 1,
  limit: number = 50
): Promise<{
  history: CleaningOperation[];
  pagination: any;
}> => {
  try {
    const url = CLEANING_ENDPOINTS.GET_HISTORY
      .replace(':datasetId', datasetId)
      .concat(`?page=${page}&limit=${limit}`);

    const response = await apiGet<{
      history: CleaningOperation[];
      pagination: any;
    }>(url);

    console.debug('[CleaningService] Cleaning history fetched', {
      datasetId,
      historyCount: response.history.length,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to fetch cleaning history', error);
    throw error;
  }
};

/**
 * Apply batch cleaning operations
 */
export const batchCleaningOperations = async (
  datasetId: string,
  operations: Array<{
    type: string;
    config: any;
  }>,
  executeSequentially: boolean = true
): Promise<{
  results: CleaningOperation[];
  totalRecordsAffected: number;
  duration: number;
}> => {
  try {
    const response = await apiPost<{
      results: CleaningOperation[];
      totalRecordsAffected: number;
      duration: number;
    }>(CLEANING_ENDPOINTS.BATCH_OPERATIONS, {
      datasetId,
      operations,
      executeSequentially,
    });

    console.debug('[CleaningService] Batch operations completed', {
      datasetId,
      operationsCount: operations.length,
      totalRecordsAffected: response.totalRecordsAffected,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to execute batch operations', error);
    throw error;
  }
};

/**
 * Export cleaned data
 */
export const exportCleanedData = async (
  datasetId: string,
  format: 'csv' | 'json' | 'excel' | 'parquet' = 'csv'
): Promise<Blob> => {
  try {
    const url = CLEANING_ENDPOINTS.EXPORT_CLEANED_DATA
      .replace(':datasetId', datasetId)
      .replace(':format', format);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.responseType = 'blob';
      xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('auth_access_token')}`);

      xhr.onload = () => {
        if (xhr.status === 200) {
          console.debug('[CleaningService] Cleaned data exported', { datasetId, format });
          resolve(xhr.response);
        } else {
          reject(new Error(`Export failed with status ${xhr.status}`));
        }
      };

      xhr.onerror = () => reject(new Error('Export request failed'));
      xhr.send();
    });
  } catch (error) {
    console.error('[CleaningService] Failed to export cleaned data', error);
    throw error;
  }
};

/**
 * Validate cleaned data
 */
export const validateCleanedData = async (
  datasetId: string,
  rules?: Array<{
    column: string;
    rule: string;
    parameters?: Record<string, any>;
  }>
): Promise<{
  valid: boolean;
  violations: Array<{
    rule: string;
    column: string;
    count: number;
    description: string;
  }>;
  qualityScore: number;
}> => {
  try {
    const response = await apiPost<{
      valid: boolean;
      violations: Array<{
        rule: string;
        column: string;
        count: number;
        description: string;
      }>;
      qualityScore: number;
    }>(CLEANING_ENDPOINTS.VALIDATE_CLEANED_DATA, { datasetId, rules });

    console.debug('[CleaningService] Cleaned data validated', {
      datasetId,
      valid: response.valid,
      qualityScore: response.qualityScore,
    });

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to validate cleaned data', error);
    throw error;
  }
};

/**
 * Generate cleaning report
 */
export const generateCleaningReport = async (
  datasetId: string
): Promise<CleaningReport> => {
  try {
    const url = CLEANING_ENDPOINTS.GENERATE_CLEANING_REPORT.replace(':datasetId', datasetId);
    const response = await apiGet<CleaningReport>(url);

    console.debug('[CleaningService] Cleaning report generated', datasetId);

    return response;
  } catch (error) {
    console.error('[CleaningService] Failed to generate cleaning report', error);
    throw error;
  }
};

/**
 * Download cleaned data as file
 */
export const downloadCleanedData = async (
  datasetId: string,
  format: 'csv' | 'json' | 'excel' | 'parquet' = 'csv'
): Promise<void> => {
  try {
    const blob = await exportCleanedData(datasetId, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `cleaned_data_${datasetId}.${format}`;
    link.click();
    URL.revokeObjectURL(url);

    console.debug('[CleaningService] Cleaned data downloaded', { datasetId, format });
  } catch (error) {
    console.error('[CleaningService] Failed to download cleaned data', error);
    throw error;
  }
};

export default {
  detectDataIssues,
  handleMissingValues,
  removeDuplicates,
  standardizeFormat,
  handleOutliers,
  normalizeData,
  encodeCategorical,
  scaleFeatures,
  cleanText,
  parseDates,
  mergeDatasets,
  splitColumn,
  combineColumns,
  filterRows,
  renameColumns,
  reorderColumns,
  convertDataTypes,
  createFeatures,
  getCleaningOperations,
  previewCleaningOperation,
  applyCleaningOperation,
  rollbackOperation,
  getCleaningHistory,
  batchCleaningOperations,
  exportCleanedData,
  validateCleanedData,
  generateCleaningReport,
  downloadCleanedData,
};
