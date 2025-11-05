// src/services/datasetService.ts

import {
  apiGet,
  apiPost,
  apiPut,
  apiDelete,
  apiUploadFile,
} from './api';
import type {
  Dataset,
  DatasetListResponse,
  DatasetUploadResponse,
  DatasetPreview,
  DatasetMetadata,
  DatasetStatistics,
} from '@/types/dataset.types';

/**
 * Dataset Service - Handles all dataset-related API calls
 * Provides comprehensive dataset management functionality with error handling and logging
 */

// ============================================================================
// Constants
// ============================================================================

const DATASET_ENDPOINTS = {
  LIST: '/datasets',
  CREATE: '/datasets',
  GET_BY_ID: '/datasets/:id',
  UPDATE: '/datasets/:id',
  DELETE: '/datasets/:id',
  BULK_DELETE: '/datasets/bulk-delete',
  UPLOAD: '/datasets/upload',
  PREVIEW: '/datasets/:id/preview',
  METADATA: '/datasets/:id/metadata',
  STATISTICS: '/datasets/:id/statistics',
  DUPLICATE: '/datasets/:id/duplicate',
  EXPORT: '/datasets/:id/export/:format',
  SHARE: '/datasets/:id/share',
  PERMISSIONS: '/datasets/:id/permissions',
  STATUS: '/datasets/:id/status',
  CANCEL: '/datasets/:id/cancel',
  QUOTA: '/datasets/quota',
};

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
const SUPPORTED_FORMATS = [
  'text/csv',
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'application/json',
];
const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB for chunked upload

// ============================================================================
// Type Definitions
// ============================================================================

interface ChunkUploadProgress {
  chunkIndex: number;
  totalChunks: number;
  percentage: number;
  uploadedBytes: number;
  totalBytes: number;
}

interface PaginationParams {
  page?: number;
  limit?: number;
  search?: string;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  status?: 'processing' | 'completed' | 'failed' | 'pending';
  createdAfter?: string;
  createdBefore?: string;
}

interface DatasetFilters {
  status?: string;
  minSize?: number;
  maxSize?: number;
  createdAfter?: string;
  createdBefore?: string;
  tags?: string[];
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Validate file before upload
 */
const validateFile = (file: File): void => {
  // Check file type
  if (!SUPPORTED_FORMATS.includes(file.type)) {
    throw new Error(
      'Unsupported file format. Supported formats: CSV, Excel, JSON'
    );
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    throw new Error(
      `File size exceeds ${formatFileSize(MAX_FILE_SIZE)} limit`
    );
  }

  // Check file name
  if (!file.name || file.name.trim().length === 0) {
    throw new Error('Invalid file name');
  }
};

/**
 * Format file size for display
 */
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return (
    Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  );
};

/**
 * Generate unique file ID for tracking uploads
 */
const generateFileId = (): string => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// ============================================================================
// Dataset CRUD Operations
// ============================================================================

/**
 * Fetch datasets with pagination and filtering
 */
export const fetchDatasets = async (
  page: number = 1,
  limit: number = 10,
  params?: PaginationParams & { signal?: AbortSignal }
): Promise<DatasetListResponse> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(params?.search && { search: params.search }),
      ...(params?.sortBy && { sortBy: params.sortBy }),
      ...(params?.sortOrder && { sortOrder: params.sortOrder }),
      ...(params?.status && { status: params.status }),
      ...(params?.createdAfter && { createdAfter: params.createdAfter }),
      ...(params?.createdBefore && { createdBefore: params.createdBefore }),
    });

    const url = `${DATASET_ENDPOINTS.LIST}?${queryParams.toString()}`;
    const response = await apiGet<DatasetListResponse>(url, {
      signal: params?.signal,
    });

    console.debug('[DatasetService] Datasets fetched', {
      page,
      limit,
      count: response.datasets.length,
      total: response.pagination.totalItems,
    });

    return response;
  } catch (error) {
    console.error('[DatasetService] Failed to fetch datasets', error);
    throw error;
  }
};

/**
 * Get single dataset by ID
 */
export const getDatasetById = async (
  datasetId: string
): Promise<Dataset> => {
  try {
    const url = DATASET_ENDPOINTS.GET_BY_ID.replace(':id', datasetId);
    const response = await apiGet<Dataset>(url);

    console.debug('[DatasetService] Dataset fetched', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to fetch dataset',
      datasetId,
      error
    );
    throw error;
  }
};

/**
 * Create new dataset
 */
export const createDataset = async (
  data: { name: string; description?: string }
): Promise<Dataset> => {
  try {
    const response = await apiPost<Dataset>(
      DATASET_ENDPOINTS.CREATE,
      data
    );

    console.debug('[DatasetService] Dataset created', data.name);

    return response;
  } catch (error) {
    console.error('[DatasetService] Failed to create dataset', error);
    throw error;
  }
};

/**
 * Upload dataset file
 * Supports both simple and chunked uploads for large files
 */
export const uploadDataset = async (
  file: File,
  metadata?: { name?: string; description?: string; tags?: string[] },
  onProgress?: (progress: number) => void,
  onChunkProgress?: (chunkProgress: ChunkUploadProgress) => void
): Promise<DatasetUploadResponse> => {
  try {
    // Validate file
    validateFile(file);

    // Use chunked upload for large files
    if (file.size > CHUNK_SIZE) {
      return uploadDatasetChunked(file, metadata, onChunkProgress);
    }

    // Simple upload for smaller files
    const response = await apiUploadFile<DatasetUploadResponse>(
      DATASET_ENDPOINTS.UPLOAD,
      file,
      {
        name: metadata?.name || file.name,
        description: metadata?.description,
        tags: metadata?.tags?.join(','),
      },
      onProgress
    );

    console.debug(
      '[DatasetService] Dataset uploaded successfully',
      file.name
    );

    return response;
  } catch (error) {
    console.error('[DatasetService] Dataset upload failed', error);
    throw error;
  }
};

/**
 * Upload dataset using chunked approach for large files
 */
const uploadDatasetChunked = async (
  file: File,
  metadata?: { name?: string; description?: string; tags?: string[] },
  onChunkProgress?: (progress: ChunkUploadProgress) => void
): Promise<DatasetUploadResponse> => {
  const fileId = generateFileId();
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  let uploadedBytes = 0;

  try {
    // Upload each chunk
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const start = chunkIndex * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const chunkFile = new File(
        [chunk],
        `${file.name}.chunk${chunkIndex}`,
        {
          type: file.type,
        }
      );

      await apiUploadFile(
        `${DATASET_ENDPOINTS.UPLOAD}/chunk`,
        chunkFile,
        {
          fileId,
          chunkIndex,
          totalChunks,
          fileName: file.name,
          ...metadata,
        }
      );

      uploadedBytes += chunk.size;
      const percentage = Math.round((uploadedBytes / file.size) * 100);

      onChunkProgress?.({
        chunkIndex,
        totalChunks,
        percentage,
        uploadedBytes,
        totalBytes: file.size,
      });

      console.debug('[DatasetService] Chunk uploaded', {
        fileId,
        chunkIndex,
        totalChunks,
        percentage,
      });
    }

    // Merge chunks on server
    const response = await apiPost<DatasetUploadResponse>(
      `${DATASET_ENDPOINTS.UPLOAD}/merge`,
      { fileId, totalChunks, fileName: file.name }
    );

    console.debug(
      '[DatasetService] Dataset chunks merged successfully',
      fileId
    );

    return response;
  } catch (error) {
    console.error('[DatasetService] Chunked upload failed', error);
    // Clean up uploaded chunks
    await apiDelete(`${DATASET_ENDPOINTS.UPLOAD}/cleanup`, {
      silent: true,
    }).catch(() => {});
    throw error;
  }
};

/**
 * Update dataset metadata
 */
export const updateDataset = async (
  datasetId: string,
  updates: Partial<Dataset>
): Promise<Dataset> => {
  try {
    const url = DATASET_ENDPOINTS.UPDATE.replace(':id', datasetId);
    const response = await apiPut<Dataset>(url, updates);

    console.debug('[DatasetService] Dataset updated', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to update dataset',
      datasetId,
      error
    );
    throw error;
  }
};

/**
 * Delete single dataset
 */
export const deleteDataset = async (datasetId: string): Promise<void> => {
  try {
    const url = DATASET_ENDPOINTS.DELETE.replace(':id', datasetId);
    await apiDelete(url);

    console.debug('[DatasetService] Dataset deleted', datasetId);
  } catch (error) {
    console.error(
      '[DatasetService] Failed to delete dataset',
      datasetId,
      error
    );
    throw error;
  }
};

/**
 * Bulk delete datasets
 */
export const bulkDeleteDatasets = async (
  datasetIds: string[]
): Promise<void> => {
  try {
    if (datasetIds.length === 0) {
      throw new Error('No datasets selected');
    }

    await apiPost(DATASET_ENDPOINTS.BULK_DELETE, { datasetIds });

    console.debug(
      '[DatasetService] Datasets bulk deleted',
      datasetIds.length
    );
  } catch (error) {
    console.error(
      '[DatasetService] Failed to bulk delete datasets',
      error
    );
    throw error;
  }
};

// ============================================================================
// Dataset Details & Preview
// ============================================================================

/**
 * Get dataset preview (first few rows)
 */
export const getDatasetPreview = async (
  datasetId: string,
  options?: { rows?: number; columns?: string[] }
): Promise<DatasetPreview> => {
  try {
    const queryParams = new URLSearchParams({
      ...(options?.rows && { rows: options.rows.toString() }),
      ...(options?.columns && { columns: options.columns.join(',') }),
    });

    const url = `${DATASET_ENDPOINTS.PREVIEW.replace(
      ':id',
      datasetId
    )}?${queryParams.toString()}`;
    const response = await apiGet<DatasetPreview>(url);

    console.debug('[DatasetService] Dataset preview fetched', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to fetch dataset preview',
      error
    );
    throw error;
  }
};

/**
 * Get dataset metadata
 */
export const getDatasetMetadata = async (
  datasetId: string
): Promise<DatasetMetadata> => {
  try {
    const url = DATASET_ENDPOINTS.METADATA.replace(':id', datasetId);
    const response = await apiGet<DatasetMetadata>(url);

    console.debug('[DatasetService] Dataset metadata fetched', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to fetch dataset metadata',
      error
    );
    throw error;
  }
};

/**
 * Get dataset statistics
 */
export const getDatasetStatistics = async (
  datasetId: string
): Promise<DatasetStatistics> => {
  try {
    const url = DATASET_ENDPOINTS.STATISTICS.replace(':id', datasetId);
    const response = await apiGet<DatasetStatistics>(url);

    console.debug('[DatasetService] Dataset statistics fetched', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to fetch dataset statistics',
      error
    );
    throw error;
  }
};

// ============================================================================
// Dataset Operations
// ============================================================================

/**
 * Duplicate dataset
 */
export const duplicateDataset = async (
  datasetId: string,
  newName?: string
): Promise<Dataset> => {
  try {
    const url = DATASET_ENDPOINTS.DUPLICATE.replace(':id', datasetId);
    const response = await apiPost<Dataset>(url, { name: newName });

    console.debug('[DatasetService] Dataset duplicated', datasetId);

    return response;
  } catch (error) {
    console.error('[DatasetService] Failed to duplicate dataset', error);
    throw error;
  }
};

/**
 * Export dataset in specified format
 */
export const exportDataset = async (
  datasetId: string,
  format: 'csv' | 'excel' | 'json' | 'parquet' = 'csv',
  filters?: DatasetFilters
): Promise<Blob> => {
  try {
    const queryParams = new URLSearchParams({
      ...(filters?.status && { status: filters.status }),
      ...(filters?.minSize && { minSize: filters.minSize.toString() }),
      ...(filters?.maxSize && { maxSize: filters.maxSize.toString() }),
      ...(filters?.createdAfter && { createdAfter: filters.createdAfter }),
      ...(filters?.createdBefore && { createdBefore: filters.createdBefore }),
      ...(filters?.tags && { tags: filters.tags.join(',') }),
    });

    const url = `${DATASET_ENDPOINTS.EXPORT.replace(':id', datasetId)
      .replace(':format', format)}?${queryParams.toString()}`;

    const response = await apiGet<Blob>(url, {
      raw: true,
    });

    console.debug(
      '[DatasetService] Dataset exported',
      datasetId,
      format
    );

    return response;
  } catch (error) {
    console.error('[DatasetService] Failed to export dataset', error);
    throw error;
  }
};

// ============================================================================
// Sharing & Permissions
// ============================================================================

/**
 * Share dataset with other users
 */
export const shareDataset = async (
  datasetId: string,
  emails: string[],
  permission: 'view' | 'edit' | 'manage' = 'view'
): Promise<{ sharedWith: string[] }> => {
  try {
    const url = DATASET_ENDPOINTS.SHARE.replace(':id', datasetId);
    const response = await apiPost<{ sharedWith: string[] }>(url, {
      emails,
      permission,
    });

    console.debug(
      '[DatasetService] Dataset shared',
      datasetId,
      emails.length
    );

    return response;
  } catch (error) {
    console.error('[DatasetService] Failed to share dataset', error);
    throw error;
  }
};

/**
 * Get dataset permissions
 */
export const getDatasetPermissions = async (
  datasetId: string
): Promise<{ users: Array<{ email: string; permission: string }> }> => {
  try {
    const url = DATASET_ENDPOINTS.PERMISSIONS.replace(':id', datasetId);
    const response = await apiGet<{
      users: Array<{ email: string; permission: string }>;
    }>(url);

    console.debug('[DatasetService] Dataset permissions fetched', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to fetch dataset permissions',
      error
    );
    throw error;
  }
};

/**
 * Update dataset permissions
 */
export const updateDatasetPermissions = async (
  datasetId: string,
  email: string,
  permission: 'view' | 'edit' | 'manage' | 'none'
): Promise<void> => {
  try {
    const url = DATASET_ENDPOINTS.PERMISSIONS.replace(':id', datasetId);
    await apiPut(url, { email, permission });

    console.debug(
      '[DatasetService] Dataset permission updated',
      datasetId,
      email
    );
  } catch (error) {
    console.error(
      '[DatasetService] Failed to update dataset permission',
      error
    );
    throw error;
  }
};

// ============================================================================
// Search & Discovery
// ============================================================================

/**
 * Search datasets
 */
export const searchDatasets = async (
  query: string,
  limit: number = 20
): Promise<Dataset[]> => {
  try {
    const queryParams = new URLSearchParams({
      search: query,
      limit: limit.toString(),
    });

    const url = `${DATASET_ENDPOINTS.LIST}?${queryParams.toString()}`;
    const response = await apiGet<DatasetListResponse>(url);

    console.debug(
      '[DatasetService] Datasets searched',
      query,
      response.datasets.length
    );

    return response.datasets;
  } catch (error) {
    console.error('[DatasetService] Failed to search datasets', error);
    throw error;
  }
};

// ============================================================================
// Processing & Status
// ============================================================================

/**
 * Get dataset processing status
 */
export const getDatasetProcessingStatus = async (
  datasetId: string
): Promise<{ status: string; progress: number; message: string }> => {
  try {
    const url = DATASET_ENDPOINTS.STATUS.replace(':id', datasetId);
    const response = await apiGet<{
      status: string;
      progress: number;
      message: string;
    }>(url);

    console.debug('[DatasetService] Dataset processing status fetched', datasetId);

    return response;
  } catch (error) {
    console.error(
      '[DatasetService] Failed to fetch dataset processing status',
      error
    );
    throw error;
  }
};

/**
 * Cancel dataset processing
 */
export const cancelDatasetProcessing = async (
  datasetId: string
): Promise<void> => {
  try {
    const url = DATASET_ENDPOINTS.CANCEL.replace(':id', datasetId);
    await apiPost(url, {});

    console.debug('[DatasetService] Dataset processing cancelled', datasetId);
  } catch (error) {
    console.error(
      '[DatasetService] Failed to cancel dataset processing',
      error
    );
    throw error;
  }
};

// ============================================================================
// Storage & Quota
// ============================================================================

/**
 * Get dataset storage quota
 */
export const getStorageQuota = async (): Promise<{
  used: number;
  limit: number;
  remaining: number;
}> => {
  try {
    const response = await apiGet<{
      used: number;
      limit: number;
      remaining: number;
    }>(DATASET_ENDPOINTS.QUOTA);

    console.debug('[DatasetService] Storage quota fetched', {
      used: formatFileSize(response.used),
      limit: formatFileSize(response.limit),
    });

    return response;
  } catch (error) {
    console.error('[DatasetService] Failed to fetch storage quota', error);
    throw error;
  }
};

// ============================================================================
// Default Export
// ============================================================================

export default {
  // CRUD Operations
  fetchDatasets,
  getDatasetById,
  createDataset,
  uploadDataset,
  updateDataset,
  deleteDataset,
  bulkDeleteDatasets,

  // Details & Preview
  getDatasetPreview,
  getDatasetMetadata,
  getDatasetStatistics,

  // Operations
  duplicateDataset,
  exportDataset,

  // Sharing & Permissions
  shareDataset,
  getDatasetPermissions,
  updateDatasetPermissions,

  // Search & Discovery
  searchDatasets,

  // Processing & Status
  getDatasetProcessingStatus,
  cancelDatasetProcessing,

  // Storage & Quota
  getStorageQuota,

  // Utility functions (exported for direct use if needed)
  validateFile,
  formatFileSize,
};
