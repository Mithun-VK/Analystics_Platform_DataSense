// src/services/edaService.ts - FULLY CORRECTED PRODUCTION-GRADE VERSION (NO ERRORS)

import { apiPost, apiGet, apiDelete } from './api';
import type {
  EDAResult,
  EDAStatus,
  DescriptiveStatistics,
  CorrelationMatrix,
  MissingValues,
  OutlierAnalysis,
  DistributionAnalysis,
  AnomalyDetection,
} from '@/types/eda.types';

// ============================================================================
// Environment Utilities
// ============================================================================

/**
 * ✅ FIXED: Safe environment check for development mode
 */
const isDevelopment = (): boolean => {
  try {
    // Try Vite environment
    if (typeof import.meta !== 'undefined' && import.meta !== null) {
      const env = (import.meta as unknown as { env?: Record<string, unknown> }).env;
      if (env && typeof env === 'object') {
        return (env as Record<string, unknown>)['DEV'] === true;
      }
    }
  } catch {
    // Fallback if import.meta is not available
  }

  // Try Node.js environment
  try {
    if (typeof process !== 'undefined' && process.env) {
      const nodeEnv = process.env['NODE_ENV'];
      return nodeEnv === 'development';
    }
  } catch {
    // Fallback
  }

  return false;
};

/**
 * ✅ FIXED: Safe debug logging helper
 */
const debugLog = (message: string, data?: unknown): void => {
  if (isDevelopment()) {
    console.debug(message, data);
  }
};

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * ✅ FIXED: Explicit type definitions for API responses
 */
interface ReportResponse {
  report: {
    title: string;
    summary: string;
    sections: Array<{
      name: string;
      content: string;
      visualizations?: string[];
    }>;
    recommendations: string[];
    generatedAt: string;
  };
}

interface ComparisonResponse {
  jobId: string;
  comparison: {
    datasets: Array<{
      id: string;
      statistics: DescriptiveStatistics;
    }>;
    differences: Record<string, unknown>;
  };
}

interface InsightsResponse {
  insights: Array<{
    type: string;
    title: string;
    description: string;
    severity: 'low' | 'medium' | 'high';
    action?: string;
  }>;
}

interface ValidationResponse {
  qualityScore: number;
  issues: Array<{
    column: string;
    issue: string;
    severity: 'warning' | 'error';
    count: number;
  }>;
}

interface AnalysisOptions {
  includeOutliers?: boolean;
  includeCorrelations?: boolean;
  includeMissingValues?: boolean;
  includeDistributions?: boolean;
  includeAnomalies?: boolean;
  samplingRate?: number;
  targetColumns?: string[];
}

interface AnalysisResponse {
  jobId: string;
  datasetId: string;
  status: EDAStatus;
  progress: number;
  message: string;
  result?: EDAResult;
  error?: string;
}

interface ComparisonOptions {
  datasets: string[];
  metrics?: string[];
  visualizations?: boolean;
}

interface ScheduleOptions {
  datasetId: string;
  frequency: 'daily' | 'weekly' | 'monthly';
  time?: string;
  enabled: boolean;
}

interface ScheduledJob {
  id: string;
  datasetId: string;
  frequency: string;
  nextRun: string;
  lastRun?: string;
  enabled: boolean;
}

// ============================================================================
// Constants
// ============================================================================

const EDA_ENDPOINTS = {
  TRIGGER_ANALYSIS: '/eda/analyze',
  GET_RESULTS: '/eda/results/:jobId',
  GET_ANALYSIS_BY_DATASET: '/eda/dataset/:datasetId',
  CANCEL_ANALYSIS: '/eda/jobs/:jobId/cancel',
  DELETE_ANALYSIS: '/eda/results/:jobId',
  LIST_ANALYSES: '/eda/analyses',
  EXPORT_REPORT: '/eda/results/:jobId/export/:format',
  COMPARE_DATASETS: '/eda/compare',
  GENERATE_INSIGHTS: '/eda/results/:jobId/insights',
  VALIDATE_DATA: '/eda/validate',
  DETECT_ANOMALIES: '/eda/anomalies/:datasetId',
  GENERATE_REPORT: '/eda/results/:jobId/report',
  SCHEDULE_ANALYSIS: '/eda/schedule',
  GET_SCHEDULED_JOBS: '/eda/scheduled-jobs',
  CANCEL_SCHEDULED_JOB: '/eda/scheduled-jobs/:jobId/cancel',
} as const;

// ============================================================================
// API Service Functions
// ============================================================================

/**
 * Trigger EDA analysis on dataset
 * Initiates async analysis job on the server
 */
export const triggerAnalysis = async (
  datasetId: string,
  options?: AnalysisOptions & { signal?: AbortSignal }
): Promise<{ jobId: string }> => {
  try {
    const payload = {
      datasetId,
      includeOutliers: options?.includeOutliers ?? true,
      includeCorrelations: options?.includeCorrelations ?? true,
      includeMissingValues: options?.includeMissingValues ?? true,
      includeDistributions: options?.includeDistributions ?? true,
      includeAnomalies: options?.includeAnomalies ?? false,
      samplingRate: options?.samplingRate ?? 1.0,
      targetColumns: options?.targetColumns,
    };

    const response = await apiPost<{ jobId: string }>(
      EDA_ENDPOINTS.TRIGGER_ANALYSIS,
      payload,
      { signal: options?.signal }
    );

    debugLog('[EDAService] Analysis triggered', {
      jobId: response.jobId,
      datasetId,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to trigger analysis', error);
    throw error;
  }
};

/**
 * Get analysis results by job ID
 * Polls for results or retrieves completed analysis
 */
export const getAnalysisResults = async (
  jobId: string,
  options?: { signal?: AbortSignal }
): Promise<AnalysisResponse> => {
  try {
    const url = EDA_ENDPOINTS.GET_RESULTS.replace(':jobId', jobId);
    const response = await apiGet<AnalysisResponse>(url, {
      signal: options?.signal,
    });

    debugLog('[EDAService] Analysis results fetched', {
      jobId,
      status: response.status,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to fetch analysis results', error);
    throw error;
  }
};

/**
 * Get latest analysis for a dataset
 */
export const getAnalysisByDataset = async (
  datasetId: string
): Promise<EDAResult | null> => {
  try {
    const url = EDA_ENDPOINTS.GET_ANALYSIS_BY_DATASET.replace(':datasetId', datasetId);
    const response = await apiGet<{ result: EDAResult | null }>(url);

    debugLog('[EDAService] Dataset analysis fetched', datasetId);

    return response.result;
  } catch (error) {
    console.error('[EDAService] Failed to fetch dataset analysis', error);
    throw error;
  }
};

/**
 * List all analyses for user
 */
export const listAnalyses = async (
  page: number = 1,
  limit: number = 20,
  filters?: { datasetId?: string; status?: EDAStatus }
): Promise<{
  analyses: AnalysisResponse[];
  pagination: { page: number; limit: number; total: number };
}> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(filters?.datasetId && { datasetId: filters.datasetId }),
      ...(filters?.status && { status: filters.status }),
    });

    const url = `${EDA_ENDPOINTS.LIST_ANALYSES}?${queryParams.toString()}`;
    const response = await apiGet<{
      analyses: AnalysisResponse[];
      pagination: { page: number; limit: number; total: number };
    }>(url);

    debugLog('[EDAService] Analyses listed', response.analyses.length);

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to list analyses', error);
    throw error;
  }
};

/**
 * Cancel ongoing analysis job
 */
export const cancelAnalysis = async (jobId: string): Promise<void> => {
  try {
    const url = EDA_ENDPOINTS.CANCEL_ANALYSIS.replace(':jobId', jobId);
    await apiPost<void>(url, {}, { skipRetry: true });

    debugLog('[EDAService] Analysis cancelled', jobId);
  } catch (error) {
    console.error('[EDAService] Failed to cancel analysis', error);
    throw error;
  }
};

/**
 * Delete analysis results
 */
export const deleteAnalysis = async (jobId: string): Promise<void> => {
  try {
    const url = EDA_ENDPOINTS.DELETE_ANALYSIS.replace(':jobId', jobId);
    await apiDelete<void>(url);

    debugLog('[EDAService] Analysis deleted', jobId);
  } catch (error) {
    console.error('[EDAService] Failed to delete analysis', error);
    throw error;
  }
};

/**
 * Export analysis report in specified format
 */
export const exportAnalysisReport = async (
  jobId: string,
  format: 'pdf' | 'html' | 'json' | 'excel' = 'pdf',
  onProgress?: (progress: number) => void
): Promise<Blob> => {
  try {
    const url = EDA_ENDPOINTS.EXPORT_REPORT
      .replace(':jobId', jobId)
      .replace(':format', format);

    // Download file with progress tracking
    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.open('GET', url, true);
      xhr.responseType = 'blob';

      // Get auth token safely
      const token = localStorage.getItem('auth_access_token');
      if (token && typeof token === 'string') {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }

      xhr.onprogress = (event) => {
        if (event.lengthComputable && event.total > 0) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress?.(progress);
        }
      };

      xhr.onload = () => {
        if (xhr.status === 200) {
          debugLog('[EDAService] Report exported', { jobId, format });
          resolve(xhr.response as Blob);
        } else {
          reject(new Error(`Export failed with status ${xhr.status}`));
        }
      };

      xhr.onerror = () => {
        reject(new Error('Export request failed'));
      };

      xhr.send();
    });
  } catch (error) {
    console.error('[EDAService] Failed to export report', error);
    throw error;
  }
};

/**
 * Compare multiple datasets with proper type
 */
export const compareDatasets = async (
  options: ComparisonOptions & { signal?: AbortSignal }
): Promise<ComparisonResponse> => {
  try {
    const payload = {
      datasets: options.datasets,
      metrics: options.metrics || [
        'mean',
        'median',
        'std',
        'min',
        'max',
        'nullCount',
      ],
      visualizations: options.visualizations ?? true,
    };

    const response = await apiPost<ComparisonResponse>(
      EDA_ENDPOINTS.COMPARE_DATASETS,
      payload,
      { signal: options.signal }
    );

    debugLog('[EDAService] Datasets compared', {
      datasetCount: options.datasets.length,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to compare datasets', error);
    throw error;
  }
};

/**
 * Generate AI insights from analysis results with proper type
 */
export const generateInsights = async (
  jobId: string
): Promise<InsightsResponse> => {
  try {
    const url = EDA_ENDPOINTS.GENERATE_INSIGHTS.replace(':jobId', jobId);

    const response = await apiPost<InsightsResponse>(url, {});

    debugLog('[EDAService] Insights generated', {
      jobId,
      insightCount: response.insights.length,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to generate insights', error);
    throw error;
  }
};

/**
 * Validate data quality with proper type
 */
export const validateData = async (
  datasetId: string
): Promise<ValidationResponse> => {
  try {
    const response = await apiPost<ValidationResponse>(
      EDA_ENDPOINTS.VALIDATE_DATA,
      { datasetId }
    );

    debugLog('[EDAService] Data validated', {
      datasetId,
      qualityScore: response.qualityScore,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to validate data', error);
    throw error;
  }
};

/**
 * Detect anomalies in dataset
 */
export const detectAnomalies = async (
  datasetId: string,
  options?: {
    method?: 'isolation_forest' | 'local_outlier_factor' | 'statistical';
    threshold?: number;
    targetColumns?: string[];
  }
): Promise<AnomalyDetection> => {
  try {
    const url = EDA_ENDPOINTS.DETECT_ANOMALIES.replace(':datasetId', datasetId);
    const payload = {
      method: options?.method ?? 'isolation_forest',
      threshold: options?.threshold ?? 0.05,
      targetColumns: options?.targetColumns,
    };

    const response = await apiPost<AnomalyDetection>(url, payload);

    debugLog('[EDAService] Anomalies detected', {
      datasetId,
      anomalyCount: response.anomalies?.length || 0,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to detect anomalies', error);
    throw error;
  }
};

/**
 * Generate comprehensive EDA report with proper type
 */
export const generateReport = async (
  jobId: string,
  options?: {
    includeCharts?: boolean;
    includeRecommendations?: boolean;
    detail?: 'summary' | 'detailed' | 'comprehensive';
  }
): Promise<ReportResponse> => {
  try {
    const url = EDA_ENDPOINTS.GENERATE_REPORT.replace(':jobId', jobId);
    const payload = {
      includeCharts: options?.includeCharts ?? true,
      includeRecommendations: options?.includeRecommendations ?? true,
      detail: options?.detail ?? 'detailed',
    };

    const response = await apiPost<ReportResponse>(url, payload);

    debugLog('[EDAService] Report generated', jobId);

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to generate report', error);
    throw error;
  }
};

/**
 * Schedule recurring EDA analysis
 */
export const scheduleAnalysis = async (
  options: ScheduleOptions
): Promise<ScheduledJob> => {
  try {
    const response = await apiPost<ScheduledJob>(
      EDA_ENDPOINTS.SCHEDULE_ANALYSIS,
      {
        datasetId: options.datasetId,
        frequency: options.frequency,
        time: options.time,
        enabled: options.enabled,
      }
    );

    debugLog('[EDAService] Analysis scheduled', {
      datasetId: options.datasetId,
      frequency: options.frequency,
    });

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to schedule analysis', error);
    throw error;
  }
};

/**
 * Get scheduled analysis jobs
 */
export const getScheduledJobs = async (): Promise<ScheduledJob[]> => {
  try {
    const response = await apiGet<ScheduledJob[]>(
      EDA_ENDPOINTS.GET_SCHEDULED_JOBS
    );

    debugLog('[EDAService] Scheduled jobs fetched', response.length);

    return response;
  } catch (error) {
    console.error('[EDAService] Failed to fetch scheduled jobs', error);
    throw error;
  }
};

/**
 * Cancel scheduled analysis job
 */
export const cancelScheduledJob = async (jobId: string): Promise<void> => {
  try {
    const url = EDA_ENDPOINTS.CANCEL_SCHEDULED_JOB.replace(':jobId', jobId);
    await apiPost<void>(url, {});

    debugLog('[EDAService] Scheduled job cancelled', jobId);
  } catch (error) {
    console.error('[EDAService] Failed to cancel scheduled job', error);
    throw error;
  }
};

// ============================================================================
// Data Extraction Utilities
// ============================================================================

/**
 * Get descriptive statistics from analysis result
 */
export const extractStatistics = (
  result: EDAResult
): DescriptiveStatistics | null => {
  return result?.statistics || null;
};

/**
 * Get correlation matrix from analysis result
 */
export const extractCorrelations = (
  result: EDAResult
): CorrelationMatrix | null => {
  return result?.correlationMatrix || null;
};

/**
 * Get missing values analysis from result
 */
export const extractMissingValues = (
  result: EDAResult
): MissingValues | null => {
  return result?.missingValues || null;
};

/**
 * Get outlier analysis from result
 */
export const extractOutliers = (
  result: EDAResult
): OutlierAnalysis | null => {
  return result?.outlierAnalysis || null;
};

/**
 * Get distribution analysis from result
 */
export const extractDistributions = (
  result: EDAResult
): DistributionAnalysis[] | null => {
  return result?.distributionAnalysis || null;
};

// ============================================================================
// Download & Export Utilities
// ============================================================================

/**
 * Download analysis as file
 */
export const downloadAnalysis = async (
  jobId: string,
  format: 'pdf' | 'html' | 'json' | 'excel' = 'pdf'
): Promise<void> => {
  try {
    const blob = await exportAnalysisReport(jobId, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `eda_analysis_${jobId}.${
      format === 'excel' ? 'xlsx' : format
    }`;
    link.click();
    URL.revokeObjectURL(url);

    debugLog('[EDAService] Analysis downloaded', { jobId, format });
  } catch (error) {
    console.error('[EDAService] Failed to download analysis', error);
    throw error;
  }
};

/**
 * Get analysis job status with polling
 */
export const pollAnalysisStatus = async (
  jobId: string,
  maxAttempts: number = 180,
  interval: number = 2000
): Promise<AnalysisResponse> => {
  let attempts = 0;

  return new Promise((resolve, reject) => {
    const poll = async (): Promise<void> => {
      try {
        const result = await getAnalysisResults(jobId);

        if (result.status === 'completed' || result.status === 'failed') {
          resolve(result);
        } else if (attempts < maxAttempts) {
          attempts += 1;
          setTimeout(() => {
            void poll();
          }, interval);
        } else {
          reject(new Error('Analysis polling timeout exceeded'));
        }
      } catch (error) {
        reject(error);
      }
    };

    void poll();
  });
};

// ============================================================================
// Default Export
// ============================================================================

export default {
  triggerAnalysis,
  getAnalysisResults,
  getAnalysisByDataset,
  listAnalyses,
  cancelAnalysis,
  deleteAnalysis,
  exportAnalysisReport,
  compareDatasets,
  generateInsights,
  validateData,
  detectAnomalies,
  generateReport,
  scheduleAnalysis,
  getScheduledJobs,
  cancelScheduledJob,
  extractStatistics,
  extractCorrelations,
  extractMissingValues,
  extractOutliers,
  extractDistributions,
  downloadAnalysis,
  pollAnalysisStatus,
};
