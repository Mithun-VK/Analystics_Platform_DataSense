// src/hooks/useEDA.ts

import { useState, useCallback, useEffect, useRef } from 'react';
import edaService from '@/services/edaService';
import { uiStore } from '@/store/uiStore';
import type {
  EDAResult,
  EDAStatus,
  DescriptiveStatistics,
  CorrelationMatrix,
  MissingValues,
  OutlierAnalysis,
  DistributionAnalysis,
} from '@/types/eda.types';

/**
 * Progress tracking for EDA analysis stages
 */
interface EDAProgress {
  stage:
    | 'initializing'
    | 'loading'
    | 'processing'
    | 'generating_statistics'
    | 'analyzing_correlations'
    | 'detecting_outliers'
    | 'completed';
  percentage: number;
  message: string;
  startedAt: number;
  estimatedTimeRemaining?: number;
}

/**
 * ✅ FIXED: Generic analysis result with explicit data type
 */
interface EDAAnalysisResult<T = undefined> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp?: number;
}

/**
 * Configuration options for EDA hook
 */
interface UseEDAOptions {
  pollInterval?: number;
  maxRetries?: number;
  retryDelay?: number;
  timeout?: number;
}

/**
 * Cache entry for EDA results
 */
interface EDACache {
  datasetId: string;
  result: EDAResult;
  timestamp: number;
  expiresAt: number;
}

// ============================================================================
// Constants
// ============================================================================

const POLL_INTERVAL = 2000; // 2 seconds
const MAX_RETRIES = 5;
const RETRY_DELAY = 1000; // 1 second
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes
const TIMEOUT = 30 * 60 * 1000; // 30 minutes

const STAGE_WEIGHTS: Record<EDAProgress['stage'], number> = {
  initializing: 5,
  loading: 15,
  processing: 30,
  generating_statistics: 25,
  analyzing_correlations: 15,
  detecting_outliers: 10,
  completed: 100,
};

// ============================================================================
// Custom Hook
// ============================================================================

/**
 * Custom hook for managing EDA (Exploratory Data Analysis) operations
 * Handles async analysis triggering, polling, progress tracking, and caching
 *
 * @param options - Configuration options for polling, retries, and timeout
 * @returns EDA hook return value with analysis methods and state
 *
 * @example
 * const { startAnalysis, progress, edaResult } = useEDA();
 * await startAnalysis(datasetId);
 */
export const useEDA = (options: UseEDAOptions = {}) => {
  const {
    pollInterval = POLL_INTERVAL,
    maxRetries = MAX_RETRIES,
    retryDelay = RETRY_DELAY,
    timeout = TIMEOUT,
  } = options;

  const addNotification = uiStore((state) => state.addNotification);

  // ============================================================================
  // State Management
  // ============================================================================

  const [edaResult, setEdaResult] = useState<EDAResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<EDAProgress>({
    stage: 'initializing',
    percentage: 0,
    message: 'Ready to analyze',
    startedAt: 0,
  });

  // Analysis metadata
  const [currentDatasetId, setCurrentDatasetId] = useState<string | null>(null);
  // ✅ FIXED: Change EDAStatus to allow 'idle' literal
  const [analysisStatus, setAnalysisStatus] = useState<EDAStatus>('idle' as EDAStatus);
  const [retryCount, setRetryCount] = useState(0);

  // ============================================================================
  // Refs for Cleanup & Polling
  // ============================================================================

  const abortControllerRef = useRef<AbortController | null>(null);
  const pollTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const analysisTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const cacheRef = useRef<Map<string, EDACache>>(new Map());
  const startTimeRef = useRef<number>(0);

  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * Error handler utility
   */
  const handleError = useCallback(
    (err: unknown, context: string) => {
      const errorMessage =
        err instanceof Error ? err.message : `An error occurred in ${context}`;
      setError(errorMessage);
      setAnalysisStatus('failed' as EDAStatus);
      console.error(`[useEDA] ${context}:`, err);
      addNotification({
        type: 'error',
        message: errorMessage,
        duration: 5000,
      });
    },
    [addNotification]
  );

  /**
   * Calculate estimated time remaining based on current stage
   */
  const calculateEstimatedTime = useCallback(
    (stage: EDAProgress['stage']) => {
      if (!startTimeRef.current) return undefined;

      const elapsedTime = Date.now() - startTimeRef.current;
      const currentWeight = STAGE_WEIGHTS[stage];
      const estimatedTotalTime = (elapsedTime / currentWeight) * 100;
      const remaining = Math.max(0, estimatedTotalTime - elapsedTime);

      return remaining > 0 ? remaining : undefined;
    },
    []
  );

  /**
   * Update progress with enhanced metrics
   */
  const updateProgress = useCallback(
    (stage: EDAProgress['stage'], percentage: number, message: string) => {
      setProgress({
        stage,
        percentage,
        message,
        startedAt: startTimeRef.current,
        estimatedTimeRemaining: calculateEstimatedTime(stage),
      });
    },
    [calculateEstimatedTime]
  );

  /**
   * Get from cache
   */
  const getFromCache = useCallback((datasetId: string): EDAResult | null => {
    const cached = cacheRef.current.get(datasetId);
    if (cached && Date.now() < cached.expiresAt) {
      return cached.result;
    }
    cacheRef.current.delete(datasetId);
    return null;
  }, []);

  /**
   * Save to cache
   */
  const saveToCache = useCallback((datasetId: string, result: EDAResult) => {
    cacheRef.current.set(datasetId, {
      datasetId,
      result,
      timestamp: Date.now(),
      expiresAt: Date.now() + CACHE_DURATION,
    });
  }, []);

  /**
   * Exponential backoff retry logic
   */
  const exponentialBackoff = (attempt: number): number => {
    const delayMs = retryDelay * Math.pow(2, attempt);
    const jitterMs = Math.random() * 1000;
    return delayMs + jitterMs;
  };

  // ============================================================================
  // Analysis Operations
  // ============================================================================

  /**
   * ✅ FIXED: Trigger EDA analysis with proper return type
   */
  const triggerAnalysis = useCallback(
    async (
      datasetId: string
    ): Promise<EDAAnalysisResult<{ jobId: string }>> => {
      // Check cache first
      const cached = getFromCache(datasetId);
      if (cached) {
        setEdaResult(cached);
        setAnalysisStatus('completed' as EDAStatus);
        addNotification({
          type: 'info',
          message: 'Using cached EDA results',
          duration: 3000,
        });
        return {
          success: true,
          data: { jobId: 'cached' },
          timestamp: Date.now(),
        };
      }

      setIsLoading(true);
      setError(null);
      setCurrentDatasetId(datasetId);
      setAnalysisStatus('pending' as EDAStatus);
      setRetryCount(0);
      startTimeRef.current = Date.now();
      updateProgress('initializing', 5, 'Starting EDA analysis...');

      // Cancel previous request if ongoing
      abortControllerRef.current?.abort();
      abortControllerRef.current = new AbortController();

      try {
        updateProgress('loading', 10, 'Initializing analysis engine...');

        const response = await edaService.triggerAnalysis(datasetId, {
          signal: abortControllerRef.current.signal,
        });

        setAnalysisStatus('processing' as EDAStatus);
        updateProgress('processing', 15, 'Analysis queued successfully');

        addNotification({
          type: 'info',
          message: 'EDA analysis started. Analyzing your dataset...',
          duration: 3000,
        });

        return {
          success: true,
          data: { jobId: response.jobId },
          timestamp: Date.now(),
        };
      } catch (err) {
        if (!(err instanceof Error && err.name === 'AbortError')) {
          handleError(err, 'triggerAnalysis');
        }
        return {
          success: false,
          error:
            err instanceof Error ? err.message : 'Failed to trigger analysis',
          timestamp: Date.now(),
        };
      } finally {
        setIsLoading(false);
      }
    },
    [getFromCache, addNotification, handleError, updateProgress]
  );

  /**
   * Poll for analysis results with retry logic
   */
  const pollAnalysisResults = useCallback(
    async (jobId: string, datasetId: string) => {
      const poll = async () => {
        try {
          updateProgress('processing', 30, 'Fetching analysis results...');

          const response = await edaService.getAnalysisResults(jobId, {
            signal: abortControllerRef.current?.signal,
          });

          if (response.status === 'completed' && response.result) {
            // Update progress through stages
            updateProgress(
              'generating_statistics',
              50,
              'Generating statistics...'
            );
            await new Promise((resolve) => setTimeout(resolve, 500));

            updateProgress(
              'analyzing_correlations',
              70,
              'Analyzing correlations...'
            );
            await new Promise((resolve) => setTimeout(resolve, 500));

            updateProgress('detecting_outliers', 85, 'Detecting outliers...');
            await new Promise((resolve) => setTimeout(resolve, 500));

            setEdaResult(response.result);
            setAnalysisStatus('completed' as EDAStatus);
            updateProgress('completed', 100, 'Analysis complete!');
            saveToCache(datasetId, response.result);

            addNotification({
              type: 'success',
              message: 'EDA analysis completed successfully',
              duration: 3000,
            });

            return true;
          } else if (response.status === 'processing') {
            updateProgress(
              'processing',
              Math.min(99, 15 + (response.progress || 0)),
              response.message || 'Analysis in progress...'
            );

            // Check timeout
            if (Date.now() - startTimeRef.current > timeout) {
              throw new Error('Analysis timeout exceeded');
            }

            // Continue polling
            pollTimeoutRef.current = setTimeout(poll, pollInterval);
            return false;
          } else if (response.status === 'failed') {
            throw new Error(response.error || 'Analysis failed');
          }

          return false;
        } catch (err) {
          if (!(err instanceof Error && err.name === 'AbortError')) {
            if (retryCount < maxRetries) {
              const delay = exponentialBackoff(retryCount);
              setRetryCount((prev) => prev + 1);
              updateProgress(
                'processing',
                Math.min(99, 15 + retryCount * 5),
                `Retrying... (Attempt ${retryCount + 1}/${maxRetries})`
              );

              pollTimeoutRef.current = setTimeout(poll, delay);
              return false;
            } else {
              handleError(err, `pollAnalysisResults after ${maxRetries} retries`);
              setAnalysisStatus('failed' as EDAStatus);
              return true; // Stop polling
            }
          }
          return false;
        }
      };

      return poll();
    },
    [
      pollInterval,
      timeout,
      maxRetries,
      retryCount,
      updateProgress,
      saveToCache,
      addNotification,
      handleError,
    ]
  );

  /**
   * ✅ FIXED: Start analysis workflow with proper return type
   */
  const startAnalysis = useCallback(
    async (datasetId: string): Promise<EDAAnalysisResult<EDAResult>> => {
      try {
        const triggerResult = await triggerAnalysis(datasetId);

        if (!triggerResult.success || !triggerResult.data) {
          return {
            success: false,
            error: triggerResult.error || 'Failed to trigger analysis',
            timestamp: Date.now(),
          };
        }

        const jobId = triggerResult.data.jobId;

        // Only poll if not using cache
        if (jobId !== 'cached') {
          // Set timeout for entire analysis
          analysisTimeoutRef.current = setTimeout(() => {
            setError('Analysis timeout exceeded');
            setAnalysisStatus('failed' as EDAStatus);
            abortControllerRef.current?.abort();
            addNotification({
              type: 'error',
              message: 'Analysis took too long and was cancelled',
              duration: 5000,
            });
          }, timeout);

          // Start polling
          await pollAnalysisResults(jobId, datasetId);
        }

        return {
          success: true,
          data: edaResult || undefined,
          timestamp: Date.now(),
        };
      } catch (err) {
        handleError(err, 'startAnalysis');
        return {
          success: false,
          error: err instanceof Error ? err.message : 'Analysis failed',
          timestamp: Date.now(),
        };
      }
    },
    [
      triggerAnalysis,
      pollAnalysisResults,
      edaResult,
      timeout,
      addNotification,
      handleError,
    ]
  );

  /**
   * Cancel ongoing analysis
   */
  const cancelAnalysis = useCallback(() => {
    abortControllerRef.current?.abort();
    if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
    if (analysisTimeoutRef.current) clearTimeout(analysisTimeoutRef.current);

    setAnalysisStatus('idle' as EDAStatus);
    setIsLoading(false);
    updateProgress('initializing', 0, 'Analysis cancelled');

    addNotification({
      type: 'info',
      message: 'Analysis cancelled',
      duration: 2000,
    });
  }, [updateProgress, addNotification]);

  // ============================================================================
  // Data Access Methods
  // ============================================================================

  /**
   * Get analysis statistics
   */
  const getStatistics = useCallback((): DescriptiveStatistics | null => {
    return edaResult?.statistics || null;
  }, [edaResult]);

  /**
   * Get correlation matrix
   */
  const getCorrelations = useCallback((): CorrelationMatrix | null => {
    return edaResult?.correlationMatrix || null;
  }, [edaResult]);

  /**
   * Get missing values analysis
   */
  const getMissingValues = useCallback((): MissingValues | null => {
    return edaResult?.missingValues || null;
  }, [edaResult]);

  /**
   * Get outlier analysis
   */
  const getOutliers = useCallback((): OutlierAnalysis | null => {
    return edaResult?.outlierAnalysis || null;
  }, [edaResult]);

  /**
   * Get distribution analysis
   */
  const getDistributions = useCallback((): DistributionAnalysis[] | null => {
    return edaResult?.distributionAnalysis || null;
  }, [edaResult]);

  // ============================================================================
  // Utility Methods
  // ============================================================================

  /**
   * Clear results
   */
  const clearResults = useCallback(() => {
    setEdaResult(null);
    setAnalysisStatus('idle' as EDAStatus);
    setError(null);
    setCurrentDatasetId(null);
    updateProgress('initializing', 0, 'Ready to analyze');
  }, [updateProgress]);

  /**
   * Clear cache
   */
  const clearCache = useCallback((datasetId?: string) => {
    if (datasetId) {
      cacheRef.current.delete(datasetId);
    } else {
      cacheRef.current.clear();
    }
  }, []);

  /**
   * Export results
   */
  const exportResults = useCallback(
    (format: 'json' | 'csv' = 'json'): EDAAnalysisResult<undefined> => {
      if (!edaResult) {
        return {
          success: false,
          error: 'No analysis results to export',
          timestamp: Date.now(),
        };
      }

      try {
        let content = '';
        let filename = `eda_analysis_${Date.now()}`;

        if (format === 'json') {
          content = JSON.stringify(edaResult, null, 2);
          filename += '.json';
        } else if (format === 'csv') {
          // Export statistics to CSV
          const headers = ['Statistic', 'Value'];
          const rows = Object.entries(edaResult.statistics || {}).map(
            ([key, value]) => [key, value]
          );
          content =
            [headers, ...rows]
              .map((row) => row.join(','))
              .join('\n') +
            '\n\nCorrelations:\n' +
            JSON.stringify(edaResult.correlationMatrix, null, 2);
          filename += '.csv';
        }

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);

        addNotification({
          type: 'success',
          message: `Results exported as ${format.toUpperCase()}`,
          duration: 3000,
        });

        return {
          success: true,
          timestamp: Date.now(),
        };
      } catch (err) {
        handleError(err, 'exportResults');
        return {
          success: false,
          error: err instanceof Error ? err.message : 'Export failed',
          timestamp: Date.now(),
        };
      }
    },
    [edaResult, addNotification, handleError]
  );

  // ============================================================================
  // Effects
  // ============================================================================

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
      if (analysisTimeoutRef.current) clearTimeout(analysisTimeoutRef.current);
    };
  }, []);

  // ============================================================================
  // Return Hook Value
  // ============================================================================

  return {
    // State
    edaResult,
    isLoading,
    error,
    progress,
    analysisStatus,
    currentDatasetId,

    // Analysis operations
    startAnalysis,
    triggerAnalysis,
    cancelAnalysis,

    // Data access
    getStatistics,
    getCorrelations,
    getMissingValues,
    getOutliers,
    getDistributions,

    // Utilities
    clearResults,
    clearCache,
    exportResults,
  };
};

export type UseEDAReturn = ReturnType<typeof useEDA>;
