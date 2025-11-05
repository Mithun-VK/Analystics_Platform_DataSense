// src/services/insightService.ts
import { apiPost, apiGet, apiDelete } from './api';

/**
 * Insight Service - Handles AI-generated insights, recommendations, and advanced analytics
 * Provides intelligent analysis and actionable recommendations for data-driven decisions
 */

const INSIGHT_ENDPOINTS = {
  GENERATE_INSIGHTS: '/insights/generate',
  GET_INSIGHTS: '/insights/:datasetId',
  GET_INSIGHT_DETAIL: '/insights/:insightId',
  LIST_INSIGHTS: '/insights',
  DELETE_INSIGHT: '/insights/:insightId',
  RATE_INSIGHT: '/insights/:insightId/rate',
  BOOKMARK_INSIGHT: '/insights/:insightId/bookmark',
  SHARE_INSIGHT: '/insights/:insightId/share',
  GET_RECOMMENDATIONS: '/insights/recommendations/:datasetId',
  ANOMALY_DETECTION: '/insights/anomalies/:datasetId',
  TREND_ANALYSIS: '/insights/trends/:datasetId',
  CORRELATION_ANALYSIS: '/insights/correlations/:datasetId',
  PREDICTIVE_ANALYSIS: '/insights/predictions/:datasetId',
  CLUSTERING_ANALYSIS: '/insights/clustering/:datasetId',
  CLASSIFICATION_ANALYSIS: '/insights/classification/:datasetId',
  TIME_SERIES_ANALYSIS: '/insights/timeseries/:datasetId',
  CUSTOM_ANALYSIS: '/insights/custom',
  EXPORT_INSIGHTS: '/insights/:datasetId/export/:format',
  GENERATE_REPORT: '/insights/:datasetId/report',
  SCHEDULE_INSIGHTS: '/insights/schedule',
  GET_SCHEDULED_INSIGHTS: '/insights/scheduled',
  CANCEL_SCHEDULED_INSIGHT: '/insights/scheduled/:jobId/cancel',
  COMPARE_DATASETS: '/insights/compare',
  FEATURE_IMPORTANCE: '/insights/feature-importance/:datasetId',
  BUSINESS_METRICS: '/insights/business-metrics/:datasetId',
  SENTIMENT_ANALYSIS: '/insights/sentiment/:datasetId',
  NLP_ANALYSIS: '/insights/nlp/:datasetId',
  MARKET_ANALYSIS: '/insights/market/:datasetId',
  COMPETITOR_ANALYSIS: '/insights/competitor/:datasetId',
};

interface InsightConfig {
  datasetId: string;
  analysisTypes?: Array<
    | 'summary'
    | 'anomalies'
    | 'trends'
    | 'correlations'
    | 'patterns'
    | 'predictions'
    | 'recommendations'
    | 'risks'
    | 'opportunities'
  >;
  columns?: string[];
  depth?: 'basic' | 'intermediate' | 'advanced';
  includeVisualizations?: boolean;
  includeRecommendations?: boolean;
  includeComparisons?: boolean;
  focusArea?: string;
  industry?: string;
  businessContext?: string;
}

interface Insight {
  id: string;
  datasetId: string;
  type: string;
  title: string;
  description: string;
  finding: string;
  significance: 'low' | 'medium' | 'high' | 'critical';
  confidence: number; // 0-1
  impact: number; // 0-1
  evidence: {
    metric: string;
    value: number | string;
    change?: number;
    trend?: 'up' | 'down' | 'stable';
  }[];
  relatedColumns: string[];
  visualizationSuggestion?: {
    type: string;
    config: any;
  };
  recommendation?: string;
  actionItems?: Array<{
    action: string;
    priority: 'low' | 'medium' | 'high';
    impact: string;
  }>;
  tags: string[];
  createdAt: string;
  updatedAt: string;
  rating?: number; // 1-5
  userNotes?: string;
  bookmarked?: boolean;
}

interface AnomalyInsight extends Insight {
  anomalies: Array<{
    index: number;
    values: Record<string, any>;
    anomalyScore: number;
    anomalyType: 'univariate' | 'multivariate' | 'contextual';
  }>;
}

interface TrendInsight extends Insight {
  trend: {
    direction: 'upward' | 'downward' | 'cyclical' | 'stable';
    slope: number;
    seasonality?: string;
    forecastedValues?: number[];
    forecastConfidence: number;
  };
}

interface CorrelationInsight extends Insight {
  correlations: Array<{
    column1: string;
    column2: string;
    coefficient: number;
    strength: 'weak' | 'moderate' | 'strong' | 'very_strong';
    type: 'positive' | 'negative';
    pvalue?: number;
  }>;
}

interface PredictionInsight extends Insight {
  predictions: {
    model: string;
    accuracy: number;
    predictions: Array<{ features: Record<string, any>; prediction: number | string }>;
    featureImportance: Record<string, number>;
  };
}

interface Recommendation {
  id: string;
  title: string;
  description: string;
  category: 'optimization' | 'risk_mitigation' | 'opportunity' | 'improvement';
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedImpact: string;
  implementationDifficulty: 'easy' | 'medium' | 'hard';
  timeline: string;
  resources: string[];
  dependencies?: string[];
  successMetrics: string[];
}

interface InsightReport {
  datasetId: string;
  generatedAt: string;
  summaryFindings: string;
  keyMetrics: Record<string, any>;
  insights: Insight[];
  recommendations: Recommendation[];
  riskAssessment: {
    overallRisk: 'low' | 'medium' | 'high' | 'critical';
    risks: Array<{ description: string; severity: string; mitigation: string }>;
  };
  opportunityAssessment: {
    opportunities: Array<{ description: string; potential: string; timeline: string }>;
  };
}

interface ScheduleConfig {
  datasetId: string;
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  analysisTypes: string[];
  deliveryMethod: 'email' | 'dashboard' | 'both';
  recipients?: string[];
  enabled: boolean;
  startDate?: string;
}

interface FeatureImportanceResult {
  features: Array<{
    name: string;
    importance: number;
    rank: number;
  }>;
  method: string;
  topFeatures: string[];
}

/**
 * Generate comprehensive AI insights from dataset
 */
export const generateInsights = async (
  config: InsightConfig & { signal?: AbortSignal }
): Promise<Insight[]> => {
  try {
    const payload = {
      datasetId: config.datasetId,
      analysisTypes: config.analysisTypes || [
        'summary',
        'anomalies',
        'trends',
        'correlations',
      ],
      columns: config.columns,
      depth: config.depth || 'intermediate',
      includeVisualizations: config.includeVisualizations ?? true,
      includeRecommendations: config.includeRecommendations ?? true,
      includeComparisons: config.includeComparisons ?? false,
      focusArea: config.focusArea,
      industry: config.industry,
      businessContext: config.businessContext,
    };

    const response = await apiPost<Insight[]>(
      INSIGHT_ENDPOINTS.GENERATE_INSIGHTS,
      payload,
      { signal: config.signal }
    );

    console.debug('[InsightService] Insights generated', {
      datasetId: config.datasetId,
      insightCount: response.length,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to generate insights', error);
    throw error;
  }
};

/**
 * Get insights for dataset
 */
export const getDatasetInsights = async (
  datasetId: string,
  page: number = 1,
  limit: number = 20,
  filters?: { type?: string; significance?: string; bookmarked?: boolean }
): Promise<{ insights: Insight[]; pagination: any }> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(filters?.type && { type: filters.type }),
      ...(filters?.significance && { significance: filters.significance }),
      ...(filters?.bookmarked && { bookmarked: filters.bookmarked.toString() }),
    });

    const url = `${INSIGHT_ENDPOINTS.GET_INSIGHTS
      .replace(':datasetId', datasetId)}?${queryParams.toString()}`;

    const response = await apiGet<{ insights: Insight[]; pagination: any }>(url);

    console.debug('[InsightService] Dataset insights fetched', {
      datasetId,
      insightCount: response.insights.length,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to fetch dataset insights', error);
    throw error;
  }
};

/**
 * Get single insight detail
 */
export const getInsightDetail = async (insightId: string): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.GET_INSIGHT_DETAIL.replace(':insightId', insightId);
    const response = await apiGet<Insight>(url);

    console.debug('[InsightService] Insight detail fetched', insightId);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to fetch insight detail', error);
    throw error;
  }
};

/**
 * List all insights with filters
 */
export const listInsights = async (
  page: number = 1,
  limit: number = 20,
  filters?: {
    datasetId?: string;
    type?: string;
    significance?: string;
    bookmarked?: boolean;
    sortBy?: 'createdAt' | 'significance' | 'confidence';
  }
): Promise<{ insights: Insight[]; pagination: any }> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(filters?.datasetId && { datasetId: filters.datasetId }),
      ...(filters?.type && { type: filters.type }),
      ...(filters?.significance && { significance: filters.significance }),
      ...(filters?.bookmarked && { bookmarked: filters.bookmarked.toString() }),
      ...(filters?.sortBy && { sortBy: filters.sortBy }),
    });

    const url = `${INSIGHT_ENDPOINTS.LIST_INSIGHTS}?${queryParams.toString()}`;
    const response = await apiGet<{ insights: Insight[]; pagination: any }>(url);

    console.debug('[InsightService] Insights listed', response.insights.length);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to list insights', error);
    throw error;
  }
};

/**
 * Delete insight
 */
export const deleteInsight = async (insightId: string): Promise<void> => {
  try {
    const url = INSIGHT_ENDPOINTS.DELETE_INSIGHT.replace(':insightId', insightId);
    await apiDelete(url);

    console.debug('[InsightService] Insight deleted', insightId);
  } catch (error) {
    console.error('[InsightService] Failed to delete insight', error);
    throw error;
  }
};

/**
 * Rate insight
 */
export const rateInsight = async (insightId: string, rating: number, notes?: string): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.RATE_INSIGHT.replace(':insightId', insightId);
    const response = await apiPost<Insight>(url, { rating, notes });

    console.debug('[InsightService] Insight rated', { insightId, rating });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to rate insight', error);
    throw error;
  }
};

/**
 * Bookmark insight
 */
export const bookmarkInsight = async (insightId: string, bookmarked: boolean): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.BOOKMARK_INSIGHT.replace(':insightId', insightId);
    const response = await apiPost<Insight>(url, { bookmarked });

    console.debug('[InsightService] Insight bookmarked', { insightId, bookmarked });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to bookmark insight', error);
    throw error;
  }
};

/**
 * Share insight
 */
export const shareInsight = async (
  insightId: string,
  emails: string[],
  permission: 'view' | 'edit' = 'view'
): Promise<void> => {
  try {
    const url = INSIGHT_ENDPOINTS.SHARE_INSIGHT.replace(':insightId', insightId);
    await apiPost(url, { emails, permission });

    console.debug('[InsightService] Insight shared', { insightId, emailCount: emails.length });
  } catch (error) {
    console.error('[InsightService] Failed to share insight', error);
    throw error;
  }
};

/**
 * Get recommendations
 */
export const getRecommendations = async (
  datasetId: string,
  category?: 'optimization' | 'risk_mitigation' | 'opportunity' | 'improvement'
): Promise<Recommendation[]> => {
  try {
    const queryParams = new URLSearchParams({
      ...(category && { category }),
    });

    const url = `${INSIGHT_ENDPOINTS.GET_RECOMMENDATIONS
      .replace(':datasetId', datasetId)}?${queryParams.toString()}`;

    const response = await apiGet<Recommendation[]>(url);

    console.debug('[InsightService] Recommendations fetched', {
      datasetId,
      recommendationCount: response.length,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to fetch recommendations', error);
    throw error;
  }
};

/**
 * Detect anomalies
 */
export const detectAnomalies = async (
  datasetId: string,
  config?: {
    method?: 'isolation_forest' | 'lof' | 'statistical';
    threshold?: number;
    columns?: string[];
  }
): Promise<AnomalyInsight> => {
  try {
    const url = INSIGHT_ENDPOINTS.ANOMALY_DETECTION.replace(':datasetId', datasetId);
    const response = await apiPost<AnomalyInsight>(url, config || {});

    console.debug('[InsightService] Anomalies detected', {
      datasetId,
      anomalyCount: response.anomalies?.length || 0,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to detect anomalies', error);
    throw error;
  }
};

/**
 * Analyze trends
 */
export const analyzeTrends = async (
  datasetId: string,
  config?: {
    timeColumn?: string;
    valueColumns?: string[];
    period?: 'daily' | 'weekly' | 'monthly' | 'yearly';
    forecastPeriods?: number;
  }
): Promise<TrendInsight> => {
  try {
    const url = INSIGHT_ENDPOINTS.TREND_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<TrendInsight>(url, config || {});

    console.debug('[InsightService] Trends analyzed', {
      datasetId,
      direction: response.trend?.direction,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to analyze trends', error);
    throw error;
  }
};

/**
 * Analyze correlations
 */
export const analyzeCorrelations = async (
  datasetId: string,
  columns?: string[]
): Promise<CorrelationInsight> => {
  try {
    const url = INSIGHT_ENDPOINTS.CORRELATION_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<CorrelationInsight>(url, { columns });

    console.debug('[InsightService] Correlations analyzed', {
      datasetId,
      correlationCount: response.correlations?.length || 0,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to analyze correlations', error);
    throw error;
  }
};

/**
 * Predictive analysis
 */
export const performPredictiveAnalysis = async (
  datasetId: string,
  config?: {
    targetColumn?: string;
    featureColumns?: string[];
    modelType?: 'regression' | 'classification' | 'timeseries';
    testSize?: number;
  }
): Promise<PredictionInsight> => {
  try {
    const url = INSIGHT_ENDPOINTS.PREDICTIVE_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<PredictionInsight>(url, config || {});

    console.debug('[InsightService] Predictive analysis completed', {
      datasetId,
      accuracy: response.predictions?.accuracy,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to perform predictive analysis', error);
    throw error;
  }
};

/**
 * Clustering analysis
 */
export const performClusteringAnalysis = async (
  datasetId: string,
  config?: {
    columns?: string[];
    numClusters?: number;
    method?: 'kmeans' | 'hierarchical' | 'dbscan';
  }
): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.CLUSTERING_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<Insight>(url, config || {});

    console.debug('[InsightService] Clustering analysis completed', datasetId);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to perform clustering analysis', error);
    throw error;
  }
};

/**
 * Feature importance analysis
 */
export const analyzeFeatureImportance = async (
  datasetId: string,
  config?: {
    targetColumn?: string;
    method?: 'permutation' | 'shap' | 'tree_based';
  }
): Promise<FeatureImportanceResult> => {
  try {
    const url = INSIGHT_ENDPOINTS.FEATURE_IMPORTANCE.replace(':datasetId', datasetId);
    const response = await apiPost<FeatureImportanceResult>(url, config || {});

    console.debug('[InsightService] Feature importance analyzed', {
      datasetId,
      topFeatureCount: response.topFeatures?.length || 0,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to analyze feature importance', error);
    throw error;
  }
};

/**
 * Time series analysis
 */
export const analyzeTimeSeries = async (
  datasetId: string,
  config?: {
    timeColumn?: string;
    valueColumn?: string;
    frequency?: 'D' | 'W' | 'M' | 'Q' | 'Y';
    decompose?: boolean;
  }
): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.TIME_SERIES_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<Insight>(url, config || {});

    console.debug('[InsightService] Time series analysis completed', datasetId);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to analyze time series', error);
    throw error;
  }
};

/**
 * Business metrics analysis
 */
export const analyzeBusinessMetrics = async (
  datasetId: string,
  config?: {
    metrics?: string[];
    benchmarks?: Record<string, number>;
    industry?: string;
  }
): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.BUSINESS_METRICS.replace(':datasetId', datasetId);
    const response = await apiPost<Insight>(url, config || {});

    console.debug('[InsightService] Business metrics analyzed', datasetId);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to analyze business metrics', error);
    throw error;
  }
};

/**
 * Sentiment analysis (for text data)
 */
export const performSentimentAnalysis = async (
  datasetId: string,
  config?: {
    textColumn?: string;
    language?: string;
  }
): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.SENTIMENT_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<Insight>(url, config || {});

    console.debug('[InsightService] Sentiment analysis completed', datasetId);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to perform sentiment analysis', error);
    throw error;
  }
};

/**
 * NLP analysis
 */
export const performNLPAnalysis = async (
  datasetId: string,
  config?: {
    textColumn?: string;
    tasks?: Array<'tokenization' | 'pos_tagging' | 'ner' | 'topic_modeling'>;
    language?: string;
  }
): Promise<Insight> => {
  try {
    const url = INSIGHT_ENDPOINTS.NLP_ANALYSIS.replace(':datasetId', datasetId);
    const response = await apiPost<Insight>(url, config || {});

    console.debug('[InsightService] NLP analysis completed', datasetId);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to perform NLP analysis', error);
    throw error;
  }
};

/**
 * Compare datasets
 */
export const compareDatasets = async (
  datasetIds: string[],
  config?: {
    metrics?: string[];
    compareMetrics?: boolean;
    compareDistributions?: boolean;
  }
): Promise<Insight> => {
  try {
    const response = await apiPost<Insight>(
      INSIGHT_ENDPOINTS.COMPARE_DATASETS,
      { datasetIds, config }
    );

    console.debug('[InsightService] Datasets compared', {
      datasetCount: datasetIds.length,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to compare datasets', error);
    throw error;
  }
};

/**
 * Export insights
 */
export const exportInsights = async (
  datasetId: string,
  format: 'pdf' | 'html' | 'json' | 'excel' = 'pdf'
): Promise<Blob> => {
  try {
    const url = INSIGHT_ENDPOINTS.EXPORT_INSIGHTS
      .replace(':datasetId', datasetId)
      .replace(':format', format);

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.responseType = 'blob';
      xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('auth_access_token')}`);

      xhr.onload = () => {
        if (xhr.status === 200) {
          console.debug('[InsightService] Insights exported', { datasetId, format });
          resolve(xhr.response);
        } else {
          reject(new Error(`Export failed with status ${xhr.status}`));
        }
      };

      xhr.onerror = () => reject(new Error('Export request failed'));
      xhr.send();
    });
  } catch (error) {
    console.error('[InsightService] Failed to export insights', error);
    throw error;
  }
};

/**
 * Generate comprehensive insight report
 */
export const generateInsightReport = async (
  datasetId: string,
  config?: {
    includeRecommendations?: boolean;
    includeRiskAssessment?: boolean;
    includeOpportunityAssessment?: boolean;
    format?: 'detailed' | 'summary' | 'executive';
  }
): Promise<InsightReport> => {
  try {
    const url = INSIGHT_ENDPOINTS.GENERATE_REPORT.replace(':datasetId', datasetId);
    const response = await apiPost<InsightReport>(url, config || {});

    console.debug('[InsightService] Insight report generated', {
      datasetId,
      insightCount: response.insights?.length || 0,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to generate insight report', error);
    throw error;
  }
};

/**
 * Schedule recurring insights
 */
export const scheduleInsights = async (config: ScheduleConfig): Promise<any> => {
  try {
    const response = await apiPost<any>(INSIGHT_ENDPOINTS.SCHEDULE_INSIGHTS, config);

    console.debug('[InsightService] Insights scheduled', {
      datasetId: config.datasetId,
      frequency: config.frequency,
    });

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to schedule insights', error);
    throw error;
  }
};

/**
 * Get scheduled insight jobs
 */
export const getScheduledInsights = async (): Promise<Array<ScheduleConfig & { id: string }>> => {
  try {
    const response = await apiGet<Array<ScheduleConfig & { id: string }>>(
      INSIGHT_ENDPOINTS.GET_SCHEDULED_INSIGHTS
    );

    console.debug('[InsightService] Scheduled insights fetched', response.length);

    return response;
  } catch (error) {
    console.error('[InsightService] Failed to fetch scheduled insights', error);
    throw error;
  }
};

/**
 * Cancel scheduled insight job
 */
export const cancelScheduledInsight = async (jobId: string): Promise<void> => {
  try {
    const url = INSIGHT_ENDPOINTS.CANCEL_SCHEDULED_INSIGHT.replace(':jobId', jobId);
    await apiPost(url, {});

    console.debug('[InsightService] Scheduled insight cancelled', jobId);
  } catch (error) {
    console.error('[InsightService] Failed to cancel scheduled insight', error);
    throw error;
  }
};

/**
 * Download insights as file
 */
export const downloadInsights = async (
  datasetId: string,
  format: 'pdf' | 'html' | 'json' | 'excel' = 'pdf'
): Promise<void> => {
  try {
    const blob = await exportInsights(datasetId, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `insights_${datasetId}.${format}`;
    link.click();
    URL.revokeObjectURL(url);

    console.debug('[InsightService] Insights downloaded', { datasetId, format });
  } catch (error) {
    console.error('[InsightService] Failed to download insights', error);
    throw error;
  }
};

export default {
  generateInsights,
  getDatasetInsights,
  getInsightDetail,
  listInsights,
  deleteInsight,
  rateInsight,
  bookmarkInsight,
  shareInsight,
  getRecommendations,
  detectAnomalies,
  analyzeTrends,
  analyzeCorrelations,
  performPredictiveAnalysis,
  performClusteringAnalysis,
  analyzeFeatureImportance,
  analyzeTimeSeries,
  analyzeBusinessMetrics,
  performSentimentAnalysis,
  performNLPAnalysis,
  compareDatasets,
  exportInsights,
  generateInsightReport,
  scheduleInsights,
  getScheduledInsights,
  cancelScheduledInsight,
  downloadInsights,
};
