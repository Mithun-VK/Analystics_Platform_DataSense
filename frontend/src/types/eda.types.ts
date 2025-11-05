// src/types/eda.types.ts

/**
 * Exploratory Data Analysis (EDA) Types
 * Comprehensive type definitions for all EDA-related operations and results
 */

// ============================================================================
// Core EDA Types
// ============================================================================
export interface Column {
  name: string;
  type: string;
  nullable: boolean;
  unique?: boolean;
  sampleValues?: any[];
}

/**
 * EDA Analysis status
 */
export type EDAStatus =
  | 'idle'
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * EDA Analysis result
 */
// ✅ ADDED: Additional EDAResult properties
export interface EDAResult {
  id: string;
  datasetId: string;
  jobId: string;
  status: EDAStatus;
  progress: number;
  analysisType: string[];

  // Core dataset info
  totalRows: number;
  totalColumns: number;
  columns: Column[];
  // ✅ ADDED: Required property
  datasetName?: string;

  // Core Analysis Results
  summary?: DatasetSummary;
  statistics?: DescriptiveStatistics;
  correlationMatrix?: CorrelationMatrix;
  // ✅ ADDED: Additional properties
  correlations?: Array<{
    column1: string;
    column2: string;
    coefficient: number;
  }>;
  missingValues?: MissingValues;
  outlierAnalysis?: OutlierAnalysis;
  // ✅ ADDED: Additional properties
  outliers?: Array<{
    rowIndex: number;
    column: string;
    value: number;
    outlierScore: number;
  }>;
  distributionAnalysis?: DistributionAnalysis[];
  // ✅ ADDED: Additional properties
  distributions?: Record<string, any>;
  anomalies?: AnomalyDetection;
  dataQualityReport?: DataQualityReport;
  // ✅ ADDED: Additional properties
  dataQuality?: {
    score: number;
    issues: Array<{
      type: string;
      severity: string;
      description: string;
    }>;
  };

  // Advanced Analysis
  dimensionalityReduction?: DimensionalityReduction;
  clusteringAnalysis?: ClusteringAnalysis;
  classificationAnalysis?: ClassificationAnalysis;
  timeSeriesAnalysis?: TimeSeriesAnalysis;

  // Insights & Recommendations
  insights?: EDAInsight[];
  recommendations?: EDARecommendation[];
  warnings?: EDAWarning[];

  // Metadata
  createdAt: string;
  completedAt?: string;
  duration?: number;
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
  metadata?: Record<string, any>;
}


/**
 * Dataset summary
 */
export interface DatasetSummary {
  rowCount: number;
  columnCount: number;
  totalSize: number;
  memoryUsage: number;
  duplicateRows: number;
  duplicateRowsPercentage: number;
  completeRows: number;
  completeRowsPercentage: number;
  timeRange?: {
    start: string;
    end: string;
  };
  keyFindings: string[];
}

// ============================================================================
// Descriptive Statistics Types
// ============================================================================

/**
 * Descriptive statistics for a numeric column
 */
export interface NumericStatistics {
  fieldName: string;
  count: number;
  nullCount: number;
  nullPercentage: number;
  uniqueCount: number;
  uniquePercentage: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  mode: number;
  std: number; // Standard deviation
  variance: number;
  skewness: number;
  kurtosis: number;
  range: number;
  iqr: number; // Interquartile range
  lowerFence: number;
  upperFence: number;
  percentiles: {
    p1: number;
    p5: number;
    p10: number;
    p25: number;
    p50: number;
    p75: number;
    p90: number;
    p95: number;
    p99: number;
  };
}

/**
 * Descriptive statistics for a categorical column
 */
export interface CategoricalStatistics {
  fieldName: string;
  count: number;
  nullCount: number;
  nullPercentage: number;
  uniqueCount: number;
  uniquePercentage: number;
  topValues: {
    value: string;
    count: number;
    percentage: number;
  }[];
  diversity: number; // Shannon entropy
  dominance: number; // % of top value
}

/**
 * Descriptive statistics for a date/time column
 */
export interface DateTimeStatistics {
  fieldName: string;
  count: number;
  nullCount: number;
  nullPercentage: number;
  uniqueCount: number;
  minDate: string;
  maxDate: string;
  daysBetween: number;
  frequency: 'daily' | 'weekly' | 'monthly' | 'yearly' | 'irregular';
  missingDates: number;
}

/**
 * Descriptive statistics (union type)
 */
export type ColumnStatistic =
  | NumericStatistics
  | CategoricalStatistics
  | DateTimeStatistics;

/**
 * Descriptive statistics collection
 */
export interface DescriptiveStatistics {
  columns: ColumnStatistic[];
  generatedAt: string;
  summary?: {
    numericColumns: number;
    categoricalColumns: number;
    dateColumns: number;
    textColumns: number;
  };
}

// ============================================================================
// Correlation Analysis Types
// ============================================================================

/**
 * Correlation between two variables
 */
export interface Correlation {
  column1: string;
  column2: string;
  coefficient: number; // -1 to 1
  strength: 'very_weak' | 'weak' | 'moderate' | 'strong' | 'very_strong';
  direction: 'positive' | 'negative';
  pvalue?: number;
  sampleSize?: number;
  method: 'pearson' | 'spearman' | 'kendall';
}

/**
 * Correlation matrix
 */
export interface CorrelationMatrix {
  columns: string[];
  correlations: Correlation[];
  matrix: number[][]; // 2D array of correlation coefficients
  strongCorrelations: Correlation[]; // Correlations above threshold
  threshold: number;
  method: string;
  analysisDate: string;
}

// ============================================================================
// Missing Values Types
// ============================================================================

/**
 * Missing values analysis
 */
export interface MissingValues {
  columns: {
    columnName: string;
    missingCount: number;
    missingPercentage: number;
    dataType: string;
    pattern?: 'random' | 'systematic' | 'mixed';
    suggestedHandling:
      | 'drop'
      | 'mean'
      | 'median'
      | 'mode'
      | 'forward_fill'
      | 'backward_fill'
      | 'interpolate';
  }[];
  totalMissing: number;
  totalMissingPercentage: number;
  missingDataMechanism: 'MCAR' | 'MAR' | 'MNAR' | 'unknown';
  notes: string[];
}

// ============================================================================
// Outlier Analysis Types
// ============================================================================

/**
 * Outlier information
 */
export interface Outlier {
  rowIndex: number;
  values: Record<string, any>;
  outlierScore: number;
  outlierType: 'univariate' | 'multivariate' | 'contextual';
  detectionMethod: string;
  severity: 'mild' | 'extreme';
}

/**
 * Outlier analysis result
 */
export interface OutlierAnalysis {
  columns: {
    columnName: string;
    outliersDetected: number;
    outliersPercentage: number;
    method: 'iqr' | 'zscore' | 'isolation_forest' | 'mahalanobis';
    threshold?: number;
    bounds?: {
      lower: number;
      upper: number;
    };
  }[];
  totalOutliers: number;
  totalOutliersPercentage: number;
  outlierRows: Outlier[];
  recommendations: string[];
}

// ============================================================================
// Distribution Analysis Types
// ============================================================================

/**
 * Distribution analysis for a column
 */
export interface DistributionAnalysis {
  columnName: string;
  dataType: string;
  distributionType?:
    | 'normal'
    | 'log_normal'
    | 'exponential'
    | 'uniform'
    | 'poisson'
    | 'binomial'
    | 'unknown';
  normality: {
    shapiroWilkTest?: number;
    kurtosisTest?: number;
    isNormal: boolean;
  };
  histogram?: {
    bins: number;
    frequencies: number[];
    binEdges: number[];
  };
  kde?: {
    values: number[];
    density: number[];
  };
  skewness: number;
  kurtosis: number;
  visualizationSuggestion: 'histogram' | 'kde' | 'boxplot' | 'violin' | 'qqplot';
}

// ============================================================================
// Anomaly Detection Types
// ============================================================================

/**
 * Anomaly detection result
 */
export interface AnomalyDetection {
  method: 'isolation_forest' | 'lof' | 'statistical' | 'autoencoder';
  anomalies: {
    rowIndex: number;
    anomalyScore: number;
    features: Record<string, any>;
    anomalyType: string;
  }[];
  totalAnomalies: number;
  anomalyPercentage: number;
  threshold: number;
  parameters?: Record<string, any>;
}

// ============================================================================
// Data Quality Report Types
// ============================================================================

/**
 * Data quality issue
 */
export interface DataQualityIssue {
  id: string;
  column: string;
  type: 'missing' | 'duplicate' | 'outlier' | 'inconsistent' | 'invalid' | 'other';
  severity: 'low' | 'medium' | 'high' | 'critical';
  count: number;
  percentage: number;
  description: string;
  examples?: any[];
  suggestion?: string;
}

/**
 * Data quality report
 */
export interface DataQualityReport {
  overallScore: number; // 0-100
  dimensions: {
    completeness: number; // % non-null
    uniqueness: number; // % unique
    consistency: number; // % consistent format
    validity: number; // % valid values
    accuracy: number; // % accurate
    timeliness: number; // % timely
  };
  issues: DataQualityIssue[];
  criticalIssues: DataQualityIssue[];
  recommendations: string[];
  lastAssessmentDate: string;
}

// ============================================================================
// Advanced Analysis Types
// ============================================================================

/**
 * Dimensionality reduction result
 */
export interface DimensionalityReduction {
  method: 'pca' | 'tsne' | 'umap' | 'auto_encoder';
  originalDimensions: number;
  reducedDimensions: number;
  varianceExplained: number; // percentage
  components?: {
    componentNumber: number;
    variance: number;
    cumulativeVariance: number;
  }[];
  embeddingCoordinates?: number[][];
}

/**
 * Clustering analysis result
 */
export interface ClusteringAnalysis {
  method: 'kmeans' | 'hierarchical' | 'dbscan' | 'gaussian_mixture';
  numClusters: number;
  silhouetteScore: number;
  daviesBouldinIndex: number;
  inertia?: number;
  clusters: {
    clusterId: number;
    size: number;
    percentage: number;
    centroid?: Record<string, number>;
    characteristics?: string[];
  }[];
}

/**
 * Classification analysis result
 */
export interface ClassificationAnalysis {
  targetColumn: string;
  classDistribution: {
    className: string;
    count: number;
    percentage: number;
  }[];
  imbalanceRatio?: number;
  multiclass: boolean;
  recommendations?: string[];
}

/**
 * Time series analysis result
 */
export interface TimeSeriesAnalysis {
  timeColumn: string;
  valueColumn: string;
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly' | 'irregular';
  trend?: {
    direction: 'upward' | 'downward' | 'stable' | 'cyclical';
    slope?: number;
    changePercentage?: number;
  };
  seasonality?: {
    detected: boolean;
    period?: number;
    strength?: number;
  };
  decomposition?: {
    trend: number[];
    seasonal: number[];
    residual: number[];
  };
  forecast?: {
    values: number[];
    confidenceInterval: {
      lower: number[];
      upper: number[];
    };
  };
  stationarity?: {
    isStationary: boolean;
    adfullerTest?: number;
  };
}

// ============================================================================
// EDA Insights & Recommendations Types
// ============================================================================

/**
 * EDA insight
 */
// src/types/eda.types.ts - UPDATE EDAInsight

export interface EDAInsight {
  id: string;
  type: string;
  title: string;
  description: string;
  finding: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high';
  evidence?: Array<{ metric: string; value: number | string }>;
  relatedColumns?: string[];
  actionable?: boolean;
  // ✅ ADDED: Missing properties
  recommendation?: string;
  createdAt?: string;
  visualization?: {
    type: string;
    config?: Record<string, any>;
  };
}


/**
 * EDA recommendation
 */
export interface EDARecommendation {
  id: string;
  category:
    | 'data_cleaning'
    | 'feature_engineering'
    | 'analysis'
    | 'modeling'
    | 'other';
  title: string;
  description: string;
  reasoning: string;
  difficulty: 'easy' | 'medium' | 'hard';
  impact: 'low' | 'medium' | 'high';
  relatedColumns?: string[];
  implementation?: {
    steps: string[];
    tools?: string[];
    estimatedTime?: number;
  };
}

/**
 * EDA warning
 */
export interface EDAWarning {
  id: string;
  type: string;
  severity: 'info' | 'warning' | 'error';
  message: string;
  affectedColumns?: string[];
  suggestion?: string;
  canBeCorrected: boolean;
}

// ============================================================================
// Request/Response Types
// ============================================================================

/**
 * EDA analysis request
 */
export interface EDAAnalysisRequest {
  datasetId: string;
  analysisTypes?: string[];
  columns?: string[];
  depth?: 'basic' | 'intermediate' | 'advanced';
  includeVisualizations?: boolean;
  includeRecommendations?: boolean;
  excludeColumns?: string[];
  customParameters?: Record<string, any>;
}

/**
 * EDA analysis response
 */
export interface EDAAnalysisResponse {
  jobId: string;
  datasetId: string;
  status: EDAStatus;
  progress: number;
  message: string;
  result?: EDAResult;
  error?: {
    code: string;
    message: string;
  };
  estimatedTimeRemaining?: number;
}

/**
 * EDA comparison request
 */
export interface EDAComparisonRequest {
  datasets: string[];
  metrics?: string[];
  compareDistributions?: boolean;
  compareQuality?: boolean;
}

/**
 * EDA comparison result
 */
export interface EDAComparisonResult {
  datasets: Array<{
    datasetId: string;
    statistics: DescriptiveStatistics;
    quality: DataQualityReport;
  }>;
  differences: {
    column: string;
    differences: Record<string, any>;
  }[];
  recommendations: string[];
}

// ============================================================================
// Export & Report Types
// ============================================================================

/**
 * EDA report
 */
export interface EDAReport {
  id: string;
  jobId: string;
  datasetId: string;
  title: string;
  generatedAt: string;
  generatedBy: string;
  sections: ReportSection[];
  summary: string;
  keyFindings: string[];
  recommendations: string[];
  appendix?: {
    methodologyNotes: string;
    limitations: string[];
    references: string[];
  };
}

/**
 * Report section
 */
export interface ReportSection {
  id: string;
  title: string;
  content: string;
  visualizations?: {
    type: string;
    data: any;
    caption: string;
  }[];
  tables?: {
    columns: string[];
    data: Record<string, any>[];
    caption: string;
  }[];
}

/**
 * Export options
 */
export interface EDAExportOptions {
  format: 'pdf' | 'html' | 'json' | 'markdown' | 'powerpoint';
  includeData: boolean;
  includePlots: boolean;
  detail: 'summary' | 'standard' | 'comprehensive';
  theme?: 'light' | 'dark';
}

// ============================================================================
// Cache & History Types
// ============================================================================

/**
 * EDA analysis cache entry
 */
export interface EDAAnalysisCache {
  jobId: string;
  datasetId: string;
  result: EDAResult;
  cachedAt: string;
  expiresAt: string;
  hitCount: number;
}

/**
 * EDA analysis history
 */
export interface EDAAnalysisHistory {
  jobId: string;
  datasetId: string;
  status: EDAStatus;
  startedAt: string;
  completedAt?: string;
  duration?: number;
  analysisTypes: string[];
  resultSummary?: {
    rowsAnalyzed: number;
    columnsAnalyzed: number;
    issuesFound: number;
    insightsGenerated: number;
  };
}

// ============================================================================
// Enums
// ============================================================================

/**
 * EDA status enumeration
 */
export enum EDAStatusEnum {
  Idle = 'idle',
  Pending = 'pending',
  Processing = 'processing',
  Completed = 'completed',
  Failed = 'failed',
  Cancelled = 'cancelled',
}

/**
 * Analysis type enumeration
 */
export enum AnalysisTypeEnum {
  Summary = 'summary',
  Statistics = 'statistics',
  Correlation = 'correlation',
  MissingValues = 'missing_values',
  Outliers = 'outliers',
  Distribution = 'distribution',
  Anomalies = 'anomalies',
  Quality = 'quality',
  TimeSeries = 'time_series',
  Clustering = 'clustering',
}

/**
 * Analysis depth enumeration
 */
export enum AnalysisDepthEnum {
  Basic = 'basic',
  Intermediate = 'intermediate',
  Advanced = 'advanced',
}

/**
 * Distribution type enumeration
 */
export enum DistributionTypeEnum {
  Normal = 'normal',
  LogNormal = 'log_normal',
  Exponential = 'exponential',
  Uniform = 'uniform',
  Poisson = 'poisson',
  Binomial = 'binomial',
  Unknown = 'unknown',
}

/**
 * Correlation strength enumeration
 */
export enum CorrelationStrengthEnum {
  VeryWeak = 'very_weak',
  Weak = 'weak',
  Moderate = 'moderate',
  Strong = 'strong',
  VeryStrong = 'very_strong',
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
 * Recommendation category enumeration
 */
export enum RecommendationCategoryEnum {
  DataCleaning = 'data_cleaning',
  FeatureEngineering = 'feature_engineering',
  Analysis = 'analysis',
  Modeling = 'modeling',
  Other = 'other',
}

// ============================================================================
// Form Types
// ============================================================================

/**
 * EDA analysis form values
 */
export interface EDAAnalysisFormValues {
  analysisTypes: string[];
  depth: 'basic' | 'intermediate' | 'advanced';
  includeVisualizations: boolean;
  includeRecommendations: boolean;
  columns: string[];
  excludeColumns: string[];
}

/**
 * EDA export form values
 */
export interface EDAExportFormValues {
  format: 'pdf' | 'html' | 'json' | 'markdown' | 'powerpoint';
  includeData: boolean;
  includePlots: boolean;
  detail: 'summary' | 'standard' | 'comprehensive';
  theme: 'light' | 'dark';
}

// ============================================================================
// Type Guards & Predicates
// ============================================================================

/**
 * Type guard for EDAResult
 */
export function isEDAResult(obj: any): obj is EDAResult {
  return (
    obj &&
    typeof obj === 'object' &&
    typeof obj.id === 'string' &&
    typeof obj.datasetId === 'string' &&
    typeof obj.jobId === 'string'
  );
}

/**
 * Check if analysis is complete
 */
export function isAnalysisComplete(result: EDAResult): boolean {
  return (
    result.status === 'completed' && result.completedAt !== undefined
  );
}

/**
 * Check if analysis has errors
 */
export function hasAnalysisErrors(result: EDAResult): boolean {
  return result.status === 'failed' && result.error !== undefined;
}

/**
 * Get analysis progress percentage
 */
export function getAnalysisProgress(result: EDAResult): number {
  return Math.max(0, Math.min(100, result.progress));
}

/**
 * Check if column has quality issues
 */
export function hasColumnQualityIssues(
  report: DataQualityReport | undefined,
  columnName: string
): boolean {
  if (!report) return false;
  return report.issues.some(
    (issue) =>
      issue.column === columnName && issue.severity !== 'low'
  );
}

/**
 * Get correlation strength label
 */
export function getCorrelationStrengthLabel(
  coefficient: number
): CorrelationStrengthEnum {
  const absCoeff = Math.abs(coefficient);
  if (absCoeff >= 0.9) return CorrelationStrengthEnum.VeryStrong;
  if (absCoeff >= 0.7) return CorrelationStrengthEnum.Strong;
  if (absCoeff >= 0.5) return CorrelationStrengthEnum.Moderate;
  if (absCoeff >= 0.3) return CorrelationStrengthEnum.Weak;
  return CorrelationStrengthEnum.VeryWeak;
}

/**
 * Check if column has missing values
 */
export function hasSignificantMissingValues(
  stats: NumericStatistics | CategoricalStatistics | DateTimeStatistics,
  threshold: number = 0.05
): boolean {
  return stats.nullPercentage > threshold * 100;
}

/**
 * Check if column is numeric
 */
export function isNumericStatistic(
  stat: ColumnStatistic
): stat is NumericStatistics {
  return 'mean' in stat;
}

/**
 * Check if column is categorical
 */
export function isCategoricalStatistic(
  stat: ColumnStatistic
): stat is CategoricalStatistics {
  return 'topValues' in stat;
}

/**
 * Check if column is datetime
 */
export function isDateTimeStatistic(
  stat: ColumnStatistic
): stat is DateTimeStatistics {
  return 'minDate' in stat;
}

/**
 * Get quality score color
 */
export function getQualityScoreColor(score: number): string {
  if (score >= 90) return 'text-green-600 bg-green-100';
  if (score >= 70) return 'text-yellow-600 bg-yellow-100';
  if (score >= 50) return 'text-orange-600 bg-orange-100';
  return 'text-red-600 bg-red-100';
}

/**
 * Format correlation coefficient
 */
export function formatCorrelation(coefficient: number): string {
  return coefficient.toFixed(3);
}

/**
 * Format percentage
 */
export function formatPercentage(value: number, decimals: number = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Get recommendation difficulty color
 */
export function getRecommendationDifficultyColor(
  difficulty: 'easy' | 'medium' | 'hard'
): string {
  switch (difficulty) {
    case 'easy':
      return 'text-green-600';
    case 'medium':
      return 'text-yellow-600';
    case 'hard':
      return 'text-red-600';
    default:
      return 'text-gray-600';
  }
}

/**
 * Get severity color
 */
export function getSeverityColor(
  severity: 'low' | 'medium' | 'high' | 'critical'
): string {
  switch (severity) {
    case 'low':
      return 'text-blue-600 bg-blue-100';
    case 'medium':
      return 'text-yellow-600 bg-yellow-100';
    case 'high':
      return 'text-orange-600 bg-orange-100';
    case 'critical':
      return 'text-red-600 bg-red-100';
    default:
      return 'text-gray-600 bg-gray-100';
  }
}

/**
 * Get analysis status color
 */
export function getAnalysisStatusColor(status: EDAStatus): string {
  switch (status) {
    case 'idle':
      return 'text-gray-600';
    case 'pending':
      return 'text-blue-600';
    case 'processing':
      return 'text-blue-600';
    case 'completed':
      return 'text-green-600';
    case 'failed':
      return 'text-red-600';
    case 'cancelled':
      return 'text-gray-600';
    default:
      return 'text-gray-600';
  }
}
