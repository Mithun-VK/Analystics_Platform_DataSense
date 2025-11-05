// src/types/components.types.ts - COMPLETE FINAL VERSION

import type {
  EDAResult,
} from './eda.types';

// ============================================================================
// EDA Report Component
// ============================================================================

export interface EDAReportProps {
  edaResult: EDAResult;
}

// ============================================================================
// Statistics Table Component
// ============================================================================

/**
 * ✅ FIXED: All numeric fields optional
 */
export interface Statistic {
  columnName: string;
  column: string;
  type: string;
  dataType: 'numeric' | 'categorical' | 'datetime' | 'text';
  count: number;
  nullCount: number;
  nullPercentage: number;
  uniqueCount: number;
  uniquePercentage?: number;
  // ✅ ALL numeric fields optional
  mean?: number;
  median?: number;
  mode?: number;
  min?: number;
  max?: number;
  std?: number;
  variance?: number;
  skewness?: number;
  kurtosis?: number;
  range?: number;
  iqr?: number;
  q25?: number;
  q75?: number;
  p1?: number;
  p5?: number;
  p10?: number;
  p25?: number;
  p50?: number;
  p75?: number;
  p90?: number;
  p95?: number;
  p99?: number;
  topValues?: Array<{ value: string; count: number; percentage: number }>;
  diversity?: number;
  dominance?: number;
  minDate?: string;
  maxDate?: string;
  daysBetween?: number;
  frequency?: 'daily' | 'weekly' | 'monthly' | 'yearly' | 'irregular';
  missingDates?: number;
}

export interface StatisticsTableProps {
  statistics: Statistic[];
  viewMode?: 'grid' | 'list';
  sortBy?: 'name' | 'type' | 'nulls';
  sortOrder?: 'asc' | 'desc';
  searchQuery?: string;
}

// ============================================================================
// Correlation Matrix Component
// ============================================================================

/**
 * ✅ FIXED: Columns is optional with default
 */
export interface CorrelationMatrixProps {
  matrix: number[][];
  height?: number;
  width?: number;
  showValues?: boolean;
  colorScheme?: 'default' | 'cool' | 'warm' | 'diverging';
  threshold?: number;
  columns?: string[]; // ✅ Already optional - keep as is
  onCellClick?: (col1: string, col2: string) => void;
}

// ============================================================================
// Insight Card Component - FIXED
// ============================================================================

/**
 * ✅ FIXED: All properties match EDAInsight exactly
 */
export type InsightType = string; // ✅ Changed to accept any string

export interface Insight {
  id: string;
  type: InsightType; // ✅ Now accepts any string
  title: string;
  description: string;
  finding: string;
  confidence: number;
  priority: 'low' | 'medium' | 'high';
  evidence?: Array<{ metric: string; value: number | string }>;
  relatedColumns?: string[];
  actionable?: boolean;
  recommendation?: string;
  createdAt?: string;
  visualization?: {
    type: string;
    config?: Record<string, any>;
  };
  severity?: 'critical' | 'high' | 'medium' | 'low' | 'info';
  category?: 'performance' | 'quality' | 'pattern' | 'opportunity' | 'risk';
  impact?: string;
}

export interface InsightCardProps {
  insight: Insight;
  onExplore?: (insightId: string) => void;
}

// ============================================================================
// Data Quality Component
// ============================================================================

export interface DataQualityProps {
  score: number;
  dimensions: {
    completeness: number;
    uniqueness: number;
    consistency: number;
    validity: number;
    accuracy: number;
    timeliness: number;
  };
  issues: Array<{
    id: string;
    column: string;
    type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    count: number;
    percentage: number;
    description: string;
  }>;
}

// ============================================================================
// Distribution Chart Component
// ============================================================================

export interface DistributionChartProps {
  columnName: string;
  data: {
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
  };
  height?: number;
  type?: 'histogram' | 'kde' | 'both';
}

// ============================================================================
// Outlier Chart Component
// ============================================================================

export interface OutlierChartProps {
  columnName: string;
  data: {
    bounds?: { lower: number; upper: number };
    outliersDetected: number;
    method: string;
  };
  values: number[];
  height?: number;
}
