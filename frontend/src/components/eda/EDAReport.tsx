// src/components/eda/EDAReport.tsx - FINAL COMPLETE VERSION WITH FIXES

import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  Share2,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  BarChart3,
  PieChart,
  Activity,
  Database,
  FileText,
  ChevronDown,
  ChevronUp,
  Info,
  Eye,
  Calendar,
  Clock,
} from 'lucide-react';
import StatisticsTable from './StatisticsTable';
import CorrelationMatrix from './CorrelationMatrix';
import { formatDistanceToNow } from 'date-fns';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * ✅ FIXED: Added 'cancelled' and 'pending' to status union type
 */
export interface EDAResult {
  id: string;
  datasetId: string;
  datasetName?: string;
  createdAt: string;
  status: 'completed' | 'processing' | 'failed' | 'pending' | 'cancelled';
  summary: {
    totalRows: number;
    totalColumns: number;
    missingValues: number;
    duplicateRows: number;
    memoryUsage: number;
  };
  statistics: {
    columnName: string;
    count: number;
    mean?: number;
    std?: number;
    min?: number;
    q25?: number;
    median?: number;
    q75?: number;
    max?: number;
    dataType: string;
  }[];
  correlations: {
    matrix: number[][];
    columns?: string[];
  };
  distributions: {
    [columnName: string]: {
      values: number[];
      frequencies: number[];
    };
  };
  insights: {
    type: 'info' | 'warning' | 'success' | 'error';
    title: string;
    message: string;
    severity: 'high' | 'medium' | 'low';
  }[];
  outliers: {
    columnName: string;
    count: number;
    percentage: number;
  }[];
  dataQuality: {
    completeness: number;
    consistency: number;
    validity: number;
    overall: number;
  };
}

export interface EDAReportProps {
  edaResult: EDAResult;
  onRerun?: () => void;
  onExport?: () => void;
}

// ============================================================================
// Component
// ============================================================================

/**
 * EDAReport - Comprehensive exploratory data analysis results display
 * Features: Summary statistics, correlations, distributions, insights, data quality
 * Provides interactive visualizations and actionable recommendations
 */
const EDAReport: React.FC<EDAReportProps> = ({
  edaResult,
  onRerun,
  onExport,
}) => {
  const navigate = useNavigate();

  // ============================================================================
  // State Management
  // ============================================================================

  const [activeSection, setActiveSection] = useState<string>('overview');
  const [expandedInsights, setExpandedInsights] = useState<Set<number>>(
    new Set([0])
  );
  const [showAllInsights, setShowAllInsights] = useState(false);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  /**
   * Toggle insight expansion
   */
  const toggleInsight = (index: number) => {
    const newExpanded = new Set(expandedInsights);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedInsights(newExpanded);
  };

  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * Get insight icon based on type
   */
  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-red-600" />;
      default:
        return <Info className="w-5 h-5 text-blue-600" />;
    }
  };

  /**
   * Get insight color classes based on type
   */
  const getInsightColorClasses = (type: string) => {
    switch (type) {
      case 'success':
        return 'border-l-green-500 border-green-200 bg-green-50';
      case 'warning':
        return 'border-l-yellow-500 border-yellow-200 bg-yellow-50';
      case 'error':
        return 'border-l-red-500 border-red-200 bg-red-50';
      default:
        return 'border-l-blue-500 border-blue-200 bg-blue-50';
    }
  };

  /**
   * Format file size
   */
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  /**
   * Get data quality score color
   */
  const getQualityColor = (score: number) => {
    if (score >= 90) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  /**
   * Get data quality grade
   */
  const getQualityGrade = (score: number) => {
    if (score >= 90) return 'Excellent';
    if (score >= 80) return 'Good';
    if (score >= 70) return 'Fair';
    if (score >= 60) return 'Poor';
    return 'Critical';
  };

  /**
   * Calculate trend direction with icon
   */
  const getTrendIndicator = (
    current: number,
    previous: number
  ): { icon: React.ReactNode; color: string; label: string } => {
    const change = ((current - previous) / previous) * 100;
    if (change > 0) {
      return {
        icon: <TrendingUp className="w-4 h-4" />,
        color: 'text-green-600',
        label: `+${change.toFixed(1)}%`,
      };
    } else if (change < 0) {
      return {
        icon: <TrendingDown className="w-4 h-4" />,
        color: 'text-red-600',
        label: `${change.toFixed(1)}%`,
      };
    }
    return {
      icon: <Activity className="w-4 h-4" />,
      color: 'text-gray-600',
      label: 'No change',
    };
  };

  // ============================================================================
  // Memoized Computations
  // ============================================================================

  /**
   * Sort insights by severity
   */
  const sortedInsights = useMemo(() => {
    const severityOrder = { high: 0, medium: 1, low: 2 };
    return [...edaResult.insights].sort(
      (a, b) => severityOrder[a.severity] - severityOrder[b.severity]
    );
  }, [edaResult.insights]);

  /**
   * Get displayed insights (limited or all)
   */
  const displayedInsights = showAllInsights
    ? sortedInsights
    : sortedInsights.slice(0, 5);

  /**
   * Calculate insight statistics
   */
  const insightStats = useMemo(() => {
    return {
      total: edaResult.insights.length,
      high: edaResult.insights.filter((i) => i.severity === 'high').length,
      medium: edaResult.insights.filter((i) => i.severity === 'medium').length,
      low: edaResult.insights.filter((i) => i.severity === 'low').length,
      success: edaResult.insights.filter((i) => i.type === 'success').length,
      warning: edaResult.insights.filter((i) => i.type === 'warning').length,
    };
  }, [edaResult.insights]);

  /**
   * ✅ FIXED: Get correlation columns with safe fallback
   */
  const correlationColumns = useMemo(() => {
    if (!edaResult.correlations.columns || edaResult.correlations.columns.length === 0) {
      return undefined;
    }
    return edaResult.correlations.columns;
  }, [edaResult.correlations.columns]);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className="space-y-6">
      {/* ====================================================================
          Header Section
          ==================================================================== */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate(`/datasets/${edaResult.datasetId}`)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              EDA Report: {edaResult.datasetName || 'Dataset Analysis'}
            </h1>
            <div className="flex items-center space-x-4 mt-1 flex-wrap">
              <span className="text-sm text-gray-600 flex items-center space-x-1">
                <Calendar className="w-4 h-4" />
                <span>
                  {formatDistanceToNow(new Date(edaResult.createdAt), {
                    addSuffix: true,
                  })}
                </span>
              </span>
              <span
                className={`inline-flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${
                  edaResult.status === 'completed'
                    ? 'bg-green-100 text-green-700'
                    : edaResult.status === 'processing' || edaResult.status === 'pending'
                    ? 'bg-yellow-100 text-yellow-700'
                    : edaResult.status === 'cancelled'
                    ? 'bg-gray-100 text-gray-700'
                    : 'bg-red-100 text-red-700'
                }`}
              >
                <span className="capitalize">{edaResult.status}</span>
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {onRerun && (
            <button
              onClick={onRerun}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span className="text-sm font-medium">Rerun</span>
            </button>
          )}
          {onExport && (
            <button
              onClick={onExport}
              className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <Download className="w-4 h-4" />
              <span className="text-sm font-medium">Export</span>
            </button>
          )}
          <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            <Share2 className="w-4 h-4" />
            <span className="text-sm font-medium">Share</span>
          </button>
        </div>
      </div>

      {/* ====================================================================
          Data Quality Score Section
          ==================================================================== */}
      <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border border-blue-200 p-6">
        <div className="flex items-center justify-between gap-6 flex-wrap">
          <div className="flex-1 min-w-[300px]">
            <h2 className="text-lg font-semibold text-gray-900 mb-2">
              Overall Data Quality
            </h2>
            <p className="text-sm text-gray-600 mb-4">
              Comprehensive assessment of your dataset's quality and readiness
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-gray-600 mb-1">Completeness</p>
                <p className="text-2xl font-bold text-gray-900">
                  {edaResult.dataQuality.completeness.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 mb-1">Consistency</p>
                <p className="text-2xl font-bold text-gray-900">
                  {edaResult.dataQuality.consistency.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 mb-1">Validity</p>
                <p className="text-2xl font-bold text-gray-900">
                  {edaResult.dataQuality.validity.toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-600 mb-1">Overall Score</p>
                <p
                  className={`text-2xl font-bold ${getQualityColor(
                    edaResult.dataQuality.overall
                  )}`}
                >
                  {edaResult.dataQuality.overall.toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          <div className="flex-shrink-0">
            <div className="relative w-32 h-32">
              <svg className="w-32 h-32 transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="#E5E7EB"
                  strokeWidth="8"
                  fill="none"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke={
                    edaResult.dataQuality.overall >= 90
                      ? '#10B981'
                      : edaResult.dataQuality.overall >= 70
                      ? '#F59E0B'
                      : '#EF4444'
                  }
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${
                    (edaResult.dataQuality.overall / 100) * 352
                  } 352`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span
                  className={`text-3xl font-bold ${getQualityColor(
                    edaResult.dataQuality.overall
                  )}`}
                >
                  {edaResult.dataQuality.overall.toFixed(0)}
                </span>
                <span className="text-xs text-gray-600 mt-1">
                  {getQualityGrade(edaResult.dataQuality.overall)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ====================================================================
          Key Metrics Grid
          ==================================================================== */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <Database className="w-5 h-5 text-blue-600" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-1">Total Rows</p>
          <p className="text-2xl font-bold text-gray-900">
            {edaResult.summary.totalRows.toLocaleString()}
          </p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-green-600" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-1">Total Columns</p>
          <p className="text-2xl font-bold text-gray-900">
            {edaResult.summary.totalColumns}
          </p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-yellow-600" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-1">Missing Values</p>
          <p className="text-2xl font-bold text-gray-900">
            {edaResult.summary.missingValues.toLocaleString()}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {(
              (edaResult.summary.missingValues /
                (edaResult.summary.totalRows * edaResult.summary.totalColumns)) *
              100
            ).toFixed(2)}
            % of total
          </p>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <Activity className="w-5 h-5 text-purple-600" />
            </div>
          </div>
          <p className="text-sm text-gray-600 mb-1">Duplicate Rows</p>
          <p className="text-2xl font-bold text-gray-900">
            {edaResult.summary.duplicateRows.toLocaleString()}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {(
              (edaResult.summary.duplicateRows / edaResult.summary.totalRows) *
              100
            ).toFixed(2)}
            % of total
          </p>
        </div>
      </div>

      {/* ====================================================================
          Insight Statistics Summary
          ==================================================================== */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="w-4 h-4 text-red-600" />
            <span className="text-sm font-medium text-gray-900">
              High Severity
            </span>
          </div>
          <p className="text-2xl font-bold text-red-600">{insightStats.high}</p>
        </div>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="w-4 h-4 text-yellow-600" />
            <span className="text-sm font-medium text-gray-900">
              Medium Severity
            </span>
          </div>
          <p className="text-2xl font-bold text-yellow-600">
            {insightStats.medium}
          </p>
        </div>
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-gray-900">
              Low Severity
            </span>
          </div>
          <p className="text-2xl font-bold text-green-600">{insightStats.low}</p>
        </div>
      </div>

      {/* ====================================================================
          Key Insights Section
          ==================================================================== */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="border-b border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Activity className="w-5 h-5 text-blue-600" />
              <h2 className="text-lg font-semibold text-gray-900">
                Key Insights
              </h2>
            </div>
            <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
              {edaResult.insights.length} insights
            </span>
          </div>
        </div>

        <div className="p-6 space-y-3">
          {displayedInsights.map((insight, index) => {
            const isExpanded = expandedInsights.has(index);
            return (
              <div
                key={index}
                className={`border-l-4 rounded-lg p-4 transition-all cursor-pointer hover:bg-gray-50 ${getInsightColorClasses(
                  insight.type
                )}`}
                onClick={() => toggleInsight(index)}
              >
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getInsightIcon(insight.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2 gap-2">
                      <h3 className="text-sm font-semibold text-gray-900">
                        {insight.title}
                      </h3>
                      <div className="flex items-center space-x-2">
                        <span
                          className={`text-xs font-medium px-2 py-1 rounded ${
                            insight.severity === 'high'
                              ? 'bg-red-200 text-red-700'
                              : insight.severity === 'medium'
                              ? 'bg-yellow-200 text-yellow-700'
                              : 'bg-gray-200 text-gray-700'
                          }`}
                        >
                          {insight.severity}
                        </span>
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4 text-gray-500" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-gray-500" />
                        )}
                      </div>
                    </div>
                    {isExpanded && (
                      <p className="text-sm text-gray-700 leading-relaxed">
                        {insight.message}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}

          {edaResult.insights.length > 5 && (
            <button
              onClick={() => setShowAllInsights(!showAllInsights)}
              className="w-full py-2 text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors"
            >
              {showAllInsights
                ? 'Show Less'
                : `Show ${edaResult.insights.length - 5} More Insights`}
            </button>
          )}
        </div>
      </div>

      {/* ====================================================================
          Tabbed Content Section
          ==================================================================== */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="border-b border-gray-200">
          <nav className="flex overflow-x-auto space-x-8 px-6" aria-label="Tabs">
            {[
              { id: 'overview', label: 'Overview', icon: Eye },
              { id: 'statistics', label: 'Statistics', icon: BarChart3 },
              { id: 'correlations', label: 'Correlations', icon: Activity },
              { id: 'distribution', label: 'Distribution', icon: PieChart },
              { id: 'outliers', label: 'Outliers', icon: AlertTriangle },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveSection(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors whitespace-nowrap ${
                    activeSection === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {/* Overview Tab */}
          {activeSection === 'overview' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Dataset Overview
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-3">
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm text-gray-600">Total Records</span>
                      <span className="text-sm font-medium text-gray-900">
                        {edaResult.summary.totalRows.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm text-gray-600">Total Features</span>
                      <span className="text-sm font-medium text-gray-900">
                        {edaResult.summary.totalColumns}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm text-gray-600">Memory Usage</span>
                      <span className="text-sm font-medium text-gray-900">
                        {formatFileSize(edaResult.summary.memoryUsage)}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm text-gray-600">Missing Values</span>
                      <span className="text-sm font-medium text-gray-900">
                        {edaResult.summary.missingValues.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm text-gray-600">Duplicate Rows</span>
                      <span className="text-sm font-medium text-gray-900">
                        {edaResult.summary.duplicateRows.toLocaleString()}
                      </span>
                    </div>
                    <div className="flex justify-between py-2 border-b border-gray-100">
                      <span className="text-sm text-gray-600">Analysis Date</span>
                      <span className="text-sm font-medium text-gray-900">
                        {new Date(edaResult.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Statistics Tab */}
          {activeSection === 'statistics' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Descriptive Statistics
              </h3>
              <StatisticsTable statistics={edaResult.statistics} />
            </div>
          )}

          {/* Correlations Tab */}
          {activeSection === 'correlations' && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Feature Correlations
              </h3>
              {/* ✅ FIXED: Pass columns safely with fallback */}
              <CorrelationMatrix
                matrix={edaResult.correlations.matrix}
                columns={correlationColumns || []}
              />
            </div>
          )}

          {/* Distribution Tab */}
          {activeSection === 'distribution' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Feature Distributions
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Object.entries(edaResult.distributions).map(
                  ([columnName, data]) => (
                    <div
                      key={columnName}
                      className="border border-gray-200 rounded-lg p-4"
                    >
                      <div className="flex items-center space-x-2 mb-3">
                        <PieChart className="w-4 h-4 text-blue-600" />
                        <h4 className="font-medium text-gray-900">
                          {columnName}
                        </h4>
                      </div>
                      <div className="bg-gray-50 rounded p-3">
                        <p className="text-xs text-gray-600 mb-2">
                          {data.values.length} unique values
                        </p>
                        <div className="space-y-1">
                          {data.values.slice(0, 5).map((_, idx) => (
                            <div key={idx} className="flex items-center space-x-2">
                              <div className="flex-1 bg-blue-200 rounded h-4">
                                <div
                                  className="bg-blue-600 h-full rounded"
                                  style={{
                                    width: `${
                                      (data.frequencies[idx] /
                                        Math.max(...data.frequencies)) *
                                      100
                                    }%`,
                                  }}
                                />
                              </div>
                              <span className="text-xs text-gray-600 w-12 text-right">
                                {data.frequencies[idx]}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          {/* Outliers Tab */}
          {activeSection === 'outliers' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Outlier Detection
              </h3>
              {edaResult.outliers.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 px-4 font-semibold text-gray-900">
                          Column Name
                        </th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-900">
                          Outlier Count
                        </th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-900">
                          Percentage
                        </th>
                        <th className="text-left py-3 px-4 font-semibold text-gray-900">
                          Severity
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {edaResult.outliers.map((outlier, index) => (
                        <tr key={index} className="border-b border-gray-100">
                          <td className="py-3 px-4 font-medium text-gray-900">
                            {outlier.columnName}
                          </td>
                          <td className="py-3 px-4 text-gray-600">
                            {outlier.count.toLocaleString()}
                          </td>
                          <td className="py-3 px-4 text-gray-600">
                            {outlier.percentage.toFixed(2)}%
                          </td>
                          <td className="py-3 px-4">
                            <span
                              className={`text-xs font-medium px-2 py-1 rounded ${
                                outlier.percentage > 5
                                  ? 'bg-red-100 text-red-700'
                                  : outlier.percentage > 2
                                  ? 'bg-yellow-100 text-yellow-700'
                                  : 'bg-green-100 text-green-700'
                              }`}
                            >
                              {outlier.percentage > 5
                                ? 'High'
                                : outlier.percentage > 2
                                ? 'Medium'
                                : 'Low'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-12">
                  <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                  <p className="text-gray-600">
                    No significant outliers detected
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ====================================================================
          Recommendations Section with Trend Indicators
          ==================================================================== */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="border-b border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900">
            Recommended Next Steps
          </h2>
        </div>
        <div className="p-6 space-y-3">
          {/* Data Cleaning */}
          <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
            <div className="flex-shrink-0">
              {getTrendIndicator(
                edaResult.dataQuality.completeness,
                75
              ).icon}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Data Cleaning</p>
              <p className="text-sm text-gray-600 mt-1">
                Completeness:{' '}
                <span
                  className={getTrendIndicator(
                    edaResult.dataQuality.completeness,
                    75
                  ).color}
                >
                  {getTrendIndicator(
                    edaResult.dataQuality.completeness,
                    75
                  ).label}
                </span>{' '}
                Address missing values and remove duplicate rows to improve data
                quality
              </p>
            </div>
          </div>

          {/* Feature Engineering */}
          <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
            <div className="flex-shrink-0">
              {getTrendIndicator(
                edaResult.dataQuality.consistency,
                80
              ).icon}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">
                Feature Engineering
              </p>
              <p className="text-sm text-gray-600 mt-1">
                Consistency:{' '}
                <span
                  className={getTrendIndicator(
                    edaResult.dataQuality.consistency,
                    80
                  ).color}
                >
                  {getTrendIndicator(
                    edaResult.dataQuality.consistency,
                    80
                  ).label}
                </span>{' '}
                Create new features based on correlation analysis and domain
                knowledge
              </p>
            </div>
          </div>

          {/* Visualization */}
          <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
            <div className="flex-shrink-0">
              {getTrendIndicator(edaResult.dataQuality.validity, 85).icon}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900">Visualization</p>
              <p className="text-sm text-gray-600 mt-1">
                Validity:{' '}
                <span
                  className={getTrendIndicator(
                    edaResult.dataQuality.validity,
                    85
                  ).color}
                >
                  {getTrendIndicator(
                    edaResult.dataQuality.validity,
                    85
                  ).label}
                </span>{' '}
                Create custom charts to explore patterns and relationships in your
                data
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EDAReport;
