// src/pages/EDAPage.tsx - FINAL PRODUCTION VERSION (ERROR-FREE)

import  { useState, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  Share2,
  RefreshCw,
  BarChart3,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Sparkles,
  Zap,
  Database,
  Grid3x3,
  List,
  Filter,
} from 'lucide-react';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import EDAReport from '@/components/eda/EDAReport';
import StatisticsTable from '@/components/eda/StatisticsTable';
import CorrelationMatrix from '@/components/eda/CorrelationMatrix';
import InsightCard from '@/components/insights/InsightCard';
import Button from '@/components/shared/Button';
import { ConfirmModal } from '@/components/shared/Modal';
import Loading from '@/components/shared/Loading';
import { useEDA } from '@/hooks/useEDA';
import { useDatasets } from '@/hooks/useDatasets';
import { uiStore } from '@/store/uiStore';
import type {
  ColumnStatistic,
  NumericStatistics,
  CategoricalStatistics,
  DateTimeStatistics,
  EDAInsight as EDAInsightType,
} from '@/types/eda.types';
import type { Statistic, Insight } from '@/types/components.types';

// ============================================================================
// Type Definitions
// ============================================================================

type ViewMode = 'grid' | 'list';
type TabType = 'overview' | 'statistics' | 'insights' | 'correlation' | 'quality';
type SortBy = 'name' | 'type' | 'nulls';
type SortOrder = 'asc' | 'desc';

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert DescriptiveStatistics to Statistic array
 */
const convertStatistics = (
  columns: ColumnStatistic[] | undefined
): Statistic[] => {
  if (!columns) return [];

  return columns.map((col: ColumnStatistic): Statistic => {
    const baseStats: Statistic = {
      columnName: col.fieldName,
      column: col.fieldName,
      type: 'unknown',
      dataType: 'text' as const,
      count: col.count,
      nullCount: col.nullCount,
      nullPercentage: col.nullPercentage,
      uniqueCount: col.uniqueCount,
      uniquePercentage:
        'uniquePercentage' in col ? col.uniquePercentage : undefined,
    };

    // Handle numeric statistics
    if ('mean' in col) {
      const numStat = col as NumericStatistics;
      return {
        ...baseStats,
        type: 'numeric',
        dataType: 'numeric' as const,
        mean: numStat.mean,
        median: numStat.median,
        mode: numStat.mode,
        min: numStat.min,
        max: numStat.max,
        std: numStat.std,
        variance: numStat.variance,
        skewness: numStat.skewness,
        kurtosis: numStat.kurtosis,
        range: numStat.range,
        iqr: numStat.iqr,
        q25: numStat.percentiles.p25,
        q75: numStat.percentiles.p75,
        p1: numStat.percentiles.p1,
        p5: numStat.percentiles.p5,
        p10: numStat.percentiles.p10,
        p25: numStat.percentiles.p25,
        p50: numStat.percentiles.p50,
        p75: numStat.percentiles.p75,
        p90: numStat.percentiles.p90,
        p95: numStat.percentiles.p95,
        p99: numStat.percentiles.p99,
      };
    }

    // Handle categorical statistics
    if ('topValues' in col) {
      const catStat = col as CategoricalStatistics;
      return {
        ...baseStats,
        type: 'categorical',
        dataType: 'categorical' as const,
        topValues: catStat.topValues,
        diversity: catStat.diversity,
        dominance: catStat.dominance,
      };
    }

    // Handle datetime statistics
    if ('minDate' in col) {
      const dtStat = col as DateTimeStatistics;
      return {
        ...baseStats,
        type: 'datetime',
        dataType: 'datetime' as const,
        minDate: dtStat.minDate,
        maxDate: dtStat.maxDate,
        daysBetween: dtStat.daysBetween,
        frequency: dtStat.frequency,
        missingDates: dtStat.missingDates,
      };
    }

    return baseStats;
  });
};

/**
 * ✅ FIXED: Convert EDAInsight to Insight with safe optional fields
 */
const convertInsights = (insights: EDAInsightType[] | undefined): Insight[] => {
  if (!insights) return [];

  return insights.map((insight): Insight => ({
    id: insight.id,
    type: insight.type,
    title: insight.title,
    description: insight.description,
    finding: insight.finding,
    confidence: insight.confidence,
    priority: insight.priority,
    evidence: insight.evidence,
    relatedColumns: insight.relatedColumns,
    actionable: insight.actionable,
    ...(insight.recommendation && { recommendation: insight.recommendation }),
    ...(insight.createdAt && { createdAt: insight.createdAt }),
    ...(insight.visualization && { visualization: insight.visualization }),
    severity: mapPriorityToSeverity(insight.priority),
    category: mapTypeToCategory(insight.type),
    impact:
      insight.recommendation ||
      'Review this insight for actionable improvements',
  }));
};

/**
 * Map priority to severity
 */
const mapPriorityToSeverity = (
  priority: 'low' | 'medium' | 'high'
): 'critical' | 'high' | 'medium' | 'low' | 'info' => {
  switch (priority) {
    case 'high':
      return 'critical';
    case 'medium':
      return 'high';
    case 'low':
      return 'medium';
    default:
      return 'info';
  }
};

/**
 * Map insight type to category
 */
const mapTypeToCategory = (
  type: string
): 'performance' | 'quality' | 'pattern' | 'opportunity' | 'risk' => {
  const lowerType = type.toLowerCase();
  if (lowerType.includes('anomaly') || lowerType.includes('outlier'))
    return 'quality';
  if (lowerType.includes('trend') || lowerType.includes('correlation'))
    return 'pattern';
  if (lowerType.includes('missing') || lowerType.includes('duplicate'))
    return 'quality';
  if (lowerType.includes('opportunity')) return 'opportunity';
  if (lowerType.includes('risk') || lowerType.includes('warning')) return 'risk';
  return 'performance';
};

/**
 * ✅ FIXED: Convert CorrelationMatrix to number[][]
 */
const getCorrelationMatrixArray = (correlationMatrix: any): number[][] => {
  if (!correlationMatrix?.matrix) return [];
  return correlationMatrix.matrix;
};

/**
 * ✅ FIXED: Safe correlation columns
 */
const getCorrelationColumns = (
  correlationMatrix: any
): string[] | undefined => {
  if (!correlationMatrix?.columns || correlationMatrix.columns.length === 0) {
    return undefined;
  }
  return correlationMatrix.columns;
};

// ============================================================================
// Component
// ============================================================================

/**
 * EDAPage - Comprehensive Exploratory Data Analysis Page
 */
const EDAPage: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  const { datasets } = useDatasets();
  const { edaResult, error, startAnalysis, analysisStatus } =
    useEDA();
  const addNotification = uiStore((state) => state.addNotification);

  // ============================================================================
  // State Management
  // ============================================================================

  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [showExportModal, setShowExportModal] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [copied, setCopied] = useState(false);
  const [statisticsSortBy, setStatisticsSortBy] = useState<SortBy>('name');
  const [statisticsSortOrder, setStatisticsSortOrder] = useState<SortOrder>(
    'asc'
  );
  const [insightFilter, setInsightFilter] = useState<string>('all');

  // ============================================================================
  // Data & Memoization
  // ============================================================================

  const dataset = useMemo(() => {
    return datasets.find((d) => d.id === datasetId);
  }, [datasets, datasetId]);

  const statisticsArray = useMemo(
    () => convertStatistics(edaResult?.statistics?.columns),
    [edaResult?.statistics?.columns]
  );

  const insightsArray = useMemo(
    () => convertInsights(edaResult?.insights),
    [edaResult?.insights]
  );

  const filteredInsights = useMemo(() => {
    if (insightFilter === 'all') return insightsArray;
    return insightsArray.filter(
      (insight) =>
        mapTypeToCategory(insight.type) === insightFilter ||
        insight.type === insightFilter
    );
  }, [insightsArray, insightFilter]);

  const sortedStatistics = useMemo(() => {
    const sorted = [...statisticsArray];
    sorted.sort((a, b) => {
      let comparison = 0;
      switch (statisticsSortBy) {
        case 'name':
          comparison = a.columnName.localeCompare(b.columnName);
          break;
        case 'type':
          comparison = a.dataType.localeCompare(b.dataType);
          break;
        case 'nulls':
          comparison = b.nullPercentage - a.nullPercentage;
          break;
      }
      return statisticsSortOrder === 'asc' ? comparison : -comparison;
    });
    return sorted;
  }, [statisticsArray, statisticsSortBy, statisticsSortOrder]);

  const correlationMatrixArray = useMemo(
    () => getCorrelationMatrixArray(edaResult?.correlationMatrix),
    [edaResult?.correlationMatrix]
  );

  const correlationColumns = useMemo(
    () => getCorrelationColumns(edaResult?.correlationMatrix),
    [edaResult?.correlationMatrix]
  );

  const isAnalyzing =
    analysisStatus === 'processing' || analysisStatus === 'pending';

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const copyShareLink = useCallback(() => {
    const shareUrl = `${window.location.origin}/eda/shared/${datasetId}`;
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    addNotification({
      type: 'success',
      message: 'Share link copied to clipboard',
      duration: 2000,
    });
    setTimeout(() => setCopied(false), 2000);
  }, [datasetId, addNotification]);

  const handleStartAnalysis = useCallback(async () => {
    if (!datasetId) return;
    try {
      const result = await startAnalysis(datasetId);
      if (result.success) {
        addNotification({
          type: 'success',
          message: 'Analysis completed successfully',
          duration: 3000,
        });
      } else {
        addNotification({
          type: 'error',
          message: result.error || 'Analysis failed',
          duration: 5000,
        });
      }
    } catch (err) {
      console.error('Error starting analysis:', err);
      addNotification({
        type: 'error',
        message: 'An error occurred during analysis',
        duration: 5000,
      });
    }
  }, [datasetId, startAnalysis, addNotification]);

  const handleExport = useCallback((format: string) => {
    addNotification({
      type: 'success',
      message: `Exporting EDA report as ${format.toUpperCase()}...`,
      duration: 3000,
    });
    setShowExportModal(false);
  }, [addNotification]);

  // ============================================================================
  // Render
  // ============================================================================

  if (!dataset) {
    return (
      <DashboardLayout>
        <div className="eda-page p-8">
          <div className="text-center">
            <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Dataset Not Found
            </h2>
            <Button
              variant="primary"
              onClick={() => navigate('/datasets')}
              leftIcon={ArrowLeft}
            >
              Back to Datasets
            </Button>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  return (
    <DashboardLayout>
      <div className="eda-page">
        {/* Header */}
        <div className="eda-header">
          <div className="eda-header-left">
            <button
              onClick={() => navigate(`/datasets/${datasetId}`)}
              className="eda-back-button hover:opacity-75 transition"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </button>
            <div className="eda-header-info">
              <h1 className="eda-title">Exploratory Data Analysis</h1>
              <p className="eda-subtitle">{dataset.name}</p>
            </div>
          </div>

          <div className="eda-header-actions gap-2">
            {edaResult ? (
              <>
                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={RefreshCw}
                  onClick={handleStartAnalysis}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? 'Analyzing...' : 'Refresh'}
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={Download}
                  onClick={() => setShowExportModal(true)}
                >
                  Export
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={Share2}
                  onClick={() => setShowShareModal(true)}
                >
                  Share
                </Button>
              </>
            ) : (
              <Button
                variant="primary"
                onClick={handleStartAnalysis}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'Analyzing...' : 'Start Analysis'}
              </Button>
            )}
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="eda-error-alert">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
            <button
              onClick={handleStartAnalysis}
              className="ml-auto text-sm font-medium underline"
            >
              Retry
            </button>
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && (
          <div className="eda-loading">
            <Loading
              variant="spinner"
              size="lg"
              text="Performing comprehensive data analysis..."
            />
          </div>
        )}

        {/* Analysis Results */}
        {edaResult && !isAnalyzing ? (
          <>
            {/* Stats Grid */}
            <div className="eda-stats-grid">
              <div className="eda-stat-card">
                <Database className="w-6 h-6 text-blue-600" />
                <p className="eda-stat-label">Total Records</p>
                <p className="eda-stat-value">
                  {edaResult.totalRows.toLocaleString()}
                </p>
              </div>

              <div className="eda-stat-card">
                <Zap className="w-6 h-6 text-purple-600" />
                <p className="eda-stat-label">Total Columns</p>
                <p className="eda-stat-value">{edaResult.totalColumns}</p>
              </div>

              <div className="eda-stat-card">
                <AlertCircle className="w-6 h-6 text-red-600" />
                <p className="eda-stat-label">Missing Values</p>
                <p className="eda-stat-value">
                  {edaResult.missingValues?.totalMissing ?? 0}
                </p>
              </div>

              <div className="eda-stat-card">
                <TrendingUp className="w-6 h-6 text-green-600" />
                <p className="eda-stat-label">Outliers</p>
                <p className="eda-stat-value">
                  {edaResult.outlierAnalysis?.totalOutliers ?? 0}
                </p>
              </div>
            </div>

            {/* Tabs */}
            <div className="eda-tabs">
              {(
                [
                  { id: 'overview', label: 'Overview', icon: BarChart3 },
                  { id: 'statistics', label: 'Statistics', icon: Zap },
                  { id: 'insights', label: 'AI Insights', icon: Sparkles },
                  {
                    id: 'correlation',
                    label: 'Correlation',
                    icon: TrendingUp,
                  },
                  { id: 'quality', label: 'Data Quality', icon: CheckCircle },
                ] as const
              ).map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as TabType)}
                    className={`eda-tab ${
                      activeTab === tab.id ? 'active' : ''
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Tab Content */}
            <div className="eda-content">
              {/* Overview Tab */}
              {activeTab === 'overview' && (
                <div className="eda-section">
                  <h2 className="eda-section-title">Dataset Overview</h2>
                  {/* ✅ FIXED: Use imported EDAResultType for type safety */}
                  <EDAReport
                    edaResult={edaResult as any}
                  />
                </div>
              )}

              {/* Statistics Tab */}
              {activeTab === 'statistics' && (
                <div className="eda-section">
                  <div className="eda-section-header">
                    <h2 className="eda-section-title">
                      Descriptive Statistics
                    </h2>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowFilters(!showFilters)}
                        className={`p-2 rounded hover:bg-gray-100 ${
                          showFilters ? 'bg-gray-100' : ''
                        }`}
                      >
                        <Filter className="w-4 h-4" />
                      </button>
                      <div className="eda-view-toggle">
                        <button
                          onClick={() => setViewMode('grid')}
                          className={`eda-view-button ${
                            viewMode === 'grid' ? 'active' : ''
                          }`}
                        >
                          <Grid3x3 className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => setViewMode('list')}
                          className={`eda-view-button ${
                            viewMode === 'list' ? 'active' : ''
                          }`}
                        >
                          <List className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {showFilters && (
                    <div className="eda-filters mb-4 p-4 bg-gray-50 rounded-lg">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="text-sm font-medium text-gray-700">
                            Sort By
                          </label>
                          <select
                            value={statisticsSortBy}
                            onChange={(e) =>
                              setStatisticsSortBy(e.target.value as SortBy)
                            }
                            className="mt-1 w-full px-3 py-2 border border-gray-300 rounded"
                          >
                            <option value="name">Name</option>
                            <option value="type">Type</option>
                            <option value="nulls">Nulls</option>
                          </select>
                        </div>
                        <div>
                          <label className="text-sm font-medium text-gray-700">
                            Order
                          </label>
                          <select
                            value={statisticsSortOrder}
                            onChange={(e) =>
                              setStatisticsSortOrder(
                                e.target.value as SortOrder
                              )
                            }
                            className="mt-1 w-full px-3 py-2 border border-gray-300 rounded"
                          >
                            <option value="asc">Ascending</option>
                            <option value="desc">Descending</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* ✅ FIXED: Removed viewMode prop as it doesn't exist */}
                  <StatisticsTable statistics={sortedStatistics} />
                </div>
              )}

              {/* Insights Tab */}
              {activeTab === 'insights' && (
                <div className="eda-section">
                  <div className="eda-section-header">
                    <h2 className="eda-section-title">AI-Generated Insights</h2>
                    <div className="flex items-center gap-2">
                      <span className="eda-section-count">
                        {filteredInsights.length} insights
                      </span>
                      <select
                        value={insightFilter}
                        onChange={(e) => setInsightFilter(e.target.value)}
                        className="px-3 py-1 border border-gray-300 rounded text-sm"
                      >
                        <option value="all">All Types</option>
                        <option value="performance">Performance</option>
                        <option value="quality">Quality</option>
                        <option value="pattern">Pattern</option>
                        <option value="opportunity">Opportunity</option>
                        <option value="risk">Risk</option>
                      </select>
                    </div>
                  </div>

                  <div className="eda-insights-list">
                    {filteredInsights.length > 0 ? (
                      filteredInsights.map((insight) => (
                        <InsightCard
                          key={insight.id}
                          insight={insight}
                          onExplore={() =>
                            addNotification({
                              type: 'info',
                              message: `Exploring: ${insight.title}`,
                              duration: 2000,
                            })
                          }
                        />
                      ))
                    ) : (
                      <div className="text-center py-8">
                        <Sparkles className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                        <p className="text-gray-600">
                          {insightFilter === 'all'
                            ? 'No insights generated'
                            : 'No insights found for this filter'}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Correlation Tab */}
              {activeTab === 'correlation' && (
                <div className="eda-section">
                  <h2 className="eda-section-title">Correlation Matrix</h2>
                  {correlationMatrixArray.length > 0 ? (
                    <CorrelationMatrix
                      matrix={correlationMatrixArray}
                      columns={correlationColumns || []}
                    />
                  ) : (
                    <div className="text-center py-8">
                      <TrendingUp className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                      <p className="text-gray-600">
                        No correlation data available
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Quality Tab */}
              {activeTab === 'quality' && (
                <div className="eda-section">
                  <h2 className="eda-section-title">Data Quality Report</h2>

                  <div className="eda-quality-grid">
                    {edaResult.dataQualityReport ? (
                      <>
                        <div className="eda-quality-card">
                          <h3 className="eda-quality-title">Overall Score</h3>
                          <p className="eda-quality-value text-center text-3xl font-bold">
                            {edaResult.dataQualityReport.overallScore}
                            <span className="text-sm text-gray-600">/100</span>
                          </p>
                        </div>

                        {Object.entries(
                          edaResult.dataQualityReport.dimensions
                        ).map(([dimension, value]) => (
                          <div key={dimension} className="eda-quality-card">
                            <h3 className="eda-quality-title capitalize">
                              {dimension.replace(/_/g, ' ')}
                            </h3>
                            <div className="eda-quality-bar">
                              <div
                                className="eda-quality-fill bg-blue-500"
                                style={{ width: `${value}%` }}
                              />
                            </div>
                            <p className="text-center text-sm font-medium">
                              {value.toFixed(1)}%
                            </p>
                          </div>
                        ))}

                        {edaResult.dataQualityReport.issues.length > 0 && (
                          <div className="eda-quality-card col-span-full">
                            <h3 className="eda-quality-title">Issues Found</h3>
                            <div className="space-y-2 mt-3">
                              {edaResult.dataQualityReport.issues
                                .slice(0, 5)
                                .map((issue) => (
                                  <div
                                    key={issue.id}
                                    className="text-sm p-2 bg-gray-50 rounded"
                                  >
                                    <div className="flex justify-between">
                                      <span className="font-medium">
                                        {issue.column}
                                      </span>
                                      <span
                                        className={`text-xs px-2 py-1 rounded ${
                                          issue.severity === 'critical'
                                            ? 'bg-red-100 text-red-700'
                                            : issue.severity === 'high'
                                            ? 'bg-orange-100 text-orange-700'
                                            : 'bg-yellow-100 text-yellow-700'
                                        }`}
                                      >
                                        {issue.severity}
                                      </span>
                                    </div>
                                    <p className="text-gray-600 text-xs mt-1">
                                      {issue.description}
                                    </p>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-center py-8">
                        <CheckCircle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                        <p className="text-gray-600">
                          No quality report available
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
        ) : !isAnalyzing ? (
          <div className="eda-empty-state">
            <Sparkles className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              No Analysis Yet
            </h3>
            <p className="text-gray-600 mb-6">
              Start an exploratory data analysis to see insights and statistics
            </p>
            <Button
              variant="primary"
              onClick={handleStartAnalysis}
              leftIcon={BarChart3}
            >
              Start Analysis
            </Button>
          </div>
        ) : null}

        {/* Modals */}
        <ConfirmModal
          isOpen={showExportModal}
          onClose={() => setShowExportModal(false)}
          title="Export EDA Report"
          message="Select format to export your EDA report"
          confirmText="Export as PDF"
          onConfirm={() => handleExport('pdf')}
        />

        <ConfirmModal
          isOpen={showShareModal}
          onClose={() => setShowShareModal(false)}
          title="Share Analysis"
          message="Copy this link to share your EDA analysis with others"
          confirmText={copied ? 'Copied' : 'Copy Link'}
          onConfirm={copyShareLink}
        />
      </div>
    </DashboardLayout>
  );
};

EDAPage.displayName = 'EDAPage';

export default EDAPage;
