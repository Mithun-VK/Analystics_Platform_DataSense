// src/components/cleaning/CleaningConfig.tsx

import { useState } from 'react';
import {
  Droplet,
  Copy,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Play,
  Settings,
  Info,
  ChevronDown,
  ChevronUp,
  FileText,
  Download,
} from 'lucide-react';

interface CleaningOperation {
  id: string;
  type: 'missing' | 'duplicates' | 'outliers' | 'standardize' | 'transform';
  enabled: boolean;
  config: any;
}

interface ColumnInfo {
  name: string;
  type: string;
  missingCount: number;
  missingPercentage: number;
  duplicateCount: number;
  outlierCount: number;
}

interface CleaningConfigProps {
  datasetId: string;
  columns: ColumnInfo[];
  totalRows: number;
  onApply?: (operations: CleaningOperation[]) => void;
  onPreview?: (operations: CleaningOperation[]) => void;
}

type ImputationMethod = 'mean' | 'median' | 'mode' | 'forward_fill' | 'backward_fill' | 'drop';
type OutlierMethod = 'zscore' | 'iqr' | 'isolation_forest' | 'none';
type StandardizeMethod = 'zscore' | 'minmax' | 'robust' | 'none';

/**
 * CleaningConfig - Configuration panel for data cleaning operations
 * Features: Missing value handling, duplicate removal, outlier detection
 * Supports: Multiple imputation methods, preview mode, batch operations
 */
const CleaningConfig: React.FC<CleaningConfigProps> = ({
  columns,
  totalRows,
  onApply,
  onPreview,
}) => {
  const [operations, setOperations] = useState<CleaningOperation[]>([
    {
      id: 'missing',
      type: 'missing',
      enabled: false,
      config: {
        method: 'mean' as ImputationMethod,
        threshold: 50,
        columns: [],
      },
    },
    {
      id: 'duplicates',
      type: 'duplicates',
      enabled: false,
      config: {
        keepFirst: true,
        considerColumns: [],
      },
    },
    {
      id: 'outliers',
      type: 'outliers',
      enabled: false,
      config: {
        method: 'iqr' as OutlierMethod,
        threshold: 1.5,
        action: 'remove',
      },
    },
    {
      id: 'standardize',
      type: 'standardize',
      enabled: false,
      config: {
        method: 'zscore' as StandardizeMethod,
        columns: [],
      },
    },
  ]);

  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['missing'])
  );
  const [isApplying, setIsApplying] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);

  // Toggle operation enabled state
  const toggleOperation = (id: string) => {
    setOperations((prev) =>
      prev.map((op) => (op.id === id ? { ...op, enabled: !op.enabled } : op))
    );
  };

  // Update operation config
  const updateConfig = (id: string, config: any) => {
    setOperations((prev) =>
      prev.map((op) => (op.id === id ? { ...op, config: { ...op.config, ...config } } : op))
    );
  };

  // Toggle section expansion
  const toggleSection = (id: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(id)) {
        newSet.delete(id);
      } else {
        newSet.add(id);
      }
      return newSet;
    });
  };

  // Handle apply
  const handleApply = async () => {
    setIsApplying(true);
    try {
      const enabledOps = operations.filter((op) => op.enabled);
      await onApply?.(enabledOps);
    } finally {
      setIsApplying(false);
    }
  };

  // Handle preview
  const handlePreview = async () => {
    setIsPreviewing(true);
    try {
      const enabledOps = operations.filter((op) => op.enabled);
      await onPreview?.(enabledOps);
    } finally {
      setIsPreviewing(false);
    }
  };

  // Get data quality summary
  const totalMissing = columns.reduce((sum, col) => sum + col.missingCount, 0);
  const totalDuplicates = columns.reduce((sum, col) => sum + col.duplicateCount, 0);
  const totalOutliers = columns.reduce((sum, col) => sum + col.outlierCount, 0);
  const enabledCount = operations.filter((op) => op.enabled).length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
          <Droplet className="w-6 h-6 text-blue-600" />
          <span>Data Cleaning Configuration</span>
        </h2>
        <p className="text-gray-600 mt-1">
          Configure cleaning operations to improve data quality
        </p>
      </div>

      {/* Data Quality Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-yellow-100 rounded-lg flex items-center justify-center">
              <AlertCircle className="w-5 h-5 text-yellow-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Missing Values</p>
              <p className="text-xl font-bold text-gray-900">
                {totalMissing.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500">
                {((totalMissing / (totalRows * columns.length)) * 100).toFixed(2)}% of total
              </p>
            </div>
          </div>
        </div>

        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
              <Copy className="w-5 h-5 text-red-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Duplicate Rows</p>
              <p className="text-xl font-bold text-gray-900">
                {totalDuplicates.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500">
                {((totalDuplicates / totalRows) * 100).toFixed(2)}% of total
              </p>
            </div>
          </div>
        </div>

        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Outliers</p>
              <p className="text-xl font-bold text-gray-900">
                {totalOutliers.toLocaleString()}
              </p>
              <p className="text-xs text-gray-500">Detected values</p>
            </div>
          </div>
        </div>
      </div>

      {/* Cleaning Operations */}
      <div className="space-y-4">
        {/* Missing Values */}
        <div className="card overflow-hidden">
          <div
            className="card-header cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => toggleSection('missing')}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={operations.find((op) => op.id === 'missing')?.enabled}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleOperation('missing');
                  }}
                  className="w-5 h-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
                />
                <AlertCircle className="w-5 h-5 text-yellow-600" />
                <div>
                  <h3 className="text-base font-semibold text-gray-900">
                    Handle Missing Values
                  </h3>
                  <p className="text-sm text-gray-600">
                    {totalMissing.toLocaleString()} missing values found
                  </p>
                </div>
              </div>
              {expandedSections.has('missing') ? (
                <ChevronUp className="w-5 h-5 text-gray-500" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-500" />
              )}
            </div>
          </div>

          {expandedSections.has('missing') && (
            <div className="card-body space-y-4 bg-gray-50">
              <div>
                <label className="label">Imputation Method</label>
                <select
                  value={operations.find((op) => op.id === 'missing')?.config.method}
                  onChange={(e) =>
                    updateConfig('missing', { method: e.target.value as ImputationMethod })
                  }
                  className="select w-full"
                >
                  <option value="mean">Mean (Average value)</option>
                  <option value="median">Median (Middle value)</option>
                  <option value="mode">Mode (Most frequent value)</option>
                  <option value="forward_fill">Forward Fill (Previous value)</option>
                  <option value="backward_fill">Backward Fill (Next value)</option>
                  <option value="drop">Drop Rows (Remove entries)</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  Choose how to handle missing values in numeric columns
                </p>
              </div>

              <div>
                <label className="label">
                  Missing Value Threshold: {operations.find((op) => op.id === 'missing')?.config.threshold}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={operations.find((op) => op.id === 'missing')?.config.threshold}
                  onChange={(e) =>
                    updateConfig('missing', { threshold: parseInt(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Drop columns with missing values above this threshold
                </p>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-start space-x-2">
                  <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-gray-700">
                    <strong>Recommendation:</strong> Use mean for normally distributed data,
                    median for skewed data, and mode for categorical data.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Duplicates */}
        <div className="card overflow-hidden">
          <div
            className="card-header cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => toggleSection('duplicates')}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={operations.find((op) => op.id === 'duplicates')?.enabled}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleOperation('duplicates');
                  }}
                  className="w-5 h-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
                />
                <Copy className="w-5 h-5 text-red-600" />
                <div>
                  <h3 className="text-base font-semibold text-gray-900">
                    Remove Duplicates
                  </h3>
                  <p className="text-sm text-gray-600">
                    {totalDuplicates.toLocaleString()} duplicate rows found
                  </p>
                </div>
              </div>
              {expandedSections.has('duplicates') ? (
                <ChevronUp className="w-5 h-5 text-gray-500" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-500" />
              )}
            </div>
          </div>

          {expandedSections.has('duplicates') && (
            <div className="card-body space-y-4 bg-gray-50">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">Keep First Occurrence</label>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={operations.find((op) => op.id === 'duplicates')?.config.keepFirst}
                    onChange={(e) =>
                      updateConfig('duplicates', { keepFirst: e.target.checked })
                    }
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                </label>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-start space-x-2">
                  <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-gray-700">
                    When enabled, the first occurrence of duplicate rows will be kept.
                    Otherwise, the last occurrence will be retained.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Outliers */}
        <div className="card overflow-hidden">
          <div
            className="card-header cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => toggleSection('outliers')}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={operations.find((op) => op.id === 'outliers')?.enabled}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleOperation('outliers');
                  }}
                  className="w-5 h-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
                />
                <TrendingUp className="w-5 h-5 text-purple-600" />
                <div>
                  <h3 className="text-base font-semibold text-gray-900">
                    Handle Outliers
                  </h3>
                  <p className="text-sm text-gray-600">
                    {totalOutliers.toLocaleString()} outliers detected
                  </p>
                </div>
              </div>
              {expandedSections.has('outliers') ? (
                <ChevronUp className="w-5 h-5 text-gray-500" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-500" />
              )}
            </div>
          </div>

          {expandedSections.has('outliers') && (
            <div className="card-body space-y-4 bg-gray-50">
              <div>
                <label className="label">Detection Method</label>
                <select
                  value={operations.find((op) => op.id === 'outliers')?.config.method}
                  onChange={(e) =>
                    updateConfig('outliers', { method: e.target.value as OutlierMethod })
                  }
                  className="select w-full"
                >
                  <option value="iqr">IQR (Interquartile Range)</option>
                  <option value="zscore">Z-Score (Standard Deviations)</option>
                  <option value="isolation_forest">Isolation Forest (ML-based)</option>
                  <option value="none">None (Keep all values)</option>
                </select>
              </div>

              <div>
                <label className="label">Action for Outliers</label>
                <select
                  value={operations.find((op) => op.id === 'outliers')?.config.action}
                  onChange={(e) => updateConfig('outliers', { action: e.target.value })}
                  className="select w-full"
                >
                  <option value="remove">Remove (Delete rows)</option>
                  <option value="cap">Cap (Limit to threshold)</option>
                  <option value="replace">Replace (With median)</option>
                </select>
              </div>

              <div>
                <label className="label">
                  Sensitivity: {operations.find((op) => op.id === 'outliers')?.config.threshold}
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="3"
                  step="0.1"
                  value={operations.find((op) => op.id === 'outliers')?.config.threshold}
                  onChange={(e) =>
                    updateConfig('outliers', { threshold: parseFloat(e.target.value) })
                  }
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Higher values = less aggressive outlier detection
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Standardization */}
        <div className="card overflow-hidden">
          <div
            className="card-header cursor-pointer hover:bg-gray-50 transition-colors"
            onClick={() => toggleSection('standardize')}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  checked={operations.find((op) => op.id === 'standardize')?.enabled}
                  onChange={(e) => {
                    e.stopPropagation();
                    toggleOperation('standardize');
                  }}
                  className="w-5 h-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
                />
                <Settings className="w-5 h-5 text-green-600" />
                <div>
                  <h3 className="text-base font-semibold text-gray-900">
                    Standardize Values
                  </h3>
                  <p className="text-sm text-gray-600">
                    Normalize data for consistent scaling
                  </p>
                </div>
              </div>
              {expandedSections.has('standardize') ? (
                <ChevronUp className="w-5 h-5 text-gray-500" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-500" />
              )}
            </div>
          </div>

          {expandedSections.has('standardize') && (
            <div className="card-body space-y-4 bg-gray-50">
              <div>
                <label className="label">Scaling Method</label>
                <select
                  value={operations.find((op) => op.id === 'standardize')?.config.method}
                  onChange={(e) =>
                    updateConfig('standardize', { method: e.target.value as StandardizeMethod })
                  }
                  className="select w-full"
                >
                  <option value="zscore">Z-Score (Mean=0, Std=1)</option>
                  <option value="minmax">Min-Max (Scale to 0-1)</option>
                  <option value="robust">Robust (Median & IQR)</option>
                  <option value="none">None (No scaling)</option>
                </select>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex items-start space-x-2">
                  <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                  <p className="text-xs text-gray-700">
                    Standardization is useful when features have different units or scales.
                    Required for many machine learning algorithms.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Summary */}
      <div className="card card-body bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-semibold text-gray-900 mb-1">
              {enabledCount} Operation{enabledCount !== 1 ? 's' : ''} Enabled
            </h3>
            <p className="text-sm text-gray-600">
              {enabledCount === 0
                ? 'Select operations to clean your data'
                : 'Review and apply the selected cleaning operations'}
            </p>
          </div>
          <CheckCircle className="w-8 h-8 text-blue-600" />
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between space-x-4 pt-4 border-t border-gray-200">
        <button
          onClick={handlePreview}
          disabled={enabledCount === 0 || isPreviewing}
          className="btn btn-secondary flex items-center space-x-2"
        >
          {isPreviewing ? (
            <RefreshCw className="w-4 h-4 animate-spin" />
          ) : (
            <FileText className="w-4 h-4" />
          )}
          <span>{isPreviewing ? 'Generating Preview...' : 'Preview Changes'}</span>
        </button>

        <div className="flex items-center space-x-3">
          <button className="btn btn-secondary flex items-center space-x-2">
            <Download className="w-4 h-4" />
            <span>Export Config</span>
          </button>
          <button
            onClick={handleApply}
            disabled={enabledCount === 0 || isApplying}
            className="btn btn-primary flex items-center space-x-2"
          >
            {isApplying ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            <span>{isApplying ? 'Applying...' : 'Apply Cleaning'}</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default CleaningConfig;
