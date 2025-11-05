// src/components/eda/StatisticsTable.tsx - FINAL PRODUCTION VERSION

import { useState, useMemo } from 'react';
import {
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Download,
  Search,
  Eye,
  EyeOff,
  TrendingUp,
  AlertCircle,
  Copy,
  Check,
} from 'lucide-react';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * ✅ FIXED: All numeric fields are optional to match API reality
 */
export interface Statistic {
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
}

export interface StatisticsTableProps {
  statistics: Statistic[];
  onColumnClick?: (columnName: string) => void;
}

type SortField = keyof Statistic;
type SortOrder = 'asc' | 'desc' | null;

// ============================================================================
// Component
// ============================================================================

/**
 * StatisticsTable - Comprehensive descriptive statistics display
 * Features: Sortable columns, data type filtering, search, export, column visibility
 * Displays key statistical measures: count, mean, std, quartiles, min/max
 */
const StatisticsTable: React.FC<StatisticsTableProps> = ({
  statistics,
  onColumnClick,
}) => {
  // ============================================================================
  // State Management
  // ============================================================================

  const [sortField, setSortField] = useState<SortField | null>(null);
  const [sortOrder, setSortOrder] = useState<SortOrder>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterDataType, setFilterDataType] = useState<string>('all');
  const [hiddenColumns, setHiddenColumns] = useState<Set<string>>(new Set());
  const [copiedValue, setCopiedValue] = useState<string | null>(null);

  // ============================================================================
  // Constants
  // ============================================================================

  /**
   * Available columns for visibility toggle
   */
  const availableColumns = [
    { key: 'count', label: 'Count' },
    { key: 'mean', label: 'Mean' },
    { key: 'std', label: 'Std Dev' },
    { key: 'min', label: 'Min' },
    { key: 'q25', label: '25%' },
    { key: 'median', label: 'Median' },
    { key: 'q75', label: '75%' },
    { key: 'max', label: 'Max' },
  ];

  // ============================================================================
  // Memoized Computations
  // ============================================================================

  /**
   * Get unique data types from statistics
   */
  const dataTypes = useMemo(() => {
    const types = new Set(statistics.map((s) => s.dataType));
    return ['all', ...Array.from(types)];
  }, [statistics]);

  /**
   * Filter and sort statistics with proper null handling
   */
  const processedStatistics = useMemo(() => {
    let filtered = [...statistics];

    // Apply search filter
    if (searchQuery) {
      filtered = filtered.filter((stat) =>
        stat.columnName.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Apply data type filter
    if (filterDataType !== 'all') {
      filtered = filtered.filter((stat) => stat.dataType === filterDataType);
    }

    // Apply sorting
    if (sortField && sortOrder) {
      filtered.sort((a, b) => {
        const aValue = a[sortField];
        const bValue = b[sortField];

        // Handle string comparisons
        if (typeof aValue === 'string' && typeof bValue === 'string') {
          return sortOrder === 'asc'
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
        }

        // Handle numeric comparisons with null safety
        const aNum = typeof aValue === 'number' ? aValue : -Infinity;
        const bNum = typeof bValue === 'number' ? bValue : -Infinity;
        return sortOrder === 'asc' ? aNum - bNum : bNum - aNum;
      });
    }

    return filtered;
  }, [statistics, searchQuery, filterDataType, sortField, sortOrder]);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  /**
   * Handle column sorting
   */
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      if (sortOrder === 'asc') {
        setSortOrder('desc');
      } else if (sortOrder === 'desc') {
        setSortField(null);
        setSortOrder(null);
      }
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  /**
   * Toggle column visibility
   */
  const toggleColumnVisibility = (column: string) => {
    const newHidden = new Set(hiddenColumns);
    if (newHidden.has(column)) {
      newHidden.delete(column);
    } else {
      newHidden.add(column);
    }
    setHiddenColumns(newHidden);
  };

  /**
   * Copy value to clipboard
   */
  const copyToClipboard = (value: string) => {
    navigator.clipboard.writeText(value);
    setCopiedValue(value);
    setTimeout(() => setCopiedValue(null), 2000);
  };

  /**
   * Export to CSV
   */
  const exportToCSV = () => {
    const headers = [
      'Column Name',
      'Data Type',
      ...availableColumns
        .filter((col) => !hiddenColumns.has(col.key))
        .map((col) => col.label),
    ];

    const rows = processedStatistics.map((stat) => {
      const row: any[] = [stat.columnName, stat.dataType];
      availableColumns.forEach((col) => {
        if (!hiddenColumns.has(col.key)) {
          const value = (stat as any)[col.key];
          row.push(value !== undefined ? value : '-');
        }
      });
      return row;
    });

    const csvContent = [
      headers.join(','),
      ...rows.map((row) => row.join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'statistics.csv';
    link.click();
    window.URL.revokeObjectURL(url);
  };

  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * Get sort icon based on current sort state
   */
  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 text-gray-400" />;
    }
    return sortOrder === 'asc' ? (
      <ArrowUp className="w-4 h-4 text-blue-600" />
    ) : (
      <ArrowDown className="w-4 h-4 text-blue-600" />
    );
  };

  /**
   * Format number for display with null safety
   */
  const formatNumber = (value: number | undefined, decimals: number = 2): string => {
    if (value === null || value === undefined || isNaN(value)) {
      return '-';
    }
    return value.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  };

  /**
   * Get data type badge color
   */
  const getDataTypeBadge = (dataType: string) => {
    const colors: { [key: string]: string } = {
      int64: 'bg-blue-100 text-blue-700',
      int32: 'bg-blue-100 text-blue-700',
      float64: 'bg-green-100 text-green-700',
      float32: 'bg-green-100 text-green-700',
      object: 'bg-purple-100 text-purple-700',
      string: 'bg-purple-100 text-purple-700',
      bool: 'bg-yellow-100 text-yellow-700',
      boolean: 'bg-yellow-100 text-yellow-700',
      datetime: 'bg-pink-100 text-pink-700',
      datetime64: 'bg-pink-100 text-pink-700',
      category: 'bg-indigo-100 text-indigo-700',
      numeric: 'bg-green-100 text-green-700',
      text: 'bg-purple-100 text-purple-700',
    };
    return colors[dataType.toLowerCase()] || 'bg-gray-100 text-gray-700';
  };

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div className="space-y-4">
      {/* ====================================================================
          Toolbar with Search, Filter, and Actions
          ==================================================================== */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Search and Filter */}
        <div className="flex flex-col sm:flex-row gap-3 flex-1">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search columns..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <select
            value={filterDataType}
            onChange={(e) => setFilterDataType(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent min-w-[140px]"
          >
            {dataTypes.map((type) => (
              <option key={type} value={type}>
                {type === 'all' ? 'All Types' : type}
              </option>
            ))}
          </select>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-2">
          <button
            onClick={exportToCSV}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors text-sm font-medium"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </button>

          <div className="relative group">
            <button className="flex items-center space-x-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors text-sm font-medium">
              <Eye className="w-4 h-4" />
              <span>Columns</span>
            </button>

            {/* Column Visibility Dropdown */}
            <div className="absolute right-0 top-10 z-20 w-56 bg-white rounded-lg shadow-xl border border-gray-200 py-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
              <div className="px-4 py-2 border-b border-gray-200">
                <p className="text-xs font-semibold text-gray-700 uppercase">
                  Toggle Columns
                </p>
              </div>
              {availableColumns.map((col) => (
                <button
                  key={col.key}
                  onClick={() => toggleColumnVisibility(col.key)}
                  className="w-full flex items-center justify-between px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                >
                  <span>{col.label}</span>
                  {hiddenColumns.has(col.key) ? (
                    <EyeOff className="w-4 h-4 text-gray-400" />
                  ) : (
                    <Eye className="w-4 h-4 text-blue-600" />
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ====================================================================
          Statistics Summary Banner
          ==================================================================== */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-sm">
            <TrendingUp className="w-5 h-5 text-blue-600" />
            <span className="font-medium text-gray-900">
              Showing {processedStatistics.length} of {statistics.length} columns
            </span>
          </div>
          {searchQuery || filterDataType !== 'all' ? (
            <button
              onClick={() => {
                setSearchQuery('');
                setFilterDataType('all');
              }}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              Clear filters
            </button>
          ) : null}
        </div>
      </div>

      {/* ====================================================================
          Statistics Table
          ==================================================================== */}
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table className="w-full border-collapse">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr className="border-b border-gray-200">
              <th className="sticky left-0 bg-gray-50 z-20 px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('columnName')}
                  className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                >
                  <span>Column Name</span>
                  {getSortIcon('columnName')}
                </button>
              </th>
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('dataType')}
                  className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                >
                  <span>Type</span>
                  {getSortIcon('dataType')}
                </button>
              </th>
              {!hiddenColumns.has('count') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('count')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>Count</span>
                    {getSortIcon('count')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('mean') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('mean')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>Mean</span>
                    {getSortIcon('mean')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('std') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('std')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>Std Dev</span>
                    {getSortIcon('std')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('min') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('min')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>Min</span>
                    {getSortIcon('min')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('q25') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('q25')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>25%</span>
                    {getSortIcon('q25')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('median') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('median')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>Median</span>
                    {getSortIcon('median')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('q75') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('q75')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>75%</span>
                    {getSortIcon('q75')}
                  </button>
                </th>
              )}
              {!hiddenColumns.has('max') && (
                <th className="px-4 py-3 text-left">
                  <button
                    onClick={() => handleSort('max')}
                    className="flex items-center space-x-2 hover:text-blue-600 transition-colors font-semibold text-gray-900"
                  >
                    <span>Max</span>
                    {getSortIcon('max')}
                  </button>
                </th>
              )}
              <th className="w-12"></th>
            </tr>
          </thead>
          <tbody>
            {processedStatistics.length > 0 ? (
              processedStatistics.map((stat, index) => (
                <tr key={index} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                  <td className="sticky left-0 bg-white hover:bg-gray-50 px-4 py-3">
                    <button
                      onClick={() => onColumnClick?.(stat.columnName)}
                      className="font-medium text-blue-600 hover:text-blue-700 hover:underline text-left"
                    >
                      {stat.columnName}
                    </button>
                  </td>
                  <td className="px-4 py-3">
                    <span
                      className={`badge ${getDataTypeBadge(
                        stat.dataType
                      )} uppercase text-xs px-2 py-1 rounded font-medium`}
                    >
                      {stat.dataType}
                    </span>
                  </td>
                  {!hiddenColumns.has('count') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {stat.count.toLocaleString()}
                    </td>
                  )}
                  {!hiddenColumns.has('mean') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.mean)}
                    </td>
                  )}
                  {!hiddenColumns.has('std') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.std)}
                    </td>
                  )}
                  {!hiddenColumns.has('min') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.min)}
                    </td>
                  )}
                  {!hiddenColumns.has('q25') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.q25)}
                    </td>
                  )}
                  {!hiddenColumns.has('median') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.median)}
                    </td>
                  )}
                  {!hiddenColumns.has('q75') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.q75)}
                    </td>
                  )}
                  {!hiddenColumns.has('max') && (
                    <td className="font-mono text-sm px-4 py-3">
                      {formatNumber(stat.max)}
                    </td>
                  )}
                  <td className="px-4 py-3">
                    <button
                      onClick={() => copyToClipboard(stat.columnName)}
                      className="p-1 hover:bg-gray-100 rounded transition-colors"
                      title="Copy column name"
                    >
                      {copiedValue === stat.columnName ? (
                        <Check className="w-4 h-4 text-green-600" />
                      ) : (
                        <Copy className="w-4 h-4 text-gray-500" />
                      )}
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={11} className="text-center py-12">
                  <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-3" />
                  <p className="text-gray-600">No statistics found</p>
                  {searchQuery || filterDataType !== 'all' ? (
                    <button
                      onClick={() => {
                        setSearchQuery('');
                        setFilterDataType('all');
                      }}
                      className="text-sm text-blue-600 hover:text-blue-700 font-medium mt-2"
                    >
                      Clear filters
                    </button>
                  ) : null}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* ====================================================================
          Legend
          ==================================================================== */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <h4 className="text-sm font-semibold text-gray-900 mb-3">
          Statistical Measures
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div>
            <span className="font-medium text-gray-700">Count:</span>
            <span className="text-gray-600 ml-1">Number of non-null values</span>
          </div>
          <div>
            <span className="font-medium text-gray-700">Mean:</span>
            <span className="text-gray-600 ml-1">Average value</span>
          </div>
          <div>
            <span className="font-medium text-gray-700">Std Dev:</span>
            <span className="text-gray-600 ml-1">Standard deviation</span>
          </div>
          <div>
            <span className="font-medium text-gray-700">Quartiles:</span>
            <span className="text-gray-600 ml-1">25%, 50%, 75% percentiles</span>
          </div>
          <div>
            <span className="font-medium text-gray-700">Min/Max:</span>
            <span className="text-gray-600 ml-1">Minimum and maximum values</span>
          </div>
          <div>
            <span className="font-medium text-gray-700">-:</span>
            <span className="text-gray-600 ml-1">Missing or undefined value</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatisticsTable;
