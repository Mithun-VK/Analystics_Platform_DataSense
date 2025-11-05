// src/components/visualizations/ChartTypeSelector.tsx

import { useState, useCallback, useMemo } from 'react';
import {
  BarChart3,
  LineChart,
  PieChart,
  Activity,
  TrendingUp,
  TrendingDown,
  Radar,
  Layers,
  ChevronDown,
  Check,
  Info,
  Grid3x3,
  List,
} from 'lucide-react';

export type ChartType =
  | 'line'
  | 'bar'
  | 'area'
  | 'pie'
  | 'scatter'
  | 'doughnut'
  | 'radar'
  | 'mixed';

interface ChartTypeOption {
  type: ChartType;
  label: string;
  icon: React.ElementType;
  description: string;
  recommended: string[];
  examples: string;
  disabled?: boolean;
  badge?: string;
  trend?: 'up' | 'down';
  category: 'basic' | 'advanced';
}

interface ChartTypeSelectorProps {
  selectedType: ChartType;
  onChange: (type: ChartType) => void;
  mode?: 'grid' | 'dropdown' | 'list';
  showDescriptions?: boolean;
  disabled?: boolean;
  className?: string;
  showTrending?: boolean;
}

// ✅ Chart type options with all metadata
const CHART_TYPES: ChartTypeOption[] = [
  {
    type: 'bar',
    label: 'Bar Chart',
    icon: BarChart3,
    description: 'Compare values across categories with vertical bars',
    recommended: ['categorical data', 'comparisons', 'rankings', 'performance'],
    examples: 'Sales by region, survey results, market segments',
    category: 'basic',
    trend: 'up',
  },
  {
    type: 'line',
    label: 'Line Chart',
    icon: LineChart,
    description: 'Show trends and changes over time with connected points',
    recommended: ['time series', 'trends', 'continuous data', 'monitoring'],
    examples: 'Stock prices, temperature changes, website traffic',
    category: 'basic',
    trend: 'up',
  },
  {
    type: 'area',
    label: 'Area Chart',
    icon: Activity,
    description: 'Display cumulative values and areas under curves',
    recommended: ['time series', 'volume', 'cumulative data', 'stacked metrics'],
    examples: 'Website traffic, revenue growth, resource usage',
    category: 'basic',
  },
  {
    type: 'pie',
    label: 'Pie Chart',
    icon: PieChart,
    description: 'Show proportions and percentages as slices',
    recommended: ['proportions', 'percentages', 'composition', 'distribution'],
    examples: 'Market share, budget allocation, voter distribution',
    category: 'basic',
  },
  {
    type: 'scatter',
    label: 'Scatter Plot',
    icon: Grid3x3,
    description: 'Visualize relationships and correlations between variables',
    recommended: ['correlations', 'distributions', 'outliers', 'bivariate analysis'],
    examples: 'Height vs weight, price vs demand, age vs income',
    category: 'basic',
  },
  {
    type: 'doughnut',
    label: 'Doughnut Chart',
    icon: PieChart,
    description: 'Similar to pie chart with a hollow center for better layout',
    recommended: ['proportions', 'percentages', 'multiple series', 'composition'],
    examples: 'Resource allocation, completion rates, department breakdown',
    category: 'advanced',
    badge: 'Pro',
    disabled: true,
  },
  {
    type: 'radar',
    label: 'Radar Chart',
    icon: Radar,
    description: 'Compare multiple variables in a circular format',
    recommended: [
      'multivariate data',
      'performance metrics',
      'skill profiles',
      'capabilities',
    ],
    examples: 'Skill assessments, product comparisons, team capabilities',
    category: 'advanced',
    badge: 'Pro',
    disabled: true,
  },
  {
    type: 'mixed',
    label: 'Mixed Chart',
    icon: Layers,
    description: 'Combine multiple chart types for complex data analysis',
    recommended: [
      'complex data',
      'multiple metrics',
      'advanced analysis',
      'hybrid visualization',
    ],
    examples: 'Revenue and profit trends, dual-axis analysis',
    category: 'advanced',
    badge: 'Pro',
    disabled: true,
  },
];

/**
 * ChartTypeSelector - Production-grade visual selector for chart types
 * Features: Grid/dropdown/list modes, descriptions, recommendations, trending
 * Supports: All major chart types with visual previews, guidance, and metrics
 *
 * @example
 * <ChartTypeSelector
 *   selectedType="bar"
 *   onChange={handleChartTypeChange}
 *   mode="grid"
 *   showTrending
 * />
 */
const ChartTypeSelector: React.FC<ChartTypeSelectorProps> = ({
  selectedType,
  onChange,
  mode = 'grid',
  showDescriptions = true,
  disabled = false,
  className = '',
  showTrending = false,
}) => {
  // ✅ State management
  const [isOpen, setIsOpen] = useState(false);
  const [hoveredType, setHoveredType] = useState<ChartType | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>(
    mode === 'grid' ? 'grid' : 'list'
  );

  // ✅ Get selected option
  const selectedOption = useMemo(
    () => CHART_TYPES.find((opt) => opt.type === selectedType),
    [selectedType]
  );

  // ✅ Get category breakdown
  const categoryStats = useMemo(() => {
    const basic = CHART_TYPES.filter((c) => c.category === 'basic').length;
    const advanced = CHART_TYPES.filter((c) => c.category === 'advanced').length;
    return { basic, advanced };
  }, []);

  // ✅ Handle selection
  const handleSelect = useCallback(
    (type: ChartType, optionDisabled?: boolean) => {
      if (disabled || optionDisabled) return;
      onChange(type);
      setIsOpen(false);
    },
    [disabled, onChange]
  );

  // ✅ Get trend icon
  const getTrendIcon = (trend?: 'up' | 'down') => {
    if (!showTrending) return null;
    if (trend === 'up') {
      return <TrendingUp className="w-4 h-4 text-green-600" />;
    }
    if (trend === 'down') {
      return <TrendingDown className="w-4 h-4 text-red-600" />;
    }
    return null;
  };

  // ✅ Grid View
  if (mode === 'grid') {
    return (
      <div className={className}>
        {/* View Toggle */}
        {showDescriptions && (
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                Select Chart Type
              </h3>
              <p className="text-sm text-gray-600 mt-1">
                {categoryStats.basic} basic • {categoryStats.advanced} advanced
              </p>
            </div>
            <div className="flex items-center space-x-2 bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'grid'
                    ? 'bg-white shadow-sm text-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Grid view"
              >
                <Grid3x3 className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'list'
                    ? 'bg-white shadow-sm text-blue-600'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
                title="List view"
              >
                <List className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Grid Layout */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {CHART_TYPES.map((option) => {
            const Icon = option.icon;
            const isSelected = selectedType === option.type;
            const isHovered = hoveredType === option.type;

            return (
              <button
                key={option.type}
                onClick={() => handleSelect(option.type, option.disabled)}
                onMouseEnter={() => setHoveredType(option.type)}
                onMouseLeave={() => setHoveredType(null)}
                disabled={disabled || option.disabled}
                className={`relative group p-6 rounded-xl border-2 transition-all duration-200 ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 shadow-lg'
                    : 'border-gray-200 hover:border-blue-300 hover:shadow-md'
                } ${
                  disabled || option.disabled
                    ? 'opacity-50 cursor-not-allowed'
                    : 'cursor-pointer'
                }`}
              >
                {/* Badge */}
                {option.badge && (
                  <span className="absolute top-3 right-3 px-2 py-1 bg-purple-100 text-purple-700 text-xs font-semibold rounded-full">
                    {option.badge}
                  </span>
                )}

                {/* Trending Indicator */}
                {getTrendIcon(option.trend) && (
                  <div className="absolute top-3 right-12">
                    {getTrendIcon(option.trend)}
                  </div>
                )}

                {/* Icon */}
                <div className="flex justify-center mb-3">
                  <div
                    className={`w-16 h-16 rounded-lg flex items-center justify-center transition-all duration-200 ${
                      isSelected
                        ? 'bg-blue-600 scale-110'
                        : 'bg-gray-100 group-hover:bg-blue-100'
                    }`}
                  >
                    <Icon
                      className={`w-8 h-8 ${
                        isSelected
                          ? 'text-white'
                          : 'text-gray-600 group-hover:text-blue-600'
                      }`}
                    />
                  </div>
                </div>

                {/* Label */}
                <h3
                  className={`text-center font-semibold mb-2 ${
                    isSelected ? 'text-blue-700' : 'text-gray-900'
                  }`}
                >
                  {option.label}
                </h3>

                {/* Description */}
                {showDescriptions && (
                  <p className="text-xs text-center text-gray-600 leading-relaxed">
                    {option.description}
                  </p>
                )}

                {/* Selected Indicator */}
                {isSelected && (
                  <div className="absolute top-3 left-3">
                    <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" />
                    </div>
                  </div>
                )}

                {/* Hover Info */}
                {isHovered && showDescriptions && (
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 z-50 w-64 bg-gray-900 text-white text-xs rounded-lg p-3 shadow-xl pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                    <p className="font-semibold mb-2">{option.label}</p>
                    <p className="mb-2 text-gray-300">{option.description}</p>
                    <p className="text-gray-400 text-xs mb-2">
                      <strong>Best for:</strong> {option.recommended.join(', ')}
                    </p>
                    <p className="text-gray-400 text-xs">
                      <strong>Examples:</strong> {option.examples}
                    </p>
                    <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 translate-y-1/2 rotate-45 w-2 h-2 bg-gray-900"></div>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  // ✅ Dropdown View
  if (mode === 'dropdown') {
    const Icon = selectedOption?.icon || BarChart3;

    return (
      <div className={`relative ${className}`}>
        <button
          onClick={() => setIsOpen(!isOpen)}
          disabled={disabled}
          className={`w-full flex items-center justify-between px-4 py-3 border-2 rounded-lg transition-all ${
            isOpen
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
        >
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
              <Icon className="w-5 h-5 text-blue-600" />
            </div>
            <div className="text-left">
              <p className="text-sm font-medium text-gray-900">
                {selectedOption?.label || 'Select Chart Type'}
              </p>
              {showDescriptions && selectedOption && (
                <p className="text-xs text-gray-600">
                  {selectedOption.description}
                </p>
              )}
            </div>
          </div>
          <ChevronDown
            className={`w-5 h-5 text-gray-500 transition-transform ${
              isOpen ? 'rotate-180' : ''
            }`}
          />
        </button>

        {isOpen && (
          <>
            <div
              className="fixed inset-0 z-10"
              onClick={() => setIsOpen(false)}
            />
            <div className="absolute top-full mt-2 w-full bg-white rounded-lg shadow-xl border border-gray-200 z-20 max-h-96 overflow-y-auto animate-slide-in-down">
              {CHART_TYPES.map((option) => {
                const Icon = option.icon;
                const isSelected = selectedType === option.type;

                return (
                  <button
                    key={option.type}
                    onClick={() => handleSelect(option.type, option.disabled)}
                    disabled={option.disabled}
                    className={`w-full flex items-start space-x-3 px-4 py-3 transition-colors ${
                      isSelected
                        ? 'bg-blue-50 border-l-4 border-blue-600'
                        : 'hover:bg-gray-50 border-l-4 border-transparent'
                    } ${option.disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${
                        isSelected ? 'bg-blue-600' : 'bg-gray-100'
                      }`}
                    >
                      <Icon
                        className={`w-5 h-5 ${
                          isSelected ? 'text-white' : 'text-gray-600'
                        }`}
                      />
                    </div>
                    <div className="flex-1 text-left">
                      <div className="flex items-center justify-between mb-1">
                        <p
                          className={`text-sm font-medium ${
                            isSelected ? 'text-blue-700' : 'text-gray-900'
                          }`}
                        >
                          {option.label}
                        </p>
                        <div className="flex items-center space-x-2">
                          {getTrendIcon(option.trend)}
                          {option.badge && (
                            <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs font-semibold rounded-full">
                              {option.badge}
                            </span>
                          )}
                          {isSelected && (
                            <Check className="w-4 h-4 text-blue-600" />
                          )}
                        </div>
                      </div>
                      {showDescriptions && (
                        <p className="text-xs text-gray-600">
                          {option.description}
                        </p>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </>
        )}
      </div>
    );
  }

  // ✅ List View
  return (
    <div className={`space-y-3 ${className}`}>
      {/* Header */}
      {showDescriptions && (
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Select Chart Type
          </h3>
          <p className="text-sm text-gray-600 mt-1">
            Choose the best visualization for your data
          </p>
        </div>
      )}

      {/* List Items */}
      {CHART_TYPES.map((option) => {
        const Icon = option.icon;
        const isSelected = selectedType === option.type;

        return (
          <button
            key={option.type}
            onClick={() => handleSelect(option.type, option.disabled)}
            disabled={disabled || option.disabled}
            className={`w-full flex items-start space-x-4 p-4 rounded-lg border-2 transition-all ${
              isSelected
                ? 'border-blue-500 bg-blue-50 shadow-md'
                : 'border-gray-200 hover:border-blue-300 hover:shadow-sm'
            } ${
              disabled || option.disabled
                ? 'opacity-50 cursor-not-allowed'
                : 'cursor-pointer'
            }`}
          >
            {/* Icon */}
            <div
              className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${
                isSelected ? 'bg-blue-600' : 'bg-gray-100'
              }`}
            >
              <Icon
                className={`w-6 h-6 ${isSelected ? 'text-white' : 'text-gray-600'}`}
              />
            </div>

            {/* Content */}
            <div className="flex-1 text-left">
              <div className="flex items-center justify-between mb-2">
                <h3
                  className={`text-base font-semibold ${
                    isSelected ? 'text-blue-700' : 'text-gray-900'
                  }`}
                >
                  {option.label}
                </h3>
                <div className="flex items-center space-x-2">
                  {getTrendIcon(option.trend)}
                  {option.badge && (
                    <span className="px-2 py-1 bg-purple-100 text-purple-700 text-xs font-semibold rounded-full">
                      {option.badge}
                    </span>
                  )}
                  {isSelected && (
                    <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                      <Check className="w-4 h-4 text-white" />
                    </div>
                  )}
                </div>
              </div>

              {showDescriptions && (
                <>
                  <p className="text-sm text-gray-600 mb-2">
                    {option.description}
                  </p>
                  <div className="flex flex-wrap gap-2 mb-2">
                    {option.recommended.map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-md"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                  <p className="text-xs text-gray-500">
                    <strong>Examples:</strong> {option.examples}
                  </p>
                </>
              )}
            </div>
          </button>
        );
      })}

      {/* Info Section */}
      {showDescriptions && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-6">
          <div className="flex items-start space-x-3">
            <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1 text-sm text-gray-700">
              <p className="font-semibold mb-1">How to Choose the Right Chart</p>
              <ul className="text-xs text-gray-600 space-y-1 list-disc list-inside">
                <li>
                  <strong>Bar/Line:</strong> For comparing values or showing trends
                </li>
                <li>
                  <strong>Pie/Doughnut:</strong> For showing proportions of a whole
                </li>
                <li>
                  <strong>Scatter:</strong> For displaying relationships between variables
                </li>
                <li>
                  <strong>Radar:</strong> For comparing multiple variables across categories
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

ChartTypeSelector.displayName = 'ChartTypeSelector';

export default ChartTypeSelector;

// ============================================================================
// Chart Type Utilities
// ============================================================================

/**
 * Get detailed information about a chart type
 * @param type - The chart type
 * @returns Chart type option or undefined
 */
export const getChartTypeInfo = (
  type: ChartType
): ChartTypeOption | undefined => {
  return CHART_TYPES.find((opt) => opt.type === type);
};

/**
 * Get recommended chart type based on data characteristics
 * @param dataType - The type of data
 * @returns Recommended chart type
 */
export const getRecommendedChartType = (
  dataType: 'categorical' | 'time-series' | 'proportional' | 'correlation'
): ChartType => {
  switch (dataType) {
    case 'categorical':
      return 'bar';
    case 'time-series':
      return 'line';
    case 'proportional':
      return 'pie';
    case 'correlation':
      return 'scatter';
    default:
      return 'bar';
  }
};

/**
 * Get all available chart types filtered by category
 * @param category - 'basic' or 'advanced'
 * @returns Filtered chart type options
 */
export const getChartsByCategory = (
  category: 'basic' | 'advanced'
): ChartTypeOption[] => {
  return CHART_TYPES.filter((opt) => opt.category === category);
};

/**
 * Check if a chart type is available (not disabled)
 * @param type - The chart type
 * @returns True if available, false otherwise
 */
export const isChartTypeAvailable = (type: ChartType): boolean => {
  const option = CHART_TYPES.find((opt) => opt.type === type);
  return option ? !option.disabled : false;
};

/**
 * Get all trending chart types
 * @returns Chart types with trending indicator
 */
export const getTrendingChartTypes = (): ChartTypeOption[] => {
  return CHART_TYPES.filter((opt) => opt.trend === 'up');
};
