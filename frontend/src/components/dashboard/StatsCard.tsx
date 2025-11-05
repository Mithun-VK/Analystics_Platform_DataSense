// src/components/dashboard/StatsCard.tsx

import { useEffect, useState } from 'react';
import type { LucideIcon } from 'lucide-react';
import {
  TrendingUp,
  TrendingDown,
  Minus,
  MoreVertical,
  Database,
  Upload,
  HardDrive,
  BarChart3,
} from 'lucide-react';

interface StatsCardProps {
  title: string;
  value: number | string;
  icon: LucideIcon;
  iconColor?: string;
  iconBgColor?: string;
  change?: number;
  changeLabel?: string;
  prefix?: string;
  suffix?: string;
  loading?: boolean;
  trend?: 'up' | 'down' | 'neutral';
  subtitle?: string;
  formatValue?: (value: string | number) => string;
  onClick?: () => void;
  showMenu?: boolean;
  onMenuAction?: (action: string) => void;
  animated?: boolean;
  className?: string;
}

/**
 * StatsCard - Reusable card component for displaying user statistics
 * Features animated counter, trend indicators, and customizable styling
 * @param title - Card title/label
 * @param value - Numeric or string value to display
 * @param icon - Lucide icon component
 * @param iconColor - Icon color (default: text-blue-600)
 * @param iconBgColor - Icon background color (default: bg-blue-100)
 * @param change - Percentage change value
 * @param changeLabel - Label for change indicator (default: "vs last month")
 * @param prefix - Value prefix (e.g., "$")
 * @param suffix - Value suffix (e.g., "K", "M")
 * @param loading - Show loading skeleton
 * @param trend - Trend direction indicator
 * @param subtitle - Additional descriptive text
 * @param formatValue - Custom value formatter function
 * @param onClick - Click handler for card
 * @param showMenu - Show menu dropdown
 * @param onMenuAction - Menu action handler
 * @param animated - Enable animated counter (default: true)
 * @param className - Additional CSS classes
 */
const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  icon: Icon,
  iconColor = 'text-blue-600',
  iconBgColor = 'bg-blue-100',
  change,
  changeLabel = 'vs last month',
  prefix = '',
  suffix = '',
  loading = false,
  trend,
  subtitle,
  formatValue,
  onClick,
  showMenu = false,
  onMenuAction,
  animated = true,
  className = '',
}) => {
  const [displayValue, setDisplayValue] = useState<number>(0);
  const [showMenuDropdown, setShowMenuDropdown] = useState(false);

  // Animated counter effect
  useEffect(() => {
    if (!animated || typeof value !== 'number' || loading) return;

    const duration = 1500; // Animation duration in ms
    const steps = 60;
    const increment = value / steps;
    const stepDuration = duration / steps;

    let currentStep = 0;
    const timer = setInterval(() => {
      currentStep++;
      if (currentStep >= steps) {
        setDisplayValue(value);
        clearInterval(timer);
      } else {
        setDisplayValue(Math.floor(increment * currentStep));
      }
    }, stepDuration);

    return () => clearInterval(timer);
  }, [value, animated, loading]);

  // Format displayed value
  const getFormattedValue = (): string => {
    if (formatValue) {
      return formatValue(
        animated && typeof value === 'number' ? displayValue : value
      );
    }

    const displayNum = animated && typeof value === 'number' ? displayValue : value;

    if (typeof displayNum === 'number') {
      return `${prefix}${formatNumber(displayNum)}${suffix}`;
    }

    return `${prefix}${displayNum}${suffix}`;
  };

  // Number formatting helper
  const formatNumber = (num: number): string => {
    if (num >= 1_000_000) {
      return (num / 1_000_000).toFixed(1) + 'M';
    }
    if (num >= 1_000) {
      return num.toLocaleString();
    }
    return num.toString();
  };

  // Determine trend based on change value if trend not explicitly provided
  const getTrend = (): 'up' | 'down' | 'neutral' => {
    if (trend) return trend;
    if (change === undefined || change === null) return 'neutral';
    if (change > 0) return 'up';
    if (change < 0) return 'down';
    return 'neutral';
  };

  const currentTrend = getTrend();

  // Get trend icon
  const getTrendIcon = () => {
    switch (currentTrend) {
      case 'up':
        return <TrendingUp className="w-4 h-4" />;
      case 'down':
        return <TrendingDown className="w-4 h-4" />;
      default:
        return <Minus className="w-4 h-4" />;
    }
  };

  // Get trend color classes
  const getTrendColorClasses = () => {
    switch (currentTrend) {
      case 'up':
        return 'text-green-700 bg-green-100';
      case 'down':
        return 'text-red-700 bg-red-100';
      default:
        return 'text-gray-700 bg-gray-100';
    }
  };

  // Loading skeleton
  if (loading) {
    return (
      <div className={`card card-body space-y-4 ${className}`}>
        <div className="flex items-start justify-between">
          <div className="skeleton h-4 w-24 rounded"></div>
          <div className="skeleton h-10 w-10 rounded-lg"></div>
        </div>
        <div className="space-y-2">
          <div className="skeleton h-8 w-32 rounded"></div>
          <div className="skeleton h-4 w-20 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`card card-body group hover:shadow-xl transition-all duration-300 ${
        onClick ? 'cursor-pointer' : ''
      } ${className}`}
      onClick={onClick}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-sm font-medium text-gray-600 mb-1">{title}</h3>
          {subtitle && <p className="text-xs text-gray-500 mt-1">{subtitle}</p>}
        </div>

        {/* Icon and Menu */}
        <div className="flex items-center space-x-2">
          <div
            className={`${iconBgColor} ${iconColor} p-2.5 rounded-lg group-hover:scale-110 transition-transform duration-300`}
          >
            <Icon className="w-5 h-5" />
          </div>

          {showMenu && (
            <div className="relative">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowMenuDropdown(!showMenuDropdown);
                }}
                className="p-1.5 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <MoreVertical className="w-4 h-4 text-gray-500" />
              </button>

              {showMenuDropdown && (
                <>
                  <div
                    className="fixed inset-0 z-10"
                    onClick={() => setShowMenuDropdown(false)}
                  />
                  <div className="absolute right-0 top-8 z-20 w-48 bg-white rounded-lg shadow-xl border border-gray-200 py-1 animate-slide-in-down">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onMenuAction?.('view');
                        setShowMenuDropdown(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      View Details
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onMenuAction?.('export');
                        setShowMenuDropdown(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      Export Data
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onMenuAction?.('refresh');
                        setShowMenuDropdown(false);
                      }}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      Refresh
                    </button>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Value */}
      <div className="space-y-3">
        <div className="flex items-baseline space-x-2">
          <h2 className="text-3xl font-bold text-gray-900 tracking-tight">
            {getFormattedValue()}
          </h2>

          {/* Change Indicator */}
          {change !== undefined && change !== null && (
            <span
              className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-xs font-medium ${getTrendColorClasses()}`}
            >
              {getTrendIcon()}
              <span>{Math.abs(change)}%</span>
            </span>
          )}
        </div>

        {/* Change Label */}
        {change !== undefined &&
          change !== null &&
          changeLabel && (
            <div className="flex items-center text-xs text-gray-500 pt-2 border-t border-gray-100">
              <span>
                <span
                  className={
                    currentTrend === 'up'
                      ? 'text-green-600'
                      : currentTrend === 'down'
                        ? 'text-red-600'
                        : 'text-gray-600'
                  }
                >
                  {change > 0 ? '+' : ''}
                  {change}%
                </span>{' '}
                {changeLabel}
              </span>
            </div>
          )}
      </div>
    </div>
  );
};

export default StatsCard;

// ============================================================================
// Preset StatsCard Variants
// ============================================================================

/**
 * DatasetStatsCard - Pre-configured card for dataset statistics
 */
export const DatasetStatsCard: React.FC<{
  count: number;
  change?: number;
  onClick?: () => void;
}> = ({ count, change, onClick }) => {
  return (
    <StatsCard
      title="Total Datasets"
      value={count}
      icon={Database}
      iconColor="text-blue-600"
      iconBgColor="bg-blue-100"
      change={change}
      onClick={onClick}
    />
  );
};

/**
 * UploadStatsCard - Pre-configured card for upload statistics
 */
export const UploadStatsCard: React.FC<{
  count: number;
  change?: number;
  onClick?: () => void;
}> = ({ count, change, onClick }) => {
  return (
    <StatsCard
      title="Uploads This Month"
      value={count}
      icon={Upload}
      iconColor="text-green-600"
      iconBgColor="bg-green-100"
      change={change}
      onClick={onClick}
    />
  );
};

/**
 * StorageStatsCard - Pre-configured card for storage statistics
 */
export const StorageStatsCard: React.FC<{
  sizeInMB: number;
  change?: number;
  onClick?: () => void;
}> = ({ sizeInMB, change, onClick }) => {
  // Properly typed formatter function
  const formatStorage = (value: string | number): string => {
    // Ensure we're working with a number
    const mb = typeof value === 'string' ? parseFloat(value) : value;

    if (isNaN(mb)) return '0 MB';

    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb.toFixed(0)} MB`;
  };

  return (
    <StatsCard
      title="Storage Used"
      value={sizeInMB}
      icon={HardDrive}
      iconColor="text-purple-600"
      iconBgColor="bg-purple-100"
      change={change}
      formatValue={formatStorage}
      onClick={onClick}
      animated={false}
    />
  );
};

/**
 * AnalysisStatsCard - Pre-configured card for analysis statistics
 */
export const AnalysisStatsCard: React.FC<{
  count: number;
  change?: number;
  onClick?: () => void;
}> = ({ count, change, onClick }) => {
  return (
    <StatsCard
      title="Analyses Completed"
      value={count}
      icon={BarChart3}
      iconColor="text-orange-600"
      iconBgColor="bg-orange-100"
      change={change}
      onClick={onClick}
    />
  );
};
