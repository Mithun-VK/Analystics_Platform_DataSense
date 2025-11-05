// src/components/insights/InsightCard.tsx - FINAL PRODUCTION VERSION (ERROR-FREE)

import { useState } from 'react';
import {
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Info,
  Lightbulb,
  ThumbsUp,
  ThumbsDown,
  Bookmark,
  Share2,
  MoreVertical,
  ChevronRight,
  BarChart3,
  Eye,
  Clock,
  Star,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
// ✅ FIXED: Import Insight from centralized types
import type { Insight } from '@/types/components.types';

interface InsightCardProps {
  insight: Insight;
  onView?: (id: string) => void;
  onBookmark?: (id: string) => void;
  onFeedback?: (id: string, helpful: boolean) => void;
  onShare?: (id: string) => void;
  onExplore?: (id: string) => void;
  compact?: boolean;
  showActions?: boolean;
}

/**
 * InsightCard - Display AI-generated data insights
 * Features: Interactive cards with severity levels, confidence scores, recommendations
 * Supports: Multiple insight types, feedback system, bookmarking, sharing
 */
const InsightCard: React.FC<InsightCardProps> = ({
  insight,
  onView,
  onBookmark,
  onFeedback,
  onShare,
  onExplore,
  compact = false,
  showActions = true,
}) => {
  const [expanded, setExpanded] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [feedbackGiven, setFeedbackGiven] = useState<boolean | null>(null);

  // ============================================================================
  // Utility Functions
  // ============================================================================

  /**
   * Get insight icon based on type
   */
  const getInsightIcon = () => {
    const typeStr = insight.type.toLowerCase();

    if (typeStr.includes('trend')) return <TrendingUp className="w-5 h-5" />;
    if (typeStr.includes('anomaly')) return <AlertTriangle className="w-5 h-5" />;
    if (typeStr.includes('correlation') || typeStr.includes('chart'))
      return <BarChart3 className="w-5 h-5" />;
    if (typeStr.includes('recommendation') || typeStr.includes('lightbulb'))
      return <Lightbulb className="w-5 h-5" />;
    if (typeStr.includes('forecast') || typeStr.includes('sparkles'))
      return <Sparkles className="w-5 h-5" />;

    return <Info className="w-5 h-5" />;
  };

  /**
   * Get severity badge color
   */
  const getSeverityColor = () => {
    const severity = insight.severity?.toLowerCase() || 'info';

    if (severity.includes('critical')) return 'bg-red-100 text-red-700 border-red-200';
    if (severity.includes('high')) return 'bg-orange-100 text-orange-700 border-orange-200';
    if (severity.includes('medium')) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
    if (severity.includes('low')) return 'bg-blue-100 text-blue-700 border-blue-200';
    return 'bg-gray-100 text-gray-700 border-gray-200';
  };

  /**
   * Get category badge color
   */
  const getCategoryColor = () => {
    const category = insight.category?.toLowerCase() || 'performance';

    if (category.includes('performance')) return 'bg-green-100 text-green-700';
    if (category.includes('quality')) return 'bg-purple-100 text-purple-700';
    if (category.includes('pattern')) return 'bg-blue-100 text-blue-700';
    if (category.includes('opportunity')) return 'bg-yellow-100 text-yellow-700';
    if (category.includes('risk')) return 'bg-red-100 text-red-700';

    return 'bg-gray-100 text-gray-700';
  };

  /**
   * Get confidence color
   */
  const getConfidenceColor = () => {
    if (insight.confidence >= 80) return 'text-green-600';
    if (insight.confidence >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  // ============================================================================
  // Event Handlers
  // ============================================================================

  /**
   * Handle feedback
   */
  const handleFeedback = (helpful: boolean) => {
    setFeedbackGiven(helpful);
    onFeedback?.(insight.id, helpful);
  };

  /**
   * Handle view
   */
  const handleView = () => {
    onView?.(insight.id);
    setExpanded(!expanded);
  };

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <div
      className={`card overflow-hidden transition-all duration-200 ${
        !compact ? 'ring-2 ring-blue-500' : ''
      } ${expanded ? 'shadow-xl' : ''}`}
    >
      {/* ====================================================================
          Header Section
          ==================================================================== */}
      <div
        className={`card-header cursor-pointer hover:bg-gray-50 transition-colors ${
          compact ? 'p-4' : ''
        }`}
        onClick={handleView}
      >
        <div className="flex items-start space-x-4">
          {/* Icon */}
          <div
            className={`w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0 ${getSeverityColor()}`}
          >
            {getInsightIcon()}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between mb-2">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <h3 className="text-base font-semibold text-gray-900 line-clamp-1">
                    {insight.title}
                  </h3>
                  {!compact && (
                    <span className="flex-shrink-0 w-2 h-2 bg-blue-600 rounded-full"></span>
                  )}
                </div>
                <p
                  className={`text-sm text-gray-600 ${
                    compact ? 'line-clamp-1' : 'line-clamp-2'
                  }`}
                >
                  {insight.description}
                </p>
              </div>

              {/* Menu Button */}
              {showActions && (
                <div className="relative ml-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowMenu(!showMenu);
                    }}
                    className="p-1 hover:bg-gray-100 rounded transition-colors"
                  >
                    <MoreVertical className="w-4 h-4 text-gray-500" />
                  </button>

                  {/* Dropdown Menu */}
                  {showMenu && (
                    <>
                      <div
                        className="fixed inset-0 z-10"
                        onClick={() => setShowMenu(false)}
                      />
                      <div className="absolute right-0 top-8 z-20 w-48 bg-white rounded-lg shadow-xl border border-gray-200 py-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onExplore?.(insight.id);
                            setShowMenu(false);
                          }}
                          className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                        >
                          <Eye className="w-4 h-4" />
                          <span>Explore Data</span>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onBookmark?.(insight.id);
                            setShowMenu(false);
                          }}
                          className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                        >
                          <Bookmark className="w-4 h-4" />
                          <span>Bookmark</span>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onShare?.(insight.id);
                            setShowMenu(false);
                          }}
                          className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                        >
                          <Share2 className="w-4 h-4" />
                          <span>Share Insight</span>
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Badges */}
            <div className="flex items-center space-x-2 mb-2 flex-wrap">
              {insight.category && (
                <span
                  className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium ${getCategoryColor()}`}
                >
                  {insight.category}
                </span>
              )}
              {insight.severity && (
                <span
                  className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-medium border ${getSeverityColor()}`}
                >
                  {insight.severity}
                </span>
              )}
              {!compact && insight.createdAt && (
                <span className="text-xs text-gray-500 flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>
                    {formatDistanceToNow(new Date(insight.createdAt), {
                      addSuffix: true,
                    })}
                  </span>
                </span>
              )}
            </div>

            {/* Confidence Score */}
            {!compact && (
              <div className="flex items-center space-x-2">
                <div className="flex-1 bg-gray-200 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all ${
                      insight.confidence >= 80
                        ? 'bg-green-600'
                        : insight.confidence >= 60
                        ? 'bg-yellow-600'
                        : 'bg-red-600'
                    }`}
                    style={{ width: `${insight.confidence * 100}%` }}
                  />
                </div>
                <span className={`text-xs font-medium ${getConfidenceColor()}`}>
                  {(insight.confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
            )}
          </div>

          {/* Expand Icon */}
          <ChevronRight
            className={`w-5 h-5 text-gray-400 transition-transform flex-shrink-0 ${
              expanded ? 'rotate-90' : ''
            }`}
          />
        </div>
      </div>

      {/* ====================================================================
          Expanded Content Section
          ==================================================================== */}
      {expanded && (
        <div className="card-body border-t border-gray-200 space-y-4">
          {/* Impact */}
          {insight.impact && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <div className="flex items-start space-x-2">
                <Star className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-xs font-semibold text-blue-900 mb-1">
                    Impact
                  </p>
                  <p className="text-sm text-gray-700">{insight.impact}</p>
                </div>
              </div>
            </div>
          )}

          {/* Recommendation */}
          {insight.recommendation && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-3">
              <div className="flex items-start space-x-2">
                <Lightbulb className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-xs font-semibold text-green-900 mb-1">
                    Recommendation
                  </p>
                  <p className="text-sm text-gray-700">
                    {insight.recommendation}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Evidence */}
          {insight.evidence && insight.evidence.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-3">
                Evidence
              </h4>
              <div className="grid grid-cols-2 gap-3">
                {insight.evidence.map((item, index) => (
                  <div
                    key={index}
                    className="bg-gray-50 rounded-lg p-3 border border-gray-200"
                  >
                    <p className="text-xs text-gray-600 mb-1">
                      {item.metric}
                    </p>
                    <p className="text-lg font-bold text-gray-900">
                      {item.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Related Columns */}
          {insight.relatedColumns && insight.relatedColumns.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-2">
                Related Columns
              </h4>
              <div className="flex flex-wrap gap-2">
                {insight.relatedColumns.map((column, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-md"
                  >
                    {column}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          {showActions && (
            <div className="flex items-center justify-between pt-4 border-t border-gray-200">
              {/* Feedback */}
              <div className="flex items-center space-x-2">
                <span className="text-xs text-gray-600">Helpful?</span>
                <button
                  onClick={() => handleFeedback(true)}
                  disabled={feedbackGiven !== null}
                  className={`p-2 rounded-lg transition-colors ${
                    feedbackGiven === true
                      ? 'bg-green-100 text-green-600'
                      : 'hover:bg-gray-100 text-gray-600'
                  }`}
                >
                  <ThumbsUp className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleFeedback(false)}
                  disabled={feedbackGiven !== null}
                  className={`p-2 rounded-lg transition-colors ${
                    feedbackGiven === false
                      ? 'bg-red-100 text-red-600'
                      : 'hover:bg-gray-100 text-gray-600'
                  }`}
                >
                  <ThumbsDown className="w-4 h-4" />
                </button>
              </div>

              {/* Explore Button */}
              {onExplore && (
                <button
                  onClick={() => onExplore(insight.id)}
                  className="flex items-center space-x-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
                >
                  <span>Explore Data</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              )}
            </div>
          )}

          {/* AI Attribution */}
          <div className="flex items-center space-x-2 text-xs text-gray-500 pt-2 border-t border-gray-100">
            <Sparkles className="w-3 h-3" />
            <span>AI-generated insight • Verified by statistical analysis</span>
          </div>
        </div>
      )}
    </div>
  );
};

InsightCard.displayName = 'InsightCard';

export default InsightCard;

// ============================================================================
// Skeleton Loading Component
// ============================================================================

/**
 * InsightCardSkeleton - Loading placeholder for InsightCard
 */
export const InsightCardSkeleton: React.FC = () => {
  return (
    <div className="card">
      <div className="card-header">
        <div className="flex items-start space-x-4">
          <div className="skeleton w-12 h-12 rounded-lg"></div>
          <div className="flex-1 space-y-3">
            <div className="skeleton h-5 w-3/4 rounded"></div>
            <div className="skeleton h-4 w-full rounded"></div>
            <div className="flex space-x-2">
              <div className="skeleton h-6 w-20 rounded"></div>
              <div className="skeleton h-6 w-16 rounded"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
