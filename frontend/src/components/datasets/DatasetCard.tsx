// src/components/datasets/DatasetCard.tsx

import { useState } from 'react';
import {
  FileText,
  Calendar,
  Database,
  HardDrive,
  MoreVertical,
  Eye,
  Download,
  Trash2,
  Play,
  AlertCircle,
  CheckCircle,
  Clock,
  FileSpreadsheet,
  FileJson,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import type { Dataset } from '@/types/dataset.types';

interface DatasetCardProps {
  dataset: Dataset;
  isSelected?: boolean;
  onSelect?: (id: string) => void;
  onView?: (id: string) => void;
  onDelete?: (id: string) => void;
  onDownload?: (id: string) => void;
  onAnalyze?: (id: string) => void;
}

/**
 * DatasetCard - Individual dataset card component with hover effects
 * Features: Status indicators, file type icons, action menu, selection checkbox
 */
const DatasetCard: React.FC<DatasetCardProps> = ({
  dataset,
  isSelected = false,
  onSelect,
  onView,
  onDelete,
  onDownload,
  onAnalyze,
}) => {
  const [showMenu, setShowMenu] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  // ✅ Type-safe values with defaults
  const fileSize: number = dataset.size ?? dataset.fileSize ?? 0;
  const rowCount: number = dataset.rowCount ?? dataset.totalRows ?? 0;
  const columnCount: number = dataset.columnCount ?? dataset.totalColumns ?? 0;

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  // Get file type icon
  const getFileTypeIcon = () => {
    const type = dataset.fileType.toLowerCase();
    switch (type) {
      case 'csv':
        return <FileText className="w-6 h-6 text-green-600" />;
      case 'xlsx':
      case 'xls':
        return <FileSpreadsheet className="w-6 h-6 text-blue-600" />;
      case 'json':
        return <FileJson className="w-6 h-6 text-purple-600" />;
      default:
        return <FileText className="w-6 h-6 text-gray-600" />;
    }
  };

  // Get file type color
  const getFileTypeColor = () => {
    const type = dataset.fileType.toLowerCase();
    switch (type) {
      case 'csv':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'xlsx':
      case 'xls':
        return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'json':
        return 'bg-purple-100 text-purple-700 border-purple-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  // Get status badge
  const getStatusBadge = () => {
    switch (dataset.status) {
      case 'completed':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded-full">
            <CheckCircle className="w-3 h-3" />
            <span>Ready</span>
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-1 bg-yellow-100 text-yellow-700 text-xs font-medium rounded-full">
            <Clock className="w-3 h-3 animate-spin" />
            <span>Processing</span>
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded-full">
            <AlertCircle className="w-3 h-3" />
            <span>Error</span>
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center space-x-1 px-2 py-1 bg-gray-100 text-gray-700 text-xs font-medium rounded-full">
            <Clock className="w-3 h-3" />
            <span>Pending</span>
          </span>
        );
    }
  };

  // Handle card click
  const handleCardClick = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (
      target.closest('button') ||
      target.closest('input') ||
      target.closest('.action-menu')
    ) {
      return;
    }
    onView?.(dataset.id);
  };

  return (
    <div
      className={`card group relative overflow-hidden transition-all duration-300 cursor-pointer ${
        isSelected ? 'ring-2 ring-blue-500 shadow-lg' : ''
      } ${isHovered ? 'shadow-xl -translate-y-1' : ''}`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setShowMenu(false);
      }}
      onClick={handleCardClick}
    >
      {/* Selection Checkbox */}
      {onSelect && (
        <div className="absolute top-3 left-3 z-10">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={(e) => {
              e.stopPropagation();
              onSelect(dataset.id);
            }}
            className={`w-5 h-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer transition-opacity ${
              isHovered || isSelected ? 'opacity-100' : 'opacity-0'
            }`}
          />
        </div>
      )}

      {/* Status Badge */}
      <div className="absolute top-3 right-3 z-10">{getStatusBadge()}</div>

      {/* Card Header - File Icon & Type */}
      <div className="card-header">
        <div className="flex items-center justify-between">
          <div
            className={`w-14 h-14 rounded-xl border-2 flex items-center justify-center transition-transform duration-300 ${getFileTypeColor()} ${
              isHovered ? 'scale-110' : ''
            }`}
          >
            {getFileTypeIcon()}
          </div>

          {/* Action Menu */}
          <div className="relative action-menu">
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className={`p-2 rounded-lg hover:bg-gray-100 transition-all duration-200 ${
                isHovered || showMenu ? 'opacity-100' : 'opacity-0'
              }`}
            >
              <MoreVertical className="w-5 h-5 text-gray-600" />
            </button>

            {/* Dropdown Menu */}
            {showMenu && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowMenu(false);
                  }}
                />
                <div className="absolute right-0 top-10 z-20 w-48 bg-white rounded-lg shadow-xl border border-gray-200 py-1 animate-slide-in-down">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onView?.(dataset.id);
                      setShowMenu(false);
                    }}
                    className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    <Eye className="w-4 h-4" />
                    <span>View Details</span>
                  </button>
                  {onAnalyze && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onAnalyze(dataset.id);
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Play className="w-4 h-4" />
                      <span>Run Analysis</span>
                    </button>
                  )}
                  {onDownload && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDownload(dataset.id);
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download</span>
                    </button>
                  )}
                  <div className="border-t border-gray-200 my-1"></div>
                  {onDelete && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(dataset.id);
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>Delete</span>
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Card Body - Dataset Info */}
      <div className="card-body space-y-4">
        {/* Dataset Name */}
        <div>
          <h3
            className="text-lg font-semibold text-gray-900 truncate mb-1 group-hover:text-blue-600 transition-colors"
            title={dataset.name}
          >
            {dataset.name}
          </h3>
          {dataset.description && (
            <p
              className="text-sm text-gray-600 line-clamp-2"
              title={dataset.description}
            >
              {dataset.description}
            </p>
          )}
        </div>

        {/* Dataset Metadata */}
        <div className="space-y-2">
          {/* File Type */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Type</span>
            <span
              className={`px-2 py-0.5 rounded-md font-medium uppercase text-xs ${getFileTypeColor()}`}
            >
              {dataset.fileType}
            </span>
          </div>

          {/* File Size */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2 text-gray-600">
              <HardDrive className="w-4 h-4" />
              <span>Size</span>
            </div>
            <span className="font-medium text-gray-900">
              {formatFileSize(fileSize)} {/* ✅ Now properly typed */}
            </span>
          </div>

          {/* Row Count */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2 text-gray-600">
              <Database className="w-4 h-4" />
              <span>Rows</span>
            </div>
            <span className="font-medium text-gray-900">
              {rowCount.toLocaleString()} {/* ✅ Now properly typed */}
            </span>
          </div>

          {/* Column Count */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-2 text-gray-600">
              <FileText className="w-4 h-4" />
              <span>Columns</span>
            </div>
            <span className="font-medium text-gray-900">
              {columnCount} {/* ✅ Now properly typed */}
            </span>
          </div>
        </div>
      </div>

      {/* Card Footer - Date Info */}
      <div className="card-footer">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-1">
            <Calendar className="w-3.5 h-3.5" />
            <span>
              {formatDistanceToNow(new Date(dataset.createdAt), {
                addSuffix: true,
              })}
            </span>
          </div>
          {dataset.updatedAt && dataset.updatedAt !== dataset.createdAt && (
            <span className="text-gray-400">
              Updated{' '}
              {formatDistanceToNow(new Date(dataset.updatedAt), {
                addSuffix: true,
              })}
            </span>
          )}
        </div>
      </div>

      {/* Hover Overlay Effect */}
      <div
        className={`absolute inset-0 bg-gradient-to-t from-blue-600/5 to-transparent pointer-events-none transition-opacity duration-300 ${
          isHovered ? 'opacity-100' : 'opacity-0'
        }`}
      />
    </div>
  );
};

export default DatasetCard;

// ============================================================================
// Skeleton Loading Component
// ============================================================================

export const DatasetCardSkeleton: React.FC = () => {
  return (
    <div className="card card-body space-y-4">
      {/* Header Skeleton */}
      <div className="flex items-center justify-between">
        <div className="skeleton w-14 h-14 rounded-xl"></div>
        <div className="skeleton w-16 h-6 rounded-full"></div>
      </div>

      {/* Title Skeleton */}
      <div className="space-y-2">
        <div className="skeleton h-6 w-3/4 rounded"></div>
        <div className="skeleton h-4 w-full rounded"></div>
        <div className="skeleton h-4 w-2/3 rounded"></div>
      </div>

      {/* Metadata Skeleton */}
      <div className="space-y-2">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="flex items-center justify-between">
            <div className="skeleton h-4 w-16 rounded"></div>
            <div className="skeleton h-4 w-20 rounded"></div>
          </div>
        ))}
      </div>

      {/* Footer Skeleton */}
      <div className="skeleton h-4 w-32 rounded"></div>
    </div>
  );
};

// ============================================================================
// Empty State Component
// ============================================================================

export const DatasetCardEmpty: React.FC<{ onUpload?: () => void }> = ({
  onUpload,
}) => {
  return (
    <div className="card card-body text-center py-12">
      <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <FileText className="w-8 h-8 text-gray-400" />
      </div>
      <h3 className="text-lg font-semibold text-gray-900 mb-2">
        No datasets yet
      </h3>
      <p className="text-gray-600 mb-6">
        Upload your first dataset to start analyzing your data
      </p>
      {onUpload && (
        <button onClick={onUpload} className="btn btn-primary mx-auto">
          Upload Dataset
        </button>
      )}
    </div>
  );
};
