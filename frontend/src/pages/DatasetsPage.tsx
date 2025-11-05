// src/pages/DatasetsPage.tsx

/**
 * DatasetsPage - Modern Dataset Management Interface
 * ✅ ENHANCED: Clean UI, smooth animations, dark mode, better organization
 * Features: Search, filter, sort, bulk actions, grid/list views, responsive design
 */

import { useState, useMemo, useCallback, memo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Plus,
  Upload,
  Search,
  Filter,
  Grid3x3,
  List,
  MoreVertical,
  Trash2,
  Eye,
  Download,
  Share2,
  Clock,
  HardDrive,
  Database,
  AlertCircle,
  CheckCircle,
  BarChart3,
  ArrowUpDown,
  X,
  Loader,
} from 'lucide-react';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import DataUpload from '@/components/datasets/DataUpload';
import Button from '@/components/shared/Button';
import { ConfirmModal } from '@/components/shared/Modal';
import { Skeleton } from '@/components/shared/Loading';
import { useDatasets } from '@/hooks/useDatasets';
import { useDebounce } from '@/hooks/useDebounce';
import { uiStore } from '@/store/uiStore';
import { formatDistanceToNow } from 'date-fns';

// ============================================================================
// Type Definitions
// ============================================================================

interface DatasetItem {
  id: string;
  name: string;
  description?: string;
  fileSize?: number;
  totalRows?: number;
  totalColumns?: number;
  createdAt: string;
  updatedAt: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  tags?: string[];
}

interface StatusInfo {
  icon: React.ElementType;
  color: string;
  bgColor: string;
  label: string;
}

type StatusMap = Record<
  'pending' | 'processing' | 'completed' | 'failed',
  StatusInfo
>;

type ViewMode = 'grid' | 'list';
type SortBy = 'name' | 'date' | 'size';
type SortOrder = 'asc' | 'desc';

// ============================================================================
// Constants
// ============================================================================

const STATUS_MAP: StatusMap = {
  pending: {
    icon: Clock,
    color: 'text-yellow-600 dark:text-yellow-400',
    bgColor: 'bg-yellow-100 dark:bg-yellow-900/30',
    label: 'Pending',
  },
  processing: {
    icon: Loader,
    color: 'text-blue-600 dark:text-blue-400',
    bgColor: 'bg-blue-100 dark:bg-blue-900/30',
    label: 'Processing',
  },
  completed: {
    icon: CheckCircle,
    color: 'text-green-600 dark:text-green-400',
    bgColor: 'bg-green-100 dark:bg-green-900/30',
    label: 'Ready',
  },
  failed: {
    icon: AlertCircle,
    color: 'text-red-600 dark:text-red-400',
    bgColor: 'bg-red-100 dark:bg-red-900/30',
    label: 'Failed',
  },
};

// ============================================================================
// Memoized Sub-Components
// ============================================================================

/**
 * ✅ Search Bar Component
 */
interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
}

const SearchBar = memo<SearchBarProps>(({ value, onChange }) => (
  <div className="relative flex-1 max-w-md">
    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400 pointer-events-none" />
    <input
      type="text"
      placeholder="Search datasets..."
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full pl-10 pr-10 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-lg text-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white dark:focus:bg-gray-600 transition-all"
      aria-label="Search datasets"
    />
    {value && (
      <button
        onClick={() => onChange('')}
        className="absolute right-3 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
        aria-label="Clear search"
      >
        <X className="w-4 h-4 text-gray-400" />
      </button>
    )}
  </div>
));

SearchBar.displayName = 'SearchBar';

/**
 * ✅ Filter Badge Component
 */
interface FilterBadgeProps {
  count: number;
  isActive: boolean;
  onClick: () => void;
}

const FilterBadge = memo<FilterBadgeProps>(({ count, isActive, onClick }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-2 px-4 py-2.5 rounded-lg border transition-all ${
      isActive
        ? 'bg-blue-50 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-600 dark:text-blue-400'
        : 'bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
    }`}
  >
    <Filter className="w-5 h-5" />
    <span className="font-medium">Filter</span>
    {count > 0 && (
      <span className="ml-1 px-2 py-0.5 bg-red-500 text-white text-xs font-bold rounded-full">
        {count}
      </span>
    )}
  </button>
));

FilterBadge.displayName = 'FilterBadge';

/**
 * ✅ Sort Controls Component
 */
interface SortControlsProps {
  sortBy: SortBy;
  sortOrder: SortOrder;
  onSortByChange: (value: SortBy) => void;
  onSortOrderChange: () => void;
}

const SortControls = memo<SortControlsProps>(
  ({ sortBy, sortOrder, onSortByChange, onSortOrderChange }) => (
    <div className="flex items-center gap-2">
      <select
        value={sortBy}
        onChange={(e) => onSortByChange(e.target.value as SortBy)}
        className="px-3 py-2.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        <option value="date">Newest First</option>
        <option value="name">Name (A-Z)</option>
        <option value="size">Size (Largest)</option>
      </select>

      <button
        onClick={onSortOrderChange}
        className="p-2.5 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
        title={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
      >
        <ArrowUpDown className={`w-5 h-5 text-gray-700 dark:text-gray-300 ${sortOrder === 'desc' && 'rotate-180'} transition-transform`} />
      </button>
    </div>
  )
);

SortControls.displayName = 'SortControls';

/**
 * ✅ View Mode Toggle Component
 */
interface ViewToggleProps {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
}

const ViewToggle = memo<ViewToggleProps>(({ mode, onChange }) => (
  <div className="flex items-center gap-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-1">
    <button
      onClick={() => onChange('grid')}
      className={`p-2 rounded transition-colors ${
        mode === 'grid'
          ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
          : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
      }`}
      title="Grid view"
    >
      <Grid3x3 className="w-5 h-5" />
    </button>
    <button
      onClick={() => onChange('list')}
      className={`p-2 rounded transition-colors ${
        mode === 'list'
          ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
          : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
      }`}
      title="List view"
    >
      <List className="w-5 h-5" />
    </button>
  </div>
));

ViewToggle.displayName = 'ViewToggle';

/**
 * ✅ Filters Panel Component
 */
interface FiltersPanelProps {
  allTags: string[];
  selectedTags: string[];
  selectedStatus: Array<'pending' | 'processing' | 'completed' | 'failed'>;
  onTagToggle: (tag: string) => void;
  onStatusToggle: (status: 'pending' | 'processing' | 'completed' | 'failed') => void;
  onClear: () => void;
}

const FiltersPanel = memo<FiltersPanelProps>(
  ({
    allTags,
    selectedTags,
    selectedStatus,
    onTagToggle,
    onStatusToggle,
    onClear,
  }) => (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 mb-6 animate-in fade-in slide-in-from-top-2">
      <div className="flex items-center justify-between mb-6">
        <h3 className="font-semibold text-gray-900 dark:text-white">
          Filters
        </h3>
        {(selectedTags.length > 0 || selectedStatus.length > 0) && (
          <button
            onClick={onClear}
            className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium"
          >
            Clear All
          </button>
        )}
      </div>

      <div className="space-y-6">
        {/* Status Filter */}
        <div>
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-3">
            Status
          </label>
          <div className="space-y-2">
            {(
              [
                'pending',
                'processing',
                'completed',
                'failed',
              ] as const
            ).map((status) => {
              const statusInfo = STATUS_MAP[status];
              return (
                <label key={status} className="flex items-center gap-3 cursor-pointer group">
                  <input
                    type="checkbox"
                    checked={selectedStatus.includes(status)}
                    onChange={() => onStatusToggle(status)}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  />
                  <div className="flex items-center gap-2">
                    <statusInfo.icon className={`w-4 h-4 ${statusInfo.color}`} />
                    <span className="text-sm text-gray-700 dark:text-gray-300 group-hover:text-gray-900 dark:group-hover:text-white">
                      {statusInfo.label}
                    </span>
                  </div>
                </label>
              );
            })}
          </div>
        </div>

        {/* Tags Filter */}
        {allTags.length > 0 && (
          <div>
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-3">
              Tags
            </label>
            <div className="flex flex-wrap gap-2">
              {allTags.map((tag) => (
                <button
                  key={tag}
                  onClick={() => onTagToggle(tag)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                    selectedTags.includes(tag)
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {tag}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
);

FiltersPanel.displayName = 'FiltersPanel';

/**
 * ✅ Bulk Actions Bar Component
 */
interface BulkActionsBarProps {
  selectedCount: number;
  totalCount: number;
  onSelectAll: () => void;
  onExport: () => void;
  onShare: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}

const BulkActionsBar = memo<BulkActionsBarProps>(
  ({
    selectedCount,
    totalCount,
    onSelectAll,
    onExport,
    onShare,
    onDelete,
    isDeleting,
  }) => (
    <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 mb-6 flex items-center justify-between animate-in fade-in slide-in-from-top-2">
      <div className="flex items-center gap-4">
        <input
          type="checkbox"
          checked={selectedCount === totalCount && totalCount > 0}
          onChange={onSelectAll}
          className="w-5 h-5 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
        />
        <span className="font-medium text-gray-900 dark:text-white">
          {selectedCount} selected
        </span>
      </div>

      <div className="flex items-center gap-3">
        <Button
          variant="secondary"
          size="sm"
          leftIcon={Download}
          onClick={onExport}
        >
          Export
        </Button>
        <Button
          variant="secondary"
          size="sm"
          leftIcon={Share2}
          onClick={onShare}
        >
          Share
        </Button>
        <Button
          variant="danger"
          size="sm"
          leftIcon={Trash2}
          onClick={onDelete}
          disabled={isDeleting}
        >
          {isDeleting ? 'Deleting...' : 'Delete'}
        </Button>
      </div>
    </div>
  )
);

BulkActionsBar.displayName = 'BulkActionsBar';

/**
 * ✅ Dataset Grid Item Component
 */
interface DatasetGridItemProps {
  dataset: DatasetItem;
  isSelected: boolean;
  onSelect: (id: string) => void;
  onView: (id: string) => void;
  onAnalyze: (id: string) => void;
  onDelete: (id: string) => void;
}

const DatasetGridItem = memo<DatasetGridItemProps>(
  ({ dataset, isSelected, onSelect, onView, onAnalyze, onDelete }) => {
    const [showMenu, setShowMenu] = useState(false);
    const statusInfo = STATUS_MAP[dataset.status];
    const StatusIcon = statusInfo.icon;

    const formatFileSize = (bytes: number | undefined): string => {
      const safeBytes = bytes ?? 0;
      if (safeBytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(safeBytes) / Math.log(k));
      return (
        Math.round((safeBytes / Math.pow(k, i)) * 100) / 100 +
        ' ' +
        sizes[i]
      );
    };

    return (
      <div
        className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden hover:shadow-lg dark:hover:shadow-xl hover:border-gray-300 dark:hover:border-gray-600 transition-all group cursor-pointer"
        onClick={() => onView(dataset.id)}
      >
        {/* Card Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between bg-gray-50 dark:bg-gray-700/50">
          <label
            className="flex items-center gap-2"
            onClick={(e) => e.stopPropagation()}
          >
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => onSelect(dataset.id)}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500 cursor-pointer"
            />
          </label>

          <div className="relative" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="p-2 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
            >
              <MoreVertical className="w-5 h-5 text-gray-600 dark:text-gray-400" />
            </button>

            {showMenu && (
              <div className="absolute right-0 mt-2 w-40 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50">
                <button
                  onClick={() => {
                    onView(dataset.id);
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
                >
                  <Eye className="w-4 h-4" />
                  View
                </button>
                <button
                  onClick={() => {
                    onAnalyze(dataset.id);
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
                >
                  <BarChart3 className="w-4 h-4" />
                  Analyze
                </button>
                <button className="w-full px-4 py-2 text-left flex items-center gap-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm">
                  <Download className="w-4 h-4" />
                  Download
                </button>
                <div className="border-t border-gray-200 dark:border-gray-700" />
                <button
                  onClick={() => {
                    onDelete(dataset.id);
                    setShowMenu(false);
                  }}
                  className="w-full px-4 py-2 text-left flex items-center gap-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-sm"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Card Body */}
        <div className="p-6 space-y-4">
          {/* Icon */}
          <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
            <Database className="w-6 h-6 text-blue-600 dark:text-blue-400" />
          </div>

          {/* Info */}
          <div className="min-w-0">
            <h3 className="font-semibold text-gray-900 dark:text-white truncate group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
              {dataset.name}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {(dataset.totalRows ?? 0).toLocaleString()} rows •{' '}
              {dataset.totalColumns ?? 0} columns
            </p>
          </div>

          {/* Meta */}
          <div className="flex items-center justify-between pt-2 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-600 dark:text-gray-400">
            <div className="flex items-center gap-2">
              <HardDrive className="w-4 h-4" />
              <span>{formatFileSize(dataset.fileSize)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              <span>
                {formatDistanceToNow(new Date(dataset.updatedAt), {
                  addSuffix: true,
                })}
              </span>
            </div>
          </div>

          {/* Status */}
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full w-fit ${statusInfo.bgColor}`}>
            <StatusIcon className={`w-4 h-4 ${statusInfo.color}`} />
            <span className={`text-xs font-medium ${statusInfo.color}`}>
              {statusInfo.label}
            </span>
          </div>
        </div>
      </div>
    );
  }
);

DatasetGridItem.displayName = 'DatasetGridItem';

/**
 * ✅ Empty State Component
 */
interface EmptyStateProps {
  showUpload: boolean;
  hasFilters: boolean;
  onUpload: () => void;
}

const EmptyState = memo<EmptyStateProps>(({ showUpload, hasFilters, onUpload }) => (
  <div className="text-center py-16">
    <div className="mb-4">
      <Database className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto" />
    </div>
    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
      {hasFilters ? 'No datasets found' : 'No datasets yet'}
    </h3>
    <p className="text-gray-600 dark:text-gray-400 mb-6">
      {hasFilters
        ? 'Try adjusting your search or filters'
        : 'Upload your first dataset to get started'}
    </p>
    {showUpload && (
      <Button variant="primary" leftIcon={Upload} onClick={onUpload}>
        Upload Dataset
      </Button>
    )}
  </div>
));

EmptyState.displayName = 'EmptyState';

// ============================================================================
// Main Component
// ============================================================================

/**
 * ✅ DatasetsPage - Enhanced Main Component
 */
const DatasetsPage = memo(() => {
  const navigate = useNavigate();
  const { datasets, isLoading, error, deleteDataset, bulkDeleteDatasets } =
    useDatasets();
  const addNotification = uiStore((state) => state.addNotification);

  // ========================================================================
  // State
  // ========================================================================

  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<SortBy>('date');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string>>(
    new Set()
  );
  const [datasetToDelete, setDatasetToDelete] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [statusFilter, setStatusFilter] = useState<
    Array<'pending' | 'processing' | 'completed' | 'failed'>
  >([]);

  // ========================================================================
  // Memoized Values
  // ========================================================================

  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  const allTags = useMemo(() => {
    const tags = new Set<string>();
    datasets.forEach((dataset) => {
      (dataset as unknown as DatasetItem).tags?.forEach((tag) =>
        tags.add(tag)
      );
    });
    return Array.from(tags).sort();
  }, [datasets]);

  const filteredAndSortedDatasets = useMemo(() => {
    let filtered = datasets.map((d) => d as unknown as DatasetItem);

    if (debouncedSearchQuery) {
      const query = debouncedSearchQuery.toLowerCase();
      filtered = filtered.filter(
        (dataset) =>
          dataset.name.toLowerCase().includes(query) ||
          dataset.description?.toLowerCase().includes(query)
      );
    }

    if (selectedTags.length > 0) {
      filtered = filtered.filter((dataset) =>
        selectedTags.some((tag) => dataset.tags?.includes(tag))
      );
    }

    if (statusFilter.length > 0) {
      filtered = filtered.filter((dataset) =>
        statusFilter.includes(dataset.status)
      );
    }

    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'date':
          comparison =
            new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime();
          break;
        case 'size':
          const aSize = a.fileSize ?? 0;
          const bSize = b.fileSize ?? 0;
          comparison = bSize - aSize;
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [datasets, debouncedSearchQuery, selectedTags, statusFilter, sortBy, sortOrder]);

  // ========================================================================
  // Handlers
  // ========================================================================

  const handleDeleteDataset = useCallback(async () => {
    if (!datasetToDelete) return;

    setIsDeleting(true);
    try {
      const result = await deleteDataset(datasetToDelete);
      if (result.success) {
        addNotification({
          type: 'success',
          message: 'Dataset deleted successfully',
          duration: 3000,
        });
        setDatasetToDelete(null);
      } else {
        addNotification({
          type: 'error',
          message: result.error || 'Failed to delete dataset',
          duration: 5000,
        });
      }
    } catch (err) {
      console.error('Failed to delete dataset:', err);
      addNotification({
        type: 'error',
        message: 'An error occurred while deleting the dataset',
        duration: 5000,
      });
    } finally {
      setIsDeleting(false);
    }
  }, [datasetToDelete, deleteDataset, addNotification]);

  const handleBulkDelete = useCallback(async () => {
    if (selectedDatasets.size === 0) return;

    setIsDeleting(true);
    try {
      const result = await bulkDeleteDatasets(Array.from(selectedDatasets));
      if (result.success) {
        addNotification({
          type: 'success',
          message: `${selectedDatasets.size} dataset(s) deleted successfully`,
          duration: 3000,
        });
        setSelectedDatasets(new Set());
      } else {
        addNotification({
          type: 'error',
          message: result.error || 'Failed to delete datasets',
          duration: 5000,
        });
      }
    } catch (err) {
      console.error('Failed to bulk delete datasets:', err);
      addNotification({
        type: 'error',
        message: 'An error occurred during bulk deletion',
        duration: 5000,
      });
    } finally {
      setIsDeleting(false);
    }
  }, [selectedDatasets, bulkDeleteDatasets, addNotification]);

  const toggleDatasetSelection = useCallback((id: string) => {
    setSelectedDatasets((prev) => {
      const newSet = new Set(prev);
      newSet.has(id) ? newSet.delete(id) : newSet.add(id);
      return newSet;
    });
  }, []);

  const toggleAllSelection = useCallback(() => {
    if (selectedDatasets.size === filteredAndSortedDatasets.length) {
      setSelectedDatasets(new Set());
    } else {
      setSelectedDatasets(
        new Set(filteredAndSortedDatasets.map((d) => d.id))
      );
    }
  }, [selectedDatasets.size, filteredAndSortedDatasets]);

  const handleTagToggle = useCallback((tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  }, []);

  const handleStatusToggle = useCallback(
    (status: 'pending' | 'processing' | 'completed' | 'failed') => {
      setStatusFilter((prev) =>
        prev.includes(status)
          ? prev.filter((s) => s !== status)
          : [...prev, status]
      );
    },
    []
  );

  const clearAllFilters = useCallback(() => {
    setSelectedTags([]);
    setStatusFilter([]);
    setSearchQuery('');
  }, []);

  const hasActiveFilters =
    selectedTags.length > 0 || statusFilter.length > 0 || searchQuery.length > 0;

  // ========================================================================
  // Render
  // ========================================================================

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Page Header */}
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-8">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
                Datasets
              </h1>
              <p className="text-gray-600 dark:text-gray-400 mt-1">
                Manage and organize your data files
              </p>
            </div>
            <Button
              variant="primary"
              leftIcon={Plus}
              onClick={() => setShowUploadModal(true)}
            >
              New Dataset
            </Button>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 rounded-lg flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          {/* Controls Bar */}
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 mb-6">
            <div className="flex flex-col md:flex-row md:items-center gap-4">
              <SearchBar value={searchQuery} onChange={setSearchQuery} />

              <div className="flex items-center gap-3 ml-auto">
                <FilterBadge
                  count={selectedTags.length + statusFilter.length}
                  isActive={showFilters}
                  onClick={() => setShowFilters(!showFilters)}
                />
                <SortControls
                  sortBy={sortBy}
                  sortOrder={sortOrder}
                  onSortByChange={setSortBy}
                  onSortOrderChange={() =>
                    setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
                  }
                />
                <ViewToggle mode={viewMode} onChange={setViewMode} />
              </div>
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <FiltersPanel
              allTags={allTags}
              selectedTags={selectedTags}
              selectedStatus={statusFilter}
              onTagToggle={handleTagToggle}
              onStatusToggle={handleStatusToggle}
              onClear={clearAllFilters}
            />
          )}

          {/* Bulk Actions Bar */}
          {selectedDatasets.size > 0 && (
            <BulkActionsBar
              selectedCount={selectedDatasets.size}
              totalCount={filteredAndSortedDatasets.length}
              onSelectAll={toggleAllSelection}
              onExport={() => addNotification({
                type: 'info',
                message: 'Export feature coming soon',
                duration: 2000,
              })}
              onShare={() => addNotification({
                type: 'info',
                message: 'Share feature coming soon',
                duration: 2000,
              })}
              onDelete={handleBulkDelete}
              isDeleting={isDeleting}
            />
          )}

          {/* Content */}
          {isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <Skeleton key={i} height="280px" variant="rectangular" />
              ))}
            </div>
          ) : filteredAndSortedDatasets.length === 0 ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <EmptyState
                showUpload={!hasActiveFilters}
                hasFilters={hasActiveFilters}
                onUpload={() => setShowUploadModal(true)}
              />
            </div>
          ) : viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredAndSortedDatasets.map((dataset) => (
                <DatasetGridItem
                  key={dataset.id}
                  dataset={dataset}
                  isSelected={selectedDatasets.has(dataset.id)}
                  onSelect={toggleDatasetSelection}
                  onView={(id) => navigate(`/datasets/${id}`)}
                  onAnalyze={(id) => navigate(`/eda/${id}`)}
                  onDelete={setDatasetToDelete}
                />
              ))}
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
              <table className="w-full">
                <thead className="bg-gray-50 dark:bg-gray-700 border-b border-gray-200 dark:border-gray-600">
                  <tr>
                    <th className="px-6 py-3 text-left">
                      <input
                        type="checkbox"
                        checked={
                          selectedDatasets.size ===
                            filteredAndSortedDatasets.length &&
                          filteredAndSortedDatasets.length > 0
                        }
                        onChange={toggleAllSelection}
                        className="w-4 h-4 rounded border-gray-300 text-blue-600"
                      />
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-900 dark:text-white uppercase">
                      Name
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-900 dark:text-white uppercase">
                      Details
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-900 dark:text-white uppercase">
                      Status
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold text-gray-900 dark:text-white uppercase">
                      Modified
                    </th>
                    <th className="px-6 py-3 text-right text-xs font-semibold text-gray-900 dark:text-white uppercase">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                  {filteredAndSortedDatasets.map((dataset) => {
                    const statusInfo = STATUS_MAP[dataset.status];
                    const StatusIcon = statusInfo.icon;

                    return (
                      <tr
                        key={dataset.id}
                        className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors cursor-pointer"
                        onClick={() => navigate(`/datasets/${dataset.id}`)}
                      >
                        <td className="px-6 py-4" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={selectedDatasets.has(dataset.id)}
                            onChange={() => toggleDatasetSelection(dataset.id)}
                            className="w-4 h-4 rounded border-gray-300 text-blue-600"
                          />
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <Database className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
                            <span className="font-medium text-gray-900 dark:text-white truncate">
                              {dataset.name}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-400">
                          {(dataset.totalRows ?? 0).toLocaleString()} rows •{' '}
                          {dataset.totalColumns ?? 0} columns
                        </td>
                        <td className="px-6 py-4">
                          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full w-fit ${statusInfo.bgColor}`}>
                            <StatusIcon className={`w-4 h-4 ${statusInfo.color}`} />
                            <span className={`text-xs font-medium ${statusInfo.color}`}>
                              {statusInfo.label}
                            </span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-sm text-gray-600 dark:text-gray-400">
                          {formatDistanceToNow(new Date(dataset.updatedAt), {
                            addSuffix: true,
                          })}
                        </td>
                        <td
                          className="px-6 py-4 text-right"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            onClick={() => navigate(`/eda/${dataset.id}`)}
                            className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 font-medium text-sm"
                          >
                            Analyze
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Upload Modal */}
          <DataUpload
            isOpen={showUploadModal}
            onClose={() => setShowUploadModal(false)}
          />

          {/* Delete Modal */}
          <ConfirmModal
            isOpen={!!datasetToDelete}
            onClose={() => setDatasetToDelete(null)}
            onConfirm={handleDeleteDataset}
            title="Delete Dataset"
            message="Are you sure you want to delete this dataset? This action cannot be undone."
            confirmText="Delete"
            cancelText="Cancel"
            variant="danger"
            isLoading={isDeleting}
          />
        </div>
      </div>
    </DashboardLayout>
  );
});

DatasetsPage.displayName = 'DatasetsPage';

export default DatasetsPage;
