// src/pages/VisualizationsPage.tsx - UPDATED TO WORK WITH EXISTING ChartGenerator

import { useState, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Plus,
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
  BarChart3,
  LineChart,
  PieChart,
  Settings,
  Copy,
  Check,
} from 'lucide-react';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import ChartGenerator from '@/components/visualizations/ChartGenerator';
import ChartPreview from '@/components/visualizations/ChartPreview';
import Button from '@/components/shared/Button';
import Modal, { ConfirmModal } from '@/components/shared/Modal';
import { Skeleton, ListSkeleton } from '@/components/shared/Loading';
import { useDebounce } from '@/hooks/useDebounce';
import { formatDistanceToNow } from 'date-fns';

// ============================================================================
// Type Definitions
// ============================================================================

interface Visualization {
  id: string;
  name: string;
  description?: string;
  chartType: string;
  datasetId: string;
  datasetName: string;
  config: {
    xAxis?: string;
    yAxis?: string;
    groupBy?: string;
    aggregation?: string;
  };
  createdAt: string;
  updatedAt: string;
  status: 'ready' | 'processing' | 'failed';
  views: number;
  shared: boolean;
  tags?: string[];
}

type ViewMode = 'grid' | 'list';
type SortBy = 'name' | 'date' | 'views';
type SortOrder = 'asc' | 'desc';
type FilterStatus = 'all' | 'shared' | 'private';

// ============================================================================
// Mock Data
// ============================================================================

const MOCK_COLUMNS = [
  { name: 'month', type: 'categorical' as const },
  { name: 'revenue', type: 'numeric' as const },
  { name: 'quantity', type: 'numeric' as const },
  { name: 'category', type: 'categorical' as const },
];

const MOCK_DATA = [
  { month: 'Jan', revenue: 4000, quantity: 240, category: 'A' },
  { month: 'Feb', revenue: 3000, quantity: 139, category: 'B' },
  { month: 'Mar', revenue: 2000, quantity: 980, category: 'A' },
  { month: 'Apr', revenue: 2780, quantity: 390, category: 'C' },
  { month: 'May', revenue: 1890, quantity: 480, category: 'B' },
  { month: 'Jun', revenue: 2390, quantity: 380, category: 'A' },
];

// ============================================================================
// Component
// ============================================================================

/**
 * VisualizationsPage - Enterprise-grade visualization gallery
 * ✅ FIXED: Works with ChartGenerator advanced mode
 */
const VisualizationsPage: React.FC = () => {
  const navigate = useNavigate();

  // ============================================================================
  // State Management
  // ============================================================================

  const [visualizations, setVisualizations] = useState<Visualization[]>([
    {
      id: '1',
      name: 'Sales Trend Analysis',
      description: 'Monthly sales revenue trend over the past year',
      chartType: 'line',
      datasetId: 'dataset-1',
      datasetName: 'Sales Data',
      config: {
        xAxis: 'month',
        yAxis: 'revenue',
        aggregation: 'sum',
      },
      createdAt: new Date(Date.now() - 3600000).toISOString(),
      updatedAt: new Date(Date.now() - 3600000).toISOString(),
      status: 'ready',
      views: 124,
      shared: true,
      tags: ['sales', 'trend'],
    },
    {
      id: '2',
      name: 'Category Distribution',
      description: 'Product category distribution by revenue',
      chartType: 'pie',
      datasetId: 'dataset-1',
      datasetName: 'Sales Data',
      config: {
        groupBy: 'category',
        aggregation: 'sum',
      },
      createdAt: new Date(Date.now() - 86400000).toISOString(),
      updatedAt: new Date(Date.now() - 86400000).toISOString(),
      status: 'ready',
      views: 87,
      shared: false,
      tags: ['distribution', 'category'],
    },
  ]);

  // UI State
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortBy>('date');
  const [sortOrder] = useState<SortOrder>('desc');
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedViz, setSelectedViz] = useState<Visualization | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [loading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showMoreMenu, setShowMoreMenu] = useState<string | null>(null);

  // Debounce search query
  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  // ============================================================================
  // Computations
  // ============================================================================

  const filteredAndSortedVisualizations = useMemo(() => {
    let filtered = [...visualizations];

    // Search filter
    if (debouncedSearchQuery) {
      const query = debouncedSearchQuery.toLowerCase();
      filtered = filtered.filter(
        (viz) =>
          viz.name.toLowerCase().includes(query) ||
          viz.description?.toLowerCase().includes(query) ||
          viz.datasetName.toLowerCase().includes(query)
      );
    }

    // Status filter
    if (filterStatus === 'shared') {
      filtered = filtered.filter((viz) => viz.shared);
    } else if (filterStatus === 'private') {
      filtered = filtered.filter((viz) => !viz.shared);
    }

    // Sorting
    filtered.sort((a, b) => {
      let comparison = 0;

      switch (sortBy) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'date':
          comparison =
            new Date(b.updatedAt).getTime() -
            new Date(a.updatedAt).getTime();
          break;
        case 'views':
          comparison = b.views - a.views;
          break;
      }

      return sortOrder === 'asc' ? -comparison : comparison;
    });

    return filtered;
  }, [visualizations, debouncedSearchQuery, sortBy, sortOrder, filterStatus]);

  // ============================================================================
  // Event Handlers
  // ============================================================================

  const getChartIcon = useCallback((chartType: string) => {
    switch (chartType) {
      case 'line':
        return <LineChart className="w-5 h-5" />;
      case 'bar':
        return <BarChart3 className="w-5 h-5" />;
      case 'pie':
        return <PieChart className="w-5 h-5" />;
      default:
        return <BarChart3 className="w-5 h-5" />;
    }
  }, []);

  const getChartLabel = useCallback((chartType: string) => {
    return chartType.charAt(0).toUpperCase() + chartType.slice(1);
  }, []);

  const handleDeleteViz = useCallback(async () => {
    if (!selectedViz) return;
    setIsDeleting(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setVisualizations((prev) =>
        prev.filter((v) => v.id !== selectedViz.id)
      );
      setShowDeleteModal(false);
      setSelectedViz(null);
    } catch (error) {
      console.error('Failed to delete visualization:', error);
    } finally {
      setIsDeleting(false);
    }
  }, [selectedViz]);

  const copyShareLink = useCallback(() => {
    if (!selectedViz) return;
    const shareUrl = `${window.location.origin}/visualizations/shared/${selectedViz.id}`;
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [selectedViz]);

  /**
   * ✅ FIXED: Handle chart creation with ChartGenerator
   */
  const handleChartCreated = useCallback(async (config: any) => {
    setShowCreateModal(false);
    // Add new visualization to list
    const newViz: Visualization = {
      id: `${Date.now()}`,
      name: config.title,
      description: config.subtitle,
      chartType: config.type,
      datasetId: 'dataset-1',
      datasetName: 'Sales Data',
      config: {
        xAxis: config.xAxis,
        yAxis: config.yAxis,
      },
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      status: 'ready',
      views: 0,
      shared: false,
    };
    setVisualizations((prev) => [newViz, ...prev]);
  }, []);

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <DashboardLayout>
      <div className="space-y-6 p-6">
        {/* Page Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Visualizations</h1>
            <p className="text-gray-600 mt-1">
              Create and manage data visualizations
            </p>
          </div>
          <Button
            variant="primary"
            leftIcon={Plus}
            onClick={() => setShowCreateModal(true)}
          >
            New Visualization
          </Button>
        </div>

        {/* Controls Bar */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
            <div className="flex gap-3 flex-1 w-full md:w-auto">
              {/* Search */}
              <div className="flex-1 relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search visualizations..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all"
                />
              </div>

              {/* Filter Button */}
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`px-4 py-2 rounded-lg border transition-colors flex items-center gap-2 ${
                  showFilters
                    ? 'bg-primary-50 border-primary-300 text-primary-700'
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Filter className="w-5 h-5" />
                <span className="hidden sm:inline">Filter</span>
              </button>
            </div>

            <div className="flex gap-3 items-center w-full sm:w-auto">
              {/* Sort Dropdown */}
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as SortBy)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent outline-none transition-all text-sm"
              >
                <option value="date">Newest</option>
                <option value="name">Name</option>
                <option value="views">Most Viewed</option>
              </select>

              {/* View Toggle */}
              <div className="flex border border-gray-300 rounded-lg">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`px-3 py-2 transition-colors ${
                    viewMode === 'grid'
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                  title="Grid view"
                >
                  <Grid3x3 className="w-5 h-5" />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`px-3 py-2 border-l border-gray-300 transition-colors ${
                    viewMode === 'list'
                      ? 'bg-primary-50 text-primary-700'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                  title="List view"
                >
                  <List className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Filters Panel */}
        {showFilters && (
          <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-4 animate-in slide-in-from-top-2">
            <div>
              <p className="text-sm font-semibold text-gray-900 mb-3">Visibility</p>
              <div className="space-y-2">
                {[
                  { id: 'all', label: 'All Visualizations' },
                  { id: 'shared', label: 'Shared with Me' },
                  { id: 'private', label: 'My Visualizations' },
                ].map((option) => (
                  <label key={option.id} className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="radio"
                      name="status"
                      value={option.id}
                      checked={filterStatus === option.id}
                      onChange={(e) =>
                        setFilterStatus(e.target.value as FilterStatus)
                      }
                      className="w-4 h-4 rounded"
                    />
                    <span className="text-sm">{option.label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Content Area */}
        {loading ? (
          <div className="space-y-6">
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {[...Array(6)].map((_, i) => (
                  <Skeleton key={i} height="300px" variant="rectangular" />
                ))}
              </div>
            ) : (
              <ListSkeleton items={5} />
            )}
          </div>
        ) : filteredAndSortedVisualizations.length === 0 ? (
          <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
            <BarChart3 className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              {searchQuery ? 'No visualizations found' : 'No visualizations yet'}
            </h3>
            <p className="text-gray-600 mb-6">
              {searchQuery
                ? 'Try adjusting your search'
                : 'Create your first visualization to get started'}
            </p>
            {!searchQuery && (
              <Button
                variant="primary"
                leftIcon={Plus}
                onClick={() => setShowCreateModal(true)}
              >
                Create Visualization
              </Button>
            )}
          </div>
        ) : viewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredAndSortedVisualizations.map((viz) => (
              <div
                key={viz.id}
                className="group bg-white rounded-lg border border-gray-200 hover:shadow-lg hover:border-gray-300 transition-all overflow-hidden"
              >
                {/* Card Header */}
                <div className="p-4 bg-gradient-to-r from-gray-50 to-white border-b border-gray-200 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-gray-700">
                    {getChartIcon(viz.chartType)}
                    <span className="text-sm font-medium">
                      {getChartLabel(viz.chartType)}
                    </span>
                  </div>
                  <div className="relative">
                    <button
                      onClick={() =>
                        setShowMoreMenu(
                          showMoreMenu === viz.id ? null : viz.id
                        )
                      }
                      className="p-1 hover:bg-gray-200 rounded transition-colors"
                    >
                      <MoreVertical className="w-5 h-5 text-gray-600" />
                    </button>

                    {showMoreMenu === viz.id && (
                      <div className="absolute right-0 top-8 bg-white rounded-lg shadow-lg border border-gray-200 z-10 min-w-[160px]">
                        {[
                          {
                            icon: Eye,
                            label: 'View',
                            onClick: () => {
                              setSelectedViz(viz);
                              setShowPreviewModal(true);
                              setShowMoreMenu(null);
                            },
                          },
                          {
                            icon: Settings,
                            label: 'Edit',
                            onClick: () => {
                              navigate(`/visualizations/edit/${viz.id}`);
                              setShowMoreMenu(null);
                            },
                          },
                          {
                            icon: Share2,
                            label: 'Share',
                            onClick: () => {
                              setSelectedViz(viz);
                              setShowShareModal(true);
                              setShowMoreMenu(null);
                            },
                          },
                          {
                            icon: Download,
                            label: 'Download',
                            onClick: () => setShowMoreMenu(null),
                          },
                          {
                            icon: Trash2,
                            label: 'Delete',
                            onClick: () => {
                              setSelectedViz(viz);
                              setShowDeleteModal(true);
                              setShowMoreMenu(null);
                            },
                            className: 'text-red-600',
                          },
                        ].map((item) => (
                          <button
                            key={item.label}
                            onClick={item.onClick}
                            className={`w-full px-4 py-2 text-left text-sm flex items-center gap-2 hover:bg-gray-50 border-t border-gray-200 first:border-t-0 ${
                              item.className || ''
                            }`}
                          >
                            <item.icon className="w-4 h-4" />
                            {item.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Card Content */}
                <div
                  className="p-6 cursor-pointer hover:bg-gray-50 transition-colors"
                  onClick={() => {
                    setSelectedViz(viz);
                    setShowPreviewModal(true);
                  }}
                >
                  <h3 className="font-semibold text-gray-900 mb-1 truncate">
                    {viz.name}
                  </h3>
                  <p className="text-sm text-gray-600 mb-4">{viz.datasetName}</p>
                  {viz.description && (
                    <p className="text-xs text-gray-500 line-clamp-2 mb-4">
                      {viz.description}
                    </p>
                  )}
                  {viz.tags && viz.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {viz.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="inline-block px-2 py-1 text-xs bg-primary-100 text-primary-700 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Card Footer */}
                <div className="px-6 py-3 bg-gray-50 border-t border-gray-200 flex items-center justify-between text-xs text-gray-600">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1">
                      <Eye className="w-4 h-4" />
                      {viz.views}
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-4 h-4" />
                      {formatDistanceToNow(new Date(viz.updatedAt), {
                        addSuffix: true,
                      })}
                    </span>
                  </div>
                  {viz.shared && (
                    <span className="px-2 py-1 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                      Shared
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2 bg-white rounded-lg border border-gray-200 divide-y">
            {filteredAndSortedVisualizations.map((viz) => (
              <div
                key={viz.id}
                className="p-4 flex items-center justify-between hover:bg-gray-50 transition-colors group"
              >
                <div
                  className="flex-1 flex items-center gap-4 cursor-pointer"
                  onClick={() => {
                    setSelectedViz(viz);
                    setShowPreviewModal(true);
                  }}
                >
                  <div className="text-gray-600">{getChartIcon(viz.chartType)}</div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 truncate">{viz.name}</p>
                    <p className="text-sm text-gray-600">
                      {viz.datasetName} • {getChartLabel(viz.chartType)}
                    </p>
                  </div>
                </div>

                <div className="hidden md:flex items-center gap-6 text-sm text-gray-600">
                  <span>{viz.views} views</span>
                  <span>
                    {formatDistanceToNow(new Date(viz.updatedAt), {
                      addSuffix: true,
                    })}
                  </span>
                </div>

                <div className="relative">
                  <button
                    onClick={() =>
                      setShowMoreMenu(
                        showMoreMenu === viz.id ? null : viz.id
                      )
                    }
                    className="p-2 hover:bg-gray-200 rounded transition-colors"
                  >
                    <MoreVertical className="w-5 h-5 text-gray-600" />
                  </button>

                  {showMoreMenu === viz.id && (
                    <div className="absolute right-0 top-8 bg-white rounded-lg shadow-lg border border-gray-200 z-10 min-w-[160px]">
                      {[
                        {
                          icon: Eye,
                          label: 'View',
                          onClick: () => {
                            setSelectedViz(viz);
                            setShowPreviewModal(true);
                            setShowMoreMenu(null);
                          },
                        },
                        {
                          icon: Settings,
                          label: 'Edit',
                          onClick: () => {
                            navigate(`/visualizations/edit/${viz.id}`);
                            setShowMoreMenu(null);
                          },
                        },
                        {
                          icon: Share2,
                          label: 'Share',
                          onClick: () => {
                            setSelectedViz(viz);
                            setShowShareModal(true);
                            setShowMoreMenu(null);
                          },
                        },
                        {
                          icon: Download,
                          label: 'Download',
                          onClick: () => setShowMoreMenu(null),
                        },
                        {
                          icon: Trash2,
                          label: 'Delete',
                          onClick: () => {
                            setSelectedViz(viz);
                            setShowDeleteModal(true);
                            setShowMoreMenu(null);
                          },
                          className: 'text-red-600',
                        },
                      ].map((item) => (
                        <button
                          key={item.label}
                          onClick={item.onClick}
                          className={`w-full px-4 py-2 text-left text-sm flex items-center gap-2 hover:bg-gray-50 border-t border-gray-200 first:border-t-0 ${
                            item.className || ''
                          }`}
                        >
                          <item.icon className="w-4 h-4" />
                          {item.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ✅ Create Visualization Modal - FIXED WITH ChartGenerator */}
        <Modal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          title="Create New Visualization"
          size="lg"
        >
          <ChartGenerator
            datasetId="dataset-1"
            columns={MOCK_COLUMNS}
            data={MOCK_DATA}
            onSave={handleChartCreated}
            onExport={(config) => {
              console.log('Export config:', config);
            }}
          />
        </Modal>

        {/* Preview Modal */}
        {selectedViz && (
          <Modal
            isOpen={showPreviewModal}
            onClose={() => {
              setShowPreviewModal(false);
              setSelectedViz(null);
            }}
            title={selectedViz.name}
            size="xl"
          >
            <ChartPreview
              data={selectedViz}
            />
          </Modal>
        )}

        {/* Share Modal */}
        <Modal
          isOpen={showShareModal}
          onClose={() => setShowShareModal(false)}
          title="Share Visualization"
          size="sm"
        >
          {selectedViz && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Share Link
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={`${window.location.origin}/visualizations/shared/${selectedViz.id}`}
                    readOnly
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg bg-gray-50 text-gray-600 text-sm"
                  />
                  <Button
                    variant="secondary"
                    size="sm"
                    leftIcon={copied ? Check : Copy}
                    onClick={copyShareLink}
                  >
                    {copied ? 'Copied' : 'Copy'}
                  </Button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Share Via
                </label>
                <div className="grid grid-cols-3 gap-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      window.open(
                        `mailto:?subject=Check out this visualization&body=${window.location.origin}/visualizations/shared/${selectedViz.id}`
                      )
                    }
                  >
                    Email
                  </Button>
                  <Button variant="outline" size="sm">
                    Twitter
                  </Button>
                  <Button variant="outline" size="sm">
                    LinkedIn
                  </Button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Visibility
                </label>
                <div className="space-y-2">
                  {[
                    { id: 'private', label: 'Private (Only me)' },
                    { id: 'public', label: 'Public (Anyone with link)' },
                  ].map((option) => (
                    <label
                      key={option.id}
                      className="flex items-center gap-3 cursor-pointer"
                    >
                      <input
                        type="radio"
                        name="visibility"
                        defaultChecked={
                          option.id === 'private' && !selectedViz.shared
                        }
                        className="w-4 h-4"
                      />
                      <span className="text-sm">{option.label}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          )}
        </Modal>

        {/* Delete Confirmation Modal */}
        <ConfirmModal
          isOpen={showDeleteModal}
          onClose={() => {
            setShowDeleteModal(false);
            setSelectedViz(null);
          }}
          onConfirm={handleDeleteViz}
          title="Delete Visualization"
          message={`Are you sure you want to delete "${selectedViz?.name}"? This action cannot be undone.`}
          confirmText="Delete"
          variant="danger"
          isLoading={isDeleting}
        />
      </div>
    </DashboardLayout>
  );
};

VisualizationsPage.displayName = 'VisualizationsPage';

export default VisualizationsPage;
