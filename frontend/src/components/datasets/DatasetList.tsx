// src/components/datasets/DatasetList.tsx

import {useState, useMemo, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  Filter,
  Grid3x3,
  List,
  SortAsc,
  SortDesc,
  Download,
  Trash2,
  MoreVertical,
  Calendar,
  HardDrive,
  FileText,
  Plus,
  RefreshCw,
  X,
} from 'lucide-react';
import { useDatasets } from '@/hooks/useDatasets';
import DatasetCard from './DatasetCard';
import { useDebounce } from '@/hooks/useDebounce';

type ViewMode = 'grid' | 'list';
type SortField = 'name' | 'createdAt' | 'size' | 'rowCount';
type SortOrder = 'asc' | 'desc';
type FilterType = 'all' | 'csv' | 'excel' | 'json';

interface DatasetListProps {
  onUpload?: () => void;
}

/**
 * DatasetList - Comprehensive dataset management component
 * Features: Grid/List view toggle, search, filtering, sorting, bulk actions
 * Supports responsive layout with pagination and loading states
 */
const DatasetList: React.FC<DatasetListProps> = ({ onUpload }) => {
  const navigate = useNavigate();
  const { datasets, isLoading, deleteDataset, refreshDatasets } = useDatasets();

  // View and Filter States
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<FilterType>('all');
  const [sortField, setSortField] = useState<SortField>('createdAt');
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc');
  const [showFilters, setShowFilters] = useState(false);
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = viewMode === 'grid' ? 12 : 10;

  // Debounce search query
  const debouncedSearchQuery = useDebounce(searchQuery, 300);

  // Filter and Sort Logic
  const filteredAndSortedDatasets = useMemo(() => {
    let filtered = [...datasets];

    // Apply search filter
    if (debouncedSearchQuery) {
      filtered = filtered.filter(
        (dataset) =>
          dataset.name.toLowerCase().includes(debouncedSearchQuery.toLowerCase()) ||
          (dataset.description?.toLowerCase().includes(debouncedSearchQuery.toLowerCase()) ?? false)
      );
    }

    // Apply type filter
    if (filterType !== 'all') {
      filtered = filtered.filter((dataset) => {
        const fileExt = dataset.fileType.toLowerCase();
        switch (filterType) {
          case 'csv':
            return fileExt === 'csv';
          case 'excel':
            return fileExt === 'xlsx' || fileExt === 'xls';
          case 'json':
            return fileExt === 'json';
          default:
            return true;
        }
      });
    }

    // Apply sorting - safely access properties
    filtered.sort((a, b) => {
      // Create sortable values based on sortField
      let aValue: string | number;
      let bValue: string | number;

      switch (sortField) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'createdAt':
          aValue = new Date(a.createdAt).getTime();
          bValue = new Date(b.createdAt).getTime();
          break;
        case 'size':
          aValue = a.size ?? 0;
          bValue = b.size ?? 0;
          break;
        case 'rowCount':
          aValue = a.rowCount ?? 0;
          bValue = b.rowCount ?? 0;
          break;
        default:
          return 0;
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    return filtered;
  }, [datasets, debouncedSearchQuery, filterType, sortField, sortOrder]);

  // Pagination
  const totalPages = Math.ceil(filteredAndSortedDatasets.length / itemsPerPage);
  const paginatedDatasets = useMemo(() => {
    const startIndex = (currentPage - 1) * itemsPerPage;
    return filteredAndSortedDatasets.slice(startIndex, startIndex + itemsPerPage);
  }, [filteredAndSortedDatasets, currentPage, itemsPerPage]);

  // Reset to page 1 when filters change
  useEffect(() => {
    setCurrentPage(1);
  }, [debouncedSearchQuery, filterType, sortField, sortOrder]);

  // Handlers
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('asc');
    }
  };

  const handleSelectDataset = (id: string) => {
    setSelectedDatasets((prev) =>
      prev.includes(id) ? prev.filter((item) => item !== id) : [...prev, id]
    );
  };

  const handleSelectAll = () => {
    if (selectedDatasets.length === paginatedDatasets.length) {
      setSelectedDatasets([]);
    } else {
      setSelectedDatasets(paginatedDatasets.map((d) => d.id));
    }
  };

  const handleBulkDelete = async () => {
    try {
      await Promise.all(selectedDatasets.map((id) => deleteDataset(id)));
      setSelectedDatasets([]);
      setShowDeleteConfirm(false);
      await refreshDatasets();
    } catch (error) {
      console.error('Bulk delete failed:', error);
    }
  };

  const handleViewDataset = (id: string) => {
    navigate(`/datasets/${id}`);
  };

  // Get statistics
const stats = useMemo(() => {
  // ← Use totalRows and fileSize instead of rowCount and size
  const totalSize = datasets.reduce((sum, d) => sum + (d.fileSize ?? 0), 0);
  const totalRows = datasets.reduce((sum, d) => sum + (d.totalRows ?? 0), 0);
  return {
    total: datasets.length,
    filtered: filteredAndSortedDatasets.length,
    totalSize,
    totalRows,
  };
}, [datasets, filteredAndSortedDatasets]);

return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Datasets</h1>
          <p className="text-gray-600 mt-1">
            {stats.filtered} of {stats.total} datasets
            {stats.filtered !== stats.total && ' (filtered)'}
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={refreshDatasets}
            className="btn btn-secondary btn-sm flex items-center space-x-2"
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <button
            onClick={onUpload}
            className="btn btn-primary btn-sm flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>Upload Dataset</span>
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Datasets</p>
              <p className="text-2xl font-bold text-gray-900">{stats.total}</p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <HardDrive className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Storage</p>
              <p className="text-2xl font-bold text-gray-900">
                {(stats.totalSize / 1024 / 1024).toFixed(1)} MB
              </p>
            </div>
          </div>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <Calendar className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Rows</p>
              <p className="text-2xl font-bold text-gray-900">
                {stats.totalRows.toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex flex-col lg:flex-row lg:items-center gap-4">
          {/* Search Bar */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search datasets by name or description..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Filter Button */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`btn btn-secondary flex items-center space-x-2 ${
              filterType !== 'all' || showFilters ? 'bg-blue-50 border-blue-300' : ''
            }`}
          >
            <Filter className="w-4 h-4" />
            <span>Filters</span>
            {filterType !== 'all' && (
              <span className="badge badge-primary ml-1">1</span>
            )}
          </button>

          {/* View Mode Toggle */}
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-md transition-colors ${
                viewMode === 'grid'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              aria-label="Grid view"
            >
              <Grid3x3 className="w-5 h-5" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-md transition-colors ${
                viewMode === 'list'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
              aria-label="List view"
            >
              <List className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Expanded Filters */}
        {showFilters && (
          <div className="mt-4 pt-4 border-t border-gray-200 animate-slide-in-down">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* File Type Filter */}
              <div>
                <label className="label">File Type</label>
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value as FilterType)}
                  className="select w-full"
                >
                  <option value="all">All Types</option>
                  <option value="csv">CSV Files</option>
                  <option value="excel">Excel Files</option>
                  <option value="json">JSON Files</option>
                </select>
              </div>

              {/* Sort Field */}
              <div>
                <label className="label">Sort By</label>
                <select
                  value={sortField}
                  onChange={(e) => setSortField(e.target.value as SortField)}
                  className="select w-full"
                >
                  <option value="createdAt">Date Created</option>
                  <option value="name">Name</option>
                  <option value="size">File Size</option>
                  <option value="rowCount">Row Count</option>
                </select>
              </div>

              {/* Sort Order */}
              <div>
                <label className="label">Order</label>
                <button
                  onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                  className="btn btn-secondary w-full flex items-center justify-center space-x-2"
                >
                  {sortOrder === 'asc' ? (
                    <>
                      <SortAsc className="w-4 h-4" />
                      <span>Ascending</span>
                    </>
                  ) : (
                    <>
                      <SortDesc className="w-4 h-4" />
                      <span>Descending</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bulk Actions */}
      {selectedDatasets.length > 0 && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center justify-between animate-slide-in-down">
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-blue-900">
              {selectedDatasets.length} selected
            </span>
            <button
              onClick={handleSelectAll}
              className="text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              {selectedDatasets.length === paginatedDatasets.length
                ? 'Deselect All'
                : 'Select All'}
            </button>
          </div>
          <div className="flex items-center space-x-2">
            <button className="btn btn-secondary btn-sm flex items-center space-x-2">
              <Download className="w-4 h-4" />
              <span>Export</span>
            </button>
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="btn btn-danger btn-sm flex items-center space-x-2"
            >
              <Trash2 className="w-4 h-4" />
              <span>Delete</span>
            </button>
          </div>
        </div>
      )}

      {/* Dataset Grid/List */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {[...Array(8)].map((_, index) => (
            <div key={index} className="card card-body space-y-4">
              <div className="skeleton h-4 w-3/4 rounded"></div>
              <div className="skeleton h-24 w-full rounded"></div>
              <div className="skeleton h-4 w-1/2 rounded"></div>
            </div>
          ))}
        </div>
      ) : filteredAndSortedDatasets.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            {searchQuery || filterType !== 'all'
              ? 'No datasets found'
              : 'No datasets yet'}
          </h3>
          <p className="text-gray-600 mb-6">
            {searchQuery || filterType !== 'all'
              ? 'Try adjusting your search or filters'
              : 'Upload your first dataset to get started with data analysis'}
          </p>
          {!searchQuery && filterType === 'all' && (
            <button onClick={onUpload} className="btn btn-primary">
              <Plus className="w-4 h-4 mr-2" />
              Upload Dataset
            </button>
          )}
        </div>
      ) : viewMode === 'grid' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {paginatedDatasets.map((dataset) => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              isSelected={selectedDatasets.includes(dataset.id)}
              onSelect={handleSelectDataset}
              onView={handleViewDataset}
            />
          ))}
        </div>
      ) : (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="table">
            <thead>
              <tr>
                <th className="w-12">
                  <input
                    type="checkbox"
                    checked={
                      selectedDatasets.length === paginatedDatasets.length &&
                      paginatedDatasets.length > 0
                    }
                    onChange={handleSelectAll}
                    className="w-4 h-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
                  />
                </th>
                <th
                  onClick={() => handleSort('name')}
                  className="cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center space-x-2">
                    <span>Name</span>
                    {sortField === 'name' &&
                      (sortOrder === 'asc' ? (
                        <SortAsc className="w-4 h-4" />
                      ) : (
                        <SortDesc className="w-4 h-4" />
                      ))}
                  </div>
                </th>
                <th>Type</th>
                <th
                  onClick={() => handleSort('size')}
                  className="cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center space-x-2">
                    <span>Size</span>
                    {sortField === 'size' &&
                      (sortOrder === 'asc' ? (
                        <SortAsc className="w-4 h-4" />
                      ) : (
                        <SortDesc className="w-4 h-4" />
                      ))}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('rowCount')}
                  className="cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center space-x-2">
                    <span>Rows</span>
                    {sortField === 'rowCount' &&
                      (sortOrder === 'asc' ? (
                        <SortAsc className="w-4 h-4" />
                      ) : (
                        <SortDesc className="w-4 h-4" />
                      ))}
                  </div>
                </th>
                <th
                  onClick={() => handleSort('createdAt')}
                  className="cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center space-x-2">
                    <span>Created</span>
                    {sortField === 'createdAt' &&
                      (sortOrder === 'asc' ? (
                        <SortAsc className="w-4 h-4" />
                      ) : (
                        <SortDesc className="w-4 h-4" />
                      ))}
                  </div>
                </th>
                <th>Status</th>
                <th className="w-12"></th>
              </tr>
            </thead>
            <tbody>
              {paginatedDatasets.map((dataset) => (
                <tr key={dataset.id} className="hover:bg-gray-50">
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedDatasets.includes(dataset.id)}
                      onChange={() => handleSelectDataset(dataset.id)}
                      className="w-4 h-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded cursor-pointer"
                    />
                  </td>
                  <td>
                    <button
                      onClick={() => handleViewDataset(dataset.id)}
                      className="font-medium text-blue-600 hover:text-blue-700 text-left"
                    >
                      {dataset.name}
                    </button>
                  </td>
                  <td>
                    <span className="badge badge-gray uppercase">
                      {dataset.fileType}
                    </span>
                  </td>
                  <td>{((dataset.size ?? 0) / 1024 / 1024).toFixed(2)} MB</td>
                  <td>{(dataset.rowCount ?? 0).toLocaleString()}</td>
                  <td>
                    {new Date(dataset.createdAt).toLocaleDateString('en-US', {
                      month: 'short',
                      day: 'numeric',
                      year: 'numeric',
                    })}
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        dataset.status === 'completed'
                          ? 'badge-success'
                          : dataset.status === 'processing'
                            ? 'badge-warning'
                            : dataset.status === 'failed'
                              ? 'badge-danger'
                              : 'badge-gray'
                      }`}
                    >
                      {dataset.status}
                    </span>
                  </td>
                  <td>
                    <button className="p-1 hover:bg-gray-100 rounded transition-colors">
                      <MoreVertical className="w-4 h-4 text-gray-500" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between bg-white rounded-lg border border-gray-200 px-4 py-3">
          <div className="text-sm text-gray-600">
            Showing {(currentPage - 1) * itemsPerPage + 1} to{' '}
            {Math.min(currentPage * itemsPerPage, filteredAndSortedDatasets.length)} of{' '}
            {filteredAndSortedDatasets.length} results
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
              disabled={currentPage === 1}
              className="btn btn-secondary btn-sm"
            >
              Previous
            </button>
            <div className="flex items-center space-x-1">
              {[...Array(totalPages)].map((_, index) => {
                const page = index + 1;
                if (
                  page === 1 ||
                  page === totalPages ||
                  (page >= currentPage - 1 && page <= currentPage + 1)
                ) {
                  return (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={`w-8 h-8 rounded-lg text-sm font-medium transition-colors ${
                        page === currentPage
                          ? 'bg-blue-600 text-white'
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      {page}
                    </button>
                  );
                } else if (page === currentPage - 2 || page === currentPage + 2) {
                  return (
                    <span key={page} className="text-gray-400 px-1">
                      ...
                    </span>
                  );
                }
                return null;
              })}
            </div>
            <button
              onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={currentPage === totalPages}
              className="btn btn-secondary btn-sm"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="modal-overlay" onClick={() => setShowDeleteConfirm(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                Delete {selectedDatasets.length} dataset(s)?
              </h3>
              <p className="text-gray-600 mb-6">
                This action cannot be undone. All data and analysis results will be
                permanently deleted.
              </p>
              <div className="flex items-center justify-end space-x-3">
                <button
                  onClick={() => setShowDeleteConfirm(false)}
                  className="btn btn-secondary"
                >
                  Cancel
                </button>
                <button onClick={handleBulkDelete} className="btn btn-danger">
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DatasetList;
