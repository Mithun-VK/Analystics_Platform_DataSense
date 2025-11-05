// src/components/datasets/DatasetDetails.tsx

import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  Play,
  Trash2,
  Edit,
  Share2,
  MoreVertical,
  FileText,
  Database,
  HardDrive,
  Calendar,
  CheckCircle,
  AlertCircle,
  Clock,
  Eye,
  EyeOff,
  Filter,
  Search,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface Column {
  name: string;
  type: string;
  nullCount: number;
  uniqueCount: number;
}

interface Dataset {
  id: string;
  name: string;
  description?: string;
  size: number;
  rowCount: number;
  columnCount: number;
  fileType: string;
  createdAt: string;
  updatedAt: string;
  status: 'active' | 'processing' | 'error';
  createdBy: string;
  columns: Column[];
}

interface DatasetDetailsProps {
  dataset: Dataset;
  previewData?: any[][];
  onDelete?: () => void;
  onDownload?: () => void;
  onAnalyze?: () => void;
  onEdit?: () => void;
}

/**
 * DatasetDetails - Comprehensive dataset view with metadata and data preview
 * Features: Data preview table with pagination, column info, metadata cards
 * Supports search, filter, and column visibility controls
 */
const DatasetDetails: React.FC<DatasetDetailsProps> = ({
  dataset,
  previewData = [],
  onDelete,
  onDownload,
  onAnalyze,
  onEdit,
}) => {
  const navigate = useNavigate();

  // State management
  const [activeTab, setActiveTab] = useState<'preview' | 'columns' | 'metadata'>('preview');
  const [currentPage, setCurrentPage] = useState(1);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchQuery, setSearchQuery] = useState('');
  const [hiddenColumns, setHiddenColumns] = useState<Set<number>>(new Set());
  const [showMenu, setShowMenu] = useState(false);

  // Get column headers
  const headers = dataset.columns.map((col) => col.name);

  // Filter data based on search
  const filteredData = useMemo(() => {
    if (!searchQuery) return previewData;

    return previewData.filter((row) =>
      row.some((cell) =>
        String(cell).toLowerCase().includes(searchQuery.toLowerCase())
      )
    );
  }, [previewData, searchQuery]);

  // Paginate data
  const totalPages = Math.ceil(filteredData.length / rowsPerPage);
  const paginatedData = useMemo(() => {
    const startIndex = (currentPage - 1) * rowsPerPage;
    return filteredData.slice(startIndex, startIndex + rowsPerPage);
  }, [filteredData, currentPage, rowsPerPage]);

  // Toggle column visibility
  const toggleColumnVisibility = (index: number) => {
    const newHidden = new Set(hiddenColumns);
    if (newHidden.has(index)) {
      newHidden.delete(index);
    } else {
      newHidden.add(index);
    }
    setHiddenColumns(newHidden);
  };

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  // Get status badge
  const getStatusBadge = () => {
    switch (dataset.status) {
      case 'active':
        return (
          <span className="inline-flex items-center space-x-1 px-3 py-1 bg-green-100 text-green-700 text-sm font-medium rounded-full">
            <CheckCircle className="w-4 h-4" />
            <span>Active</span>
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center space-x-1 px-3 py-1 bg-yellow-100 text-yellow-700 text-sm font-medium rounded-full">
            <Clock className="w-4 h-4 animate-spin" />
            <span>Processing</span>
          </span>
        );
      case 'error':
        return (
          <span className="inline-flex items-center space-x-1 px-3 py-1 bg-red-100 text-red-700 text-sm font-medium rounded-full">
            <AlertCircle className="w-4 h-4" />
            <span>Error</span>
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <button
            onClick={() => navigate('/datasets')}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-5 h-5 text-gray-600" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{dataset.name}</h1>
            {dataset.description && (
              <p className="text-gray-600 mt-1">{dataset.description}</p>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {getStatusBadge()}

          {/* Actions Menu */}
          <div className="relative">
            <button
              onClick={() => setShowMenu(!showMenu)}
              className="btn btn-secondary flex items-center space-x-2"
            >
              <span>Actions</span>
              <MoreVertical className="w-4 h-4" />
            </button>

            {showMenu && (
              <>
                <div
                  className="fixed inset-0 z-10"
                  onClick={() => setShowMenu(false)}
                />
                <div className="absolute right-0 top-12 z-20 w-56 bg-white rounded-lg shadow-xl border border-gray-200 py-2 animate-slide-in-down">
                  {onAnalyze && (
                    <button
                      onClick={() => {
                        onAnalyze();
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
                      onClick={() => {
                        onDownload();
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      <span>Download Dataset</span>
                    </button>
                  )}
                  {onEdit && (
                    <button
                      onClick={() => {
                        onEdit();
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Edit className="w-4 h-4" />
                      <span>Edit Details</span>
                    </button>
                  )}
                  <button
                    onClick={() => setShowMenu(false)}
                    className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                  >
                    <Share2 className="w-4 h-4" />
                    <span>Share Dataset</span>
                  </button>
                  <div className="border-t border-gray-200 my-2"></div>
                  {onDelete && (
                    <button
                      onClick={() => {
                        onDelete();
                        setShowMenu(false);
                      }}
                      className="w-full flex items-center space-x-3 px-4 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                      <span>Delete Dataset</span>
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
              <Database className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Rows</p>
              <p className="text-xl font-bold text-gray-900">
                {dataset.rowCount.toLocaleString()}
              </p>
            </div>
          </div>
        </div>

        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Columns</p>
              <p className="text-xl font-bold text-gray-900">{dataset.columnCount}</p>
            </div>
          </div>
        </div>

        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <HardDrive className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">File Size</p>
              <p className="text-xl font-bold text-gray-900">
                {formatFileSize(dataset.size)}
              </p>
            </div>
          </div>
        </div>

        <div className="card card-body">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center">
              <Calendar className="w-5 h-5 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Created</p>
              <p className="text-sm font-medium text-gray-900">
                {formatDistanceToNow(new Date(dataset.createdAt), {
                  addSuffix: true,
                })}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6" aria-label="Tabs">
            <button
              onClick={() => setActiveTab('preview')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'preview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Data Preview
            </button>
            <button
              onClick={() => setActiveTab('columns')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'columns'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Column Info
            </button>
            <button
              onClick={() => setActiveTab('metadata')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'metadata'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Metadata
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {/* Data Preview Tab */}
          {activeTab === 'preview' && (
            <div className="space-y-4">
              {/* Toolbar */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div className="flex-1 relative max-w-md">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search in data..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div className="flex items-center space-x-3">
                  <select
                    value={rowsPerPage}
                    onChange={(e) => {
                      setRowsPerPage(Number(e.target.value));
                      setCurrentPage(1);
                    }}
                    className="select"
                  >
                    <option value={10}>10 rows</option>
                    <option value={25}>25 rows</option>
                    <option value={50}>50 rows</option>
                    <option value={100}>100 rows</option>
                  </select>

                  <button className="btn btn-secondary btn-sm flex items-center space-x-2">
                    <Filter className="w-4 h-4" />
                    <span>Columns</span>
                  </button>
                </div>
              </div>

              {/* Data Table */}
              <div className="overflow-x-auto border border-gray-200 rounded-lg">
                <table className="table">
                  <thead className="sticky top-0 z-10">
                    <tr>
                      {headers.map((header, index) => {
                        if (hiddenColumns.has(index)) return null;
                        return (
                          <th key={index} className="relative group">
                            <div className="flex items-center justify-between">
                              <span className="truncate">{header}</span>
                              <button
                                onClick={() => toggleColumnVisibility(index)}
                                className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity"
                              >
                                <EyeOff className="w-4 h-4 text-gray-400" />
                              </button>
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedData.length > 0 ? (
                      paginatedData.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {row.map((cell, cellIndex) => {
                            if (hiddenColumns.has(cellIndex)) return null;
                            return (
                              <td key={cellIndex} className="max-w-xs truncate">
                                {cell !== null && cell !== undefined
                                  ? String(cell)
                                  : '-'}
                              </td>
                            );
                          })}
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td
                          colSpan={headers.length - hiddenColumns.size}
                          className="text-center py-12"
                        >
                          <p className="text-gray-500">
                            {searchQuery
                              ? 'No matching results found'
                              : 'No preview data available'}
                          </p>
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalPages > 1 && (
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-600">
                    Showing {(currentPage - 1) * rowsPerPage + 1} to{' '}
                    {Math.min(currentPage * rowsPerPage, filteredData.length)} of{' '}
                    {filteredData.length} rows
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                      disabled={currentPage === 1}
                      className="btn btn-secondary btn-sm"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="text-sm text-gray-600">
                      Page {currentPage} of {totalPages}
                    </span>
                    <button
                      onClick={() =>
                        setCurrentPage((prev) => Math.min(totalPages, prev + 1))
                      }
                      disabled={currentPage === totalPages}
                      className="btn btn-secondary btn-sm"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Column Info Tab */}
          {activeTab === 'columns' && (
            <div className="space-y-4">
              <div className="overflow-x-auto">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Column Name</th>
                      <th>Data Type</th>
                      <th>Null Count</th>
                      <th>Unique Values</th>
                      <th>Completeness</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {dataset.columns.map((column, index) => {
                      const completeness =
                        ((dataset.rowCount - column.nullCount) /
                          dataset.rowCount) *
                        100;
                      return (
                        <tr key={index}>
                          <td className="font-medium">{column.name}</td>
                          <td>
                            <span className="badge badge-gray uppercase">
                              {column.type}
                            </span>
                          </td>
                          <td>{column.nullCount.toLocaleString()}</td>
                          <td>{column.uniqueCount.toLocaleString()}</td>
                          <td>
                            <div className="flex items-center space-x-2">
                              <div className="flex-1 w-24 bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    completeness >= 90
                                      ? 'bg-green-600'
                                      : completeness >= 70
                                      ? 'bg-yellow-600'
                                      : 'bg-red-600'
                                  }`}
                                  style={{ width: `${completeness}%` }}
                                />
                              </div>
                              <span className="text-sm text-gray-600">
                                {completeness.toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td>
                            <button
                              onClick={() => toggleColumnVisibility(index)}
                              className="p-1 hover:bg-gray-100 rounded transition-colors"
                            >
                              {hiddenColumns.has(index) ? (
                                <Eye className="w-4 h-4 text-gray-500" />
                              ) : (
                                <EyeOff className="w-4 h-4 text-gray-500" />
                              )}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Metadata Tab */}
          {activeTab === 'metadata' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="font-semibold text-gray-900">General Information</h3>
                  <dl className="space-y-3">
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">File Name:</dt>
                      <dd className="text-sm font-medium text-gray-900">
                        {dataset.name}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">File Type:</dt>
                      <dd className="text-sm font-medium text-gray-900 uppercase">
                        {dataset.fileType}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">File Size:</dt>
                      <dd className="text-sm font-medium text-gray-900">
                        {formatFileSize(dataset.size)}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">Created By:</dt>
                      <dd className="text-sm font-medium text-gray-900">
                        {dataset.createdBy}
                      </dd>
                    </div>
                  </dl>
                </div>

                <div className="space-y-4">
                  <h3 className="font-semibold text-gray-900">Timestamps</h3>
                  <dl className="space-y-3">
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">Created:</dt>
                      <dd className="text-sm font-medium text-gray-900">
                        {new Date(dataset.createdAt).toLocaleString()}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">Last Updated:</dt>
                      <dd className="text-sm font-medium text-gray-900">
                        {new Date(dataset.updatedAt).toLocaleString()}
                      </dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="text-sm text-gray-600">Time Ago:</dt>
                      <dd className="text-sm font-medium text-gray-900">
                        {formatDistanceToNow(new Date(dataset.createdAt), {
                          addSuffix: true,
                        })}
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>

              {dataset.description && (
                <div className="space-y-2">
                  <h3 className="font-semibold text-gray-900">Description</h3>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    {dataset.description}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DatasetDetails;
