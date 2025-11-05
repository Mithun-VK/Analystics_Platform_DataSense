// src/pages/DatasetDetailPage.tsx

import  { useState, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  Download,
  Share2,
  Trash2,
  BarChart3,
  RefreshCw,
  Copy,
  Check,
  AlertCircle,
  Columns,
  Rows,
  HardDrive,
  Clock,
  User,
  Tag,
  Search,
  Lock,
  Globe,
} from 'lucide-react';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import Button from '@/components/shared/Button';
import Modal, { ConfirmModal } from '@/components/shared/Modal';
import { CardSkeleton } from '@/components/shared/Loading';
import { useDatasets } from '@/hooks/useDatasets';
import { formatDistanceToNow } from 'date-fns';

interface Column {
  name: string;
  type: string;
  nullable: boolean;
  sampleValues?: any[];
}

interface DatasetInfo {
  id: string;
  name: string;
  description?: string;
  size: number;
  rowCount: number;
  columnCount: number;
  columns: Column[];
  createdAt: string;
  updatedAt: string;
  owner: {
    id: string;
    name: string;
    email: string;
  };
  tags?: string[];
  isPublic: boolean;
  status: 'ready' | 'processing' | 'failed';
  preview?: any[][];
}

/**
 * DatasetDetailPage - Detailed view of single dataset with actions and preview
 * Features: Preview table, column info, metadata, actions, sharing
 * Responsive design with comprehensive dataset management
 */
const DatasetDetailPage: React.FC = () => {
  const { datasetId } = useParams<{ datasetId: string }>();
  const navigate = useNavigate();
  const { datasets, deleteDataset } = useDatasets();

  // Find dataset
  const dataset = useMemo(() => {
    return datasets.find((d) => d.id === datasetId);
  }, [datasets, datasetId]);

  // State management
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showShareModal, setShowShareModal] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedColumns, setSelectedColumns] = useState<Set<string>>(
    new Set()
  );
  const [isDeleting, setIsDeleting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [activeTab, setActiveTab] = useState<'preview' | 'columns' | 'info'>(
    'preview'
  );

  // Filter columns based on search
  const filteredColumns = useMemo(() => {
    if (!dataset || !('columns' in dataset)) return [];
    return (dataset as DatasetInfo).columns.filter((col) =>
      col.name.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [dataset, searchQuery]);

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  // Handle delete
  const handleDelete = async () => {
    if (!datasetId) return;
    setIsDeleting(true);
    try {
      await deleteDataset(datasetId);
      navigate('/datasets');
    } catch (error) {
      console.error('Failed to delete dataset:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  // Copy share link
  const copyShareLink = () => {
    const shareUrl = `${window.location.origin}/datasets/shared/${datasetId}`;
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (!dataset) {
    return (
      <DashboardLayout>
        <div className="detail-page p-8">
          <div className="text-center">
            <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Dataset Not Found
            </h2>
            <p className="text-gray-600 mb-6">
              The dataset you're looking for doesn't exist or has been deleted.
            </p>
            <Button
              variant="primary"
              onClick={() => navigate('/datasets')}
              leftIcon={ArrowLeft}
            >
              Back to Datasets
            </Button>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  const typedDataset = dataset as DatasetInfo;

  return (
    <DashboardLayout>
      <div className="detail-page">
        {/* Header */}
        <div className="detail-header">
          <div className="detail-header-left">
            <button
              onClick={() => navigate('/datasets')}
              className="detail-back-button"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </button>
            <div className="detail-header-title">
              <h1 className="detail-title">{typedDataset.name}</h1>
              <p className="detail-subtitle">{typedDataset.description}</p>
            </div>
          </div>

          <div className="detail-header-actions">
            <Button
              variant="secondary"
              size="sm"
              leftIcon={BarChart3}
              onClick={() => navigate(`/eda/${datasetId}`)}
            >
              Analyze
            </Button>
            <Button
              variant="secondary"
              size="sm"
              leftIcon={Share2}
              onClick={() => setShowShareModal(true)}
            >
              Share
            </Button>
            <Button
              variant="danger"
              size="sm"
              leftIcon={Trash2}
              onClick={() => setShowDeleteModal(true)}
            >
              Delete
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <div className="detail-tabs">
          <button
            onClick={() => setActiveTab('preview')}
            className={`detail-tab ${activeTab === 'preview' ? 'active' : ''}`}
          >
            Preview
          </button>
          <button
            onClick={() => setActiveTab('columns')}
            className={`detail-tab ${activeTab === 'columns' ? 'active' : ''}`}
          >
            Columns
          </button>
          <button
            onClick={() => setActiveTab('info')}
            className={`detail-tab ${activeTab === 'info' ? 'active' : ''}`}
          >
            Information
          </button>
        </div>

        {/* Content */}
        <div className="detail-content">
          {/* Preview Tab */}
          {activeTab === 'preview' && (
            <div className="detail-section">
              <div className="detail-section-header">
                <h2 className="detail-section-title">Data Preview</h2>
                <span className="detail-section-meta">
                  Showing first 100 rows
                </span>
              </div>

              {typedDataset.preview ? (
                <div className="detail-preview-container">
                  <div className="detail-preview-scroll">
                    <table className="detail-preview-table">
                      <thead>
                        <tr>
                          {typedDataset.columns.map((col) => (
                            <th
                              key={col.name}
                              className="detail-table-header"
                            >
                              {col.name}
                              <span className="detail-table-type">
                                {col.type}
                              </span>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {typedDataset.preview.slice(0, 20).map((row, idx) => (
                          <tr key={idx} className="detail-table-row">
                            {row.map((cell, cellIdx) => (
                              <td
                                key={cellIdx}
                                className="detail-table-cell"
                                title={String(cell)}
                              >
                                {cell === null ? (
                                  <span className="detail-null-value">
                                    NULL
                                  </span>
                                ) : (
                                  String(cell).substring(0, 50)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <CardSkeleton />
              )}

              {/* Preview Actions */}
              <div className="detail-preview-actions">
                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={Download}
                >
                  Download Preview
                </Button>
                <Button
                  variant="secondary"
                  size="sm"
                  leftIcon={RefreshCw}
                >
                  Refresh
                </Button>
              </div>
            </div>
          )}

          {/* Columns Tab */}
          {activeTab === 'columns' && (
            <div className="detail-section">
              <div className="detail-section-header">
                <h2 className="detail-section-title">
                  Columns ({typedDataset.columnCount})
                </h2>
                <div className="detail-search">
                  <Search className="w-4 h-4" />
                  <input
                    type="text"
                    placeholder="Search columns..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="detail-search-input"
                  />
                </div>
              </div>

              <div className="detail-columns-list">
                {filteredColumns.map((column, idx) => (
                  <div key={idx} className="detail-column-item">
                    <div className="detail-column-header">
                      <input
                        type="checkbox"
                        checked={selectedColumns.has(column.name)}
                        onChange={(e) => {
                          const newSelected = new Set(selectedColumns);
                          if (e.target.checked) {
                            newSelected.add(column.name);
                          } else {
                            newSelected.delete(column.name);
                          }
                          setSelectedColumns(newSelected);
                        }}
                        className="detail-checkbox"
                      />
                      <div className="detail-column-info">
                        <p className="detail-column-name">{column.name}</p>
                        <span className="detail-column-type">
                          {column.type}
                        </span>
                      </div>
                      <div className="detail-column-meta">
                        {column.nullable && (
                          <span className="detail-column-badge nullable">
                            Nullable
                          </span>
                        )}
                      </div>
                    </div>

                    {column.sampleValues && (
                      <div className="detail-column-samples">
                        <p className="text-xs font-medium text-gray-700 mb-2">
                          Sample Values:
                        </p>
                        <div className="detail-samples-grid">
                          {column.sampleValues.slice(0, 5).map((val, i) => (
                            <span key={i} className="detail-sample-value">
                              {val}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {filteredColumns.length === 0 && (
                <div className="detail-empty-state">
                  <Columns className="w-12 h-12 text-gray-400" />
                  <p>No columns found</p>
                </div>
              )}
            </div>
          )}

          {/* Information Tab */}
          {activeTab === 'info' && (
            <div className="detail-section">
              <h2 className="detail-section-title mb-6">Dataset Information</h2>

              <div className="detail-info-grid">
                {/* Stats */}
                <div className="detail-info-card">
                  <h3 className="detail-info-title">Statistics</h3>
                  <div className="detail-info-items">
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        <Rows className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="detail-info-label">Total Rows</p>
                        <p className="detail-info-value">
                          {typedDataset.rowCount.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        <Columns className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="detail-info-label">Total Columns</p>
                        <p className="detail-info-value">
                          {typedDataset.columnCount}
                        </p>
                      </div>
                    </div>
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        <HardDrive className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="detail-info-label">File Size</p>
                        <p className="detail-info-value">
                          {formatFileSize(typedDataset.size)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Timeline */}
                <div className="detail-info-card">
                  <h3 className="detail-info-title">Timeline</h3>
                  <div className="detail-info-items">
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        <Clock className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="detail-info-label">Created</p>
                        <p className="detail-info-value">
                          {formatDistanceToNow(
                            new Date(typedDataset.createdAt),
                            { addSuffix: true }
                          )}
                        </p>
                      </div>
                    </div>
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        <RefreshCw className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="detail-info-label">Last Modified</p>
                        <p className="detail-info-value">
                          {formatDistanceToNow(
                            new Date(typedDataset.updatedAt),
                            { addSuffix: true }
                          )}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Owner & Access */}
                <div className="detail-info-card">
                  <h3 className="detail-info-title">Access</h3>
                  <div className="detail-info-items">
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        <User className="w-5 h-5" />
                      </div>
                      <div>
                        <p className="detail-info-label">Owner</p>
                        <p className="detail-info-value">
                          {typedDataset.owner.name}
                        </p>
                        <p className="detail-info-email">
                          {typedDataset.owner.email}
                        </p>
                      </div>
                    </div>
                    <div className="detail-info-item">
                      <div className="detail-info-icon">
                        {typedDataset.isPublic ? (
                          <Globe className="w-5 h-5" />
                        ) : (
                          <Lock className="w-5 h-5" />
                        )}
                      </div>
                      <div>
                        <p className="detail-info-label">Visibility</p>
                        <p className="detail-info-value">
                          {typedDataset.isPublic ? 'Public' : 'Private'}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Tags */}
                {typedDataset.tags && typedDataset.tags.length > 0 && (
                  <div className="detail-info-card">
                    <h3 className="detail-info-title">Tags</h3>
                    <div className="detail-tags">
                      {typedDataset.tags.map((tag) => (
                        <span key={tag} className="detail-tag">
                          <Tag className="w-4 h-4" />
                          {tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Column Types Summary */}
              <div className="detail-info-card mt-6">
                <h3 className="detail-info-title">Column Types</h3>
                <div className="detail-column-types">
                  {Array.from(
                    new Set(typedDataset.columns.map((c) => c.type))
                  ).map((type) => {
                    const count = typedDataset.columns.filter(
                      (c) => c.type === type
                    ).length;
                    return (
                      <div key={type} className="detail-type-item">
                        <div className="detail-type-badge">{type}</div>
                        <span className="detail-type-count">
                          {count} column{count !== 1 ? 's' : ''}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Share Modal */}
        <Modal
          isOpen={showShareModal}
          onClose={() => setShowShareModal(false)}
          title="Share Dataset"
          size="md"
          footer={
            <Button
              variant="primary"
              onClick={() => setShowShareModal(false)}
            >
              Close
            </Button>
          }
        >
          <div className="detail-share-content">
            <div className="space-y-4">
              <div>
                <label className="label">Share Link</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={`${window.location.origin}/datasets/shared/${datasetId}`}
                    readOnly
                    className="input flex-1"
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
                <label className="label">Share Via</label>
                <div className="grid grid-cols-3 gap-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      window.open(
                        `mailto:?subject=Check out this dataset&body=${window.location.origin}/datasets/shared/${datasetId}`
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
                <label className="label">Visibility</label>
                <div className="space-y-2">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="radio"
                      name="visibility"
                      defaultChecked={!typedDataset.isPublic}
                      className="detail-radio"
                    />
                    <span>Private (Only me)</span>
                  </label>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="radio"
                      name="visibility"
                      defaultChecked={typedDataset.isPublic}
                      className="detail-radio"
                    />
                    <span>Public (Anyone with link)</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </Modal>

        {/* Delete Confirmation */}
        <ConfirmModal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          onConfirm={handleDelete}
          title="Delete Dataset"
          message={`Are you sure you want to delete "${typedDataset.name}"? This action cannot be undone.`}
          confirmText="Delete"
          variant="danger"
          isLoading={isDeleting}
        />
      </div>
    </DashboardLayout>
  );
};

DatasetDetailPage.displayName = 'DatasetDetailPage';

export default DatasetDetailPage;
