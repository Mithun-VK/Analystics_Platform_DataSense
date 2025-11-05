// src/components/datasets/DataUpload.tsx

import { useState, useRef, useCallback } from 'react';
import {
  Upload,
  X,
  File,
  CheckCircle,
  AlertCircle,
  Loader2,
  FileSpreadsheet,
  FileText,
  FileJson,
} from 'lucide-react';
import { useDatasets } from '@/hooks/useDatasets';

interface DataUploadProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: () => void;
}

interface UploadedFile {
  file: File;
  id: string;
  progress: number;
  status: 'pending' | 'uploading' | 'success' | 'error';
  errorMessage?: string;
  preview?: string;
}

// Allowed file types and size limits
const ALLOWED_FILE_TYPES = {
  'text/csv': ['.csv'],
  'application/vnd.ms-excel': ['.xls'],
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
  'application/json': ['.json'],
};

const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB
const MAX_FILES = 5;

/**
 * DataUpload - Modal component for uploading datasets
 * Features: Drag & drop, multiple file upload, progress tracking, validation
 */
const DataUpload: React.FC<DataUploadProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  const { uploadDataset } = useDatasets();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // Validate file type
  const isValidFileType = (file: File): boolean => {
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    return Object.values(ALLOWED_FILE_TYPES)
      .flat()
      .includes(fileExtension);
  };

  // Validate file size
  const isValidFileSize = (file: File): boolean => {
    return file.size <= MAX_FILE_SIZE;
  };

  // Get file icon
  const getFileIcon = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'csv':
        return <FileText className="w-8 h-8 text-green-600" />;
      case 'xls':
      case 'xlsx':
        return <FileSpreadsheet className="w-8 h-8 text-blue-600" />;
      case 'json':
        return <FileJson className="w-8 h-8 text-purple-600" />;
      default:
        return <File className="w-8 h-8 text-gray-600" />;
    }
  };

  // Handle file selection
  const handleFileSelect = useCallback(
    (files: FileList | null) => {
      if (!files) return;

      const newFiles: UploadedFile[] = [];
      const fileArray = Array.from(files);

      // Check max files limit
      if (uploadedFiles.length + fileArray.length > MAX_FILES) {
        alert(`You can only upload up to ${MAX_FILES} files at once.`);
        return;
      }

      fileArray.forEach((file) => {
        // Validate file type
        if (!isValidFileType(file)) {
          alert(
            `Invalid file type: ${file.name}. Please upload CSV, Excel, or JSON files.`
          );
          return;
        }

        // Validate file size
        if (!isValidFileSize(file)) {
          alert(
            `File too large: ${file.name}. Maximum size is ${
              MAX_FILE_SIZE / 1024 / 1024
            }MB.`
          );
          return;
        }

        // Add file to upload queue
        newFiles.push({
          file,
          id: Math.random().toString(36).substr(2, 9),
          progress: 0,
          status: 'pending',
        });
      });

      setUploadedFiles((prev) => [...prev, ...newFiles]);
    },
    [uploadedFiles]
  );

  // Handle drag events
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    handleFileSelect(files);
  };

  // Handle file input change
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files);
  };

  // Remove file from upload queue
  const handleRemoveFile = (id: string) => {
    setUploadedFiles((prev) => prev.filter((f) => f.id !== id));
  };

  // Upload single file with progress
  const uploadSingleFile = async (uploadedFile: UploadedFile): Promise<void> => {
    try {
      // Update status to uploading
      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === uploadedFile.id ? { ...f, status: 'uploading', progress: 0 } : f
        )
      );

      // Simulate progress (in real app, use XMLHttpRequest or axios with onUploadProgress)
      const progressInterval = setInterval(() => {
        setUploadedFiles((prev) =>
          prev.map((f) => {
            if (f.id === uploadedFile.id && f.progress < 90) {
              return { ...f, progress: f.progress + 10 };
            }
            return f;
          })
        );
      }, 200);

      // ✅ Pass File object directly, not FormData
      await uploadDataset(uploadedFile.file, {
        name: uploadedFile.file.name,
        description: `Uploaded on ${new Date().toLocaleDateString()}`,
      });

      clearInterval(progressInterval);

      // Update status to success
      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === uploadedFile.id
            ? { ...f, status: 'success', progress: 100 }
            : f
        )
      );
    } catch (error: any) {
      // Update status to error
      setUploadedFiles((prev) =>
        prev.map((f) =>
          f.id === uploadedFile.id
            ? {
                ...f,
                status: 'error',
                errorMessage: error.message || 'Upload failed',
              }
            : f
        )
      );
    }
  };

  // Upload all files
  const handleUploadAll = async () => {
    const filesToUpload = uploadedFiles.filter((f) => f.status === 'pending');

    if (filesToUpload.length === 0) {
      return;
    }

    setIsUploading(true);

    try {
      // Upload files sequentially (can be made parallel if needed)
      for (const file of filesToUpload) {
        await uploadSingleFile(file);
      }

      // Call success callback after all uploads complete
      setTimeout(() => {
        const allSuccess = uploadedFiles.every(
          (f) => f.status === 'success' || f.status === 'error'
        );
        if (allSuccess) {
          onUploadSuccess?.();
        }
      }, 500);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle close
  const handleClose = () => {
    if (isUploading) {
      if (!confirm('Upload in progress. Are you sure you want to cancel?')) {
        return;
      }
    }
    setUploadedFiles([]);
    onClose();
  };

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
  };

  if (!isOpen) return null;

  const hasFiles = uploadedFiles.length > 0;
  const allUploaded = uploadedFiles.every((f) => f.status === 'success');
  const hasErrors = uploadedFiles.some((f) => f.status === 'error');
  const pendingFiles = uploadedFiles.filter((f) => f.status === 'pending').length;

  return (
    <div className="modal-overlay animate-fade-in" onClick={handleClose}>
      <div
        className="modal-content max-w-3xl animate-slide-in-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
          <div>
            <h2 className="text-xl font-bold text-gray-900">Upload Dataset</h2>
            <p className="text-sm text-gray-600 mt-1">
              Support for CSV, Excel, and JSON files (Max {MAX_FILE_SIZE / 1024 / 1024}
              MB)
            </p>
          </div>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            disabled={isUploading}
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-6 max-h-[60vh] overflow-y-auto">
          {/* Drop Zone */}
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-200 ${
              isDragging
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
            }`}
          >
            <div className="flex flex-col items-center space-y-4">
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center transition-colors ${
                  isDragging ? 'bg-blue-100' : 'bg-gray-100'
                }`}
              >
                <Upload
                  className={`w-8 h-8 ${
                    isDragging ? 'text-blue-600' : 'text-gray-600'
                  }`}
                />
              </div>

              <div>
                <p className="text-lg font-semibold text-gray-900 mb-1">
                  {isDragging ? 'Drop files here' : 'Drag & drop files here'}
                </p>
                <p className="text-sm text-gray-600">
                  or{' '}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    browse files
                  </button>
                </p>
              </div>

              <div className="flex items-center space-x-4 text-xs text-gray-500">
                <span className="flex items-center space-x-1">
                  <FileText className="w-4 h-4" />
                  <span>CSV</span>
                </span>
                <span className="flex items-center space-x-1">
                  <FileSpreadsheet className="w-4 h-4" />
                  <span>Excel</span>
                </span>
                <span className="flex items-center space-x-1">
                  <FileJson className="w-4 h-4" />
                  <span>JSON</span>
                </span>
              </div>
            </div>
          </div>

          {/* Hidden File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".csv,.xls,.xlsx,.json"
            onChange={handleFileInputChange}
            className="hidden"
          />

          {/* Uploaded Files List */}
          {hasFiles && (
            <div className="mt-6 space-y-3">
              <h3 className="text-sm font-semibold text-gray-900">
                Files ({uploadedFiles.length}/{MAX_FILES})
              </h3>

              {uploadedFiles.map((uploadedFile) => (
                <div
                  key={uploadedFile.id}
                  className="bg-gray-50 rounded-lg p-4 border border-gray-200"
                >
                  <div className="flex items-start space-x-4">
                    {/* File Icon */}
                    <div className="flex-shrink-0">
                      {getFileIcon(uploadedFile.file.name)}
                    </div>

                    {/* File Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-gray-900 truncate">
                            {uploadedFile.file.name}
                          </p>
                          <p className="text-xs text-gray-500">
                            {formatFileSize(uploadedFile.file.size)}
                          </p>
                        </div>

                        {/* Status Icon */}
                        <div className="ml-4 flex-shrink-0">
                          {uploadedFile.status === 'success' && (
                            <CheckCircle className="w-5 h-5 text-green-600" />
                          )}
                          {uploadedFile.status === 'error' && (
                            <AlertCircle className="w-5 h-5 text-red-600" />
                          )}
                          {uploadedFile.status === 'uploading' && (
                            <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                          )}
                          {uploadedFile.status === 'pending' && (
                            <button
                              onClick={() => handleRemoveFile(uploadedFile.id)}
                              className="p-1 hover:bg-gray-200 rounded transition-colors"
                            >
                              <X className="w-4 h-4 text-gray-500" />
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Progress Bar */}
                      {(uploadedFile.status === 'uploading' ||
                        uploadedFile.status === 'success') && (
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className={`h-1.5 rounded-full transition-all duration-300 ${
                              uploadedFile.status === 'success'
                                ? 'bg-green-600'
                                : 'bg-blue-600'
                            }`}
                            style={{ width: `${uploadedFile.progress}%` }}
                          />
                        </div>
                      )}

                      {/* Error Message */}
                      {uploadedFile.status === 'error' && (
                        <p className="text-xs text-red-600 mt-1">
                          {uploadedFile.errorMessage}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="text-sm text-gray-600">
            {hasFiles && (
              <>
                {allUploaded && (
                  <span className="flex items-center space-x-2 text-green-600 font-medium">
                    <CheckCircle className="w-4 h-4" />
                    <span>All files uploaded successfully</span>
                  </span>
                )}
                {hasErrors && !allUploaded && (
                  <span className="flex items-center space-x-2 text-red-600 font-medium">
                    <AlertCircle className="w-4 h-4" />
                    <span>Some files failed to upload</span>
                  </span>
                )}
                {!allUploaded && !hasErrors && (
                  <span>{pendingFiles} file(s) ready to upload</span>
                )}
              </>
            )}
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={handleClose}
              className="btn btn-secondary"
              disabled={isUploading}
            >
              {allUploaded ? 'Done' : 'Cancel'}
            </button>

            {!allUploaded && (
              <button
                onClick={handleUploadAll}
                disabled={isUploading || pendingFiles === 0}
                className="btn btn-primary flex items-center space-x-2"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    <span>Upload {pendingFiles > 0 ? `(${pendingFiles})` : ''}</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;
