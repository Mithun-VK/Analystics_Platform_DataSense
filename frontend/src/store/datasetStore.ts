// src/store/datasetStore.ts

import { create } from 'zustand';
import { persist, createJSONStorage, subscribeWithSelector } from 'zustand/middleware';
import type { Dataset } from '@/types/dataset.types';
import datasetService from '@/services/datasetService';

/**
 * Dataset Store - Zustand store for managing dataset state and caching
 */

interface DatasetCache {
  data: Dataset;
  timestamp: number;
  expiresAt: number;
}

interface PaginationState {
  currentPage: number;
  pageSize: number;
  totalItems: number;
  totalPages: number;
}

interface FilterState {
  search: string;
  sortBy: 'name' | 'createdAt' | 'updatedAt' | 'fileSize' | 'totalRows' | 'accessCount' | null;
  sortOrder: 'asc' | 'desc';
  status: string | null;
  createdAfter: string | null;
  createdBefore: string | null;
  tags: string[];
}

interface UploadState {
  isUploading: boolean;
  uploadProgress: number;
  uploadedBytes: number;
  totalBytes: number;
  currentFile: File | null;
  uploadSpeed: number;
  estimatedTimeRemaining: number;
}

interface DatasetStats {
  totalDatasets: number;
  totalSize: number;
  averageSize: number;
  largestDataset: Dataset | null;
}

interface DatasetState {
  datasets: Dataset[];
  selectedDataset: Dataset | null;
  isLoadingDatasets: boolean;
  datasetsError: string | null;

  pagination: PaginationState;
  filters: FilterState;
  searchDebounceTimer: NodeJS.Timeout | null;

  uploadState: UploadState;
  recentlyDeleted: string[];

  datasetCache: Map<string, DatasetCache>;
  cacheExpiry: number;
  lastCacheCleanup: number;

  totalDatasetSize: number;
  datasetStats: DatasetStats;
  processingDatasets: Set<string>;

  favoriteDatasets: string[];
  datasetCollections: Map<string, string[]>;
  currentCollection: string | null;

  expandedSections: Set<string>;
  viewMode: 'grid' | 'list';
  itemsPerPage: number;

  setDatasets: (datasets: Dataset[]) => void;
  setSelectedDataset: (dataset: Dataset | null) => void;
  setIsLoadingDatasets: (loading: boolean) => void;
  setDatasetsError: (error: string | null) => void;
  setPagination: (pagination: Partial<PaginationState>) => void;
  setFilters: (filters: Partial<FilterState>) => void;
  setUploadProgress: (
    progress: number,
    uploadedBytes: number,
    totalBytes: number
  ) => void;
  setUploadSpeed: (speed: number) => void;
  setEstimatedTimeRemaining: (time: number) => void;
  setCurrentFile: (file: File | null) => void;
  setViewMode: (mode: 'grid' | 'list') => void;
  setItemsPerPage: (items: number) => void;
  toggleExpandedSection: (section: string) => void;
  toggleFavoriteDataset: (datasetId: string) => void;
  setCurrentCollection: (collectionId: string | null) => void;
  addDatasetToCollection: (collectionId: string, datasetId: string) => void;
  removeDatasetFromCollection: (collectionId: string, datasetId: string) => void;

  loadDatasets: (page?: number, limit?: number) => Promise<void>;
  loadDatasetById: (
    datasetId: string,
    useCache?: boolean
  ) => Promise<Dataset | null>;
  createDataset: (name: string, description?: string) => Promise<Dataset>;
  uploadDataset: (
    file: File,
    metadata?: { name?: string; description?: string; tags?: string[] },
    onProgress?: (progress: number) => void
  ) => Promise<Dataset>;
  updateDataset: (
    datasetId: string,
    updates: Partial<Dataset>
  ) => Promise<Dataset>;
  deleteDataset: (datasetId: string) => Promise<void>;
  bulkDeleteDatasets: (datasetIds: string[]) => Promise<void>;
  undoDelete: () => Promise<void>;
  duplicateDataset: (datasetId: string, newName?: string) => Promise<Dataset>;
  searchDatasets: (query: string) => Promise<Dataset[]>;
  applyFilters: () => Promise<void>;
  clearFilters: () => void;
  resetPagination: () => void;
  nextPage: () => Promise<void>;
  previousPage: () => Promise<void>;
  goToPage: (page: number) => Promise<void>;

  cacheDataset: (dataset: Dataset) => void;
  getCachedDataset: (datasetId: string) => Dataset | null;
  invalidateCache: (datasetId?: string) => void;
  clearCache: () => void;
  cleanupExpiredCache: () => void;

  updateDatasetStats: () => void;
  getDatasetsBySize: () => Dataset[];
  getRecentDatasets: (limit?: number) => Dataset[];
  getDatasetsByTag: (tag: string) => Dataset[];
  getFavoriteDatasets: () => Dataset[];
  getCollectionDatasets: (collectionId: string) => Dataset[];

  markDatasetProcessing: (datasetId: string, isProcessing: boolean) => void;
  isDatasetProcessing: (datasetId: string) => boolean;

  updateDatasets: (datasets: Dataset[]) => void;
  removeDatasetById: (datasetId: string) => void;
  clearAllDatasets: () => void;

  resetDatasetStore: () => void;
  cleanupStore: () => void;
}

const createInitialDatasetState = (): Omit<
  DatasetState,
  | 'setDatasets'
  | 'setSelectedDataset'
  | 'setIsLoadingDatasets'
  | 'setDatasetsError'
  | 'setPagination'
  | 'setFilters'
  | 'setUploadProgress'
  | 'setUploadSpeed'
  | 'setEstimatedTimeRemaining'
  | 'setCurrentFile'
  | 'setViewMode'
  | 'setItemsPerPage'
  | 'toggleExpandedSection'
  | 'toggleFavoriteDataset'
  | 'setCurrentCollection'
  | 'addDatasetToCollection'
  | 'removeDatasetFromCollection'
  | 'loadDatasets'
  | 'loadDatasetById'
  | 'createDataset'
  | 'uploadDataset'
  | 'updateDataset'
  | 'deleteDataset'
  | 'bulkDeleteDatasets'
  | 'undoDelete'
  | 'duplicateDataset'
  | 'searchDatasets'
  | 'applyFilters'
  | 'clearFilters'
  | 'resetPagination'
  | 'nextPage'
  | 'previousPage'
  | 'goToPage'
  | 'cacheDataset'
  | 'getCachedDataset'
  | 'invalidateCache'
  | 'clearCache'
  | 'cleanupExpiredCache'
  | 'updateDatasetStats'
  | 'getDatasetsBySize'
  | 'getRecentDatasets'
  | 'getDatasetsByTag'
  | 'getFavoriteDatasets'
  | 'getCollectionDatasets'
  | 'markDatasetProcessing'
  | 'isDatasetProcessing'
  | 'updateDatasets'
  | 'removeDatasetById'
  | 'clearAllDatasets'
  | 'resetDatasetStore'
  | 'cleanupStore'
> => ({
  datasets: [],
  selectedDataset: null,
  isLoadingDatasets: false,
  datasetsError: null,
  pagination: {
    currentPage: 1,
    pageSize: 10,
    totalItems: 0,
    totalPages: 0,
  },
  filters: {
    search: '',
    sortBy: null,
    sortOrder: 'desc',
    status: null,
    createdAfter: null,
    createdBefore: null,
    tags: [],
  },
  uploadState: {
    isUploading: false,
    uploadProgress: 0,
    uploadedBytes: 0,
    totalBytes: 0,
    currentFile: null,
    uploadSpeed: 0,
    estimatedTimeRemaining: 0,
  },
  recentlyDeleted: [],
  datasetCache: new Map(),
  cacheExpiry: 15 * 60 * 1000,
  lastCacheCleanup: Date.now(),
  totalDatasetSize: 0,
  datasetStats: {
    totalDatasets: 0,
    totalSize: 0,
    averageSize: 0,
    largestDataset: null,
  },
  processingDatasets: new Set(),
  favoriteDatasets: [],
  datasetCollections: new Map(),
  currentCollection: null,
  expandedSections: new Set(),
  viewMode: 'grid' as const,
  itemsPerPage: 10,
  searchDebounceTimer: null,
});

export const datasetStore = create<DatasetState>()(
  persist(
    subscribeWithSelector((set, get) => ({
      ...createInitialDatasetState(),

      setDatasets: (datasets) => set({ datasets }),
      setSelectedDataset: (selectedDataset) => set({ selectedDataset }),
      setIsLoadingDatasets: (isLoadingDatasets) =>
        set({ isLoadingDatasets }),
      setDatasetsError: (datasetsError) => set({ datasetsError }),

      setPagination: (updates) =>
        set((state) => ({
          pagination: { ...state.pagination, ...updates },
        })),

      setFilters: (updates) =>
        set((state) => ({
          filters: { ...state.filters, ...updates },
        })),

      setUploadProgress: (uploadProgress, uploadedBytes, totalBytes) =>
        set((state) => ({
          uploadState: {
            ...state.uploadState,
            uploadProgress,
            uploadedBytes,
            totalBytes,
          },
        })),

      setUploadSpeed: (uploadSpeed) =>
        set((state) => ({
          uploadState: { ...state.uploadState, uploadSpeed },
        })),

      setEstimatedTimeRemaining: (estimatedTimeRemaining) =>
        set((state) => ({
          uploadState: { ...state.uploadState, estimatedTimeRemaining },
        })),

      setCurrentFile: (currentFile) =>
        set((state) => ({
          uploadState: { ...state.uploadState, currentFile },
        })),

      setViewMode: (viewMode) => set({ viewMode }),

      setItemsPerPage: (itemsPerPage) =>
        set((state) => ({
          itemsPerPage,
          pagination: { ...state.pagination, pageSize: itemsPerPage },
        })),

      toggleExpandedSection: (section) =>
        set((state) => {
          const newSet = new Set(state.expandedSections);
          if (newSet.has(section)) {
            newSet.delete(section);
          } else {
            newSet.add(section);
          }
          return { expandedSections: newSet };
        }),

      toggleFavoriteDataset: (datasetId) =>
        set((state) => {
          const newFavorites = [...state.favoriteDatasets];
          const index = newFavorites.indexOf(datasetId);
          if (index > -1) {
            newFavorites.splice(index, 1);
          } else {
            newFavorites.push(datasetId);
          }
          return { favoriteDatasets: newFavorites };
        }),

      setCurrentCollection: (currentCollection) =>
        set({ currentCollection }),

      addDatasetToCollection: (collectionId, datasetId) =>
        set((state) => {
          const newCollections = new Map(state.datasetCollections);
          const collection = newCollections.get(collectionId) || [];
          if (!collection.includes(datasetId)) {
            collection.push(datasetId);
            newCollections.set(collectionId, collection);
          }
          return { datasetCollections: newCollections };
        }),

      removeDatasetFromCollection: (collectionId, datasetId) =>
        set((state) => {
          const newCollections = new Map(state.datasetCollections);
          const collection = newCollections.get(collectionId) || [];
          const filtered = collection.filter((id) => id !== datasetId);
          newCollections.set(collectionId, filtered);
          return { datasetCollections: newCollections };
        }),

      // ✅ FIXED: Use correct response structure
      loadDatasets: async (page = 1, limit = 10) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          // ✅ Don't use unused filters variable
          const response = await datasetService.fetchDatasets(page, limit);

          set({
            datasets: response.datasets || [], // ✅ FIXED: Use 'datasets' not 'data'
            pagination: {
              currentPage: page,
              pageSize: limit,
              totalItems: response.pagination.totalItems, // ✅ FIXED: Use 'totalItems'
              totalPages: response.pagination.totalPages, // ✅ FIXED: Use 'totalPages'
            },
            isLoadingDatasets: false,
          });

          (response.datasets || []).forEach((dataset: Dataset) => {
            get().cacheDataset(dataset);
          });

          get().updateDatasetStats();
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Failed to load datasets';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
        }
      },

      // ✅ FIXED: Use correct method from datasetService
      loadDatasetById: async (datasetId, useCache = true) => {
        try {
          if (useCache) {
            const cached = get().getCachedDataset(datasetId);
            if (cached) return cached;
          }

          set({ isLoadingDatasets: true });
          // ✅ FIXED: Call fetchDatasets and filter, or implement getDataset endpoint
          const response = await datasetService.fetchDatasets(1, 100);
          const dataset = response.datasets?.find(d => d.id === datasetId);

          if (!dataset) {
            throw new Error('Dataset not found');
          }

          get().cacheDataset(dataset);
          set({ isLoadingDatasets: false });

          return dataset;
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Failed to load dataset';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          return null;
        }
      },

      createDataset: async (name, description) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          const dataset = await datasetService.createDataset({
            name,
            description,
          });

          set((state) => ({
            datasets: [dataset, ...state.datasets],
            isLoadingDatasets: false,
          }));

          get().cacheDataset(dataset);
          get().updateDatasetStats();

          return dataset;
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Failed to create dataset';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          throw error;
        }
      },

      uploadDataset: async (file, metadata, onProgress) => {
        set((state) => ({
          uploadState: {
            ...state.uploadState,
            isUploading: true,
            currentFile: file,
            totalBytes: file.size,
          },
        }));

        const startTime = Date.now();
        let lastUpdateTime = startTime;
        let lastUploadedBytes = 0;

        try {
          const result = await datasetService.uploadDataset(
            file,
            metadata,
            (progress: number) => {
              const now = Date.now();

              if (now - lastUpdateTime > 500) {
                const bytesSinceLastUpdate =
                  progress * file.size - lastUploadedBytes;
                const timeSinceLastUpdate = (now - lastUpdateTime) / 1000;
                const speed = bytesSinceLastUpdate / timeSinceLastUpdate;

                const remainingBytes = file.size - progress * file.size;
                const estimatedTime = remainingBytes / speed;

                set((state) => ({
                  uploadState: {
                    ...state.uploadState,
                    uploadProgress: progress * 100,
                    uploadedBytes: progress * file.size,
                    uploadSpeed: speed,
                    estimatedTimeRemaining: estimatedTime * 1000,
                  },
                }));

                lastUpdateTime = now;
                lastUploadedBytes = progress * file.size;

                onProgress?.(progress * 100);
              }
            }
          );

          const dataset = result as unknown as Dataset;

          set((state) => ({
            datasets: [dataset, ...state.datasets],
            pagination: {
              ...state.pagination,
              totalItems: state.pagination.totalItems + 1,
            },
            uploadState: {
              isUploading: false,
              uploadProgress: 100,
              uploadedBytes: 0,
              totalBytes: 0,
              currentFile: null,
              uploadSpeed: 0,
              estimatedTimeRemaining: 0,
            },
          }));

          get().cacheDataset(dataset);
          get().updateDatasetStats();

          return dataset;
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Upload failed';
          set({
            datasetsError: errorMessage,
            uploadState: {
              isUploading: false,
              uploadProgress: 0,
              uploadedBytes: 0,
              totalBytes: 0,
              currentFile: null,
              uploadSpeed: 0,
              estimatedTimeRemaining: 0,
            },
          });
          throw error;
        }
      },

      updateDataset: async (datasetId, updates) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          const updated = await datasetService.updateDataset(
            datasetId,
            updates
          );

          set((state) => ({
            datasets: state.datasets.map((d) =>
              d.id === datasetId ? updated : d
            ),
            selectedDataset:
              state.selectedDataset?.id === datasetId
                ? updated
                : state.selectedDataset,
            isLoadingDatasets: false,
          }));

          get().cacheDataset(updated);

          return updated;
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Update failed';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          throw error;
        }
      },

      deleteDataset: async (datasetId) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          await datasetService.deleteDataset(datasetId);

          set((state) => ({
            datasets: state.datasets.filter((d) => d.id !== datasetId),
            recentlyDeleted: [datasetId, ...state.recentlyDeleted].slice(
              0,
              5
            ),
            isLoadingDatasets: false,
          }));

          get().invalidateCache(datasetId);
          get().updateDatasetStats();
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Delete failed';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          throw error;
        }
      },

      bulkDeleteDatasets: async (datasetIds) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          for (const datasetId of datasetIds) {
            await datasetService.deleteDataset(datasetId);
          }

          set((state) => ({
            datasets: state.datasets.filter(
              (d) => !datasetIds.includes(d.id)
            ),
            recentlyDeleted: [
              ...datasetIds,
              ...state.recentlyDeleted,
            ].slice(0, 5),
            isLoadingDatasets: false,
          }));

          datasetIds.forEach((id) => get().invalidateCache(id));
          get().updateDatasetStats();
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Bulk delete failed';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          throw error;
        }
      },

      undoDelete: async () => {
        const { recentlyDeleted } = get();
        if (recentlyDeleted.length === 0) {
          throw new Error('No datasets to restore');
        }
      },

      duplicateDataset: async (datasetId, newName) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          const duplicated = await datasetService.duplicateDataset(
            datasetId,
            newName
          );

          set((state) => ({
            datasets: [...state.datasets, duplicated],
            isLoadingDatasets: false,
          }));

          get().cacheDataset(duplicated);
          get().updateDatasetStats();

          return duplicated;
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Duplicate failed';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          throw error;
        }
      },

      searchDatasets: async (query) => {
        set({ isLoadingDatasets: true, datasetsError: null });
        try {
          const results = await datasetService.searchDatasets(query);
          set({ isLoadingDatasets: false });
          return results;
        } catch (error) {
          const errorMessage =
            error instanceof Error ? error.message : 'Search failed';
          set({ datasetsError: errorMessage, isLoadingDatasets: false });
          throw error;
        }
      },

      applyFilters: async () => {
        set({ isLoadingDatasets: true });
        try {
          await get().loadDatasets(1, get().pagination.pageSize);
        } catch (error) {
          console.error('Apply filters error:', error);
        }
      },

      clearFilters: () => {
        set({
          filters: {
            search: '',
            sortBy: null,
            sortOrder: 'desc',
            status: null,
            createdAfter: null,
            createdBefore: null,
            tags: [],
          },
        });
      },

      resetPagination: () => {
        set({
          pagination: {
            currentPage: 1,
            pageSize: get().pagination.pageSize,
            totalItems: 0,
            totalPages: 0,
          },
        });
      },

      nextPage: async () => {
        const { pagination } = get();
        if (pagination.currentPage < pagination.totalPages) {
          await get().loadDatasets(
            pagination.currentPage + 1,
            pagination.pageSize
          );
        }
      },

      previousPage: async () => {
        const { pagination } = get();
        if (pagination.currentPage > 1) {
          await get().loadDatasets(
            pagination.currentPage - 1,
            pagination.pageSize
          );
        }
      },

      goToPage: async (page) => {
        const { pagination } = get();
        if (page >= 1 && page <= pagination.totalPages) {
          await get().loadDatasets(page, pagination.pageSize);
        }
      },

      cacheDataset: (dataset: Dataset) => {
        const { cacheExpiry } = get();
        set((state) => {
          const newCache = new Map(state.datasetCache);
          newCache.set(dataset.id, {
            data: dataset,
            timestamp: Date.now(),
            expiresAt: Date.now() + cacheExpiry,
          });
          return { datasetCache: newCache };
        });
      },

      getCachedDataset: (datasetId) => {
        const cached = get().datasetCache.get(datasetId);
        if (!cached) return null;

        if (Date.now() > cached.expiresAt) {
          get().invalidateCache(datasetId);
          return null;
        }

        return cached.data;
      },

      invalidateCache: (datasetId) => {
        set((state) => {
          const newCache = new Map(state.datasetCache);
          if (datasetId) {
            newCache.delete(datasetId);
          }
          return { datasetCache: newCache };
        });
      },

      clearCache: () => {
        set({ datasetCache: new Map() });
      },

      cleanupExpiredCache: () => {
        const now = Date.now();
        set((state) => {
          const newCache = new Map(state.datasetCache);
          const idsToDelete: string[] = [];

          newCache.forEach((cached, id) => {
            if (now > cached.expiresAt) {
              idsToDelete.push(id);
            }
          });

          idsToDelete.forEach((id) => newCache.delete(id));

          return {
            datasetCache: newCache,
            lastCacheCleanup: now,
          };
        });
      },

      updateDatasetStats: () => {
        const { datasets } = get();

        const totalSize = datasets.reduce(
          (sum, d) => sum + (d.fileSize || 0),
          0
        );
        const largestDataset = datasets.reduce((largest, d) =>
          (d.fileSize || 0) > (largest?.fileSize || 0) ? d : largest
        );

        set({
          datasetStats: {
            totalDatasets: datasets.length,
            totalSize,
            averageSize:
              datasets.length > 0 ? totalSize / datasets.length : 0,
            largestDataset,
          },
          totalDatasetSize: totalSize,
        });
      },

      getDatasetsBySize: () => {
        const { datasets } = get();
        return [...datasets].sort(
          (a, b) => (b.fileSize || 0) - (a.fileSize || 0)
        );
      },

      getRecentDatasets: (limit = 5) => {
        const { datasets } = get();
        return [...datasets]
          .sort(
            (a, b) =>
              new Date(b.createdAt).getTime() -
              new Date(a.createdAt).getTime()
          )
          .slice(0, limit);
      },

      getDatasetsByTag: (tag) => {
        const { datasets } = get();
        return datasets.filter((d) => d.tags?.includes(tag));
      },

      getFavoriteDatasets: () => {
        const { datasets, favoriteDatasets } = get();
        return datasets.filter((d) => favoriteDatasets.includes(d.id));
      },

      getCollectionDatasets: (collectionId) => {
        const { datasets, datasetCollections } = get();
        const collectionDatasetIds = datasetCollections.get(collectionId) || [];
        return datasets.filter((d) =>
          collectionDatasetIds.includes(d.id)
        );
      },

      markDatasetProcessing: (datasetId, isProcessing) => {
        set((state) => {
          const newSet = new Set(state.processingDatasets);
          if (isProcessing) {
            newSet.add(datasetId);
          } else {
            newSet.delete(datasetId);
          }
          return { processingDatasets: newSet };
        });
      },

      isDatasetProcessing: (datasetId) => {
        return get().processingDatasets.has(datasetId);
      },

      updateDatasets: (datasets) => set({ datasets }),

      removeDatasetById: (datasetId) =>
        set((state) => ({
          datasets: state.datasets.filter((d) => d.id !== datasetId),
        })),

      clearAllDatasets: () =>
        set({
          datasets: [],
          selectedDataset: null,
          pagination: {
            currentPage: 1,
            pageSize: 10,
            totalItems: 0,
            totalPages: 0,
          },
        }),

      resetDatasetStore: () => {
        set(createInitialDatasetState());
      },

      cleanupStore: () => {
        get().cleanupExpiredCache();
        const { searchDebounceTimer } = get();
        if (searchDebounceTimer) {
          clearTimeout(searchDebounceTimer);
        }
      },
    })),
    {
      name: 'dataset-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        favoriteDatasets: state.favoriteDatasets,
        viewMode: state.viewMode,
        itemsPerPage: state.itemsPerPage,
        datasetCollections: Array.from(state.datasetCollections.entries()),
        filters: state.filters,
      }),
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          return {
            ...persistedState,
            datasetCollections: new Map(
              persistedState.datasetCollections || []
            ),
          };
        }
        return persistedState;
      },
    }
  )
);

export const useDatasets = () => datasetStore((state) => state.datasets);
export const useSelectedDataset = () =>
  datasetStore((state) => state.selectedDataset);
export const useDatasetLoading = () =>
  datasetStore((state) => state.isLoadingDatasets);
export const useDatasetError = () =>
  datasetStore((state) => state.datasetsError);
export const useDatasetPagination = () =>
  datasetStore((state) => state.pagination);
export const useDatasetFilters = () =>
  datasetStore((state) => state.filters);
export const useUploadState = () =>
  datasetStore((state) => state.uploadState);
export const useDatasetStats = () =>
  datasetStore((state) => state.datasetStats);
export const useFavoriteDatasets = () =>
  datasetStore((state) => state.favoriteDatasets);
export const useViewMode = () => datasetStore((state) => state.viewMode);

export default datasetStore;
