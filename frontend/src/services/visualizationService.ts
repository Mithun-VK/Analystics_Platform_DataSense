// src/services/visualizationService.ts
import { apiPost, apiGet, apiPut, apiDelete } from './api';
import type {
  ChartConfig,
  ChartData,
  ChartType,
  ChartStyle,
  ExportFormat,
} from '@/types/visualization.types';

/**
 * Visualization Service - Handles all chart/visualization-related API calls
 */

const VISUALIZATION_ENDPOINTS = {
  GENERATE_CHART: '/visualizations/generate',
  GET_CHART: '/visualizations/:id',
  LIST_CHARTS: '/visualizations',
  UPDATE_CHART: '/visualizations/:id',
  DELETE_CHART: '/visualizations/:id',
  DUPLICATE_CHART: '/visualizations/:id/duplicate',
  EXPORT_CHART: '/visualizations/:id/export/:format',
  SHARE_CHART: '/visualizations/:id/share',
  GET_PREVIEW: '/visualizations/:id/preview',
  VALIDATE_CONFIG: '/visualizations/validate',
  CHART_TEMPLATES: '/visualizations/templates',
  GET_TEMPLATE: '/visualizations/templates/:templateId',
  SAVE_CHART: '/visualizations/save',
  GET_CHART_HISTORY: '/visualizations/:id/history',
  RESTORE_VERSION: '/visualizations/:id/restore/:version',
  GET_SUPPORTED_TYPES: '/visualizations/chart-types',
  BULK_GENERATE: '/visualizations/bulk-generate',
};

interface ChartGenerationRequest {
  datasetId?: string;
  data?: any[];
  chartType: ChartType;
  xAxis: string;
  yAxis: string | string[];
  title?: string;
  description?: string;
  filters?: Record<string, any>;
  aggregation?: {
    type: 'sum' | 'avg' | 'count' | 'min' | 'max';
    field?: string;
    groupBy?: string;
  };
  style?: ChartStyle;
  options?: {
    showLegend?: boolean;
    showGrid?: boolean;
    showTooltip?: boolean;
    responsive?: boolean;
    height?: number;
    width?: number;
  };
}

interface GeneratedChart {
  id: string;
  chartType: ChartType;
  data: ChartData;
  config: ChartConfig;
  datasetId?: string;
  createdAt: string;
  updatedAt: string;
  thumbnail?: string;
}

interface ChartTemplate {
  id: string;
  name: string;
  description: string;
  chartType: ChartType;
  defaultConfig: ChartConfig;
  category: string;
  thumbnail?: string;
}

interface ChartShare {
  id: string;
  chartId: string;
  type: 'link' | 'public' | 'users';
  permission: 'view' | 'edit' | 'manage';
  expiresAt?: string;
  sharedWith?: string[];
}

interface ChartVersion {
  versionId: string;
  timestamp: string;
  changes: string;
  createdBy: string;
}

/**
 * Generate chart from dataset
 */
export const generateChart = async (
  request: ChartGenerationRequest & { signal?: AbortSignal }
): Promise<GeneratedChart> => {
  try {
    const payload = {
      datasetId: request.datasetId,
      data: request.data,
      chartType: request.chartType,
      xAxis: request.xAxis,
      yAxis: request.yAxis,
      title: request.title,
      description: request.description,
      filters: request.filters,
      aggregation: request.aggregation,
      style: request.style,
      options: request.options,
    };

    const response = await apiPost<GeneratedChart>(
      VISUALIZATION_ENDPOINTS.GENERATE_CHART,
      payload,
      { signal: request.signal }
    );

    console.debug('[VisualizationService] Chart generated', {
      id: response.id,
      type: response.chartType,
    });

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to generate chart', error);
    throw error;
  }
};

/**
 * Get chart by ID
 */
export const getChart = async (chartId: string): Promise<GeneratedChart> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.GET_CHART.replace(':id', chartId);
    const response = await apiGet<GeneratedChart>(url);

    console.debug('[VisualizationService] Chart fetched', chartId);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart', error);
    throw error;
  }
};

/**
 * List all charts with pagination and filtering
 */
export const listCharts = async (
  page: number = 1,
  limit: number = 20,
  filters?: {
    datasetId?: string;
    chartType?: ChartType;
    search?: string;
    sortBy?: 'createdAt' | 'updatedAt' | 'name';
    sortOrder?: 'asc' | 'desc';
  }
): Promise<{
  charts: GeneratedChart[];
  pagination: { page: number; limit: number; total: number };
}> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(filters?.datasetId && { datasetId: filters.datasetId }),
      ...(filters?.chartType && { chartType: filters.chartType }),
      ...(filters?.search && { search: filters.search }),
      ...(filters?.sortBy && { sortBy: filters.sortBy }),
      ...(filters?.sortOrder && { sortOrder: filters.sortOrder }),
    });

    const url = `${VISUALIZATION_ENDPOINTS.LIST_CHARTS}?${queryParams.toString()}`;
    const response = await apiGet<{
      charts: GeneratedChart[];
      pagination: { page: number; limit: number; total: number };
    }>(url);

    console.debug('[VisualizationService] Charts listed', response.charts.length);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to list charts', error);
    throw error;
  }
};

/**
 * Update chart configuration
 */
export const updateChart = async (
  chartId: string,
  updates: Partial<{
    title: string;
    description: string;
    chartType: ChartType;
    config: Partial<ChartConfig>;
    style: Partial<ChartStyle>;
  }>
): Promise<GeneratedChart> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.UPDATE_CHART.replace(':id', chartId);
    const response = await apiPut<GeneratedChart>(url, updates);

    console.debug('[VisualizationService] Chart updated', chartId);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to update chart', error);
    throw error;
  }
};

/**
 * Delete chart
 */
export const deleteChart = async (chartId: string): Promise<void> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.DELETE_CHART.replace(':id', chartId);
    await apiDelete(url);

    console.debug('[VisualizationService] Chart deleted', chartId);
  } catch (error) {
    console.error('[VisualizationService] Failed to delete chart', error);
    throw error;
  }
};

/**
 * Duplicate chart
 */
export const duplicateChart = async (
  chartId: string,
  newName?: string
): Promise<GeneratedChart> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.DUPLICATE_CHART.replace(':id', chartId);
    const response = await apiPost<GeneratedChart>(url, {
      name: newName,
    });

    console.debug('[VisualizationService] Chart duplicated', chartId);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to duplicate chart', error);
    throw error;
  }
};

/**
 * Export chart in multiple formats
 */
export const exportChart = async (
  chartId: string,
  format: ExportFormat = 'png',
  options?: {
    width?: number;
    height?: number;
    quality?: number;
  }
): Promise<Blob> => {
  try {
    const queryParams = new URLSearchParams({
      ...(options?.width && { width: options.width.toString() }),
      ...(options?.height && { height: options.height.toString() }),
      ...(options?.quality && { quality: options.quality.toString() }),
    });

    const url = `${VISUALIZATION_ENDPOINTS.EXPORT_CHART
      .replace(':id', chartId)
      .replace(':format', format)}?${queryParams.toString()}`;

    // Use XMLHttpRequest for blob response
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.responseType = 'blob';
      xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('auth_access_token')}`);

      xhr.onload = () => {
        if (xhr.status === 200) {
          console.debug('[VisualizationService] Chart exported', {
            chartId,
            format,
          });
          resolve(xhr.response);
        } else {
          reject(new Error(`Export failed with status ${xhr.status}`));
        }
      };

      xhr.onerror = () => {
        reject(new Error('Export request failed'));
      };

      xhr.send();
    });
  } catch (error) {
    console.error('[VisualizationService] Failed to export chart', error);
    throw error;
  }
};

/**
 * Download chart as file
 */
export const downloadChart = async (
  chartId: string,
  format: ExportFormat = 'png'
): Promise<void> => {
  try {
    const blob = await exportChart(chartId, format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `chart_${chartId}.${format}`;
    link.click();
    URL.revokeObjectURL(url);

    console.debug('[VisualizationService] Chart downloaded', {
      chartId,
      format,
    });
  } catch (error) {
    console.error('[VisualizationService] Failed to download chart', error);
    throw error;
  }
};

/**
 * Get chart preview (thumbnail)
 */
export const getChartPreview = async (chartId: string): Promise<string> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.GET_PREVIEW.replace(':id', chartId);
    const response = await apiGet<{ thumbnail: string }>(url);

    console.debug('[VisualizationService] Chart preview fetched', chartId);

    return response.thumbnail;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart preview', error);
    throw error;
  }
};

/**
 * Validate chart configuration
 */
export const validateChartConfig = async (
  config: ChartGenerationRequest
): Promise<{ valid: boolean; errors?: string[] }> => {
  try {
    const response = await apiPost<{
      valid: boolean;
      errors?: string[];
    }>(VISUALIZATION_ENDPOINTS.VALIDATE_CONFIG, config);

    console.debug('[VisualizationService] Chart config validated', {
      valid: response.valid,
    });

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to validate chart config', error);
    throw error;
  }
};

/**
 * Get chart templates
 */
export const getChartTemplates = async (
  category?: string
): Promise<ChartTemplate[]> => {
  try {
    const queryParams = new URLSearchParams({
      ...(category && { category }),
    });

    const url = `${VISUALIZATION_ENDPOINTS.CHART_TEMPLATES}?${queryParams.toString()}`;
    const response = await apiGet<ChartTemplate[]>(url);

    console.debug('[VisualizationService] Chart templates fetched', response.length);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart templates', error);
    throw error;
  }
};

/**
 * Get specific chart template
 */
export const getChartTemplate = async (templateId: string): Promise<ChartTemplate> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.GET_TEMPLATE.replace(':templateId', templateId);
    const response = await apiGet<ChartTemplate>(url);

    console.debug('[VisualizationService] Chart template fetched', templateId);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart template', error);
    throw error;
  }
};

/**
 * Save chart
 */
export const saveChart = async (
  chartData: GeneratedChart,
  metadata?: { tags?: string[]; folder?: string }
): Promise<GeneratedChart> => {
  try {
    const response = await apiPost<GeneratedChart>(
      VISUALIZATION_ENDPOINTS.SAVE_CHART,
      {
        ...chartData,
        ...metadata,
      }
    );

    console.debug('[VisualizationService] Chart saved', response.id);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to save chart', error);
    throw error;
  }
};

/**
 * Get chart version history
 */
export const getChartHistory = async (chartId: string): Promise<ChartVersion[]> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.GET_CHART_HISTORY.replace(':id', chartId);
    const response = await apiGet<ChartVersion[]>(url);

    console.debug('[VisualizationService] Chart history fetched', response.length);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart history', error);
    throw error;
  }
};

/**
 * Restore chart to previous version
 */
export const restoreChartVersion = async (
  chartId: string,
  versionId: string
): Promise<GeneratedChart> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.RESTORE_VERSION
      .replace(':id', chartId)
      .replace(':version', versionId);
    const response = await apiPost<GeneratedChart>(url, {});

    console.debug('[VisualizationService] Chart version restored', {
      chartId,
      versionId,
    });

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to restore chart version', error);
    throw error;
  }
};

/**
 * Share chart with users
 */
export const shareChart = async (
  chartId: string,
  shareConfig: {
    type: 'link' | 'public' | 'users';
    permission: 'view' | 'edit' | 'manage';
    emails?: string[];
    expiresIn?: number;
  }
): Promise<ChartShare> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.SHARE_CHART.replace(':id', chartId);
    const response = await apiPost<ChartShare>(url, shareConfig);

    console.debug('[VisualizationService] Chart shared', {
      chartId,
      type: shareConfig.type,
    });

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to share chart', error);
    throw error;
  }
};

/**
 * Get supported chart types
 */
export const getSupportedChartTypes = async (): Promise<
  Array<{
    type: ChartType;
    label: string;
    description: string;
    icon?: string;
    supportedDataTypes: string[];
  }>
> => {
  try {
    const response = await apiGet<
      Array<{
        type: ChartType;
        label: string;
        description: string;
        icon?: string;
        supportedDataTypes: string[];
      }>
    >(VISUALIZATION_ENDPOINTS.GET_SUPPORTED_TYPES);

    console.debug('[VisualizationService] Supported chart types fetched', response.length);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart types', error);
    throw error;
  }
};

/**
 * Generate multiple charts in bulk
 */
export const generateChartsInBulk = async (
  requests: ChartGenerationRequest[]
): Promise<GeneratedChart[]> => {
  try {
    const response = await apiPost<GeneratedChart[]>(
      VISUALIZATION_ENDPOINTS.BULK_GENERATE,
      { charts: requests }
    );

    console.debug('[VisualizationService] Charts bulk generated', response.length);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to bulk generate charts', error);
    throw error;
  }
};

/**
 * Generate chart from template
 */
export const generateChartFromTemplate = async (
  templateId: string,
  datasetId: string,
  overrides?: Partial<ChartGenerationRequest>
): Promise<GeneratedChart> => {
  try {
    const template = await getChartTemplate(templateId);

    const request: ChartGenerationRequest = {
      datasetId,
      chartType: template.chartType,
      xAxis: '',
      yAxis: '',
      ...overrides,
    };

    return generateChart(request);
  } catch (error) {
    console.error('[VisualizationService] Failed to generate chart from template', error);
    throw error;
  }
};

/**
 * Create dashboard with multiple charts
 */
export const createChartDashboard = async (
  name: string,
  chartIds: string[],
  layout?: {
    columns?: number;
    rows?: number;
  }
): Promise<{
  id: string;
  name: string;
  charts: string[];
  layout: any;
}> => {
  try {
    const response = await apiPost<{
      id: string;
      name: string;
      charts: string[];
      layout: any;
    }>('/visualizations/dashboards', {
      name,
      chartIds,
      layout,
    });

    console.debug('[VisualizationService] Dashboard created', response.id);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to create dashboard', error);
    throw error;
  }
};

/**
 * Get chart data in different formats
 */
export const getChartData = async (
  chartId: string,
  format: 'json' | 'csv' = 'json'
): Promise<any> => {
  try {
    const queryParams = new URLSearchParams({ format });
    const url = `${VISUALIZATION_ENDPOINTS.GET_CHART.replace(':id', chartId)}/data?${queryParams.toString()}`;
    const response = await apiGet<any>(url);

    console.debug('[VisualizationService] Chart data fetched', { chartId, format });

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to fetch chart data', error);
    throw error;
  }
};

/**
 * Update chart data in real-time
 */
export const updateChartDataRealtime = async (
  chartId: string,
  newData: any[]
): Promise<GeneratedChart> => {
  try {
    const url = VISUALIZATION_ENDPOINTS.UPDATE_CHART.replace(':id', chartId);
    const response = await apiPut<GeneratedChart>(url, {
      data: newData,
    });

    console.debug('[VisualizationService] Chart data updated', chartId);

    return response;
  } catch (error) {
    console.error('[VisualizationService] Failed to update chart data', error);
    throw error;
  }
};

export default {
  generateChart,
  getChart,
  listCharts,
  updateChart,
  deleteChart,
  duplicateChart,
  exportChart,
  downloadChart,
  getChartPreview,
  validateChartConfig,
  getChartTemplates,
  getChartTemplate,
  saveChart,
  getChartHistory,
  restoreChartVersion,
  shareChart,
  getSupportedChartTypes,
  generateChartsInBulk,
  generateChartFromTemplate,
  createChartDashboard,
  getChartData,
  updateChartDataRealtime,
};
