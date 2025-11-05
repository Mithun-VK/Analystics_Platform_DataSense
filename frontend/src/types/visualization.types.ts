// src/types/visualization.types.ts
/**
 * Visualization and Chart Types
 * Comprehensive type definitions for all chart and visualization-related operations
 */

// ============================================================================
// Core Chart Types
// ============================================================================

/**
 * Supported chart types
 */
export type ChartType =
  | 'bar'
  | 'line'
  | 'pie'
  | 'donut'
  | 'scatter'
  | 'bubble'
  | 'area'
  | 'areaStacked'
  | 'histogram'
  | 'boxplot'
  | 'violin'
  | 'heatmap'
  | 'treemap'
  | 'sunburst'
  | 'sankey'
  | 'gauge'
  | 'indicator'
  | 'waterfall'
  | 'funnel'
  | 'timeline'
  | 'network'
  | 'map'
  | 'candlestick'
  | 'qqplot'
  | 'hexbin'
  | 'contour'
  | 'parallel'
  | 'radial'
  | 'custom';

/**
 * Export format options
 */
export type ExportFormat = 'png' | 'jpg' | 'pdf' | 'svg' | 'json' | 'csv' | 'html';

/**
 * Chart orientation
 */
export type ChartOrientation = 'vertical' | 'horizontal';

/**
 * Aggregation method
 */
export type AggregationMethod = 'sum' | 'avg' | 'count' | 'min' | 'max' | 'median' | 'stdDev' | 'variance' | 'custom';

/**
 * Chart data point
 */
export interface DataPoint {
  x: string | number | Date;
  y: number | number[];
  label?: string;
  color?: string;
  size?: number;
  meta?: Record<string, any>;
}

/**
 * Chart data series
 */
export interface DataSeries {
  name: string;
  data: DataPoint[];
  type?: string;
  color?: string;
  opacity?: number;
  lineStyle?: 'solid' | 'dashed' | 'dotted';
  markerStyle?: 'circle' | 'square' | 'triangle' | 'diamond' | 'cross';
  visible?: boolean;
}

/**
 * Chart data
 */
export interface ChartData {
  series: DataSeries[];
  categories?: (string | number | Date)[];
  xAxisData?: (string | number | Date)[];
  yAxisData?: (string | number | Date)[];
  metadata?: {
    source?: string;
    generatedAt?: string;
    lastUpdated?: string;
    rowsIncluded?: number;
  };
}

// ============================================================================
// Chart Configuration Types
// ============================================================================

/**
 * Chart axis configuration
 */
export interface ChartAxis {
  name?: string;
  type?: 'category' | 'value' | 'time' | 'log';
  position?: 'left' | 'right' | 'top' | 'bottom';
  show?: boolean;
  axisLabel?: {
    show?: boolean;
    rotate?: number;
    fontSize?: number;
    color?: string;
    formatter?: string;
    interval?: number;
  };
  axisLine?: {
    show?: boolean;
    color?: string;
    lineStyle?: 'solid' | 'dashed' | 'dotted';
  };
  axisTick?: {
    show?: boolean;
    length?: number;
  };
  scale?: boolean;
  min?: number | 'dataMin';
  max?: number | 'dataMax';
  splitNumber?: number;
  data?: (string | number)[];
}

/**
 * Chart legend configuration
 */
export interface ChartLegend {
  show?: boolean;
  position?: 'top' | 'bottom' | 'left' | 'right' | 'inside-top-left' | 'inside-top-right' | 'inside-bottom-left' | 'inside-bottom-right';
  orient?: 'horizontal' | 'vertical';
  align?: 'auto' | 'left' | 'right' | 'center';
  verticalAlign?: 'auto' | 'top' | 'middle' | 'bottom';
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  borderRadius?: number;
  padding?: number | number[];
  itemGap?: number;
  itemHeight?: number;
  textStyle?: TextStyle;
  selected?: Record<string, boolean>;
  formatter?: string;
}

/**
 * Chart tooltip configuration
 */
export interface ChartTooltip {
  show?: boolean;
  trigger?: 'item' | 'axis' | 'none';
  triggerOn?: 'mousemove' | 'click' | 'mousemove|click';
  axisPointer?: {
    type?: 'line' | 'shadow' | 'cross' | 'none';
    lineStyle?: LineStyle;
    shadowStyle?: ShadowStyle;
  };
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  padding?: number | number[];
  textStyle?: TextStyle;
  formatter?: string;
  contentStyle?: Record<string, any>;
  position?: 'top' | 'right' | 'bottom' | 'left' | 'follow';
  alwaysShowContent?: boolean;
}

/**
 * Line style configuration
 */
export interface LineStyle {
  color?: string;
  width?: number;
  type?: 'solid' | 'dashed' | 'dotted';
  opacity?: number;
  cap?: 'butt' | 'round' | 'square';
  join?: 'bevel' | 'miter' | 'round';
}

/**
 * Shadow style configuration
 */
export interface ShadowStyle {
  color?: string;
  opacity?: number;
}

/**
 * Text style configuration
 */
export interface TextStyle {
  color?: string;
  fontSize?: number;
  fontWeight?: 'normal' | 'bold' | 'bolder' | 'lighter';
  fontFamily?: string;
  fontStyle?: 'normal' | 'italic' | 'oblique';
  textAlign?: 'left' | 'center' | 'right';
  textShadow?: string;
  lineHeight?: number;
  textDecoration?: 'none' | 'underline' | 'overline' | 'line-through';
}

/**
 * Grid configuration
 */
export interface ChartGrid {
  show?: boolean;
  containLabel?: boolean;
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  shadowBlur?: number;
  shadowColor?: string;
  shadowOffsetX?: number;
  shadowOffsetY?: number;
  left?: number | string;
  right?: number | string;
  top?: number | string;
  bottom?: number | string;
  width?: number | string;
  height?: number | string;
}

/**
 * Color scheme
 */
export interface ChartColorScheme {
  name: string;
  colors: string[];
  backgroundColor?: string;
  textColor?: string;
  borderColor?: string;
  gridColor?: string;
  axisLineColor?: string;
  tooltipBackground?: string;
}

/**
 * Animation configuration
 */
export interface AnimationConfig {
  enabled?: boolean;
  duration?: number;
  easing?: 'linear' | 'quadraticIn' | 'quadraticOut' | 'quadraticInOut' | 'cubicIn' | 'cubicOut' | 'cubicInOut';
  delay?: number;
  stagger?: boolean;
}

/**
 * Chart style configuration
 */
export interface ChartStyle {
  theme?: 'light' | 'dark' | 'custom';
  colorScheme?: ChartColorScheme | string;
  fontSize?: number;
  fontFamily?: string;
  containerBackground?: string;
  borderRadius?: number;
  boxShadow?: string;
  padding?: number | number[];
  margin?: number | number[];
}

/**
 * Main chart configuration
 */
export interface ChartConfig {
  // Basic configuration
  id?: string;
  type: ChartType;
  title?: string;
  subtitle?: string;
  description?: string;
  
  // Dimensions
  width?: number | string;
  height?: number | string;
  responsive?: boolean;
  maintainAspectRatio?: boolean;
  aspectRatio?: number;

  // Data configuration
  xAxis?: ChartAxis;
  yAxis?: ChartAxis | ChartAxis[];
  xField?: string;
  yField?: string | string[];
  seriesField?: string;
  colorField?: string;
  sizeField?: string;
  shapeField?: string;

  // Series configuration
  series?: {
    type?: ChartType;
    name?: string;
    stack?: string;
    label?: LabelConfig;
    smooth?: boolean;
    connectNulls?: boolean;
    sampling?: 'average' | 'max' | 'min' | 'sum' | 'lttb';
    symbol?: 'circle' | 'square' | 'triangle' | 'diamond' | 'cross';
    symbolSize?: number | number[];
    lineStyle?: LineStyle;
    areaStyle?: AreaStyle;
    itemStyle?: ItemStyle;
    emphasis?: EmphasisStyle;
  }[];

  // Visual components
  legend?: ChartLegend;
  tooltip?: ChartTooltip;
  grid?: ChartGrid;
  style?: ChartStyle;

  // Advanced options
  animation?: AnimationConfig;
  dataZoom?: DataZoomConfig[];
  visualMap?: VisualMapConfig;
  markLine?: MarkLineConfig;
  markPoint?: MarkPointConfig;
  markArea?: MarkAreaConfig;

  // Interaction
  interactive?: boolean;
  selectable?: boolean;
  clickable?: boolean;
  hoverable?: boolean;
  draggable?: boolean;
  zoomable?: boolean;
  resizable?: boolean;

  // Export & sharing
  exportable?: boolean;
  printable?: boolean;
  shareable?: boolean;

  // Custom options
  customOptions?: Record<string, any>;
  toolbox?: ToolboxConfig;
}

/**
 * Label configuration
 */
export interface LabelConfig {
  show?: boolean;
  position?: 'top' | 'bottom' | 'left' | 'right' | 'inside' | 'outside' | 'center';
  distance?: number;
  rotate?: number;
  fontSize?: number;
  fontWeight?: string;
  color?: string;
  formatter?: string;
  backgroundColor?: string;
  borderColor?: string;
  borderWidth?: number;
  borderRadius?: number;
  padding?: number | number[];
}

/**
 * Area style configuration
 */
export interface AreaStyle {
  color?: string;
  opacity?: number;
}

/**
 * Item style configuration
 */
export interface ItemStyle {
  color?: string;
  borderColor?: string;
  borderWidth?: number;
  opacity?: number;
  shadowColor?: string;
  shadowBlur?: number;
  shadowOffsetX?: number;
  shadowOffsetY?: number;
  shadowOpacity?: number;
}

/**
 * Emphasis style configuration
 */
export interface EmphasisStyle {
  itemStyle?: ItemStyle;
  lineStyle?: LineStyle;
  labelLine?: LabelLineConfig;
  label?: LabelConfig;
}

/**
 * Label line configuration
 */
export interface LabelLineConfig {
  show?: boolean;
  length?: number;
  length2?: number;
  smooth?: boolean;
  lineStyle?: LineStyle;
}

/**
 * Data zoom configuration
 */
export interface DataZoomConfig {
  type?: 'slider' | 'inside' | 'select';
  show?: boolean;
  start?: number;
  end?: number;
  startValue?: number;
  endValue?: number;
  xAxisIndex?: number[];
  yAxisIndex?: number[];
  minSpan?: number;
  maxSpan?: number;
  fillerColor?: string;
  handleSize?: number;
  textStyle?: TextStyle;
}

/**
 * Visual map configuration
 */
export interface VisualMapConfig {
  type?: 'continuous' | 'piecewise';
  min?: number;
  max?: number;
  inRange?: {
    color?: string[];
    colorAlpha?: number[];
    opacity?: number[];
    symbolSize?: number[];
  };
  outOfRange?: {
    color?: string[];
  };
  calculable?: boolean;
  realtime?: boolean;
  splitNumber?: number;
  pieces?: Array<{
    min?: number;
    max?: number;
    label?: string;
    color?: string;
  }>;
  textStyle?: TextStyle;
}

/**
 * Mark line configuration
 */
export interface MarkLineConfig {
  show?: boolean;
  data: Array<{
    name?: string;
    type?: 'average' | 'median' | 'max' | 'min' | 'custom';
    yAxis?: number;
    lineStyle?: LineStyle;
    label?: LabelConfig;
    symbol?: string;
    symbolSize?: number;
  }>;
  lineStyle?: LineStyle;
  label?: LabelConfig;
  animation?: boolean;
}

/**
 * Mark point configuration
 */
export interface MarkPointConfig {
  show?: boolean;
  data: Array<{
    name?: string;
    type?: 'max' | 'min' | 'average' | 'custom';
    coord?: (string | number)[];
    value?: number;
    itemStyle?: ItemStyle;
    label?: LabelConfig;
    symbol?: string;
    symbolSize?: number;
  }>;
  itemStyle?: ItemStyle;
  label?: LabelConfig;
  animation?: boolean;
}

/**
 * Mark area configuration
 */
export interface MarkAreaConfig {
  show?: boolean;
  data: Array<{
    name?: string;
    itemStyle?: ItemStyle;
    label?: LabelConfig;
    data: any[];
  }>;
  itemStyle?: ItemStyle;
  label?: LabelConfig;
}

/**
 * Toolbox configuration
 */
export interface ToolboxConfig {
  show?: boolean;
  orient?: 'horizontal' | 'vertical';
  itemSize?: number;
  itemGap?: number;
  showTitle?: boolean;
  iconStyle?: Record<string, any>;
  feature?: {
    saveAsImage?: { show?: boolean; pixelRatio?: number };
    dataView?: { show?: boolean; readOnly?: boolean };
    restore?: { show?: boolean };
    dataZoom?: { show?: boolean };
    magicType?: { show?: boolean; types?: ChartType[] };
  };
}

// ============================================================================
// Chart Generation Request/Response Types
// ============================================================================

/**
 * Chart generation request
 */
export interface ChartGenerationRequest {
  datasetId?: string;
  data?: any[];
  chartType: ChartType;
  xAxis: string | string[];
  yAxis: string | string[];
  title?: string;
  description?: string;
  filters?: Record<string, any>;
  aggregation?: {
    type: AggregationMethod;
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
    interactive?: boolean;
  };
  customConfig?: Partial<ChartConfig>;
}

/**
 * Generated chart
 */
export interface GeneratedChart {
  id: string;
  chartType: ChartType;
  data: ChartData;
  config: ChartConfig;
  datasetId?: string;
  createdAt: string;
  updatedAt: string;
  createdBy: string;
  thumbnail?: string;
  previewUrl?: string;
  metadata?: Record<string, any>;
}

/**
 * Chart template
 */
export interface ChartTemplate {
  id: string;
  name: string;
  description?: string;
  category: string;
  chartType: ChartType;
  defaultConfig: ChartConfig;
  defaultStyle?: ChartStyle;
  previewImage?: string;
  tags?: string[];
  isPublic: boolean;
  createdAt: string;
}

/**
 * Chart share configuration
 */
export interface ChartShare {
  id: string;
  chartId: string;
  type: 'link' | 'public' | 'users';
  permission: 'view' | 'edit' | 'manage';
  expiresAt?: string;
  sharedWith?: string[];
  shareUrl?: string;
  shareable?: boolean;
}

/**
 * Chart version
 */
export interface ChartVersion {
  versionId: string;
  timestamp: string;
  changes: string;
  createdBy: string;
  config: ChartConfig;
  data?: ChartData;
}

// ============================================================================
// Chart Export & Download Types
// ============================================================================

/**
 * Export options
 */
export interface ExportOptions {
  format: ExportFormat;
  width?: number;
  height?: number;
  quality?: number;
  backgroundColor?: string;
  includeTitle?: boolean;
  includeData?: boolean;
  scale?: number;
  compression?: 'none' | 'lz' | 'deflate';
}

/**
 * Export result
 */
export interface ExportResult {
  exportId: string;
  chartId: string;
  format: ExportFormat;
  fileSize: number;
  downloadUrl: string;
  expiresAt: string;
  createdAt: string;
}

// ============================================================================
// Dashboard Types
// ============================================================================

/**
 * Dashboard layout item
 */
export interface DashboardLayoutItem {
  id: string;
  chartId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  minWidth?: number;
  minHeight?: number;
  maxWidth?: number;
  maxHeight?: number;
  resizable?: boolean;
  draggable?: boolean;
  static?: boolean;
}

/**
 * Dashboard
 */
export interface Dashboard {
  id: string;
  name: string;
  description?: string;
  userId: string;
  charts: GeneratedChart[];
  layout: DashboardLayoutItem[];
  theme?: 'light' | 'dark';
  backgroundColor?: string;
  isPublic: boolean;
  refreshInterval?: number;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, any>;
}

/**
 * Dashboard creation request
 */
export interface DashboardCreateRequest {
  name: string;
  description?: string;
  chartIds: string[];
  layout?: DashboardLayoutItem[];
  theme?: 'light' | 'dark';
  isPublic?: boolean;
}

// ============================================================================
// Chart Comparison Types
// ============================================================================

/**
 * Chart comparison
 */
export interface ChartComparison {
  chart1Id: string;
  chart2Id: string;
  similarities: string[];
  differences: string[];
  recommendations: string[];
  bestFor: string;
}

// ============================================================================
// Enums
// ============================================================================

/**
 * Chart type enumeration
 */
export enum ChartTypeEnum {
  Bar = 'bar',
  Line = 'line',
  Pie = 'pie',
  Donut = 'donut',
  Scatter = 'scatter',
  Bubble = 'bubble',
  Area = 'area',
  AreaStacked = 'areaStacked',
  Histogram = 'histogram',
  BoxPlot = 'boxplot',
  Violin = 'violin',
  Heatmap = 'heatmap',
  Treemap = 'treemap',
  Sunburst = 'sunburst',
  Sankey = 'sankey',
  Gauge = 'gauge',
  Indicator = 'indicator',
  Waterfall = 'waterfall',
  Funnel = 'funnel',
  Timeline = 'timeline',
  Network = 'network',
  Map = 'map',
  Candlestick = 'candlestick',
  QQPlot = 'qqplot',
  Hexbin = 'hexbin',
  Contour = 'contour',
  Parallel = 'parallel',
  Radial = 'radial',
  Custom = 'custom',
}

/**
 * Export format enumeration
 */
export enum ExportFormatEnum {
  PNG = 'png',
  JPG = 'jpg',
  PDF = 'pdf',
  SVG = 'svg',
  JSON = 'json',
  CSV = 'csv',
  HTML = 'html',
}

/**
 * Chart orientation enumeration
 */
export enum ChartOrientationEnum {
  Vertical = 'vertical',
  Horizontal = 'horizontal',
}

/**
 * Aggregation method enumeration
 */
export enum AggregationMethodEnum {
  Sum = 'sum',
  Avg = 'avg',
  Count = 'count',
  Min = 'min',
  Max = 'max',
  Median = 'median',
  StdDev = 'stdDev',
  Variance = 'variance',
  Custom = 'custom',
}

/**
 * Data zoom type enumeration
 */
export enum DataZoomTypeEnum {
  Slider = 'slider',
  Inside = 'inside',
  Select = 'select',
}

/**
 * Visual map type enumeration
 */
export enum VisualMapTypeEnum {
  Continuous = 'continuous',
  Piecewise = 'piecewise',
}

/**
 * Mark type enumeration
 */
export enum MarkTypeEnum {
  Average = 'average',
  Median = 'median',
  Max = 'max',
  Min = 'min',
  Custom = 'custom',
}

/**
 * Position enumeration
 */
export enum PositionEnum {
  Top = 'top',
  Bottom = 'bottom',
  Left = 'left',
  Right = 'right',
  Inside = 'inside',
  Outside = 'outside',
  Center = 'center',
}

// ============================================================================
// Form Types
// ============================================================================

/**
 * Chart creation form values
 */
export interface ChartFormValues {
  title: string;
  description?: string;
  chartType: ChartType;
  xAxis: string | string[];
  yAxis: string | string[];
  filters?: Record<string, any>;
  aggregation?: {
    type: AggregationMethod;
    field?: string;
  };
  style?: ChartStyle;
  interactiveOptions?: {
    showLegend: boolean;
    showGrid: boolean;
    showTooltip: boolean;
    interactive: boolean;
  };
}

/**
 * Chart export form values
 */
export interface ChartExportFormValues {
  format: ExportFormat;
  width: number;
  height: number;
  quality: number;
  includeTitle: boolean;
  includeData: boolean;
}

// ============================================================================
// Type Guards & Predicates
// ============================================================================

/**
 * Type guard for ChartConfig
 */
export function isChartConfig(obj: any): obj is ChartConfig {
  return obj && typeof obj === 'object' && typeof obj.type === 'string';
}

/**
 * Type guard for GeneratedChart
 */
export function isGeneratedChart(obj: any): obj is GeneratedChart {
  return (
    obj &&
    typeof obj === 'object' &&
    typeof obj.id === 'string' &&
    typeof obj.chartType === 'string'
  );
}

/**
 * Check if chart type supports data zoom
 */
export function supportsDataZoom(chartType: ChartType): boolean {
  return ['line', 'bar', 'scatter', 'area', 'candlestick', 'heatmap'].includes(chartType);
}

/**
 * Check if chart type supports legend
 */
export function supportsLegend(chartType: ChartType): boolean {
  return ![
    'gauge',
    'indicator',
    'qqplot',
  ].includes(chartType);
}

/**
 * Check if chart type supports tooltip
 */
export function supportsTooltip(chartType: ChartType): boolean {
  return chartType !== 'custom';
}

/**
 * Check if chart type is numeric-based
 */
export function isNumericChartType(chartType: ChartType): boolean {
  return [
    'bar',
    'line',
    'scatter',
    'bubble',
    'area',
    'histogram',
    'boxplot',
    'candlestick',
  ].includes(chartType);
}

/**
 * Check if chart type is categorical-based
 */
export function isCategoricalChartType(chartType: ChartType): boolean {
  return ['pie', 'donut', 'funnel', 'gauge', 'sunburst'].includes(chartType);
}

/**
 * Get default chart height
 */
export function getDefaultChartHeight(chartType: ChartType): number {
  const heightMap: Record<ChartType, number> = {
    bar: 400,
    line: 350,
    pie: 400,
    donut: 400,
    scatter: 400,
    bubble: 450,
    area: 350,
    areaStacked: 350,
    histogram: 400,
    boxplot: 400,
    violin: 400,
    heatmap: 500,
    treemap: 400,
    sunburst: 400,
    sankey: 500,
    gauge: 350,
    indicator: 200,
    waterfall: 400,
    funnel: 400,
    timeline: 400,
    network: 500,
    map: 500,
    candlestick: 400,
    qqplot: 400,
    hexbin: 400,
    contour: 400,
    parallel: 500,
    radial: 400,
    custom: 400,
  };
  return heightMap[chartType] || 400;
}

/**
 * Get supported data types for chart type
 */
export function getSupportedDataTypes(chartType: ChartType): string[] {
  const typeMap: Record<ChartType, string[]> = {
    bar: ['numeric', 'categorical'],
    line: ['numeric', 'date'],
    pie: ['categorical', 'numeric'],
    donut: ['categorical', 'numeric'],
    scatter: ['numeric'],
    bubble: ['numeric'],
    area: ['numeric', 'date'],
    areaStacked: ['numeric', 'date'],
    histogram: ['numeric'],
    boxplot: ['numeric'],
    violin: ['numeric', 'categorical'],
    heatmap: ['numeric'],
    treemap: ['categorical', 'numeric'],
    sunburst: ['categorical', 'numeric'],
    sankey: ['categorical'],
    gauge: ['numeric'],
    indicator: ['numeric'],
    waterfall: ['numeric'],
    funnel: ['categorical', 'numeric'],
    timeline: ['date'],
    network: ['categorical'],
    map: ['geographic'],
    candlestick: ['numeric', 'date'],
    qqplot: ['numeric'],
    hexbin: ['numeric'],
    contour: ['numeric'],
    parallel: ['numeric', 'categorical'],
    radial: ['numeric'],
    custom: ['any'],
  };
  return typeMap[chartType] || [];
}
