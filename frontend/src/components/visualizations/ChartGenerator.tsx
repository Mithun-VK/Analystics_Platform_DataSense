// src/components/visualizations/ChartGenerator.tsx

import { useState, useMemo, useCallback, useRef } from 'react';
import {
  BarChart3,
  LineChart as LineChartIcon,
  PieChart as PieChartIcon,
  Activity,
  TrendingUp,
  Settings,
  Download,
  Eye,
  Save,
  Palette,
  Layers,
  Grid3x3,
  Sliders,
  RefreshCw,
  Copy,
  Check,
  AlertCircle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export interface Column {
  name: string;
  type: 'numeric' | 'categorical' | 'datetime';
}

export interface ChartConfig {
  type: 'line' | 'bar' | 'area' | 'pie' | 'scatter';
  xAxis: string;
  yAxis: string | string[];
  title: string;
  subtitle?: string;
  colors: string[];
  showGrid: boolean;
  showLegend: boolean;
  showTooltip: boolean;
  smoothCurve: boolean;
  stacked: boolean;
  animationDuration: number;
  dataLabels?: boolean;
  dotSize?: number;
}

export interface ChartGeneratorProps {
  datasetId: string;
  columns: Column[];
  data: any[];
  onSave?: (config: ChartConfig) => Promise<void>;
  onExport?: (config: ChartConfig) => void;
}

const DEFAULT_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // yellow
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
];

const CHART_TYPES = [
  { value: 'bar' as const, label: 'Bar Chart', icon: BarChart3 },
  { value: 'line' as const, label: 'Line Chart', icon: LineChartIcon },
  { value: 'area' as const, label: 'Area Chart', icon: Activity },
  { value: 'pie' as const, label: 'Pie Chart', icon: PieChartIcon },
  { value: 'scatter' as const, label: 'Scatter Plot', icon: Grid3x3 },
];

/**
 * ChartGenerator - Production-grade interactive chart builder
 * Features: Multiple chart types, live preview, full customization, export
 * Supports: Line, Bar, Area, Pie, and Scatter charts with advanced styling
 *
 * @example
 * <ChartGenerator
 *   datasetId="ds-123"
 *   columns={columns}
 *   data={data}
 *   onSave={saveChart}
 *   onExport={exportChart}
 * />
 */
const ChartGenerator: React.FC<ChartGeneratorProps> = ({
  columns,
  data,
  onSave,
  onExport,
}) => {
  // ✅ State management
  const [config, setConfig] = useState<ChartConfig>({
    type: 'bar',
    xAxis: columns[0]?.name || '',
    yAxis:
      columns.find((c) => c.type === 'numeric')?.name || columns[1]?.name || '',
    title: 'My Chart',
    subtitle: '',
    colors: DEFAULT_COLORS,
    showGrid: true,
    showLegend: true,
    showTooltip: true,
    smoothCurve: false,
    stacked: false,
    animationDuration: 1000,
    dataLabels: false,
    dotSize: 4,
  });

  const [activeTab, setActiveTab] = useState<'data' | 'style' | 'advanced'>(
    'data'
  );
  const [isSaving, setIsSaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const chartRef = useRef<HTMLDivElement>(null);

  // ✅ Get numeric columns for Y-axis
  const numericColumns = useMemo(
    () => columns.filter((c) => c.type === 'numeric'),
    [columns]
  );

  // ✅ Get categorical columns for X-axis
  const categoricalColumns = useMemo(
    () =>
      columns.filter(
        (c) => c.type === 'categorical' || c.type === 'datetime'
      ),
    [columns]
  );

  // ✅ Update config
  const updateConfig = useCallback((updates: Partial<ChartConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  }, []);

  // ✅ Update color at index
  const updateColor = useCallback((index: number, color: string) => {
    setConfig((prev) => {
      const newColors = [...prev.colors];
      newColors[index] = color;
      return { ...prev, colors: newColors };
    });
  }, []);

  // ✅ Handle save
  const handleSave = useCallback(async () => {
    if (!onSave) return;

    setIsSaving(true);
    try {
      await onSave(config);
    } catch (error) {
      console.error('Failed to save chart:', error);
    } finally {
      setIsSaving(false);
    }
  }, [config, onSave]);

  // ✅ Export chart as image
  const handleExport = useCallback(() => {
    try {
      onExport?.(config);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to export chart:', error);
    }
  }, [config, onExport]);

  // ✅ Copy chart config to clipboard
  const copyConfig = useCallback(() => {
    navigator.clipboard.writeText(JSON.stringify(config, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [config]);

  // ✅ Render chart based on type
  const renderChart = useCallback(() => {
    const commonProps = {
      data: data.slice(0, 100),
      margin: { top: 20, right: 30, left: 20, bottom: 20 },
    };

    if (!config.xAxis || !config.yAxis) {
      return (
        <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg">
          <div className="text-center">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">Please select X and Y axes</p>
          </div>
        </div>
      );
    }

    switch (config.type) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart {...commonProps}>
              {config.showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e5e7eb"
                  vertical={false}
                />
              )}
              <XAxis
                dataKey={config.xAxis}
                stroke="#6b7280"
                style={{ fontSize: '12px' }}
              />
              <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} />
              {config.showTooltip && (
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
              )}
              {config.showLegend && <Legend />}
              {Array.isArray(config.yAxis) ? (
                config.yAxis.map((key, index) => (
                  <Line
                    key={key}
                    type={config.smoothCurve ? 'monotone' : 'linear'}
                    dataKey={key}
                    stroke={
                      config.colors[index % config.colors.length]
                    }
                    strokeWidth={2}
                    dot={{ r: config.dotSize }}
                    activeDot={{ r: (config.dotSize || 4) + 2 }}
                    animationDuration={config.animationDuration}
                    isAnimationActive={true}
                  />
                ))
              ) : (
                <Line
                  type={config.smoothCurve ? 'monotone' : 'linear'}
                  dataKey={config.yAxis}
                  stroke={config.colors[0]}
                  strokeWidth={2}
                  dot={{ r: config.dotSize }}
                  activeDot={{ r: (config.dotSize || 4) + 2 }}
                  animationDuration={config.animationDuration}
                  isAnimationActive={true}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        );

      case 'bar':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart {...commonProps}>
              {config.showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e5e7eb"
                  vertical={false}
                />
              )}
              <XAxis
                dataKey={config.xAxis}
                stroke="#6b7280"
                style={{ fontSize: '12px' }}
              />
              <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} />
              {config.showTooltip && (
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
              )}
              {config.showLegend && <Legend />}
              {Array.isArray(config.yAxis) ? (
                config.yAxis.map((key, index) => (
                  <Bar
                    key={key}
                    dataKey={key}
                    fill={
                      config.colors[index % config.colors.length]
                    }
                    stackId={config.stacked ? 'stack' : undefined}
                    animationDuration={config.animationDuration}
                    radius={[8, 8, 0, 0]}
                  />
                ))
              ) : (
                <Bar
                  dataKey={config.yAxis}
                  fill={config.colors[0]}
                  animationDuration={config.animationDuration}
                  radius={[8, 8, 0, 0]}
                />
              )}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart {...commonProps}>
              {config.showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e5e7eb"
                  vertical={false}
                />
              )}
              <XAxis
                dataKey={config.xAxis}
                stroke="#6b7280"
                style={{ fontSize: '12px' }}
              />
              <YAxis stroke="#6b7280" style={{ fontSize: '12px' }} />
              {config.showTooltip && (
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
              )}
              {config.showLegend && <Legend />}
              {Array.isArray(config.yAxis) ? (
                config.yAxis.map((key, index) => (
                  <Area
                    key={key}
                    type={config.smoothCurve ? 'monotone' : 'linear'}
                    dataKey={key}
                    stroke={
                      config.colors[index % config.colors.length]
                    }
                    fill={
                      config.colors[index % config.colors.length]
                    }
                    fillOpacity={0.6}
                    stackId={config.stacked ? 'stack' : undefined}
                    animationDuration={config.animationDuration}
                  />
                ))
              ) : (
                <Area
                  type={config.smoothCurve ? 'monotone' : 'linear'}
                  dataKey={config.yAxis}
                  stroke={config.colors[0]}
                  fill={config.colors[0]}
                  fillOpacity={0.6}
                  animationDuration={config.animationDuration}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data.slice(0, 10)}
                dataKey={
                  Array.isArray(config.yAxis) ? config.yAxis[0] : config.yAxis
                }
                nameKey={config.xAxis}
                cx="50%"
                cy="50%"
                outerRadius={120}
                label={config.dataLabels}
                animationDuration={config.animationDuration}
              >
                {data.slice(0, 10).map((_, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      config.colors[index % config.colors.length]
                    }
                  />
                ))}
              </Pie>
              {config.showTooltip && (
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
              )}
              {config.showLegend && <Legend />}
            </PieChart>
          </ResponsiveContainer>
        );

      case 'scatter':
        return (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart {...commonProps}>
              {config.showGrid && (
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="#e5e7eb"
                  vertical={false}
                />
              )}
              <XAxis
                dataKey={config.xAxis}
                stroke="#6b7280"
                style={{ fontSize: '12px' }}
              />
              <YAxis
                dataKey={
                  Array.isArray(config.yAxis) ? config.yAxis[0] : config.yAxis
                }
                stroke="#6b7280"
                style={{ fontSize: '12px' }}
              />
              {config.showTooltip && (
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
              )}
              {config.showLegend && <Legend />}
              <Scatter
                name={
                  Array.isArray(config.yAxis) ? config.yAxis[0] : config.yAxis
                }
                data={data.slice(0, 100)}
                fill={config.colors[0]}
                animationDuration={config.animationDuration}
              />
            </ScatterChart>
          </ResponsiveContainer>
        );

      default:
        return null;
    }
  }, [config, data]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full">
      {/* Configuration Panel */}
      <div className="lg:col-span-1 space-y-4">
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
              <Settings className="w-5 h-5 text-blue-600" />
              <span>Chart Configuration</span>
            </h2>
          </div>

          <div className="card-body space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
            {/* Tabs */}
            <div className="flex space-x-2 border-b border-gray-200">
              {[
                { id: 'data', label: 'Data', icon: Layers },
                { id: 'style', label: 'Style', icon: Palette },
                { id: 'advanced', label: 'Advanced', icon: Sliders },
              ].map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() =>
                      setActiveTab(tab.id as 'data' | 'style' | 'advanced')
                    }
                    className={`flex items-center space-x-2 px-3 py-2 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab.id
                        ? 'border-blue-500 text-blue-600'
                        : 'border-transparent text-gray-600 hover:text-gray-900'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Data Tab */}
            {activeTab === 'data' && (
              <div className="space-y-4">
                {/* Chart Type */}
                <div>
                  <label className="label text-sm font-semibold">
                    Chart Type
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {CHART_TYPES.map((type) => {
                      const Icon = type.icon;
                      return (
                        <button
                          key={type.value}
                          onClick={() =>
                            updateConfig({ type: type.value })
                          }
                          className={`flex items-center justify-center space-x-2 px-3 py-2 rounded-lg border-2 transition-all ${
                            config.type === type.value
                              ? 'border-blue-500 bg-blue-50 text-blue-700'
                              : 'border-gray-200 hover:border-gray-300 text-gray-600'
                          }`}
                          title={type.label}
                        >
                          <Icon className="w-4 h-4" />
                          <span className="text-xs font-medium hidden sm:inline">
                            {type.label}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* X-Axis */}
                <div>
                  <label className="label text-sm font-semibold">
                    X-Axis (Category)
                  </label>
                  <select
                    value={config.xAxis}
                    onChange={(e) =>
                      updateConfig({ xAxis: e.target.value })
                    }
                    className="select w-full text-sm"
                  >
                    <option value="">Select X-Axis</option>
                    {categoricalColumns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Y-Axis */}
                <div>
                  <label className="label text-sm font-semibold">
                    Y-Axis (Value)
                  </label>
                  <select
                    value={
                      Array.isArray(config.yAxis)
                        ? config.yAxis[0]
                        : config.yAxis
                    }
                    onChange={(e) =>
                      updateConfig({ yAxis: e.target.value })
                    }
                    className="select w-full text-sm"
                  >
                    <option value="">Select Y-Axis</option>
                    {numericColumns.map((col) => (
                      <option key={col.name} value={col.name}>
                        {col.name}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Title */}
                <div>
                  <label className="label text-sm font-semibold">
                    Chart Title
                  </label>
                  <input
                    type="text"
                    value={config.title}
                    onChange={(e) =>
                      updateConfig({ title: e.target.value })
                    }
                    className="input w-full text-sm"
                    placeholder="Enter chart title"
                  />
                </div>

                {/* Subtitle */}
                <div>
                  <label className="label text-sm font-semibold">
                    Subtitle (Optional)
                  </label>
                  <input
                    type="text"
                    value={config.subtitle || ''}
                    onChange={(e) =>
                      updateConfig({ subtitle: e.target.value })
                    }
                    className="input w-full text-sm"
                    placeholder="Enter subtitle"
                  />
                </div>
              </div>
            )}

            {/* Style Tab */}
            {activeTab === 'style' && (
              <div className="space-y-4">
                {/* Colors */}
                <div>
                  <label className="label text-sm font-semibold">Colors</label>
                  <div className="grid grid-cols-4 gap-2">
                    {config.colors.map((color, index) => (
                      <div key={index} className="relative">
                        <input
                          type="color"
                          value={color}
                          onChange={(e) =>
                            updateColor(index, e.target.value)
                          }
                          className="w-full h-10 rounded-lg cursor-pointer border-2 border-gray-200 hover:border-gray-300"
                          title={`Color ${index + 1}`}
                        />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Toggle Options */}
                {[
                  { key: 'showGrid', label: 'Show Grid' },
                  { key: 'showLegend', label: 'Show Legend' },
                  { key: 'showTooltip', label: 'Show Tooltip' },
                  {
                    key: 'smoothCurve',
                    label: 'Smooth Curve',
                    show: config.type === 'line' || config.type === 'area',
                  },
                  {
                    key: 'stacked',
                    label: 'Stacked',
                    show: config.type === 'bar' || config.type === 'area',
                  },
                  {
                    key: 'dataLabels',
                    label: 'Show Data Labels',
                    show: config.type === 'pie',
                  },
                ].map(
                  (option) =>
                    (option.show !== false) && (
                      <div
                        key={option.key}
                        className="flex items-center justify-between"
                      >
                        <label className="text-sm font-medium text-gray-700">
                          {option.label}
                        </label>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={
                              config[
                                option.key as keyof ChartConfig
                              ] as boolean
                            }
                            onChange={(e) =>
                              updateConfig({
                                [option.key]: e.target.checked,
                              })
                            }
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                        </label>
                      </div>
                    )
                )}
              </div>
            )}

            {/* Advanced Tab */}
            {activeTab === 'advanced' && (
              <div className="space-y-4">
                {/* Animation Duration */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="label text-sm font-semibold">
                      Animation Duration
                    </label>
                    <span className="text-sm font-mono text-blue-600">
                      {config.animationDuration}ms
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="3000"
                    step="100"
                    value={config.animationDuration}
                    onChange={(e) =>
                      updateConfig({
                        animationDuration: parseInt(e.target.value),
                      })
                    }
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>0ms</span>
                    <span>3000ms</span>
                  </div>
                </div>

                {/* Dot Size (for line/scatter) */}
                {(config.type === 'line' || config.type === 'scatter') && (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="label text-sm font-semibold">
                        Dot Size
                      </label>
                      <span className="text-sm font-mono text-blue-600">
                        {config.dotSize}px
                      </span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="10"
                      step="1"
                      value={config.dotSize || 4}
                      onChange={(e) =>
                        updateConfig({ dotSize: parseInt(e.target.value) })
                      }
                      className="w-full"
                    />
                  </div>
                )}

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                  <p className="text-xs text-gray-700">
                    <TrendingUp className="w-4 h-4 inline mr-1" />
                    <strong>Tip:</strong> Adjust animation duration for smoother
                    or faster transitions.
                  </p>
                </div>

                {/* Chart Config Display */}
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="label text-sm font-semibold">
                      Chart Config (JSON)
                    </label>
                    <button
                      onClick={copyConfig}
                      className="text-sm text-blue-600 hover:text-blue-700 flex items-center space-x-1"
                    >
                      {copied ? (
                        <>
                          <Check className="w-3 h-3" />
                          <span>Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>
                  <code className="block bg-gray-900 text-green-400 p-2 rounded text-xs overflow-auto max-h-32 font-mono">
                    {JSON.stringify(config, null, 2)}
                  </code>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center space-x-3 pt-4 border-t border-gray-200">
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="btn btn-primary flex-1 flex items-center justify-center space-x-2"
              >
                {isSaving ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Save className="w-4 h-4" />
                )}
                <span className="text-sm">
                  {isSaving ? 'Saving...' : 'Save Chart'}
                </span>
              </button>
              <button
                onClick={handleExport}
                className="btn btn-secondary flex items-center space-x-2"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-green-600" />
                ) : (
                  <Download className="w-4 h-4" />
                )}
                <span className="text-sm">Export</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Preview */}
      <div className="lg:col-span-2" ref={chartRef}>
        <div className="card h-full">
          <div className="card-header">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">
                  {config.title}
                </h2>
                {config.subtitle && (
                  <p className="text-sm text-gray-600 mt-1">
                    {config.subtitle}
                  </p>
                )}
              </div>
              <Eye className="w-5 h-5 text-gray-400" />
            </div>
          </div>
          <div
            className="card-body"
            style={{
              height: 'calc(100% - 80px)',
              minHeight: '500px',
            }}
          >
            {renderChart()}
          </div>
        </div>
      </div>
    </div>
  );
};

ChartGenerator.displayName = 'ChartGenerator';

export default ChartGenerator;
