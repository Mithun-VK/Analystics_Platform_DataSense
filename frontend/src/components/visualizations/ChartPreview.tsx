// src/components/visualizations/ChartPreview.tsx - FIXED VERSION

import { useMemo } from 'react';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import { AlertCircle, Share2, Edit, Trash2 } from 'lucide-react';
import Button from '@/components/shared/Button';

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * ✅ FIXED: Accept Visualization object, not array
 */
export interface ChartPreviewProps {
  data: {
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
  };
  chartData?: any[];
  createdAt?: string;
  onEdit?: () => void;
  onDelete?: () => void;
  onShare?: (platform: 'email' | 'slack' | 'teams') => void;
}

// ============================================================================
// Mock Data
// ============================================================================

const MOCK_CHART_DATA = [
  { name: 'Jan', value: 4000, revenue: 2400 },
  { name: 'Feb', value: 3000, revenue: 1398 },
  { name: 'Mar', value: 2000, revenue: 9800 },
  { name: 'Apr', value: 2780, revenue: 3908 },
  { name: 'May', value: 1890, revenue: 4800 },
  { name: 'Jun', value: 2390, revenue: 3800 },
];

const COLORS = [
  '#3B82F6',
  '#10B981',
  '#F59E0B',
  '#EF4444',
  '#8B5CF6',
  '#EC4899',
];

// ============================================================================
// Component
// ============================================================================

/**
 * ChartPreview - Display chart preview for visualizations
 * ✅ FIXED: Accepts Visualization object
 */
const ChartPreview: React.FC<ChartPreviewProps> = ({
  data,
  chartData = MOCK_CHART_DATA,
  onEdit,
  onDelete,
  onShare,
}) => {
  const renderChart = useMemo(() => {
    if (!data) {
      return (
        <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
          <div className="text-center">
            <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-2" />
            <p className="text-gray-600">No data available</p>
          </div>
        </div>
      );
    }

    if (data.status === 'processing') {
      return (
        <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="animate-spin mb-4">
              <div className="h-8 w-8 border-4 border-primary-500 border-t-primary-600 rounded-full mx-auto"></div>
            </div>
            <p className="text-gray-600">Processing chart...</p>
          </div>
        </div>
      );
    }

    if (data.status === 'failed') {
      return (
        <div className="flex items-center justify-center h-96 bg-red-50 rounded-lg">
          <div className="text-center">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-2" />
            <p className="text-red-600">Failed to load chart</p>
          </div>
        </div>
      );
    }

    switch (data.chartType) {
      case 'line':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={data.config.xAxis || 'name'} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey={data.config.yAxis || 'revenue'}
                stroke="#3B82F6"
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        );

      case 'bar':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={data.config.xAxis || 'name'} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey={data.config.yAxis || 'value'} fill="#3B82F6" />
              {data.config.groupBy && (
                <Bar dataKey={data.config.groupBy} fill="#10B981" />
              )}
            </BarChart>
          </ResponsiveContainer>
        );

      case 'area':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={data.config.xAxis || 'name'} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Area
                type="monotone"
                dataKey={data.config.yAxis || 'revenue'}
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        );

      case 'pie':
        return (
          <ResponsiveContainer width="100%" height={400}>
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={120}
                fill="#8884d8"
                dataKey={data.config.yAxis || 'value'}
              >
                {chartData.map((_entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        );

      default:
        return (
          <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
            <p className="text-gray-600">Unknown chart type</p>
          </div>
        );
    }
  }, [data, chartData]);

  return (
    <div className="space-y-4">
      {/* Chart Information */}
      {data && (
        <div className="space-y-2 pb-4 border-b border-gray-200">
          {data.description && (
            <p className="text-sm text-gray-600">{data.description}</p>
          )}
          {data.tags && data.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {data.tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-block px-2 py-1 text-xs bg-primary-100 text-primary-800 rounded-full"
                >
                  {tag}
                </span>
              ))}
            </div>
          )}
          <div className="text-xs text-gray-500 space-y-1">
            <p>Dataset: {data.datasetName}</p>
            <p>Type: {data.config.xAxis} vs {data.config.yAxis}</p>
          </div>
        </div>
      )}

      {/* Chart Display */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        {renderChart}
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2 pt-4 border-t border-gray-200">
        {onEdit && (
          <Button
            variant="secondary"
            size="sm"
            leftIcon={Edit}
            onClick={onEdit}
            fullWidth
          >
            Edit
          </Button>
        )}
        {onShare && (
          <Button
            variant="secondary"
            size="sm"
            leftIcon={Share2}
            onClick={() => onShare('email')}
            fullWidth
          >
            Share
          </Button>
        )}
        {onDelete && (
          <Button
            variant="secondary"
            size="sm"
            leftIcon={Trash2}
            onClick={onDelete}
            fullWidth
            className="text-red-600 hover:text-red-700"
          >
            Delete
          </Button>
        )}
      </div>
    </div>
  );
};

ChartPreview.displayName = 'ChartPreview';

export default ChartPreview;
