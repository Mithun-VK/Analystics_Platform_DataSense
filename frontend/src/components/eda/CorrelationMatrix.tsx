// src/components/eda/CorrelationMatrix.tsx

import {useState, useMemo, useRef } from 'react';
import {
  Download,
  ZoomIn,
  ZoomOut,
  Info,
  Eye,
  Filter,
  Search,
} from 'lucide-react';

interface CorrelationMatrixProps {
  matrix: number[][];
  columns: string[];
  width?: number;
  height?: number;
}

/**
 * CorrelationMatrix - Interactive heatmap visualization of feature correlations
 * Features: Color-coded cells, tooltips, zoom controls, filtering, export
 * Displays correlation coefficients with diverging color scale (-1 to +1)
 */
const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  matrix,
  columns,
  width = 800,
  height = 800,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{
    row: number;
    col: number;
    value: number;
  } | null>(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showValues, setShowValues] = useState(true);
  const [filterThreshold, setFilterThreshold] = useState(0);
  const [searchQuery, setSearchQuery] = useState('');
  const [displayMode, setDisplayMode] = useState<'full' | 'lower' | 'upper'>('full');

  // Filter columns based on search
  const filteredIndices = useMemo(() => {
    if (!searchQuery) return columns.map((_, i) => i);
    
    return columns
      .map((col, i) => ({ col, i }))
      .filter(({ col }) => col.toLowerCase().includes(searchQuery.toLowerCase()))
      .map(({ i }) => i);
  }, [columns, searchQuery]);

  // Calculate dimensions
  const margin = { top: 120, right: 20, bottom: 20, left: 120 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const cellSize = Math.min(
    innerWidth / filteredIndices.length,
    innerHeight / filteredIndices.length
  );

  // Color scale for correlation values
  const getColor = (value: number): string => {
    if (value === null || value === undefined || isNaN(value)) {
      return '#f3f4f6';
    }

    // Apply filter threshold
    if (Math.abs(value) < filterThreshold) {
      return '#f3f4f6';
    }

    // Diverging color scale: blue (negative) -> white (0) -> red (positive)
    if (value < 0) {
      const intensity = Math.abs(value);
      const r = Math.round(255 - intensity * 186); // 255 -> 69
      const g = Math.round(255 - intensity * 116); // 255 -> 139
      const b = 255; // stays 255
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const intensity = value;
      const r = 255; // stays 255
      const g = Math.round(255 - intensity * 156); // 255 -> 99
      const b = Math.round(255 - intensity * 138); // 255 -> 117
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // Get text color based on background
  const getTextColor = (value: number): string => {
    if (Math.abs(value) > 0.5) return '#ffffff';
    return '#374151';
  };

  // Handle mouse move for tooltip
  const handleMouseMove = (
    e: React.MouseEvent,
    row: number,
    col: number,
    value: number
  ) => {
    setHoveredCell({ row, col, value });
    setTooltipPos({ x: e.clientX, y: e.clientY });
  };

  // Export as PNG
  const exportAsPNG = () => {
    if (!svgRef.current) return;

    const svgData = new XMLSerializer().serializeToString(svgRef.current);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');

    const img = new Image();
    const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      if (ctx) {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, width, height);
        ctx.drawImage(img, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) {
            const link = document.createElement('a');
            link.download = 'correlation-matrix.png';
            link.href = URL.createObjectURL(blob);
            link.click();
          }
        });
      }
      URL.revokeObjectURL(url);
    };

    img.src = url;
  };

  // Export correlation data as CSV
  const exportAsCSV = () => {
    const headers = ['Variable', ...columns];
    const rows = matrix.map((row, i) => [columns[i], ...row.map(v => v.toFixed(3))]);
    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const link = document.createElement('a');
    link.download = 'correlation-matrix.csv';
    link.href = URL.createObjectURL(blob);
    link.click();
  };

  // Should cell be displayed based on display mode
  const shouldDisplayCell = (row: number, col: number): boolean => {
    if (displayMode === 'full') return true;
    if (displayMode === 'lower') return row >= col;
    if (displayMode === 'upper') return row <= col;
    return true;
  };

  // Get correlation strength label
  const getCorrelationStrength = (value: number): string => {
    const abs = Math.abs(value);
    if (abs >= 0.9) return 'Very Strong';
    if (abs >= 0.7) return 'Strong';
    if (abs >= 0.5) return 'Moderate';
    if (abs >= 0.3) return 'Weak';
    return 'Very Weak';
  };

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        {/* Search and Filter */}
        <div className="flex flex-col sm:flex-row gap-3 flex-1">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search variables..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-600" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={filterThreshold}
              onChange={(e) => setFilterThreshold(parseFloat(e.target.value))}
              className="w-32"
            />
            <span className="text-sm text-gray-600 w-12">
              {filterThreshold.toFixed(1)}
            </span>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-2">
          <select
            value={displayMode}
            onChange={(e) => setDisplayMode(e.target.value as any)}
            className="select text-sm"
          >
            <option value="full">Full Matrix</option>
            <option value="lower">Lower Triangle</option>
            <option value="upper">Upper Triangle</option>
          </select>

          <button
            onClick={() => setShowValues(!showValues)}
            className={`btn btn-secondary btn-sm ${showValues ? 'bg-blue-50' : ''}`}
            title="Toggle values"
          >
            <Eye className="w-4 h-4" />
          </button>

          <button
            onClick={() => setZoomLevel(Math.max(0.5, zoomLevel - 0.25))}
            className="btn btn-secondary btn-sm"
            disabled={zoomLevel <= 0.5}
            title="Zoom out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>

          <button
            onClick={() => setZoomLevel(Math.min(2, zoomLevel + 0.25))}
            className="btn btn-secondary btn-sm"
            disabled={zoomLevel >= 2}
            title="Zoom in"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <button
            onClick={exportAsPNG}
            className="btn btn-secondary btn-sm"
            title="Export as PNG"
          >
            <Download className="w-4 h-4" />
          </button>

          <button
            onClick={exportAsCSV}
            className="btn btn-secondary btn-sm"
            title="Export as CSV"
          >
            CSV
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <span className="text-sm font-medium text-gray-700">
              Correlation Strength:
            </span>
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(69, 139, 255)' }}></div>
                <span className="text-xs text-gray-600">-1.0 (Perfect Negative)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded bg-white border border-gray-300"></div>
                <span className="text-xs text-gray-600">0.0 (No Correlation)</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(255, 99, 117)' }}></div>
                <span className="text-xs text-gray-600">+1.0 (Perfect Positive)</span>
              </div>
            </div>
          </div>
          {filterThreshold > 0 && (
            <span className="text-xs text-gray-600">
              Hiding correlations below {filterThreshold.toFixed(1)}
            </span>
          )}
        </div>
      </div>

      {/* Heatmap Container */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 overflow-auto">
        <div
          style={{
            transform: `scale(${zoomLevel})`,
            transformOrigin: 'top left',
            transition: 'transform 0.2s',
          }}
        >
          <svg
            ref={svgRef}
            width={width}
            height={height}
            className="mx-auto"
          >
            <g transform={`translate(${margin.left}, ${margin.top})`}>
              {/* Column labels (top) */}
              {filteredIndices.map((colIdx, i) => (
                <text
                  key={`col-label-${i}`}
                  x={i * cellSize + cellSize / 2}
                  y={-10}
                  textAnchor="start"
                  transform={`rotate(-45, ${i * cellSize + cellSize / 2}, -10)`}
                  className="text-xs fill-gray-700"
                  style={{ fontSize: '11px' }}
                >
                  {columns[colIdx].length > 15
                    ? columns[colIdx].substring(0, 15) + '...'
                    : columns[colIdx]}
                </text>
              ))}

              {/* Row labels (left) */}
              {filteredIndices.map((rowIdx, i) => (
                <text
                  key={`row-label-${i}`}
                  x={-10}
                  y={i * cellSize + cellSize / 2 + 4}
                  textAnchor="end"
                  className="text-xs fill-gray-700"
                  style={{ fontSize: '11px' }}
                >
                  {columns[rowIdx].length > 15
                    ? columns[rowIdx].substring(0, 15) + '...'
                    : columns[rowIdx]}
                </text>
              ))}

              {/* Heatmap cells */}
              {filteredIndices.map((rowIdx, i) =>
                filteredIndices.map((colIdx, j) => {
                  if (!shouldDisplayCell(rowIdx, colIdx)) return null;

                  const value = matrix[rowIdx][colIdx];
                  const isHovered =
                    hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;

                  return (
                    <g key={`cell-${i}-${j}`}>
                      <rect
                        x={j * cellSize}
                        y={i * cellSize}
                        width={cellSize}
                        height={cellSize}
                        fill={getColor(value)}
                        stroke={isHovered ? '#3b82f6' : '#e5e7eb'}
                        strokeWidth={isHovered ? 2 : 0.5}
                        className="cursor-pointer transition-all duration-150"
                        onMouseMove={(e) => handleMouseMove(e, rowIdx, colIdx, value)}
                        onMouseLeave={() => setHoveredCell(null)}
                        style={{
                          opacity: isHovered ? 1 : 0.95,
                        }}
                      />
                      {showValues && cellSize > 40 && (
                        <text
                          x={j * cellSize + cellSize / 2}
                          y={i * cellSize + cellSize / 2 + 4}
                          textAnchor="middle"
                          className="pointer-events-none"
                          fill={getTextColor(value)}
                          style={{ fontSize: '10px', fontWeight: 600 }}
                        >
                          {value.toFixed(2)}
                        </text>
                      )}
                    </g>
                  );
                })
              )}
            </g>
          </svg>
        </div>
      </div>

      {/* Tooltip */}
      {hoveredCell && (
        <div
          className="fixed z-50 bg-gray-900 text-white px-4 py-3 rounded-lg shadow-2xl pointer-events-none"
          style={{
            left: tooltipPos.x + 15,
            top: tooltipPos.y + 15,
            maxWidth: '300px',
          }}
        >
          <div className="space-y-2">
            <div className="flex items-center space-x-2 pb-2 border-b border-gray-700">
              <Info className="w-4 h-4" />
              <span className="font-semibold text-sm">Correlation Details</span>
            </div>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Variable 1:</span>
                <span className="font-medium">{columns[hoveredCell.row]}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Variable 2:</span>
                <span className="font-medium">{columns[hoveredCell.col]}</span>
              </div>
              <div className="flex justify-between pt-1 border-t border-gray-700">
                <span className="text-gray-400">Coefficient:</span>
                <span
                  className="font-bold"
                  style={{
                    color:
                      hoveredCell.value > 0
                        ? '#ff6375'
                        : hoveredCell.value < 0
                        ? '#458bff'
                        : '#ffffff',
                  }}
                >
                  {hoveredCell.value.toFixed(4)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Strength:</span>
                <span className="font-medium">
                  {getCorrelationStrength(hoveredCell.value)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Info Panel */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start space-x-3">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1 text-sm">
            <p className="text-gray-700">
              <strong>Interpretation:</strong> Values close to +1 indicate strong positive
              correlation, values close to -1 indicate strong negative correlation, and
              values near 0 indicate little to no linear correlation. Hover over cells for
              detailed information.
            </p>
          </div>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-600 mb-1">Total Correlations</p>
          <p className="text-2xl font-bold text-gray-900">
            {(filteredIndices.length * (filteredIndices.length - 1)) / 2}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-600 mb-1">Strong Correlations</p>
          <p className="text-2xl font-bold text-gray-900">
            {matrix.reduce((count, row, i) => {
              return (
                count +
                row.filter((val, j) => i < j && Math.abs(val) >= 0.7).length
              );
            }, 0)}
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-gray-600 mb-1">Variables Displayed</p>
          <p className="text-2xl font-bold text-gray-900">
            {filteredIndices.length} / {columns.length}
          </p>
        </div>
      </div>
    </div>
  );
};

export default CorrelationMatrix;
