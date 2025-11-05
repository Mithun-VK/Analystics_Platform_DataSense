// src/components/shared/Loading.tsx

import { Loader2 } from 'lucide-react';

export type LoadingVariant = 'spinner' | 'skeleton' | 'dots' | 'pulse';
export type LoadingSize = 'sm' | 'md' | 'lg';

export interface LoadingProps {
  /** Loading indicator variant */
  variant?: LoadingVariant;
  /** Size of the loading indicator */
  size?: LoadingSize;
  /** Loading text to display */
  text?: string;
  /** Full screen overlay loading */
  fullScreen?: boolean;
  /** Custom className */
  className?: string;
}

/**
 * Loading - Loading spinner and skeleton component for async operations
 * Features: Multiple variants (spinner, skeleton, dots, pulse), sizes, overlay mode
 * Accessible with ARIA live region for screen readers
 * 
 * @example
 * <Loading variant="spinner" text="Loading data..." />
 * <Loading variant="skeleton" />
 */
const Loading: React.FC<LoadingProps> = ({
  variant = 'spinner',
  size = 'md',
  text,
  fullScreen = false,
  className = '',
}) => {
  // Get size classes
  const getSizeClasses = () => {
    const sizeMap = {
      sm: 'w-4 h-4',
      md: 'w-8 h-8',
      lg: 'w-12 h-12',
    };
    return sizeMap[size];
  };

  // Render spinner variant
  const renderSpinner = () => (
    <div className={`flex flex-col items-center justify-center ${className}`}>
      <Loader2 className={`${getSizeClasses()} text-blue-600 animate-spin`} />
      {text && <p className="mt-3 text-sm text-gray-600">{text}</p>}
    </div>
  );

  // Render dots variant
  const renderDots = () => (
    <div className={`flex items-center justify-center space-x-2 ${className}`}>
      <div
        className={`${
          size === 'sm' ? 'w-2 h-2' : size === 'lg' ? 'w-4 h-4' : 'w-3 h-3'
        } bg-blue-600 rounded-full animate-bounce`}
        style={{ animationDelay: '0ms' }}
      />
      <div
        className={`${
          size === 'sm' ? 'w-2 h-2' : size === 'lg' ? 'w-4 h-4' : 'w-3 h-3'
        } bg-blue-600 rounded-full animate-bounce`}
        style={{ animationDelay: '150ms' }}
      />
      <div
        className={`${
          size === 'sm' ? 'w-2 h-2' : size === 'lg' ? 'w-4 h-4' : 'w-3 h-3'
        } bg-blue-600 rounded-full animate-bounce`}
        style={{ animationDelay: '300ms' }}
      />
      {text && <p className="ml-3 text-sm text-gray-600">{text}</p>}
    </div>
  );

  // Render pulse variant
  const renderPulse = () => (
    <div className={`flex items-center justify-center ${className}`}>
      <div className={`${getSizeClasses()} relative`}>
        <div className="absolute inset-0 bg-blue-600 rounded-full animate-ping opacity-75" />
        <div className="relative bg-blue-600 rounded-full w-full h-full" />
      </div>
      {text && <p className="ml-3 text-sm text-gray-600">{text}</p>}
    </div>
  );

  // Render skeleton variant (handled separately below)
  const renderContent = () => {
    switch (variant) {
      case 'dots':
        return renderDots();
      case 'pulse':
        return renderPulse();
      case 'skeleton':
        return null; // Skeleton is a separate component
      default:
        return renderSpinner();
    }
  };

  // Full screen overlay
  if (fullScreen) {
    return (
      <div
        className="fixed inset-0 bg-white bg-opacity-90 backdrop-blur-sm flex items-center justify-center z-50"
        role="status"
        aria-live="polite"
        aria-label={text || 'Loading'}
      >
        {renderContent()}
      </div>
    );
  }

  return (
    <div role="status" aria-live="polite" aria-label={text || 'Loading'}>
      {renderContent()}
      <span className="sr-only">{text || 'Loading...'}</span>
    </div>
  );
};

Loading.displayName = 'Loading';

export default Loading;

// Skeleton component for content placeholders
export interface SkeletonProps {
  /** Width of skeleton */
  width?: string | number;
  /** Height of skeleton */
  height?: string | number;
  /** Shape of skeleton */
  variant?: 'text' | 'circular' | 'rectangular';
  /** Number of skeleton lines */
  count?: number;
  /** Custom className */
  className?: string;
  /** Animation type */
  animation?: 'pulse' | 'wave' | 'none';
}

export const Skeleton: React.FC<SkeletonProps> = ({
  width = '100%',
  height,
  variant = 'text',
  count = 1,
  className = '',
  animation = 'pulse',
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'circular':
        return 'rounded-full';
      case 'rectangular':
        return 'rounded-lg';
      default:
        return 'rounded';
    }
  };

  const getAnimationClass = () => {
    switch (animation) {
      case 'wave':
        return 'animate-shimmer';
      case 'pulse':
        return 'animate-pulse';
      default:
        return '';
    }
  };

  const getHeight = () => {
    if (height) return height;
    if (variant === 'text') return '1rem';
    return '100%';
  };

  const skeletonStyle = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof getHeight() === 'number' ? `${getHeight()}px` : getHeight(),
  };

  return (
    <>
      {Array.from({ length: count }).map((_, index) => (
        <div
          key={index}
          className={`skeleton ${getVariantClasses()} ${getAnimationClass()} ${className} ${
            index < count - 1 ? 'mb-2' : ''
          }`}
          style={skeletonStyle}
          role="status"
          aria-label="Loading..."
        />
      ))}
    </>
  );
};

Skeleton.displayName = 'Skeleton';

// Common skeleton patterns
export const CardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`card card-body space-y-4 ${className}`}>
    <div className="flex items-center space-x-4">
      <Skeleton variant="circular" width={48} height={48} />
      <div className="flex-1 space-y-2">
        <Skeleton width="60%" />
        <Skeleton width="40%" />
      </div>
    </div>
    <Skeleton count={3} />
  </div>
);

export const TableSkeleton: React.FC<{ rows?: number; columns?: number }> = ({
  rows = 5,
  columns = 4,
}) => (
  <div className="space-y-2">
    {Array.from({ length: rows }).map((_, rowIndex) => (
      <div key={rowIndex} className="flex space-x-4">
        {Array.from({ length: columns }).map((_, colIndex) => (
          <Skeleton key={colIndex} height="2rem" />
        ))}
      </div>
    ))}
  </div>
);

export const ListSkeleton: React.FC<{ items?: number }> = ({ items = 5 }) => (
  <div className="space-y-4">
    {Array.from({ length: items }).map((_, index) => (
      <div key={index} className="flex items-center space-x-4">
        <Skeleton variant="circular" width={40} height={40} />
        <div className="flex-1 space-y-2">
          <Skeleton width="80%" />
          <Skeleton width="60%" />
        </div>
      </div>
    ))}
  </div>
);
