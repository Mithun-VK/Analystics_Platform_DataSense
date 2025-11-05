// src/components/shared/ErrorBoundary.tsx

import type { ReactNode, ErrorInfo, FC, ComponentType } from 'react';
import { Component } from 'react';
import {
  AlertTriangle,
  RefreshCw,
  Home,
  MessageSquare,
  ChevronDown,
} from 'lucide-react';
import Button from './Button';

export interface ErrorBoundaryProps {
  children: ReactNode;
  /** Fallback component to render on error */
  fallback?: (error: Error, retry: () => void) => ReactNode;
  /** Called when an error is caught */
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  /** Show error details in development */
  showDetails?: boolean;
  /** Reset keys to reset error boundary */
  resetKeys?: Array<string | number>;
  /** Custom error message */
  errorMessage?: string;
  /** Custom reset message */
  resetMessage?: string;
  /** Report errors to service */
  onErrorReport?: (error: Error, errorInfo: ErrorInfo) => Promise<void>;
  /** Maximum error count before showing warning */
  maxErrorCount?: number;
  /** Automatically reset after N milliseconds */
  autoResetTimeout?: number;
}

export interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorCount: number;
  isReporting: boolean;
  lastErrorTime: number | null;
}

/**
 * ErrorBoundary - Production-grade error boundary component
 * Features: Error catching, retry mechanism, error reporting, development details,
 * error count tracking, auto-reset, fully accessible
 *
 * @example
 * <ErrorBoundary
 *   onErrorReport={reportToSentry}
 *   onError={(error, errorInfo) => console.log(error)}
 * >
 *   <YourComponent />
 * </ErrorBoundary>
 */
class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  // ✅ Added static displayName property
  static displayName = 'ErrorBoundary';

  private resetTimeoutId: NodeJS.Timeout | null = null;
  private reportTimeoutId: NodeJS.Timeout | null = null;
  private isDevelopment: boolean;

  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorCount: 0,
      isReporting: false,
      lastErrorTime: null,
    };
    // ✅ Cache NODE_ENV check to avoid repeated lookups
    this.isDevelopment = import.meta.env.MODE === 'development';
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
      lastErrorTime: Date.now(),
    };
  }

  override componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    const { maxErrorCount = 5, autoResetTimeout = 5 * 60 * 1000 } = this.props;

    // Update state with error details
    this.setState((prevState) => ({
      errorInfo,
      errorCount: prevState.errorCount + 1,
    }));

    // Call custom error handler
    this.props.onError?.(error, errorInfo);

    // Log error to console in development
    if (this.isDevelopment) {
      console.error('🚨 Error caught by ErrorBoundary:', {
        error,
        errorInfo,
        timestamp: new Date().toISOString(),
      });
    }

    // Auto-report errors if handler provided and under max error count
    if (this.props.onErrorReport && this.state.errorCount < maxErrorCount) {
      // Debounce error reporting
      if (this.reportTimeoutId) {
        clearTimeout(this.reportTimeoutId);
      }

      this.reportTimeoutId = setTimeout(() => {
        this.reportError(error, errorInfo);
      }, 1000);
    }

    // Reset after timeout (for recovery from transient errors)
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
    }

    this.resetTimeoutId = setTimeout(() => {
      this.setState({ errorCount: 0 });
    }, autoResetTimeout);
  }

  override componentDidUpdate(
    prevProps: ErrorBoundaryProps,
    _prevState: ErrorBoundaryState
  ): void {
    // Reset error boundary when reset keys change
    if (this.state.hasError && this.props.resetKeys) {
      const hasResetKeyChanged = this.props.resetKeys.some(
        (key, idx) => key !== prevProps.resetKeys?.[idx]
      );

      if (hasResetKeyChanged) {
        this.resetErrorBoundary();
      }
    }
  }

  override componentWillUnmount(): void {
    if (this.resetTimeoutId) {
      clearTimeout(this.resetTimeoutId);
    }
    if (this.reportTimeoutId) {
      clearTimeout(this.reportTimeoutId);
    }
  }

  private resetErrorBoundary = (): void => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  private reportError = async (
    error: Error,
    errorInfo: ErrorInfo
  ): Promise<void> => {
    if (!this.props.onErrorReport) return;

    this.setState({ isReporting: true });

    try {
      await this.props.onErrorReport(error, errorInfo);

      if (this.isDevelopment) {
        console.log('✅ Error reported successfully');
      }
    } catch (reportError) {
      console.error('❌ Failed to report error:', reportError);
    } finally {
      this.setState({ isReporting: false });
    }
  };

  private navigateHome = (): void => {
    window.location.href = '/';
  };

  private navigateToContact = (): void => {
    window.location.href = '/contact';
  };

  private getErrorSeverity = (): 'low' | 'medium' | 'high' => {
    const { errorCount } = this.state;
    if (errorCount >= 5) return 'high';
    if (errorCount >= 3) return 'medium';
    return 'low';
  };

  private getSeverityColor = (): string => {
    const severity = this.getErrorSeverity();
    switch (severity) {
      case 'high':
        return 'bg-red-50 border-red-200 text-red-700';
      case 'medium':
        return 'bg-yellow-50 border-yellow-200 text-yellow-700';
      default:
        return 'bg-blue-50 border-blue-200 text-blue-700';
    }
  };

  private getSeverityIcon = (): ReactNode => {
    const severity = this.getErrorSeverity();
    const iconClass =
      severity === 'high' ? 'text-red-600' : 'text-yellow-600';
    return <AlertTriangle className={`w-5 h-5 ${iconClass}`} />;
  };

  private generateErrorId = (): string => {
    return Math.random().toString(36).substr(2, 9).toUpperCase();
  };

  override render(): ReactNode {
    const { hasError, error, errorInfo, isReporting, errorCount } = this.state;
    const {
      children,
      fallback,
      showDetails = this.isDevelopment,
      errorMessage,
      resetMessage,
      maxErrorCount = 5,
    } = this.props;

    if (hasError && error) {
      // Use custom fallback if provided
      if (fallback) {
        return fallback(error, this.resetErrorBoundary);
      }

      const severity = this.getErrorSeverity();
      const severityColor = this.getSeverityColor();
      const severityLabel =
        severity === 'high' ? 'Critical' : severity === 'medium' ? 'Warning' : 'Error';
      const errorId = this.generateErrorId();

      // Show error UI
      return (
        <div
          className="min-h-screen bg-gray-50 flex items-center justify-center p-4"
          role="alert"
          aria-live="assertive"
        >
          <div className="max-w-2xl w-full">
            {/* Main Error Card */}
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
              {/* Header */}
              <div className="bg-gradient-to-r from-red-500 to-red-600 px-6 py-4">
                <div className="flex items-center space-x-4">
                  <AlertTriangle className="w-8 h-8 text-white flex-shrink-0" />
                  <div>
                    <h1 className="text-2xl font-bold text-white">
                      Something went wrong
                    </h1>
                    <p className="text-red-100 text-sm mt-1">Error {severityLabel}</p>
                  </div>
                </div>
              </div>

              {/* Content */}
              <div className="px-6 py-6 space-y-6">
                {/* Error Message */}
                <div>
                  <p className="text-gray-700 text-base leading-relaxed">
                    {errorMessage ||
                      "We're sorry, but something unexpected happened. Our team has been notified and is working on a fix."}
                  </p>
                </div>

                {/* Error Count Badge */}
                {errorCount >= 2 && (
                  <div
                    className={`border rounded-lg p-4 flex items-center space-x-3 ${severityColor}`}
                  >
                    {this.getSeverityIcon()}
                    <div>
                      <p className="font-semibold">
                        Multiple errors detected ({errorCount}/{maxErrorCount})
                      </p>
                      <p className="text-sm opacity-90 mt-1">
                        If this persists, please try clearing your cache or contact
                        support.
                      </p>
                    </div>
                  </div>
                )}

                {/* Error Details (Development Only) */}
                {showDetails && errorInfo && (
                  <details className="bg-gray-50 rounded-lg border border-gray-200 overflow-hidden group">
                    <summary className="px-4 py-3 cursor-pointer hover:bg-gray-100 font-medium text-gray-900 flex items-center justify-between transition-colors">
                      <span>Error Details (Development Only)</span>
                      <ChevronDown className="w-5 h-5 transform group-open:rotate-180 transition-transform" />
                    </summary>

                    <div className="border-t border-gray-200 px-4 py-4 space-y-4 bg-white">
                      {/* Error Message */}
                      <div>
                        <h4 className="font-semibold text-gray-900 mb-2 text-sm">
                          Error Message:
                        </h4>
                        <code className="block bg-gray-50 border border-gray-200 rounded p-3 text-sm text-red-700 overflow-auto max-h-32 font-mono">
                          {error.toString()}
                        </code>
                      </div>

                      {/* Component Stack */}
                      {errorInfo.componentStack && (
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-2 text-sm">
                            Component Stack:
                          </h4>
                          <code className="block bg-gray-50 border border-gray-200 rounded p-3 text-sm text-gray-700 overflow-auto max-h-32 font-mono whitespace-pre-wrap">
                            {errorInfo.componentStack}
                          </code>
                        </div>
                      )}

                      {/* Full Stack Trace */}
                      {error.stack && (
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-2 text-sm">
                            Full Stack Trace:
                          </h4>
                          <code className="block bg-gray-50 border border-gray-200 rounded p-3 text-sm text-gray-700 overflow-auto max-h-32 font-mono whitespace-pre-wrap">
                            {error.stack}
                          </code>
                        </div>
                      )}

                      {/* Timestamp */}
                      <div className="text-xs text-gray-500 pt-2 border-t border-gray-200">
                        Timestamp:{' '}
                        {new Date(
                          this.state.lastErrorTime || Date.now()
                        ).toISOString()}
                      </div>
                    </div>
                  </details>
                )}

                {/* Help Section */}
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="font-semibold text-gray-900 mb-3 flex items-center space-x-2">
                    <span>What can you do?</span>
                  </h3>
                  <ul className="space-y-2 text-sm text-gray-700">
                    <li className="flex items-start space-x-2">
                      <span className="text-blue-600 font-bold mt-0.5">•</span>
                      <span>Try refreshing the page</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-blue-600 font-bold mt-0.5">•</span>
                      <span>Clear your browser cache and cookies</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-blue-600 font-bold mt-0.5">•</span>
                      <span>Try using a different browser or incognito mode</span>
                    </li>
                    <li className="flex items-start space-x-2">
                      <span className="text-blue-600 font-bold mt-0.5">•</span>
                      <span>Contact our support team if the problem persists</span>
                    </li>
                  </ul>
                </div>

                {/* Report Status */}
                {isReporting && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 flex items-center space-x-2">
                    <div className="animate-spin">
                      <RefreshCw className="w-4 h-4 text-yellow-600" />
                    </div>
                    <p className="text-sm text-yellow-700 font-medium">
                      Reporting error to our team...
                    </p>
                  </div>
                )}
              </div>

              {/* Footer - Actions */}
              <div className="bg-gray-50 border-t border-gray-200 px-6 py-4 flex flex-col sm:flex-row gap-3">
                <Button
                  variant="primary"
                  leftIcon={RefreshCw}
                  onClick={this.resetErrorBoundary}
                  disabled={isReporting}
                  className="flex-1"
                >
                  {resetMessage || 'Try Again'}
                </Button>
                <Button
                  variant="secondary"
                  leftIcon={Home}
                  onClick={this.navigateHome}
                  className="flex-1"
                >
                  Go Home
                </Button>
                <Button
                  variant="outline"
                  leftIcon={MessageSquare}
                  onClick={this.navigateToContact}
                  className="flex-1"
                >
                  Contact Support
                </Button>
              </div>
            </div>

            {/* Footer Text */}
            <p className="text-center text-gray-600 text-sm mt-4">
              Error ID:{' '}
              <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono">
                {errorId}
              </code>
            </p>
          </div>
        </div>
      );
    }

    return children;
  }
}

export default ErrorBoundary;

// ============================================================================
// Higher-Order Component for wrapping components
// ============================================================================

export interface WithErrorBoundaryOptions
  extends Omit<ErrorBoundaryProps, 'children'> {
  fallback?: (error: Error, retry: () => void) => ReactNode;
}

/**
 * withErrorBoundary - HOC for wrapping components with error boundary
 * @example
 * export default withErrorBoundary(YourComponent, { onError: console.log })
 */
export function withErrorBoundary<P extends Record<string, unknown>>(
  Component: ComponentType<P>,
  errorBoundaryProps?: WithErrorBoundaryOptions
): FC<P> {
  const WrappedComponent: FC<P> = (props: P) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} />
    </ErrorBoundary>
  );

  const displayName =
    (Component as ComponentType & { displayName?: string }).displayName ||
    (Component as ComponentType & { name?: string }).name ||
    'Component';
  WrappedComponent.displayName = `withErrorBoundary(${displayName})`;

  return WrappedComponent;
}

// ============================================================================
// Hook for manual error handling
// ============================================================================

/**
 * useErrorHandler - Hook for manual error handling
 * Logs errors and throws for error boundary to catch
 * @example
 * const handleError = useErrorHandler();
 * handleError(new Error('Something failed'), 'context info');
 */
export const useErrorHandler = (): ((
  error: Error | string,
  info?: string
) => never) => {
  const isDevelopment = import.meta.env.MODE === 'development';

  return (error: Error | string, info?: string): never => {
    const errorMessage =
      typeof error === 'string' ? error : error.message;
    const fullMessage = info ? `${errorMessage}: ${info}` : errorMessage;

    // Log to console
    console.error('🚨 Error from useErrorHandler:', {
      message: fullMessage,
      error,
      info,
      timestamp: new Date().toISOString(),
    });

    // In production, send to error tracking service
    if (!isDevelopment) {
      // Send to error tracking service like Sentry
      // captureException(new Error(fullMessage))
    }

    throw new Error(fullMessage);
  };
};

// ============================================================================
// Error Logger Utility
// ============================================================================

export interface ErrorLog {
  id: string;
  error: {
    message: string;
    name: string;
    stack?: string;
  };
  errorInfo: ErrorInfo;
  timestamp: number;
  userAgent: string;
  url: string;
}

/**
 * createErrorLogger - Factory for creating error loggers
 * Useful for sending errors to external services
 * @example
 * const logger = createErrorLogger('/api/errors');
 * <ErrorBoundary onErrorReport={logger}>...</ErrorBoundary>
 */
export const createErrorLogger =
  (
    endpoint: string
  ): ((error: Error, errorInfo: ErrorInfo) => Promise<void>) =>
  async (error: Error, errorInfo: ErrorInfo): Promise<void> => {
    try {
      const errorLog: Omit<ErrorLog, 'id'> = {
        error: {
          message: error.message,
          name: error.name,
          stack: error.stack,
        },
        errorInfo,
        timestamp: Date.now(),
        userAgent: navigator.userAgent,
        url: window.location.href,
      };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(errorLog),
      });

      if (!response.ok) {
        console.error('Failed to log error:', response.statusText);
      }
    } catch (logError) {
      console.error('Error logging failed:', logError);
    }
  };
