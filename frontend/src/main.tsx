// src/main.tsx

/**
 * Application Entry Point
 * React 18 root initialization with React Query and other configurations
 * ✅ ENHANCED: React Profiler logging, performance monitoring, and error handling
 */

/// <reference types="vite/client" />

import { createRoot } from 'react-dom/client';
import { Profiler, ProfilerOnRenderCallback } from 'react';
import App from './App';
import './styles/global.css';

// ============================================================================
// Type Definitions
// ============================================================================

interface PerformanceEntryWithDuration extends PerformanceEntry {
  duration: number;
}

interface PerformanceMetric {
  name: string;
  duration: number;
  startTime: number;
  timestamp: number;
}

// ============================================================================
// Performance Monitoring
// ============================================================================

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private isProduction: boolean;
  private maxMetrics: number = 100;

  constructor(isProduction: boolean) {
    this.isProduction = isProduction;
  }

  /**
   * ✅ FIXED: React Profiler callback for component render times
   */
  onRenderCallback: ProfilerOnRenderCallback = (
    id: any, // The unique string id of the Profiler
    phase: any, // "mount" or "update"
    actualDuration: number, // Time spent rendering the component
    baseDuration: number, // Estimated time without memoization
    startTime: number, // When React started rendering
    commitTime: number, // When React committed an update
  ) => {
    const metric: PerformanceMetric = {
      name: `Profiler: ${id} (${phase})`,
      duration: actualDuration,
      startTime,
      timestamp: Date.now(),
    };

    this.recordMetric(metric);

    // Log slow renders (> 5ms)
    if (actualDuration > 5) {
      console.warn(`⚠️ Slow render detected: ${id} took ${actualDuration.toFixed(2)}ms`);
    }

    // Debug logging in development
    if (!this.isProduction && actualDuration > 1) {
      console.debug(`✅ Profiler: ${id} (${phase})`, {
        actualDuration: `${actualDuration.toFixed(2)}ms`,
        baseDuration: `${baseDuration.toFixed(2)}ms`,
        startTime: `${startTime.toFixed(2)}ms`,
        commitTime: `${commitTime.toFixed(2)}ms`,
      });
    }
  };

  /**
   * ✅ FIXED: Monitor Web Vitals
   */
  monitorWebVitals() {
    // Largest Contentful Paint
    if ('PerformanceObserver' in window) {
      try {
        const lcpObserver = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1] as PerformanceEntryWithDuration;
          
          const metric: PerformanceMetric = {
            name: 'Web Vital: LCP (Largest Contentful Paint)',
            duration: lastEntry.duration,
            startTime: lastEntry.startTime,
            timestamp: Date.now(),
          };

          this.recordMetric(metric);

          if (!this.isProduction) {
            console.log(`📊 LCP: ${lastEntry.startTime.toFixed(2)}ms`);
          }
        });

        lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

        // Cumulative Layout Shift
        const clsObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            const metric: PerformanceMetric = {
              name: 'Web Vital: CLS (Cumulative Layout Shift)',
              duration: (entry as any).value || 0,
              startTime: entry.startTime,
              timestamp: Date.now(),
            };

            this.recordMetric(metric);

            if (!this.isProduction) {
              console.log(`📊 CLS: ${((entry as any).value || 0).toFixed(3)}`);
            }
          }
        });

        clsObserver.observe({ entryTypes: ['layout-shift'] });

        // First Input Delay
        const fidObserver = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            const metric: PerformanceMetric = {
              name: 'Web Vital: FID (First Input Delay)',
              duration: (entry as any).processingDuration || 0,
              startTime: entry.startTime,
              timestamp: Date.now(),
            };

            this.recordMetric(metric);

            if (!this.isProduction) {
              console.log(`📊 FID: ${((entry as any).processingDuration || 0).toFixed(2)}ms`);
            }
          }
        });

        fidObserver.observe({ entryTypes: ['first-input'] });
      } catch (error) {
        console.warn('Web Vitals monitoring setup failed:', error);
      }
    }
  }

  /**
   * ✅ FIXED: Monitor performance entries
   */
  monitorPerformanceEntries() {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            const metric: PerformanceMetric = {
              name: entry.name,
              duration: (entry as PerformanceEntryWithDuration).duration || 0,
              startTime: entry.startTime,
              timestamp: Date.now(),
            };

            this.recordMetric(metric);

            if (!this.isProduction) {
              console.debug('⏱️ Performance Entry:', {
                name: entry.name,
                duration: `${((entry as PerformanceEntryWithDuration).duration || 0).toFixed(2)}ms`,
                startTime: `${entry.startTime.toFixed(2)}ms`,
              });
            }
          }
        });

        observer.observe({
          entryTypes: ['measure', 'navigation', 'resource', 'paint'],
          buffered: true,
        });
      } catch (error) {
        console.warn('Performance monitoring setup failed:', error);
      }
    }
  }

  /**
   * Record a performance metric
   */
  private recordMetric(metric: PerformanceMetric) {
    this.metrics.push(metric);

    // Keep only recent metrics
    if (this.metrics.length > this.maxMetrics) {
      this.metrics.shift();
    }
  }

  /**
   * Get all recorded metrics
   */
  getMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  /**
   * Get metrics summary
   */
  getSummary() {
    const avgDuration = this.metrics.length > 0
      ? this.metrics.reduce((sum, m) => sum + m.duration, 0) / this.metrics.length
      : 0;

    const maxDuration = Math.max(...this.metrics.map(m => m.duration), 0);
    const slowRenders = this.metrics.filter(m => m.duration > 5).length;

    return {
      totalMetrics: this.metrics.length,
      averageDuration: avgDuration.toFixed(2),
      maxDuration: maxDuration.toFixed(2),
      slowRenders,
    };
  }

  /**
   * Export metrics for analysis
   */
  exportMetrics(format: 'json' | 'csv' = 'json') {
    if (format === 'json') {
      return JSON.stringify(this.metrics, null, 2);
    }

    // CSV format
    const headers = ['name', 'duration', 'startTime', 'timestamp'];
    const rows = this.metrics.map(m => [m.name, m.duration, m.startTime, m.timestamp]);
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');

    return csv;
  }

  /**
   * Log summary to console
   */
  logSummary() {
    const summary = this.getSummary();
    console.group('📊 Performance Summary');
    console.table(summary);
    console.groupEnd();
  }
}

// ============================================================================
// Main Initialization
// ============================================================================

/**
 * Initialize and render the application
 */
const main = async () => {
  // Get root element
  const rootElement = document.getElementById('root');

  if (!rootElement) {
    throw new Error('Root element not found in DOM');
  }

  // ✅ FIXED: Use proper type checking for production mode
  const isProduction = import.meta.env.MODE === 'production';

  // Initialize performance monitor
  const perfMonitor = new PerformanceMonitor(isProduction);

  // ✅ Expose performance monitor globally for debugging
  if (!isProduction) {
    (window as any).__perfMonitor = perfMonitor;
    console.log('💡 Performance monitor available at window.__perfMonitor');
  }

  // Setup error logging
  if (isProduction) {
    // Handle uncaught errors
    window.addEventListener('error', (event) => {
      console.error('❌ Uncaught error:', event.error);
      // Send to error tracking service (e.g., Sentry)
    });

    // Handle unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      console.error('❌ Unhandled promise rejection:', event.reason);
      // Send to error tracking service
    });
  }

  // ✅ Setup performance monitoring
  perfMonitor.monitorPerformanceEntries();
  perfMonitor.monitorWebVitals();

  // Setup network status monitoring
  const handleNetworkChange = () => {
    const isOnline = navigator.onLine;
    console.log(`🌐 Network status: ${isOnline ? '✅ Online' : '❌ Offline'}`);

    // Dispatch event to stores if needed
    if (isOnline) {
      // Resume operations
      document.dispatchEvent(new CustomEvent('app:online'));
    } else {
      // Pause operations
      document.dispatchEvent(new CustomEvent('app:offline'));
    }
  };

  window.addEventListener('online', handleNetworkChange);
  window.addEventListener('offline', handleNetworkChange);

  // Setup page visibility monitoring
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      console.debug('👁️ Page is hidden');
      perfMonitor.logSummary();
    } else {
      console.debug('👁️ Page is visible');
    }
  });

  // ✅ Create React 18 root with Profiler
  const root = createRoot(rootElement);

  // ✅ FIXED: Wrap App with Profiler for performance monitoring
  root.render(
    <Profiler id="App" onRender={perfMonitor.onRenderCallback}>
      <App />
    </Profiler>
  );

  // Log initial performance metrics
  if (!isProduction) {
    console.log('✅ Application initialized successfully');
  }

  // Cleanup on app unload
  window.addEventListener('beforeunload', () => {
    // Log final metrics before unload
    if (!isProduction) {
      perfMonitor.logSummary();
    }

    // Cleanup code
    window.removeEventListener('online', handleNetworkChange);
    window.removeEventListener('offline', handleNetworkChange);
  });

  // Log performance summary periodically (every 30s in development)
  if (!isProduction) {
    setInterval(() => {
      perfMonitor.logSummary();
    }, 30000);
  }
};

// Run main function
main().catch((error) => {
  console.error('❌ Failed to initialize application:', error);
  // Render error page
  const rootElement = document.getElementById('root');
  if (rootElement) {
    rootElement.innerHTML = `
      <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        font-family: system-ui, -apple-system, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      ">
        <div style="
          text-align: center;
          padding: 2rem;
          background: white;
          border-radius: 0.75rem;
          box-shadow: 0 20px 25px rgba(0,0,0,0.15);
          max-width: 500px;
        ">
          <div style="
            font-size: 3rem;
            margin-bottom: 1rem;
          ">
            ⚠️
          </div>
          <h1 style="
            font-size: 1.875rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #1f2937;
          ">
            Failed to Initialize
          </h1>
          <p style="
            color: #6b7280;
            margin-bottom: 1.5rem;
            font-size: 1rem;
            line-height: 1.5;
          ">
            ${error instanceof Error ? error.message : 'An unknown error occurred'}
          </p>
          <button 
            onclick="location.reload()"
            style="
              padding: 0.75rem 1.5rem;
              background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: white;
              border: none;
              border-radius: 0.375rem;
              cursor: pointer;
              font-size: 1rem;
              font-weight: 600;
              transition: all 0.3s;
            "
            onmouseover="this.style.transform='scale(1.05); this.style.boxShadow='0 10px 15px rgba(0,0,0,0.2)'"
            onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none'"
          >
            Try Again
          </button>
          <p style="
            color: #9ca3af;
            margin-top: 1rem;
            font-size: 0.875rem;
          ">
            If this persists, please clear your browser cache and try again.
          </p>
        </div>
      </div>
    `;
  }
});

// ============================================================================
// Type Definitions
// ============================================================================

declare global {
  interface Window {
    __perfMonitor?: PerformanceMonitor;
  }
}

export {};
