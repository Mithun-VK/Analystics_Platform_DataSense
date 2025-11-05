// src/App.tsx

/**
 * Root Application Component
 * ✅ PRODUCTION READY: Full workflow with real API
 * ✅ TESTING MODE: Authentication only (via environment variable)
 * Main entry point with routing, providers, error boundaries, and global configuration
 */

import { useState, useEffect, Suspense, lazy, useCallback, useRef, memo } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
  useLocation,
} from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'sonner';

// Stores
import { uiStore } from '@/store/uiStore';
import { authStore } from '@/store/authStore';

// Components
import ErrorBoundary from '@/components/shared/ErrorBoundary';
import ProtectedRoute from '@/components/auth/ProtectedRoute';
import Loading from '@/components/shared/Loading';
import Header from '@/components/shared/Header';
import Footer from '@/components/shared/Footer';

// API & Services
import queryClient from '@/services/api';

// ============================================================================
// Configuration
// ============================================================================

/**
 * ✅ AUTHENTICATION MODE: Read from environment
 * VITE_AUTH_MODE=testing → Mock authentication
 * VITE_AUTH_MODE=production → Real authentication
 */
const AUTH_MODE = (import.meta.env.VITE_AUTH_MODE || 'production') as 'testing' | 'production';

const ENABLE_DEBUG = import.meta.env.DEV;

// ============================================================================
// Lazy Page Imports
// ============================================================================

const LandingPage = lazy(() =>
  import('@/pages/LandingPage').then((module) => ({
    default: module.default,
  }))
);

const LoginPage = lazy(() =>
  import('@/pages/LoginPage').then((module) => ({
    default: module.default,
  }))
);

const RegisterPage = lazy(() =>
  import('@/pages/RegisterPage').then((module) => ({
    default: module.default,
  }))
);

const DashboardPage = lazy(() =>
  import('@/pages/DashboardPage').then((module) => ({
    default: module.default,
  }))
);

const DatasetsPage = lazy(() =>
  import('@/pages/DatasetsPage').then((module) => ({
    default: module.default,
  }))
);

const DatasetDetailPage = lazy(() =>
  import('@/pages/DatasetDetailPage').then((module) => ({
    default: module.default,
  }))
);

const EDAPage = lazy(() =>
  import('@/pages/EDAPage').then((module) => ({
    default: module.default,
  }))
);

const VisualizationsPage = lazy(() =>
  import('@/pages/VisualizationsPage').then((module) => ({
    default: module.default,
  }))
);

const ProfilePage = lazy(() =>
  import('@/pages/ProfilePage').then((module) => ({
    default: module.default,
  }))
);

const NotFoundPage = lazy(() =>
  import('@/pages/NotFoundPage').then((module) => ({
    default: module.default,
  }))
);

// ============================================================================
// Loading Components
// ============================================================================

/**
 * ✅ Memoized page loading component
 */
const PageLoading = memo(({ message = 'Loading page...' }: { message?: string }) => (
  <div className="flex items-center justify-center min-h-screen bg-white dark:bg-gray-900">
    <div className="flex flex-col items-center gap-4">
      <Loading size="md" />
      {message && (
        <p className="text-gray-600 dark:text-gray-400 text-sm font-medium">
          {message}
        </p>
      )}
    </div>
  </div>
));

PageLoading.displayName = 'PageLoading';

/**
 * ✅ Memoized app loading component
 */
const AppLoading = memo(({ message = 'Loading...' }: { message?: string }) => (
  <div className="flex items-center justify-center min-h-screen bg-white dark:bg-gray-900">
    <div className="flex flex-col items-center gap-4">
      <Loading size="lg" />
      {message && (
        <p className="text-gray-600 dark:text-gray-400 text-sm font-medium">
          {message}
        </p>
      )}
    </div>
  </div>
));

AppLoading.displayName = 'AppLoading';

// ============================================================================
// Route Content Component
// ============================================================================

/**
 * ✅ Separate component for routes with useLocation inside Router context
 */
const AppRoutes = memo(({ isAuthenticated }: { isAuthenticated: boolean }) => {
  const location = useLocation();

  return (
    <ErrorBoundary>
      <Suspense fallback={<PageLoading message="Loading page..." />}>
        <Routes>
          {/* ============================================================================
              Public Routes
              ============================================================================ */}

          <Route path="/" element={<LandingPage />} />

          {/* ✅ Redirect authenticated users away from login/register */}
          <Route
            path="/login"
            element={
              isAuthenticated ? (
                <Navigate to="/dashboard" replace state={{ from: location }} />
              ) : (
                <LoginPage />
              )
            }
          />

          <Route
            path="/register"
            element={
              isAuthenticated ? (
                <Navigate to="/dashboard" replace state={{ from: location }} />
              ) : (
                <RegisterPage />
              )
            }
          />

          {/* ============================================================================
              Protected Routes - Dashboard & Analytics
              ============================================================================ */}

          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <ErrorBoundary>
                  <Suspense fallback={<PageLoading message="Loading dashboard..." />}>
                    <DashboardPage />
                  </Suspense>
                </ErrorBoundary>
              </ProtectedRoute>
            }
          />

          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <ErrorBoundary>
                  <Suspense fallback={<PageLoading message="Loading profile..." />}>
                    <ProfilePage />
                  </Suspense>
                </ErrorBoundary>
              </ProtectedRoute>
            }
          />

          {/* ============================================================================
              Protected Routes - Datasets & Analysis
              ============================================================================ */}

          {/* Datasets List */}
          <Route
            path="/datasets"
            element={
              <ProtectedRoute>
                <ErrorBoundary>
                  <Suspense fallback={<PageLoading message="Loading datasets..." />}>
                    <DatasetsPage />
                  </Suspense>
                </ErrorBoundary>
              </ProtectedRoute>
            }
          />

          {/* Dataset Detail */}
          <Route
            path="/datasets/:id"
            element={
              <ProtectedRoute>
                <ErrorBoundary>
                  <Suspense fallback={<PageLoading message="Loading dataset..." />}>
                    <DatasetDetailPage />
                  </Suspense>
                </ErrorBoundary>
              </ProtectedRoute>
            }
          />

          {/* ✅ EDA with Dataset ID Parameter */}
          <Route
            path="/eda/:datasetId"
            element={
              <ProtectedRoute>
                <ErrorBoundary>
                  <Suspense fallback={<PageLoading message="Loading EDA..." />}>
                    <EDAPage />
                  </Suspense>
                </ErrorBoundary>
              </ProtectedRoute>
            }
          />

          {/* ✅ Redirect /eda to /datasets (missing datasetId) */}
          <Route path="/eda" element={<Navigate to="/datasets" replace />} />

          {/* Visualizations */}
          <Route
            path="/visualizations"
            element={
              <ProtectedRoute>
                <ErrorBoundary>
                  <Suspense fallback={<PageLoading message="Loading visualizations..." />}>
                    <VisualizationsPage />
                  </Suspense>
                </ErrorBoundary>
              </ProtectedRoute>
            }
          />

          {/* ============================================================================
              Error Routes - Must be Last
              ============================================================================ */}

          {/* 404 Not Found - Must be last route */}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
});

AppRoutes.displayName = 'AppRoutes';

// ============================================================================
// Main App Component
// ============================================================================

/**
 * ✅ PRODUCTION READY: Main App component
 */
const App = memo(() => {
  // ✅ Use memoized selectors to prevent unnecessary re-renders
  const isAuthenticated = authStore((state) => state.isAuthenticated);

  // ✅ Get UI store methods with proper selectors
  const isDarkMode = uiStore((state) => state.isDarkMode);
  const setDarkMode = uiStore((state) => state.setDarkMode);
  const updateWindowDimensions = uiStore((state) => state.updateWindowDimensions);

  // ✅ Separate loading state for initialization
  const [isAppInitialized, setIsAppInitialized] = useState(false);
  const initializeRef = useRef(false);

  /**
   * ✅ Memoized initialization function
   */
  const initializeApp = useCallback(async () => {
    try {
      if (ENABLE_DEBUG) {
        console.debug('[App] Initializing application...', {
          authMode: AUTH_MODE,
          environment: import.meta.env.MODE,
        });
      }

      // Set theme based on preference or system
      const savedTheme = localStorage.getItem('theme');
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

      if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        setDarkMode(true);
        localStorage.setItem('theme', 'dark');
        document.documentElement.classList.add('dark');
      } else {
        setDarkMode(false);
        localStorage.setItem('theme', 'light');
        document.documentElement.classList.remove('dark');
      }

      // Initial window dimensions
      updateWindowDimensions(window.innerWidth, window.innerHeight);

      if (ENABLE_DEBUG) {
        console.debug('[App] Initialization complete');
      }
    } catch (error) {
      console.error('[App] Initialization error:', error);
    } finally {
      setIsAppInitialized(true);
    }
  }, [setDarkMode, updateWindowDimensions]);

  /**
   * ✅ Initialize app on mount (runs once)
   */
  useEffect(() => {
    // Prevent double initialization in StrictMode
    if (initializeRef.current) {
      if (ENABLE_DEBUG) {
        console.debug('[App] Already initialized, skipping...');
      }
      return;
    }

    initializeRef.current = true;
    initializeApp();
  }, [initializeApp]);

  /**
   * ✅ Setup window resize listener with proper cleanup
   */
  useEffect(() => {
    const handleResize = () => {
      updateWindowDimensions(window.innerWidth, window.innerHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [updateWindowDimensions]);

  /**
   * ✅ Setup online/offline listeners with proper cleanup
   */
  useEffect(() => {
    const handleOnline = () => {
      if (ENABLE_DEBUG) {
        console.log('[App] Application is online');
      }
    };

    const handleOffline = () => {
      if (ENABLE_DEBUG) {
        console.log('[App] Application is offline');
      }
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  /**
   * ✅ Apply dark mode to document
   */
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // ========================================================================
  // Render - Loading State
  // ========================================================================

  if (!isAppInitialized) {
    return <AppLoading message="Loading application..." />;
  }

  // ========================================================================
  // Render - Main Content
  // ========================================================================

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <div className="min-h-screen flex flex-col bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-50 transition-colors duration-300">
          {/* Header - Always show when authenticated */}
          {isAuthenticated && <Header />}

          {/* Main Content */}
          <main className="flex-1 w-full">
            <AppRoutes isAuthenticated={isAuthenticated} />
          </main>

          {/* Footer - Show when not authenticated */}
          {!isAuthenticated && <Footer />}

          {/* Toast/Notification Container */}
          <Toaster
            position="top-right"
            richColors
            theme={isDarkMode ? 'dark' : 'light'}
            expand={true}
            closeButton
            duration={4000}
          />

          {/* Debug Indicator - Testing Mode Only */}
          {AUTH_MODE === 'testing' && (
            <div
              className="fixed bottom-4 right-4 bg-yellow-100 dark:bg-yellow-900/30 border border-yellow-300 dark:border-yellow-700 text-yellow-800 dark:text-yellow-200 px-4 py-2 rounded-lg text-xs font-medium z-50 shadow-lg"
              title="Authentication is in testing mode"
            >
              🧪 Auth Testing Mode
            </div>
          )}
        </div>
      </QueryClientProvider>
    </ErrorBoundary>
  );
});

App.displayName = 'App';

// ============================================================================
// Root App Wrapper with Router
// ============================================================================

/**
 * ✅ Root component - Router wrapper
 */
const RootApp = memo(() => {
  return (
    <Router>
      <App />
    </Router>
  );
});

RootApp.displayName = 'RootApp';

export default RootApp;
