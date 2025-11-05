// src/components/auth/ProtectedRoute.tsx

/**
 * Protected Route Components
 * ✅ TESTING MODE: All routes unprotected for development testing
 * Role-based access control, subscription management, and email verification
 * Fully typed with auth.types.ts
 */

import { useEffect, useState } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Loader2, AlertCircle, Lock } from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';

// ============================================================================
// Feature Flags - TESTING MODE
// ============================================================================

/**
 * ✅ TESTING MODE: Set to TRUE to disable all protection
 * Change to FALSE in production
 */
const TESTING_MODE_DISABLE_AUTH = true;

/**
 * ✅ Log all route access for debugging
 */
const DEBUG_ROUTE_ACCESS = true;

// ============================================================================
// Type Definitions
// ============================================================================

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: string | string[];
  redirectTo?: string;
  fallback?: React.ReactNode;
  requireEmailVerification?: boolean;
  bypassProtection?: boolean;
}

interface AdminRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  bypassProtection?: boolean;
}

interface ManagerRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  bypassProtection?: boolean;
}

interface VerifiedUserRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  bypassProtection?: boolean;
}

interface GuestRouteProps {
  children: React.ReactNode;
  redirectTo?: string;
  bypassProtection?: boolean;
}

interface ConditionalRouteProps {
  authenticatedComponent: React.ReactNode;
  unauthenticatedComponent: React.ReactNode;
  loadingComponent?: React.ReactNode;
}

interface SubscriptionRouteProps {
  children: React.ReactNode;
  requiredPlan?: 'free' | 'pro' | 'enterprise';
  redirectTo?: string;
  bypassProtection?: boolean;
}

interface LocationState {
  from?: string;
  returnUrl?: string;
  reason?: string;
  requiredPlan?: string;
  currentPlan?: string;
}

// ============================================================================
// Main ProtectedRoute Component
// ============================================================================

/**
 * ✅ TESTING MODE: ProtectedRoute - NOW ALLOWS ALL ACCESS
 * Restricts access to authenticated users
 * Supports role-based access control and custom loading states
 * @param children - Child components to render if authenticated
 * @param requiredRole - Optional role(s) required to access the route
 * @param redirectTo - Custom redirect path (defaults to /login)
 * @param fallback - Custom loading component
 * @param requireEmailVerification - Whether email must be verified
 * @param bypassProtection - Override testing mode for specific routes
 */
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredRole,
  redirectTo = '/login',
  fallback,
  requireEmailVerification = false,
  bypassProtection = false,
}) => {
  const { isAuthenticated, user, isLoading, isCheckingAuth } = useAuth();
  const location = useLocation();
  const [isChecking, setIsChecking] = useState(true);

  // ✅ TESTING: Log route access
  useEffect(() => {
    if (DEBUG_ROUTE_ACCESS) {
      console.log(
        `[ProtectedRoute] Access attempt to ${location.pathname}`,
        {
          testingMode: TESTING_MODE_DISABLE_AUTH,
          bypassProtection,
          isAuthenticated,
          user: user?.email,
          requiredRole,
        }
      );
    }
  }, [location.pathname, isAuthenticated, user, requiredRole]);

  // Check authentication on mount
  useEffect(() => {
    const verifyAuth = async () => {
      try {
        setIsChecking(true);
        // Allow time for auth state to hydrate
        await new Promise((resolve) => setTimeout(resolve, 100));
      } catch (error) {
        console.error('Authentication check failed:', error);
      } finally {
        setIsChecking(false);
      }
    };

    verifyAuth();
  }, []);

  // ✅ TESTING MODE: If disabled, allow all access
  if (TESTING_MODE_DISABLE_AUTH && !bypassProtection) {
    if (DEBUG_ROUTE_ACCESS) {
      console.log(
        `[ProtectedRoute] TESTING MODE: Allowing access to ${location.pathname}`
      );
    }
    return <>{children}</>;
  }

  // Show loading state while checking authentication
  if (isLoading || isChecking || isCheckingAuth) {
    return fallback ? (
      <>{fallback}</>
    ) : (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-blue-600 dark:text-blue-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-600 dark:text-gray-300 font-medium">
            Verifying authentication...
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            Please wait a moment
          </p>
        </div>
      </div>
    );
  }

  // ✅ TESTING MODE: If disabled, show content with warning
  if (TESTING_MODE_DISABLE_AUTH) {
    return (
      <div>
        {/* ✅ TESTING: Visual indicator */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 p-3 mb-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Testing Mode: All routes are unprotected
          </p>
        </div>
        {children}
      </div>
    );
  }

  // Production mode: Redirect to login if not authenticated
  if (!isAuthenticated || !user) {
    console.log('[ProtectedRoute] User not authenticated, redirecting to login');
    return (
      <Navigate
        to={redirectTo}
        state={{
          from: location.pathname,
          returnUrl: location.pathname + location.search,
        } as LocationState}
        replace
      />
    );
  }

  // Check email verification if required
  if (requireEmailVerification && !user.emailVerified) {
    console.log('[ProtectedRoute] Email not verified, redirecting to verify-email');
    return (
      <Navigate
        to="/verify-email"
        state={{ from: location.pathname } as LocationState}
        replace
      />
    );
  }

  // Check role-based access if required
  if (requiredRole) {
    // ✅ FIXED: Use 'roles' (plural) instead of 'role'
    const hasRequiredRole = checkUserRole(user.roles, requiredRole);

    if (!hasRequiredRole) {
      console.log('[ProtectedRoute] User lacks required role, redirecting to unauthorized');
      return (
        <Navigate
          to="/unauthorized"
          state={{ from: location.pathname } as LocationState}
          replace
        />
      );
    }
  }

  // User is authenticated and has required role - render children
  return <>{children}</>;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * ✅ FIXED: Check if user has required role
 * Handles both single role and array of roles
 * @param userRoles - User's roles array
 * @param requiredRole - Required role(s) for access
 * @returns boolean indicating if user has required role
 */
const checkUserRole = (
  userRoles: string | string[] | undefined,
  requiredRole: string | string[]
): boolean => {
  if (!userRoles) return false;

  // Normalize userRoles to array
  const userRolesArray = Array.isArray(userRoles) ? userRoles : [userRoles];

  // Normalize requiredRole to array
  const requiredRoles = Array.isArray(requiredRole) ? requiredRole : [requiredRole];

  // Check if user has any of the required roles
  return userRolesArray.some((role) => requiredRoles.includes(role));
};

/**
 * Check subscription plan hierarchy
 * @param userPlan - User's subscription plan
 * @param requiredPlan - Required subscription plan
 * @returns boolean indicating if user's plan meets requirement
 */
const checkSubscriptionPlan = (
  userPlan: string | undefined,
  requiredPlan: 'free' | 'pro' | 'enterprise'
): boolean => {
  const planHierarchy: Record<string, number> = {
    free: 1,
    pro: 2,
    enterprise: 3,
  };

  const userPlanLevel = planHierarchy[userPlan || 'free'] || 1;
  const requiredPlanLevel = planHierarchy[requiredPlan];

  return userPlanLevel >= requiredPlanLevel;
};

// ============================================================================
// Specialized Route Components - TESTING MODE
// ============================================================================

/**
 * ✅ TESTING: AdminRoute - NOW ALLOWS ALL ACCESS
 */
export const AdminRoute: React.FC<AdminRouteProps> = ({
  children,
  redirectTo = '/login',
  bypassProtection = false,
}) => {
  if (DEBUG_ROUTE_ACCESS) {
    console.log('[AdminRoute] Access attempt');
  }

  if (TESTING_MODE_DISABLE_AUTH && !bypassProtection) {
    return (
      <div>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 p-3 mb-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Testing Mode: AdminRoute bypassed (normally requires admin role)
          </p>
        </div>
        {children}
      </div>
    );
  }

  return (
    <ProtectedRoute
      requiredRole="admin"
      redirectTo={redirectTo}
      requireEmailVerification={true}
    >
      {children}
    </ProtectedRoute>
  );
};

/**
 * ✅ TESTING: ManagerRoute - NOW ALLOWS ALL ACCESS
 */
export const ManagerRoute: React.FC<ManagerRouteProps> = ({
  children,
  redirectTo = '/login',
  bypassProtection = false,
}) => {
  if (DEBUG_ROUTE_ACCESS) {
    console.log('[ManagerRoute] Access attempt');
  }

  if (TESTING_MODE_DISABLE_AUTH && !bypassProtection) {
    return (
      <div>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 p-3 mb-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Testing Mode: ManagerRoute bypassed (normally requires manager/admin role)
          </p>
        </div>
        {children}
      </div>
    );
  }

  return (
    <ProtectedRoute
      requiredRole={['admin', 'manager']}
      redirectTo={redirectTo}
      requireEmailVerification={true}
    >
      {children}
    </ProtectedRoute>
  );
};

/**
 * ✅ TESTING: VerifiedUserRoute - NOW ALLOWS ALL ACCESS
 */
export const VerifiedUserRoute: React.FC<VerifiedUserRouteProps> = ({
  children,
  redirectTo = '/login',
  bypassProtection = false,
}) => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (DEBUG_ROUTE_ACCESS) {
    console.log('[VerifiedUserRoute] Access attempt');
  }

  if (TESTING_MODE_DISABLE_AUTH && !bypassProtection) {
    return (
      <div>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 p-3 mb-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Testing Mode: VerifiedUserRoute bypassed (normally requires email verification)
          </p>
        </div>
        {children}
      </div>
    );
  }

  if (isLoading) {
    return <RouteLoadingFallback />;
  }

  if (user && !user.emailVerified) {
    return (
      <Navigate
        to="/verify-email"
        state={{ from: location.pathname } as LocationState}
        replace
      />
    );
  }

  return (
    <ProtectedRoute redirectTo={redirectTo}>
      {children}
    </ProtectedRoute>
  );
};

/**
 * ✅ TESTING: GuestRoute - NOW ALLOWS ALL ACCESS
 * Useful for login/register pages
 */
export const GuestRoute: React.FC<GuestRouteProps> = ({
  children,
  redirectTo = '/dashboard',
  bypassProtection = false,
}) => {
  const { isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (DEBUG_ROUTE_ACCESS) {
    console.log('[GuestRoute] Access attempt', { isAuthenticated });
  }

  if (TESTING_MODE_DISABLE_AUTH && !bypassProtection) {
    return <>{children}</>;
  }

  if (isLoading) {
    return <RouteLoadingFallback />;
  }

  if (isAuthenticated) {
    const state = location.state as LocationState | undefined;
    const returnUrl = state?.returnUrl || redirectTo;
    return <Navigate to={returnUrl} replace />;
  }

  return <>{children}</>;
};

/**
 * ConditionalRoute - Renders different content based on authentication status
 * Useful for components that need to show different content for auth/unauth users
 */
export const ConditionalRoute: React.FC<ConditionalRouteProps> = ({
  authenticatedComponent,
  unauthenticatedComponent,
  loadingComponent,
}) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <>
        {loadingComponent || <RouteLoadingFallback />}
      </>
    );
  }

  return (
    <>
      {isAuthenticated ? authenticatedComponent : unauthenticatedComponent}
    </>
  );
};

/**
 * ✅ TESTING: SubscriptionRoute - NOW ALLOWS ALL ACCESS
 */
export const SubscriptionRoute: React.FC<SubscriptionRouteProps> = ({
  children,
  requiredPlan = 'free',
  redirectTo,
  bypassProtection = false,
}) => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (DEBUG_ROUTE_ACCESS) {
    console.log('[SubscriptionRoute] Access attempt', { requiredPlan });
  }

  if (TESTING_MODE_DISABLE_AUTH && !bypassProtection) {
    return (
      <div>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 p-3 mb-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Testing Mode: SubscriptionRoute bypassed (normally requires {requiredPlan} plan)
          </p>
        </div>
        {children}
      </div>
    );
  }

  if (isLoading) {
    return <RouteLoadingFallback />;
  }

  if (!user) {
    return (
      <Navigate
        to="/login"
        state={{ from: location.pathname } as LocationState}
        replace
      />
    );
  }

  // Check subscription status
  if (user.subscriptionStatus && user.subscriptionStatus !== 'active') {
    return (
      <Navigate
        to={redirectTo || '/upgrade'}
        state={{
          from: location.pathname,
          reason: 'inactive_subscription',
        } as LocationState}
        replace
      />
    );
  }

  // ✅ FIXED: Properly check subscription plan
  const userPlan = user.subscriptionPlan as string | undefined;
  const hasPlan = checkSubscriptionPlan(userPlan, requiredPlan);

  if (!hasPlan) {
    return (
      <Navigate
        to={redirectTo || '/upgrade'}
        state={{
          from: location.pathname,
          requiredPlan,
          currentPlan: userPlan,
        } as LocationState}
        replace
      />
    );
  }

  return (
    <ProtectedRoute>
      {children}
    </ProtectedRoute>
  );
};

/**
 * ✅ TESTING: ActiveAccountRoute - NOW ALLOWS ALL ACCESS
 */
export const ActiveAccountRoute: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (DEBUG_ROUTE_ACCESS) {
    console.log('[ActiveAccountRoute] Access attempt');
  }

  if (TESTING_MODE_DISABLE_AUTH) {
    return (
      <div>
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 p-3 mb-4 rounded">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            ⚠️ Testing Mode: ActiveAccountRoute bypassed (normally requires active account)
          </p>
        </div>
        {children}
      </div>
    );
  }

  if (isLoading) {
    return <RouteLoadingFallback />;
  }

  if (!user?.isActive) {
    return (
      <Navigate
        to="/account-suspended"
        state={{ from: location.pathname } as LocationState}
        replace
      />
    );
  }

  return (
    <ProtectedRoute>
      {children}
    </ProtectedRoute>
  );
};

// ============================================================================
// Loading Components
// ============================================================================

/**
 * Custom Loading Component for ProtectedRoute
 */
export const RouteLoadingFallback: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <div className="text-center space-y-6">
        {/* Animated Logo */}
        <div className="flex justify-center">
          <div className="relative">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-600 to-blue-700 rounded-2xl flex items-center justify-center shadow-xl animate-pulse dark:from-blue-500 dark:to-blue-600">
              <svg
                className="w-12 h-12 text-white"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
            <div className="absolute inset-0 bg-blue-600 rounded-2xl opacity-50 animate-ping dark:bg-blue-500"></div>
          </div>
        </div>

        {/* Loading Text */}
        <div className="space-y-2">
          <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-50">
            Loading your workspace
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Please wait while we prepare everything
          </p>
        </div>

        {/* Loading Spinner */}
        <div className="flex justify-center">
          <Loader2 className="w-8 h-8 text-blue-600 dark:text-blue-400 animate-spin" />
        </div>

        {/* Progress Dots */}
        <div className="flex justify-center space-x-2">
          <div className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"></div>
          <div
            className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"
            style={{ animationDelay: '0.1s' }}
          ></div>
          <div
            className="w-2 h-2 bg-blue-600 dark:bg-blue-400 rounded-full animate-bounce"
            style={{ animationDelay: '0.2s' }}
          ></div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Error & Exception Pages
// ============================================================================

/**
 * Unauthorized Access Component
 */
export const UnauthorizedPage: React.FC = () => {
  const { user } = useAuth();

  // ✅ FIXED: Use 'roles' instead of 'role'
  const userRoles = Array.isArray(user?.roles)
    ? user.roles.join(', ')
    : user?.roles || 'unknown';

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full text-center space-y-6">
        {/* Icon */}
        <div className="flex justify-center">
          <div className="w-20 h-20 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
            <Lock className="w-10 h-10 text-red-600 dark:text-red-400" />
          </div>
        </div>

        {/* Message */}
        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
            Access Denied
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            You don't have permission to access this page.
          </p>
          {userRoles && (
            <p className="text-sm text-gray-500 dark:text-gray-500">
              Your current role:
              <span className="font-medium block mt-1">{userRoles}</span>
            </p>
          )}
        </div>

        {/* Actions */}
        <div className="space-y-3">
          <button
            onClick={() => window.history.back()}
            className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-50 font-medium rounded-lg transition-colors"
          >
            Go Back
          </button>
          <a
            href="/dashboard"
            className="block w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 text-white font-medium rounded-lg transition-colors text-center"
          >
            Go to Dashboard
          </a>
        </div>

        {/* Help Link */}
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Need help?{' '}
          <a
            href="/support"
            className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
          >
            Contact Support
          </a>
        </p>
      </div>
    </div>
  );
};

/**
 * Account Suspended Component
 */
export const AccountSuspendedPage: React.FC = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full text-center space-y-6">
        <div className="flex justify-center">
          <div className="w-20 h-20 bg-yellow-100 dark:bg-yellow-900/30 rounded-full flex items-center justify-center">
            <AlertCircle className="w-10 h-10 text-yellow-600 dark:text-yellow-400" />
          </div>
        </div>

        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
            Account Suspended
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            Your account has been suspended. Please contact support for more
            information.
          </p>
        </div>

        <div className="space-y-3">
          <a
            href="/support"
            className="block w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 text-white font-medium rounded-lg transition-colors text-center"
          >
            Contact Support
          </a>
          <button
            onClick={() => window.history.back()}
            className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-50 font-medium rounded-lg transition-colors"
          >
            Go Back
          </button>
        </div>
      </div>
    </div>
  );
};

/**
 * Upgrade Required Component
 */
export const UpgradeRequiredPage: React.FC<{
  currentPlan?: string;
  requiredPlan?: string;
}> = ({ currentPlan = 'free', requiredPlan = 'pro' }) => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full text-center space-y-6">
        <div className="flex justify-center">
          <div className="w-20 h-20 bg-blue-100 dark:bg-blue-900/30 rounded-full flex items-center justify-center">
            <svg
              className="w-10 h-10 text-blue-600 dark:text-blue-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
        </div>

        <div className="space-y-2">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-50">
            Upgrade Required
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            This feature requires a <span className="font-medium">{requiredPlan}</span>{' '}
            subscription. You're currently on the{' '}
            <span className="font-medium">{currentPlan}</span> plan.
          </p>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-lg p-4">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            Upgrade to unlock premium features and get the most out of your
            experience.
          </p>
        </div>

        <div className="space-y-3">
          <a
            href="/upgrade"
            className="block w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 text-white font-medium rounded-lg transition-colors text-center"
          >
            Upgrade Now
          </a>
          <button
            onClick={() => window.history.back()}
            className="w-full px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-50 font-medium rounded-lg transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProtectedRoute;
