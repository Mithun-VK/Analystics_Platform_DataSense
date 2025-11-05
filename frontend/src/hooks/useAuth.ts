// src/hooks/useAuth.ts

/**
 * useAuth - Authentication Hook
 * ✅ PRODUCTION READY: Proper hook order and cleanup
 * ✅ FIXED: Hook order violations and cleanup function errors
 */

import { useCallback, useEffect, useState, useMemo, useRef } from 'react';
import { authStore } from '@/store/authStore';
import authService from '@/services/authService';
import userService from '@/services/userService';
import type { User } from '@/types/auth.types';

// ============================================================================
// Configuration
// ============================================================================

const AUTH_MODE = (import.meta.env.VITE_AUTH_MODE || 'production') as 'testing' | 'production';
const DEBUG_AUTH = import.meta.env.DEV;

// ============================================================================
// Mock User for Testing
// ============================================================================

const MOCK_TEST_USER: User = {
  id: 'test-user-123',
  email: import.meta.env.VITE_MOCK_USER_EMAIL || 'test@datasense.app',
  firstName: 'Test',
  lastName: 'User',
  fullName: import.meta.env.VITE_MOCK_USER_NAME || 'Test User',
  avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=TestUser',
  bio: 'Testing user for development',
  permissions: ['read', 'write', 'delete', 'admin'],
  roles: ['admin', 'user'],
  twoFactorEnabled: false,
  emailVerified: true,
  passwordChangeRequired: false,
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  isActive: false,
  accountLocked: false,
  loginAttempts: 0
};

const MOCK_TEST_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test-token';
const MOCK_TEST_REFRESH_TOKEN = 'refresh-token-test-12345';

// ============================================================================
// Type Definitions
// ============================================================================

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterCredentials {
  fullName: string;
  email: string;
  password: string;
}

export interface AuthUser extends User {
  fullName?: string;
}

export interface AuthState {
  user: AuthUser | null;
  token: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  isCheckingAuth: boolean;
  login: (credentials: LoginCredentials) => Promise<any>;
  logout: () => Promise<boolean>;
  register: (credentials: RegisterCredentials) => Promise<any>;
  refreshAuthToken: () => Promise<any>;
  checkAuth: () => Promise<boolean>;
  updateProfile: (updates: Partial<AuthUser>) => Promise<AuthUser>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<boolean>;
  requestPasswordReset: (email: string) => Promise<boolean>;
  resetPassword: (token: string, newPassword: string) => Promise<boolean>;
  verifyEmail: (token: string) => Promise<boolean>;
  enableTwoFactorAuth: () => Promise<any>;
  disableTwoFactorAuth: (password: string) => Promise<boolean>;
  verifyTwoFactorCode: (code: string) => Promise<any>;
  clearError: () => void;
  isAdmin: boolean;
  isModerator: boolean;
  isVerified: boolean;
}

// ============================================================================
// Hook Implementation
// ============================================================================

/**
 * ✅ useAuth Hook - Authentication State Management
 * Proper hook order to avoid violations
 */
export const useAuth = (): AuthState => {
  // ========================================================================
  // ✅ CRITICAL: All useState calls MUST come first
  // ========================================================================

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isCheckingAuth, setIsCheckingAuth] = useState(
    AUTH_MODE === 'testing' ? false : true
  );

  // ========================================================================
  // ✅ Store Selectors - After useState
  // ========================================================================

  const storeUser = authStore((state) => state.user as AuthUser | null);
  const storeToken = authStore((state) => state.token);
  const storeRefreshToken = authStore((state) => state.refreshToken);
  const storeIsAuthenticated = authStore((state) => state.isAuthenticated);
  const setUser = authStore((state) => state.setUser);
  const setToken = authStore((state) => state.setToken);
  const setRefreshToken = authStore((state) => state.setRefreshToken);
  const setIsAuthenticated = authStore((state) => state.setIsAuthenticated);
  const clearAuth = authStore((state) => state.clearAuth);

  // ========================================================================
  // ✅ Refs - After hooks
  // ========================================================================

  const initializeRef = useRef(false);

  // ========================================================================
  // ✅ Callbacks - Defined early, used in effects
  // ========================================================================

  const normalizeUser = useCallback((user: User): AuthUser => {
    return {
      ...user,
      fullName:
        user.fullName ||
        `${user.firstName || ''} ${user.lastName || ''}`.trim(),
    };
  }, []);

  const decodeJWT = useCallback((token: string) => {
    try {
      const base64Url = token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(
        atob(base64)
          .split('')
          .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
          .join('')
      );
      return JSON.parse(jsonPayload);
    } catch (error) {
      if (DEBUG_AUTH) console.error('[Auth] Failed to decode JWT:', error);
      return null;
    }
  }, []);

  const logAuthEvent = useCallback(
    (event: string, userId?: string, metadata?: string) => {
      if (DEBUG_AUTH) {
        console.log('[Auth Event]', {
          event,
          userId,
          timestamp: new Date().toISOString(),
          authMode: AUTH_MODE,
          metadata,
        });
      }
    },
    []
  );

  // ========================================================================
  // ✅ Login Handler
  // ========================================================================

  const login = useCallback(
    async (credentials: LoginCredentials) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Login attempt:', credentials.email);

        if (AUTH_MODE === 'testing') {
          if (DEBUG_AUTH) console.log('[Auth] TESTING MODE: Mock login');

          const normalizedUser = normalizeUser(MOCK_TEST_USER);
          setUser(normalizedUser);
          setToken(MOCK_TEST_TOKEN);
          setRefreshToken(MOCK_TEST_REFRESH_TOKEN);
          setIsAuthenticated(true);
          logAuthEvent('login_success', normalizedUser.id);

          return {
            user: normalizedUser,
            accessToken: MOCK_TEST_TOKEN,
            refreshToken: MOCK_TEST_REFRESH_TOKEN,
          };
        }

        // Production mode - real API call
        const response = await authService.login(
          credentials.email,
          credentials.password,
          credentials.rememberMe
        );

        localStorage.setItem('authToken', response.accessToken);
        localStorage.setItem('refreshToken', response.refreshToken);

        if (credentials.rememberMe) {
          localStorage.setItem('rememberMe', 'true');
        }

        const normalizedUser = normalizeUser(response.user);
        setUser(normalizedUser);
        setToken(response.accessToken);
        setRefreshToken(response.refreshToken);
        setIsAuthenticated(true);

        logAuthEvent('login_success', normalizedUser.id);
        return response;
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.message || err.message || 'Login failed';
        setError(errorMessage);
        logAuthEvent('login_failed', undefined, errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [setUser, setToken, setRefreshToken, setIsAuthenticated, normalizeUser, logAuthEvent]
  );

  // ========================================================================
  // ✅ Logout Handler
  // ========================================================================

  const logout = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      if (DEBUG_AUTH) console.log('[Auth] Logout attempt');

      if (AUTH_MODE !== 'testing') {
        await authService.logout();
      }

      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
      localStorage.removeItem('rememberMe');

      logAuthEvent('logout_success', storeUser?.id);

      clearAuth();
      return true;
    } catch (err: any) {
      const errorMessage = err.message || 'Logout failed';
      setError(errorMessage);
      logAuthEvent('logout_failed', storeUser?.id, errorMessage);

      clearAuth();
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');

      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [storeUser?.id, clearAuth, logAuthEvent]);

  // ========================================================================
  // ✅ Register Handler
  // ========================================================================

  const register = useCallback(
    async (credentials: RegisterCredentials) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Register attempt:', credentials.email);

        if (AUTH_MODE === 'testing') {
          if (DEBUG_AUTH) console.log('[Auth] TESTING MODE: Mock registration');

          const testUser: AuthUser = {
            ...MOCK_TEST_USER,
            email: credentials.email,
            fullName: credentials.fullName,
          };

          const normalizedUser = normalizeUser(testUser);
          setUser(normalizedUser);
          setToken(MOCK_TEST_TOKEN);
          setRefreshToken(MOCK_TEST_REFRESH_TOKEN);
          setIsAuthenticated(true);
          logAuthEvent('registration_success', normalizedUser.id);

          return {
            user: normalizedUser,
            accessToken: MOCK_TEST_TOKEN,
            refreshToken: MOCK_TEST_REFRESH_TOKEN,
          };
        }

        const response = await authService.register(credentials);

        if (response.accessToken && response.refreshToken) {
          localStorage.setItem('authToken', response.accessToken);
          localStorage.setItem('refreshToken', response.refreshToken);

          const normalizedUser = normalizeUser(response.user);

          setUser(normalizedUser);
          setToken(response.accessToken);
          setRefreshToken(response.refreshToken);
          setIsAuthenticated(true);

          logAuthEvent('registration_success', normalizedUser.id);
        }

        return response;
      } catch (err: any) {
        const errorMessage =
          err.response?.data?.message || err.message || 'Registration failed';
        setError(errorMessage);
        logAuthEvent('registration_failed', undefined, errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [setUser, setToken, setRefreshToken, setIsAuthenticated, normalizeUser, logAuthEvent]
  );

  // ========================================================================
  // ✅ Token Refresh Handler
  // ========================================================================

  const refreshAuthToken = useCallback(async () => {
    if (!storeRefreshToken) {
      return;
    }

    try {
      if (DEBUG_AUTH) console.debug('[Auth] Refreshing token...');

      if (AUTH_MODE === 'testing') {
        if (DEBUG_AUTH) console.log('[Auth] TESTING MODE: Mock token refresh');
        return {
          accessToken: MOCK_TEST_TOKEN,
          refreshToken: MOCK_TEST_REFRESH_TOKEN,
        };
      }

      const response = await authService.refreshToken(storeRefreshToken);

      localStorage.setItem('authToken', response.accessToken);
      localStorage.setItem('refreshToken', response.refreshToken);

      setToken(response.accessToken);
      setRefreshToken(response.refreshToken);

      return response;
    } catch (err: any) {
      clearAuth();
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
      setError('Session expired. Please login again.');
      throw err;
    }
  }, [storeRefreshToken, setToken, setRefreshToken, clearAuth]);

  // ========================================================================
  // ✅ Auth Check Handler
  // ========================================================================

  const checkAuth = useCallback(async (): Promise<boolean> => {
    if (!storeToken) {
      return false;
    }

    try {
      if (DEBUG_AUTH) console.debug('[Auth] Checking authentication...');

      if (AUTH_MODE === 'testing') {
        if (DEBUG_AUTH) console.log('[Auth] TESTING MODE: Auth check passed');
        return true;
      }

      const userData = await userService.getCurrentUser();
      const normalizedUser = normalizeUser(userData);
      setUser(normalizedUser);
      return true;
    } catch (error) {
      clearAuth();
      localStorage.removeItem('authToken');
      localStorage.removeItem('refreshToken');
      return false;
    }
  }, [storeToken, setUser, clearAuth, normalizeUser]);

  // ========================================================================
  // ✅ Additional Handlers
  // ========================================================================

  const updateProfile = useCallback(
    async (updates: Partial<AuthUser>) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Updating profile...');

        if (AUTH_MODE === 'testing') {
          const updatedUser: AuthUser = {
            ...storeUser,
            ...updates,
          } as AuthUser;

          setUser(updatedUser);
          logAuthEvent('profile_updated', updatedUser.id);
          return updatedUser;
        }

        const [firstName = '', ...lastNameParts] = (updates.fullName || '').split(' ');
        const lastName = lastNameParts.join(' ');

        const updatedUser = await userService.updateProfile({
          ...updates,
          firstName,
          lastName,
        });
        const normalizedUser = normalizeUser(updatedUser);
        setUser(normalizedUser);
        logAuthEvent('profile_updated', normalizedUser.id);
        return normalizedUser;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || err.message || 'Failed to update profile';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [storeUser, setUser, normalizeUser, logAuthEvent]
  );

  const changePassword = useCallback(
    async (currentPassword: string, newPassword: string) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Changing password...');

        if (AUTH_MODE !== 'testing') {
          await authService.changePassword(currentPassword, newPassword);
        }

        logAuthEvent('password_changed', storeUser?.id);
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || err.message || 'Failed to change password';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [storeUser?.id, logAuthEvent]
  );

  const requestPasswordReset = useCallback(async (email: string) => {
    setIsLoading(true);
    setError(null);

    try {
      if (DEBUG_AUTH) console.log('[Auth] Requesting password reset...');

      if (AUTH_MODE !== 'testing') {
        await authService.requestPasswordReset(email);
      }

      logAuthEvent('password_reset_requested', undefined, email);
      return true;
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || err.message || 'Failed to request password reset';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [logAuthEvent]);

  const resetPassword = useCallback(
    async (token: string, newPassword: string) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Resetting password...');

        if (AUTH_MODE !== 'testing') {
          await authService.confirmPasswordReset(token, newPassword);
        }

        logAuthEvent('password_reset_completed');
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || err.message || 'Failed to reset password';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [logAuthEvent]
  );

  const verifyEmail = useCallback(
    async (token: string) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Verifying email...');

        if (AUTH_MODE !== 'testing') {
          await authService.confirmEmailVerification(token);
        }

        if (storeUser) {
          const updatedUser: AuthUser = {
            ...storeUser,
            emailVerified: true,
          };
          setUser(updatedUser);
        }
        logAuthEvent('email_verified', storeUser?.id);
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || err.message || 'Failed to verify email';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [storeUser, setUser, logAuthEvent]
  );

  const enableTwoFactorAuth = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      if (DEBUG_AUTH) console.log('[Auth] Enabling 2FA...');

      if (AUTH_MODE === 'testing') {
        if (storeUser) {
          const updatedUser: AuthUser = {
            ...storeUser,
            twoFactorEnabled: true,
          };
          setUser(updatedUser);
        }
        logAuthEvent('2fa_enabled', storeUser?.id);

        return {
          secret: 'JBSWY3DPEBLW64TMMQ2HY2LQMFRGG33JFZQ====',
          qrCode: 'data:image/png;base64,mock-qr-code',
        };
      }

      const response = await authService.setup2FA();
      if (storeUser) {
        const updatedUser: AuthUser = {
          ...storeUser,
          twoFactorEnabled: true,
        };
        setUser(updatedUser);
      }
      logAuthEvent('2fa_enabled', storeUser?.id);
      return response;
    } catch (err: any) {
      const errorMessage = err.response?.data?.message || err.message || 'Failed to enable 2FA';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [storeUser, setUser, logAuthEvent]);

  const disableTwoFactorAuth = useCallback(
    async (password: string) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Disabling 2FA...');

        if (AUTH_MODE !== 'testing') {
          await authService.disable2FA(password);
        }

        if (storeUser) {
          const updatedUser: AuthUser = {
            ...storeUser,
            twoFactorEnabled: false,
          };
          setUser(updatedUser);
        }
        logAuthEvent('2fa_disabled', storeUser?.id);
        return true;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || err.message || 'Failed to disable 2FA';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [storeUser, setUser, logAuthEvent]
  );

  const verifyTwoFactorCode = useCallback(
    async (code: string) => {
      setIsLoading(true);
      setError(null);

      try {
        if (DEBUG_AUTH) console.log('[Auth] Verifying 2FA code...');

        if (AUTH_MODE === 'testing') {
          logAuthEvent('2fa_verified', storeUser?.id);
          return { backupCodes: ['XXXX-XXXX', 'YYYY-YYYY'] };
        }

        const response = await authService.verify2FA(code);
        logAuthEvent('2fa_verified', storeUser?.id);
        return response;
      } catch (err: any) {
        const errorMessage = err.response?.data?.message || err.message || 'Invalid 2FA code';
        setError(errorMessage);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [storeUser?.id, logAuthEvent]
  );

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  // ========================================================================
  // ✅ CRITICAL: Effects MUST come after all hooks
  // ========================================================================

  // Initialize authentication on mount
  useEffect(() => {
    if (initializeRef.current) {
      if (DEBUG_AUTH) console.debug('[Auth] Already initialized, skipping...');
      return;
    }

    initializeRef.current = true;

    const initializeAuth = async () => {
      try {
        if (DEBUG_AUTH) console.debug('[Auth] Starting initialization...', { AUTH_MODE });

        // Testing mode: instant setup
        if (AUTH_MODE === 'testing') {
          if (DEBUG_AUTH) console.log('[Auth] TESTING MODE: Using mock user');

          const normalizedUser = normalizeUser(MOCK_TEST_USER);
          setUser(normalizedUser);
          setToken(MOCK_TEST_TOKEN);
          setRefreshToken(MOCK_TEST_REFRESH_TOKEN);
          setIsAuthenticated(true);
          setIsCheckingAuth(false);

          logAuthEvent('test_mode_initialized', normalizedUser.id);
          return;
        }

        // Production mode: validate tokens
        const token = localStorage.getItem('authToken');
        const refreshToken = localStorage.getItem('refreshToken');

        if (!token || !refreshToken) {
          if (DEBUG_AUTH) console.debug('[Auth] No tokens found');
          setIsCheckingAuth(false);
          return;
        }

        try {
          if (DEBUG_AUTH) console.debug('[Auth] Fetching current user...');
          const userData = await userService.getCurrentUser();

          const normalizedUser = normalizeUser(userData);
          setUser(normalizedUser);
          setToken(token);
          setRefreshToken(refreshToken);
          setIsAuthenticated(true);

          if (DEBUG_AUTH) console.debug('[Auth] User authenticated successfully');
        } catch (fetchError) {
          if (DEBUG_AUTH) console.debug('[Auth] User fetch failed, attempting token refresh...');

          try {
            const response = await authService.refreshToken(refreshToken);

            localStorage.setItem('authToken', response.accessToken);
            localStorage.setItem('refreshToken', response.refreshToken);

            setToken(response.accessToken);
            setRefreshToken(response.refreshToken);
            setIsAuthenticated(true);

            if (DEBUG_AUTH) console.debug('[Auth] Token refreshed successfully');
          } catch (refreshError) {
            if (DEBUG_AUTH) console.debug('[Auth] Token refresh failed');

            clearAuth();
            localStorage.removeItem('authToken');
            localStorage.removeItem('refreshToken');
          }
        }
      } catch (error) {
        console.error('[Auth] Initialization error:', error);
        clearAuth();
      } finally {
        setIsCheckingAuth(false);
      }
    };

    initializeAuth();
  }, []); // Empty dependency array - run only once

  // Token refresh effect
  useEffect(() => {
    if (AUTH_MODE === 'testing') {
      return; // No token refresh in testing mode
    }

    if (!storeToken || !storeRefreshToken) {
      return;
    }

    const decoded = decodeJWT(storeToken);
    if (!decoded?.exp) {
      return;
    }

    const expirationTime = decoded.exp * 1000;
    const currentTime = Date.now();
    const timeUntilExpiry = expirationTime - currentTime;
    const refreshBufferTime = 5 * 60 * 1000; // 5 minutes
    const refreshTime = timeUntilExpiry - refreshBufferTime;

    if (refreshTime <= 0) {
      if (DEBUG_AUTH) console.debug('[Auth] Token expired, refreshing immediately');
      refreshAuthToken();
      return;
    }

    // ✅ FIXED: Proper cleanup function
    const timeoutId = setTimeout(() => {
      if (DEBUG_AUTH) console.debug('[Auth] Token refresh scheduled');
      refreshAuthToken();
    }, refreshTime);

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [storeToken, storeRefreshToken, decodeJWT, refreshAuthToken, AUTH_MODE]);

  // ========================================================================
  // ✅ Computed Values (Memoized)
  // ========================================================================

  const computedValues = useMemo(
    () => ({
      isAdmin: storeUser?.roles?.includes('admin') || false,
      isModerator: storeUser?.roles?.includes('moderator') || false,
      isVerified: storeUser?.emailVerified || false,
    }),
    [storeUser]
  );

  // ========================================================================
  // Return State Object
  // ========================================================================

  return useMemo(
    () => ({
      user: storeUser,
      token: storeToken,
      refreshToken: storeRefreshToken,
      isAuthenticated: storeIsAuthenticated,
      isLoading,
      error,
      isCheckingAuth,
      login,
      logout,
      register,
      refreshAuthToken,
      checkAuth,
      updateProfile,
      changePassword,
      requestPasswordReset,
      resetPassword,
      verifyEmail,
      enableTwoFactorAuth,
      disableTwoFactorAuth,
      verifyTwoFactorCode,
      clearError,
      ...computedValues,
    }),
    [
      storeUser,
      storeToken,
      storeRefreshToken,
      storeIsAuthenticated,
      isLoading,
      error,
      isCheckingAuth,
      login,
      logout,
      register,
      refreshAuthToken,
      checkAuth,
      updateProfile,
      changePassword,
      requestPasswordReset,
      resetPassword,
      verifyEmail,
      enableTwoFactorAuth,
      disableTwoFactorAuth,
      verifyTwoFactorCode,
      clearError,
      computedValues,
    ]
  );
};

export default useAuth;
