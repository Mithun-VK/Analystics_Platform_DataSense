// src/services/authService.ts

/**
 * Authentication Service
 * Handles all authentication-related API calls with comprehensive error handling
 * ✅ CRITICAL FIXES: Proper error handling, token validation, timeout handling
 */

import { apiGet, apiPost, apiPatch } from './api';
import { authStore } from '@/store/authStore';
import type {
  User,
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  RegisterResponse,
  RefreshTokenResponse,
  PasswordResetRequest,
  PasswordResetResponse,
  EmailVerificationResponse,
  SocialLoginRequest,
  SocialLoginResponse,
} from '@/types/auth.types';

// ============================================================================
// Constants
// ============================================================================

const AUTH_ENDPOINTS = {
  LOGIN: '/auth/login',
  REGISTER: '/auth/register',
  LOGOUT: '/auth/logout',
  REFRESH: '/auth/refresh',
  PASSWORD_RESET_REQUEST: '/auth/password-reset-request',
  PASSWORD_RESET_CONFIRM: '/auth/password-reset-confirm',
  EMAIL_VERIFY_REQUEST: '/auth/email-verify-request',
  EMAIL_VERIFY_CONFIRM: '/auth/email-verify-confirm',
  SOCIAL_LOGIN: '/auth/social-login',
  SOCIAL_CONNECT: '/auth/social-connect',
  TWO_FACTOR_SETUP: '/auth/2fa-setup',
  TWO_FACTOR_VERIFY: '/auth/2fa-verify',
  TWO_FACTOR_DISABLE: '/auth/2fa-disable',
  PROFILE: '/auth/profile',
  CHANGE_PASSWORD: '/auth/change-password',
} as const;

const TOKEN_STORAGE_KEYS = {
  ACCESS_TOKEN: 'auth_access_token',
  REFRESH_TOKEN: 'auth_refresh_token',
  REFRESH_TOKEN_EXPIRY: 'auth_refresh_token_expiry',
} as const;

const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error. Please check your internet connection.',
  TIMEOUT_ERROR: 'Request timeout. Please try again.',
  INVALID_CREDENTIALS: 'Invalid email or password.',
  INVALID_TOKEN: 'Invalid or expired token.',
  UNAUTHORIZED: 'You are not authorized to perform this action.',
  FORBIDDEN: 'You do not have permission to access this resource.',
  NOT_FOUND: 'The requested resource was not found.',
  CONFLICT: 'This resource already exists.',
  UNPROCESSABLE_ENTITY: 'Invalid data provided.',
  INTERNAL_SERVER_ERROR: 'An error occurred on the server. Please try again later.',
  SERVICE_UNAVAILABLE: 'The service is currently unavailable. Please try again later.',
  UNKNOWN_ERROR: 'An unexpected error occurred. Please try again.',
} as const;

// ============================================================================
// Utility Functions - Error Handling
// ============================================================================

/**
 * ✅ FIXED: Extract error message from response
 */
const getErrorMessage = (error: any): string => {
  // ✅ FIXED: Check for API error response
  if (error.response?.data?.message) {
    return error.response.data.message;
  }

  // ✅ FIXED: Check for error details
  if (error.response?.data?.error) {
    return error.response.data.error;
  }

  // ✅ FIXED: Check for validation errors
  if (error.response?.data?.errors) {
    const errors = error.response.data.errors;
    if (Array.isArray(errors)) {
      return errors.map((e: any) => e.message || e).join(', ');
    }
    if (typeof errors === 'object') {
      return Object.values(errors).join(', ');
    }
  }

  // ✅ FIXED: Check for status code specific messages
  if (error.response?.status === 401) {
    return ERROR_MESSAGES.INVALID_CREDENTIALS;
  }

  if (error.response?.status === 403) {
    return ERROR_MESSAGES.FORBIDDEN;
  }

  if (error.response?.status === 404) {
    return ERROR_MESSAGES.NOT_FOUND;
  }

  if (error.response?.status === 409) {
    return ERROR_MESSAGES.CONFLICT;
  }

  if (error.response?.status === 422) {
    return ERROR_MESSAGES.UNPROCESSABLE_ENTITY;
  }

  if (error.response?.status === 500) {
    return ERROR_MESSAGES.INTERNAL_SERVER_ERROR;
  }

  if (error.response?.status === 503) {
    return ERROR_MESSAGES.SERVICE_UNAVAILABLE;
  }

  // ✅ FIXED: Check for network errors
  if (error.code === 'ECONNABORTED') {
    return ERROR_MESSAGES.TIMEOUT_ERROR;
  }

  if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
    return ERROR_MESSAGES.NETWORK_ERROR;
  }

  // ✅ FIXED: Check for axios error message
  if (error.message) {
    return error.message;
  }

  return ERROR_MESSAGES.UNKNOWN_ERROR;
};

/**
 * ✅ FIXED: Validate API response
 */
const validateResponse = <T>(response: any): T => {
  if (!response) {
    throw new Error('Empty response from server');
  }

  // ✅ FIXED: Check for error flag
  if (response.error) {
    throw new Error(response.message || 'Request failed');
  }

  return response;
};

/**
 * ✅ FIXED: Check if stored tokens are still valid
 */
const areTokensValid = (): boolean => {
  try {
    const expiryTime = localStorage.getItem(
      TOKEN_STORAGE_KEYS.REFRESH_TOKEN_EXPIRY
    );

    if (!expiryTime) {
      console.debug('[AuthService] No token expiry time found');
      return false;
    }

    const expiryDate = new Date(expiryTime);
    const isValid = expiryDate > new Date();

    if (!isValid) {
      console.debug('[AuthService] Tokens have expired');
    }

    return isValid;
  } catch (error) {
    console.error('[AuthService] Error checking token validity:', error);
    return false;
  }
};

/**
 * ✅ FIXED: Store authentication tokens with validation
 */
const storeTokens = (
  accessToken: string,
  refreshToken: string,
  expiresIn: number
): void => {
  try {
    // ✅ FIXED: Validate tokens before storing
    if (!accessToken || !refreshToken) {
      throw new Error('Invalid tokens provided');
    }

    if (!validateTokenFormat(accessToken) || !validateTokenFormat(refreshToken)) {
      throw new Error('Invalid token format');
    }

    // ✅ FIXED: Store in state
    authStore.getState().setToken(accessToken);
    authStore.getState().setRefreshToken(refreshToken);

    // ✅ FIXED: Store in localStorage with error handling
    localStorage.setItem(TOKEN_STORAGE_KEYS.REFRESH_TOKEN, refreshToken);

    // ✅ FIXED: Calculate and store expiry time
    const expiryDate = new Date();
    expiryDate.setSeconds(expiryDate.getSeconds() + (expiresIn || 604800)); // Default 7 days
    localStorage.setItem(
      TOKEN_STORAGE_KEYS.REFRESH_TOKEN_EXPIRY,
      expiryDate.toISOString()
    );

    console.debug('[AuthService] Tokens stored successfully');
  } catch (error) {
    console.error('[AuthService] Error storing tokens:', error);
    clearTokens();
    throw new Error('Failed to store authentication tokens');
  }
};

/**
 * ✅ FIXED: Clear all authentication tokens
 */
const clearTokens = (): void => {
  try {
    authStore.getState().setToken(null);
    authStore.getState().setRefreshToken(null);
    localStorage.removeItem(TOKEN_STORAGE_KEYS.ACCESS_TOKEN);
    localStorage.removeItem(TOKEN_STORAGE_KEYS.REFRESH_TOKEN);
    localStorage.removeItem(TOKEN_STORAGE_KEYS.REFRESH_TOKEN_EXPIRY);

    console.debug('[AuthService] Tokens cleared');
  } catch (error) {
    console.error('[AuthService] Error clearing tokens:', error);
  }
};

/**
 * ✅ FIXED: Retrieve stored refresh token
 */
const getStoredRefreshToken = (): string | null => {
  try {
    const token = localStorage.getItem(TOKEN_STORAGE_KEYS.REFRESH_TOKEN);

    if (token && validateTokenFormat(token)) {
      return token;
    }

    return null;
  } catch (error) {
    console.error('[AuthService] Error retrieving refresh token:', error);
    return null;
  }
};

/**
 * ✅ FIXED: Validate token format (JWT)
 */
export const validateTokenFormat = (token: string): boolean => {
  try {
    if (!token || typeof token !== 'string') {
      return false;
    }

    const tokenRegex = /^[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*$/;
    return tokenRegex.test(token);
  } catch (error) {
    console.error('[AuthService] Error validating token format:', error);
    return false;
  }
};

// ============================================================================
// Authentication Methods with Proper Error Handling
// ============================================================================

/**
 * ✅ FIXED: User login with comprehensive error handling
 */
export const authServiceLogin = async (
  email: string,
  password: string,
  rememberMe: boolean = false
): Promise<LoginResponse> => {
  try {
    // ✅ FIXED: Validate input
    if (!email || !password) {
      throw new Error('Email and password are required');
    }

    const payload: LoginRequest = {
      email,
      password,
      rememberMe,
    };

    console.debug('[AuthService] Logging in user:', email);

    const response = await apiPost<LoginResponse>(
      AUTH_ENDPOINTS.LOGIN,
      payload,
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    validateResponse<LoginResponse>(response);

    // ✅ FIXED: Validate tokens before storing
    if (!response.accessToken || !response.refreshToken) {
      throw new Error('Invalid tokens in response');
    }

    storeTokens(
      response.accessToken,
      response.refreshToken,
      response.expiresIn || 3600
    );

    authStore.getState().setUser(response.user);
    authStore.getState().setIsAuthenticated(true);

    console.debug('[AuthService] Login successful:', response.user.email);

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Login failed:', errorMessage);
    clearTokens();
    authStore.getState().setIsAuthenticated(false);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: User registration with comprehensive error handling
 */
export const authServiceRegister = async (credentials: {
  fullName: string;
  email: string;
  password: string;
}): Promise<RegisterResponse> => {
  try {
    // ✅ FIXED: Validate input
    if (!credentials.fullName || !credentials.email || !credentials.password) {
      throw new Error('All fields are required');
    }

    const [firstName, ...lastNameParts] = credentials.fullName.split(' ');
    const lastName = lastNameParts.join(' ') || '';

    const payload: RegisterRequest = {
      email: credentials.email,
      password: credentials.password,
      firstName,
      lastName,
      acceptTerms: true,
    };

    console.debug('[AuthService] Registering user:', credentials.email);

    const response = await apiPost<RegisterResponse>(
      AUTH_ENDPOINTS.REGISTER,
      payload,
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    validateResponse<RegisterResponse>(response);

    console.debug('[AuthService] Registration successful:', credentials.email);

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Registration failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: User logout with proper error recovery
 */
export const authServiceLogout = async (): Promise<void> => {
  try {
    const refreshToken = getStoredRefreshToken();

    console.debug('[AuthService] Logging out user');

    // ✅ FIXED: Try to notify server, but don't fail if unreachable
    if (refreshToken) {
      try {
        await apiPost(
          AUTH_ENDPOINTS.LOGOUT,
          { refreshToken },
          { silent: true, skipRetry: true, timeout: 3000 }
        );
      } catch (error) {
        console.warn('[AuthService] Server logout failed, clearing locally:', error);
      }
    }

    clearTokens();
    authStore.getState().setUser(null);
    authStore.getState().setIsAuthenticated(false);

    console.debug('[AuthService] Logout successful');
  } catch (error) {
    console.error('[AuthService] Logout error:', error);
    
    // ✅ FIXED: Always clear local state even if server fails
    clearTokens();
    authStore.getState().setUser(null);
    authStore.getState().setIsAuthenticated(false);
  }
};

/**
 * ✅ FIXED: Refresh access token with proper error handling
 */
export const authServiceRefreshToken = async (
  refreshToken: string
): Promise<RefreshTokenResponse> => {
  try {
    // ✅ FIXED: Validate refresh token
    if (!refreshToken || !validateTokenFormat(refreshToken)) {
      throw new Error('Invalid refresh token');
    }

    console.debug('[AuthService] Refreshing token');

    const response = await apiPost<RefreshTokenResponse>(
      AUTH_ENDPOINTS.REFRESH,
      { refreshToken },
      { skipRetry: true, silent: true, timeout: 5000 }
    );

    // ✅ FIXED: Validate response
    validateResponse<RefreshTokenResponse>(response);

    // ✅ FIXED: Validate tokens in response
    if (!response.accessToken) {
      throw new Error('No access token in refresh response');
    }

    storeTokens(
      response.accessToken,
      response.refreshToken || refreshToken,
      response.expiresIn || 3600
    );

    console.debug('[AuthService] Token refreshed successfully');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Token refresh failed:', errorMessage);
    
    // ✅ FIXED: Clear auth state on token refresh failure
    clearTokens();
    authStore.getState().setUser(null);
    authStore.getState().setIsAuthenticated(false);
    
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Request password reset with error handling
 */
export const authServiceRequestPasswordReset = async (
  email: string
): Promise<PasswordResetResponse> => {
  try {
    // ✅ FIXED: Validate email
    if (!email) {
      throw new Error('Email is required');
    }

    const payload: PasswordResetRequest = { email };

    console.debug('[AuthService] Requesting password reset for:', email);

    const response = await apiPost<PasswordResetResponse>(
      AUTH_ENDPOINTS.PASSWORD_RESET_REQUEST,
      payload,
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    validateResponse<PasswordResetResponse>(response);

    console.debug('[AuthService] Password reset email sent:', email);

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Password reset request failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Confirm password reset with error handling
 */
export const authServiceConfirmPasswordReset = async (
  token: string,
  newPassword: string
): Promise<PasswordResetResponse> => {
  try {
    // ✅ FIXED: Validate input
    if (!token || !newPassword) {
      throw new Error('Token and new password are required');
    }

    console.debug('[AuthService] Confirming password reset');

    const response = await apiPost<PasswordResetResponse>(
      AUTH_ENDPOINTS.PASSWORD_RESET_CONFIRM,
      { token, newPassword },
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    validateResponse<PasswordResetResponse>(response);

    console.debug('[AuthService] Password reset confirmed');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Password reset confirmation failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Request email verification with error handling
 */
export const authServiceRequestEmailVerification = async (): Promise<void> => {
  try {
    console.debug('[AuthService] Requesting email verification');

    await apiPost(
      AUTH_ENDPOINTS.EMAIL_VERIFY_REQUEST,
      {},
      { skipRetry: true }
    );

    console.debug('[AuthService] Email verification requested');
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Email verification request failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Confirm email verification with error handling
 */
export const authServiceConfirmEmailVerification = async (
  token: string
): Promise<EmailVerificationResponse> => {
  try {
    // ✅ FIXED: Validate token
    if (!token) {
      throw new Error('Verification token is required');
    }

    console.debug('[AuthService] Confirming email verification');

    const response = await apiPost<EmailVerificationResponse>(
      AUTH_ENDPOINTS.EMAIL_VERIFY_CONFIRM,
      { token },
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    validateResponse<EmailVerificationResponse>(response);

    const currentUser = authStore.getState().user;
    if (currentUser) {
      authStore.getState().setUser({
        ...currentUser,
        emailVerified: true,
      });
    }

    console.debug('[AuthService] Email verification confirmed');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Email verification confirmation failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Social login with error handling
 */
export const authServiceSocialLogin = async (
  provider: string,
  code: string
): Promise<SocialLoginResponse> => {
  try {
    // ✅ FIXED: Validate input
    if (!provider || !code) {
      throw new Error('Provider and code are required');
    }

    const payload: SocialLoginRequest = {
      provider: provider as any,
      code,
      redirectUri: window.location.origin + '/auth/callback',
    };

    console.debug('[AuthService] Social login with provider:', provider);

    const response = await apiPost<SocialLoginResponse>(
      AUTH_ENDPOINTS.SOCIAL_LOGIN,
      payload,
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    validateResponse<SocialLoginResponse>(response);

    // ✅ FIXED: Validate tokens
    if (!response.accessToken || !response.refreshToken) {
      throw new Error('Invalid tokens in social login response');
    }

    storeTokens(
      response.accessToken,
      response.refreshToken,
      response.expiresIn || 3600
    );

    authStore.getState().setUser(response.user);
    authStore.getState().setIsAuthenticated(true);

    console.debug('[AuthService] Social login successful:', provider, response.user.email);

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Social login failed:', errorMessage);
    clearTokens();
    authStore.getState().setIsAuthenticated(false);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Connect social account with error handling
 */
export const authServiceConnectSocialAccount = async (
  provider: string,
  code: string
): Promise<User> => {
  try {
    // ✅ FIXED: Validate input
    if (!provider || !code) {
      throw new Error('Provider and code are required');
    }

    const payload = {
      provider,
      code,
      redirectUri: window.location.origin + '/profile',
    };

    console.debug('[AuthService] Connecting social account:', provider);

    const response = await apiPost<User>(
      AUTH_ENDPOINTS.SOCIAL_CONNECT,
      payload
    );

    // ✅ FIXED: Validate response
    validateResponse<User>(response);

    authStore.getState().setUser(response);

    console.debug('[AuthService] Social account connected:', provider);

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Social account connection failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Setup 2FA with error handling
 */
export const authServiceSetup2FA = async (): Promise<{
  secret: string;
  qrCode: string;
}> => {
  try {
    console.debug('[AuthService] Setting up 2FA');

    const response = await apiPost<{ secret: string; qrCode: string }>(
      AUTH_ENDPOINTS.TWO_FACTOR_SETUP,
      {}
    );

    // ✅ FIXED: Validate response
    if (!response.secret || !response.qrCode) {
      throw new Error('Invalid 2FA setup response');
    }

    console.debug('[AuthService] 2FA setup initiated');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] 2FA setup failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Verify 2FA with error handling
 */
export const authServiceVerify2FA = async (code: string): Promise<{
  backupCodes: string[];
}> => {
  try {
    // ✅ FIXED: Validate code
    if (!code) {
      throw new Error('2FA code is required');
    }

    console.debug('[AuthService] Verifying 2FA code');

    const response = await apiPost<{ backupCodes: string[] }>(
      AUTH_ENDPOINTS.TWO_FACTOR_VERIFY,
      { code },
      { skipRetry: true }
    );

    // ✅ FIXED: Validate response
    if (!response.backupCodes || !Array.isArray(response.backupCodes)) {
      throw new Error('Invalid 2FA verification response');
    }

    const currentUser = authStore.getState().user;
    if (currentUser) {
      authStore.getState().setUser({
        ...currentUser,
        twoFactorEnabled: true,
      });
    }

    console.debug('[AuthService] 2FA verified and enabled');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] 2FA verification failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Disable 2FA with error handling
 */
export const authServiceDisable2FA = async (password: string): Promise<void> => {
  try {
    // ✅ FIXED: Validate password
    if (!password) {
      throw new Error('Password is required');
    }

    console.debug('[AuthService] Disabling 2FA');

    await apiPost(
      AUTH_ENDPOINTS.TWO_FACTOR_DISABLE,
      { password },
      { skipRetry: true }
    );

    const currentUser = authStore.getState().user;
    if (currentUser) {
      authStore.getState().setUser({
        ...currentUser,
        twoFactorEnabled: false,
      });
    }

    console.debug('[AuthService] 2FA disabled');
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] 2FA disable failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Get current user profile with error handling
 */
export const authServiceGetProfile = async (): Promise<User> => {
  try {
    console.debug('[AuthService] Fetching user profile');

    const response = await apiGet<User>(AUTH_ENDPOINTS.PROFILE);

    // ✅ FIXED: Validate response
    if (!response || typeof response !== 'object') {
      throw new Error('Invalid profile response');
    }

    authStore.getState().setUser(response);

    console.debug('[AuthService] Profile fetched successfully');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Profile fetch failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Update user profile with error handling
 */
export const authServiceUpdateProfile = async (
  updates: Partial<User>
): Promise<User> => {
  try {
    // ✅ FIXED: Validate updates
    if (!updates || typeof updates !== 'object') {
      throw new Error('Invalid profile updates');
    }

    console.debug('[AuthService] Updating user profile');

    const response = await apiPatch<User>(
      AUTH_ENDPOINTS.PROFILE,
      updates
    );

    // ✅ FIXED: Validate response
    if (!response || typeof response !== 'object') {
      throw new Error('Invalid profile update response');
    }

    authStore.getState().setUser(response);

    console.debug('[AuthService] Profile updated successfully');

    return response;
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Profile update failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Change password with error handling
 */
export const authServiceChangePassword = async (
  currentPassword: string,
  newPassword: string
): Promise<void> => {
  try {
    // ✅ FIXED: Validate input
    if (!currentPassword || !newPassword) {
      throw new Error('Current password and new password are required');
    }

    if (currentPassword === newPassword) {
      throw new Error('New password must be different from current password');
    }

    console.debug('[AuthService] Changing password');

    await apiPatch(
      AUTH_ENDPOINTS.CHANGE_PASSWORD,
      { currentPassword, newPassword },
      { skipRetry: true }
    );

    console.debug('[AuthService] Password changed successfully');
  } catch (error) {
    const errorMessage = getErrorMessage(error);
    console.error('[AuthService] Password change failed:', errorMessage);
    throw new Error(errorMessage);
  }
};

/**
 * ✅ FIXED: Restore session from stored tokens with comprehensive error handling
 */
export const authServiceRestoreSession = async (): Promise<boolean> => {
  try {
    const refreshToken = getStoredRefreshToken();

    console.debug('[AuthService] Attempting to restore session');

    // ✅ FIXED: Check if tokens exist and are valid
    if (!refreshToken) {
      console.debug('[AuthService] No stored refresh token');
      return false;
    }

    if (!areTokensValid()) {
      console.debug('[AuthService] Stored tokens are invalid or expired');
      clearTokens();
      return false;
    }

    // ✅ FIXED: Try to refresh token to validate it
    try {
      await authServiceRefreshToken(refreshToken);
      authStore.getState().setIsAuthenticated(true);
      console.debug('[AuthService] Session restored successfully');
      return true;
    } catch (error) {
      console.debug('[AuthService] Token refresh failed during session restore');
      clearTokens();
      return false;
    }
  } catch (error) {
    console.error('[AuthService] Session restore error:', error);
    clearTokens();
    return false;
  }
};

/**
 * ✅ FIXED: Check if user is authenticated
 */
export const isUserAuthenticated = (): boolean => {
  try {
    const store = authStore.getState();
    return (
      store.isAuthenticated &&
      store.token !== null &&
      validateTokenFormat(store.token)
    );
  } catch (error) {
    console.error('[AuthService] Error checking authentication:', error);
    return false;
  }
};

/**
 * ✅ FIXED: Get current access token
 */
export const getCurrentToken = (): string | null => {
  try {
    const token = authStore.getState().token;
    return token && validateTokenFormat(token) ? token : null;
  } catch (error) {
    console.error('[AuthService] Error getting current token:', error);
    return null;
  }
};

/**
 * ✅ FIXED: Get current user
 */
export const getCurrentUser = (): User | null => {
  try {
    return authStore.getState().user || null;
  } catch (error) {
    console.error('[AuthService] Error getting current user:', error);
    return null;
  }
};

// ============================================================================
// Export
// ============================================================================

export default {
  login: authServiceLogin,
  register: authServiceRegister,
  logout: authServiceLogout,
  refreshToken: authServiceRefreshToken,
  requestPasswordReset: authServiceRequestPasswordReset,
  confirmPasswordReset: authServiceConfirmPasswordReset,
  requestEmailVerification: authServiceRequestEmailVerification,
  confirmEmailVerification: authServiceConfirmEmailVerification,
  socialLogin: authServiceSocialLogin,
  connectSocialAccount: authServiceConnectSocialAccount,
  setup2FA: authServiceSetup2FA,
  verify2FA: authServiceVerify2FA,
  disable2FA: authServiceDisable2FA,
  getProfile: authServiceGetProfile,
  updateProfile: authServiceUpdateProfile,
  changePassword: authServiceChangePassword,
  restoreSession: authServiceRestoreSession,
  isUserAuthenticated,
  getCurrentToken,
  getCurrentUser,
  validateTokenFormat,
};
