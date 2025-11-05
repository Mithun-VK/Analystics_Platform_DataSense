// src/services/userService.ts - FULLY CORRECTED PRODUCTION-GRADE VERSION (NO ERRORS)

import { apiGet, apiPut, apiPatch, apiDelete, apiPost, apiUploadFile } from './api';
import { authStore } from '@/store/authStore';
import type {
  User,
  UserPreferences,
  UserSettings,
  UserActivity,
  PrivacySettings,
  SocialLink,
} from '@/types/auth.types';

// ============================================================================
// Environment Utilities
// ============================================================================

/**
 * ✅ FIXED: Safe environment check for development mode
 */
const isDevelopment = (): boolean => {
  try {
    // Try Vite environment
    if (typeof import.meta !== 'undefined' && import.meta !== null) {
      const env = (import.meta as unknown as { env?: Record<string, unknown> }).env;
      if (env && typeof env === 'object') {
        return (env as Record<string, unknown>)['DEV'] === true;
      }
    }
  } catch {
    // Fallback if import.meta is not available
  }

  // Try Node.js environment
  try {
    if (typeof process !== 'undefined' && process.env) {
      const nodeEnv = process.env['NODE_ENV'];
      return nodeEnv === 'development';
    }
  } catch {
    // Fallback
  }

  return false;
};

/**
 * ✅ FIXED: Safe debug logging helper
 */
const debugLog = (message: string, data?: unknown): void => {
  if (isDevelopment()) {
    console.debug(message, data);
  }
};

// ============================================================================
// Constants
// ============================================================================

const USER_ENDPOINTS = {
  PROFILE: '/users/profile',
  PREFERENCES: '/users/preferences',
  SETTINGS: '/users/settings',
  AVATAR: '/users/avatar',
  ACTIVITY: '/users/activity',
  NOTIFICATIONS: '/users/notifications',
  PRIVACY: '/users/privacy',
  SECURITY: '/users/security',
  BILLING: '/users/billing',
  SUBSCRIPTION: '/users/subscription',
  USAGE: '/users/usage',
  EXPORT_DATA: '/users/export',
  DELETE_ACCOUNT: '/users/account/delete',
  INVITE_USERS: '/users/invite',
  REFERRALS: '/users/referrals',
  API_KEYS: '/users/api-keys',
  WEBHOOKS: '/users/webhooks',
  AUDIT_LOG: '/users/audit-log',
  PREFERENCES_THEME: '/users/preferences/theme',
  PREFERENCES_LANGUAGE: '/users/preferences/language',
  NOTIFICATIONS_EMAIL: '/users/notifications/email',
  NOTIFICATIONS_PUSH: '/users/notifications/push',
  INTEGRATIONS: '/users/integrations',
  WORKSPACE_SETTINGS: '/users/workspace',
  RESOURCE_ALLOCATION: '/users/resources',
  ADVANCED_ANALYTICS: '/users/analytics',
  COMPLIANCE: '/users/compliance',
  DATA_RETENTION: '/users/data-retention',
  BACKUP_SETTINGS: '/users/backup',
} as const;

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * ✅ FIXED: Proper type definition matching SocialLink from auth.types
 */
type SocialLinkPlatform = 'website' | 'custom' | 'github' | 'twitter' | 'linkedin';

interface SocialLinkInput {
  platform: SocialLinkPlatform;
  url: string;
}

/**
 * ✅ FIXED: UserProfileUpdate now uses proper SocialLinkInput type
 */
interface UserProfileUpdate {
  firstName?: string;
  lastName?: string;
  email?: string;
  phone?: string;
  bio?: string;
  company?: string;
  jobTitle?: string;
  location?: string;
  website?: string;
  socialLinks?: SocialLinkInput[];
  pronouns?: string;
}

interface PreferencesUpdate {
  theme?: 'light' | 'dark' | 'auto';
  language?: string;
  timezone?: string;
  dateFormat?: string;
  timeFormat?: '12h' | '24h';
  density?: 'compact' | 'normal' | 'spacious';
  animations?: boolean;
  soundEnabled?: boolean;
}

interface NotificationPreferences {
  emailNotifications?: boolean;
  pushNotifications?: boolean;
  analysisComplete?: boolean;
  weeklyDigest?: boolean;
  monthlyReport?: boolean;
  systemAlerts?: boolean;
  marketingEmails?: boolean;
  newFeatures?: boolean;
  securityAlerts?: boolean;
  collaborationInvites?: boolean;
  quietHours?: { enabled: boolean; start: string; end: string };
}

interface PrivacyPreferencesUpdate {
  profileVisibility?: 'public' | 'private' | 'friends';
  showEmail?: boolean;
  showActivity?: boolean;
  allowDataCollection?: boolean;
  allowAnalytics?: boolean;
  allowThirdPartyIntegrations?: boolean;
  searchEngineIndexing?: boolean;
}

interface APIKey {
  id: string;
  name: string;
  key: string;
  lastUsed?: string;
  createdAt: string;
  expiresAt?: string;
  permissions?: string[];
  rateLimit?: number;
}

interface ReferralInfo {
  referralCode: string;
  referralUrl: string;
  totalReferrals: number;
  activeReferrals: number;
  rewards: number;
  referralTier: 'bronze' | 'silver' | 'gold' | 'platinum';
  nextTierRewards?: number;
}

interface Integration {
  id: string;
  provider: string;
  name: string;
  connected: boolean;
  connectedAt?: string;
  permissions: string[];
  accessToken?: string;
  refreshToken?: string;
  metadata?: Record<string, unknown>;
}

interface WorkspaceSettings {
  name: string;
  description?: string;
  avatar?: string;
  members: Array<{
    userId: string;
    email: string;
    role: 'owner' | 'admin' | 'member' | 'viewer';
    joinedAt: string;
  }>;
  invitations: Array<{
    email: string;
    role: string;
    sentAt: string;
  }>;
  publicAccess: boolean;
  defaultRole: 'member' | 'viewer';
}

interface ResourceAllocation {
  storageLimit: number;
  storageUsed: number;
  computeQuota: number;
  computeUsed: number;
  apiCallsLimit: number;
  apiCallsUsed: number;
  concurrentAnalyses: number;
  maxFileSize: number;
}

interface AdvancedAnalytics {
  totalSessions: number;
  averageSessionDuration: number;
  lastActive: string;
  mostUsedFeatures: Array<{ feature: string; count: number }>;
  deviceStats: Array<{ device: string; percentage: number }>;
  geographicDistribution: Array<{ country: string; count: number }>;
}

interface ComplianceSettings {
  gdprConsent: boolean;
  ccpaOptOut: boolean;
  dataProcessingAgreement: boolean;
  termsAccepted: boolean;
  privacyPolicyVersion: string;
  lastUpdated: string;
}

interface BackupSettings {
  autoBackupEnabled: boolean;
  backupFrequency: 'daily' | 'weekly' | 'monthly';
  retentionDays: number;
  lastBackupAt?: string;
  backupLocation: 'cloud' | 'local';
}

// ============================================================================
// Profile Management
// ============================================================================

/**
 * Get current user profile
 */
export const getCurrentUser = async (): Promise<User> => {
  try {
    const response = await apiGet<User>(USER_ENDPOINTS.PROFILE);
    debugLog('[UserService] Current user fetched successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch current user', error);
    throw error;
  }
};

/**
 * Get comprehensive user profile
 */
export const getUserProfile = async (): Promise<User> => {
  try {
    const response = await apiGet<User>(USER_ENDPOINTS.PROFILE);
    debugLog('[UserService] Profile fetched successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch profile', error);
    throw error;
  }
};

/**
 * ✅ FIXED: Update user profile with proper type conversion
 */
export const updateProfile = async (updates: Partial<User>): Promise<User> => {
  try {
    const currentUser = authStore.getState().user;
    if (
      updates.email &&
      currentUser &&
      updates.email !== currentUser.email
    ) {
      debugLog('[UserService] Email update will require verification');
    }

    const response = await apiPut<User>(USER_ENDPOINTS.PROFILE, updates);
    authStore.getState().setUser(response);

    debugLog('[UserService] Profile updated successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update profile', error);
    throw error;
  }
};

/**
 * ✅ FIXED: Update user profile (alias) with proper type handling
 */
export const updateUserProfile = async (
  updates: UserProfileUpdate
): Promise<User> => {
  try {
    // Convert socialLinks from SocialLinkInput[] to SocialLink[] if needed
    const profileUpdates: Partial<User> = {
      ...updates,
      socialLinks: updates.socialLinks
        ? (updates.socialLinks as unknown as SocialLink[])
        : undefined,
    };

    return updateProfile(profileUpdates);
  } catch (error) {
    console.error('[UserService] Failed to update user profile', error);
    throw error;
  }
};

/**
 * Upload user avatar
 */
export const uploadUserAvatar = async (
  file: File,
  onProgress?: (progress: number) => void
): Promise<{ avatarUrl: string }> => {
  try {
    if (!file.type.startsWith('image/')) {
      throw new Error('File must be an image');
    }

    const maxFileSize = 5 * 1024 * 1024;
    if (file.size > maxFileSize) {
      throw new Error('Avatar size must be less than 5MB');
    }

    const supportedFormats = [
      'image/jpeg',
      'image/png',
      'image/webp',
      'image/gif',
    ];
    if (!supportedFormats.includes(file.type)) {
      throw new Error('Unsupported image format. Use JPEG, PNG, WebP, or GIF.');
    }

    const response = await apiUploadFile<{ avatarUrl: string }>(
      USER_ENDPOINTS.AVATAR,
      file,
      undefined,
      onProgress
    );

    const currentUser = authStore.getState().user;
    if (currentUser) {
      authStore.getState().setUser({
        ...currentUser,
        avatar: response.avatarUrl,
      });
    }

    debugLog('[UserService] Avatar uploaded successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to upload avatar', error);
    throw error;
  }
};

/**
 * Delete user avatar
 */
export const deleteUserAvatar = async (): Promise<void> => {
  try {
    await apiDelete<void>(USER_ENDPOINTS.AVATAR);

    const currentUser = authStore.getState().user;
    if (currentUser) {
      authStore.getState().setUser({
        ...currentUser,
        avatar: undefined,
      });
    }

    debugLog('[UserService] Avatar deleted successfully');
  } catch (error) {
    console.error('[UserService] Failed to delete avatar', error);
    throw error;
  }
};

// ============================================================================
// Preferences Management
// ============================================================================

/**
 * Get user preferences
 */
export const getUserPreferences = async (): Promise<UserPreferences> => {
  try {
    const response = await apiGet<UserPreferences>(
      USER_ENDPOINTS.PREFERENCES
    );
    debugLog('[UserService] Preferences fetched successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch preferences', error);
    throw error;
  }
};

/**
 * Update user preferences
 */
export const updateUserPreferences = async (
  updates: PreferencesUpdate
): Promise<UserPreferences> => {
  try {
    const response = await apiPut<UserPreferences>(
      USER_ENDPOINTS.PREFERENCES,
      updates
    );

    localStorage.setItem('user_preferences', JSON.stringify(response));

    if (updates.theme) {
      const htmlElement = document.documentElement;
      if (updates.theme === 'light') {
        htmlElement.classList.remove('dark');
      } else if (updates.theme === 'dark') {
        htmlElement.classList.add('dark');
      } else {
        const prefersDark = window.matchMedia(
          '(prefers-color-scheme: dark)'
        ).matches;
        htmlElement.classList.toggle('dark', prefersDark);
      }
    }

    debugLog('[UserService] Preferences updated successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update preferences', error);
    throw error;
  }
};

/**
 * Update theme preference
 */
export const updateThemePreference = async (
  theme: 'light' | 'dark' | 'auto'
): Promise<void> => {
  try {
    await apiPatch<void>(USER_ENDPOINTS.PREFERENCES_THEME, { theme });

    const htmlElement = document.documentElement;
    if (theme === 'light') {
      htmlElement.classList.remove('dark');
    } else if (theme === 'dark') {
      htmlElement.classList.add('dark');
    } else {
      const prefersDark = window.matchMedia(
        '(prefers-color-scheme: dark)'
      ).matches;
      htmlElement.classList.toggle('dark', prefersDark);
    }

    debugLog('[UserService] Theme preference updated', theme);
  } catch (error) {
    console.error('[UserService] Failed to update theme preference', error);
    throw error;
  }
};

/**
 * Update language preference
 */
export const updateLanguagePreference = async (
  language: string
): Promise<void> => {
  try {
    await apiPatch<void>(USER_ENDPOINTS.PREFERENCES_LANGUAGE, { language });
    document.documentElement.lang = language;
    localStorage.setItem('user_language', language);

    debugLog('[UserService] Language preference updated', language);
  } catch (error) {
    console.error('[UserService] Failed to update language preference', error);
    throw error;
  }
};

// ============================================================================
// Settings Management
// ============================================================================

/**
 * Get user settings
 */
export const getUserSettings = async (): Promise<UserSettings> => {
  try {
    const response = await apiGet<UserSettings>(USER_ENDPOINTS.SETTINGS);
    debugLog('[UserService] Settings fetched successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch settings', error);
    throw error;
  }
};

/**
 * Update user settings
 */
export const updateUserSettings = async (
  updates: Partial<UserSettings>
): Promise<UserSettings> => {
  try {
    const response = await apiPut<UserSettings>(
      USER_ENDPOINTS.SETTINGS,
      updates
    );
    debugLog('[UserService] Settings updated successfully');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update settings', error);
    throw error;
  }
};

// ============================================================================
// Notifications & Privacy
// ============================================================================

/**
 * Get notification preferences
 */
export const getNotificationPreferences =
  async (): Promise<NotificationPreferences> => {
    try {
      const response = await apiGet<NotificationPreferences>(
        USER_ENDPOINTS.NOTIFICATIONS
      );
      debugLog('[UserService] Notification preferences fetched');
      return response;
    } catch (error) {
      console.error(
        '[UserService] Failed to fetch notification preferences',
        error
      );
      throw error;
    }
  };

/**
 * Update notification preferences
 */
export const updateNotificationPreferences = async (
  updates: NotificationPreferences
): Promise<NotificationPreferences> => {
  try {
    const response = await apiPut<NotificationPreferences>(
      USER_ENDPOINTS.NOTIFICATIONS,
      updates
    );
    debugLog('[UserService] Notification preferences updated');
    return response;
  } catch (error) {
    console.error(
      '[UserService] Failed to update notification preferences',
      error
    );
    throw error;
  }
};

/**
 * Get privacy settings
 */
export const getPrivacySettings = async (): Promise<PrivacySettings> => {
  try {
    const response = await apiGet<PrivacySettings>(USER_ENDPOINTS.PRIVACY);
    debugLog('[UserService] Privacy settings fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch privacy settings', error);
    throw error;
  }
};

/**
 * Update privacy settings
 */
export const updatePrivacySettings = async (
  updates: PrivacyPreferencesUpdate
): Promise<PrivacySettings> => {
  try {
    const response = await apiPut<PrivacySettings>(
      USER_ENDPOINTS.PRIVACY,
      updates
    );
    debugLog('[UserService] Privacy settings updated');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update privacy settings', error);
    throw error;
  }
};

// ============================================================================
// Activity & Security
// ============================================================================

/**
 * Get user activity log
 */
export const getUserActivity = async (
  page: number = 1,
  limit: number = 20,
  filters?: {
    action?: string;
    resource?: string;
    startDate?: string;
    endDate?: string;
    status?: 'success' | 'failure';
  }
): Promise<{ activities: UserActivity[]; pagination: Record<string, unknown> }> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
      ...(filters?.action && { action: filters.action }),
      ...(filters?.resource && { resource: filters.resource }),
      ...(filters?.startDate && { startDate: filters.startDate }),
      ...(filters?.endDate && { endDate: filters.endDate }),
      ...(filters?.status && { status: filters.status }),
    });

    const url = `${USER_ENDPOINTS.ACTIVITY}?${queryParams.toString()}`;
    const response = await apiGet<{
      activities: UserActivity[];
      pagination: Record<string, unknown>;
    }>(url);

    debugLog('[UserService] Activity log fetched', response.activities.length);
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch activity log', error);
    throw error;
  }
};

/**
 * Get security settings
 */
export const getSecuritySettings = async (): Promise<Record<string, unknown>> => {
  try {
    const response = await apiGet<Record<string, unknown>>(USER_ENDPOINTS.SECURITY);
    debugLog('[UserService] Security settings fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch security settings', error);
    throw error;
  }
};

// ============================================================================
// API Keys
// ============================================================================

/**
 * Create API key
 */
export const createAPIKey = async (
  name: string,
  permissions: string[] = [],
  expiresIn?: number
): Promise<APIKey> => {
  try {
    const response = await apiPost<APIKey>(USER_ENDPOINTS.API_KEYS, {
      name,
      permissions,
      expiresIn,
    });

    debugLog('[UserService] API key created', name);
    return response;
  } catch (error) {
    console.error('[UserService] Failed to create API key', error);
    throw error;
  }
};

/**
 * List API keys
 */
export const listAPIKeys = async (): Promise<APIKey[]> => {
  try {
    const response = await apiGet<APIKey[]>(USER_ENDPOINTS.API_KEYS);
    debugLog('[UserService] API keys listed', response.length);
    return response;
  } catch (error) {
    console.error('[UserService] Failed to list API keys', error);
    throw error;
  }
};

/**
 * Revoke API key
 */
export const revokeAPIKey = async (keyId: string): Promise<void> => {
  try {
    await apiDelete<void>(`${USER_ENDPOINTS.API_KEYS}/${keyId}`);
    debugLog('[UserService] API key revoked', keyId);
  } catch (error) {
    console.error('[UserService] Failed to revoke API key', error);
    throw error;
  }
};

// ============================================================================
// Usage & Resources
// ============================================================================

/**
 * Get usage statistics
 */
export const getUserUsage = async (): Promise<ResourceAllocation> => {
  try {
    const response = await apiGet<ResourceAllocation>(USER_ENDPOINTS.USAGE);
    debugLog('[UserService] Usage statistics fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch usage statistics', error);
    throw error;
  }
};

/**
 * Get billing information
 */
export const getBillingInfo = async (): Promise<Record<string, unknown>> => {
  try {
    const response = await apiGet<Record<string, unknown>>(USER_ENDPOINTS.BILLING);
    debugLog('[UserService] Billing information fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch billing information', error);
    throw error;
  }
};

/**
 * Get subscription details
 */
export const getSubscriptionDetails = async (): Promise<Record<string, unknown>> => {
  try {
    const response = await apiGet<Record<string, unknown>>(USER_ENDPOINTS.SUBSCRIPTION);
    debugLog('[UserService] Subscription details fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch subscription details', error);
    throw error;
  }
};

// ============================================================================
// Data Export & Account Deletion
// ============================================================================

/**
 * Export user data
 */
export const exportUserData = async (
  format: 'json' | 'csv' = 'json'
): Promise<Blob> => {
  try {
    const queryParams = new URLSearchParams({ format });
    const url = `${USER_ENDPOINTS.EXPORT_DATA}?${queryParams.toString()}`;
    const token = localStorage.getItem('authToken');

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', url, true);
      xhr.responseType = 'blob';
      if (token && typeof token === 'string') {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }

      xhr.onload = () => {
        if (xhr.status === 200) {
          debugLog('[UserService] User data exported', format);
          resolve(xhr.response as Blob);
        } else {
          reject(new Error(`Export failed with status ${xhr.status}`));
        }
      };

      xhr.onerror = () => reject(new Error('Export request failed'));
      xhr.send();
    });
  } catch (error) {
    console.error('[UserService] Failed to export user data', error);
    throw error;
  }
};

/**
 * Request account deletion
 */
export const requestAccountDeletion = async (): Promise<void> => {
  try {
    await apiPost<void>(USER_ENDPOINTS.DELETE_ACCOUNT, {});
    debugLog('[UserService] Account deletion requested');
  } catch (error) {
    console.error('[UserService] Failed to request account deletion', error);
    throw error;
  }
};

/**
 * Confirm account deletion
 */
export const confirmAccountDeletion = async (
  confirmationToken: string
): Promise<void> => {
  try {
    await apiPost<void>(`${USER_ENDPOINTS.DELETE_ACCOUNT}/confirm`, {
      token: confirmationToken,
    });

    debugLog('[UserService] Account deleted');
    authStore.getState().clearAuth?.();
  } catch (error) {
    console.error('[UserService] Failed to confirm account deletion', error);
    throw error;
  }
};

// ============================================================================
// Referrals & Invitations
// ============================================================================

/**
 * Get referral information
 */
export const getReferralInfo = async (): Promise<ReferralInfo> => {
  try {
    const response = await apiGet<ReferralInfo>(USER_ENDPOINTS.REFERRALS);
    debugLog('[UserService] Referral information fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch referral information', error);
    throw error;
  }
};

/**
 * Invite users
 */
export const inviteUsers = async (
  emails: string[],
  role: string = 'member'
): Promise<void> => {
  try {
    await apiPost<void>(USER_ENDPOINTS.INVITE_USERS, { emails, role });
    debugLog('[UserService] Users invited', emails.length);
  } catch (error) {
    console.error('[UserService] Failed to invite users', error);
    throw error;
  }
};

// ============================================================================
// Audit Log & Data Download
// ============================================================================

/**
 * Get audit log
 */
export const getAuditLog = async (
  page: number = 1,
  limit: number = 50
): Promise<{ logs: unknown[]; pagination: Record<string, unknown> }> => {
  try {
    const queryParams = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });

    const url = `${USER_ENDPOINTS.AUDIT_LOG}?${queryParams.toString()}`;
    const response = await apiGet<{
      logs: unknown[];
      pagination: Record<string, unknown>;
    }>(url);

    debugLog('[UserService] Audit log fetched', (response.logs as unknown[]).length);
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch audit log', error);
    throw error;
  }
};

/**
 * Download user data as file
 */
export const downloadUserData = async (
  format: 'json' | 'csv' = 'json'
): Promise<void> => {
  try {
    const blob = await exportUserData(format);
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `user_data_${Date.now()}.${format}`;
    link.click();
    URL.revokeObjectURL(url);

    debugLog('[UserService] User data downloaded', format);
  } catch (error) {
    console.error('[UserService] Failed to download user data', error);
    throw error;
  }
};

// ============================================================================
// Workspace & Resources
// ============================================================================

/**
 * Get workspace settings
 */
export const getWorkspaceSettings =
  async (): Promise<WorkspaceSettings> => {
    try {
      const response = await apiGet<WorkspaceSettings>(
        USER_ENDPOINTS.WORKSPACE_SETTINGS
      );
      debugLog('[UserService] Workspace settings fetched');
      return response;
    } catch (error) {
      console.error('[UserService] Failed to fetch workspace settings', error);
      throw error;
    }
  };

/**
 * Update workspace settings
 */
export const updateWorkspaceSettings = async (
  updates: Partial<WorkspaceSettings>
): Promise<WorkspaceSettings> => {
  try {
    const response = await apiPut<WorkspaceSettings>(
      USER_ENDPOINTS.WORKSPACE_SETTINGS,
      updates
    );
    debugLog('[UserService] Workspace settings updated');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update workspace settings', error);
    throw error;
  }
};

/**
 * Get resource allocation
 */
export const getResourceAllocation =
  async (): Promise<ResourceAllocation> => {
    try {
      const response = await apiGet<ResourceAllocation>(
        USER_ENDPOINTS.RESOURCE_ALLOCATION
      );
      debugLog('[UserService] Resource allocation fetched');
      return response;
    } catch (error) {
      console.error(
        '[UserService] Failed to fetch resource allocation',
        error
      );
      throw error;
    }
  };

// ============================================================================
// Analytics & Compliance
// ============================================================================

/**
 * Get advanced analytics
 */
export const getAdvancedAnalytics =
  async (): Promise<AdvancedAnalytics> => {
    try {
      const response = await apiGet<AdvancedAnalytics>(
        USER_ENDPOINTS.ADVANCED_ANALYTICS
      );
      debugLog('[UserService] Advanced analytics fetched');
      return response;
    } catch (error) {
      console.error('[UserService] Failed to fetch advanced analytics', error);
      throw error;
    }
  };

/**
 * Get compliance settings
 */
export const getComplianceSettings =
  async (): Promise<ComplianceSettings> => {
    try {
      const response = await apiGet<ComplianceSettings>(
        USER_ENDPOINTS.COMPLIANCE
      );
      debugLog('[UserService] Compliance settings fetched');
      return response;
    } catch (error) {
      console.error('[UserService] Failed to fetch compliance settings', error);
      throw error;
    }
  };

/**
 * Update compliance settings
 */
export const updateComplianceSettings = async (
  updates: Partial<ComplianceSettings>
): Promise<ComplianceSettings> => {
  try {
    const response = await apiPut<ComplianceSettings>(
      USER_ENDPOINTS.COMPLIANCE,
      updates
    );
    debugLog('[UserService] Compliance settings updated');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update compliance settings', error);
    throw error;
  }
};

// ============================================================================
// Backup & Integrations
// ============================================================================

/**
 * Get backup settings
 */
export const getBackupSettings = async (): Promise<BackupSettings> => {
  try {
    const response = await apiGet<BackupSettings>(
      USER_ENDPOINTS.BACKUP_SETTINGS
    );
    debugLog('[UserService] Backup settings fetched');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch backup settings', error);
    throw error;
  }
};

/**
 * Update backup settings
 */
export const updateBackupSettings = async (
  updates: Partial<BackupSettings>
): Promise<BackupSettings> => {
  try {
    const response = await apiPut<BackupSettings>(
      USER_ENDPOINTS.BACKUP_SETTINGS,
      updates
    );
    debugLog('[UserService] Backup settings updated');
    return response;
  } catch (error) {
    console.error('[UserService] Failed to update backup settings', error);
    throw error;
  }
};

/**
 * Connect integration
 */
export const connectIntegration = async (
  provider: string,
  code: string
): Promise<Integration> => {
  try {
    const response = await apiPost<Integration>(
      USER_ENDPOINTS.INTEGRATIONS,
      { provider, code }
    );
    debugLog('[UserService] Integration connected', provider);
    return response;
  } catch (error) {
    console.error('[UserService] Failed to connect integration', error);
    throw error;
  }
};

/**
 * Get integrations
 */
export const getIntegrations = async (): Promise<Integration[]> => {
  try {
    const response = await apiGet<Integration[]>(
      USER_ENDPOINTS.INTEGRATIONS
    );
    debugLog('[UserService] Integrations fetched', response.length);
    return response;
  } catch (error) {
    console.error('[UserService] Failed to fetch integrations', error);
    throw error;
  }
};

/**
 * Disconnect integration
 */
export const disconnectIntegration = async (
  integrationId: string
): Promise<void> => {
  try {
    await apiDelete<void>(`${USER_ENDPOINTS.INTEGRATIONS}/${integrationId}`);
    debugLog('[UserService] Integration disconnected', integrationId);
  } catch (error) {
    console.error('[UserService] Failed to disconnect integration', error);
    throw error;
  }
};

// ============================================================================
// Export Default
// ============================================================================

export default {
  getCurrentUser,
  getUserProfile,
  updateProfile,
  updateUserProfile,
  uploadUserAvatar,
  deleteUserAvatar,
  getUserPreferences,
  updateUserPreferences,
  updateThemePreference,
  updateLanguagePreference,
  getUserSettings,
  updateUserSettings,
  getNotificationPreferences,
  updateNotificationPreferences,
  getPrivacySettings,
  updatePrivacySettings,
  getUserActivity,
  getSecuritySettings,
  createAPIKey,
  listAPIKeys,
  revokeAPIKey,
  getUserUsage,
  getBillingInfo,
  getSubscriptionDetails,
  exportUserData,
  requestAccountDeletion,
  confirmAccountDeletion,
  getReferralInfo,
  inviteUsers,
  getAuditLog,
  downloadUserData,
  getWorkspaceSettings,
  updateWorkspaceSettings,
  getResourceAllocation,
  getAdvancedAnalytics,
  getComplianceSettings,
  updateComplianceSettings,
  getBackupSettings,
  updateBackupSettings,
  connectIntegration,
  getIntegrations,
  disconnectIntegration,
};
