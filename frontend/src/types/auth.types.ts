// src/types/auth.types.ts
/**
 * Authentication and User Management Types
 * Comprehensive type definitions for all auth-related operations
 */

// ============================================================================
// User & Profile Types
// ============================================================================

/**
 * Core user object
 */
export interface User {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  fullName?: string;
  avatar?: string;
  phone?: string;
  bio?: string;
  company?: string;
  jobTitle?: string;
  location?: string;
  website?: string;
  socialLinks?: SocialLink[];
  pronouns?: string;

  // Account Status
  isActive: boolean;
  emailVerified: boolean;
  phoneVerified?: boolean;
  twoFactorEnabled: boolean;
  accountLocked: boolean;
  lastLoginAt?: string;
  createdAt: string;
  updatedAt: string;
  deletedAt?: string;

  // Roles & Permissions
  roles: string[];
  permissions: string[];
  capabilities?: Record<string, boolean>;
  isAdmin?: boolean;

  // Security & Compliance
  passwordChangeRequired: boolean;
  passwordChangedAt?: string;
  passwordExpiresAt?: string;
  lastPasswordReset?: string;
  loginAttempts: number;
  lastFailedLoginAt?: string;
  acceptedTermsVersion?: string;
  acceptedPrivacyVersion?: string;
  acceptedCookiesVersion?: string;

  // Preferences
  preferences?: UserPreferences;
  settings?: UserSettings;
  notificationSettings?: NotificationSettings;
  privacySettings?: PrivacySettings;

  // Subscription & Plan
  subscriptionPlan?: 'free' | 'pro' | 'enterprise';
  subscriptionStatus?: 'active' | 'inactive' | 'cancelled' | 'suspended';
  subscriptionExpiresAt?: string;
  trialEndsAt?: string;

  // Metadata
  metadata?: Record<string, any>;
  tags?: string[];
  customFields?: Record<string, any>;
}

/**
 * User profile (extended user info)
 */
export interface UserProfile extends User {
  // Additional profile information
  bio: string;
  avatar: string;
  bannerImage?: string;
  followerCount: number;
  followingCount: number;
  totalDatasets: number;
  totalAnalyses: number;
  joinDate: string;
  accountAge: number; // in days
}

/**
 * Social link
 */
export interface SocialLink {
  platform: 'github' | 'twitter' | 'linkedin' | 'website' | 'custom';
  url: string;
  username?: string;
}

/**
 * User preferences
 */
export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  sidebarTheme?: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  dateFormat: string;
  timeFormat: '12h' | '24h';
  density?: 'compact' | 'normal' | 'spacious';
  animations: boolean;
  soundEnabled: boolean;
  helpTooltipsEnabled: boolean;
  keyboardShortcutsEnabled: boolean;
}

/**
 * User settings
 */
export interface UserSettings {
  emailNotifications: boolean;
  pushNotifications: boolean;
  smsNotifications: boolean;
  twoFactorMethod?: 'email' | 'authenticator' | 'sms';
  recoveryEmail?: string;
  recoveryPhone?: string;
  sessionTimeout: number; // in minutes
  rememberMe: boolean;
  autoLogout: boolean;
  activityLogging: boolean;
  dataCollection: boolean;
}

/**
 * Notification settings
 */
export interface NotificationSettings {
  emailNotifications: boolean;
  pushNotifications: boolean;
  smsNotifications: boolean;
  analysisComplete: boolean;
  weeklyDigest: boolean;
  monthlyReport: boolean;
  systemAlerts: boolean;
  marketingEmails: boolean;
  newFeatures: boolean;
  securityAlerts: boolean;
  collaborationInvites: boolean;
  quietHours?: {
    enabled: boolean;
    startTime: string; // HH:mm format
    endTime: string;
    timezone: string;
  };
}

/**
 * Privacy settings
 */
export interface PrivacySettings {
  profileVisibility: 'public' | 'private' | 'friends';
  showEmail: boolean;
  showPhoneNumber: boolean;
  showActivity: boolean;
  showConnections: boolean;
  allowDataCollection: boolean;
  allowAnalytics: boolean;
  allowThirdPartyIntegrations: boolean;
  searchEngineIndexing: boolean;
  directMessages: 'everyone' | 'friends' | 'none';
  showOnlineStatus: boolean;
}

// ============================================================================
// Authentication Request/Response Types
// ============================================================================

/**
 * Login request
 */
export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
  deviceId?: string;
  deviceName?: string;
}

/**
 * Login response
 */
export interface LoginResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn: number; // in seconds
  tokenType: string;
  requiresMFA?: boolean;
  mfaMethod?: string;
}

/**
 * Register request
 */
export interface RegisterRequest {
  email: string;
  password: string;
  confirmPassword?: string;
  firstName: string;
  lastName: string;
  acceptTerms: boolean;
  acceptPrivacy?: boolean;
  company?: string;
  referralCode?: string;
}

/**
 * Register response
 */
export interface RegisterResponse {
  user: User;
  accessToken?: string;
  refreshToken?: string;
  emailVerificationRequired: boolean;
  verificationEmail?: string;
  message: string;
}

/**
 * Token refresh request
 */
export interface TokenRefreshRequest {
  refreshToken: string;
  deviceId?: string;
}

/**
 * Token refresh response
 */
export interface RefreshTokenResponse {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: string;
}

/**
 * Password reset request
 */
export interface PasswordResetRequest {
  email: string;
}

/**
 * Password reset response
 */
export interface PasswordResetResponse {
  message: string;
  resetToken?: string;
  expiresIn?: number;
}

/**
 * Password reset confirm request
 */
export interface PasswordResetConfirmRequest {
  token: string;
  newPassword: string;
  confirmPassword: string;
}

/**
 * Email verification request
 */
export interface EmailVerificationRequest {
  email?: string;
  token?: string;
}

/**
 * Email verification response
 */
export interface EmailVerificationResponse {
  message: string;
  verified: boolean;
  user?: User;
}

/**
 * Social login request
 */
export interface SocialLoginRequest {
  provider: 'google' | 'github' | 'facebook' | 'microsoft' | 'apple';
  code: string;
  state?: string;
  idToken?: string;
  accessToken?: string;
  redirectUri?: string;
}

/**
 * Social login response
 */
export interface SocialLoginResponse {
  user: User;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  isNewUser: boolean;
  requiresMFA?: boolean;
}

// ============================================================================
// MFA (Multi-Factor Authentication) Types
// ============================================================================

/**
 * MFA setup request
 */
export interface MFASetupRequest {
  method: 'email' | 'authenticator' | 'sms';
}

/**
 * MFA setup response
 */
export interface MFASetupResponse {
  secret?: string;
  qrCode?: string;
  backupCodes?: string[];
  message: string;
}

/**
 * MFA verify request
 */
export interface MFAVerifyRequest {
  code: string;
  method: 'email' | 'authenticator' | 'sms';
  rememberDevice?: boolean;
}

/**
 * MFA verify response
 */
export interface MFAVerifyResponse {
  verified: boolean;
  backupCodes?: string[];
  message: string;
}

/**
 * MFA disable request
 */
export interface MFADisableRequest {
  password: string;
  method?: string;
}

/**
 * MFA status
 */
export interface MFAStatus {
  enabled: boolean;
  method?: 'email' | 'authenticator' | 'sms';
  verifiedAt?: string;
  backupCodesCount: number;
}

// ============================================================================
// Session & Token Types
// ============================================================================

/**
 * Session information
 */
export interface Session {
  id: string;
  userId: string;
  refreshToken: string;
  accessToken?: string;
  deviceId?: string;
  deviceName?: string;
  ipAddress: string;
  userAgent: string;
  expiresAt: string;
  createdAt: string;
  lastActivityAt: string;
  isActive: boolean;
}

/**
 * Token payload
 */
export interface TokenPayload {
  sub: string; // user id
  email: string;
  roles: string[];
  permissions: string[];
  iat: number;
  exp: number;
  iss: string;
  aud: string;
}

/**
 * Refresh token payload
 */
export interface RefreshTokenPayload extends TokenPayload {
  deviceId?: string;
}

// ============================================================================
// Security & Audit Types
// ============================================================================

/**
 * User activity log
 */
export interface UserActivity {
  id: string;
  userId: string;
  action: string;
  resource: string;
  resourceId?: string;
  status: 'success' | 'failure' | 'pending';
  ipAddress: string;
  userAgent: string;
  location?: {
    country: string;
    city: string;
    latitude: number;
    longitude: number;
  };
  metadata?: Record<string, any>;
  timestamp: string;
}

/**
 * Login attempt
 */
export interface LoginAttempt {
  id: string;
  email: string;
  ipAddress: string;
  userAgent: string;
  success: boolean;
  reason?: string;
  timestamp: string;
}

/**
 * Security event
 */
export interface SecurityEvent {
  id: string;
  userId: string;
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  ipAddress: string;
  location?: string;
  timestamp: string;
  resolved: boolean;
  resolvedAt?: string;
  metadata?: Record<string, any>;
}

/**
 * Audit log entry
 */
export interface AuditLogEntry {
  id: string;
  userId: string;
  action: string;
  resourceType: string;
  resourceId: string;
  changes?: {
    field: string;
    oldValue: any;
    newValue: any;
  }[];
  ipAddress: string;
  userAgent: string;
  timestamp: string;
  metadata?: Record<string, any>;
}

// ============================================================================
// Password & Credential Types
// ============================================================================

/**
 * Password strength
 */
export interface PasswordStrength {
  score: number; // 0-5
  feedback: string[];
  suggestions: string[];
  isStrong: boolean;
}

/**
 * Password policy
 */
export interface PasswordPolicy {
  minLength: number;
  maxLength: number;
  requireUppercase: boolean;
  requireLowercase: boolean;
  requireNumbers: boolean;
  requireSpecialChars: boolean;
  specialChars: string;
  expirationDays?: number;
  historyCount?: number;
}

/**
 * Credential change request
 */
export interface CredentialChangeRequest {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

/**
 * Credential change response
 */
export interface CredentialChangeResponse {
  success: boolean;
  message: string;
  requiresReauth?: boolean;
}

// ============================================================================
// Device & Trust Types
// ============================================================================

/**
 * Device information
 */
export interface DeviceInfo {
  id: string;
  name: string;
  type: 'desktop' | 'mobile' | 'tablet' | 'unknown';
  os: string;
  osVersion: string;
  browser: string;
  browserVersion: string;
  userAgent: string;
  ipAddress: string;
  lastUsedAt: string;
  createdAt: string;
}

/**
 * Trusted device
 */
export interface TrustedDevice extends DeviceInfo {
  userId: string;
  isTrusted: boolean;
  trustExpiresAt?: string;
}

// ============================================================================
// OAuth & Social Auth Types
// ============================================================================

/**
 * OAuth provider config
 */
export interface OAuthProviderConfig {
  provider: string;
  clientId: string;
  clientSecret?: string;
  redirectUri: string;
  scopes: string[];
  authorizationEndpoint: string;
  tokenEndpoint: string;
  userInfoEndpoint: string;
}

/**
 * Social account
 */
export interface SocialAccount {
  provider: string;
  providerId: string;
  email?: string;
  name?: string;
  avatar?: string;
  accessToken?: string;
  refreshToken?: string;
  expiresAt?: string;
  connectedAt: string;
  lastUsedAt?: string;
}

/**
 * OAuth token
 */
export interface OAuthToken {
  accessToken: string;
  refreshToken?: string;
  expiresIn: number;
  tokenType: string;
  scope: string;
}

// ============================================================================
// Authorization & Permission Types
// ============================================================================

/**
 * Permission
 */
export interface Permission {
  id: string;
  name: string;
  description?: string;
  category: string;
  scope?: string;
}

/**
 * Role
 */
export interface Role {
  id: string;
  name: string;
  description?: string;
  permissions: Permission[];
  isSystem: boolean;
  createdAt: string;
  updatedAt: string;
}

/**
 * ACL (Access Control List) entry
 */
export interface ACLEntry {
  id: string;
  subjectId: string;
  subjectType: 'user' | 'role' | 'group';
  resourceId: string;
  resourceType: string;
  action: string;
  effect: 'allow' | 'deny';
  conditions?: Record<string, any>;
  expiresAt?: string;
}

// ============================================================================
// Compliance & GDPR Types
// ============================================================================

/**
 * Compliance status
 */
export interface ComplianceStatus {
  gdprConsent: boolean;
  ccpaOptOut: boolean;
  dataProcessingAgreement: boolean;
  termsAccepted: boolean;
  privacyPolicyVersion: string;
  cookieConsent: boolean;
  lastUpdated: string;
}

/**
 * Data export request
 */
export interface DataExportRequest {
  format: 'json' | 'csv' | 'xml';
  includePersonalData: boolean;
  includeActivityLog: boolean;
  includeDatasets: boolean;
}

/**
 * Data export response
 */
export interface DataExportResponse {
  exportId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  downloadUrl?: string;
  expiresAt?: string;
  createdAt: string;
}

// ============================================================================
// Subscription & Billing Types
// ============================================================================

/**
 * Subscription plan
 */
export interface SubscriptionPlan {
  id: string;
  name: string;
  description?: string;
  price: number;
  currency: string;
  billingCycle: 'monthly' | 'yearly' | 'lifetime';
  features: string[];
  limits: Record<string, number>;
}

/**
 * Subscription
 */
export interface Subscription {
  id: string;
  userId: string;
  planId: string;
  status: 'active' | 'inactive' | 'cancelled' | 'suspended';
  startDate: string;
  endDate?: string;
  renewalDate?: string;
  autoRenew: boolean;
  paymentMethod?: string;
  nextBillingDate?: string;
}

// ============================================================================
// Error & Exception Types
// ============================================================================

/**
 * Authentication error response
 */
export interface AuthErrorResponse {
  code: string;
  message: string;
  details?: Record<string, any>;
  timestamp: string;
  requestId: string;
}

/**
 * Validation error
 */
export interface ValidationError {
  field: string;
  message: string;
  code: string;
  value?: any;
}

// ============================================================================
// Form Types
// ============================================================================

/**
 * Login form values
 */
export interface LoginFormValues {
  email: string;
  password: string;
  rememberMe: boolean;
}

/**
 * Register form values
 */
export interface RegisterFormValues {
  email: string;
  password: string;
  confirmPassword: string;
  firstName: string;
  lastName: string;
  acceptTerms: boolean;
  acceptPrivacy: boolean;
}

/**
 * Password reset form values
 */
export interface PasswordResetFormValues {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
}

/**
 * Profile form values
 */
export interface ProfileFormValues {
  firstName: string;
  lastName: string;
  email: string;
  phone?: string;
  company?: string;
  jobTitle?: string;
  location?: string;
  bio?: string;
  website?: string;
}

// ============================================================================
// API Integration Types
// ============================================================================

/**
 * Auth context value
 */
export interface AuthContextValue {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  register: (data: RegisterRequest) => Promise<void>;
  updateProfile: (data: Partial<User>) => Promise<void>;
  changePassword: (data: CredentialChangeRequest) => Promise<void>;
}

// ============================================================================
// Enums for Auth Types
// ============================================================================

/**
 * User roles enumeration
 */
export enum UserRoleEnum {
  Admin = 'admin',
  Manager = 'manager',
  User = 'user',
  Viewer = 'viewer',
  Guest = 'guest',
}

/**
 * Account status enumeration
 */
export enum AccountStatusEnum {
  Active = 'active',
  Inactive = 'inactive',
  Suspended = 'suspended',
  Deleted = 'deleted',
  Locked = 'locked',
}

/**
 * MFA method enumeration
 */
export enum MFAMethodEnum {
  Email = 'email',
  Authenticator = 'authenticator',
  SMS = 'sms',
  WebAuthn = 'webauthn',
}

/**
 * OAuth provider enumeration
 */
export enum OAuthProviderEnum {
  Google = 'google',
  GitHub = 'github',
  Facebook = 'facebook',
  Microsoft = 'microsoft',
  Apple = 'apple',
}

/**
 * Subscription status enumeration
 */
export enum SubscriptionStatusEnum {
  Active = 'active',
  Inactive = 'inactive',
  Cancelled = 'cancelled',
  Suspended = 'suspended',
  Expired = 'expired',
}

/**
 * Theme enumeration
 */
export enum ThemeEnum {
  Light = 'light',
  Dark = 'dark',
  Auto = 'auto',
}

// ============================================================================
// Guard & Predicate Types
// ============================================================================

/**
 * Type guard for User
 */
export function isUser(obj: any): obj is User {
  return (
    obj &&
    typeof obj === 'object' &&
    typeof obj.id === 'string' &&
    typeof obj.email === 'string' &&
    typeof obj.firstName === 'string' &&
    typeof obj.lastName === 'string'
  );
}

/**
 * Type guard for LoginResponse
 */
export function isLoginResponse(obj: any): obj is LoginResponse {
  return (
    obj &&
    typeof obj === 'object' &&
    isUser(obj.user) &&
    typeof obj.accessToken === 'string' &&
    typeof obj.refreshToken === 'string'
  );
}

/**
 * Type guard for authenticated user
 */
export function isAuthenticated(user: User | null): user is User {
  return user !== null && user.id !== undefined;
}
