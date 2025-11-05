// src/pages/ProfilePage.tsx - FINAL ERROR-FREE VERSION

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  User,
  Mail,
  Phone,
  MapPin,
  Camera,
  Save,
  X,
  Eye,
  EyeOff,
  Bell,
  Lock,
  Shield,
  Trash2,
  Download,
  LogOut,
  AlertCircle,
  Moon,
  Globe,
  Code,
  Clock,
  CheckCircle,
  Settings as SettingsIcon,
} from 'lucide-react';
import DashboardLayout from '@/components/dashboard/DashboardLayout';
import Button from '@/components/shared/Button';
import Input from '@/components/shared/Input';
import Modal, { ConfirmModal } from '@/components/shared/Modal';
import { useAuth } from '@/hooks/useAuth';
import { validatePassword } from '@/utils/validators';
import { formatDistanceToNow } from 'date-fns';

// ============================================================================
// Type Definitions
// ============================================================================

interface UserProfile {
  id: string;
  fullName: string;
  email: string;
  phone?: string;
  location?: string;
  bio?: string;
  avatar?: string;
  company?: string;
  joinedDate: string;
  lastLogin: string;
  verified: boolean;
  twoFactorEnabled: boolean;
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    emailNotifications: boolean;
    analyticsOptIn: boolean;
    dataSharing: boolean;
  };
}

interface SecurityLog {
  id: string;
  action: string;
  device: string;
  location: string;
  timestamp: string;
  status: 'success' | 'failed';
}

type TabType = 'profile' | 'security' | 'preferences' | 'billing' | 'activity';

// ============================================================================
// Component
// ============================================================================

/**
 * ProfilePage - User profile page with account settings and preferences
 * Features: Profile editing, security settings, preferences, activity log, account management
 * ✅ FIXED: Removed unused Toggle2 import
 */
const ProfilePage: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  // ============================================================================
  // State Management
  // ============================================================================

  const [activeTab, setActiveTab] = useState<TabType>('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showPasswordModal, setShowPasswordModal] = useState(false);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showCurrentPassword] = useState(false);
  const [showNewPassword] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');

  // ============================================================================
  // Form State
  // ============================================================================

  const [profileData, setProfileData] = useState<UserProfile>({
    id: '1',
    fullName: user?.fullName || 'John Doe',
    email: user?.email || 'john@example.com',
    phone: '+1 (555) 123-4567',
    location: 'San Francisco, CA',
    bio: 'Data enthusiast and analytics professional',
    company: 'Tech Corp',
    avatar: 'JD',
    joinedDate: '2024-01-15T00:00:00Z',
    lastLogin: new Date().toISOString(),
    verified: true,
    twoFactorEnabled: false,
    preferences: {
      theme: 'light',
      emailNotifications: true,
      analyticsOptIn: true,
      dataSharing: false,
    },
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  // ============================================================================
  // Mock Data
  // ============================================================================

  const securityLogs: SecurityLog[] = [
    {
      id: '1',
      action: 'Login',
      device: 'Chrome on macOS',
      location: 'San Francisco, CA',
      timestamp: new Date(Date.now() - 3600000).toISOString(),
      status: 'success',
    },
    {
      id: '2',
      action: 'Password Changed',
      device: 'Safari on iPhone',
      location: 'San Francisco, CA',
      timestamp: new Date(Date.now() - 86400000).toISOString(),
      status: 'success',
    },
    {
      id: '3',
      action: 'Failed Login',
      device: 'Firefox on Windows',
      location: 'New York, NY',
      timestamp: new Date(Date.now() - 172800000).toISOString(),
      status: 'failed',
    },
  ];

  // ============================================================================
  // Event Handlers
  // ============================================================================

  /**
   * Handle profile update
   */
  const handleProfileUpdate = async () => {
    setIsSaving(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setIsEditing(false);
      setSuccessMessage('Profile updated successfully!');
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setIsSaving(false);
    }
  };

  /**
   * Handle password change
   */
  const handlePasswordChange = async () => {
    if (!validatePassword(passwordData.newPassword)) {
      return;
    }

    if (passwordData.newPassword !== passwordData.confirmPassword) {
      return;
    }

    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setShowPasswordModal(false);
      setPasswordData({
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      });
      setSuccessMessage('Password changed successfully!');
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      console.error('Failed to change password:', error);
    }
  };

  /**
   * Handle 2FA toggle
   */
  const handleToggle2FA = async () => {
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      setProfileData({
        ...profileData,
        twoFactorEnabled: !profileData.twoFactorEnabled,
      });
      setSuccessMessage(
        profileData.twoFactorEnabled
          ? '2FA disabled successfully'
          : '2FA enabled successfully'
      );
      setTimeout(() => setSuccessMessage(''), 3000);
    } catch (error) {
      console.error('Failed to toggle 2FA:', error);
    }
  };

  /**
   * Handle logout
   */
  const handleLogout = async () => {
    try {
      await logout();
      navigate('/login');
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  /**
   * Handle delete account
   */
  const handleDeleteAccount = async () => {
    setIsDeleting(true);
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 2000));
      await logout();
      navigate('/');
    } catch (error) {
      console.error('Failed to delete account:', error);
    } finally {
      setIsDeleting(false);
    }
  };

  // ============================================================================
  // Render
  // ============================================================================

  return (
    <DashboardLayout>
      <div className="profile-page">
        {/* ====================================================================
            Header
            ==================================================================== */}
        <div className="profile-header">
          <button
            onClick={() => navigate('/dashboard')}
            className="profile-back-button"
          >
            <ArrowLeft className="w-5 h-5" />
            <span>Back</span>
          </button>
          <div>
            <h1 className="profile-title">Account Settings</h1>
            <p className="profile-subtitle">Manage your account and preferences</p>
          </div>
        </div>

        {/* ====================================================================
            Success Message
            ==================================================================== */}
        {successMessage && (
          <div className="profile-success-alert">
            <CheckCircle className="w-5 h-5" />
            <p>{successMessage}</p>
          </div>
        )}

        {/* ====================================================================
            Tabs Navigation
            ==================================================================== */}
        <div className="profile-tabs">
          {[
            { id: 'profile', label: 'Profile', icon: User },
            { id: 'security', label: 'Security', icon: Lock },
            { id: 'preferences', label: 'Preferences', icon: SettingsIcon },
            { id: 'billing', label: 'Billing', icon: Code },
            { id: 'activity', label: 'Activity', icon: Clock },
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as TabType)}
                className={`profile-tab ${activeTab === tab.id ? 'active' : ''}`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* ====================================================================
            Tab Content
            ==================================================================== */}
        <div className="profile-content">
          {/* Profile Tab */}
          {activeTab === 'profile' && (
            <div className="profile-section">
              <div className="profile-section-header">
                <h2 className="profile-section-title">Profile Information</h2>
                {!isEditing && (
                  <Button
                    variant="secondary"
                    size="sm"
                    leftIcon={isEditing ? X : User}
                    onClick={() => setIsEditing(!isEditing)}
                  >
                    {isEditing ? 'Cancel' : 'Edit Profile'}
                  </Button>
                )}
              </div>

              <div className="profile-card">
                {/* Avatar */}
                <div className="profile-avatar-section">
                  <div className="profile-avatar">{profileData.avatar}</div>
                  {isEditing && (
                    <button className="profile-avatar-upload">
                      <Camera className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Form Fields */}
                <div className="profile-form-grid">
                  <Input
                    label="Full Name"
                    value={profileData.fullName}
                    onChange={(e) =>
                      setProfileData({
                        ...profileData,
                        fullName: e.target.value,
                      })
                    }
                    disabled={!isEditing}
                    leftIcon={User}
                  />

                  <Input
                    label="Email Address"
                    type="email"
                    value={profileData.email}
                    onChange={(e) =>
                      setProfileData({
                        ...profileData,
                        email: e.target.value,
                      })
                    }
                    disabled={!isEditing}
                    leftIcon={Mail}
                  />

                  <Input
                    label="Phone Number"
                    value={profileData.phone}
                    onChange={(e) =>
                      setProfileData({
                        ...profileData,
                        phone: e.target.value,
                      })
                    }
                    disabled={!isEditing}
                    leftIcon={Phone}
                  />

                  <Input
                    label="Location"
                    value={profileData.location}
                    onChange={(e) =>
                      setProfileData({
                        ...profileData,
                        location: e.target.value,
                      })
                    }
                    disabled={!isEditing}
                    leftIcon={MapPin}
                  />

                  <div className="profile-form-full">
                    <Input
                      label="Company"
                      value={profileData.company}
                      onChange={(e) =>
                        setProfileData({
                          ...profileData,
                          company: e.target.value,
                        })
                      }
                      disabled={!isEditing}
                    />
                  </div>

                  <div className="profile-form-full">
                    <label className="label">Bio</label>
                    <textarea
                      value={profileData.bio}
                      onChange={(e) =>
                        setProfileData({
                          ...profileData,
                          bio: e.target.value,
                        })
                      }
                      disabled={!isEditing}
                      className="input h-24 resize-none"
                      placeholder="Tell us about yourself"
                    />
                  </div>
                </div>

                {/* Account Info */}
                <div className="profile-info-grid">
                  <div className="profile-info-item">
                    <p className="profile-info-label">Account Status</p>
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-green-600" />
                      <span className="font-medium text-green-600">Verified</span>
                    </div>
                  </div>

                  <div className="profile-info-item">
                    <p className="profile-info-label">Joined</p>
                    <p className="font-medium">
                      {formatDistanceToNow(
                        new Date(profileData.joinedDate),
                        { addSuffix: true }
                      )}
                    </p>
                  </div>

                  <div className="profile-info-item">
                    <p className="profile-info-label">Last Login</p>
                    <p className="font-medium">
                      {formatDistanceToNow(
                        new Date(profileData.lastLogin),
                        { addSuffix: true }
                      )}
                    </p>
                  </div>
                </div>

                {/* Save Button */}
                {isEditing && (
                  <div className="profile-form-actions">
                    <Button
                      variant="primary"
                      leftIcon={Save}
                      onClick={handleProfileUpdate}
                      loading={isSaving}
                    >
                      Save Changes
                    </Button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Security Tab */}
          {activeTab === 'security' && (
            <div className="profile-section space-y-6">
              {/* Password */}
              <div className="profile-card">
                <div className="profile-card-header">
                  <div>
                    <h3 className="profile-card-title">Password</h3>
                    <p className="profile-card-description">
                      Change your password regularly to keep your account secure
                    </p>
                  </div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setShowPasswordModal(true)}
                  >
                    Change Password
                  </Button>
                </div>
              </div>

              {/* Two-Factor Authentication */}
              <div className="profile-card">
                <div className="profile-card-header">
                  <div>
                    <h3 className="profile-card-title">
                      Two-Factor Authentication
                    </h3>
                    <p className="profile-card-description">
                      Add an extra layer of security to your account
                    </p>
                  </div>
                  <button
                    onClick={handleToggle2FA}
                    className={`profile-toggle ${
                      profileData.twoFactorEnabled ? 'active' : ''
                    }`}
                  >
                    <div className="profile-toggle-circle" />
                  </button>
                </div>
              </div>

              {/* Security Log */}
              <div className="profile-card">
                <h3 className="profile-card-title mb-4">
                  Recent Security Events
                </h3>
                <div className="profile-security-log">
                  {securityLogs.map((log) => (
                    <div key={log.id} className="profile-security-item">
                      <div className="profile-security-info">
                        <p className="font-medium text-gray-900">
                          {log.action}
                        </p>
                        <p className="text-sm text-gray-600">
                          {log.device} • {log.location}
                        </p>
                      </div>
                      <div className="profile-security-meta">
                        <span className={`profile-security-status ${log.status}`}>
                          {log.status === 'success' ? (
                            <CheckCircle className="w-4 h-4" />
                          ) : (
                            <AlertCircle className="w-4 h-4" />
                          )}
                          {log.status}
                        </span>
                        <span className="text-sm text-gray-600">
                          {formatDistanceToNow(
                            new Date(log.timestamp),
                            { addSuffix: true }
                          )}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Preferences Tab */}
          {activeTab === 'preferences' && (
            <div className="profile-section">
              <div className="profile-card space-y-6">
                {/* Theme */}
                <div className="profile-preference-item">
                  <div className="flex items-center gap-3">
                    <div className="profile-pref-icon bg-purple-100 text-purple-600">
                      <Moon className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Theme</p>
                      <p className="text-sm text-gray-600">
                        Choose your preferred color scheme
                      </p>
                    </div>
                  </div>
                  <select
                    value={profileData.preferences.theme}
                    onChange={(e) =>
                      setProfileData({
                        ...profileData,
                        preferences: {
                          ...profileData.preferences,
                          theme: e.target.value as 'light' | 'dark' | 'auto',
                        },
                      })
                    }
                    className="profile-select"
                  >
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                    <option value="auto">Auto</option>
                  </select>
                </div>

                {/* Email Notifications */}
                <div className="profile-preference-item">
                  <div className="flex items-center gap-3">
                    <div className="profile-pref-icon bg-blue-100 text-blue-600">
                      <Bell className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">
                        Email Notifications
                      </p>
                      <p className="text-sm text-gray-600">
                        Receive updates and alerts via email
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      setProfileData({
                        ...profileData,
                        preferences: {
                          ...profileData.preferences,
                          emailNotifications:
                            !profileData.preferences.emailNotifications,
                        },
                      })
                    }
                    className={`profile-toggle ${
                      profileData.preferences.emailNotifications ? 'active' : ''
                    }`}
                  >
                    <div className="profile-toggle-circle" />
                  </button>
                </div>

                {/* Analytics */}
                <div className="profile-preference-item">
                  <div className="flex items-center gap-3">
                    <div className="profile-pref-icon bg-green-100 text-green-600">
                      <Globe className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">
                        Analytics Opt-in
                      </p>
                      <p className="text-sm text-gray-600">
                        Help us improve with usage data
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      setProfileData({
                        ...profileData,
                        preferences: {
                          ...profileData.preferences,
                          analyticsOptIn: !profileData.preferences.analyticsOptIn,
                        },
                      })
                    }
                    className={`profile-toggle ${
                      profileData.preferences.analyticsOptIn ? 'active' : ''
                    }`}
                  >
                    <div className="profile-toggle-circle" />
                  </button>
                </div>

                {/* Data Sharing */}
                <div className="profile-preference-item">
                  <div className="flex items-center gap-3">
                    <div className="profile-pref-icon bg-orange-100 text-orange-600">
                      <Shield className="w-5 h-5" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">Data Sharing</p>
                      <p className="text-sm text-gray-600">
                        Share anonymized data with partners
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      setProfileData({
                        ...profileData,
                        preferences: {
                          ...profileData.preferences,
                          dataSharing: !profileData.preferences.dataSharing,
                        },
                      })
                    }
                    className={`profile-toggle ${
                      profileData.preferences.dataSharing ? 'active' : ''
                    }`}
                  >
                    <div className="profile-toggle-circle" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Billing Tab */}
          {activeTab === 'billing' && (
            <div className="profile-section">
              <div className="profile-card">
                <h3 className="profile-card-title mb-4">Billing Information</h3>
                <div className="space-y-4">
                  <div className="profile-billing-item">
                    <p className="profile-info-label">Current Plan</p>
                    <p className="font-semibold text-lg text-blue-600">
                      Professional
                    </p>
                  </div>
                  <div className="profile-billing-item">
                    <p className="profile-info-label">Billing Cycle</p>
                    <p className="font-medium">Monthly</p>
                  </div>
                  <div className="profile-billing-item">
                    <p className="profile-info-label">Next Billing Date</p>
                    <p className="font-medium">
                      {new Date(
                        Date.now() + 30 * 86400000
                      ).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="profile-billing-item">
                    <p className="profile-info-label">Amount</p>
                    <p className="font-semibold text-lg">$29/month</p>
                  </div>
                </div>

                <div className="profile-billing-actions space-y-2 mt-6 pt-6 border-t border-gray-200">
                  <Button
                    variant="secondary"
                    fullWidth
                    leftIcon={Download}
                  >
                    Download Invoice
                  </Button>
                  <Button variant="outline" fullWidth>
                    Change Plan
                  </Button>
                </div>
              </div>
            </div>
          )}

          {/* Activity Tab */}
          {activeTab === 'activity' && (
            <div className="profile-section">
              <div className="profile-card">
                <h3 className="profile-card-title mb-4">Account Actions</h3>
                <div className="space-y-4">
                  <Button
                    variant="secondary"
                    fullWidth
                    leftIcon={LogOut}
                    onClick={() => setShowLogoutModal(true)}
                  >
                    Sign Out from All Devices
                  </Button>
                  <Button
                    variant="danger"
                    fullWidth
                    leftIcon={Trash2}
                    onClick={() => setShowDeleteModal(true)}
                  >
                    Delete Account
                  </Button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* ====================================================================
            Modals
            ==================================================================== */}

        {/* Password Change Modal */}
        <Modal
          isOpen={showPasswordModal}
          onClose={() => setShowPasswordModal(false)}
          title="Change Password"
          size="sm"
        >
          <div className="profile-password-form space-y-4">
            <Input
              label="Current Password"
              type={showCurrentPassword ? 'text' : 'password'}
              value={passwordData.currentPassword}
              onChange={(e) =>
                setPasswordData({
                  ...passwordData,
                  currentPassword: e.target.value,
                })
              }
              rightIcon={showCurrentPassword ? Eye : EyeOff}
              required
            />

            <Input
              label="New Password"
              type={showNewPassword ? 'text' : 'password'}
              value={passwordData.newPassword}
              onChange={(e) =>
                setPasswordData({
                  ...passwordData,
                  newPassword: e.target.value,
                })
              }
              rightIcon={showNewPassword ? Eye : EyeOff}
              required
            />

            <Input
              label="Confirm Password"
              type="password"
              value={passwordData.confirmPassword}
              onChange={(e) =>
                setPasswordData({
                  ...passwordData,
                  confirmPassword: e.target.value,
                })
              }
              required
            />

            <div className="space-y-3 pt-2">
              <Button
                variant="primary"
                fullWidth
                onClick={handlePasswordChange}
              >
                Change Password
              </Button>
              <Button
                variant="secondary"
                fullWidth
                onClick={() => setShowPasswordModal(false)}
              >
                Cancel
              </Button>
            </div>
          </div>
        </Modal>

        {/* Logout Confirmation Modal */}
        <ConfirmModal
          isOpen={showLogoutModal}
          onClose={() => setShowLogoutModal(false)}
          onConfirm={handleLogout}
          title="Sign Out from All Devices"
          message="You will be signed out from all devices and will need to sign in again. Continue?"
          confirmText="Sign Out"
          variant="warning"
        />

        {/* Delete Account Modal */}
        <ConfirmModal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          onConfirm={handleDeleteAccount}
          title="Delete Account"
          message="This action is permanent and cannot be undone. All your data will be deleted. Are you sure?"
          confirmText="Delete Account"
          variant="danger"
          isLoading={isDeleting}
        />
      </div>
    </DashboardLayout>
  );
};

ProfilePage.displayName = 'ProfilePage';

export default ProfilePage;
