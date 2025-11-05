// src/pages/LoginPage.tsx

/**
 * LoginPage - Secure authentication page with modern UI and redirect prevention
 * ✅ ENHANCED: Memoization, redirect loop prevention, performance optimized
 */

import { memo, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useNavigate, useLocation, Link } from 'react-router-dom';
import {
  Eye,
  EyeOff,
  Mail,
  Lock,
  AlertCircle,
  CheckCircle,
  Loader,
  Shield,
  ArrowRight,
  Info,
} from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import * as validators from '@/utils/validators';

// ============================================================================
// Type Definitions
// ============================================================================

interface LocationState {
  from?: {
    pathname: string;
  };
  redirectReason?: string;
  successMessage?: string;
}

interface PasswordStrength {
  score: number;
  label: string;
  color: string;
  percentage: number;
}

interface FormErrors {
  email?: string;
  password?: string;
}

interface FormTouched {
  email?: boolean;
  password?: boolean;
}

// ============================================================================
// Memoized Sub-Components
// ============================================================================

/**
 * ✅ FIXED: Memoized feature item component
 */
const FeatureItem = memo<{ icon: string; label: string; desc: string }>(
  ({ icon, label, desc }) => (
    <div className="flex items-start gap-4 group">
      <span className="text-3xl group-hover:scale-110 transition-transform">
        {icon}
      </span>
      <div>
        <h3 className="text-slate-200 font-medium">{label}</h3>
        <p className="text-sm text-slate-400">{desc}</p>
      </div>
    </div>
  )
);

FeatureItem.displayName = 'FeatureItem';

/**
 * ✅ FIXED: Memoized stat item component
 */
const StatItem = memo<{ number: string; label: string }>(
  ({ number, label }) => (
    <div>
      <p className="text-2xl font-bold text-primary-400">{number}</p>
      <p className="text-xs text-slate-400">{label}</p>
    </div>
  )
);

StatItem.displayName = 'StatItem';

/**
 * ✅ FIXED: Memoized password strength indicator
 */
interface PasswordStrengthIndicatorProps {
  password: string;
  strength: PasswordStrength;
}

const PasswordStrengthIndicator = memo<PasswordStrengthIndicatorProps>(
  ({ password, strength }) => (
    <div className="space-y-2 mt-3 p-3 bg-slate-700/30 rounded-lg border border-slate-600">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-wide">
          Password Strength
        </span>
        <span
          className={`text-xs font-medium ${strength.color.replace(
            'bg-',
            'text-'
          )}`}
        >
          {strength.label}
        </span>
      </div>

      <div className="w-full bg-slate-600 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full ${strength.color} transition-all duration-300`}
          style={{ width: `${strength.percentage}%` }}
        ></div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs mt-2">
        {[
          { label: '8+ characters', met: password.length >= 8 },
          { label: 'Uppercase letter', met: /[A-Z]/.test(password) },
          { label: 'Lowercase letter', met: /[a-z]/.test(password) },
          { label: 'Number', met: /\d/.test(password) },
        ].map((check) => (
          <div key={check.label} className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded-full ${
                check.met ? 'bg-green-500' : 'bg-slate-600'
              }`}
            ></div>
            <span className={check.met ? 'text-slate-300' : 'text-slate-500'}>
              {check.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
);

PasswordStrengthIndicator.displayName = 'PasswordStrengthIndicator';

/**
 * ✅ FIXED: Memoized demo credentials section
 */
interface DemoCredentialsProps {
  isShowing: boolean;
  isSubmitting: boolean;
  isAccountLocked: boolean;
  onUseDemoCredentials: () => void;
}

const DemoCredentialsSection = memo<DemoCredentialsProps>(
  ({ isShowing, isSubmitting, isAccountLocked, onUseDemoCredentials }) => {
    if (!isShowing) return null;

    return (
      <div className="mt-6 p-4 bg-gradient-to-br from-primary-900/30 to-primary-900/10 border border-primary-700/50 rounded-lg space-y-4">
        <div>
          <label className="text-xs font-medium text-slate-400 uppercase tracking-wide block mb-1">
            Demo Email
          </label>
          <div className="flex items-center gap-2">
            <code className="text-sm text-primary-300 font-mono bg-slate-900/50 px-3 py-2 rounded flex-1 truncate">
              demo@dataviz.com
            </code>
            <button
              onClick={() => navigator.clipboard.writeText('demo@dataviz.com')}
              className="p-2 hover:bg-slate-700 rounded transition-colors"
              title="Copy"
            >
              📋
            </button>
          </div>
        </div>

        <div>
          <label className="text-xs font-medium text-slate-400 uppercase tracking-wide block mb-1">
            Demo Password
          </label>
          <div className="flex items-center gap-2">
            <code className="text-sm text-primary-300 font-mono bg-slate-900/50 px-3 py-2 rounded flex-1">
              demo123456
            </code>
            <button
              onClick={() => navigator.clipboard.writeText('demo123456')}
              className="p-2 hover:bg-slate-700 rounded transition-colors"
              title="Copy"
            >
              📋
            </button>
          </div>
        </div>

        <button
          onClick={onUseDemoCredentials}
          className="w-full py-2 px-4 rounded-lg bg-primary-600 hover:bg-primary-700 text-white transition-colors text-sm font-medium"
          disabled={isSubmitting || isAccountLocked}
        >
          Use Demo Credentials
        </button>
      </div>
    );
  }
);

DemoCredentialsSection.displayName = 'DemoCredentialsSection';

// ============================================================================
// Main Component
// ============================================================================

/**
 * ✅ FULLY OPTIMIZED: Memoized login page component
 * Prevents redirect loops and performance issues
 */
const LoginPage = memo(() => {
  const navigate = useNavigate();
  const location = useLocation();
  const { isAuthenticated, isCheckingAuth, isLoading, login, clearError, error: authError } = useAuth();

  // ============================================================================
  // State Management
  // ============================================================================

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showDemoCredentials, setShowDemoCredentials] = useState(false);
  const [focusedField, setFocusedField] = useState<'email' | 'password' | null>(null);
  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<FormTouched>({});
  const [loginAttempts, setLoginAttempts] = useState(0);
  const [isAccountLocked, setIsAccountLocked] = useState(false);

  // ✅ FIXED: Use refs to prevent redirect loops
  const redirectHandledRef = useRef(false);
  const initializeRef = useRef(false);

  const locationState = location.state as LocationState;
  const redirectReason = locationState?.redirectReason;

  // ============================================================================
  // Redirect Prevention Effects
  // ============================================================================

  /**
   * ✅ FIXED: Prevent redirect loop with proper checks
   */
  useEffect(() => {
    // Only redirect if not already checking auth and authenticated
    if (isCheckingAuth || !isAuthenticated || redirectHandledRef.current) {
      return;
    }

    // Mark as handled to prevent double navigation
    redirectHandledRef.current = true;

    const from = locationState?.from?.pathname || '/dashboard';
    navigate(from, { replace: true });
  }, [isCheckingAuth, isAuthenticated, navigate, locationState]);

  /**
   * ✅ FIXED: Initialize component only once
   */
  useEffect(() => {
    if (initializeRef.current) return;
    initializeRef.current = true;

    // Load saved email preference
    const savedEmail = localStorage.getItem('auth_email_remembered');
    const savedRememberMe = localStorage.getItem('auth_remember_me');

    if (savedEmail && savedRememberMe === 'true') {
      setEmail(savedEmail);
      setRememberMe(true);
    }

    // Clear auth errors on mount
    clearError();
  }, [clearError]);

  // ============================================================================
  // Memoized Computations
  // ============================================================================

  /**
   * ✅ FIXED: Memoized password strength calculation
   */
  const passwordStrength = useMemo((): PasswordStrength => {
    if (!password) {
      return { score: 0, label: '', color: '', percentage: 0 };
    }

    let score = 0;
    const checks = {
      hasMinLength: password.length >= 8,
      hasUpperCase: /[A-Z]/.test(password),
      hasLowerCase: /[a-z]/.test(password),
      hasNumbers: /\d/.test(password),
      hasSpecialChar: /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password),
    };

    Object.values(checks).forEach((check) => {
      if (check) score++;
    });

    const strengthLevels: PasswordStrength[] = [
      { score: 0, label: '', color: '', percentage: 0 },
      { score: 1, label: 'Weak', color: 'bg-red-500', percentage: 20 },
      { score: 2, label: 'Fair', color: 'bg-yellow-500', percentage: 40 },
      { score: 3, label: 'Good', color: 'bg-blue-500', percentage: 60 },
      { score: 4, label: 'Strong', color: 'bg-green-500', percentage: 80 },
      { score: 5, label: 'Very Strong', color: 'bg-emerald-500', percentage: 100 },
    ];

    return strengthLevels[score] || strengthLevels[0];
  }, [password]);

  /**
   * ✅ FIXED: Memoized features data
   */
  const features = useMemo(
    () => [
      { icon: '📊', label: 'Automated EDA Analysis', desc: 'Real-time statistical insights' },
      { icon: '📈', label: 'Interactive Visualizations', desc: 'Drag-and-drop chart builder' },
      { icon: '🤖', label: 'AI-Powered Insights', desc: 'Smart pattern detection' },
      { icon: '⚡', label: 'Real-time Processing', desc: 'Instant data computation' },
    ],
    []
  );

  /**
   * ✅ FIXED: Memoized stats data
   */
  const stats = useMemo(
    () => [
      { number: '10K+', label: 'Active Users' },
      { number: '500K+', label: 'Datasets Processed' },
      { number: '99.9%', label: 'Uptime' },
    ],
    []
  );

  /**
   * ✅ FIXED: Memoized form validation state
   */
  const isFormValid = useMemo(
    () => !errors.email && !errors.password && email && password,
    [errors, email, password]
  );

  // ============================================================================
  // Validation Functions (Memoized)
  // ============================================================================

  /**
   * ✅ FIXED: Memoized email validation
   */
  const validateEmailField = useCallback((value: string): string => {
    if (!value.trim()) return 'Email is required';
    if (!validators.validateEmail(value)) return 'Please enter a valid email address';
    return '';
  }, []);

  /**
   * ✅ FIXED: Memoized password validation
   */
  const validatePasswordField = useCallback((value: string): string => {
    if (!value) return 'Password is required';
    if (!validators.validatePassword(value)) return 'Password must be at least 6 characters';
    return '';
  }, []);

  // ============================================================================
  // Event Handlers (Memoized)
  // ============================================================================

  /**
   * ✅ FIXED: Memoized email change handler
   */
  const handleEmailChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setEmail(value);

      if (touched.email) {
        const error = validateEmailField(value);
        setErrors((prev) => ({ ...prev, email: error }));
      }
    },
    [touched.email, validateEmailField]
  );

  /**
   * ✅ FIXED: Memoized password change handler
   */
  const handlePasswordChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value;
      setPassword(value);

      if (touched.password) {
        const error = validatePasswordField(value);
        setErrors((prev) => ({ ...prev, password: error }));
      }
    },
    [touched.password, validatePasswordField]
  );

  /**
   * ✅ FIXED: Memoized field blur handler
   */
  const handleFieldBlur = useCallback(
    (field: 'email' | 'password') => {
      setTouched((prev) => ({ ...prev, [field]: true }));
      setFocusedField(null);

      if (field === 'email') {
        const error = validateEmailField(email);
        setErrors((prev) => ({ ...prev, email: error }));
      } else if (field === 'password') {
        const error = validatePasswordField(password);
        setErrors((prev) => ({ ...prev, password: error }));
      }
    },
    [email, password, validateEmailField, validatePasswordField]
  );

  /**
   * ✅ FIXED: Memoized form submission handler
   */
  const handleSubmit = useCallback(
    async (e: React.FormEvent<HTMLFormElement>) => {
      e.preventDefault();

      if (isAccountLocked || isSubmitting) return;

      const emailError = validateEmailField(email);
      const passwordError = validatePasswordField(password);

      setTouched({ email: true, password: true });

      if (emailError || passwordError) {
        setErrors({ email: emailError, password: passwordError });
        return;
      }

      setIsSubmitting(true);

      try {
        await login({
          email,
          password,
          rememberMe,
        });

        // Save remember me preference
        if (rememberMe) {
          localStorage.setItem('auth_email_remembered', email);
          localStorage.setItem('auth_remember_me', 'true');
        } else {
          localStorage.removeItem('auth_email_remembered');
          localStorage.setItem('auth_remember_me', 'false');
        }

        setLoginAttempts(0);
        setErrors({});
      } catch (error) {
        const newAttempts = loginAttempts + 1;
        setLoginAttempts(newAttempts);

        if (newAttempts >= 5) {
          setIsAccountLocked(true);
          setErrors({
            email: 'Too many failed login attempts. Account temporarily locked.',
          });

          setTimeout(() => {
            setIsAccountLocked(false);
            setLoginAttempts(0);
          }, 15 * 60 * 1000);
        } else {
          const errorMessage =
            error instanceof Error ? error.message : 'Invalid email or password';
          setErrors({ email: errorMessage });
        }
      } finally {
        setIsSubmitting(false);
      }
    },
    [email, password, rememberMe, isAccountLocked, isSubmitting, loginAttempts, login, validateEmailField, validatePasswordField]
  );

  /**
   * ✅ FIXED: Memoized demo credentials handler
   */
  const handleUseDemoCredentials = useCallback(() => {
    setEmail('demo@dataviz.com');
    setPassword('demo123456');
    setTouched({ email: true, password: true });
    setShowDemoCredentials(false);
  }, []);

  // ============================================================================
  // Render - Loading/Checking Auth State
  // ============================================================================

  if (isCheckingAuth || isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin">
            <div className="h-12 w-12 border-4 border-primary-500 border-t-primary-600 rounded-full"></div>
          </div>
          <p className="text-slate-400 text-sm">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  // ============================================================================
  // Render - Main Content
  // ============================================================================

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 overflow-hidden">
      {/* Left Section - Hero */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12 relative overflow-hidden">
        {/* Animated background blobs */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-20 left-20 w-64 h-64 bg-primary-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob"></div>
          <div className="absolute top-40 right-20 w-64 h-64 bg-secondary-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-2000"></div>
          <div className="absolute -bottom-8 left-1/2 w-64 h-64 bg-accent-500 rounded-full mix-blend-multiply filter blur-3xl animate-blob animation-delay-4000"></div>
        </div>

        {/* Content */}
        <div className="relative z-10">
          {/* Logo */}
          <div className="flex items-center gap-3 mb-12">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-400 to-primary-600 rounded-lg flex items-center justify-center">
              <Shield className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">DataViz Pro</span>
          </div>

          {/* Hero Content */}
          <h1 className="text-5xl font-bold text-white mb-6 leading-tight">
            Transform Your Data Into Actionable Insights
          </h1>
          <p className="text-lg text-slate-300 mb-8 leading-relaxed">
            Upload your datasets, generate automatic EDA reports, create stunning visualizations,
            and uncover patterns that drive business decisions.
          </p>

          {/* Feature List */}
          <div className="space-y-4 mb-8">
            {features.map((feature) => (
              <FeatureItem
                key={feature.label}
                icon={feature.icon}
                label={feature.label}
                desc={feature.desc}
              />
            ))}
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-6 pt-8 border-t border-slate-700">
            {stats.map((stat) => (
              <StatItem key={stat.label} number={stat.number} label={stat.label} />
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="relative z-10 pt-8 border-t border-slate-700">
          <p className="text-sm text-slate-400">
            ✨ Join thousands of data professionals using DataViz Pro
          </p>
        </div>
      </div>

      {/* Right Section - Login Form */}
      <div className="w-full lg:w-1/2 flex flex-col justify-center items-center p-6 sm:p-8 relative">
        {/* Background gradient */}
        <div className="absolute inset-0 opacity-5">
          <div className="absolute inset-0 bg-gradient-to-b from-primary-500 to-transparent"></div>
        </div>

        <div className="w-full max-w-md relative z-10">
          {/* Mobile Logo */}
          <div className="lg:hidden mb-8 text-center">
            <div className="w-12 h-12 bg-gradient-to-br from-primary-400 to-primary-600 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white">DataViz Pro</h1>
            <p className="text-slate-400 text-sm mt-2">Analyze. Visualize. Understand.</p>
          </div>

          {/* Session Expired Alert */}
          {redirectReason && (
            <div
              className={`mb-6 p-4 rounded-lg flex items-start gap-3 border backdrop-blur-sm ${
                redirectReason === 'session-expired'
                  ? 'bg-yellow-900/30 border-yellow-700 text-yellow-200'
                  : 'bg-red-900/30 border-red-700 text-red-200'
              }`}
            >
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-sm font-medium">
                  {redirectReason === 'session-expired'
                    ? 'Your session has expired. Please log in again to continue.'
                    : 'You need to log in to access that resource.'}
                </p>
              </div>
            </div>
          )}

          {/* Success Message */}
          {locationState?.successMessage && (
            <div className="mb-6 p-4 bg-green-900/30 border border-green-700 rounded-lg flex items-start gap-3">
              <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-green-200">{locationState.successMessage}</p>
            </div>
          )}

          {/* Auth Error Alert */}
          {authError && (
            <div className="mb-6 p-4 bg-red-900/30 border border-red-700 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-red-200">{authError}</p>
            </div>
          )}

          {/* Form Header */}
          <div className="mb-8">
            <h2 className="text-3xl font-bold text-white mb-2">Welcome Back</h2>
            <p className="text-slate-400">Sign in to your account to continue exploring</p>
          </div>

          {/* Main Form */}
          <form onSubmit={handleSubmit} className="space-y-6" noValidate>
            {/* Email Field */}
            <div className="space-y-2">
              <label htmlFor="email" className="block text-sm font-medium text-slate-200">
                Email Address
              </label>
              <div className="relative group">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500 group-focus-within:text-primary-400 transition-colors" />
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={handleEmailChange}
                  onBlur={() => handleFieldBlur('email')}
                  onFocus={() => setFocusedField('email')}
                  placeholder="you@example.com"
                  className={`w-full pl-10 pr-4 py-3 rounded-lg bg-slate-700/50 border transition-all duration-200 outline-none backdrop-blur-sm ${
                    errors.email && touched.email
                      ? 'border-red-500 focus:border-red-400 focus:ring-2 focus:ring-red-500/20'
                      : focusedField === 'email'
                      ? 'border-primary-400 focus:border-primary-400 focus:ring-2 focus:ring-primary-500/20'
                      : 'border-slate-600 focus:border-primary-400'
                  } text-slate-100 placeholder-slate-500`}
                  disabled={isSubmitting || isAccountLocked}
                />
              </div>
              {errors.email && touched.email && (
                <p className="text-sm text-red-400 flex items-center gap-2 mt-1">
                  <AlertCircle className="w-4 h-4" />
                  {errors.email}
                </p>
              )}
            </div>

            {/* Password Field */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <label htmlFor="password" className="block text-sm font-medium text-slate-200">
                  Password
                </label>
                {password && (
                  <span className="text-xs text-slate-400">
                    Strength: {passwordStrength.label}
                  </span>
                )}
              </div>
              <div className="relative group">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500 group-focus-within:text-primary-400 transition-colors" />
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={handlePasswordChange}
                  onBlur={() => handleFieldBlur('password')}
                  onFocus={() => setFocusedField('password')}
                  placeholder="••••••••"
                  className={`w-full pl-10 pr-12 py-3 rounded-lg bg-slate-700/50 border transition-all duration-200 outline-none backdrop-blur-sm ${
                    errors.password && touched.password
                      ? 'border-red-500 focus:border-red-400 focus:ring-2 focus:ring-red-500/20'
                      : focusedField === 'password'
                      ? 'border-primary-400 focus:border-primary-400 focus:ring-2 focus:ring-primary-500/20'
                      : 'border-slate-600 focus:border-primary-400'
                  } text-slate-100 placeholder-slate-500`}
                  disabled={isSubmitting || isAccountLocked}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                  disabled={isSubmitting}
                >
                  {showPassword ? (
                    <EyeOff className="w-5 h-5" />
                  ) : (
                    <Eye className="w-5 h-5" />
                  )}
                </button>
              </div>

              {/* Password Strength Indicator */}
              {password && (
                <PasswordStrengthIndicator
                  password={password}
                  strength={passwordStrength}
                />
              )}

              {errors.password && touched.password && (
                <p className="text-sm text-red-400 flex items-center gap-2 mt-1">
                  <AlertCircle className="w-4 h-4" />
                  {errors.password}
                </p>
              )}
            </div>

            {/* Remember Me & Forgot Password */}
            <div className="flex items-center justify-between pt-2">
              <label className="flex items-center gap-2 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(e) => setRememberMe(e.target.checked)}
                  className="w-4 h-4 rounded border-slate-600 bg-slate-700 text-primary-500 cursor-pointer"
                  disabled={isSubmitting || isAccountLocked}
                />
                <span className="text-sm text-slate-300 group-hover:text-slate-200 transition-colors">
                  Remember me
                </span>
              </label>
              <Link
                to="/forgot-password"
                className="text-sm text-primary-400 hover:text-primary-300 transition-colors"
              >
                Forgot password?
              </Link>
            </div>

            {/* Account Locked Warning */}
            {isAccountLocked && (
              <div className="p-4 bg-red-900/30 border border-red-700 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-200">
                    Account Temporarily Locked
                  </p>
                  <p className="text-xs text-red-300 mt-1">
                    Too many failed login attempts. Please try again later.
                  </p>
                </div>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting || isAccountLocked || !isFormValid}
              className={`w-full py-3 px-4 rounded-lg font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
                isFormValid && !isSubmitting && !isAccountLocked
                  ? 'bg-gradient-to-r from-primary-500 to-primary-600 text-white hover:shadow-lg hover:shadow-primary-500/50 active:scale-95'
                  : 'bg-slate-700 text-slate-400 cursor-not-allowed'
              }`}
            >
              {isSubmitting ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                <>
                  Sign In
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="relative my-8">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-700"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-3 bg-slate-800 text-slate-400">or explore as guest</span>
            </div>
          </div>

          {/* Demo Credentials Button */}
          <button
            onClick={() => setShowDemoCredentials(!showDemoCredentials)}
            className="w-full py-3 px-4 rounded-lg border border-slate-600 text-slate-300 hover:bg-slate-700/50 hover:border-slate-500 transition-all duration-200 flex items-center justify-center gap-2"
            disabled={isSubmitting || isAccountLocked}
          >
            <Info className="w-4 h-4" />
            Try Demo Account
          </button>

          {/* Demo Credentials Section */}
          <DemoCredentialsSection
            isShowing={showDemoCredentials}
            isSubmitting={isSubmitting}
            isAccountLocked={isAccountLocked}
            onUseDemoCredentials={handleUseDemoCredentials}
          />

          {/* Sign Up Link */}
          <div className="mt-8 text-center space-y-4 pt-6 border-t border-slate-700">
            <p className="text-slate-400 text-sm">
              New to DataViz Pro?{' '}
              <Link
                to="/register"
                className="text-primary-400 hover:text-primary-300 font-medium transition-colors"
              >
                Create an account
              </Link>
            </p>
            <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
              <Link to="/privacy" className="hover:text-slate-400 transition-colors">
                Privacy Policy
              </Link>
              <span>•</span>
              <Link to="/terms" className="hover:text-slate-400 transition-colors">
                Terms of Service
              </Link>
              <span>•</span>
              <Link to="/contact" className="hover:text-slate-400 transition-colors">
                Support
              </Link>
            </div>
          </div>

          {/* Security Badge */}
          <div className="mt-6 p-3 bg-slate-700/30 rounded-lg flex items-center justify-center gap-2 text-xs text-slate-400">
            <Shield className="w-4 h-4 text-green-400" />
            <span>SSL Encrypted • GDPR Compliant • 2FA Available</span>
          </div>
        </div>
      </div>
    </div>
  );
});

LoginPage.displayName = 'LoginPage';

export default LoginPage;
