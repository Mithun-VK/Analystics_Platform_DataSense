// src/components/auth/RegisterForm.tsx - COMPLETE FIXED VERSION

import { useState, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Mail,
  Lock,
  User,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Eye,
  EyeOff,
} from 'lucide-react';
import Button from '@/components/shared/Button';
import Input from '@/components/shared/Input';
import * as validators from '@/utils/validators';
import { uiStore } from '@/store/uiStore';
import axios from 'axios';

// ============================================================================
// Types
// ============================================================================

interface FormData {
  fullName: string;
  email: string;
  password: string;
  confirmPassword: string;
  agreeToTerms: boolean;
}

interface FormErrors {
  fullName?: string;
  email?: string;
  password?: string;
  confirmPassword?: string;
  agreeToTerms?: string;
  general?: string;
}

interface PasswordStrength {
  score: number;
  label: string;
  color: string;
  bgColor: string;
}

interface PasswordRequirements {
  minLength: boolean;
  lowercase: boolean;
  uppercase: boolean;
  number: boolean;
  special: boolean;
}

// ============================================================================
// Constants
// ============================================================================

const PASSWORD_REQUIREMENTS: Record<keyof PasswordRequirements, string> = {
  minLength: '8+ characters',
  lowercase: 'Lowercase letter (a-z)',
  uppercase: 'Uppercase letter (A-Z)',
  number: 'Number (0-9)',
  special: 'Special character (!@#$%)',
};

// ============================================================================
// Component
// ============================================================================

/**
 * RegisterForm - Modern registration form with real-time validation
 * ✅ FIXED: Correct API payload with backend field names
 */
const RegisterForm: React.FC = () => {
  const navigate = useNavigate();
  const addNotification = uiStore((state) => state.addNotification);

  // ✅ FIXED: Using import.meta.env
  const isDevelopment = import.meta.env.MODE === 'development';

  // State
  const [formData, setFormData] = useState<FormData>({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    agreeToTerms: false,
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [touched, setTouched] = useState<Partial<Record<keyof FormData, boolean>>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // ============================================================================
  // Utilities
  // ============================================================================

  const calculatePasswordStrength = useCallback((password: string): number => {
    if (!password) return 0;

    let score = 0;

    if (password.length >= 8) score += 20;
    if (password.length >= 12) score += 10;
    if (password.length >= 16) score += 5;

    if (/[a-z]/.test(password)) score += 15;
    if (/[A-Z]/.test(password)) score += 15;
    if (/[0-9]/.test(password)) score += 15;
    if (/[^a-zA-Z0-9]/.test(password)) score += 20;

    return Math.min(score, 100);
  }, []);

  const passwordStrength = useMemo((): PasswordStrength => {
    const score = calculatePasswordStrength(formData.password);

    if (score === 0) {
      return {
        score: 0,
        label: '',
        color: '',
        bgColor: '',
      };
    }

    if (score < 30) {
      return {
        score,
        label: 'Weak',
        color: 'text-red-600',
        bgColor: 'bg-red-100',
      };
    }

    if (score < 60) {
      return {
        score,
        label: 'Fair',
        color: 'text-yellow-600',
        bgColor: 'bg-yellow-100',
      };
    }

    if (score < 80) {
      return {
        score,
        label: 'Good',
        color: 'text-blue-600',
        bgColor: 'bg-blue-100',
      };
    }

    return {
      score,
      label: 'Strong',
      color: 'text-green-600',
      bgColor: 'bg-green-100',
    };
  }, [formData.password, calculatePasswordStrength]);

  const passwordRequirements = useMemo((): PasswordRequirements => {
    const pwd = formData.password;
    return {
      minLength: pwd.length >= 8,
      lowercase: /[a-z]/.test(pwd),
      uppercase: /[A-Z]/.test(pwd),
      number: /[0-9]/.test(pwd),
      special: /[^a-zA-Z0-9]/.test(pwd),
    };
  }, [formData.password]);

  const requirementsMet = useMemo(() => {
    return Object.values(passwordRequirements).filter(Boolean).length;
  }, [passwordRequirements]);

  const isFormValid = useMemo((): boolean => {
    return (
      formData.fullName.trim().length >= 2 &&
      validators.validateEmail(formData.email) &&
      validators.validatePassword(formData.password) &&
      formData.password === formData.confirmPassword &&
      formData.agreeToTerms &&
      Object.keys(errors).length === 0
    );
  }, [formData, errors]);

  // ============================================================================
  // Handlers
  // ============================================================================

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const { name, value, type, checked } = e.target;
      const inputValue = type === 'checkbox' ? checked : value;

      setFormData((prev) => ({
        ...prev,
        [name]: inputValue,
      }));

      if (touched[name as keyof FormData]) {
        setTouched((prev) => ({
          ...prev,
          [name]: true,
        }));
      }

      if (errors[name as keyof FormErrors]) {
        setErrors((prev) => ({
          ...prev,
          [name]: undefined,
        }));
      }
    },
    [errors, touched]
  );

  const handleFieldBlur = useCallback(
    (fieldName: keyof FormData) => {
      setTouched((prev) => ({
        ...prev,
        [fieldName]: true,
      }));
      validateField(fieldName);
    },
    [formData]
  );

  const validateField = useCallback(
    (fieldName: keyof FormData) => {
      const newErrors: FormErrors = { ...errors };

      switch (fieldName) {
        case 'fullName': {
          const name = formData.fullName.trim();
          if (!name) {
            newErrors.fullName = 'Full name is required';
          } else if (name.length < 2) {
            newErrors.fullName = 'Full name must be at least 2 characters';
          } else {
            delete newErrors.fullName;
          }
          break;
        }

        case 'email': {
          const email = formData.email.trim();
          if (!email) {
            newErrors.email = 'Email is required';
          } else if (!validators.validateEmail(email)) {
            newErrors.email = 'Please enter a valid email address';
          } else {
            delete newErrors.email;
          }
          break;
        }

        case 'password': {
          if (!formData.password) {
            newErrors.password = 'Password is required';
          } else if (!validators.validatePassword(formData.password)) {
            newErrors.password = 'Password must be at least 8 characters';
          } else {
            delete newErrors.password;
          }
          break;
        }

        case 'confirmPassword': {
          if (!formData.confirmPassword) {
            newErrors.confirmPassword = 'Please confirm your password';
          } else if (formData.password !== formData.confirmPassword) {
            newErrors.confirmPassword = 'Passwords do not match';
          } else {
            delete newErrors.confirmPassword;
          }
          break;
        }

        case 'agreeToTerms': {
          if (!formData.agreeToTerms) {
            newErrors.agreeToTerms = 'You must agree to the terms';
          } else {
            delete newErrors.agreeToTerms;
          }
          break;
        }

        default:
          break;
      }

      setErrors(newErrors);
    },
    [formData, errors]
  );

  const validateForm = useCallback((): boolean => {
    const newErrors: FormErrors = {};

    if (!formData.fullName.trim()) {
      newErrors.fullName = 'Full name is required';
    } else if (formData.fullName.trim().length < 2) {
      newErrors.fullName = 'Full name must be at least 2 characters';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'Email is required';
    } else if (!validators.validateEmail(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    } else if (!validators.validatePassword(formData.password)) {
      newErrors.password = 'Password must be at least 8 characters';
    }

    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password';
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }

    if (!formData.agreeToTerms) {
      newErrors.agreeToTerms = 'You must agree to the terms';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData]);

// ✅ FIXED: handleSubmit with proper username generation
const handleSubmit = useCallback(
  async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!validateForm()) {
      addNotification({
        type: 'error',
        message: 'Please fix the errors above',
        duration: 4000,
      });
      return;
    }

    setIsLoading(true);

    try {
      // ✅ FIXED: Generate username from email (remove @ and . chars)
      // Example: "john.doe@example.com" -> "johndoe"
      const generatedUsername = formData.email
        .split('@')[0]  // Get part before @
        .replace(/[^a-zA-Z0-9_-]/g, '')  // Remove invalid characters (keep only alphanumeric, _, -)
        .toLowerCase()
        .slice(0, 30);  // Max 30 chars

      const payload = {
        username: generatedUsername,          // ✅ FIXED: Valid username
        full_name: formData.fullName.trim(),
        email: formData.email.trim().toLowerCase(),
        password: formData.password,
      };

      if (isDevelopment) {
        console.log('📤 Sending payload:', payload);
      }

      // Call the API directly
      const response = await axios.post(
        'http://localhost:8000/api/v1/auth/register',
        payload,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      if (isDevelopment) {
        console.log('✅ Registration successful:', response.data);
      }

      addNotification({
        type: 'success',
        message: 'Account created successfully! Redirecting to dashboard...',
        duration: 2000,
      });

      // Clear form
      setFormData({
        fullName: '',
        email: '',
        password: '',
        confirmPassword: '',
        agreeToTerms: false,
      });

      // Redirect to dashboard
      setTimeout(() => navigate('/dashboard'), 1500);
    } catch (error: any) {
      if (isDevelopment) {
        console.error('❌ Registration error:', error);
        console.error('Error response:', error?.response?.data);
      }

      // Extract error message
      const errorMessage =
        error?.response?.data?.message ||
        error?.response?.data?.detail ||
        error?.response?.data?.error ||
        error?.message ||
        'Registration failed. Please try again.';

      // Handle specific validation errors
      if (error?.response?.data?.details && Array.isArray(error.response.data.details)) {
        const firstError = error.response.data.details[0];
        setErrors({ general: firstError });
      } else {
        setErrors({ general: errorMessage });
      }

      addNotification({
        type: 'error',
        message: errorMessage,
        duration: 5000,
      });
    } finally {
      setIsLoading(false);
    }
  },
  [formData, validateForm, navigate, addNotification, isDevelopment]
);

  // ============================================================================
  // Render
  // ============================================================================

  const isFormLoading = isLoading;

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-6 w-full">
      {/* Error Alert */}
      {errors.general && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-medium text-red-900 mb-1">Registration failed</h4>
            <p className="text-sm text-red-700">{errors.general}</p>
          </div>
        </div>
      )}

      {/* Form Fields */}
      <div className="space-y-5">
        {/* Full Name */}
        <Input
          label="Full Name"
          type="text"
          name="fullName"
          value={formData.fullName}
          onChange={handleInputChange}
          onBlur={() => handleFieldBlur('fullName')}
          error={touched.fullName ? errors.fullName : undefined}
          placeholder="John Doe"
          leftIcon={User}
          required
          disabled={isFormLoading}
          fullWidth
        />

        {/* Email */}
        <Input
          label="Email Address"
          type="email"
          name="email"
          value={formData.email}
          onChange={handleInputChange}
          onBlur={() => handleFieldBlur('email')}
          error={touched.email ? errors.email : undefined}
          placeholder="you@example.com"
          leftIcon={Mail}
          required
          disabled={isFormLoading}
          fullWidth
          helperText="We'll keep your email secure"
        />

        {/* Password */}
        <div className="space-y-3">
          <div className="relative">
            <Input
              label="Password"
              type={showPassword ? 'text' : 'password'}
              name="password"
              value={formData.password}
              onChange={handleInputChange}
              onBlur={() => handleFieldBlur('password')}
              error={touched.password ? errors.password : undefined}
              placeholder="Create a strong password"
              leftIcon={Lock}
              required
              disabled={isFormLoading}
              fullWidth
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute right-3 top-10 text-gray-500 hover:text-gray-700 transition-colors"
              tabIndex={-1}
            >
              {showPassword ? (
                <EyeOff className="w-5 h-5" />
              ) : (
                <Eye className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Password Strength Indicator */}
          {formData.password && (
            <div className="space-y-2 p-4 bg-slate-50 rounded-lg border border-slate-200">
              {/* Strength Label & Score */}
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-700">
                  Password strength
                </span>
                {passwordStrength.label && (
                  <span className={`text-xs font-semibold ${passwordStrength.color}`}>
                    {passwordStrength.label}
                  </span>
                )}
              </div>

              {/* Strength Bar */}
              <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 transition-all duration-300"
                  style={{ width: `${passwordStrength.score}%` }}
                />
              </div>

              {/* Requirements Checklist */}
              <div className="grid grid-cols-2 gap-2 pt-2">
                {Object.entries(PASSWORD_REQUIREMENTS).map(([key, label]) => {
                  const met = passwordRequirements[key as keyof PasswordRequirements];
                  return (
                    <div
                      key={key}
                      className={`text-xs flex items-center gap-2 transition-colors ${
                        met ? 'text-green-600' : 'text-gray-600'
                      }`}
                    >
                      {met ? (
                        <CheckCircle className="w-3.5 h-3.5 flex-shrink-0" />
                      ) : (
                        <div className="w-3.5 h-3.5 rounded-full border border-gray-300" />
                      )}
                      <span>{label}</span>
                    </div>
                  );
                })}
              </div>

              {/* Progress Text */}
              <p className="text-xs text-gray-600 pt-1">
                {requirementsMet}/5 requirements met
              </p>
            </div>
          )}
        </div>

        {/* Confirm Password */}
        <div className="relative">
          <Input
            label="Confirm Password"
            type={showConfirmPassword ? 'text' : 'password'}
            name="confirmPassword"
            value={formData.confirmPassword}
            onChange={handleInputChange}
            onBlur={() => handleFieldBlur('confirmPassword')}
            error={touched.confirmPassword ? errors.confirmPassword : undefined}
            placeholder="Re-enter your password"
            leftIcon={Lock}
            required
            disabled={isFormLoading}
            fullWidth
          />
          <button
            type="button"
            onClick={() => setShowConfirmPassword(!showConfirmPassword)}
            className="absolute right-3 top-10 text-gray-500 hover:text-gray-700 transition-colors"
            tabIndex={-1}
          >
            {showConfirmPassword ? (
              <EyeOff className="w-5 h-5" />
            ) : (
              <Eye className="w-5 h-5" />
            )}
          </button>
        </div>

        {/* Match Indicator */}
        {formData.password && formData.confirmPassword && (
          <div className="flex items-center gap-2 text-sm">
            {formData.password === formData.confirmPassword ? (
              <>
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span className="text-green-600 font-medium">Passwords match</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-4 h-4 text-red-600" />
                <span className="text-red-600 font-medium">Passwords do not match</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Terms & Conditions */}
      <div className="space-y-2 pt-2 border-t border-gray-200">
        <label className="flex items-start gap-3 cursor-pointer group">
          <input
            type="checkbox"
            name="agreeToTerms"
            checked={formData.agreeToTerms}
            onChange={handleInputChange}
            onBlur={() => handleFieldBlur('agreeToTerms')}
            disabled={isFormLoading}
            className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500 mt-1 cursor-pointer"
            required
          />
          <span className="text-sm text-gray-700 group-hover:text-gray-900 transition-colors leading-relaxed">
            I agree to the{' '}
            <a
              href="/terms"
              className="text-blue-600 hover:text-blue-700 font-medium underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Terms of Service
            </a>
            {' and '}
            <a
              href="/privacy"
              className="text-blue-600 hover:text-blue-700 font-medium underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Privacy Policy
            </a>
          </span>
        </label>

        {errors.agreeToTerms && (
          <p className="text-sm text-red-600 flex items-center gap-2 ml-7">
            <AlertCircle className="w-4 h-4" />
            {errors.agreeToTerms}
          </p>
        )}
      </div>

      {/* Submit Button */}
      <Button
        type="submit"
        variant="primary"
        fullWidth
        loading={isFormLoading}
        loadingText="Creating account..."
        disabled={!isFormValid && Object.keys(touched).length > 0}
        size="lg"
        leftIcon={ArrowRight}
        className="mt-8"
      >
        Create Account
      </Button>

      {/* Sign In Link */}
      <p className="text-center text-sm text-gray-600">
        Already have an account?{' '}
        <a
          href="/login"
          className="text-blue-600 hover:text-blue-700 font-semibold transition-colors"
        >
          Sign in instead
        </a>
      </p>

      {/* Security Notice */}
      <div className="p-3 bg-blue-50 rounded-lg border border-blue-200 flex items-start gap-2">
        <div className="text-blue-600 text-lg mt-0.5">🔒</div>
        <p className="text-xs text-blue-700 leading-relaxed">
          Your data is encrypted end-to-end and never shared with third parties.
          We comply with GDPR and other privacy regulations.
        </p>
      </div>
    </form>
  );
};

RegisterForm.displayName = 'RegisterForm';

export default RegisterForm;
