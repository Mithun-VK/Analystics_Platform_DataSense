// src/components/shared/Input.tsx

import { forwardRef, useState } from 'react';
import type {
  InputHTMLAttributes,
  TextareaHTMLAttributes,
} from 'react'; // ✅ Type-only import
import { Eye, EyeOff, AlertCircle, CheckCircle } from 'lucide-react';
import type { LucideIcon } from 'lucide-react'; // ✅ Type-only import

export interface InputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Input label text */
  label?: string;
  /** Error message to display */
  error?: string;
  /** Success message to display */
  success?: string;
  /** Helper text to display below input */
  helperText?: string;
  /** Icon to display before input */
  leftIcon?: LucideIcon;
  /** Icon to display after input */
  rightIcon?: LucideIcon;
  /** Make input full width */
  fullWidth?: boolean;
  /** Input size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Show character counter */
  showCounter?: boolean;
  /** Make label required indicator */
  required?: boolean;
  /** Additional class name for wrapper */
  wrapperClassName?: string;
  /** Additional class name for label */
  labelClassName?: string;
}

export interface TextareaProps
  extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'size'> {
  /** Textarea label text */
  label?: string;
  /** Error message to display */
  error?: string;
  /** Success message to display */
  success?: string;
  /** Helper text to display below textarea */
  helperText?: string;
  /** Make textarea full width */
  fullWidth?: boolean;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Show character counter */
  showCounter?: boolean;
  /** Make label required indicator */
  required?: boolean;
  /** Additional class name for wrapper */
  wrapperClassName?: string;
  /** Additional class name for label */
  labelClassName?: string;
  /** Resize behavior */
  resize?: 'none' | 'vertical' | 'horizontal' | 'both';
}

/**
 * Input - Controlled input component with validation and error display
 * Features: Label, error/success states, icons, character counter, password toggle
 * Fully accessible with ARIA attributes and semantic HTML
 *
 * @example
 * <Input
 *   label="Email Address"
 *   type="email"
 *   error={errors.email}
 *   leftIcon={Mail}
 *   required
 * />
 */
const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      error,
      success,
      helperText,
      leftIcon: LeftIcon,
      rightIcon: RightIcon,
      fullWidth = false,
      size = 'md',
      showCounter = false,
      required = false,
      disabled = false,
      type = 'text',
      className = '',
      wrapperClassName = '',
      labelClassName = '',
      maxLength,
      value,
      id,
      ...props
    },
    ref
  ) => {
    const [showPassword, setShowPassword] = useState(false);
    const [, setIsFocused] = useState(false);

    // Generate unique ID if not provided
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;

    // Determine input type (handle password toggle)
    const inputType = type === 'password' && showPassword ? 'text' : type;

    // Get input size classes
    const getSizeClasses = (): string => {
      switch (size) {
        case 'sm':
          return 'py-1.5 text-sm';
        case 'lg':
          return 'py-3 text-base';
        default:
          return 'py-2.5 text-sm';
      }
    };

    // Get input state classes
    const getStateClasses = (): string => {
      if (error) {
        return 'input-error border-red-300 focus:border-red-500 focus:ring-red-500';
      }
      if (success) {
        return 'input-success border-green-300 focus:border-green-500 focus:ring-green-500';
      }
      return 'border-gray-300 focus:border-blue-500 focus:ring-blue-500';
    };

    // Calculate character count
    const currentLength = value ? String(value).length : 0;
    const showCounterDisplay = showCounter && maxLength;

    return (
      <div className={`${fullWidth ? 'w-full' : ''} ${wrapperClassName}`}>
        {/* Label */}
        {label && (
          <label
            htmlFor={inputId}
            className={`label ${labelClassName}`}
          >
            {label}
            {required && (
              <span className="text-red-500 ml-1" aria-label="required">
                *
              </span>
            )}
          </label>
        )}

        {/* Input Container */}
        <div className="relative">
          {/* Left Icon */}
          {LeftIcon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <LeftIcon
                className={`w-5 h-5 ${
                  error
                    ? 'text-red-500'
                    : success
                    ? 'text-green-500'
                    : 'text-gray-400'
                }`}
                aria-hidden="true"
              />
            </div>
          )}

          {/* Input Field */}
          <input
            ref={ref}
            id={inputId}
            type={inputType}
            value={value}
            disabled={disabled}
            required={required}
            maxLength={maxLength}
            className={`
              input
              ${getSizeClasses()}
              ${getStateClasses()}
              ${LeftIcon ? 'pl-10' : ''}
              ${RightIcon || type === 'password' || error || success ? 'pr-10' : ''}
              ${fullWidth ? 'w-full' : ''}
              ${className}
            `}
            aria-invalid={error ? 'true' : 'false'}
            aria-describedby={
              error
                ? `${inputId}-error`
                : success
                ? `${inputId}-success`
                : helperText
                ? `${inputId}-helper`
                : undefined
            }
            aria-required={required}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            {...props}
          />

          {/* Right Icons */}
          <div className="absolute inset-y-0 right-0 flex items-center pr-3 space-x-1">
            {/* Error Icon */}
            {error && (
              <AlertCircle
                className="w-5 h-5 text-red-500"
                aria-hidden="true"
              />
            )}

            {/* Success Icon */}
            {success && !error && (
              <CheckCircle
                className="w-5 h-5 text-green-500"
                aria-hidden="true"
              />
            )}

            {/* Password Toggle */}
            {type === 'password' && !error && !success && (
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="text-gray-400 hover:text-gray-600 focus:outline-none focus:text-gray-600 transition-colors"
                tabIndex={-1}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? (
                  <EyeOff className="w-5 h-5" />
                ) : (
                  <Eye className="w-5 h-5" />
                )}
              </button>
            )}

            {/* Right Icon */}
            {RightIcon && !error && !success && type !== 'password' && (
              <RightIcon
                className="w-5 h-5 text-gray-400"
                aria-hidden="true"
              />
            )}
          </div>
        </div>

        {/* Helper/Error/Success Text */}
        <div className="mt-1.5 min-h-[20px]">
          {error && (
            <p
              id={`${inputId}-error`}
              className="error-message"
              role="alert"
              aria-live="polite"
            >
              {error}
            </p>
          )}

          {success && !error && (
            <p
              id={`${inputId}-success`}
              className="text-sm text-green-600 flex items-center space-x-1"
            >
              <CheckCircle className="w-4 h-4" />
              <span>{success}</span>
            </p>
          )}

          {helperText && !error && !success && (
            <p
              id={`${inputId}-helper`}
              className="text-sm text-gray-600"
            >
              {helperText}
            </p>
          )}
        </div>

        {/* Character Counter */}
        {showCounterDisplay && (
          <div className="mt-1 text-right">
            <span
              className={`text-xs ${
                currentLength > maxLength! ? 'text-red-600' : 'text-gray-500'
              }`}
              aria-live="polite"
              aria-atomic="true"
            >
              {currentLength} / {maxLength}
            </span>
          </div>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;

// ============================================================================
// Textarea variant - SEPARATE COMPONENT with proper types
// ============================================================================

/**
 * Textarea - Controlled textarea component with validation and error display
 * Features: Label, error/success states, character counter, auto-resize
 * Fully accessible with ARIA attributes and semantic HTML
 *
 * @example
 * <Textarea
 *   label="Message"
 *   error={errors.message}
 *   showCounter
 *   maxLength={500}
 * />
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      success,
      helperText,
      fullWidth = false,
      size = 'md',
      showCounter = false,
      required = false,
      disabled = false,
      className = '',
      wrapperClassName = '',
      labelClassName = '',
      maxLength,
      value,
      id,
      rows = 4,
      resize = 'vertical',
      ...props
    },
    ref
  ) => {
    // ✅ Generate unique ID if not provided
    const textareaId =
      id || `textarea-${Math.random().toString(36).substr(2, 9)}`;

    // ✅ Calculate character count safely
    const currentLength = value ? String(value).length : 0;
    const showCounterDisplay = showCounter && maxLength;

    // ✅ Get size classes
    const getSizeClasses = (): string => {
      switch (size) {
        case 'sm':
          return 'py-1.5 text-sm';
        case 'lg':
          return 'py-3 text-base';
        default:
          return 'py-2 text-sm';
      }
    };

    // ✅ Get state classes
    const getStateClasses = (): string => {
      if (error) {
        return 'input-error border-red-300 focus:border-red-500 focus:ring-red-500';
      }
      if (success) {
        return 'input-success border-green-300 focus:border-green-500 focus:ring-green-500';
      }
      return 'border-gray-300 focus:border-blue-500 focus:ring-blue-500';
    };

    // ✅ Get resize class
    const getResizeClass = (): string => {
      switch (resize) {
        case 'none':
          return 'resize-none';
        case 'horizontal':
          return 'resize-x';
        case 'both':
          return 'resize';
        default:
          return 'resize-y';
      }
    };

    return (
      <div className={`${fullWidth ? 'w-full' : ''} ${wrapperClassName}`}>
        {/* Label */}
        {label && (
          <label htmlFor={textareaId} className={`label ${labelClassName}`}>
            {label}
            {required && (
              <span className="text-red-500 ml-1" aria-label="required">
                *
              </span>
            )}
          </label>
        )}

        {/* Textarea */}
        <textarea
          ref={ref}
          id={textareaId}
          value={value}
          disabled={disabled}
          required={required}
          maxLength={maxLength}
          rows={rows}
          className={`
            input
            ${getSizeClasses()}
            ${getStateClasses()}
            ${getResizeClass()}
            ${fullWidth ? 'w-full' : ''}
            ${className}
          `}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={
            error
              ? `${textareaId}-error`
              : success
              ? `${textareaId}-success`
              : helperText
              ? `${textareaId}-helper`
              : undefined
          }
          aria-required={required}
          {...props}
        />

        {/* Helper/Error/Success Text */}
        <div className="mt-1.5 min-h-[20px]">
          {error && (
            <p
              id={`${textareaId}-error`}
              className="error-message"
              role="alert"
              aria-live="polite"
            >
              {error}
            </p>
          )}

          {success && !error && (
            <p
              id={`${textareaId}-success`}
              className="text-sm text-green-600 flex items-center space-x-1"
            >
              <CheckCircle className="w-4 h-4" />
              <span>{success}</span>
            </p>
          )}

          {helperText && !error && !success && (
            <p
              id={`${textareaId}-helper`}
              className="text-sm text-gray-600"
            >
              {helperText}
            </p>
          )}
        </div>

        {/* Character Counter */}
        {showCounterDisplay && (
          <div className="mt-1 text-right">
            <span
              className={`text-xs ${
                currentLength > maxLength! ? 'text-red-600' : 'text-gray-500'
              }`}
              aria-live="polite"
              aria-atomic="true"
            >
              {currentLength} / {maxLength}
            </span>
          </div>
        )}
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

// ============================================================================
// Select Input Component
// ============================================================================

export interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
}

export interface SelectProps
  extends Omit<InputHTMLAttributes<HTMLSelectElement>, 'size'> {
  /** Select label text */
  label?: string;
  /** Error message to display */
  error?: string;
  /** Helper text to display below select */
  helperText?: string;
  /** Select options */
  options: SelectOption[];
  /** Placeholder option text */
  placeholder?: string;
  /** Make select full width */
  fullWidth?: boolean;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Make label required indicator */
  required?: boolean;
  /** Additional class name for wrapper */
  wrapperClassName?: string;
  /** Additional class name for label */
  labelClassName?: string;
}

/**
 * Select - Controlled select component with validation
 * Features: Label, error states, options, full accessibility
 *
 * @example
 * <Select
 *   label="Country"
 *   options={countries}
 *   error={errors.country}
 * />
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      label,
      error,
      helperText,
      options,
      placeholder,
      fullWidth = false,
      size = 'md',
      required = false,
      disabled = false,
      className = '',
      wrapperClassName = '',
      labelClassName = '',
      id,
      value,
      ...props
    },
    ref
  ) => {
    const selectId = id || `select-${Math.random().toString(36).substr(2, 9)}`;

    const getSizeClasses = (): string => {
      switch (size) {
        case 'sm':
          return 'py-1.5 text-sm';
        case 'lg':
          return 'py-3 text-base';
        default:
          return 'py-2.5 text-sm';
      }
    };

    const getStateClasses = (): string => {
      if (error) {
        return 'border-red-300 focus:border-red-500 focus:ring-red-500';
      }
      return 'border-gray-300 focus:border-blue-500 focus:ring-blue-500';
    };

    return (
      <div className={`${fullWidth ? 'w-full' : ''} ${wrapperClassName}`}>
        {label && (
          <label htmlFor={selectId} className={`label ${labelClassName}`}>
            {label}
            {required && (
              <span className="text-red-500 ml-1" aria-label="required">
                *
              </span>
            )}
          </label>
        )}

        <select
          ref={ref}
          id={selectId}
          value={value}
          disabled={disabled}
          required={required}
          className={`
            input
            ${getSizeClasses()}
            ${getStateClasses()}
            ${fullWidth ? 'w-full' : ''}
            ${className}
          `}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={
            error ? `${selectId}-error` : helperText ? `${selectId}-helper` : undefined
          }
          aria-required={required}
          {...props}
        >
          {placeholder && (
            <option value="">{placeholder}</option>
          )}
          {options.map((option) => (
            <option
              key={`${option.value}-${option.label}`}
              value={option.value}
              disabled={option.disabled}
            >
              {option.label}
            </option>
          ))}
        </select>

        <div className="mt-1.5 min-h-[20px]">
          {error && (
            <p
              id={`${selectId}-error`}
              className="error-message"
              role="alert"
              aria-live="polite"
            >
              {error}
            </p>
          )}

          {helperText && !error && (
            <p id={`${selectId}-helper`} className="text-sm text-gray-600">
              {helperText}
            </p>
          )}
        </div>
      </div>
    );
  }
);

Select.displayName = 'Select';
