// src/utils/validators.ts - FINAL PRODUCTION VERSION (COMPATIBLE WITH LOGINPAGE)

/**
 * Validator Utilities
 * Form validation functions for various data types and patterns
 * ✅ FIXED: All validators return boolean for simple validation
 */

// ============================================================================
// Email & Communication Validators
// ============================================================================

/**
 * ✅ FIXED: Validate email address - returns only boolean
 * Used in LoginPage and other authentication forms
 */
export function validateEmail(email: string): boolean {
  try {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return !!email && emailRegex.test(email);
  } catch {
    return false;
  }
}

/**
 * Validate email with detailed error message
 */
export function validateEmailWithError(email: string): {
  valid: boolean;
  error?: string;
} {
  try {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

    if (!email) {
      return { valid: false, error: 'Email is required' };
    }

    if (email.length > 254) {
      return { valid: false, error: 'Email is too long' };
    }

    if (!emailRegex.test(email)) {
      return { valid: false, error: 'Invalid email format' };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating email' };
  }
}

/**
 * Validate phone number - returns only boolean
 */
export function validatePhoneNumber(phone: string): boolean {
  try {
    const cleaned = phone.replace(/\D/g, '');
    return cleaned.length >= 10 && cleaned.length <= 15;
  } catch {
    return false;
  }
}

/**
 * Validate phone number with detailed error message
 */
export function validatePhoneNumberWithError(phone: string): {
  valid: boolean;
  error?: string;
} {
  try {
    const cleaned = phone.replace(/\D/g, '');

    if (!phone) {
      return { valid: false, error: 'Phone number is required' };
    }

    if (cleaned.length < 10) {
      return { valid: false, error: 'Phone number too short' };
    }

    if (cleaned.length > 15) {
      return { valid: false, error: 'Phone number too long' };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating phone number' };
  }
}

/**
 * Validate URL - returns only boolean
 */
export function validateURL(url: string): boolean {
  try {
    if (!url) return false;
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate URL with detailed error message
 */
export function validateURLWithError(url: string): {
  valid: boolean;
  error?: string;
} {
  try {
    if (!url) {
      return { valid: false, error: 'URL is required' };
    }

    new URL(url);
    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Invalid URL format' };
  }
}

// ============================================================================
// Password & Security Validators
// ============================================================================

/**
 * Validate password strength result interface
 */
export interface PasswordStrengthResult {
  score: number;
  strength:
    | 'very_weak'
    | 'weak'
    | 'fair'
    | 'good'
    | 'strong'
    | 'very_strong';
  feedback: string[];
  suggestions: string[];
}

/**
 * Validate password strength with detailed feedback
 */
export function validatePasswordStrength(
  password: string
): PasswordStrengthResult {
  try {
    const feedback: string[] = [];
    const suggestions: string[] = [];
    let score = 0;

    if (!password) {
      return {
        score: 0,
        strength: 'very_weak',
        feedback: ['Password is required'],
        suggestions: ['Enter a password'],
      };
    }

    // Length checks
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;
    if (password.length >= 16) score++;

    // Character variety checks
    if (/[a-z]/.test(password)) {
      score++;
      feedback.push('Contains lowercase letters');
    } else {
      suggestions.push('Add lowercase letters');
    }

    if (/[A-Z]/.test(password)) {
      score++;
      feedback.push('Contains uppercase letters');
    } else {
      suggestions.push('Add uppercase letters');
    }

    if (/[0-9]/.test(password)) {
      score++;
      feedback.push('Contains numbers');
    } else {
      suggestions.push('Add numbers');
    }

    if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) {
      score++;
      feedback.push('Contains special characters');
    } else {
      suggestions.push('Add special characters');
    }

    // Common patterns
    if (!/(.)\1{2,}/.test(password)) {
      feedback.push('No repeating characters');
    } else {
      suggestions.push('Avoid repeating characters');
    }

    if (!/^\d+$|^[a-zA-Z]+$/.test(password)) {
      feedback.push('Mix of character types');
    }

    let strength: PasswordStrengthResult['strength'];
    if (score <= 1) strength = 'very_weak';
    else if (score <= 2) strength = 'weak';
    else if (score <= 3) strength = 'fair';
    else if (score <= 4) strength = 'good';
    else if (score <= 5) strength = 'strong';
    else strength = 'very_strong';

    return {
      score: Math.min(5, score),
      strength,
      feedback,
      suggestions,
    };
  } catch (error) {
    return {
      score: 0,
      strength: 'very_weak',
      feedback: ['Error validating password'],
      suggestions: [],
    };
  }
}

/**
 * ✅ FIXED: Validate password format - returns only boolean
 */
export function validatePassword(
  password: string,
  minLength: number = 6
): boolean {
  try {
    return !!password && password.length >= minLength;
  } catch {
    return false;
  }
}

/**
 * Validate password format with custom requirements
 */
export function validatePasswordFormat(
  password: string,
  minLength: number = 8,
  requireUppercase: boolean = true,
  requireNumbers: boolean = true,
  requireSpecialChars: boolean = false
): { valid: boolean; error?: string } {
  try {
    if (!password) {
      return { valid: false, error: 'Password is required' };
    }

    if (password.length < minLength) {
      return {
        valid: false,
        error: `Password must be at least ${minLength} characters`,
      };
    }

    if (requireUppercase && !/[A-Z]/.test(password)) {
      return {
        valid: false,
        error: 'Password must contain uppercase letter',
      };
    }

    if (requireNumbers && !/[0-9]/.test(password)) {
      return { valid: false, error: 'Password must contain number' };
    }

    if (
      requireSpecialChars &&
      !/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)
    ) {
      return {
        valid: false,
        error: 'Password must contain special character',
      };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating password' };
  }
}

// ============================================================================
// File Validators
// ============================================================================

/**
 * Validate file type - returns only boolean
 */
export function validateFileType(
  file: File,
  allowedTypes: string[]
): boolean {
  try {
    return !!file && allowedTypes.includes(file.type);
  } catch {
    return false;
  }
}

/**
 * Validate file type with detailed error message
 */
export function validateFileTypeWithError(
  file: File,
  allowedTypes: string[]
): { valid: boolean; error?: string } {
  try {
    if (!file) {
      return { valid: false, error: 'File is required' };
    }

    if (!allowedTypes.includes(file.type)) {
      return {
        valid: false,
        error: `File type not allowed. Allowed types: ${allowedTypes.join(', ')}`,
      };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating file type' };
  }
}

/**
 * Validate file size - returns only boolean
 */
export function validateFileSize(
  file: File,
  maxSizeBytes: number
): boolean {
  try {
    return !!file && file.size <= maxSizeBytes;
  } catch {
    return false;
  }
}

/**
 * Validate file size with detailed error message
 */
export function validateFileSizeWithError(
  file: File,
  maxSizeBytes: number
): { valid: boolean; error?: string } {
  try {
    if (!file) {
      return { valid: false, error: 'File is required' };
    }

    if (file.size > maxSizeBytes) {
      const maxSizeMB = (maxSizeBytes / 1024 / 1024).toFixed(2);
      const fileSizeMB = (file.size / 1024 / 1024).toFixed(2);
      return {
        valid: false,
        error: `File size (${fileSizeMB}MB) exceeds maximum (${maxSizeMB}MB)`,
      };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating file size' };
  }
}

/**
 * Validate file extension - returns only boolean
 */
export function validateFileExtension(
  fileName: string,
  allowedExtensions: string[]
): boolean {
  try {
    if (!fileName) return false;
    const extension = fileName.split('.').pop()?.toLowerCase();
    return !!extension && allowedExtensions.includes(`.${extension}`);
  } catch {
    return false;
  }
}

/**
 * Validate file extension with detailed error message
 */
export function validateFileExtensionWithError(
  fileName: string,
  allowedExtensions: string[]
): { valid: boolean; error?: string } {
  try {
    if (!fileName) {
      return { valid: false, error: 'File name is required' };
    }

    const extension = fileName.split('.').pop()?.toLowerCase();

    if (!extension || !allowedExtensions.includes(`.${extension}`)) {
      return {
        valid: false,
        error: `File extension not allowed. Allowed: ${allowedExtensions.join(', ')}`,
      };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating file extension' };
  }
}

/**
 * Validate file (type, size, extension)
 */
export function validateFile(
  file: File,
  options: {
    allowedTypes?: string[];
    allowedExtensions?: string[];
    maxSizeBytes?: number;
  }
): { valid: boolean; errors: string[] } {
  try {
    const errors: string[] = [];

    if (options.allowedTypes) {
      const typeValidation = validateFileTypeWithError(
        file,
        options.allowedTypes
      );
      if (!typeValidation.valid) errors.push(typeValidation.error!);
    }

    if (options.allowedExtensions) {
      const extValidation = validateFileExtensionWithError(
        file.name,
        options.allowedExtensions
      );
      if (!extValidation.valid) errors.push(extValidation.error!);
    }

    if (options.maxSizeBytes) {
      const sizeValidation = validateFileSizeWithError(
        file,
        options.maxSizeBytes
      );
      if (!sizeValidation.valid) errors.push(sizeValidation.error!);
    }

    return { valid: errors.length === 0, errors };
  } catch (error) {
    return { valid: false, errors: ['Error validating file'] };
  }
}

// ============================================================================
// Text & String Validators
// ============================================================================

/**
 * Validate required field - returns only boolean
 */
export function validateRequired(value: any): boolean {
  return value !== null && value !== undefined && value !== '';
}

/**
 * Validate text length - returns only boolean
 */
export function validateLength(
  text: string,
  minLength?: number,
  maxLength?: number
): boolean {
  try {
    if (!text) return false;
    if (minLength && text.length < minLength) return false;
    if (maxLength && text.length > maxLength) return false;
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate text pattern (regex) - returns only boolean
 */
export function validatePattern(text: string, pattern: RegExp): boolean {
  try {
    return !!text && pattern.test(text);
  } catch {
    return false;
  }
}

/**
 * Validate username - returns only boolean
 */
export function validateUsername(username: string): boolean {
  try {
    return (
      !!username &&
      username.length >= 3 &&
      username.length <= 20 &&
      /^[a-zA-Z0-9_-]+$/.test(username)
    );
  } catch {
    return false;
  }
}

/**
 * Validate username with detailed error message
 */
export function validateUsernameWithError(username: string): {
  valid: boolean;
  error?: string;
} {
  try {
    if (!username) {
      return { valid: false, error: 'Username is required' };
    }

    if (username.length < 3) {
      return { valid: false, error: 'Username must be at least 3 characters' };
    }

    if (username.length > 20) {
      return { valid: false, error: 'Username must not exceed 20 characters' };
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(username)) {
      return {
        valid: false,
        error:
          'Username can only contain letters, numbers, underscore, and hyphen',
      };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating username' };
  }
}

// ============================================================================
// Number & Range Validators
// ============================================================================

/**
 * Validate number range - returns only boolean
 */
export function validateNumberRange(
  value: number,
  min?: number,
  max?: number
): boolean {
  try {
    if (value === null || value === undefined) return false;
    if (min !== undefined && value < min) return false;
    if (max !== undefined && value > max) return false;
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate integer - returns only boolean
 */
export function validateInteger(value: any): boolean {
  try {
    return Number.isInteger(value);
  } catch {
    return false;
  }
}

/**
 * Validate decimal - returns only boolean
 */
export function validateDecimal(
  value: any,
  decimalPlaces?: number
): boolean {
  try {
    if (typeof value !== 'number') return false;

    if (decimalPlaces !== undefined) {
      const decimalRegex = new RegExp(
        `^-?\\d+(\\.\\d{1,${decimalPlaces}})?$`
      );
      return decimalRegex.test(value.toString());
    }

    return true;
  } catch {
    return false;
  }
}

// ============================================================================
// Custom Validators
// ============================================================================

/**
 * Validate field matches another field - returns only boolean
 */
export function validateMatch(value: any, compareValue: any): boolean {
  try {
    return value === compareValue;
  } catch {
    return false;
  }
}

/**
 * Validate custom condition - returns only boolean
 */
export function validateCustom(
  value: any,
  validatorFn: (value: any) => boolean
): boolean {
  try {
    return validatorFn(value);
  } catch {
    return false;
  }
}

/**
 * Validate enum value - returns only boolean
 */
export function validateEnum(value: any, allowedValues: any[]): boolean {
  try {
    return allowedValues.includes(value);
  } catch {
    return false;
  }
}

/**
 * Validate credit card number (Luhn algorithm) - returns only boolean
 */
export function validateCreditCard(cardNumber: string): boolean {
  try {
    const cleaned = cardNumber.replace(/\s/g, '');

    if (!/^\d+$/.test(cleaned)) return false;
    if (cleaned.length < 13 || cleaned.length > 19) return false;

    // Luhn algorithm
    let sum = 0;
    let isEven = false;

    for (let i = cleaned.length - 1; i >= 0; i--) {
      let digit = parseInt(cleaned[i], 10);

      if (isEven) {
        digit *= 2;
        if (digit > 9) digit -= 9;
      }

      sum += digit;
      isEven = !isEven;
    }

    return sum % 10 === 0;
  } catch {
    return false;
  }
}

/**
 * Validate credit card with detailed error message
 */
export function validateCreditCardWithError(cardNumber: string): {
  valid: boolean;
  error?: string;
} {
  try {
    const cleaned = cardNumber.replace(/\s/g, '');

    if (!/^\d+$/.test(cleaned)) {
      return {
        valid: false,
        error: 'Credit card number must contain only digits',
      };
    }

    if (cleaned.length < 13 || cleaned.length > 19) {
      return {
        valid: false,
        error: 'Credit card number must be 13-19 digits',
      };
    }

    // Luhn algorithm
    let sum = 0;
    let isEven = false;

    for (let i = cleaned.length - 1; i >= 0; i--) {
      let digit = parseInt(cleaned[i], 10);

      if (isEven) {
        digit *= 2;
        if (digit > 9) digit -= 9;
      }

      sum += digit;
      isEven = !isEven;
    }

    if (sum % 10 !== 0) {
      return { valid: false, error: 'Invalid credit card number' };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: 'Error validating credit card' };
  }
}
