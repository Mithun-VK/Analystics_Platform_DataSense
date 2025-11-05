// src/utils/formatters.ts
/**
 * Formatter Utilities
 * Functions to format dates, numbers, file sizes, currencies, and other data types
 */

// ============================================================================
// Date Formatters
// ============================================================================

/**
 * Format date to standard format
 */
export function formatDate(
  date: string | Date | number,
  format: string = 'MMM DD, YYYY'
): string {
  try {
    const dateObj = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;

    if (isNaN(dateObj.getTime())) {
      return 'Invalid date';
    }

    const months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    const monthsFull = [
      'January', 'February', 'March', 'April', 'May', 'June',
      'July', 'August', 'September', 'October', 'November', 'December',
    ];
    const days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
    const daysFull = days;

    const year = dateObj.getFullYear();
    const month = dateObj.getMonth();
    const day = dateObj.getDate();
    const hours = String(dateObj.getHours()).padStart(2, '0');
    const minutes = String(dateObj.getMinutes()).padStart(2, '0');
    const seconds = String(dateObj.getSeconds()).padStart(2, '0');
    const dayOfWeek = dateObj.getDay();

    const replacements: Record<string, string> = {
      YYYY: year.toString(),
      YY: year.toString().slice(-2),
      MMMM: monthsFull[month],
      MMM: months[month],
      MM: String(month + 1).padStart(2, '0'),
      M: (month + 1).toString(),
      DD: String(day).padStart(2, '0'),
      D: day.toString(),
      dddd: daysFull[dayOfWeek],
      ddd: days[dayOfWeek].slice(0, 3),
      HH: hours,
      H: dateObj.getHours().toString(),
      mm: minutes,
      m: dateObj.getMinutes().toString(),
      ss: seconds,
      s: dateObj.getSeconds().toString(),
    };

    let result = format;
    Object.entries(replacements).forEach(([key, value]) => {
      result = result.replace(new RegExp(key, 'g'), value);
    });

    return result;
  } catch (error) {
    console.error('Error formatting date:', error);
    return 'Invalid date';
  }
}

/**
 * Format date as relative time (e.g., "2 hours ago")
 */
export function formatDateRelative(date: string | Date | number): string {
  try {
    const dateObj = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;
    const now = new Date();
    const seconds = Math.floor((now.getTime() - dateObj.getTime()) / 1000);

    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)} hours ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)} days ago`;
    if (seconds < 2592000) return `${Math.floor(seconds / 604800)} weeks ago`;
    if (seconds < 31536000) return `${Math.floor(seconds / 2592000)} months ago`;

    return `${Math.floor(seconds / 31536000)} years ago`;
  } catch (error) {
    console.error('Error formatting relative date:', error);
    return 'Unknown';
  }
}

/**
 * Format date and time together
 */
export function formatDateTime(date: string | Date | number, includeSeconds: boolean = false): string {
  const dateStr = formatDate(date, 'MMM DD, YYYY');
  const timeStr = includeSeconds
    ? formatDate(date, 'HH:mm:ss')
    : formatDate(date, 'HH:mm');
  return `${dateStr} at ${timeStr}`;
}

/**
 * Format time only
 */
export function formatTime(date: string | Date | number, format24h: boolean = true): string {
  try {
    const dateObj = typeof date === 'string' || typeof date === 'number' ? new Date(date) : date;
    let hours = dateObj.getHours();
    const minutes = String(dateObj.getMinutes()).padStart(2, '0');
    const seconds = String(dateObj.getSeconds()).padStart(2, '0');

    if (!format24h) {
      const ampm = hours >= 12 ? 'PM' : 'AM';
      hours = hours % 12 || 12;
      return `${String(hours).padStart(2, '0')}:${minutes} ${ampm}`;
    }

    return `${String(hours).padStart(2, '0')}:${minutes}:${seconds}`;
  } catch (error) {
    console.error('Error formatting time:', error);
    return 'Invalid time';
  }
}

/**
 * Format ISO date string to readable format
 */
export function formatISODate(isoString: string): string {
  try {
    return formatDate(new Date(isoString), 'MMM DD, YYYY');
  } catch (error) {
    console.error('Error formatting ISO date:', error);
    return 'Invalid date';
  }
}

// ============================================================================
// Number Formatters
// ============================================================================

/**
 * Format number with thousands separator
 */
export function formatNumber(
  num: number,
  decimals: number = 0,
  thousandsSeparator: string = ','
): string {
  try {
    const parts = num.toFixed(decimals).split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, thousandsSeparator);
    return parts.join('.');
  } catch (error) {
    console.error('Error formatting number:', error);
    return num.toString();
  }
}

/**
 * Format percentage
 */
export function formatPercentage(value: number, decimals: number = 2): string {
  try {
    return `${(value * 100).toFixed(decimals)}%`;
  } catch (error) {
    console.error('Error formatting percentage:', error);
    return '0%';
  }
}

/**
 * Format number with abbreviation (K, M, B, etc.)
 */
export function formatNumberShort(num: number, decimals: number = 1): string {
  try {
    const absNum = Math.abs(num);

    if (absNum >= 1e9) return `${(num / 1e9).toFixed(decimals)}B`;
    if (absNum >= 1e6) return `${(num / 1e6).toFixed(decimals)}M`;
    if (absNum >= 1e3) return `${(num / 1e3).toFixed(decimals)}K`;

    return num.toFixed(decimals);
  } catch (error) {
    console.error('Error formatting short number:', error);
    return num.toString();
  }
}

/**
 * Format as decimal with fixed precision
 */
export function formatDecimal(num: number, decimals: number = 2): string {
  try {
    return num.toFixed(decimals);
  } catch (error) {
    console.error('Error formatting decimal:', error);
    return num.toString();
  }
}

/**
 * Format as scientific notation
 */
export function formatScientific(num: number, decimals: number = 2): string {
  try {
    return num.toExponential(decimals);
  } catch (error) {
    console.error('Error formatting scientific notation:', error);
    return num.toString();
  }
}

// ============================================================================
// File Size Formatters
// ============================================================================

/**
 * Format bytes to human readable format
 */
export function formatFileSize(bytes: number, decimals: number = 2): string {
  try {
    if (bytes === 0) return '0 Bytes';

    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
  } catch (error) {
    console.error('Error formatting file size:', error);
    return 'Unknown';
  }
}

/**
 * Format upload progress
 */
export function formatUploadProgress(uploaded: number, total: number): string {
  const uploadedSize = formatFileSize(uploaded);
  const totalSize = formatFileSize(total);
  return `${uploadedSize} / ${totalSize}`;
}

/**
 * Convert file size string to bytes
 */
export function parseFileSize(sizeStr: string): number {
  try {
    const units: Record<string, number> = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 };
    const match = sizeStr.match(/^([\d.]+)\s*([A-Z]+)$/i);

    if (!match) return 0;

    const value = parseFloat(match[1]);
    const unit = match[2].toUpperCase();

    return value * (units[unit] || 1);
  } catch (error) {
    console.error('Error parsing file size:', error);
    return 0;
  }
}

// ============================================================================
// Currency Formatters
// ============================================================================

/**
 * Format as currency
 */
export function formatCurrency(
  amount: number,
  currency: string = 'USD',
  decimals: number = 2
): string {
  try {
    const formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency,
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });

    return formatter.format(amount);
  } catch (error) {
    console.error('Error formatting currency:', error);
    return `$${amount.toFixed(decimals)}`;
  }
}

/**
 * Format price with symbol
 */
export function formatPrice(amount: number, symbol: string = '$'): string {
  try {
    return `${symbol}${formatNumber(amount, 2)}`;
  } catch (error) {
    console.error('Error formatting price:', error);
    return `${symbol}0.00`;
  }
}

// ============================================================================
// String Formatters
// ============================================================================

/**
 * Format text to title case
 */
export function formatTitleCase(text: string): string {
  try {
    return text
      .toLowerCase()
      .split(' ')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  } catch (error) {
    console.error('Error formatting title case:', error);
    return text;
  }
}

/**
 * Format text to sentence case
 */
export function formatSentenceCase(text: string): string {
  try {
    return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
  } catch (error) {
    console.error('Error formatting sentence case:', error);
    return text;
  }
}

/**
 * Truncate text with ellipsis
 */
export function truncateText(text: string, maxLength: number = 50, suffix: string = '...'): string {
  try {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - suffix.length) + suffix;
  } catch (error) {
    console.error('Error truncating text:', error);
    return text;
  }
}

/**
 * Capitalize first letter
 */
export function capitalize(text: string): string {
  try {
    return text.charAt(0).toUpperCase() + text.slice(1);
  } catch (error) {
    console.error('Error capitalizing text:', error);
    return text;
  }
}

/**
 * Format phone number
 */
export function formatPhoneNumber(phone: string, format: string = '(XXX) XXX-XXXX'): string {
  try {
    const cleaned = phone.replace(/\D/g, '');

    if (cleaned.length !== 10) return phone;

    let result = format;
    let cleanIndex = 0;

    result = result.replace(/X/g, () => cleaned[cleanIndex++] || '');

    return result;
  } catch (error) {
    console.error('Error formatting phone number:', error);
    return phone;
  }
}

/**
 * Format URL slug
 */
export function formatSlug(text: string): string {
  try {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-');
  } catch (error) {
    console.error('Error formatting slug:', error);
    return text;
  }
}

/**
 * Format email for display
 */
export function formatEmailDisplay(email: string, showFull: boolean = true): string {
  try {
    if (showFull) return email;

    const [local] = email.split('@');
    const hiddenLength = Math.max(1, Math.floor(local.length / 2));
    const visible = local.substring(0, local.length - hiddenLength);
    const hidden = '*'.repeat(hiddenLength);

    return `${visible}${hidden}@${email.split('@')[1]}`;
  } catch (error) {
    console.error('Error formatting email display:', error);
    return email;
  }
}

// ============================================================================
// Data Structure Formatters
// ============================================================================

/**
 * Format array as readable string
 */
export function formatArray(
  arr: any[],
  separator: string = ', ',
  lastSeparator: string = ' and '
): string {
  try {
    if (arr.length === 0) return '';
    if (arr.length === 1) return arr[0].toString();
    if (arr.length === 2) return `${arr[0]}${lastSeparator}${arr[1]}`;

    return arr.slice(0, -1).join(separator) + `${lastSeparator}${arr[arr.length - 1]}`;
  } catch (error) {
    console.error('Error formatting array:', error);
    return '';
  }
}

/**
 * Format object as readable string
 */
export function formatObject(obj: Record<string, any>, indent: number = 2): string {
  try {
    return JSON.stringify(obj, null, indent);
  } catch (error) {
    console.error('Error formatting object:', error);
    return '{}';
  }
}

/**
 * Format duration in milliseconds to human readable
 */
export function formatDuration(ms: number): string {
  try {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(2)}s`;
    if (ms < 3600000) return `${(ms / 60000).toFixed(2)}m`;

    return `${(ms / 3600000).toFixed(2)}h`;
  } catch (error) {
    console.error('Error formatting duration:', error);
    return '0ms';
  }
}

/**
 * Format bandwidth
 */
export function formatBandwidth(bytesPerSecond: number): string {
  try {
    return `${formatFileSize(bytesPerSecond)}/s`;
  } catch (error) {
    console.error('Error formatting bandwidth:', error);
    return 'Unknown';
  }
}

// ============================================================================
// Specialized Formatters
// ============================================================================

/**
 * Format correlation coefficient
 */
export function formatCorrelation(coefficient: number, decimals: number = 3): string {
  try {
    return coefficient.toFixed(decimals);
  } catch (error) {
    console.error('Error formatting correlation:', error);
    return '0.000';
  }
}

/**
 * Format statistical measure
 */
export function formatStatistic(value: number, decimals: number = 2): string {
  try {
    return value.toFixed(decimals);
  } catch (error) {
    console.error('Error formatting statistic:', error);
    return '0.00';
  }
}

/**
 * Format confidence interval
 */
export function formatConfidenceInterval(
  lower: number,
  upper: number,
  decimals: number = 2
): string {
  try {
    return `[${lower.toFixed(decimals)}, ${upper.toFixed(decimals)}]`;
  } catch (error) {
    console.error('Error formatting confidence interval:', error);
    return '[]';
  }
}
