// src/utils/helpers.ts
/**
 * Helper Utilities
 * General utility functions for string manipulation, array operations, and common tasks
 */

// ============================================================================
// String Helpers
// ============================================================================

/**
 * Check if string is empty or whitespace
 */
export function isEmpty(str: string | null | undefined): boolean {
  return !str || str.trim().length === 0;
}

/**
 * Remove whitespace from string
 */
export function removeWhitespace(str: string): string {
  return str.replace(/\s/g, '');
}

/**
 * Replace all occurrences in string
 */
export function replaceAll(str: string, search: string, replacement: string): string {
  return str.split(search).join(replacement);
}

/**
 * Convert string to camelCase
 */
export function toCamelCase(str: string): string {
  return str
    .replace(/[-_\s](.)/g, (_, c) => c.toUpperCase())
    .replace(/^(.)/, (c) => c.toLowerCase());
}

/**
 * Convert string to kebab-case
 */
export function toKebabCase(str: string): string {
  return str
    .replace(/([a-z])([A-Z])/g, '$1-$2')
    .replace(/[\s_]/g, '-')
    .toLowerCase();
}

/**
 * Convert string to snake_case
 */
export function toSnakeCase(str: string): string {
  return str
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .replace(/[\s-]/g, '_')
    .toLowerCase();
}

/**
 * Repeat string n times
 */
export function repeatString(str: string, times: number): string {
  return str.repeat(Math.max(0, times));
}

/**
 * Reverse string
 */
export function reverseString(str: string): string {
  return str.split('').reverse().join('');
}

/**
 * Pad string to specified length
 */
export function padString(str: string, length: number, char: string = ' '): string {
  if (str.length >= length) return str;
  return char.repeat(length - str.length) + str;
}

/**
 * Generate random string
 */
export function generateRandomString(length: number = 10, chars?: string): string {
  const characters = chars || 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * characters.length));
  }
  return result;
}

/**
 * Generate UUID v4
 */
export function generateUUID(): string {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// ============================================================================
// Array Helpers
// ============================================================================

/**
 * Check if array is empty
 */
export function isArrayEmpty(arr: any[]): boolean {
  return !arr || arr.length === 0;
}

/**
 * Get unique values from array
 */
export function getUniqueArray<T>(arr: T[]): T[] {
  return Array.from(new Set(arr));
}

/**
 * Get unique values from array of objects
 */
export function getUniqueByProperty<T>(arr: T[], property: keyof T): T[] {
  const seen = new Set();
  return arr.filter((item) => {
    const key = item[property];
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/**
 * Flatten nested array
 */
export function flattenArray<T>(arr: Array<T | T[]>): T[] {
  return arr.reduce<T[]>((flat, item) => {
    // ✅ Check if item is an array before treating it as one
    if (Array.isArray(item)) {
      return flat.concat(flattenArray(item));
    }
    return flat.concat(item);
  }, []);
}

/**
 * Chunk array into smaller arrays
 */
export function chunkArray<T>(arr: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    chunks.push(arr.slice(i, i + size));
  }
  return chunks;
}

/**
 * Group array by property
 */
export function groupByProperty<T>(arr: T[], property: keyof T): Record<string, T[]> {
  return arr.reduce((groups, item) => {
    const key = String(item[property]);
    if (!groups[key]) groups[key] = [];
    groups[key].push(item);
    return groups;
  }, {} as Record<string, T[]>);
}

/**
 * Sort array by property
 */
export function sortByProperty<T>(arr: T[], property: keyof T, ascending: boolean = true): T[] {
  return [...arr].sort((a, b) => {
    if (a[property] < b[property]) return ascending ? -1 : 1;
    if (a[property] > b[property]) return ascending ? 1 : -1;
    return 0;
  });
}

/**
 * Filter array by property value
 */
export function filterByProperty<T>(arr: T[], property: keyof T, value: any): T[] {
  return arr.filter((item) => item[property] === value);
}

/**
 * Find item in array
 */
export function findInArray<T>(arr: T[], predicate: (item: T) => boolean): T | undefined {
  return arr.find(predicate);
}

/**
 * Find index of item in array
 */
export function findIndexInArray<T>(arr: T[], predicate: (item: T) => boolean): number {
  return arr.findIndex(predicate);
}

/**
 * Remove item from array
 */
export function removeFromArray<T>(arr: T[], item: T): T[] {
  return arr.filter((i) => i !== item);
}

/**
 * Remove item by predicate from array
 */
export function removeFromArrayByPredicate<T>(arr: T[], predicate: (item: T) => boolean): T[] {
  return arr.filter((item) => !predicate(item));
}

/**
 * Shuffle array
 */
export function shuffleArray<T>(arr: T[]): T[] {
  const newArr = [...arr];
  for (let i = newArr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArr[i], newArr[j]] = [newArr[j], newArr[i]];
  }
  return newArr;
}

/**
 * Sum array of numbers
 */
export function sumArray(arr: number[]): number {
  return arr.reduce((sum, num) => sum + num, 0);
}

/**
 * Average of array of numbers
 */
export function averageArray(arr: number[]): number {
  return arr.length === 0 ? 0 : sumArray(arr) / arr.length;
}

/**
 * Get min value from array
 */
export function getMinFromArray(arr: number[]): number {
  return Math.min(...arr);
}

/**
 * Get max value from array
 */
export function getMaxFromArray(arr: number[]): number {
  return Math.max(...arr);
}

// ============================================================================
// Object Helpers
// ============================================================================

/**
 * Deep clone object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') return obj;

  if (obj instanceof Date) return new Date(obj.getTime()) as any;
  if (obj instanceof Array) return obj.map((item) => deepClone(item)) as any;

  const cloned = {} as T;
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }
  return cloned;
}

/**
 * Merge objects
 */
export function mergeObjects<T extends Record<string, any>>(target: T, source: Partial<T>): T {
  return { ...target, ...source };
}

/**
 * Deep merge objects
 */
export function deepMergeObjects<T extends Record<string, any>>(target: T, source: Partial<T>): T {
  const result = { ...target };

  for (const key in source) {
    if (source.hasOwnProperty(key)) {
      const sourceValue = source[key];
      const targetValue = result[key];

      if (sourceValue && typeof sourceValue === 'object' && targetValue && typeof targetValue === 'object') {
        result[key] = deepMergeObjects(targetValue, sourceValue);
      } else {
        result[key] = sourceValue as any;
      }
    }
  }

  return result;
}

/**
 * Omit properties from object
 */
export function omitProperties<T extends Record<string, any>>(
  obj: T,
  keys: (keyof T)[]
): Partial<T> {
  const result: Partial<T> = {};

  for (const key in obj) {
    if (obj.hasOwnProperty(key) && !keys.includes(key as keyof T)) {
      result[key as keyof T] = obj[key];
    }
  }

  return result;
}

/**
 * Pick properties from object
 */
export function pickProperties<T extends Record<string, any>>(obj: T, keys: (keyof T)[]): Partial<T> {
  const result: Partial<T> = {};

  for (const key of keys) {
    if (obj.hasOwnProperty(key)) {
      result[key] = obj[key];
    }
  }

  return result;
}

/**
 * Get nested property value
 */
export function getNestedProperty(obj: any, path: string, defaultValue?: any): any {
  const keys = path.split('.');
  let value = obj;

  for (const key of keys) {
    value = value?.[key];
  }

  return value !== undefined ? value : defaultValue;
}

/**
 * Set nested property value
 */
export function setNestedProperty(obj: any, path: string, value: any): any {
  const keys = path.split('.');
  let current = obj;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];
    current[key] = current[key] || {};
    current = current[key];
  }

  current[keys[keys.length - 1]] = value;
  return obj;
}

// ============================================================================
// Type & Check Helpers
// ============================================================================

/**
 * Check if value is null or undefined
 */
export function isNullOrUndefined(value: any): boolean {
  return value === null || value === undefined;
}

/**
 * Check if value is of type
 */
export function isOfType<T>(value: any, type: string): value is T {
  return typeof value === type;
}

/**
 * Check if value is empty
 */
export function isEmpty2(value: any): boolean {
  if (value === null || value === undefined) return true;
  if (typeof value === 'string') return value.trim().length === 0;
  if (Array.isArray(value)) return value.length === 0;
  if (typeof value === 'object') return Object.keys(value).length === 0;
  return false;
}

/**
 * Check if browser is online
 */
export function isOnline(): boolean {
  return typeof navigator !== 'undefined' && navigator.onLine;
}

// ============================================================================
// Async Helpers
// ============================================================================

/**
 * Delay execution
 */
export function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Retry async operation
 */
export async function retryAsync<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3,
  delayMs: number = 1000
): Promise<T> {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === maxAttempts - 1) throw error;
      await delay(delayMs);
    }
  }
  throw new Error('Retry failed');
}

/**
 * Timeout async operation
 */
export function timeoutAsync<T>(promise: Promise<T>, ms: number): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error('Operation timeout')), ms)
    ),
  ]);
}

// ============================================================================
// Comparison Helpers
// ============================================================================

/**
 * Deep equality check
 */
export function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;

  if (a == null || b == null) return false;

  if (typeof a !== 'object' || typeof b !== 'object') return false;

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (!deepEqual(a[key], b[key])) return false;
  }

  return true;
}

/**
 * Shallow equality check
 */
export function shallowEqual<T extends Record<string, any>>(a: T, b: T): boolean {
  if (a === b) return true;

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (a[key] !== b[key]) return false;
  }

  return true;
}

// ============================================================================
// Math Helpers
// ============================================================================

/**
 * Round number to decimal places
 */
export function roundTo(num: number, decimals: number = 0): number {
  return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
}

/**
 * Clamp number between min and max
 */
export function clamp(num: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, num));
}

/**
 * Map value from one range to another
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
}

/**
 * Normalize value between 0 and 1
 */
export function normalize(value: number, min: number, max: number): number {
  return (value - min) / (max - min);
}

/**
 * Check if number is even
 */
export function isEven(num: number): boolean {
  return num % 2 === 0;
}

/**
 * Check if number is odd
 */
export function isOdd(num: number): boolean {
  return num % 2 !== 0;
}

/**
 * Calculate percentage
 */
export function calculatePercentage(value: number, total: number): number {
  return (value / total) * 100;
}

/**
 * Calculate percentage change
 */
export function calculatePercentageChange(oldValue: number, newValue: number): number {
  return ((newValue - oldValue) / oldValue) * 100;
}

// ============================================================================
// DOM Helpers
// ============================================================================

/**
 * Check if element is in viewport
 */
export function isElementInViewport(el: Element): boolean {
  const rect = el.getBoundingClientRect();
  return (
    rect.top >= 0 &&
    rect.left >= 0 &&
    rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
    rect.right <= (window.innerWidth || document.documentElement.clientWidth)
  );
}

/**
 * Copy text to clipboard
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
}

/**
 * Get scroll position
 */
export function getScrollPosition(): { x: number; y: number } {
  return {
    x: window.scrollX || document.documentElement.scrollLeft,
    y: window.scrollY || document.documentElement.scrollTop,
  };
}

/**
 * Scroll to element
 */
export function scrollToElement(el: Element, smooth: boolean = true): void {
  el.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
}

// ============================================================================
// Storage Helpers
// ============================================================================

/**
 * Get item from localStorage
 */
export function getLocalStorageItem<T>(key: string, defaultValue?: T): T | null {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : (defaultValue ?? null);
  } catch (error) {
    console.error(`Error reading from localStorage:`, error);
    return defaultValue ?? null;
  }
}

/**
 * Set item in localStorage
 */
export function setLocalStorageItem<T>(key: string, value: T): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error(`Error writing to localStorage:`, error);
    return false;
  }
}

/**
 * Remove item from localStorage
 */
export function removeLocalStorageItem(key: string): boolean {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error(`Error removing from localStorage:`, error);
    return false;
  }
}

/**
 * Clear localStorage
 */
export function clearLocalStorage(): boolean {
  try {
    localStorage.clear();
    return true;
  } catch (error) {
    console.error(`Error clearing localStorage:`, error);
    return false;
  }
}
