// src/hooks/useDebounce.ts

import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Configuration options for debounce and throttle hooks
 */
interface DebounceOptions {
  leading?: boolean;
  trailing?: boolean;
  maxWait?: number;
}

/**
 * State object returned by useDebouncedValue
 */
interface DebounceState<T> {
  value: T;
  isPending: boolean;
  isExecuting: boolean;
}

// ============================================================================
// Basic Debounce Hooks
// ============================================================================

/**
 * Basic debounce hook that delays updating a value
 * Useful for search inputs, form filters, and other rapid-fire value changes
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default: 500ms)
 * @returns Debounced value
 *
 * @example
 * const debouncedSearchTerm = useDebounce(searchTerm, 500);
 */
export const useDebounce = <T,>(value: T, delay: number = 500): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    // Set up timer
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    // Cleanup timer if value or delay changes
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

/**
 * Advanced debounce hook with leading and trailing edge options
 * @param value - The value to debounce
 * @param delay - Delay in milliseconds (default: 500ms)
 * @param options - Configuration options (leading, trailing, maxWait)
 * @returns Debounced value with state information
 *
 * @example
 * const { value, isPending, isExecuting } = useDebouncedValue(
 *   searchTerm,
 *   500,
 *   { leading: false, trailing: true }
 * );
 */
export const useDebouncedValue = <T,>(
  value: T,
  delay: number = 500,
  options: DebounceOptions = {}
): DebounceState<T> => {
  const { leading = false, trailing = true, maxWait } = options;

  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  const [isPending, setIsPending] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const maxWaitTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastCallTimeRef = useRef<number>(Date.now());
  const isInitialRef = useRef(true);

  useEffect(() => {
    const now = Date.now();
    const timeSinceLastCall = now - lastCallTimeRef.current;

    // Leading edge - execute immediately on first call
    if (leading && isInitialRef.current) {
      setIsExecuting(true);
      setDebouncedValue(value);
      setIsExecuting(false);
      isInitialRef.current = false;
    }

    lastCallTimeRef.current = now;
    setIsPending(true);

    // Clear existing timeouts
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    if (maxWaitTimeoutRef.current) {
      clearTimeout(maxWaitTimeoutRef.current);
    }

    // Trailing edge - execute after delay
    if (trailing) {
      timeoutRef.current = setTimeout(() => {
        setIsExecuting(true);
        setDebouncedValue(value);
        setIsExecuting(false);
        setIsPending(false);
      }, delay);
    }

    // Max wait - force execution if waiting too long
    if (maxWait && timeSinceLastCall >= maxWait) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      setIsExecuting(true);
      setDebouncedValue(value);
      setIsExecuting(false);
      setIsPending(false);
    } else if (maxWait) {
      maxWaitTimeoutRef.current = setTimeout(() => {
        setIsExecuting(true);
        setDebouncedValue(value);
        setIsExecuting(false);
        setIsPending(false);
      }, maxWait);
    }

    // Cleanup
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (maxWaitTimeoutRef.current) {
        clearTimeout(maxWaitTimeoutRef.current);
      }
    };
  }, [value, delay, leading, trailing, maxWait]);

  return {
    value: debouncedValue,
    isPending,
    isExecuting,
  };
};

// ============================================================================
// Debounce Callback Hooks
// ============================================================================

/**
 * ✅ FIXED: Debounce hook for callback functions
 * Useful for debouncing API calls, event handlers, or async operations
 * @param callback - Function to debounce
 * @param delay - Delay in milliseconds (default: 500ms)
 * @returns Debounced callback function
 *
 * @example
 * const { execute, cancel, pending } = useDebouncedCallback(
 *   async (searchTerm: string) => {
 *     const results = await api.search(searchTerm);
 *   },
 *   500
 * );
 */
export const useDebouncedCallback = <T extends unknown[] = []>(
  callback: (...args: T) => void | Promise<void>,
  delay: number = 500
): {
  execute: (...args: T) => void;
  cancel: () => void;
  pending: boolean;
  flush: (...args: T) => void;
} => {
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [pending, setPending] = useState(false);
  const lastArgsRef = useRef<T | null>(null);

  const execute = useCallback(
    (...args: T) => {
      setPending(true);
      lastArgsRef.current = args;

      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(async () => {
        await callback(...args);
        setPending(false);
      }, delay);
    },
    [callback, delay]
  );

  const cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      setPending(false);
    }
  }, []);

  // ✅ FIXED: Proper type-safe flush implementation
  const flush = useCallback(
    (...args: T) => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      // Use provided args if available, otherwise use last saved args
      const argsToUse =
        lastArgsRef.current !== null ? lastArgsRef.current : args;
      callback(...argsToUse);
      setPending(false);
    },
    [callback]
  );

  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return { execute, cancel, pending, flush };
};

/**
 * ✅ FIXED: Advanced debounce hook with configurable leading/trailing edges
 * @param callback - Function to debounce
 * @param delay - Delay in milliseconds
 * @param options - Configuration options
 * @returns Debounced callback with enhanced controls
 *
 * @example
 * const { execute, cancel, flush, pending } = useDebouncedCallbackAdvanced(
 *   async (term: string) => {
 *     return await api.search(term);
 *   },
 *   500,
 *   { leading: false, trailing: true, maxWait: 2000 }
 * );
 */
export const useDebouncedCallbackAdvanced = <
  T extends unknown[] = [],
  R = void
>(
  callback: (...args: T) => R | Promise<R>,
  delay: number = 500,
  options: DebounceOptions = {}
): {
  execute: (...args: T) => Promise<R | undefined>;
  cancel: () => void;
  flush: (...args: T) => Promise<R | undefined>;
  pending: boolean;
  callCount: number;
  lastArgs?: T;
} => {
  const { leading = false, trailing = true, maxWait } = options;

  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const maxWaitTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [pending, setPending] = useState(false);
  const [callCount, setCallCount] = useState(0);
  const lastArgsRef = useRef<T | null>(null);
  const lastCallTimeRef = useRef<number>(Date.now());
  const isInitialRef = useRef(true);

  const executeCallback = useCallback(
    async (...args: T): Promise<R | undefined> => {
      try {
        setPending(false);
        const result = await callback(...args);
        return result;
      } catch (error) {
        console.error('[useDebouncedCallbackAdvanced] Callback error:', error);
        return undefined;
      }
    },
    [callback]
  );

  const execute = useCallback(
    async (...args: T): Promise<R | undefined> => {
      const now = Date.now();
      const timeSinceLastCall = now - lastCallTimeRef.current;

      lastArgsRef.current = args;
      lastCallTimeRef.current = now;
      setCallCount((prev) => prev + 1);

      // Leading edge execution
      if (leading && isInitialRef.current) {
        isInitialRef.current = false;
        return executeCallback(...args);
      }

      setPending(true);

      // Clear existing timeouts
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (maxWaitTimeoutRef.current) {
        clearTimeout(maxWaitTimeoutRef.current);
      }

      // Create promise for result
      return new Promise<R | undefined>((resolve) => {
        // Trailing edge timeout
        if (trailing) {
          timeoutRef.current = setTimeout(async () => {
            const result = await executeCallback(...args);
            resolve(result);
          }, delay);
        }

        // Max wait timeout
        if (maxWait) {
          if (timeSinceLastCall >= maxWait) {
            if (timeoutRef.current) {
              clearTimeout(timeoutRef.current);
            }
            executeCallback(...args).then(resolve);
          } else {
            maxWaitTimeoutRef.current = setTimeout(async () => {
              const result = await executeCallback(...args);
              resolve(result);
            }, maxWait);
          }
        }

        // If neither trailing nor maxWait, resolve immediately
        if (!trailing && !maxWait) {
          resolve(undefined);
        }
      });
    },
    [executeCallback, delay, leading, trailing, maxWait]
  );

  const cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    if (maxWaitTimeoutRef.current) {
      clearTimeout(maxWaitTimeoutRef.current);
    }
    setPending(false);
  }, []);

  // ✅ FIXED: Proper type-safe flush implementation
  const flush = useCallback(
    async (...args: T): Promise<R | undefined> => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (maxWaitTimeoutRef.current) {
        clearTimeout(maxWaitTimeoutRef.current);
      }
      // Use provided args if available, otherwise use last saved args
      const argsToUse =
        lastArgsRef.current !== null ? lastArgsRef.current : args;
      return executeCallback(...argsToUse);
    },
    [executeCallback]
  );

  // Cleanup
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (maxWaitTimeoutRef.current) {
        clearTimeout(maxWaitTimeoutRef.current);
      }
    };
  }, []);

  return {
    execute,
    cancel,
    flush,
    pending,
    callCount,
    lastArgs: lastArgsRef.current || undefined,
  };
};

// ============================================================================
// Specialized Debounce Hooks
// ============================================================================

/**
 * Specialized debounce hook for search operations
 * Combines value debouncing with callback execution
 * @param value - Search value
 * @param onSearch - Callback when search should execute
 * @param delay - Delay in milliseconds
 * @returns Search state and controls
 *
 * @example
 * const { debouncedValue, isSearching, searchResult } = useDebouncedSearch(
 *   searchTerm,
 *   async (term) => {
 *     return await api.search(term);
 *   },
 *   300
 * );
 */
export const useDebouncedSearch = <T,>(
  value: string,
  onSearch: (term: string) => T | Promise<T>,
  delay: number = 300
): {
  debouncedValue: string;
  isSearching: boolean;
  searchResult?: T;
  clearSearch: () => void;
} => {
  const [debouncedValue, setDebouncedValue] = useState(value);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<T>();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!value.trim()) {
      setDebouncedValue('');
      setSearchResult(undefined);
      return;
    }

    setIsSearching(true);

    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    timeoutRef.current = setTimeout(async () => {
      setDebouncedValue(value);
      try {
        const result = await onSearch(value);
        setSearchResult(result);
      } catch (error) {
        console.error('[useDebouncedSearch] Search error:', error);
      } finally {
        setIsSearching(false);
      }
    }, delay);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, onSearch, delay]);

  const clearSearch = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setDebouncedValue('');
    setSearchResult(undefined);
    setIsSearching(false);
  }, []);

  return {
    debouncedValue,
    isSearching,
    searchResult,
    clearSearch,
  };
};

// ============================================================================
// Throttle Hook
// ============================================================================

/**
 * Throttle hook - ensures function executes at most once per interval
 * Useful for scroll, resize, and other frequent events
 * @param value - Value to throttle
 * @param interval - Interval in milliseconds
 * @returns Throttled value
 *
 * @example
 * const throttledScrollPosition = useThrottle(scrollY, 500);
 */
export const useThrottle = <T,>(value: T, interval: number = 500): T => {
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastUpdatedRef = useRef<number>(Date.now());

  useEffect(() => {
    const now = Date.now();

    if (now >= lastUpdatedRef.current + interval) {
      lastUpdatedRef.current = now;
      setThrottledValue(value);
      return;
    }

    const timeoutId = setTimeout(
      () => {
        lastUpdatedRef.current = Date.now();
        setThrottledValue(value);
      },
      interval - (now - lastUpdatedRef.current)
    );

    return () => clearTimeout(timeoutId);
  }, [value, interval]);

  return throttledValue;
};

// ============================================================================
// Standalone Debounce Function (Non-Hook)
// ============================================================================

/**
 * ✅ FIXED: Utility function to create standalone debounce function (not a hook)
 * Can be used outside of React components
 * @param callback - Function to debounce
 * @param delay - Delay in milliseconds
 * @returns Debounced function
 *
 * @example
 * const debouncedSearch = createDebouncedFunction(
 *   async (term: string) => {
 *     const results = await api.search(term);
 *   },
 *   500
 * );
 *
 * debouncedSearch('search term');
 */
export const createDebouncedFunction = <T extends unknown[] = [], R = void>(
  callback: (...args: T) => R | Promise<R>,
  delay: number = 500
): {
  (...args: T): Promise<R | undefined>;
  cancel: () => void;
  flush: (...args: T) => Promise<R | undefined>;
} => {
  let timeoutId: NodeJS.Timeout | null = null;
  let lastArgs: T | null = null;

  const debounced = async (...args: T): Promise<R | undefined> => {
    lastArgs = args;

    if (timeoutId) {
      clearTimeout(timeoutId);
    }

    return new Promise((resolve) => {
      timeoutId = setTimeout(async () => {
        try {
          const result = await callback(...args);
          resolve(result);
        } catch (error) {
          console.error('[createDebouncedFunction] Error:', error);
          resolve(undefined);
        } finally {
          timeoutId = null;
        }
      }, delay);
    });
  };

  debounced.cancel = () => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
  };

  // ✅ FIXED: Proper type-safe flush implementation
  debounced.flush = async (...args: T): Promise<R | undefined> => {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    // Use provided args if available, otherwise use last saved args
    const argsToUse = lastArgs !== null ? lastArgs : args;
    try {
      return await callback(...argsToUse);
    } catch (error) {
      console.error('[createDebouncedFunction] Flush error:', error);
      return undefined;
    }
  };

  return debounced;
};

// ============================================================================
// Type Exports
// ============================================================================

export type UseDebounceReturn<T> = T;
export type UseDebouncedValueReturn<T> = DebounceState<T>;
export type UseDebouncedCallbackReturn<T extends unknown[] = []> = ReturnType<
  typeof useDebouncedCallback<T>
>;
export type UseDebouncedCallbackAdvancedReturn<
  T extends unknown[] = [],
  R = void
> = ReturnType<typeof useDebouncedCallbackAdvanced<T, R>>;
export type UseDebouncedSearchReturn<T> = ReturnType<
  typeof useDebouncedSearch<T>
>;
export type UseThrottleReturn<T> = T;
export type CreateDebouncedFunctionReturn<
  T extends unknown[] = [],
  R = void
> = ReturnType<typeof createDebouncedFunction<T, R>>;
