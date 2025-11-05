// src/components/shared/Button.tsx

import { forwardRef, ReactNode, Children, cloneElement, isValidElement } from 'react';
import type { ButtonHTMLAttributes, ReactElement } from 'react';
import { Loader2 } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

export type ButtonVariant =
  | 'primary'
  | 'secondary'
  | 'success'
  | 'danger'
  | 'outline'
  | 'ghost';

export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Button visual style variant */
  variant?: ButtonVariant;
  /** Button size */
  size?: ButtonSize;
  /** Show loading spinner and disable interaction */
  loading?: boolean;
  /** Icon to display before text */
  leftIcon?: LucideIcon;
  /** Icon to display after text */
  rightIcon?: LucideIcon;
  /** Make button full width */
  fullWidth?: boolean;
  /** Children content */
  children?: ReactNode;
  /** Loading text to show when loading is true */
  loadingText?: string;
  /** Additional class names */
  className?: string;
}

/**
 * Button - Reusable button component with variants and loading states
 * Features: Multiple variants, sizes, icons, loading states, full accessibility
 * Integrates with global.css utility classes for consistent styling
 *
 * @example
 * <Button variant="primary" loading={isLoading} leftIcon={Save}>
 *   Save Changes
 * </Button>
 */
const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      loading = false,
      leftIcon: LeftIcon,
      rightIcon: RightIcon,
      fullWidth = false,
      children,
      loadingText,
      className = '',
      disabled,
      type = 'button',
      ...props
    },
    ref
  ) => {
    // Combine class names based on props
    const getButtonClasses = (): string => {
      const classes = ['btn'];

      // Add variant class
      classes.push(`btn-${variant}`);

      // Add size class
      if (size !== 'md') {
        classes.push(`btn-${size}`);
      }

      // Add full width class
      if (fullWidth) {
        classes.push('w-full');
      }

      // Add custom classes
      if (className) {
        classes.push(className);
      }

      return classes.join(' ');
    };

    // Determine if button should be disabled
    const isDisabled = disabled || loading;

    // Button content based on loading state
    const buttonContent = (
      <>
        {loading && (
          <Loader2
            className={`w-4 h-4 animate-spin ${
              children || loadingText ? 'mr-2' : ''
            }`}
            aria-hidden="true"
          />
        )}

        {!loading && LeftIcon && (
          <LeftIcon
            className={`w-4 h-4 ${children ? 'mr-2' : ''}`}
            aria-hidden="true"
          />
        )}

        {loading && loadingText ? loadingText : children}

        {!loading && RightIcon && (
          <RightIcon
            className={`w-4 h-4 ${children ? 'ml-2' : ''}`}
            aria-hidden="true"
          />
        )}
      </>
    );

    return (
      <button
        ref={ref}
        type={type}
        className={getButtonClasses()}
        disabled={isDisabled}
        aria-busy={loading}
        aria-disabled={isDisabled}
        {...props}
      >
        {buttonContent}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;

// ============================================================================
// Icon-only button variant
// ============================================================================

export interface IconButtonProps
  extends Omit<ButtonProps, 'leftIcon' | 'rightIcon'> {
  icon: LucideIcon;
  label: string;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon: Icon, label, variant = 'ghost', className = '', ...props }, ref) => {
    const variantClass = `btn-${variant}`;

    return (
      <button
        ref={ref}
        className={`btn ${variantClass} inline-flex items-center justify-center p-2 hover:bg-opacity-80 transition-all ${className}`}
        aria-label={label}
        title={label}
        type="button"
        {...props}
      >
        <Icon className="w-5 h-5" aria-hidden="true" />
      </button>
    );
  }
);

IconButton.displayName = 'IconButton';

// ============================================================================
// Button Group component for grouping related buttons
// ============================================================================

export interface ButtonGroupProps {
  children: ReactNode;
  className?: string;
  orientation?: 'horizontal' | 'vertical';
  size?: ButtonSize;
  fullWidth?: boolean;
}

export const ButtonGroup = forwardRef<HTMLDivElement, ButtonGroupProps>(
  (
    {
      children,
      className = '',
      orientation = 'horizontal',
      fullWidth = false,
    },
    ref
  ) => {
    const orientationClass =
      orientation === 'vertical' ? 'flex-col' : 'flex-row';
    const widthClass = fullWidth ? 'w-full' : '';

    return (
      <div
        ref={ref}
        className={`inline-flex ${orientationClass} ${widthClass} ${className}`}
        role="group"
      >
        {Children.map(children, (child: ReactNode, index: number) => {
          if (!isValidElement(child)) return child;

          // Get total children count
          const totalChildren = Children.count(children);

          // Add border radius classes based on position
          const isFirst = index === 0;
          const isLast = index === totalChildren - 1;
          const isOnly = totalChildren === 1;

          let additionalClasses = '';

          if (isOnly) {
            // Single button - keep all borders
            return child;
          }

          if (orientation === 'horizontal') {
            if (!isFirst) additionalClasses += ' -ml-px';
            if (!isFirst && !isLast) additionalClasses += ' rounded-none';
            if (isFirst) additionalClasses += ' rounded-r-none';
            if (isLast) additionalClasses += ' rounded-l-none';
          } else {
            if (!isFirst) additionalClasses += ' -mt-px';
            if (!isFirst && !isLast) additionalClasses += ' rounded-none';
            if (isFirst) additionalClasses += ' rounded-b-none';
            if (isLast) additionalClasses += ' rounded-t-none';
          }

          const existingClassName = (child.props as { className?: string })
            .className || '';
          const newClassName = `${existingClassName} ${additionalClasses}`.trim();

          return cloneElement(child as ReactElement, {
            className: newClassName,
          });
        })}
      </div>
    );
  }
);

ButtonGroup.displayName = 'ButtonGroup';

// ============================================================================
// Preset Button Variants - Common use cases
// ============================================================================

export interface PrimaryButtonProps extends ButtonProps {
  variant?: never;
}

/**
 * PrimaryButton - Pre-configured primary action button
 */
export const PrimaryButton = forwardRef<
  HTMLButtonElement,
  PrimaryButtonProps
>((props, ref) => <Button ref={ref} variant="primary" {...props} />);

PrimaryButton.displayName = 'PrimaryButton';

/**
 * SecondaryButton - Pre-configured secondary action button
 */
export const SecondaryButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => <Button ref={ref} variant="secondary" {...props} />
);

SecondaryButton.displayName = 'SecondaryButton';

/**
 * DangerButton - Pre-configured danger/destructive action button
 */
export const DangerButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => <Button ref={ref} variant="danger" {...props} />
);

DangerButton.displayName = 'DangerButton';

/**
 * SuccessButton - Pre-configured success action button
 */
export const SuccessButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => <Button ref={ref} variant="success" {...props} />
);

SuccessButton.displayName = 'SuccessButton';

/**
 * GhostButton - Pre-configured ghost/transparent button
 */
export const GhostButton = forwardRef<HTMLButtonElement, ButtonProps>(
  (props, ref) => <Button ref={ref} variant="ghost" {...props} />
);

GhostButton.displayName = 'GhostButton';
