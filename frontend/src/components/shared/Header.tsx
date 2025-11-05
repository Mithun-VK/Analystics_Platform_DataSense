// src/components/shared/Header.tsx

/**
 * Header Component - Modern Navigation Bar
 * ✅ ENHANCED: Clean design, smooth animations, dark mode, accessibility
 * Features: Sticky header, responsive mobile menu, user dropdown, theme toggle
 */

import { useState, useEffect, useRef, useCallback, memo } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import {
  Menu,
  X,
  Bell,
  Settings,
  User,
  LogOut,
  ChevronDown,
  Search,
  HelpCircle,
  Moon,
  Sun,
  BarChart3,
  Home,
  Database,
} from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { uiStore } from '@/store/uiStore';

// ============================================================================
// Type Definitions
// ============================================================================

interface HeaderProps {
  showSearch?: boolean;
  showNotifications?: boolean;
  logo?: React.ReactNode;
  navItems?: NavItem[];
}

interface NavItem {
  label: string;
  path: string;
  icon?: React.ElementType;
}

// ============================================================================
// Memoized Sub-Components
// ============================================================================

/**
 * ✅ Logo Component
 */
const HeaderLogo = memo(({ logo }: { logo?: React.ReactNode }) => (
  <Link to="/" className="flex items-center gap-3 hover:opacity-80 transition-opacity flex-shrink-0">
    {logo ? (
      logo
    ) : (
      <>
        <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-lg flex items-center justify-center shadow-md hover:shadow-lg transition-shadow">
          <BarChart3 className="w-6 h-6 text-white" />
        </div>
        <span className="text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent hidden sm:block">
          DataSense
        </span>
      </>
    )}
  </Link>
));

HeaderLogo.displayName = 'HeaderLogo';

/**
 * ✅ Navigation Links Component
 */
interface NavLinksProps {
  items: NavItem[];
  isActive: (path: string) => boolean;
}

const NavLinks = memo<NavLinksProps>(({ items, isActive }) => (
  <div className="hidden lg:flex items-center gap-1">
    {items.map((item) => {
      const Icon = item.icon;
      const active = isActive(item.path);

      return (
        <Link
          key={item.path}
          to={item.path}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all duration-200 ${
            active
              ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          {Icon && <Icon className="w-4 h-4" />}
          <span>{item.label}</span>
        </Link>
      );
    })}
  </div>
));

NavLinks.displayName = 'NavLinks';

/**
 * ✅ Search Bar Component
 */
const SearchBar = memo(() => (
  <div className="hidden md:flex items-center relative w-full max-w-xs">
    <Search className="absolute left-3 w-4 h-4 text-gray-400 pointer-events-none" />
    <input
      type="search"
      placeholder="Search datasets, analyses..."
      className="w-full pl-10 pr-4 py-2 bg-gray-100 dark:bg-gray-800 border border-transparent dark:border-gray-700 rounded-lg text-sm text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white dark:focus:bg-gray-700 transition-colors"
      aria-label="Search"
    />
  </div>
));

SearchBar.displayName = 'SearchBar';

/**
 * ✅ Theme Toggle Button
 */
interface ThemeToggleProps {
  isDarkMode: boolean;
  onToggle: () => void;
}

const ThemeToggle = memo<ThemeToggleProps>(({ isDarkMode, onToggle }) => (
  <button
    onClick={onToggle}
    className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
    aria-label="Toggle theme"
  >
    {isDarkMode ? (
      <Sun className="w-5 h-5" />
    ) : (
      <Moon className="w-5 h-5" />
    )}
  </button>
));

ThemeToggle.displayName = 'ThemeToggle';

/**
 * ✅ Notifications Bell
 */
interface NotificationBellProps {
  unreadCount: number;
  onClick: () => void;
}

const NotificationBell = memo<NotificationBellProps>(({ unreadCount, onClick }) => (
  <button
    onClick={onClick}
    className="relative p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
    aria-label={`Notifications ${unreadCount > 0 ? `(${unreadCount} unread)` : ''}`}
  >
    <Bell className="w-5 h-5 text-gray-700 dark:text-gray-300" />
    {unreadCount > 0 && (
      <span className="absolute top-0 right-0 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center">
        {unreadCount > 9 ? '9+' : unreadCount}
      </span>
    )}
  </button>
));

NotificationBell.displayName = 'NotificationBell';

/**
 * ✅ User Menu Dropdown
 */
interface UserMenuProps {
  user: any;
  isOpen: boolean;
  onLogout: () => void;
  onNavigate: (path: string) => void;
  onClose: () => void;
}

const UserMenu = memo<UserMenuProps>(
  ({ user, isOpen, onLogout, onNavigate, onClose }) => {
    const userInitial = user?.fullName?.charAt(0).toUpperCase() || 'U';

    if (!isOpen) return null;

    return (
      <div className="absolute top-full right-0 mt-2 w-56 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden z-50 animate-in fade-in slide-in-from-top-2 duration-200">
        {/* User Info */}
        <div className="px-4 py-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-semibold">
              {userInitial}
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold text-gray-900 dark:text-white truncate">
                {user?.fullName || 'User'}
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 truncate">
                {user?.email}
              </p>
            </div>
          </div>
        </div>

        {/* Menu Items */}
        <div className="py-2">
          <button
            onClick={() => {
              onNavigate('/profile');
              onClose();
            }}
            className="w-full px-4 py-2.5 flex items-center gap-3 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
          >
            <User className="w-4 h-4" />
            <span>My Profile</span>
          </button>

          <button
            onClick={() => {
              onNavigate('/settings');
              onClose();
            }}
            className="w-full px-4 py-2.5 flex items-center gap-3 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm"
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </button>

          <button
            onClick={() => {
              onNavigate('/help');
              onClose();
            }}
            className="w-full px-4 py-2.5 flex items-center gap-3 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-sm lg:hidden"
          >
            <HelpCircle className="w-4 h-4" />
            <span>Help Center</span>
          </button>
        </div>

        {/* Logout */}
        <div className="border-t border-gray-200 dark:border-gray-700 py-2">
          <button
            onClick={() => {
              onLogout();
              onClose();
            }}
            className="w-full px-4 py-2.5 flex items-center gap-3 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-sm font-medium"
          >
            <LogOut className="w-4 h-4" />
            <span>Sign Out</span>
          </button>
        </div>
      </div>
    );
  }
);

UserMenu.displayName = 'UserMenu';

/**
 * ✅ Mobile Menu
 */
interface MobileMenuProps {
  isOpen: boolean;
  navItems: NavItem[];
  isActive: (path: string) => boolean;
  showSearch: boolean;
  onNavigate: (path: string) => void;
  onClose: () => void;
}

const MobileMenu = memo<MobileMenuProps>(
  ({ isOpen, navItems, isActive, showSearch, onNavigate, onClose }) => {
    if (!isOpen) return null;

    return (
      <>
        {/* Overlay */}
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />

        {/* Menu */}
        <div className="fixed top-16 left-0 right-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 z-50 lg:hidden animate-in slide-in-from-top duration-200">
          <div className="p-4 space-y-4 max-h-[calc(100vh-4rem)] overflow-y-auto">
            {/* Mobile Search */}
            {showSearch && (
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                <input
                  type="search"
                  placeholder="Search..."
                  className="w-full pl-10 pr-4 py-2 bg-gray-100 dark:bg-gray-700 border border-transparent rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            )}

            {/* Mobile Nav Links */}
            <nav className="space-y-1">
              {navItems.map((item) => {
                const Icon = item.icon;
                const active = isActive(item.path);

                return (
                  <button
                    key={item.path}
                    onClick={() => {
                      onNavigate(item.path);
                      onClose();
                    }}
                    className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg font-medium text-sm transition-colors ${
                      active
                        ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    {Icon && <Icon className="w-5 h-5" />}
                    <span>{item.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </>
    );
  }
);

MobileMenu.displayName = 'MobileMenu';

// ============================================================================
// Main Header Component
// ============================================================================

/**
 * ✅ Enhanced Header Component
 */
const Header = memo<HeaderProps>(
  ({
    showSearch = true,
    showNotifications = true,
    logo,
    navItems = [],
  }) => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const { pathname } = useLocation();

    const isDarkMode = uiStore((state) => state.isDarkMode);
    const setDarkMode = uiStore((state) => state.setDarkMode);

    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
    const [isScrolled, setIsScrolled] = useState(false);
    const [unreadCount] = useState(3);

    const userMenuRef = useRef<HTMLDivElement>(null);

    // ========================================================================
    // Default Navigation Items
    // ========================================================================

    const defaultNavItems: NavItem[] = [
      { label: 'Dashboard', path: '/dashboard', icon: Home },
      { label: 'Datasets', path: '/datasets', icon: Database },
      { label: 'Visualizations', path: '/visualizations', icon: BarChart3 },
    ];

    const navigationItems = navItems.length > 0 ? navItems : defaultNavItems;

    // ========================================================================
    // Effects
    // ========================================================================

    // Handle scroll for sticky effect
    useEffect(() => {
      const handleScroll = () => {
        setIsScrolled(window.scrollY > 10);
      };

      window.addEventListener('scroll', handleScroll);
      return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    // Close menus on route change
    useEffect(() => {
      setIsMobileMenuOpen(false);
      setIsUserMenuOpen(false);
    }, [pathname]);

    // Close user menu on outside click
    useEffect(() => {
      const handleClickOutside = (e: MouseEvent) => {
        if (
          userMenuRef.current &&
          !userMenuRef.current.contains(e.target as Node)
        ) {
          setIsUserMenuOpen(false);
        }
      };

      if (isUserMenuOpen) {
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
      }
    }, [isUserMenuOpen]);

    // Prevent body scroll when mobile menu is open
    useEffect(() => {
      document.body.style.overflow = isMobileMenuOpen ? 'hidden' : '';
      return () => {
        document.body.style.overflow = '';
      };
    }, [isMobileMenuOpen]);

    // ========================================================================
    // Handlers
    // ========================================================================

    const handleLogout = useCallback(async () => {
      try {
        await logout();
        navigate('/login');
      } catch (error) {
        console.error('Logout failed:', error);
      }
    }, [logout, navigate]);

    const toggleTheme = useCallback(() => {
      setDarkMode(!isDarkMode);
    }, [isDarkMode, setDarkMode]);

    const handleNavigate = useCallback(
      (path: string) => {
        navigate(path);
      },
      [navigate]
    );

    const isActive = useCallback(
      (path: string) => pathname === path,
      [pathname]
    );

    // ========================================================================
    // Render
    // ========================================================================

    return (
      <>
        {/* Main Header */}
        <header
          className={`sticky top-0 z-50 bg-white dark:bg-gray-800 transition-all duration-200 ${
            isScrolled
              ? 'shadow-md border-b border-gray-200 dark:border-gray-700'
              : 'border-b border-gray-100 dark:border-gray-700/50'
          }`}
          role="banner"
        >
          <nav
            className="max-w-full px-4 sm:px-6 lg:px-8 py-3 flex items-center justify-between gap-4"
            aria-label="Main navigation"
          >
            {/* Left Section - Logo & Nav */}
            <div className="flex items-center gap-6 min-w-0">
              {/* Mobile Menu Button */}
              <button
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                className="p-2 rounded-lg lg:hidden hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-gray-700 dark:text-gray-300"
                aria-label="Toggle menu"
                aria-expanded={isMobileMenuOpen}
              >
                {isMobileMenuOpen ? (
                  <X className="w-6 h-6" />
                ) : (
                  <Menu className="w-6 h-6" />
                )}
              </button>

              {/* Logo */}
              <HeaderLogo logo={logo} />

              {/* Desktop Nav */}
              <NavLinks items={navigationItems} isActive={isActive} />
            </div>

            {/* Middle Section - Search */}
            {showSearch && <SearchBar />}

            {/* Right Section - Actions */}
            <div className="flex items-center gap-2">
              {/* Help Button */}
              <button
                onClick={() => handleNavigate('/help')}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors hidden lg:flex text-gray-700 dark:text-gray-300"
                aria-label="Help"
              >
                <HelpCircle className="w-5 h-5" />
              </button>

              {/* Theme Toggle */}
              <ThemeToggle isDarkMode={isDarkMode} onToggle={toggleTheme} />

              {/* Notifications */}
              {showNotifications && (
                <NotificationBell
                  unreadCount={unreadCount}
                  onClick={() => handleNavigate('/notifications')}
                />
              )}

              {/* User Menu */}
              <div className="relative" ref={userMenuRef}>
                <button
                  onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  aria-label="User menu"
                  aria-expanded={isUserMenuOpen}
                >
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm font-semibold flex-shrink-0">
                    {user?.fullName?.charAt(0).toUpperCase() || 'U'}
                  </div>
                  <div className="hidden md:flex flex-col items-start min-w-0">
                    <p className="text-sm font-medium text-gray-900 dark:text-white leading-tight">
                      {user?.fullName?.split(' ')[0] || 'User'}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400 leading-tight">
                      {user?.isAdmin ? 'Admin' : 'User'}
                    </p>
                  </div>
                  <ChevronDown
                    className={`w-4 h-4 text-gray-500 transition-transform hidden md:block ${
                      isUserMenuOpen ? 'rotate-180' : ''
                    }`}
                  />
                </button>

                {/* User Dropdown Menu */}
                <UserMenu
                  user={user}
                  isOpen={isUserMenuOpen}
                  onLogout={handleLogout}
                  onNavigate={handleNavigate}
                  onClose={() => setIsUserMenuOpen(false)}
                />
              </div>
            </div>
          </nav>
        </header>

        {/* Mobile Menu */}
        <MobileMenu
          isOpen={isMobileMenuOpen}
          navItems={navigationItems}
          isActive={isActive}
          showSearch={showSearch}
          onNavigate={handleNavigate}
          onClose={() => setIsMobileMenuOpen(false)}
        />
      </>
    );
  }
);

Header.displayName = 'Header';

export default Header;
