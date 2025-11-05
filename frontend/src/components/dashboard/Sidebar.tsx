// src/components/dashboard/Sidebar.tsx

import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  BarChart3,
  LineChart,
  Sparkles,
  Droplet,
  Settings,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  FileText,
  Users,
  CreditCard,
  Crown,
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  isCollapsed: boolean;
  isMobile: boolean;
  onCollapse: () => void;
  onClose: () => void;
}

interface NavItem {
  id: string;
  label: string;
  icon: React.ElementType;
  path: string;
  badge?: string | number;
  badgeColor?: string;
  children?: NavItem[];
}

/**
 * Sidebar - Navigation sidebar component with collapsible menu
 * Features: nested navigation, active state highlighting, badge indicators, tooltips
 * Responsive: Full sidebar on desktop, overlay on mobile, collapsible on all devices
 */
const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  isCollapsed,
  isMobile,
  onCollapse,
  onClose,
}) => {
  const location = useLocation();
  const [expandedMenus, setExpandedMenus] = useState<string[]>(['']);

  // Navigation items configuration
  const navigationItems: NavItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
      path: '/dashboard',
    },
    {
      id: 'datasets',
      label: 'Datasets',
      icon: Database,
      path: '/datasets',
      badge: '12',
      badgeColor: 'bg-blue-500',
    },
    {
      id: 'eda',
      label: 'EDA & Analysis',
      icon: BarChart3,
      path: '/eda',
    },
    {
      id: 'visualizations',
      label: 'Visualizations',
      icon: LineChart,
      path: '/visualizations',
      children: [
        {
          id: 'visualizations-create',
          label: 'Create Chart',
          icon: Sparkles,
          path: '/visualizations/create',
        },
        {
          id: 'visualizations-gallery',
          label: 'Chart Gallery',
          icon: FileText,
          path: '/visualizations/gallery',
        },
      ],
    },
    {
      id: 'cleaning',
      label: 'Data Cleaning',
      icon: Droplet,
      path: '/cleaning',
      badge: 'New',
      badgeColor: 'bg-green-500',
    },
    {
      id: 'insights',
      label: 'AI Insights',
      icon: Sparkles,
      path: '/insights',
      badge: 'Pro',
      badgeColor: 'bg-purple-500',
    },
  ];

  const bottomNavigationItems: NavItem[] = [
    {
      id: 'team',
      label: 'Team',
      icon: Users,
      path: '/team',
    },
    {
      id: 'billing',
      label: 'Billing',
      icon: CreditCard,
      path: '/billing',
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Settings,
      path: '/settings',
    },
  ];

  // Toggle submenu expansion
  const toggleSubmenu = (itemId: string) => {
    setExpandedMenus((prev) =>
      prev.includes(itemId)
        ? prev.filter((id) => id !== itemId)
        : [...prev, itemId]
    );
  };

  // Check if nav item is active
  const isActive = (path: string, item: NavItem): boolean => {
    if (item.children) {
      return item.children.some((child) => location.pathname.startsWith(child.path));
    }
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  // Check if submenu should be expanded
  const isSubmenuExpanded = (item: NavItem): boolean => {
    if (isCollapsed) return false;
    return expandedMenus.includes(item.id) || isActive(item.path, item);
  };

  // Render navigation item
  const renderNavItem = (item: NavItem, level: number = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const active = isActive(item.path, item);
    const expanded = isSubmenuExpanded(item);
    const Icon = item.icon;

    if (hasChildren) {
      return (
        <div key={item.id}>
          {/* Parent Menu Item */}
          <button
            onClick={() => {
              if (isCollapsed) {
                onCollapse();
                setTimeout(() => toggleSubmenu(item.id), 100);
              } else {
                toggleSubmenu(item.id);
              }
            }}
            className={`sidebar-item group relative ${
              active ? 'active' : ''
            } ${isCollapsed ? 'justify-center' : ''}`}
            title={isCollapsed ? item.label : ''}
          >
            <div className="flex items-center flex-1 min-w-0">
              <Icon
                className={`w-5 h-5 flex-shrink-0 transition-colors ${
                  isCollapsed ? 'mx-auto' : ''
                }`}
              />
              {!isCollapsed && (
                <>
                  <span className="ml-3 font-medium truncate">{item.label}</span>
                  {item.badge && (
                    <span
                      className={`ml-auto px-2 py-0.5 text-xs font-semibold text-white rounded-full ${item.badgeColor}`}
                    >
                      {item.badge}
                    </span>
                  )}
                </>
              )}
            </div>
            {!isCollapsed && (
              <ChevronDown
                className={`w-4 h-4 ml-2 flex-shrink-0 transition-transform duration-200 ${
                  expanded ? 'rotate-180' : ''
                }`}
              />
            )}

            {/* Tooltip for collapsed state */}
            {isCollapsed && (
              <div className="absolute left-full ml-6 px-2 py-1 bg-gray-900 text-white text-sm rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-50 pointer-events-none">
                {item.label}
                <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900"></div>
              </div>
            )}
          </button>

          {/* Submenu Items */}
          {expanded && !isCollapsed && (
            <div className="ml-4 mt-1 space-y-1 animate-slide-in-down">
              {item.children?.map((child) => renderNavItem(child, level + 1))}
            </div>
          )}
        </div>
      );
    }

    // Regular menu item (no children)
    return (
      <NavLink
        key={item.id}
        to={item.path}
        onClick={isMobile ? onClose : undefined}
        className={({ isActive }) =>
          `sidebar-item group relative ${isActive ? 'active' : ''} ${
            isCollapsed ? 'justify-center' : ''
          } ${level > 0 ? 'text-sm' : ''}`
        }
        title={isCollapsed ? item.label : ''}
      >
        <div className="flex items-center flex-1 min-w-0">
          <Icon
            className={`w-5 h-5 flex-shrink-0 transition-colors ${
              isCollapsed ? 'mx-auto' : ''
            }`}
          />
          {!isCollapsed && (
            <>
              <span className="ml-3 font-medium truncate">{item.label}</span>
              {item.badge && (
                <span
                  className={`ml-auto px-2 py-0.5 text-xs font-semibold text-white rounded-full ${item.badgeColor}`}
                >
                  {item.badge}
                </span>
              )}
            </>
          )}
        </div>

        {/* Tooltip for collapsed state */}
        {isCollapsed && (
          <div className="absolute left-full ml-6 px-2 py-1 bg-gray-900 text-white text-sm rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-50 pointer-events-none">
            {item.label}
            <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900"></div>
          </div>
        )}
      </NavLink>
    );
  };

  if (!isOpen && !isMobile) return null;

  return (
    <>
      {/* Sidebar Container */}
      <aside
        className={`sidebar fixed lg:sticky top-0 h-screen bg-white border-r border-gray-200 shadow-lg transition-all duration-300 z-40 flex flex-col ${
          isCollapsed && !isMobile ? 'w-20' : 'w-64'
        } ${
          isMobile
            ? isOpen || isMobile
              ? 'translate-x-0'
              : '-translate-x-full'
            : ''
        }`}
      >
        {/* Logo Section */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200 flex-shrink-0">
          {!isCollapsed && (
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex items-center justify-center shadow-md">
                <svg
                  className="w-6 h-6 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-bold text-gray-900 leading-tight">
                  DataAnalytics
                </span>
                <span className="text-xs text-gray-500 leading-tight">
                  Pro Platform
                </span>
              </div>
            </div>
          )}

          {/* Collapse Toggle Button */}
          <button
            onClick={onCollapse}
            className="hidden lg:flex p-1.5 rounded-lg hover:bg-gray-100 transition-colors ml-auto"
            aria-label="Toggle sidebar"
          >
            {isCollapsed ? (
              <ChevronRight className="w-5 h-5 text-gray-600" />
            ) : (
              <ChevronLeft className="w-5 h-5 text-gray-600" />
            )}
          </button>

          {/* Mobile Close Button */}
          {isMobile && (
            <button
              onClick={onClose}
              className="lg:hidden p-1.5 rounded-lg hover:bg-gray-100 transition-colors ml-auto"
              aria-label="Close sidebar"
            >
              <ChevronLeft className="w-5 h-5 text-gray-600" />
            </button>
          )}
        </div>

        {/* Navigation Section */}
        <nav className="flex-1 overflow-y-auto py-4 px-2 space-y-1">
          {/* Main Navigation */}
          <div className="space-y-1">
            {!isCollapsed && (
              <div className="px-3 mb-2">
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Main Menu
                </span>
              </div>
            )}
            {navigationItems.map((item) => renderNavItem(item))}
          </div>

          {/* Divider */}
          <div className="my-4 border-t border-gray-200"></div>

          {/* Bottom Navigation */}
          <div className="space-y-1">
            {!isCollapsed && (
              <div className="px-3 mb-2">
                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  Account
                </span>
              </div>
            )}
            {bottomNavigationItems.map((item) => renderNavItem(item))}
          </div>
        </nav>

        {/* Upgrade Section */}
        {!isCollapsed && (
          <div className="p-4 m-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl border border-blue-200 flex-shrink-0">
            <div className="flex items-start space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-yellow-400 to-yellow-500 rounded-lg flex items-center justify-center flex-shrink-0 shadow-md">
                <Crown className="w-5 h-5 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-semibold text-gray-900 mb-1">
                  Upgrade to Pro
                </h4>
                <p className="text-xs text-gray-600 mb-3 leading-relaxed">
                  Unlock advanced features and unlimited datasets
                </p>
                <button className="w-full px-3 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors shadow-sm">
                  Upgrade Now
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Collapsed Upgrade Button */}
        {isCollapsed && (
          <div className="p-2 flex-shrink-0">
            <button
              className="w-full p-2 bg-gradient-to-br from-yellow-400 to-yellow-500 rounded-lg hover:from-yellow-500 hover:to-yellow-600 transition-all shadow-md group relative"
              title="Upgrade to Pro"
            >
              <Crown className="w-5 h-5 text-white mx-auto" />

              {/* Tooltip */}
              <div className="absolute left-full ml-6 px-2 py-1 bg-gray-900 text-white text-sm rounded-md opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 whitespace-nowrap z-50 pointer-events-none">
                Upgrade to Pro
                <div className="absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-gray-900"></div>
              </div>
            </button>
          </div>
        )}

        {/* Footer Info */}
        {!isCollapsed && (
          <div className="px-4 py-3 border-t border-gray-200 flex-shrink-0">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>v1.0.0</span>
              <span>© 2025 DataAnalytics</span>
            </div>
          </div>
        )}
      </aside>
    </>
  );
};

export default Sidebar;
