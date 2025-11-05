// src/store/uiStore.ts

import { create } from 'zustand';
import { persist, createJSONStorage, subscribeWithSelector } from 'zustand/middleware';

/**
 * UI Store - Zustand store for managing UI state
 * Handles modals, notifications, sidebars, and general UI interactions
 */

type NotificationType = 'success' | 'error' | 'info' | 'warning';

type ModalType =
  | 'dataUpload'
  | 'deleteDataset'
  | 'datasetDetails'
  | 'confirmAction'
  | 'settings'
  | 'userProfile'
  | 'shareDataset'
  | 'exportData'
  | 'filterDatasets'
  | 'createCollection'
  | 'editDataset'
  | 'advancedFilters'
  | 'dataPreview'
  | 'chartSettings'
  | 'custom';

interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  title?: string;
  description?: string;
  duration?: number;
  action?: { label: string; onClick: () => void };
  dismissible?: boolean;
  timestamp: number;
}

interface Modal {
  type: ModalType | string;
  isOpen: boolean;
  data?: Record<string, any>;
  isLoading?: boolean;
  error?: string;
}

interface Breadcrumb {
  label: string;
  path: string;
  icon?: string;
}

interface UIState {
  // Notifications
  notifications: Notification[];
  notificationHistory: Notification[];
  maxNotifications: number;

  // Modals
  modals: Map<ModalType | string, Modal>;
  activeModal: ModalType | string | null;
  modalStack: (ModalType | string)[];

  // Sidebar & Navigation
  isSidebarOpen: boolean;
  isSidebarCollapsed: boolean;
  sidebarWidth: number;
  collapsedSidebarWidth: number;
  isNavbarSticky: boolean;

  // Breadcrumbs & History
  breadcrumbs: Breadcrumb[];
  navigationHistory: string[];
  currentPath: string;

  // Loading & Progress
  globalLoading: boolean;
  globalLoadingMessage: string;
  progressValue: number;
  progressVisible: boolean;

  // Theme & Appearance
  isDarkMode: boolean;
  isSidebarDarkMode: boolean;
  accentColor: string;
  fontSize: 'sm' | 'base' | 'lg';
  compactMode: boolean;

  // Responsive & Layout
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  windowHeight: number;
  windowWidth: number;
  screenSize: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';

  // User Interactions
  isHelpTooltipsEnabled: boolean;
  isKeyboardShortcutsEnabled: boolean;
  showAnimations: boolean;
  soundEnabled: boolean;

  // Search & Filter State
  searchBarActive: boolean;
  globalSearchQuery: string;
  filterPanelOpen: boolean;
  advancedFiltersOpen: boolean;

  // Expandable Sections
  expandedSections: Set<string>;
  pinnedSections: Set<string>;

  // Context Menus
  contextMenu: {
    isOpen: boolean;
    x: number;
    y: number;
    items: Array<{
      label: string;
      icon?: string;
      action: () => void;
      isDangerous?: boolean;
      disabled?: boolean;
    }>;
  };

  // Toasts Queue
  toastQueue: Notification[];
  maxToasts: number;

  // Drawer State
  isRightDrawerOpen: boolean;
  rightDrawerContent: string | null;
  rightDrawerData?: Record<string, any>;

  // Dialog State
  confirmDialog: {
    isOpen: boolean;
    title: string;
    message: string;
    confirmText: string;
    cancelText: string;
    onConfirm: () => void;
    onCancel: () => void;
    isDangerous: boolean;
    isLoading: boolean;
  };

  // Dropdown State
  openDropdowns: Set<string>;

  // Notification Setter Actions
  addNotification: (notification: Omit<Notification, 'id' | 'timestamp'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  clearNotificationHistory: () => void;

  // Modal Actions
  openModal: (type: ModalType | string, data?: Record<string, any>) => void;
  closeModal: (type?: ModalType | string) => void;
  closeAllModals: () => void;
  setModalLoading: (type: ModalType | string, loading: boolean) => void;
  setModalError: (type: ModalType | string, error: string | null) => void;
  updateModalData: (type: ModalType | string, data: Record<string, any>) => void;
  pushModal: (type: ModalType | string) => void;
  popModal: () => void;

  // Sidebar Actions
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  collapseSidebar: () => void;
  expandSidebar: () => void;
  setSidebarWidth: (width: number) => void;

  // Breadcrumb Actions
  setBreadcrumbs: (breadcrumbs: Breadcrumb[]) => void;
  addBreadcrumb: (breadcrumb: Breadcrumb) => void;
  removeBreadcrumb: (path: string) => void;
  updateCurrentPath: (path: string) => void;

  // Loading Actions
  setGlobalLoading: (loading: boolean, message?: string) => void;
  setProgressValue: (value: number) => void;
  setProgressVisible: (visible: boolean) => void;

  // Theme Actions
  toggleDarkMode: () => void;
  setDarkMode: (isDark: boolean) => void;
  toggleSidebarDarkMode: () => void;
  setSidebarDarkMode: (isDark: boolean) => void;
  setAccentColor: (color: string) => void;
  setFontSize: (size: 'sm' | 'base' | 'lg') => void;
  setCompactMode: (compact: boolean) => void;

  // Responsive Actions
  updateWindowDimensions: (width: number, height: number) => void;
  setScreenSize: (size: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl') => void;

  // Search & Filter Actions
  setSearchBarActive: (active: boolean) => void;
  setGlobalSearchQuery: (query: string) => void;
  setFilterPanelOpen: (open: boolean) => void;
  setAdvancedFiltersOpen: (open: boolean) => void;

  // Section Actions
  toggleExpandedSection: (section: string) => void;
  expandSection: (section: string) => void;
  collapseSection: (section: string) => void;
  togglePinnedSection: (section: string) => void;

  // Context Menu Actions
  showContextMenu: (x: number, y: number, items: UIState['contextMenu']['items']) => void;
  hideContextMenu: () => void;

  // Drawer Actions
  openRightDrawer: (content: string, data?: Record<string, any>) => void;
  closeRightDrawer: () => void;

  // Confirm Dialog Actions
  showConfirmDialog: (config: Partial<UIState['confirmDialog']>) => void;
  closeConfirmDialog: () => void;
  setConfirmDialogLoading: (loading: boolean) => void;

  // Dropdown Actions
  openDropdown: (id: string) => void;
  closeDropdown: (id: string) => void;
  toggleDropdown: (id: string) => void;
  closeAllDropdowns: () => void;

  // Preference Actions
  setHelpTooltipsEnabled: (enabled: boolean) => void;
  setKeyboardShortcutsEnabled: (enabled: boolean) => void;
  setShowAnimations: (show: boolean) => void;
  setSoundEnabled: (enabled: boolean) => void;

  // Utility Actions
  resetUIState: () => void;
  cleanupUIState: () => void;
}

// ✅ FIXED: Create factory function that returns complete UIState (without methods)
const createInitialUIState = (): Omit<
  UIState,
  | 'addNotification'
  | 'removeNotification'
  | 'clearNotifications'
  | 'clearNotificationHistory'
  | 'openModal'
  | 'closeModal'
  | 'closeAllModals'
  | 'setModalLoading'
  | 'setModalError'
  | 'updateModalData'
  | 'pushModal'
  | 'popModal'
  | 'toggleSidebar'
  | 'setSidebarOpen'
  | 'collapseSidebar'
  | 'expandSidebar'
  | 'setSidebarWidth'
  | 'setBreadcrumbs'
  | 'addBreadcrumb'
  | 'removeBreadcrumb'
  | 'updateCurrentPath'
  | 'setGlobalLoading'
  | 'setProgressValue'
  | 'setProgressVisible'
  | 'toggleDarkMode'
  | 'setDarkMode'
  | 'toggleSidebarDarkMode'
  | 'setSidebarDarkMode'
  | 'setAccentColor'
  | 'setFontSize'
  | 'setCompactMode'
  | 'updateWindowDimensions'
  | 'setScreenSize'
  | 'setSearchBarActive'
  | 'setGlobalSearchQuery'
  | 'setFilterPanelOpen'
  | 'setAdvancedFiltersOpen'
  | 'toggleExpandedSection'
  | 'expandSection'
  | 'collapseSection'
  | 'togglePinnedSection'
  | 'showContextMenu'
  | 'hideContextMenu'
  | 'openRightDrawer'
  | 'closeRightDrawer'
  | 'showConfirmDialog'
  | 'closeConfirmDialog'
  | 'setConfirmDialogLoading'
  | 'openDropdown'
  | 'closeDropdown'
  | 'toggleDropdown'
  | 'closeAllDropdowns'
  | 'setHelpTooltipsEnabled'
  | 'setKeyboardShortcutsEnabled'
  | 'setShowAnimations'
  | 'setSoundEnabled'
  | 'resetUIState'
  | 'cleanupUIState'
> => ({
  notifications: [],
  notificationHistory: [],
  maxNotifications: 5,
  modals: new Map(),
  activeModal: null,
  modalStack: [],
  isSidebarOpen: true,
  isSidebarCollapsed: false,
  sidebarWidth: 280,
  collapsedSidebarWidth: 80,
  isNavbarSticky: true,
  breadcrumbs: [],
  navigationHistory: [],
  currentPath: '/',
  globalLoading: false,
  globalLoadingMessage: '',
  progressValue: 0,
  progressVisible: false,
  isDarkMode: false,
  isSidebarDarkMode: false,
  accentColor: '#3b82f6',
  fontSize: 'base' as const,
  compactMode: false,
  isMobile: false,
  isTablet: false,
  isDesktop: true,
  windowHeight: typeof window !== 'undefined' ? window.innerHeight : 1080,
  windowWidth: typeof window !== 'undefined' ? window.innerWidth : 1920,
  screenSize: 'lg' as const,
  isHelpTooltipsEnabled: true,
  isKeyboardShortcutsEnabled: true,
  showAnimations: true,
  soundEnabled: false,
  searchBarActive: false,
  globalSearchQuery: '',
  filterPanelOpen: false,
  advancedFiltersOpen: false,
  expandedSections: new Set<string>(),
  pinnedSections: new Set<string>(),
  contextMenu: {
    isOpen: false,
    x: 0,
    y: 0,
    items: [],
  },
  toastQueue: [],
  maxToasts: 3,
  isRightDrawerOpen: false,
  rightDrawerContent: null,
  confirmDialog: {
    isOpen: false,
    title: '',
    message: '',
    confirmText: 'Confirm',
    cancelText: 'Cancel',
    onConfirm: () => {},
    onCancel: () => {},
    isDangerous: false,
    isLoading: false,
  },
  openDropdowns: new Set<string>(),
});

/**
 * Generate unique notification ID
 */
const generateNotificationId = (): string => {
  return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

/**
 * Determine screen size from width
 */
const getScreenSize = (width: number): 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl' => {
  if (width < 640) return 'xs';
  if (width < 768) return 'sm';
  if (width < 1024) return 'md';
  if (width < 1280) return 'lg';
  if (width < 1536) return 'xl';
  return '2xl';
};

export const uiStore = create<UIState>()(
  persist(
    subscribeWithSelector((set, get) => ({
      ...createInitialUIState(),

      // Notification Actions
      addNotification: (notification) => {
        const id = generateNotificationId();
        const newNotification: Notification = {
          ...notification,
          id,
          timestamp: Date.now(),
          duration: notification.duration ?? 5000,
          dismissible: notification.dismissible ?? true,
        };

        set((state) => {
          const newNotifications = [newNotification, ...state.notifications].slice(
            0,
            state.maxNotifications
          );

          // Auto-dismiss if duration is set
          if (newNotification.duration && newNotification.duration > 0) {
            setTimeout(() => {
              get().removeNotification(id);
            }, newNotification.duration);
          }

          return {
            notifications: newNotifications,
            notificationHistory: [newNotification, ...state.notificationHistory].slice(0, 50),
          };
        });

        return id;
      },

      removeNotification: (id) => {
        set((state) => ({
          notifications: state.notifications.filter((n) => n.id !== id),
        }));
      },

      clearNotifications: () => {
        set({ notifications: [] });
      },

      clearNotificationHistory: () => {
        set({ notificationHistory: [] });
      },

      // Modal Actions
      openModal: (type, data) => {
        set((state) => {
          const newModals = new Map(state.modals);
          newModals.set(type, {
            type,
            isOpen: true,
            data,
            isLoading: false,
            error: undefined,
          });

          return {
            modals: newModals,
            activeModal: type,
            modalStack: [...state.modalStack, type],
          };
        });
      },

      closeModal: (type) => {
        set((state) => {
          const targetType = type || state.activeModal;
          if (!targetType) return state;

          const newModals = new Map(state.modals);
          newModals.set(targetType, {
            ...newModals.get(targetType),
            isOpen: false,
          } as Modal);

          return {
            modals: newModals,
            activeModal: state.modalStack[state.modalStack.length - 2] || null,
            modalStack: state.modalStack.filter((m) => m !== targetType),
          };
        });
      },

      closeAllModals: () => {
        set({ modals: new Map(), activeModal: null, modalStack: [] });
      },

      setModalLoading: (type, loading) => {
        set((state) => {
          const newModals = new Map(state.modals);
          const modal = newModals.get(type);
          if (modal) {
            newModals.set(type, { ...modal, isLoading: loading });
          }
          return { modals: newModals };
        });
      },

      setModalError: (type, error) => {
        set((state) => {
          const newModals = new Map(state.modals);
          const modal = newModals.get(type);
          if (modal) {
            newModals.set(type, { ...modal, error: error ?? undefined });
          }
          return { modals: newModals };
        });
      },

      updateModalData: (type, data) => {
        set((state) => {
          const newModals = new Map(state.modals);
          const modal = newModals.get(type);
          if (modal) {
            newModals.set(type, { ...modal, data: { ...modal.data, ...data } });
          }
          return { modals: newModals };
        });
      },

      pushModal: (type) => {
        set((state) => ({
          modalStack: [...state.modalStack, type],
          activeModal: type,
        }));
      },

      popModal: () => {
        set((state) => ({
          modalStack: state.modalStack.slice(0, -1),
          activeModal: state.modalStack[state.modalStack.length - 2] || null,
        }));
      },

      // Sidebar Actions
      toggleSidebar: () => {
        set((state) => ({ isSidebarOpen: !state.isSidebarOpen }));
      },

      setSidebarOpen: (open) => {
        set({ isSidebarOpen: open });
      },

      collapseSidebar: () => {
        set({ isSidebarCollapsed: true });
      },

      expandSidebar: () => {
        set({ isSidebarCollapsed: false });
      },

      setSidebarWidth: (width) => {
        set({ sidebarWidth: width });
      },

      // Breadcrumb Actions
      setBreadcrumbs: (breadcrumbs) => {
        set({ breadcrumbs });
      },

      addBreadcrumb: (breadcrumb) => {
        set((state) => ({
          breadcrumbs: [...state.breadcrumbs, breadcrumb],
        }));
      },

      removeBreadcrumb: (path) => {
        set((state) => ({
          breadcrumbs: state.breadcrumbs.filter((b) => b.path !== path),
        }));
      },

      updateCurrentPath: (path) => {
        set((state) => ({
          currentPath: path,
          navigationHistory: [...state.navigationHistory, path].slice(-50),
        }));
      },

      // Loading Actions
      setGlobalLoading: (loading, message = '') => {
        set({
          globalLoading: loading,
          globalLoadingMessage: message,
          progressVisible: loading,
          progressValue: loading ? 0 : 100,
        });
      },

      setProgressValue: (value) => {
        set({ progressValue: Math.min(100, Math.max(0, value)) });
      },

      setProgressVisible: (visible) => {
        set({ progressVisible: visible });
      },

      // Theme Actions
      toggleDarkMode: () => {
        set((state) => {
          const newDarkMode = !state.isDarkMode;
          if (typeof document !== 'undefined') {
            document.documentElement.classList.toggle('dark', newDarkMode);
          }
          return { isDarkMode: newDarkMode };
        });
      },

      setDarkMode: (isDark) => {
        set({ isDarkMode: isDark });
        if (typeof document !== 'undefined') {
          document.documentElement.classList.toggle('dark', isDark);
        }
      },

      toggleSidebarDarkMode: () => {
        set((state) => ({ isSidebarDarkMode: !state.isSidebarDarkMode }));
      },

      setSidebarDarkMode: (isDark) => {
        set({ isSidebarDarkMode: isDark });
      },

      setAccentColor: (color) => {
        set({ accentColor: color });
        if (typeof document !== 'undefined') {
          document.documentElement.style.setProperty('--accent-color', color);
        }
      },

      setFontSize: (size) => {
        set({ fontSize: size });
        if (typeof document !== 'undefined') {
          document.documentElement.classList.remove('text-sm', 'text-base', 'text-lg');
          document.documentElement.classList.add(`text-${size}`);
        }
      },

      setCompactMode: (compact) => {
        set({ compactMode: compact });
        if (typeof document !== 'undefined') {
          document.documentElement.classList.toggle('compact', compact);
        }
      },

      // Responsive Actions
      updateWindowDimensions: (width, height) => {
        set({
          windowWidth: width,
          windowHeight: height,
          isMobile: width < 768,
          isTablet: width >= 768 && width < 1024,
          isDesktop: width >= 1024,
          screenSize: getScreenSize(width),
        });
      },

      setScreenSize: (size) => {
        set({ screenSize: size });
      },

      // Search & Filter Actions
      setSearchBarActive: (active) => {
        set({ searchBarActive: active });
      },

      setGlobalSearchQuery: (query) => {
        set({ globalSearchQuery: query });
      },

      setFilterPanelOpen: (open) => {
        set({ filterPanelOpen: open });
      },

      setAdvancedFiltersOpen: (open) => {
        set({ advancedFiltersOpen: open });
      },

      // Section Actions
      toggleExpandedSection: (section) => {
        set((state) => {
          const newSet = new Set(state.expandedSections);
          if (newSet.has(section)) {
            newSet.delete(section);
          } else {
            newSet.add(section);
          }
          return { expandedSections: newSet };
        });
      },

      expandSection: (section) => {
        set((state) => ({
          expandedSections: new Set([...state.expandedSections, section]),
        }));
      },

      collapseSection: (section) => {
        set((state) => {
          const newSet = new Set(state.expandedSections);
          newSet.delete(section);
          return { expandedSections: newSet };
        });
      },

      togglePinnedSection: (section) => {
        set((state) => {
          const newSet = new Set(state.pinnedSections);
          if (newSet.has(section)) {
            newSet.delete(section);
          } else {
            newSet.add(section);
          }
          return { pinnedSections: newSet };
        });
      },

      // Context Menu Actions
      showContextMenu: (x, y, items) => {
        set({
          contextMenu: {
            isOpen: true,
            x,
            y,
            items,
          },
        });
      },

      hideContextMenu: () => {
        set((state) => ({
          contextMenu: { ...state.contextMenu, isOpen: false },
        }));
      },

      // Drawer Actions
      openRightDrawer: (content, data) => {
        set({ isRightDrawerOpen: true, rightDrawerContent: content, rightDrawerData: data });
      },

      closeRightDrawer: () => {
        set({ isRightDrawerOpen: false, rightDrawerContent: null, rightDrawerData: undefined });
      },

      // Confirm Dialog Actions
      showConfirmDialog: (config) => {
        set((state) => ({
          confirmDialog: { ...state.confirmDialog, ...config, isOpen: true },
        }));
      },

      closeConfirmDialog: () => {
        set((state) => ({
          confirmDialog: { ...state.confirmDialog, isOpen: false },
        }));
      },

      setConfirmDialogLoading: (loading) => {
        set((state) => ({
          confirmDialog: { ...state.confirmDialog, isLoading: loading },
        }));
      },

      // Dropdown Actions
      openDropdown: (id) => {
        set((state) => ({
          openDropdowns: new Set([...state.openDropdowns, id]),
        }));
      },

      closeDropdown: (id) => {
        set((state) => {
          const newSet = new Set(state.openDropdowns);
          newSet.delete(id);
          return { openDropdowns: newSet };
        });
      },

      toggleDropdown: (id) => {
        set((state) => {
          const newSet = new Set(state.openDropdowns);
          if (newSet.has(id)) {
            newSet.delete(id);
          } else {
            newSet.add(id);
          }
          return { openDropdowns: newSet };
        });
      },

      closeAllDropdowns: () => {
        set({ openDropdowns: new Set() });
      },

      // Preference Actions
      setHelpTooltipsEnabled: (enabled) => {
        set({ isHelpTooltipsEnabled: enabled });
      },

      setKeyboardShortcutsEnabled: (enabled) => {
        set({ isKeyboardShortcutsEnabled: enabled });
      },

      setShowAnimations: (show) => {
        set({ showAnimations: show });
      },

      setSoundEnabled: (enabled) => {
        set({ soundEnabled: enabled });
      },

      // Utility Actions
      resetUIState: () => {
        set(createInitialUIState());
      },

      cleanupUIState: () => {
        set((state) => ({
          notifications: [],
          contextMenu: { ...state.contextMenu, isOpen: false },
          openDropdowns: new Set(),
        }));
      },
    })),
    {
      name: 'ui-store',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        isDarkMode: state.isDarkMode,
        isSidebarDarkMode: state.isSidebarDarkMode,
        accentColor: state.accentColor,
        fontSize: state.fontSize,
        compactMode: state.compactMode,
        isSidebarCollapsed: state.isSidebarCollapsed,
        sidebarWidth: state.sidebarWidth,
        isHelpTooltipsEnabled: state.isHelpTooltipsEnabled,
        isKeyboardShortcutsEnabled: state.isKeyboardShortcutsEnabled,
        showAnimations: state.showAnimations,
        soundEnabled: state.soundEnabled,
      }),
      version: 1,
      migrate: (persistedState: any, version: number) => {
        if (version === 0) {
          return {
            ...persistedState,
            expandedSections: new Set(persistedState.expandedSections || []),
            pinnedSections: new Set(persistedState.pinnedSections || []),
            openDropdowns: new Set(persistedState.openDropdowns || []),
          };
        }
        return persistedState;
      },
    }
  )
);

/**
 * Selector hooks for specific state slices
 */
export const useNotifications = () => uiStore((state) => state.notifications);

export const useModals = () => uiStore((state) => state.modals);

export const useActiveModal = () => uiStore((state) => state.activeModal);

export const useSidebar = () =>
  uiStore((state) => ({
    isSidebarOpen: state.isSidebarOpen,
    isSidebarCollapsed: state.isSidebarCollapsed,
    sidebarWidth: state.sidebarWidth,
  }));

export const useDarkMode = () =>
  uiStore((state) => ({
    isDarkMode: state.isDarkMode,
    isSidebarDarkMode: state.isSidebarDarkMode,
  }));

export const useResponsive = () =>
  uiStore((state) => ({
    isMobile: state.isMobile,
    isTablet: state.isTablet,
    isDesktop: state.isDesktop,
    screenSize: state.screenSize,
    windowWidth: state.windowWidth,
    windowHeight: state.windowHeight,
  }));

export const useGlobalLoading = () =>
  uiStore((state) => ({
    globalLoading: state.globalLoading,
    globalLoadingMessage: state.globalLoadingMessage,
  }));

export const useContextMenu = () => uiStore((state) => state.contextMenu);

export const useRightDrawer = () =>
  uiStore((state) => ({
    isOpen: state.isRightDrawerOpen,
    content: state.rightDrawerContent,
    data: state.rightDrawerData,
  }));

export const useConfirmDialog = () => uiStore((state) => state.confirmDialog);

export const useSearch = () =>
  uiStore((state) => ({
    searchBarActive: state.searchBarActive,
    globalSearchQuery: state.globalSearchQuery,
  }));

export default uiStore;
