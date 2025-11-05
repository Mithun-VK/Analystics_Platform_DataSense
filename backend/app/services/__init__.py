"""
Services Package.

Business logic layer containing all service implementations
for the DataInsight application. Services handle data processing,
analysis, visualization, and AI-powered insights.

This package provides:
- Authentication and user management services
- Dataset upload and management services
- Data cleaning and preprocessing services
- Exploratory data analysis services
- AI-powered insight generation services
- Advanced visualization services
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================
# CORE SERVICE IMPORTS (Required for MVP)
# ============================================================

try:
    from app.services.auth_service import AuthService, get_auth_service
    _AUTH_AVAILABLE = True
except ImportError as e:
    logger.error(f"Auth service not available: {e}")
    _AUTH_AVAILABLE = False

try:
    from app.services.user_service import UserService, get_user_service
    _USER_AVAILABLE = True
except ImportError as e:
    logger.error(f"User service not available: {e}")
    _USER_AVAILABLE = False

try:
    from app.services.dataset_service import DatasetService, get_dataset_service
    _DATASET_AVAILABLE = True
except ImportError as e:
    logger.error(f"Dataset service not available: {e}")
    _DATASET_AVAILABLE = False

try:
    from app.services.cleaning_service import DataCleaningService, get_cleaning_service
    _CLEANING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Cleaning service not available: {e}")
    _CLEANING_AVAILABLE = False

try:
    from app.services.eda_service import EDAService, get_eda_service
    _EDA_AVAILABLE = True
except ImportError as e:
    logger.error(f"EDA service not available: {e}")
    _EDA_AVAILABLE = False

try:
    from app.services.ai_service import AIService, get_ai_service, AIProvider, InsightType
    _AI_AVAILABLE = True
except ImportError as e:
    logger.error(f"AI service not available: {e}")
    _AI_AVAILABLE = False

try:
    from app.services.visualization_service import (
        VisualizationService,
        get_visualization_service,
        ChartType,
    )
    _VIZ_AVAILABLE = True
except ImportError as e:
    logger.error(f"Visualization service not available: {e}")
    _VIZ_AVAILABLE = False

# ============================================================
# OPTIONAL ADVANCED SERVICES (Future Enhancements)
# ============================================================

try:
    from app.services.upload_service import UploadService, get_upload_service
    _UPLOAD_SERVICE_AVAILABLE = True
except ImportError:
    _UPLOAD_SERVICE_AVAILABLE = False

try:
    from app.services.analytics_service import AnalyticsService, get_analytics_service
    _ANALYTICS_SERVICE_AVAILABLE = True
except ImportError:
    _ANALYTICS_SERVICE_AVAILABLE = False

try:
    from app.services.report_service import ReportService, get_report_service
    _REPORT_SERVICE_AVAILABLE = True
except ImportError:
    _REPORT_SERVICE_AVAILABLE = False

try:
    from app.services.payment_service import PaymentService, get_payment_service
    _PAYMENT_SERVICE_AVAILABLE = True
except ImportError:
    _PAYMENT_SERVICE_AVAILABLE = False

try:
    from app.services.notification_service import NotificationService, get_notification_service
    _NOTIFICATION_SERVICE_AVAILABLE = True
except ImportError:
    _NOTIFICATION_SERVICE_AVAILABLE = False

# ============================================================
# PUBLIC API
# ============================================================

__all__ = []

# Add core services
if _AUTH_AVAILABLE:
    __all__.extend(["AuthService", "get_auth_service"])

if _USER_AVAILABLE:
    __all__.extend(["UserService", "get_user_service"])

if _DATASET_AVAILABLE:
    __all__.extend(["DatasetService", "get_dataset_service"])

if _CLEANING_AVAILABLE:
    __all__.extend(["DataCleaningService", "get_cleaning_service"])

if _EDA_AVAILABLE:
    __all__.extend(["EDAService", "get_eda_service"])

if _AI_AVAILABLE:
    __all__.extend(["AIService", "get_ai_service", "AIProvider", "InsightType"])

if _VIZ_AVAILABLE:
    __all__.extend(["VisualizationService", "get_visualization_service", "ChartType"])

# Add optional services
if _UPLOAD_SERVICE_AVAILABLE:
    __all__.extend(["UploadService", "get_upload_service"])

if _ANALYTICS_SERVICE_AVAILABLE:
    __all__.extend(["AnalyticsService", "get_analytics_service"])

if _REPORT_SERVICE_AVAILABLE:
    __all__.extend(["ReportService", "get_report_service"])

if _PAYMENT_SERVICE_AVAILABLE:
    __all__.extend(["PaymentService", "get_payment_service"])

if _NOTIFICATION_SERVICE_AVAILABLE:
    __all__.extend(["NotificationService", "get_notification_service"])

# ============================================================
# PACKAGE UTILITIES
# ============================================================

VERSION = "1.0.0"
SERVICE_LAYER_INITIALIZED = True


def get_available_services() -> dict[str, bool]:
    """
    Get dictionary of available services in the package.
    
    Returns:
        Dictionary mapping service names to availability status
    """
    return {
        "auth": _AUTH_AVAILABLE,
        "user": _USER_AVAILABLE,
        "dataset": _DATASET_AVAILABLE,
        "cleaning": _CLEANING_AVAILABLE,
        "eda": _EDA_AVAILABLE,
        "ai": _AI_AVAILABLE,
        "visualization": _VIZ_AVAILABLE,
        "upload": _UPLOAD_SERVICE_AVAILABLE,
        "analytics": _ANALYTICS_SERVICE_AVAILABLE,
        "report": _REPORT_SERVICE_AVAILABLE,
        "payment": _PAYMENT_SERVICE_AVAILABLE,
        "notification": _NOTIFICATION_SERVICE_AVAILABLE,
    }


def list_core_services() -> list[str]:
    """
    List all core MVP services.
    
    Returns:
        List of core service names
    """
    services = []
    if _AUTH_AVAILABLE:
        services.append("AuthService")
    if _USER_AVAILABLE:
        services.append("UserService")
    if _DATASET_AVAILABLE:
        services.append("DatasetService")
    if _CLEANING_AVAILABLE:
        services.append("DataCleaningService")
    if _EDA_AVAILABLE:
        services.append("EDAService")
    if _AI_AVAILABLE:
        services.append("AIService")
    if _VIZ_AVAILABLE:
        services.append("VisualizationService")
    return services


def list_advanced_services() -> list[str]:
    """
    List all advanced (optional) services that are available.
    
    Returns:
        List of available advanced service names
    """
    services = []
    
    if _UPLOAD_SERVICE_AVAILABLE:
        services.append("UploadService")
    if _ANALYTICS_SERVICE_AVAILABLE:
        services.append("AnalyticsService")
    if _REPORT_SERVICE_AVAILABLE:
        services.append("ReportService")
    if _PAYMENT_SERVICE_AVAILABLE:
        services.append("PaymentService")
    if _NOTIFICATION_SERVICE_AVAILABLE:
        services.append("NotificationService")
    
    return services


def service_health_check() -> dict[str, str]:
    """
    Check health status of all services.
    
    Returns:
        Dictionary with service health status
    """
    return {
        name: "healthy" if available else "unavailable"
        for name, available in get_available_services().items()
    }


# Log initialization
core_count = len(list_core_services())
advanced_count = len(list_advanced_services())
logger.info(
    f"Services package initialized. "
    f"Core: {core_count}/7, Advanced: {advanced_count}/5"
)
