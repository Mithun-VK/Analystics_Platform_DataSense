"""
Enhanced Production-Grade Data Cleaning Service - Phase 1
Advanced Missing Value Handling with KNN, MICE, and Custom Strategies
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.dataset import Dataset, DatasetStatus
from app.database import get_db

from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from scipy.stats.mstats import winsorize
from scipy import stats

import hashlib
from sqlalchemy import Column, Integer, String, JSON, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base

from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    RobustScaler,
    MaxAbsScaler,
    LabelEncoder, 
    OneHotEncoder
)
import pickle

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from typing import Iterator, Callable
import gc

from abc import ABC, abstractmethod
from enum import Enum
import yaml
import importlib
from typing import Type, Protocol

from functools import wraps
from time import sleep
from typing import Callable, TypeVar, ParamSpec
import traceback
import sys

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from celery import Celery
from typing import Awaitable
import asyncio
from fastapi import BackgroundTasks, APIRouter
from datetime import timedelta

logger = logging.getLogger(__name__)


# ============================================================
# EVENT SYSTEM
# ============================================================

class CleaningEvent(str, Enum):
    """Cleaning pipeline events."""
    STARTED = "cleaning_started"
    COMPLETED = "cleaning_completed"
    FAILED = "cleaning_failed"
    PROGRESS_UPDATE = "progress_update"
    STEP_COMPLETED = "step_completed"
    WARNING_RAISED = "warning_raised"
    ERROR_RECOVERED = "error_recovered"


class EventHook:
    """Event hook system for pipeline integration."""
    
    def __init__(self):
        self.subscribers: Dict[CleaningEvent, List[Callable]] = {
            event: [] for event in CleaningEvent
        }
    
    def subscribe(self, event: CleaningEvent, callback: Callable) -> None:
        """Subscribe to an event."""
        self.subscribers[event].append(callback)
        logger.info(f"📡 Subscribed to event: {event.value}")
    
    def unsubscribe(self, event: CleaningEvent, callback: Callable) -> None:
        """Unsubscribe from an event."""
        if callback in self.subscribers[event]:
            self.subscribers[event].remove(callback)
    
    def emit(self, event: CleaningEvent, data: Dict[str, Any]) -> None:
        """Emit an event to all subscribers."""
        logger.debug(f"📡 Emitting event: {event.value}")
        
        for callback in self.subscribers[event]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Event callback failed: {str(e)}")


# ============================================================
# WEBHOOK INTEGRATION
# ============================================================

class WebhookNotifier:
    """Send webhook notifications for cleaning events."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def notify(self, event: CleaningEvent, data: Dict[str, Any]) -> bool:
        """Send webhook notification."""
        import aiohttp
        
        payload = {
            "event": event.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"✅ Webhook notification sent: {event.value}")
                        return True
                    else:
                        logger.warning(f"⚠️ Webhook failed with status: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"❌ Webhook notification failed: {str(e)}")
            return False

# ============================================================
# CUSTOM EXCEPTIONS
# ============================================================

class DataCleaningError(Exception):
    """Base exception for data cleaning errors."""
    pass


class DataLoadError(DataCleaningError):
    """Error loading data from file."""
    pass


class DataValidationError(DataCleaningError):
    """Error validating data or schema."""
    pass


class TransformationError(DataCleaningError):
    """Error during data transformation."""
    pass


class PluginError(DataCleaningError):
    """Error executing a plugin."""
    pass


class ConfigurationError(DataCleaningError):
    """Error in configuration."""
    pass


# ============================================================
# ERROR CONTEXT MANAGER
# ============================================================

class ErrorContext:
    """Context manager for error handling with detailed logging."""
    
    def __init__(
        self,
        service: 'DataCleaningService',
        operation: str,
        critical: bool = False,
        fallback_value: Any = None
    ):
        self.service = service
        self.operation = operation
        self.critical = critical
        self.fallback_value = fallback_value
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"🔄 Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        if exc_type is None:
            # Success
            logger.info(f"✅ Completed operation: {self.operation} ({duration:.2f}s)")
            return True
        
        # Error occurred
        error_details = {
            "operation": self.operation,
            "error_type": exc_type.__name__,
            "error_message": str(exc_val),
            "duration_seconds": duration,
            "traceback": traceback.format_exc()
        }
        
        self.service._log_step(
            "error",
            f"Error in {self.operation}: {exc_val}",
            metadata=error_details
        )
        
        if self.critical:
            logger.error(f"❌ Critical error in {self.operation}: {exc_val}")
            return False  # Re-raise exception
        else:
            logger.warning(f"⚠️ Non-critical error in {self.operation}: {exc_val}")
            return True  # Suppress exception


# ============================================================
# RETRY DECORATOR
# ============================================================

P = ParamSpec('P')
T = TypeVar('T')

def retry_on_failure(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            current_delay = delay
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"❌ Failed after {max_attempts} attempts: {func.__name__}")
                        raise
                    
                    logger.warning(
                        f"⚠️ Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {current_delay}s..."
                    )
                    
                    sleep(current_delay)
                    current_delay *= backoff
            
            return None  # Should never reach here
        
        return wrapper
    return decorator

# ============================================================
# PLUGIN ARCHITECTURE - BASE CLASSES
# ============================================================

class CleaningPlugin(ABC):
    """Base class for cleaning plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the plugin's cleaning logic."""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate plugin configuration."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "version": "1.0",
            "description": self.__doc__ or "No description"
        }


class ImputerPlugin(CleaningPlugin):
    """Base class for custom imputer plugins."""
    
    @abstractmethod
    def impute(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Impute missing values in a column."""
        pass


class OutlierDetectorPlugin(CleaningPlugin):
    """Base class for custom outlier detector plugins."""
    
    @abstractmethod
    def detect(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers in a column. Returns boolean mask."""
        pass


# ============================================================
# EXAMPLE CUSTOM PLUGINS
# ============================================================

class MedianImputerPlugin(ImputerPlugin):
    """Custom median imputer plugin."""
    
    def validate_config(self) -> bool:
        return True
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = self.impute(df, col)
        return df
    
    def impute(self, df: pd.DataFrame, column: str) -> pd.Series:
        median_val = df[column].median()
        return df[column].fillna(median_val)


class CustomRangeValidatorPlugin(CleaningPlugin):
    """Validate values are within specified ranges."""
    
    def validate_config(self) -> bool:
        return "ranges" in self.config
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        ranges = self.config.get("ranges", {})
        
        for col, range_def in ranges.items():
            if col in df.columns:
                min_val = range_def.get("min")
                max_val = range_def.get("max")
                
                if min_val is not None:
                    df = df[df[col] >= min_val]
                if max_val is not None:
                    df = df[df[col] <= max_val]
        
        return df


# ============================================================
# CONFIGURATION PRESETS
# ============================================================

class CleaningPreset(str, Enum):
    """Pre-defined cleaning configuration presets."""
    
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    ML_READY = "ml_ready"
    PRODUCTION = "production"
    CUSTOM = "custom"


PRESET_CONFIGS = {
    CleaningPreset.MINIMAL: {
        "remove_duplicates": True,
        "missing_strategy": "drop",
        "outlier_detection": False,
        "optimize_dtypes": False,
        "scaling_method": None,
        "encoding_method": None
    },
    
    CleaningPreset.STANDARD: {
        "remove_duplicates": True,
        "missing_strategy": "median",
        "outlier_detection": True,
        "outlier_method": "iqr",
        "outlier_treatment": "cap",
        "optimize_dtypes": True,
        "category_threshold": 0.5,
        "scaling_method": None,
        "encoding_method": None
    },
    
    CleaningPreset.AGGRESSIVE: {
        "remove_duplicates": True,
        "missing_strategy": "knn",
        "knn_neighbors": 5,
        "outlier_detection": True,
        "outlier_method": "isolation_forest",
        "outlier_treatment": "remove",
        "optimize_dtypes": True,
        "category_threshold": 0.3,
        "auto_detect_dates": True,
        "downcast_integers": True,
        "downcast_floats": True,
        "scaling_method": None,
        "encoding_method": None
    },
    
    CleaningPreset.ML_READY: {
        "remove_duplicates": True,
        "missing_strategy": "knn",
        "knn_neighbors": 5,
        "outlier_detection": True,
        "outlier_method": "isolation_forest",
        "outlier_treatment": "cap",
        "optimize_dtypes": True,
        "category_threshold": 0.5,
        "auto_detect_dates": True,
        "scaling_method": "standard",
        "encoding_method": "onehot",
        "max_categories_onehot": 10,
        "save_scaler": True,
        "save_encoders": True
    },
    
    CleaningPreset.PRODUCTION: {
        "remove_duplicates": True,
        "missing_strategy": "median",
        "outlier_detection": True,
        "outlier_method": "iqr",
        "outlier_treatment": "cap",
        "optimize_dtypes": True,
        "category_threshold": 0.5,
        "auto_detect_dates": True,
        "downcast_integers": True,
        "downcast_floats": True,
        "schema_validation": None,
        "enforce_schema": True,
        "save_audit_log": True,
        "generate_quality_report": True,
        "enable_parallel_processing": True,
        "optimize_memory": True
    }
}

# ============================================================
# CONFIGURATION SCHEMA
# ============================================================

class CleaningConfig(BaseModel):
    """Configuration for data cleaning pipeline."""
    
    # Phase 1: Missing Value Handling
    missing_strategy: str = Field(default="median", description="Global missing value strategy")
    per_column_missing: Optional[Dict[str, str]] = Field(default=None)
    fill_values: Optional[Dict[str, Any]] = Field(default=None)
    knn_neighbors: int = Field(default=5)
    mice_max_iter: int = Field(default=10)
    
    # Phase 2: Duplicate Handling
    remove_duplicates: bool = Field(default=True)
    duplicate_subset: Optional[List[str]] = Field(default=None)
    
    # Phase 3: Column Management
    columns_to_drop: Optional[List[str]] = Field(default=None)
    
    # Phase 4: Outlier Detection
    outlier_detection: bool = Field(default=True)
    outlier_method: str = Field(default="iqr")
    outlier_threshold: float = Field(default=1.5)
    outlier_treatment: str = Field(default="cap")
    isolation_contamination: float = Field(default=0.05)
    dbscan_eps: float = Field(default=0.5)
    dbscan_min_samples: int = Field(default=5)
    outlier_impute_values: Optional[Dict[str, Any]] = Field(default=None)
    
    # Phase 5: Data Type Optimization
    optimize_dtypes: bool = Field(default=True)
    category_threshold: float = Field(default=0.5)
    auto_detect_dates: bool = Field(default=True)
    downcast_integers: bool = Field(default=True)
    downcast_floats: bool = Field(default=True)
    
    # Phase 6: Schema Validation
    schema_validation: Optional[Dict[str, Any]] = Field(default=None)
    enforce_schema: bool = Field(default=False)
    auto_fix_schema: bool = Field(default=True)
    
    # Phase 7: Feature Scaling
    scaling_method: Optional[str] = Field(default=None)
    columns_to_scale: Optional[List[str]] = Field(default=None)
    exclude_from_scaling: Optional[List[str]] = Field(default=None)
    save_scaler: bool = Field(default=True)
    
    # Phase 8: Categorical Encoding
    encoding_method: Optional[str] = Field(default=None)
    columns_to_encode: Optional[List[str]] = Field(default=None)
    exclude_from_encoding: Optional[List[str]] = Field(default=None)
    max_categories_onehot: int = Field(default=10)
    save_encoders: bool = Field(default=True)
    target_column_for_encoding: Optional[str] = Field(default=None)
    ordinal_mappings: Optional[Dict[str, List[str]]] = Field(default=None)
    
    # Phase 9: Performance & Scalability
    chunk_size: Optional[int] = Field(default=None)
    enable_parallel_processing: bool = Field(default=False)
    max_workers: Optional[int] = Field(default=None)
    use_multiprocessing: bool = Field(default=False)
    memory_limit_mb: Optional[float] = Field(default=None)
    enable_caching: bool = Field(default=True)
    optimize_memory: bool = Field(default=True)
    # progress_callback REMOVED - incompatible with JSON schema
    
    # Phase 10: Error Handling
    enable_error_recovery: bool = Field(default=True)
    continue_on_error: bool = Field(default=True)
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    fallback_to_defaults: bool = Field(default=True)
    strict_mode: bool = Field(default=False)
    validation_mode: str = Field(default="warn")
    error_report_path: Optional[str] = Field(default=None)
    
    # Phase 11: Integration & Automation
    webhook_url: Optional[str] = Field(default=None)
    enable_webhooks: bool = Field(default=False)
    email_notifications: Optional[List[str]] = Field(default=None)
    slack_webhook: Optional[str] = Field(default=None)
    auto_schedule: Optional[Dict[str, Any]] = Field(default=None)
    run_async: bool = Field(default=False)
    
    # Phase 12: Extensibility & Configuration
    preset: Optional[str] = Field(default=None)
    custom_plugins: Optional[List[Dict[str, Any]]] = Field(default=None)
    plugin_order: Optional[List[str]] = Field(default=None)
    config_file_path: Optional[str] = Field(default=None)
    extend_preset: bool = Field(default=False)
    
    # Phase 13: Audit & Quality
    save_audit_log: bool = Field(default=True)
    export_audit_log: bool = Field(default=False)
    user_id: Optional[int] = Field(default=None)
    generate_quality_report: bool = Field(default=True)
    quality_thresholds: Optional[Dict[str, float]] = Field(default=None)
    
    class Config:
        extra = "allow"
        # DO NOT use arbitrary_types_allowed - it causes JSON schema issues

# ============================================================
# DATA CLEANING SERVICE - PHASE 1
# ============================================================

class DataCleaningService:
    """
    Production-grade data cleaning service - Phase 1.
    
    Features:
    ✅ Advanced missing value handling (median, mean, mode, KNN, MICE)
    ✅ Per-column imputation strategies
    ✅ Custom fill values
    ✅ Duplicate removal
    ✅ Column dropping
    ✅ Comprehensive audit logging
    """
    
    def __init__(self, db: Session):
        """Initialize data cleaning service."""
        self.db = db
        self.cleaning_log: List[Dict[str, Any]] = []
        self.transformations_applied: List[str] = []
        self.quality_metrics: Dict[str, Any] = {}
        self.quality_alerts: List[str] = []
        
        # Phase 4 additions
        self.session_id: str = self._generate_session_id()
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Phase 5 additions
        self.schema_violations: List[Dict[str, Any]] = []
        self.type_conversions: List[Dict[str, Any]] = []
        
        # Phase 6 additions
        self.fitted_scalers: Dict[str, Any] = {}
        self.fitted_encoders: Dict[str, Any] = {}
        self.encoding_mappings: Dict[str, Any] = {}

        # Other initializations if needed...
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID for audit trail."""
        timestamp = datetime.now(timezone.utc).isoformat()
        random_component = str(hash(timestamp))
        session_string = f"{timestamp}_{random_component}"
        return hashlib.sha256(session_string.encode()).hexdigest()[:16]



    # ============================================================
    # PHASE 10: EVENT HOOKS
    # ============================================================
    
    def on_event(self, event: CleaningEvent, callback: Callable) -> None:
        """Register an event callback."""
        self.event_hooks.subscribe(event, callback)
    
    def _emit_event(self, event: CleaningEvent, data: Dict[str, Any]) -> None:
        """Emit an event."""
        self.event_hooks.emit(event, data)
    
    async def _send_webhook_notification(
        self,
        event: CleaningEvent,
        data: Dict[str, Any],
        config: CleaningConfig
    ) -> None:
        """Send webhook notification if configured."""
        
        if not config.enable_webhooks or not config.webhook_url:
            return
        
        if not self.webhook_notifier:
            self.webhook_notifier = WebhookNotifier(config.webhook_url)
        
        await self.webhook_notifier.notify(event, data)
    
    # ============================================================
    # PHASE 10: NOTIFICATION SYSTEM
    # ============================================================
    
    async def _send_email_notification(
        self,
        subject: str,
        body: str,
        recipients: List[str]
    ) -> bool:
        """Send email notification."""
        # Placeholder - integrate with your email service
        logger.info(f"📧 Email notification: {subject} to {recipients}")
        return True
    
    async def _send_slack_notification(
        self,
        message: str,
        webhook_url: str
    ) -> bool:
        """Send Slack notification."""
        import aiohttp
        
        payload = {
            "text": message,
            "username": "Data Cleaning Bot",
            "icon_emoji": ":robot_face:"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")
            return False
    
    async def _send_notifications(
        self,
        event: CleaningEvent,
        data: Dict[str, Any],
        config: CleaningConfig
    ) -> None:
        """Send all configured notifications."""
        
        # Webhook
        if config.enable_webhooks and config.webhook_url:
            await self._send_webhook_notification(event, data, config)
        
        # Email
        if config.email_notifications and event in [CleaningEvent.COMPLETED, CleaningEvent.FAILED]:
            subject = f"Data Cleaning {event.value}"
            body = json.dumps(data, indent=2)
            await self._send_email_notification(subject, body, config.email_notifications)
        
        # Slack
        if config.slack_webhook and event in [CleaningEvent.COMPLETED, CleaningEvent.FAILED]:
            message = f"🤖 Data Cleaning {event.value}\nDataset: {data.get('dataset_id')}\nSession: {data.get('session_id')}"
            await self._send_slack_notification(message, config.slack_webhook)
    
    # ============================================================
    # PHASE 10: BACKGROUND TASK EXECUTION
    # ============================================================
    
    async def clean_dataset_async(
        self,
        dataset_id: int,
        config: CleaningConfig,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """
        Execute cleaning asynchronously in background.
        
        Returns immediately with job_id for status tracking.
        """
        job_id = f"job_{self.session_id}"
        
        # Add to background tasks
        background_tasks.add_task(
            self._execute_cleaning_background,
            dataset_id,
            config,
            job_id
        )
        
        logger.info(f"🚀 Started background cleaning job: {job_id}")
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Cleaning job started in background",
            "session_id": self.session_id
        }
    
    async def _execute_cleaning_background(
        self,
        dataset_id: int,
        config: CleaningConfig,
        job_id: str
    ) -> None:
        """Execute cleaning in background with notifications."""
        
        try:
            # Emit started event
            self._emit_event(CleaningEvent.STARTED, {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Execute cleaning
            result = self.clean_dataset_robust(dataset_id, config)
            
            # Emit completed event
            await self._send_notifications(
                CleaningEvent.COMPLETED,
                {
                    "job_id": job_id,
                    "dataset_id": dataset_id,
                    "result": result
                },
                config
            )
            
            self._emit_event(CleaningEvent.COMPLETED, result)
            
        except Exception as e:
            # Emit failed event
            await self._send_notifications(
                CleaningEvent.FAILED,
                {
                    "job_id": job_id,
                    "dataset_id": dataset_id,
                    "error": str(e)
                },
                config
            )
            
            self._emit_event(CleaningEvent.FAILED, {"error": str(e)})
            
            logger.error(f"Background cleaning job failed: {str(e)}")
    
    # ============================================================
    # PHASE 10: SCHEDULED CLEANING
    # ============================================================
    
    @staticmethod
    def create_scheduled_cleaner(
        dataset_id: int,
        config: CleaningConfig,
        schedule: str,
        db_session_factory: Callable
    ) -> BackgroundScheduler:
        """
        Create a scheduled cleaning job.
        
        Args:
            dataset_id: Dataset to clean
            config: Cleaning configuration
            schedule: Cron expression (e.g., "0 2 * * *" for daily at 2 AM)
            db_session_factory: Function that returns a DB session
            
        Returns:
            Scheduler instance (must be started)
        """
        scheduler = BackgroundScheduler()
        
        def run_scheduled_cleaning():
            db = db_session_factory()
            try:
                service = DataCleaningService(db)
                result = service.clean_dataset_robust(dataset_id, config)
                logger.info(f"✅ Scheduled cleaning completed: {result['session_id']}")
            except Exception as e:
                logger.error(f"❌ Scheduled cleaning failed: {str(e)}")
            finally:
                db.close()
        
        scheduler.add_job(
            run_scheduled_cleaning,
            trigger=CronTrigger.from_crontab(schedule),
            id=f"cleaning_job_{dataset_id}",
            name=f"Scheduled Cleaning for Dataset {dataset_id}"
        )
        
        logger.info(f"📅 Scheduled cleaning job created: {schedule}")
        
        return scheduler
    
    # ============================================================
    # PHASE 10: BATCH PROCESSING
    # ============================================================
    
    def clean_multiple_datasets(
        self,
        dataset_ids: List[int],
        config: CleaningConfig,
        parallel: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Clean multiple datasets.
        
        Args:
            dataset_ids: List of dataset IDs
            config: Cleaning configuration
            parallel: Execute in parallel if True
            
        Returns:
            List of results for each dataset
        """
        logger.info(f"🔄 Cleaning {len(dataset_ids)} datasets")
        
        if parallel and config.enable_parallel_processing:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            max_workers = config.max_workers or min(4, len(dataset_ids))
            results = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self.clean_dataset_robust, dataset_id, config): dataset_id
                    for dataset_id in dataset_ids
                }
                
                for future in as_completed(futures):
                    dataset_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to clean dataset {dataset_id}: {str(e)}")
                        results.append({
                            "dataset_id": dataset_id,
                            "success": False,
                            "error": str(e)
                        })
            
            return results
        
        else:
            # Sequential processing
            results = []
            for dataset_id in dataset_ids:
                try:
                    result = self.clean_dataset_robust(dataset_id, config)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to clean dataset {dataset_id}: {str(e)}")
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": str(e)
                    })
            
            return results
    
    # ============================================================
    # PHASE 10: API INTEGRATION
    # ============================================================
    
    @staticmethod
    def create_api_router() -> APIRouter:
        """Create FastAPI router for cleaning endpoints."""
        
        router = APIRouter(prefix="/api/v1/cleaning", tags=["cleaning"])
        
        @router.post("/clean/{dataset_id}")
        async def clean_dataset_endpoint(
            dataset_id: int,
            config: CleaningConfig,
            background_tasks: BackgroundTasks,
            db: Session = Depends(get_db)
        ):
            """Clean a dataset endpoint."""
            service = DataCleaningService(db)
            
            if config.run_async:
                return await service.clean_dataset_async(dataset_id, config, background_tasks)
            else:
                result = service.clean_dataset_robust(dataset_id, config)
                return result
        
        @router.get("/presets")
        async def list_presets():
            """List available configuration presets."""
            return DataCleaningService.list_available_presets()
        
        @router.get("/history/{dataset_id}")
        async def get_cleaning_history(
            dataset_id: int,
            limit: int = 10,
            db: Session = Depends(get_db)
        ):
            """Get cleaning history for a dataset."""
            service = DataCleaningService(db)
            return service.get_cleaning_history(dataset_id, limit)
        
        @router.get("/audit/{session_id}")
        async def get_audit_log(
            session_id: str,
            db: Session = Depends(get_db)
        ):
            """Get audit log for a specific session."""
            service = DataCleaningService(db)
            return service.get_audit_log_details(session_id)
        
        return router
    
    # ============================================================
    # PHASE 10: DATA PIPELINE INTEGRATION
    # ============================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Export service state for pipeline integration."""
        return {
            "session_id": self.session_id,
            "transformations_applied": self.transformations_applied,
            "quality_metrics": self.quality_metrics,
            "fitted_scalers": {k: type(v).__name__ for k, v in self.fitted_scalers.items()},
            "fitted_encoders": {k: type(v).__name__ for k, v in self.fitted_encoders.items()},
            "encoding_mappings": self.encoding_mappings,
            "performance_metrics": self.performance_metrics,
            "errors_encountered": self.errors_encountered,
            "recovery_actions": self.recovery_actions
        }
    
    def export_pipeline_metadata(self, output_path: str) -> str:
        """Export complete pipeline metadata for MLOps integration."""
        
        metadata = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            
            "transformations": {
                "applied": self.transformations_applied,
                "scalers": list(self.fitted_scalers.keys()),
                "encoders": list(self.fitted_encoders.keys()),
                "encoding_mappings": self.encoding_mappings
            },
            
            "quality": self.quality_metrics,
            "performance": self.performance_metrics,
            "errors": {
                "count": len(self.errors_encountered),
                "details": self.errors_encountered
            },
            
            "audit": {
                "cleaning_log": self.cleaning_log,
                "recovery_actions": self.recovery_actions
            }
        }
        
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"📄 Exported pipeline metadata to: {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Failed to export metadata: {str(e)}")
            return ""

    # ... (keep all other Phase 1-9 methods) ...

    # ============================================================
    # PHASE 9: ROBUST ERROR HANDLING
    # ============================================================
    
    def _safe_execute(
        self,
        func: Callable,
        *args,
        operation_name: str = "",
        critical: bool = False,
        fallback_value: Any = None,
        config: Optional[CleaningConfig] = None,
        **kwargs
    ) -> Tuple[Any, bool]:
        """
        Safely execute a function with error handling.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            operation_name: Name of operation for logging
            critical: Whether this is a critical operation
            fallback_value: Value to return on failure
            config: Cleaning configuration
            **kwargs: Keyword arguments
            
        Returns:
            Tuple of (result, success)
        """
        try:
            result = func(*args, **kwargs)
            return result, True
        
        except Exception as e:
            error_info = {
                "operation": operation_name or func.__name__,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "traceback": traceback.format_exc(),
                "critical": critical
            }
            
            self.errors_encountered.append(error_info)
            
            logger.error(
                f"❌ Error in {operation_name}: {type(e).__name__}: {str(e)}"
            )
            
            self._log_step(
                "error",
                f"Failed: {operation_name} - {str(e)}",
                metadata=error_info
            )
            
            # Determine how to handle the error
            if critical or (config and config.strict_mode):
                # Re-raise in strict mode or for critical operations
                raise
            
            elif config and config.enable_error_recovery:
                # Log recovery action
                recovery_info = {
                    "operation": operation_name,
                    "action": "Used fallback value" if fallback_value is not None else "Skipped operation",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                self.recovery_actions.append(recovery_info)
                
                self._log_step(
                    "recovery",
                    f"Recovered from error in {operation_name}"
                )
                
                return fallback_value, False
            
            else:
                # Re-raise if error recovery is disabled
                raise
    
    @retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0)
    def _robust_read_dataframe(
        self,
        file_path: str,
        file_type: str,
        config: Optional[CleaningConfig] = None
    ) -> pd.DataFrame:
        """
        Read DataFrame with retry logic.
        
        Retries on transient failures like:
        - Network issues (for remote files)
        - Temporary file locks
        - Memory errors
        """
        try:
            return self._read_dataframe(file_path, file_type, config)
        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}")
            raise DataLoadError(f"Could not load data from {file_path}: {str(e)}") from e
    
    # ============================================================
    # PHASE 9: INPUT VALIDATION
    # ============================================================
    
    def _validate_input_data(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> Dict[str, Any]:
        """
        Validate input data before cleaning.
        
        Checks:
        - DataFrame is not empty
        - Column names are valid
        - No completely null columns
        - Data types are supported
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if DataFrame is empty
        if df.empty:
            validation_results["is_valid"] = False
            validation_results["errors"].append("DataFrame is empty")
            return validation_results
        
        # Check for invalid column names
        invalid_cols = []
        for col in df.columns:
            if not isinstance(col, str) or col.strip() == "":
                invalid_cols.append(col)
        
        if invalid_cols:
            if config.validation_mode == "strict":
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Invalid column names: {invalid_cols}")
            else:
                validation_results["warnings"].append(f"Invalid column names: {invalid_cols}")
        
        # Check for completely null columns
        null_cols = df.columns[df.isnull().all()].tolist()
        if null_cols:
            if config.validation_mode == "strict":
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"Completely null columns: {null_cols}")
            else:
                validation_results["warnings"].append(f"Completely null columns: {null_cols}")
        
        # Check for unsupported data types
        unsupported_types = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            if dtype.startswith('complex'):
                unsupported_types.append(col)
        
        if unsupported_types:
            validation_results["warnings"].append(f"Columns with unsupported types: {unsupported_types}")
        
        # Check data size sanity
        if len(df) > 10_000_000:  # 10 million rows
            validation_results["warnings"].append(
                f"Very large dataset ({len(df):,} rows). Consider using chunked processing."
            )
        
        return validation_results
    
    # ============================================================
    # PHASE 9: GRACEFUL DEGRADATION
    # ============================================================
    
    def _handle_missing_values_robust(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> pd.DataFrame:
        """
        Handle missing values with graceful degradation.
        
        Fallback hierarchy:
        1. Configured strategy (KNN, MICE, etc.)
        2. Simple imputation (median/mode)
        3. Drop rows (last resort)
        """
        with ErrorContext(self, "missing_value_handling", critical=False):
            try:
                return self._handle_missing_values_advanced(df, config)
            
            except Exception as e:
                logger.warning(f"⚠️ Advanced imputation failed: {str(e)}. Using simple imputation.")
                
                # Fallback to simple imputation
                for col in df.columns:
                    if df[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            if not df[col].mode().empty:
                                df[col].fillna(df[col].mode()[0], inplace=True)
                
                self._log_step("fallback", "Used simple imputation as fallback")
                self.recovery_actions.append({
                    "operation": "missing_value_handling",
                    "action": "Fallback to simple imputation",
                    "reason": str(e)
                })
                
                return df
    
    # ============================================================
    # PHASE 9: ERROR REPORTING
    # ============================================================
    
    def _generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
        
        return {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            
            "summary": {
                "total_errors": len(self.errors_encountered),
                "total_warnings": len(self.warnings_encountered),
                "total_recoveries": len(self.recovery_actions),
                "critical_errors": sum(1 for e in self.errors_encountered if e.get("critical", False))
            },
            
            "errors": self.errors_encountered,
            "warnings": self.warnings_encountered,
            "recovery_actions": self.recovery_actions,
            
            "recommendations": self._generate_error_recommendations()
        }
    
    def _generate_error_recommendations(self) -> List[str]:
        """Generate recommendations based on errors encountered."""
        
        recommendations = []
        
        # Check error patterns
        error_types = [e["error_type"] for e in self.errors_encountered]
        
        if "MemoryError" in error_types:
            recommendations.append("Enable chunked processing to reduce memory usage")
            recommendations.append("Consider using a machine with more RAM")
        
        if "KeyError" in error_types:
            recommendations.append("Verify column names in your configuration")
            recommendations.append("Check schema validation settings")
        
        if "ValueError" in error_types:
            recommendations.append("Review data type conversions")
            recommendations.append("Enable auto-fix for schema violations")
        
        if len(self.errors_encountered) > 10:
            recommendations.append("Consider using strict_mode=False for more graceful error handling")
        
        return recommendations
    
    def _save_error_report(self, output_path: str) -> str:
        """Save error report to file."""
        
        try:
            report = self._generate_error_report()
            
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📄 Saved error report to: {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Failed to save error report: {str(e)}")
            return ""
    
    # ============================================================
    # PHASE 9: HEALTH CHECKS
    # ============================================================
    
    def _perform_health_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform health check on DataFrame.
        
        Returns health status and warnings.
        """
        health = {
            "status": "healthy",
            "checks": {},
            "warnings": []
        }
        
        # Check 1: Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        health["checks"]["memory_usage_mb"] = round(memory_mb, 2)
        
        if memory_mb > 1000:  # 1GB
            health["warnings"].append(f"High memory usage: {memory_mb:.0f} MB")
            health["status"] = "warning"
        
        # Check 2: Missing data ratio
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        health["checks"]["missing_ratio"] = round(missing_ratio, 4)
        
        if missing_ratio > 0.5:
            health["warnings"].append(f"High missing data ratio: {missing_ratio*100:.1f}%")
            health["status"] = "warning"
        
        # Check 3: Duplicate ratio
        duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0
        health["checks"]["duplicate_ratio"] = round(duplicate_ratio, 4)
        
        if duplicate_ratio > 0.3:
            health["warnings"].append(f"High duplicate ratio: {duplicate_ratio*100:.1f}%")
            health["status"] = "warning"
        
        # Check 4: Data types
        unsupported = []
        for col in df.columns:
            if str(df[col].dtype).startswith('complex'):
                unsupported.append(col)
        
        if unsupported:
            health["warnings"].append(f"Unsupported data types in: {unsupported}")
            health["status"] = "warning"
        
        return health
    
    # ============================================================
    # PHASE 9: DEFENSIVE CLEANING
    # ============================================================
    
    def clean_dataset_robust(
        self,
        dataset_id: int,
        config: CleaningConfig
    ) -> Dict[str, Any]:
        """
        Execute cleaning pipeline with comprehensive error handling.
        
        This is the main entry point with full Phase 9 robustness.
        """
        # Initialize error tracking
        self.errors_encountered = []
        self.warnings_encountered = []
        self.recovery_actions = []
        
        try:
            # Validate dataset exists
            dataset = self.db.get(Dataset, dataset_id)
            if not dataset:
                raise DataValidationError(f"Dataset {dataset_id} not found")
            
            # Load configuration with error handling
            config, success = self._safe_execute(
                self._load_configuration,
                config,
                operation_name="load_configuration",
                critical=True,
                config=config
            )
            
            if not success:
                raise ConfigurationError("Failed to load configuration")
            
            # Execute main cleaning with error recovery
            return self.clean_dataset(dataset_id, config)
        
        except Exception as e:
            # Log final error
            logger.error(f"❌ Cleaning pipeline failed: {str(e)}", exc_info=True)
            
            # Generate and save error report
            if config.error_report_path:
                self._save_error_report(config.error_report_path)
            
            # Update dataset status
            try:
                dataset = self.db.get(Dataset, dataset_id)
                if dataset:
                    dataset.status = DatasetStatus.FAILED
                    dataset.processing_error = str(e)
                    self.db.commit()
            except:
                pass
            
            raise

    # ============================================================
    # PHASE 8: CONFIGURATION MANAGEMENT
    # ============================================================
    
    def _load_configuration(self, config: CleaningConfig) -> CleaningConfig:
        """
        Load and merge configuration from multiple sources.
        
        Priority (highest to lowest):
        1. Explicit config parameters
        2. External config file
        3. Preset configuration
        4. Default values
        """
        logger.info("⚙️ Loading configuration")
        
        # Start with default config
        final_config = config.dict()
        
        # Load from preset if specified
        if config.preset and config.preset in PRESET_CONFIGS:
            preset_config = PRESET_CONFIGS[config.preset]
            
            if config.extend_preset:
                # Merge: preset + explicit config
                merged = {**preset_config, **{k: v for k, v in final_config.items() if v is not None}}
                final_config = merged
            else:
                # Replace with preset, keep only explicit overrides
                explicit_overrides = {k: v for k, v in final_config.items() if v is not None and k not in ['preset', 'extend_preset']}
                final_config = {**preset_config, **explicit_overrides}
            
            logger.info(f"📋 Loaded preset: {config.preset}")
            self._log_step("config", f"Using preset: {config.preset}")
        
        # Load from external file if specified
        if config.config_file_path:
            file_config = self._load_config_from_file(config.config_file_path)
            final_config = {**final_config, **file_config}
            logger.info(f"📁 Loaded config from: {config.config_file_path}")
        
        return CleaningConfig(**final_config)
    
    def _load_config_from_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}
        
        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif path.suffix == '.json':
                    return json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {path.suffix}")
                    return {}
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            return {}
    
    def _save_config_to_file(
        self,
        config: CleaningConfig,
        output_path: str,
        format: str = "yaml"
    ) -> str:
        """Save configuration to file for reuse."""
        
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = config.dict(exclude_none=True)
            
            with open(path, 'w') as f:
                if format == "yaml":
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format == "json":
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"💾 Saved config to: {path}")
            return str(path)
            
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
            return ""
    
    # ============================================================
    # PHASE 8: PLUGIN SYSTEM
    # ============================================================
    
    def register_plugin(
        self,
        plugin: CleaningPlugin,
        execution_order: Optional[int] = None
    ) -> None:
        """
        Register a custom cleaning plugin.
        
        Args:
            plugin: Plugin instance
            execution_order: Order in which to execute (lower = earlier)
        """
        if not plugin.validate_config():
            raise ValueError(f"Invalid configuration for plugin: {plugin.name}")
        
        self.registered_plugins[plugin.name] = plugin
        
        if execution_order is not None:
            self.plugin_execution_order.insert(execution_order, plugin.name)
        else:
            self.plugin_execution_order.append(plugin.name)
        
        logger.info(f"🔌 Registered plugin: {plugin.name}")
        self._log_step("plugin_registration", f"Registered plugin: {plugin.name}")
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister a plugin."""
        if plugin_name in self.registered_plugins:
            del self.registered_plugins[plugin_name]
            if plugin_name in self.plugin_execution_order:
                self.plugin_execution_order.remove(plugin_name)
            logger.info(f"🔌 Unregistered plugin: {plugin_name}")
    
    def _execute_plugins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all registered plugins in order."""
        
        if not self.registered_plugins:
            return df
        
        logger.info(f"🔌 Executing {len(self.registered_plugins)} custom plugins")
        
        for plugin_name in self.plugin_execution_order:
            if plugin_name in self.registered_plugins:
                plugin = self.registered_plugins[plugin_name]
                
                try:
                    logger.info(f"🔌 Executing plugin: {plugin_name}")
                    df = plugin.execute(df)
                    self._log_step("plugin_execution", f"Executed plugin: {plugin_name}")
                    self.transformations_applied.append(f"Plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Plugin {plugin_name} failed: {str(e)}")
                    self._log_step("plugin_error", f"Plugin {plugin_name} failed: {str(e)}")
        
        return df
    
    def _load_plugins_from_config(self, config: CleaningConfig) -> None:
        """Load and register plugins from configuration."""
        
        if not config.custom_plugins:
            return
        
        for plugin_config in config.custom_plugins:
            try:
                plugin_class_path = plugin_config.get("class")
                plugin_params = plugin_config.get("config", {})
                
                # Dynamic import
                module_path, class_name = plugin_class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                plugin_class = getattr(module, class_name)
                
                # Instantiate and register
                plugin = plugin_class(plugin_params)
                self.register_plugin(plugin)
                
            except Exception as e:
                logger.error(f"Failed to load plugin: {str(e)}")
    
    # ============================================================
    # PHASE 8: PIPELINE BUILDER
    # ============================================================
    
    class PipelineBuilder:
        """Fluent API for building cleaning pipelines."""
        
        def __init__(self, service: 'DataCleaningService'):
            self.service = service
            self.config = CleaningConfig()
        
        def use_preset(self, preset: CleaningPreset) -> 'DataCleaningService.PipelineBuilder':
            """Use a configuration preset."""
            self.config.preset = preset.value
            return self
        
        def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'DataCleaningService.PipelineBuilder':
            """Enable duplicate removal."""
            self.config.remove_duplicates = True
            self.config.duplicate_subset = subset
            return self
        
        def handle_missing(self, strategy: str, **kwargs) -> 'DataCleaningService.PipelineBuilder':
            """Configure missing value handling."""
            self.config.missing_strategy = strategy
            for key, value in kwargs.items():
                setattr(self.config, key, value)
            return self
        
        def detect_outliers(self, method: str, treatment: str = "cap") -> 'DataCleaningService.PipelineBuilder':
            """Configure outlier detection."""
            self.config.outlier_detection = True
            self.config.outlier_method = method
            self.config.outlier_treatment = treatment
            return self
        
        def scale_features(self, method: str, columns: Optional[List[str]] = None) -> 'DataCleaningService.PipelineBuilder':
            """Configure feature scaling."""
            self.config.scaling_method = method
            self.config.columns_to_scale = columns
            return self
        
        def encode_categorical(self, method: str, columns: Optional[List[str]] = None) -> 'DataCleaningService.PipelineBuilder':
            """Configure categorical encoding."""
            self.config.encoding_method = method
            self.config.columns_to_encode = columns
            return self
        
        def add_plugin(self, plugin: CleaningPlugin) -> 'DataCleaningService.PipelineBuilder':
            """Add a custom plugin."""
            self.service.register_plugin(plugin)
            return self
        
        def enable_parallel(self, max_workers: Optional[int] = None) -> 'DataCleaningService.PipelineBuilder':
            """Enable parallel processing."""
            self.config.enable_parallel_processing = True
            self.config.max_workers = max_workers
            return self
        
        def with_quality_thresholds(self, **thresholds) -> 'DataCleaningService.PipelineBuilder':
            """Set quality thresholds."""
            self.config.quality_thresholds = thresholds
            return self
        
        def build(self) -> CleaningConfig:
            """Build and return the configuration."""
            return self.config
    
    def pipeline(self) -> PipelineBuilder:
        """Create a new pipeline builder."""
        return self.PipelineBuilder(self)
    
    # ============================================================
    # PHASE 8: CONFIGURATION TEMPLATES
    # ============================================================
    
    @staticmethod
    def get_preset_config(preset: CleaningPreset) -> CleaningConfig:
        """Get a preset configuration."""
        if preset.value not in PRESET_CONFIGS:
            raise ValueError(f"Unknown preset: {preset.value}")
        
        return CleaningConfig(**PRESET_CONFIGS[preset.value])
    
    @staticmethod
    def list_available_presets() -> List[Dict[str, Any]]:
        """List all available configuration presets."""
        return [
            {
                "name": preset.value,
                "description": f"{preset.value.replace('_', ' ').title()} cleaning configuration",
                "config": PRESET_CONFIGS[preset]
            }
            for preset in CleaningPreset if preset != CleaningPreset.CUSTOM
        ]
    
    @staticmethod
    def create_custom_preset(
        name: str,
        config: Dict[str, Any],
        description: str = ""
    ) -> None:
        """Create and register a custom preset."""
        PRESET_CONFIGS[name] = config
        logger.info(f"📋 Created custom preset: {name}")
    
    # ============================================================
    # PHASE 8: CONFIGURATION VALIDATION
    # ============================================================
    
    def _validate_configuration(self, config: CleaningConfig) -> Dict[str, Any]:
        """Validate configuration for conflicts and issues."""
        
        warnings = []
        errors = []
        
        # Check conflicting options
        if config.missing_strategy == "drop" and config.outlier_treatment == "impute":
            warnings.append("Missing strategy 'drop' may conflict with outlier imputation")
        
        # Check performance settings
        if config.enable_parallel_processing and config.chunk_size:
            warnings.append("Parallel processing with chunking may not provide optimal performance")
        
        # Check encoding limits
        if config.encoding_method == "onehot" and not config.max_categories_onehot:
            warnings.append("One-hot encoding without category limit may create too many columns")
        
        # Check required dependencies
        if config.missing_strategy == "mice":
            try:
                from sklearn.experimental import enable_iterative_imputer
            except ImportError:
                errors.append("MICE imputation requires sklearn experimental features")
        
        return {
            "is_valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }
    
    # Update clean_dataset to use configuration loading
    def clean_dataset(
        self,
        dataset_id: int,
        config: CleaningConfig
    ) -> Dict[str, Any]:
        """Execute complete cleaning pipeline with Phase 1-8."""
        
        # Load and merge configuration
        config = self._load_configuration(config)
        
        # Validate configuration
        validation = self._validate_configuration(config)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration: {validation['errors']}"
            )
        
        if validation["warnings"]:
            for warning in validation["warnings"]:
                logger.warning(f"⚠️ Config warning: {warning}")
        
        # Load plugins from config
        self._load_plugins_from_config(config)
        
        # ... (continue with existing clean_dataset logic) ...
        
        # Execute custom plugins after core cleaning
        # df = self._execute_plugins(df)
        
        # ... (rest of cleaning pipeline) ...


    # ============================================================
    # PHASE 7: INTELLIGENT DATA LOADING
    # ============================================================
    
    def _read_dataframe(
        self,
        file_path: str,
        file_type: str,
        config: Optional[CleaningConfig] = None
    ) -> pd.DataFrame:
        """
        Intelligent data reading with chunking and memory management.
        
        Features:
        - Automatic chunking for large files
        - Memory-efficient loading
        - Progress tracking
        - Type inference optimization
        """
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        
        logger.info(f"📂 Reading file: {file_path} ({file_size_mb:.2f} MB)")
        
        # Determine if chunking is needed
        should_chunk = False
        chunk_size = None
        
        if config:
            if config.chunk_size:
                should_chunk = True
                chunk_size = config.chunk_size
            elif config.memory_limit_mb and file_size_mb > config.memory_limit_mb:
                should_chunk = True
                # Estimate chunk size based on memory limit
                chunk_size = int(config.memory_limit_mb * 1000)  # Rough estimate
        
        # Default chunking for very large files (>500MB)
        if file_size_mb > 500 and not should_chunk:
            should_chunk = True
            chunk_size = 50000
        
        try:
            if file_type == ".csv":
                if should_chunk:
                    return self._read_csv_chunked(file_path, chunk_size, config)
                else:
                    return pd.read_csv(file_path, low_memory=True)
            
            elif file_type in [".xlsx", ".xls"]:
                # Excel files are read into memory entirely
                return pd.read_excel(file_path)
            
            elif file_type == ".json":
                if should_chunk:
                    return self._read_json_chunked(file_path, chunk_size)
                else:
                    return pd.read_json(file_path)
            
            elif file_type == ".parquet":
                return pd.read_parquet(file_path)
            
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read file: {str(e)}"
            ) from e
    
    def _read_csv_chunked(
        self,
        file_path: str,
        chunk_size: int,
        config: Optional[CleaningConfig] = None
    ) -> pd.DataFrame:
        """Read large CSV files in chunks and combine."""
        
        logger.info(f"📊 Reading CSV in chunks of {chunk_size:,} rows")
        
        chunks = []
        total_rows = 0
        
        try:
            # Read in chunks
            chunk_iterator = pd.read_csv(file_path, chunksize=chunk_size, low_memory=True)
            
            for i, chunk in enumerate(chunk_iterator):
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # Progress update
                if config and config.progress_callback:
                    config.progress_callback(f"Reading chunk {i+1}...")
                
                # Memory management
                if config and config.optimize_memory and i % 10 == 0:
                    gc.collect()
            
            # Combine all chunks
            logger.info(f"🔗 Combining {len(chunks)} chunks ({total_rows:,} total rows)")
            df = pd.concat(chunks, ignore_index=True)
            
            # Clear chunks from memory
            del chunks
            gc.collect()
            
            return df
            
        except Exception as e:
            logger.error(f"Chunked reading failed: {str(e)}")
            raise
    
    def _read_json_chunked(
        self,
        file_path: str,
        chunk_size: int
    ) -> pd.DataFrame:
        """Read large JSON files in chunks."""
        
        logger.info(f"📊 Reading JSON in chunks")
        
        try:
            # For line-delimited JSON
            chunks = []
            with open(file_path, 'r') as f:
                chunk = []
                for i, line in enumerate(f):
                    chunk.append(json.loads(line))
                    
                    if len(chunk) >= chunk_size:
                        chunks.append(pd.DataFrame(chunk))
                        chunk = []
                
                # Add remaining
                if chunk:
                    chunks.append(pd.DataFrame(chunk))
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            
            return df
            
        except Exception as e:
            logger.error(f"Chunked JSON reading failed: {str(e)}")
            # Fallback to regular read
            return pd.read_json(file_path)
    
    # ============================================================
    # PHASE 7: PARALLEL PROCESSING
    # ============================================================
    
    def _process_columns_parallel(
        self,
        df: pd.DataFrame,
        func: Callable,
        columns: List[str],
        config: CleaningConfig,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process multiple columns in parallel.
        
        Args:
            df: DataFrame
            func: Function to apply to each column
            columns: List of columns to process
            config: Cleaning config
            **kwargs: Additional arguments for func
        """
        if not config.enable_parallel_processing or len(columns) < 2:
            # Sequential processing
            for col in columns:
                df = func(df, col, **kwargs)
            return df
        
        logger.info(f"⚡ Processing {len(columns)} columns in parallel")
        
        max_workers = config.max_workers or multiprocessing.cpu_count()
        
        try:
            if config.use_multiprocessing:
                # Multiprocessing for CPU-intensive tasks
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(func, df.copy(), col, **kwargs): col 
                        for col in columns
                    }
                    
                    for future in as_completed(futures):
                        col = futures[future]
                        try:
                            result_df = future.result()
                            df[col] = result_df[col]
                        except Exception as e:
                            logger.error(f"Parallel processing failed for {col}: {str(e)}")
            else:
                # Threading for I/O-bound tasks
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(func, df, col, **kwargs): col 
                        for col in columns
                    }
                    
                    for future in as_completed(futures):
                        col = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Parallel processing failed for {col}: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Parallel processing failed: {str(e)}")
            # Fallback to sequential
            for col in columns:
                df = func(df, col, **kwargs)
            return df
    
    # ============================================================
    # PHASE 7: MEMORY OPTIMIZATION
    # ============================================================
    
    def _optimize_memory_usage(
        self,
        df: pd.DataFrame,
        aggressive: bool = False
    ) -> pd.DataFrame:
        """
        Aggressively optimize memory usage.
        
        Techniques:
        - Downcast numeric types
        - Convert to categorical
        - Remove unused categories
        - Optimize string storage
        """
        logger.info("💾 Optimizing memory usage")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Downcast integers
        int_cols = df.select_dtypes(include=['int']).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast floats
        float_cols = df.select_dtypes(include=['float']).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert to category (aggressive mode)
        if aggressive:
            object_cols = df.select_dtypes(include=['object']).columns
            for col in object_cols:
                num_unique = df[col].nunique()
                num_total = len(df)
                
                # Convert if less than 50% unique
                if num_unique / num_total < 0.5:
                    df[col] = df[col].astype('category')
        
        # Remove unused categories
        cat_cols = df.select_dtypes(include=['category']).columns
        for col in cat_cols:
            df[col] = df[col].cat.remove_unused_categories()
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        saved = initial_memory - final_memory
        
        if saved > 0:
            logger.info(f"💾 Memory optimized: {saved:.2f} MB saved ({saved/initial_memory*100:.1f}%)")
        
        return df
    
    def _clear_cache(self):
        """Clear internal cache and force garbage collection."""
        self.cache.clear()
        gc.collect()
        logger.info("🗑️ Cache cleared and garbage collected")
    
    # ============================================================
    # PHASE 7: BATCH PROCESSING
    # ============================================================
    
    def _process_in_batches(
        self,
        df: pd.DataFrame,
        func: Callable,
        batch_size: int,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process DataFrame in batches for memory efficiency.
        
        Args:
            df: DataFrame to process
            func: Function to apply to each batch
            batch_size: Number of rows per batch
            **kwargs: Additional arguments for func
        """
        logger.info(f"📦 Processing in batches of {batch_size:,} rows")
        
        n_batches = int(np.ceil(len(df) / batch_size))
        results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            
            batch = df.iloc[start_idx:end_idx].copy()
            processed_batch = func(batch, **kwargs)
            results.append(processed_batch)
            
            # Progress
            progress = ((i + 1) / n_batches) * 100
            logger.info(f"📊 Batch {i+1}/{n_batches} complete ({progress:.1f}%)")
            
            # Memory management
            del batch
            if i % 10 == 0:
                gc.collect()
        
        # Combine results
        result_df = pd.concat(results, ignore_index=True)
        del results
        gc.collect()
        
        return result_df
    
    # ============================================================
    # PHASE 7: PROGRESS TRACKING
    # ============================================================
    
    def _update_progress(
        self,
        current: int,
        total: int,
        message: str = "",
        config: Optional[CleaningConfig] = None
    ) -> None:
        """Update progress and call callback if configured."""
        
        if total == 0:
            return
        
        progress = (current / total) * 100
        self.current_progress = progress
        
        log_message = f"📊 Progress: {progress:.1f}% ({current}/{total})"
        if message:
            log_message += f" - {message}"
        
        logger.info(log_message)
        
        # Call progress callback if configured
        if config and config.progress_callback:
            try:
                config.progress_callback({
                    "progress": progress,
                    "current": current,
                    "total": total,
                    "message": message
                })
            except Exception as e:
                logger.warning(f"Progress callback failed: {str(e)}")
    
    # ============================================================
    # PHASE 7: PERFORMANCE MONITORING
    # ============================================================
    
    def _record_performance_metric(
        self,
        operation: str,
        duration_seconds: float,
        rows_processed: int = 0,
        memory_used_mb: float = 0
    ) -> None:
        """Record performance metrics for operations."""
        
        self.performance_metrics[operation] = {
            "duration_seconds": round(duration_seconds, 3),
            "rows_processed": rows_processed,
            "rows_per_second": round(rows_processed / duration_seconds, 2) if duration_seconds > 0 else 0,
            "memory_used_mb": round(memory_used_mb, 2)
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        
        total_duration = sum(m["duration_seconds"] for m in self.performance_metrics.values())
        total_rows = sum(m["rows_processed"] for m in self.performance_metrics.values())
        
        return {
            "total_duration_seconds": round(total_duration, 2),
            "total_rows_processed": total_rows,
            "average_throughput_rows_per_sec": round(total_rows / total_duration, 2) if total_duration > 0 else 0,
            "operations": self.performance_metrics
        }
    
    # ============================================================
    # PHASE 7: OPTIMIZED DUPLICATE REMOVAL
    # ============================================================
    
    def _remove_duplicates_optimized(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        config: Optional[CleaningConfig] = None
    ) -> pd.DataFrame:
        """Memory-efficient duplicate removal for large datasets."""
        
        start_time = datetime.now(timezone.utc)
        initial_rows = len(df)
        
        if config and config.chunk_size and len(df) > config.chunk_size * 2:
            # Process in chunks for very large datasets
            logger.info("🔄 Removing duplicates using chunked approach")
            
            # First pass: identify duplicates
            duplicate_mask = df.duplicated(subset=subset, keep='first')
            
            # Second pass: filter
            df = df[~duplicate_mask].copy()
            
            del duplicate_mask
            gc.collect()
        else:
            # Standard approach
            df = df.drop_duplicates(subset=subset, keep='first')
        
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        removed = initial_rows - len(df)
        
        self._record_performance_metric(
            "remove_duplicates",
            duration,
            rows_processed=initial_rows
        )
        
        if removed > 0:
            logger.info(f"✂️ Removed {removed:,} duplicates in {duration:.2f}s")
            self._log_step("remove_duplicates", f"Removed {removed:,} duplicate rows")
            self.transformations_applied.append(f"Duplicate Removal ({removed:,} rows)")
        
        return df
    
    # ============================================================
    # PHASE 7: LAZY EVALUATION & CACHING
    # ============================================================
    
    def _get_cached_or_compute(
        self,
        cache_key: str,
        compute_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Get cached result or compute and cache it."""
        
        if cache_key in self.cache:
            logger.info(f"📦 Using cached result for: {cache_key}")
            return self.cache[cache_key]
        
        # Compute
        result = compute_func(*args, **kwargs)
        
        # Cache
        self.cache[cache_key] = result
        
        return result
        
    # ============================================================
    # PHASE 6: FEATURE SCALING
    # ============================================================
    
    def _scale_numerical_features(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> pd.DataFrame:
        """
        Scale numerical features using specified method.
        
        Methods:
        - standard: StandardScaler (mean=0, std=1)
        - minmax: MinMaxScaler (range 0-1)
        - robust: RobustScaler (median-based, robust to outliers)
        - maxabs: MaxAbsScaler (range -1 to 1)
        """
        logger.info(f"📏 Scaling features using method: {config.scaling_method}")
        
        # Determine columns to scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if config.columns_to_scale:
            cols_to_scale = [c for c in config.columns_to_scale if c in df.columns]
        else:
            cols_to_scale = numerical_cols
        
        # Exclude specified columns
        if config.exclude_from_scaling:
            cols_to_scale = [c for c in cols_to_scale if c not in config.exclude_from_scaling]
        
        if not cols_to_scale:
            self._log_step("scaling", "No columns to scale")
            return df
        
        # Select scaler
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
            "maxabs": MaxAbsScaler()
        }
        
        if config.scaling_method not in scaler_map:
            logger.warning(f"Unknown scaling method: {config.scaling_method}")
            return df
        
        scaler = scaler_map[config.scaling_method]
        
        try:
            # Fit and transform
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            
            # Store fitted scaler
            for col in cols_to_scale:
                self.fitted_scalers[col] = {
                    "method": config.scaling_method,
                    "scaler": scaler
                }
            
            self._log_step(
                "feature_scaling",
                f"Scaled {len(cols_to_scale)} columns using {config.scaling_method}: {cols_to_scale}"
            )
            
            self.transformations_applied.append(
                f"Feature Scaling ({config.scaling_method}) on {len(cols_to_scale)} columns"
            )
            
        except Exception as e:
            self._log_step("scaling_error", f"Scaling failed: {str(e)}")
            logger.error(f"Scaling error: {str(e)}", exc_info=True)
        
        return df
    
    # ============================================================
    # PHASE 6: CATEGORICAL ENCODING
    # ============================================================
    
    def _encode_categorical_features(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> pd.DataFrame:
        """
        Encode categorical features using specified method.
        
        Methods:
        - label: LabelEncoder (0, 1, 2, ...)
        - onehot: One-Hot Encoding (binary columns)
        - ordinal: Ordinal Encoding with custom order
        - target: Target Encoding (mean of target)
        - binary: Binary Encoding
        """
        logger.info(f"🔤 Encoding categorical features using method: {config.encoding_method}")
        
        # Determine columns to encode
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if config.columns_to_encode:
            cols_to_encode = [c for c in config.columns_to_encode if c in df.columns]
        else:
            cols_to_encode = categorical_cols
        
        # Exclude specified columns
        if config.exclude_from_encoding:
            cols_to_encode = [c for c in cols_to_encode if c not in config.exclude_from_encoding]
        
        if not cols_to_encode:
            self._log_step("encoding", "No columns to encode")
            return df
        
        # Route to appropriate encoding method
        if config.encoding_method == "label":
            df = self._label_encode(df, cols_to_encode)
        
        elif config.encoding_method == "onehot":
            df = self._onehot_encode(df, cols_to_encode, config.max_categories_onehot)
        
        elif config.encoding_method == "ordinal":
            df = self._ordinal_encode(df, cols_to_encode, config.ordinal_mappings)
        
        elif config.encoding_method == "target":
            if config.target_column_for_encoding and config.target_column_for_encoding in df.columns:
                df = self._target_encode(df, cols_to_encode, config.target_column_for_encoding)
            else:
                logger.warning("Target encoding requires target_column_for_encoding to be specified")
        
        elif config.encoding_method == "binary":
            df = self._binary_encode(df, cols_to_encode)
        
        return df
    
    def _label_encode(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Label encode categorical columns."""
        
        for col in columns:
            try:
                le = LabelEncoder()
                # Handle NaN
                non_null_mask = df[col].notna()
                df.loc[non_null_mask, col] = le.fit_transform(df.loc[non_null_mask, col].astype(str))
                
                # Store encoder and mapping
                self.fitted_encoders[col] = le
                self.encoding_mappings[col] = {
                    "method": "label",
                    "classes": le.classes_.tolist(),
                    "n_classes": len(le.classes_)
                }
                
                self._log_step(
                    "label_encoding",
                    f"Label encoded '{col}' ({len(le.classes_)} classes)"
                )
                
            except Exception as e:
                self._log_step("encoding_error", f"Failed to label encode '{col}': {str(e)}")
        
        if columns:
            self.transformations_applied.append(f"Label Encoding ({len(columns)} columns)")
        
        return df
    
    def _onehot_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        max_categories: int
    ) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        
        encoded_cols = []
        
        for col in columns:
            n_unique = df[col].nunique()
            
            # Skip if too many categories
            if n_unique > max_categories:
                self._log_step(
                    "onehot_skip",
                    f"Skipped '{col}' (too many categories: {n_unique} > {max_categories})"
                )
                continue
            
            try:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                
                # Drop original column
                df = df.drop(columns=[col])
                
                # Add dummy columns
                df = pd.concat([df, dummies], axis=1)
                
                encoded_cols.append(col)
                
                self.encoding_mappings[col] = {
                    "method": "onehot",
                    "n_categories": n_unique,
                    "new_columns": dummies.columns.tolist()
                }
                
                self._log_step(
                    "onehot_encoding",
                    f"One-hot encoded '{col}' into {len(dummies.columns)} columns"
                )
                
            except Exception as e:
                self._log_step("encoding_error", f"Failed to one-hot encode '{col}': {str(e)}")
        
        if encoded_cols:
            self.transformations_applied.append(f"One-Hot Encoding ({len(encoded_cols)} columns)")
        
        return df
    
    def _ordinal_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        mappings: Optional[Dict[str, List[str]]]
    ) -> pd.DataFrame:
        """Ordinal encode with custom ordering."""
        
        if not mappings:
            logger.warning("Ordinal encoding requires ordinal_mappings to be specified")
            return df
        
        for col in columns:
            if col not in mappings:
                continue
            
            try:
                order = mappings[col]
                mapping_dict = {val: idx for idx, val in enumerate(order)}
                
                df[col] = df[col].map(mapping_dict)
                
                self.encoding_mappings[col] = {
                    "method": "ordinal",
                    "order": order,
                    "mapping": mapping_dict
                }
                
                self._log_step(
                    "ordinal_encoding",
                    f"Ordinal encoded '{col}' with order: {order}"
                )
                
            except Exception as e:
                self._log_step("encoding_error", f"Failed to ordinal encode '{col}': {str(e)}")
        
        self.transformations_applied.append(f"Ordinal Encoding ({len(columns)} columns)")
        
        return df
    
    def _target_encode(
        self,
        df: pd.DataFrame,
        columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Target encode (mean encoding) based on target variable."""
        
        for col in columns:
            try:
                # Calculate mean of target for each category
                target_means = df.groupby(col)[target_column].mean()
                
                # Map to column
                df[f"{col}_target_encoded"] = df[col].map(target_means)
                
                # Optionally drop original
                # df = df.drop(columns=[col])
                
                self.encoding_mappings[col] = {
                    "method": "target",
                    "target_column": target_column,
                    "means": target_means.to_dict(),
                    "new_column": f"{col}_target_encoded"
                }
                
                self._log_step(
                    "target_encoding",
                    f"Target encoded '{col}' based on '{target_column}'"
                )
                
            except Exception as e:
                self._log_step("encoding_error", f"Failed to target encode '{col}': {str(e)}")
        
        self.transformations_applied.append(f"Target Encoding ({len(columns)} columns)")
        
        return df
    
    def _binary_encode(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Binary encode categorical columns (more compact than one-hot)."""
        
        for col in columns:
            try:
                # Get unique values
                unique_vals = df[col].unique()
                n_unique = len(unique_vals)
                
                # Calculate number of binary columns needed
                n_bits = int(np.ceil(np.log2(n_unique)))
                
                # Create mapping
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                
                # Convert to binary
                binary_cols = []
                for bit in range(n_bits):
                    col_name = f"{col}_bin_{bit}"
                    df[col_name] = df[col].map(mapping).apply(lambda x: (x >> bit) & 1)
                    binary_cols.append(col_name)
                
                self.encoding_mappings[col] = {
                    "method": "binary",
                    "n_categories": n_unique,
                    "n_bits": n_bits,
                    "new_columns": binary_cols
                }
                
                self._log_step(
                    "binary_encoding",
                    f"Binary encoded '{col}' into {n_bits} binary columns"
                )
                
            except Exception as e:
                self._log_step("encoding_error", f"Failed to binary encode '{col}': {str(e)}")
        
        self.transformations_applied.append(f"Binary Encoding ({len(columns)} columns)")
        
        return df
    
    # ============================================================
    # SAVE TRANSFORMERS
    # ============================================================
    
    def _save_transformers(
        self,
        dataset_id: int,
        transformer_type: str
    ) -> None:
        """Save fitted scalers or encoders to disk for reuse."""
        
        try:
            # Create transformers directory
            transformers_dir = Path(settings.UPLOAD_DIR) / "transformers"
            transformers_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{transformer_type}_dataset_{dataset_id}_{self.session_id}_{timestamp}.pkl"
            file_path = transformers_dir / filename
            
            # Save transformers
            if transformer_type == "scalers":
                data_to_save = self.fitted_scalers
            elif transformer_type == "encoders":
                data_to_save = self.fitted_encoders
            else:
                return
            
            with open(file_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            logger.info(f"💾 Saved {transformer_type} to {file_path}")
            self._log_step("save_transformers", f"Saved {transformer_type} to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save {transformer_type}: {str(e)}")

    # ============================================================
    # PHASE 5: ADVANCED DATA TYPE OPTIMIZATION
    # ============================================================
    
    def _optimize_data_types_advanced(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> pd.DataFrame:
        """
        Advanced data type optimization for memory efficiency and performance.
        
        Features:
        - Auto-detect and convert date/datetime columns
        - Convert low-cardinality strings to category
        - Downcast integers and floats
        - Detect and fix string/numeric mismatches
        - Convert boolean-like columns to bool
        
        Returns:
            DataFrame with optimized types
        """
        logger.info("🔧 Optimizing data types")
        
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        # Step 1: Auto-detect and convert date columns
        if config.auto_detect_dates:
            df = self._auto_detect_dates(df)
        
        # Step 2: Convert low-cardinality strings to category
        df = self._optimize_categorical_columns(df, config.category_threshold)
        
        # Step 3: Detect and fix string/numeric mismatches
        df = self._fix_type_mismatches(df)
        
        # Step 4: Convert boolean-like columns
        df = self._convert_boolean_columns(df)
        
        # Step 5: Downcast numeric types
        if config.downcast_integers:
            df = self._downcast_integers(df)
        
        if config.downcast_floats:
            df = self._downcast_floats(df)
        
        # Calculate memory savings
        final_memory = df.memory_usage(deep=True).sum() / 1024**2
        memory_saved = initial_memory - final_memory
        
        if memory_saved > 0:
            reduction_pct = (memory_saved / initial_memory) * 100
            self._log_step(
                "memory_optimization",
                f"Reduced memory by {memory_saved:.2f} MB ({reduction_pct:.1f}% reduction)"
            )
            self.transformations_applied.append(f"Data Type Optimization (saved {memory_saved:.2f} MB)")
        
        return df
    
    def _auto_detect_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-detect and convert date/datetime columns."""
        
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{2}-\d{2}-\d{2}',  # YY-MM-DD or MM-DD-YY
        ]
        
        for col in df.select_dtypes(include=['object']).columns:
            # Skip if too many nulls
            if df[col].isnull().sum() / len(df) > 0.5:
                continue
            
            # Sample some values
            sample = df[col].dropna().head(100).astype(str)
            
            # Check if matches date patterns
            is_date = False
            for pattern in date_patterns:
                matches = sample.str.match(pattern).sum()
                if matches / len(sample) > 0.8:  # 80% match threshold
                    is_date = True
                    break
            
            if is_date:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    converted = df[col].notna().sum()
                    self._log_step(
                        "date_conversion",
                        f"Converted '{col}' to datetime ({converted} values)"
                    )
                    self.type_conversions.append({
                        "column": col,
                        "from_type": "object",
                        "to_type": "datetime64",
                        "values_converted": int(converted)
                    })
                except Exception as e:
                    self._log_step("date_conversion_error", f"Failed to convert '{col}': {str(e)}")
        
        return df
    
    def _optimize_categorical_columns(
        self,
        df: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """Convert low-cardinality object columns to category."""
        
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df[col])
            
            if unique_ratio < threshold:
                try:
                    unique_count = df[col].nunique()
                    df[col] = df[col].astype('category')
                    self._log_step(
                        "category_conversion",
                        f"Converted '{col}' to category ({unique_count} unique values, {unique_ratio:.1%} uniqueness)"
                    )
                    self.type_conversions.append({
                        "column": col,
                        "from_type": "object",
                        "to_type": "category",
                        "unique_count": int(unique_count),
                        "uniqueness_ratio": round(unique_ratio, 4)
                    })
                except Exception as e:
                    self._log_step("category_conversion_error", f"Failed to convert '{col}': {str(e)}")
        
        return df
    
    def _fix_type_mismatches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and fix columns that should be numeric but are stored as strings."""
        
        for col in df.select_dtypes(include=['object']).columns:
            # Sample values
            sample = df[col].dropna().head(1000)
            
            if len(sample) == 0:
                continue
            
            # Check if values are numeric-like
            try:
                # Remove common number separators
                cleaned = sample.astype(str).str.replace(',', '').str.replace('$', '').str.strip()
                numeric_values = pd.to_numeric(cleaned, errors='coerce')
                
                # If most values convert successfully, it's likely a numeric column
                conversion_rate = numeric_values.notna().sum() / len(cleaned)
                
                if conversion_rate > 0.9:  # 90% threshold
                    # Convert entire column
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.replace('$', '').str.strip(),
                        errors='coerce'
                    )
                    converted = df[col].notna().sum()
                    self._log_step(
                        "numeric_conversion",
                        f"Converted '{col}' from string to numeric ({converted} values, {conversion_rate:.1%} success rate)"
                    )
                    self.type_conversions.append({
                        "column": col,
                        "from_type": "object",
                        "to_type": "numeric",
                        "conversion_rate": round(conversion_rate, 4),
                        "values_converted": int(converted)
                    })
            except Exception as e:
                continue
        
        return df
    
    def _convert_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean-like columns to actual boolean type."""
        
        boolean_values = [
            {'true': ['true', 't', 'yes', 'y', '1', 'on'], 'false': ['false', 'f', 'no', 'n', '0', 'off']},
        ]
        
        for col in df.select_dtypes(include=['object']).columns:
            # Get unique values (case-insensitive)
            unique_vals = df[col].dropna().astype(str).str.lower().unique()
            
            if len(unique_vals) <= 2:
                # Check if matches boolean pattern
                for bool_map in boolean_values:
                    all_true = set(bool_map['true'])
                    all_false = set(bool_map['false'])
                    unique_set = set(unique_vals)
                    
                    if unique_set.issubset(all_true.union(all_false)):
                        # Convert to boolean
                        df[col] = df[col].astype(str).str.lower().map(
                            lambda x: True if x in all_true else (False if x in all_false else None)
                        )
                        converted = df[col].notna().sum()
                        self._log_step(
                            "boolean_conversion",
                            f"Converted '{col}' to boolean ({converted} values)"
                        )
                        self.type_conversions.append({
                            "column": col,
                            "from_type": "object",
                            "to_type": "bool",
                            "values_converted": int(converted)
                        })
                        break
        
        return df
    
    def _downcast_integers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast integer columns to smallest possible type."""
        
        for col in df.select_dtypes(include=['int']).columns:
            try:
                original_dtype = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], downcast='integer')
                new_dtype = str(df[col].dtype)
                
                if original_dtype != new_dtype:
                    self.type_conversions.append({
                        "column": col,
                        "from_type": original_dtype,
                        "to_type": new_dtype,
                        "operation": "downcast_integer"
                    })
            except Exception as e:
                continue
        
        return df
    
    def _downcast_floats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcast float columns to smallest possible type."""
        
        for col in df.select_dtypes(include=['float']).columns:
            try:
                original_dtype = str(df[col].dtype)
                df[col] = pd.to_numeric(df[col], downcast='float')
                new_dtype = str(df[col].dtype)
                
                if original_dtype != new_dtype:
                    self.type_conversions.append({
                        "column": col,
                        "from_type": original_dtype,
                        "to_type": new_dtype,
                        "operation": "downcast_float"
                    })
            except Exception as e:
                continue
        
        return df
    
    # ============================================================
    # PHASE 5: SCHEMA VALIDATION
    # ============================================================
    
    def _validate_schema(
        self,
        df: pd.DataFrame,
        schema: Dict[str, Any],
        enforce: bool = False
    ) -> Dict[str, Any]:
        """
        Validate DataFrame against schema definition.
        
        Schema format:
        {
            "required_columns": ["col1", "col2"],
            "column_types": {
                "col1": "int64",
                "col2": "object"
            },
            "column_ranges": {
                "age": {"min": 0, "max": 120},
                "price": {"min": 0}
            },
            "column_patterns": {
                "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            },
            "column_values": {
                "status": ["active", "inactive", "pending"]
            }
        }
        """
        logger.info("🔍 Validating schema")
        
        errors = []
        warnings = []
        violations = []
        
        # Check required columns
        if "required_columns" in schema:
            missing = set(schema["required_columns"]) - set(df.columns)
            if missing:
                error = f"Missing required columns: {list(missing)}"
                errors.append(error)
                violations.append({
                    "type": "missing_columns",
                    "columns": list(missing),
                    "severity": "error"
                })
        
        # Check column types
        if "column_types" in schema:
            for col, expected_type in schema["column_types"].items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if actual_type != expected_type:
                        warning = f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}"
                        warnings.append(warning)
                        violations.append({
                            "type": "type_mismatch",
                            "column": col,
                            "expected": expected_type,
                            "actual": actual_type,
                            "severity": "warning",
                            "auto_fixable": True
                        })
        
        # Check numeric ranges
        if "column_ranges" in schema:
            for col, range_def in schema["column_ranges"].items():
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    if "min" in range_def:
                        violations_count = (df[col] < range_def["min"]).sum()
                        if violations_count > 0:
                            warning = f"Column '{col}' has {violations_count} values below minimum {range_def['min']}"
                            warnings.append(warning)
                            violations.append({
                                "type": "range_violation",
                                "column": col,
                                "constraint": "min",
                                "expected": range_def["min"],
                                "violations_count": int(violations_count),
                                "severity": "warning",
                                "auto_fixable": True
                            })
                    
                    if "max" in range_def:
                        violations_count = (df[col] > range_def["max"]).sum()
                        if violations_count > 0:
                            warning = f"Column '{col}' has {violations_count} values above maximum {range_def['max']}"
                            warnings.append(warning)
                            violations.append({
                                "type": "range_violation",
                                "column": col,
                                "constraint": "max",
                                "expected": range_def["max"],
                                "violations_count": int(violations_count),
                                "severity": "warning",
                                "auto_fixable": True
                            })
        
        # Check string patterns (regex)
        if "column_patterns" in schema:
            for col, pattern in schema["column_patterns"].items():
                if col in df.columns and df[col].dtype == 'object':
                    regex = re.compile(pattern)
                    invalid = ~df[col].astype(str).str.match(regex)
                    invalid_count = invalid.sum()
                    
                    if invalid_count > 0:
                        warning = f"Column '{col}' has {invalid_count} values not matching pattern"
                        warnings.append(warning)
                        violations.append({
                            "type": "pattern_violation",
                            "column": col,
                            "pattern": pattern,
                            "violations_count": int(invalid_count),
                            "severity": "warning",
                            "auto_fixable": False
                        })
        
        # Check allowed values
        if "column_values" in schema:
            for col, allowed_values in schema["column_values"].items():
                if col in df.columns:
                    invalid = ~df[col].isin(allowed_values + [None, np.nan])
                    invalid_count = invalid.sum()
                    
                    if invalid_count > 0:
                        invalid_vals = df[col][invalid].unique()[:5]  # Show first 5
                        warning = f"Column '{col}' has {invalid_count} invalid values (e.g., {list(invalid_vals)})"
                        warnings.append(warning)
                        violations.append({
                            "type": "value_violation",
                            "column": col,
                            "allowed_values": allowed_values,
                            "violations_count": int(invalid_count),
                            "sample_invalid": [str(v) for v in invalid_vals],
                            "severity": "error",
                            "auto_fixable": False
                        })
        
        is_valid = len(errors) == 0
        
        self.schema_violations = violations
        
        if violations:
            self._log_step(
                "schema_validation",
                f"Found {len(violations)} schema violations: {len(errors)} errors, {len(warnings)} warnings"
            )
        else:
            self._log_step("schema_validation", "Schema validation passed ✅")
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "violations": violations
        }
    
    def _auto_fix_schema_violations(
        self,
        df: pd.DataFrame,
        violations: List[Dict[str, Any]],
        config: CleaningConfig
    ) -> pd.DataFrame:
        """Automatically fix fixable schema violations."""
        
        logger.info("🔧 Auto-fixing schema violations")
        
        for violation in violations:
            if not violation.get("auto_fixable", False):
                continue
            
            try:
                if violation["type"] == "type_mismatch":
                    col = violation["column"]
                    expected = violation["expected"]
                    
                    # Attempt type conversion
                    try:
                        df[col] = df[col].astype(expected)
                        self._log_step("schema_fix", f"Converted '{col}' to {expected}")
                    except Exception as e:
                        self._log_step("schema_fix_error", f"Failed to convert '{col}': {str(e)}")
                
                elif violation["type"] == "range_violation":
                    col = violation["column"]
                    constraint = violation["constraint"]
                    expected_val = violation["expected"]
                    
                    if constraint == "min":
                        df.loc[df[col] < expected_val, col] = expected_val
                    elif constraint == "max":
                        df.loc[df[col] > expected_val, col] = expected_val
                    
                    self._log_step(
                        "schema_fix",
                        f"Clipped '{col}' to {constraint} value {expected_val}"
                    )
            
            except Exception as e:
                self._log_step("schema_fix_error", f"Failed to fix violation: {str(e)}")
        
        return df

    # ============================================================
    # PHASE 4: DATABASE AUDIT LOG PERSISTENCE
    # ============================================================
    
    def _save_audit_log_to_database(
        self,
        dataset_id: int,
        config: CleaningConfig,
        original_shape: Tuple[int, int],
        final_shape: Tuple[int, int],
        original_file_path: str,
        cleaned_file_path: Optional[str],
        processing_time: float,
        status: str,
        error_message: Optional[str]
    ) -> None:
        """
        Save comprehensive audit log to database.
        
        Args:
            dataset_id: Dataset ID
            config: Cleaning configuration used
            original_shape: Original (rows, cols)
            final_shape: Final (rows, cols)
            original_file_path: Path to original file
            cleaned_file_path: Path to cleaned file
            processing_time: Time taken in seconds
            status: success, failed, partial
            error_message: Error message if failed
        """
        try:
            from app.models.dataset import CleaningAuditLog
            
            audit_log = CleaningAuditLog(
                dataset_id=dataset_id,
                session_id=self.session_id,
                
                # Timing
                started_at=self.start_time,
                completed_at=self.end_time,
                duration_seconds=processing_time,
                status=status,
                
                # Configuration
                config_used=config.dict(),
                
                # Changes
                original_shape={"rows": original_shape[0], "columns": original_shape[1]},
                final_shape={"rows": final_shape[0], "columns": final_shape[1]},
                rows_removed=original_shape[0] - final_shape[0],
                columns_removed=original_shape[1] - final_shape[1],
                
                # Transformations
                transformations_applied=self.transformations_applied,
                cleaning_steps=self.cleaning_log,
                
                # Quality
                quality_metrics=self.quality_metrics,
                quality_alerts=self.quality_alerts,
                
                # Files
                original_file_path=original_file_path,
                cleaned_file_path=cleaned_file_path,
                
                # Error
                error_message=error_message,
                
                # User
                user_id=config.user_id
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
            logger.info(f"💾 Saved audit log to database (Session: {self.session_id})")
            self._log_step("audit_save", f"Audit log saved to database with session ID: {self.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save audit log to database: {str(e)}", exc_info=True)
            self.db.rollback()
    
    # ============================================================
    # PHASE 4: EXPORT AUDIT LOG TO FILE
    # ============================================================
    
    def _export_audit_log_to_file(self, dataset_id: int) -> str:
        """
        Export audit log to JSON file for compliance and traceability.
        
        Returns:
            Path to exported audit log file
        """
        try:
            # Create audit logs directory
            audit_dir = Path(settings.UPLOAD_DIR) / "audit_logs"
            audit_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audit_log_dataset_{dataset_id}_{self.session_id}_{timestamp}.json"
            file_path = audit_dir / filename
            
            # Prepare audit data
            audit_data = {
                "session_id": self.session_id,
                "dataset_id": dataset_id,
                "started_at": self.start_time.isoformat() if self.start_time else None,
                "completed_at": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else None,
                
                "transformations_applied": self.transformations_applied,
                "cleaning_log": self.cleaning_log,
                "quality_metrics": self.quality_metrics,
                "quality_alerts": self.quality_alerts,
                
                "metadata": {
                    "export_timestamp": datetime.now(timezone.utc).isoformat(),
                    "version": "1.0",
                    "format": "json"
                }
            }
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 Exported audit log to {file_path}")
            self._log_step("audit_export", f"Audit log exported to file: {file_path}")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to export audit log to file: {str(e)}", exc_info=True)
            return ""
    
    # ============================================================
    # ENHANCED LOGGING WITH CONTEXT
    # ============================================================
    
    def _log_step(self, step: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Enhanced logging with metadata and context.
        
        Args:
            step: Step name/category
            message: Log message
            metadata: Optional additional metadata
        """
        entry = {
            "session_id": self.session_id,
            "step": step,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }
        
        self.cleaning_log.append(entry)
        logger.info(f"[{self.session_id}][{step}] {message}")
    
    # ============================================================
    # RETRIEVE AUDIT HISTORY
    # ============================================================
    
    def get_cleaning_history(
        self,
        dataset_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve cleaning history for a dataset from audit logs.
        
        Args:
            dataset_id: Dataset ID
            limit: Maximum number of records to return
            
        Returns:
            List of audit log entries
        """
        try:
            from app.models.dataset import CleaningAuditLog
            
            audit_logs = self.db.query(CleaningAuditLog).filter(
                CleaningAuditLog.dataset_id == dataset_id
            ).order_by(
                CleaningAuditLog.started_at.desc()
            ).limit(limit).all()
            
            history = []
            for log in audit_logs:
                history.append({
                    "session_id": log.session_id,
                    "started_at": log.started_at.isoformat() if log.started_at else None,
                    "completed_at": log.completed_at.isoformat() if log.completed_at else None,
                    "duration_seconds": log.duration_seconds,
                    "status": log.status,
                    "rows_removed": log.rows_removed,
                    "columns_removed": log.columns_removed,
                    "quality_score": log.quality_metrics.get("overall_score") if log.quality_metrics else None,
                    "transformations_count": len(log.transformations_applied) if log.transformations_applied else 0,
                    "error_message": log.error_message
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to retrieve cleaning history: {str(e)}")
            return []
    
    def get_audit_log_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed audit log for a specific session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Complete audit log details
        """
        try:
            from app.models.dataset import CleaningAuditLog
            
            log = self.db.query(CleaningAuditLog).filter(
                CleaningAuditLog.session_id == session_id
            ).first()
            
            if not log:
                return None
            
            return {
                "session_id": log.session_id,
                "dataset_id": log.dataset_id,
                "started_at": log.started_at.isoformat() if log.started_at else None,
                "completed_at": log.completed_at.isoformat() if log.completed_at else None,
                "duration_seconds": log.duration_seconds,
                "status": log.status,
                "config_used": log.config_used,
                "original_shape": log.original_shape,
                "final_shape": log.final_shape,
                "rows_removed": log.rows_removed,
                "columns_removed": log.columns_removed,
                "transformations_applied": log.transformations_applied,
                "cleaning_steps": log.cleaning_steps,
                "quality_metrics": log.quality_metrics,
                "quality_alerts": log.quality_alerts,
                "original_file_path": log.original_file_path,
                "cleaned_file_path": log.cleaned_file_path,
                "error_message": log.error_message,
                "user_id": log.user_id
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve audit log details: {str(e)}")
            return None

    # ============================================================
    # PHASE 3: DATA QUALITY METRICS CALCULATION
    # ============================================================
    
    def _calculate_quality_metrics(
        self,
        df: pd.DataFrame,
        original_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality metrics across 6 dimensions.
        
        Quality Dimensions:
        1. Completeness - % of non-missing data
        2. Uniqueness - Average uniqueness ratio across columns
        3. Consistency - No duplicates, uniform formats
        4. Validity - Data conforms to expected types and ranges
        5. Accuracy - Data correctness (placeholder for domain logic)
        6. Timeliness - Data recency (placeholder for temporal analysis)
        
        Returns:
            Dictionary with quality scores and detailed metrics
        """
        logger.info("📊 Calculating comprehensive quality metrics")
        
        total_cells = df.shape[0] * df.shape[1]
        missing_count = df.isnull().sum().sum()
        
        # ============================================================
        # 1. COMPLETENESS (30% weight)
        # ============================================================
        completeness = (1 - (missing_count / total_cells)) * 100 if total_cells > 0 else 100
        
        completeness_details = {
            "total_cells": int(total_cells),
            "missing_cells": int(missing_count),
            "filled_cells": int(total_cells - missing_count),
            "percentage": round(completeness, 2)
        }
        
        # ============================================================
        # 2. UNIQUENESS (20% weight)
        # ============================================================
        uniqueness_ratios = []
        column_uniqueness = {}
        
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            
            if total_count > 0:
                ratio = unique_count / total_count
                uniqueness_ratios.append(ratio)
                column_uniqueness[col] = {
                    "unique_values": int(unique_count),
                    "total_values": int(total_count),
                    "uniqueness_ratio": round(ratio, 4)
                }
        
        uniqueness = np.mean(uniqueness_ratios) * 100 if uniqueness_ratios else 100
        
        uniqueness_details = {
            "average_uniqueness": round(uniqueness, 2),
            "by_column": column_uniqueness
        }
        
        # ============================================================
        # 3. CONSISTENCY (20% weight)
        # ============================================================
        duplicate_count = df.duplicated().sum()
        consistency = (1 - (duplicate_count / len(df))) * 100 if len(df) > 0 else 100
        
        # Check data type consistency
        mixed_type_columns = []
        for col in df.select_dtypes(include=['object']).columns:
            types = df[col].apply(type).nunique()
            if types > 1:
                mixed_type_columns.append(col)
        
        consistency_details = {
            "percentage": round(consistency, 2),
            "duplicate_rows": int(duplicate_count),
            "mixed_type_columns": mixed_type_columns,
            "mixed_type_count": len(mixed_type_columns)
        }
        
        # ============================================================
        # 4. VALIDITY (10% weight)
        # ============================================================
        # Check for valid data types and ranges
        invalid_values = 0
        validity_issues = []
        
        for col in df.select_dtypes(include=[np.number]).columns:
            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                invalid_values += inf_count
                validity_issues.append({
                    "column": col,
                    "issue": "infinite_values",
                    "count": int(inf_count)
                })
            
            # Check for negative values where inappropriate (can be configured)
            # This is a placeholder - customize based on domain knowledge
        
        total_values = df.select_dtypes(include=[np.number]).size
        validity = (1 - (invalid_values / total_values)) * 100 if total_values > 0 else 100
        
        validity_details = {
            "percentage": round(validity, 2),
            "invalid_values": int(invalid_values),
            "issues": validity_issues
        }
        
        # ============================================================
        # 5. ACCURACY (10% weight)
        # ============================================================
        # Placeholder for domain-specific accuracy checks
        # Can include range checks, format validation, business rules
        accuracy = 100.0  # Default assumption
        
        accuracy_details = {
            "percentage": round(accuracy, 2),
            "note": "Placeholder - implement domain-specific validation rules"
        }
        
        # ============================================================
        # 6. TIMELINESS (10% weight)
        # ============================================================
        # Placeholder for temporal data recency checks
        timeliness = 100.0  # Default assumption
        
        timeliness_details = {
            "percentage": round(timeliness, 2),
            "note": "Placeholder - implement temporal recency validation"
        }
        
        # ============================================================
        # OVERALL QUALITY SCORE (Weighted Average)
        # ============================================================
        weights = {
            "completeness": 0.30,
            "uniqueness": 0.20,
            "consistency": 0.20,
            "validity": 0.10,
            "accuracy": 0.10,
            "timeliness": 0.10
        }
        
        overall_score = (
            completeness * weights["completeness"] +
            uniqueness * weights["uniqueness"] +
            consistency * weights["consistency"] +
            validity * weights["validity"] +
            accuracy * weights["accuracy"] +
            timeliness * weights["timeliness"]
        )
        
        quality_grade = self._get_quality_grade(overall_score)
        
        # ============================================================
        # COMPILE METRICS
        # ============================================================
        metrics = {
            # Summary scores
            "overall_score": round(overall_score, 2),
            "quality_grade": quality_grade,
            
            # Individual dimension scores
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "consistency": round(consistency, 2),
            "validity": round(validity, 2),
            "accuracy": round(accuracy, 2),
            "timeliness": round(timeliness, 2),
            
            # Detailed breakdowns
            "details": {
                "completeness": completeness_details,
                "uniqueness": uniqueness_details,
                "consistency": consistency_details,
                "validity": validity_details,
                "accuracy": accuracy_details,
                "timeliness": timeliness_details
            },
            
            # Weights used
            "weights": weights,
            
            # Timestamp
            "calculated_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._log_step(
            "quality_metrics",
            f"Quality Assessment: {overall_score:.2f}% (Grade: {quality_grade}) - "
            f"Completeness: {completeness:.1f}%, Uniqueness: {uniqueness:.1f}%, "
            f"Consistency: {consistency:.1f}%"
        )
        
        return metrics
    
    # ============================================================
    # QUALITY GRADING SYSTEM
    # ============================================================
    
    def _get_quality_grade(self, score: float) -> str:
        """
        Convert quality score to letter grade.
        
        Grading scale:
        - A+: 95-100%
        - A:  90-94%
        - B+: 85-89%
        - B:  80-84%
        - C+: 75-79%
        - C:  70-74%
        - D:  60-69%
        - F:  <60%
        """
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    # ============================================================
    # QUALITY ALERTS & MONITORING
    # ============================================================
    
    def _check_quality_alerts(
        self,
        metrics: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Check quality metrics against thresholds and generate alerts.
        
        Args:
            metrics: Quality metrics dictionary
            thresholds: Optional custom thresholds (defaults provided)
            
        Returns:
            List of alert messages for metrics below thresholds
        """
        # Default thresholds
        default_thresholds = {
            "overall_score": 85.0,
            "completeness": 95.0,
            "uniqueness": 80.0,
            "consistency": 90.0,
            "validity": 95.0,
            "accuracy": 90.0,
            "timeliness": 90.0
        }
        
        # Use custom thresholds if provided
        thresholds = thresholds or default_thresholds
        
        alerts = []
        
        for metric, threshold in thresholds.items():
            value = metrics.get(metric, 100)
            
            if value < threshold:
                severity = self._get_alert_severity(value, threshold)
                
                alert = {
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "severity": severity,
                    "message": f"⚠️ {severity.upper()}: {metric.replace('_', ' ').title()} is {value:.2f}% (threshold: {threshold}%)",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                alerts.append(alert)
                
                self._log_step(
                    "quality_alert",
                    f"{severity.upper()} - {metric}: {value:.2f}% < {threshold}%"
                )
        
        if not alerts:
            self._log_step("quality_check", "✅ All quality metrics meet thresholds")
        else:
            self._log_step("quality_check", f"⚠️ {len(alerts)} quality alerts generated")
        
        return alerts
    
    def _get_alert_severity(self, value: float, threshold: float) -> str:
        """Determine alert severity based on how far below threshold."""
        gap = threshold - value
        
        if gap >= 20:
            return "critical"
        elif gap >= 10:
            return "high"
        elif gap >= 5:
            return "medium"
        else:
            return "low"
    
    # ============================================================
    # QUALITY REPORT GENERATION
    # ============================================================
    
    def _generate_quality_report(
        self,
        metrics: Dict[str, Any],
        original_shape: Tuple[int, int],
        final_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Returns:
            Structured report with quality insights and recommendations
        """
        logger.info("📋 Generating quality report")
        
        report = {
            "title": "Data Quality Assessment Report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            
            # Executive summary
            "summary": {
                "overall_score": metrics["overall_score"],
                "quality_grade": metrics["quality_grade"],
                "status": self._get_quality_status(metrics["overall_score"]),
                "original_rows": original_shape[0],
                "final_rows": final_shape[0],
                "data_loss_percent": round(
                    ((original_shape[0] - final_shape[0]) / original_shape[0] * 100), 2
                ) if original_shape[0] > 0 else 0
            },
            
            # Dimension scores
            "dimensions": {
                "completeness": {
                    "score": metrics["completeness"],
                    "status": self._get_dimension_status(metrics["completeness"], 95),
                    "details": metrics["details"]["completeness"]
                },
                "uniqueness": {
                    "score": metrics["uniqueness"],
                    "status": self._get_dimension_status(metrics["uniqueness"], 80),
                    "details": metrics["details"]["uniqueness"]
                },
                "consistency": {
                    "score": metrics["consistency"],
                    "status": self._get_dimension_status(metrics["consistency"], 90),
                    "details": metrics["details"]["consistency"]
                },
                "validity": {
                    "score": metrics["validity"],
                    "status": self._get_dimension_status(metrics["validity"], 95),
                    "details": metrics["details"]["validity"]
                },
                "accuracy": {
                    "score": metrics["accuracy"],
                    "status": self._get_dimension_status(metrics["accuracy"], 90),
                    "details": metrics["details"]["accuracy"]
                },
                "timeliness": {
                    "score": metrics["timeliness"],
                    "status": self._get_dimension_status(metrics["timeliness"], 90),
                    "details": metrics["details"]["timeliness"]
                }
            },
            
            # Recommendations
            "recommendations": self._generate_quality_recommendations(metrics),
            
            # Alerts
            "alerts": self.quality_alerts
        }
        
        self._log_step("quality_report", "Generated comprehensive quality report")
        
        return report
    
    def _get_quality_status(self, score: float) -> str:
        """Get overall quality status."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "acceptable"
        elif score >= 60:
            return "needs_improvement"
        else:
            return "poor"
    
    def _get_dimension_status(self, score: float, threshold: float) -> str:
        """Get dimension-specific status."""
        if score >= threshold:
            return "pass"
        elif score >= threshold - 10:
            return "warning"
        else:
            return "fail"
    
    def _generate_quality_recommendations(
        self,
        metrics: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on quality metrics."""
        recommendations = []
        
        # Completeness recommendations
        if metrics["completeness"] < 95:
            missing_pct = 100 - metrics["completeness"]
            recommendations.append({
                "dimension": "completeness",
                "priority": "high" if missing_pct > 10 else "medium",
                "recommendation": f"Address {missing_pct:.1f}% missing data through additional imputation or data collection"
            })
        
        # Uniqueness recommendations
        if metrics["uniqueness"] < 80:
            recommendations.append({
                "dimension": "uniqueness",
                "priority": "medium",
                "recommendation": "Review columns with low uniqueness - consider removing redundant or constant features"
            })
        
        # Consistency recommendations
        if metrics["consistency"] < 90:
            dup_count = metrics["details"]["consistency"]["duplicate_rows"]
            if dup_count > 0:
                recommendations.append({
                    "dimension": "consistency",
                    "priority": "high",
                    "recommendation": f"Remove or investigate {dup_count} duplicate rows"
                })
            
            mixed_count = metrics["details"]["consistency"]["mixed_type_count"]
            if mixed_count > 0:
                recommendations.append({
                    "dimension": "consistency",
                    "priority": "high",
                    "recommendation": f"Standardize data types in {mixed_count} columns with mixed types"
                })
        
        # Validity recommendations
        if metrics["validity"] < 95:
            issues = metrics["details"]["validity"]["issues"]
            if issues:
                recommendations.append({
                    "dimension": "validity",
                    "priority": "high",
                    "recommendation": f"Fix {len(issues)} validity issues including infinite values and invalid ranges"
                })
        
        # General recommendation if score is low
        if metrics["overall_score"] < 70:
            recommendations.append({
                "dimension": "overall",
                "priority": "critical",
                "recommendation": "Data quality is below acceptable standards - consider re-running cleaning with more aggressive parameters"
            })
        
        return recommendations

    # ============================================================
    # PHASE 2: ADVANCED OUTLIER DETECTION
    # ============================================================
    
    def _detect_outliers(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> Dict[str, pd.Series]:
        """
        Detect outliers using multiple advanced methods.
        
        Methods:
        - IQR (Interquartile Range) - univariate
        - Z-Score - univariate
        - Isolation Forest - multivariate ML-based
        - DBSCAN - density-based clustering
        
        Returns:
            Dictionary mapping column names to boolean masks (True = outlier)
        """
        outlier_masks = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self._log_step("outlier_detection", "No numerical columns for outlier detection")
            return outlier_masks
        
        method = config.outlier_method
        threshold = config.outlier_threshold
        
        logger.info(f"🔍 Detecting outliers using method: {method}")
        
        if method == "iqr":
            total_outliers = 0
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_masks[col] = mask
                
                if mask.sum() > 0:
                    outlier_count = mask.sum()
                    total_outliers += outlier_count
                    self._log_step(
                        "outlier_iqr",
                        f"IQR detected {outlier_count} outliers in '{col}' (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                    )
            
            if total_outliers > 0:
                self.transformations_applied.append(f"IQR Outlier Detection ({total_outliers} total)")
        
        elif method == "zscore":
            total_outliers = 0
            for col in numerical_cols:
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                mask = z_scores > threshold
                outlier_masks[col] = mask
                
                if mask.sum() > 0:
                    outlier_count = mask.sum()
                    total_outliers += outlier_count
                    self._log_step(
                        "outlier_zscore",
                        f"Z-Score detected {outlier_count} outliers in '{col}' (threshold: {threshold})"
                    )
            
            if total_outliers > 0:
                self.transformations_applied.append(f"Z-Score Outlier Detection ({total_outliers} total)")
        
        elif method == "isolation_forest":
            try:
                # Prepare data - handle NaN
                data_clean = df[numerical_cols].fillna(df[numerical_cols].median())
                
                # Fit Isolation Forest
                iso_forest = IsolationForest(
                    contamination=config.isolation_contamination,
                    random_state=42,
                    n_estimators=100,
                    n_jobs=-1
                )
                
                predictions = iso_forest.fit_predict(data_clean)
                
                # -1 indicates outlier
                global_mask = pd.Series(predictions == -1, index=df.index)
                
                # Apply same mask to all numerical columns
                for col in numerical_cols:
                    outlier_masks[col] = global_mask
                
                outlier_count = global_mask.sum()
                self._log_step(
                    "outlier_isolation_forest",
                    f"Isolation Forest detected {outlier_count} multivariate outliers (contamination={config.isolation_contamination})"
                )
                
                if outlier_count > 0:
                    self.transformations_applied.append(f"Isolation Forest Detection ({outlier_count} outliers)")
                
            except Exception as e:
                self._log_step("outlier_error", f"Isolation Forest failed: {str(e)}")
                logger.error(f"Isolation Forest error: {str(e)}", exc_info=True)
        
        elif method == "dbscan":
            try:
                # Prepare data
                data_clean = df[numerical_cols].fillna(df[numerical_cols].median())
                
                # Fit DBSCAN
                clustering = DBSCAN(
                    eps=config.dbscan_eps,
                    min_samples=config.dbscan_min_samples,
                    n_jobs=-1
                ).fit(data_clean)
                
                # -1 label indicates outlier/noise
                global_mask = pd.Series(clustering.labels_ == -1, index=df.index)
                
                # Apply same mask to all numerical columns
                for col in numerical_cols:
                    outlier_masks[col] = global_mask
                
                outlier_count = global_mask.sum()
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                
                self._log_step(
                    "outlier_dbscan",
                    f"DBSCAN detected {outlier_count} density-based outliers ({n_clusters} clusters found, eps={config.dbscan_eps})"
                )
                
                if outlier_count > 0:
                    self.transformations_applied.append(f"DBSCAN Outlier Detection ({outlier_count} outliers)")
                
            except Exception as e:
                self._log_step("outlier_error", f"DBSCAN failed: {str(e)}")
                logger.error(f"DBSCAN error: {str(e)}", exc_info=True)
        
        else:
            logger.warning(f"⚠️ Unknown outlier detection method: {method}")
            self._log_step("outlier_detection", f"Unknown method '{method}', skipping outlier detection")
        
        return outlier_masks
    
    # ============================================================
    # PHASE 2: OUTLIER TREATMENT
    # ============================================================
    
    def _treat_outliers(
        self,
        df: pd.DataFrame,
        outlier_masks: Dict[str, pd.Series],
        config: CleaningConfig
    ) -> pd.DataFrame:
        """
        Treat detected outliers using specified method.
        
        Treatment methods:
        - cap: Cap at non-outlier min/max
        - remove: Remove outlier rows
        - winsorize: Winsorize distribution (5th/95th percentiles)
        - impute: Replace with median or custom values
        - none: Leave outliers unchanged
        
        Returns:
            DataFrame with outliers treated
        """
        if not outlier_masks:
            return df
        
        treatment = config.outlier_treatment
        treated_df = df.copy()
        total_treated = 0
        
        logger.info(f"🛠️ Treating outliers using method: {treatment}")
        
        for col, mask in outlier_masks.items():
            if mask.sum() == 0:
                continue
            
            outlier_count = mask.sum()
            
            try:
                if treatment == "cap":
                    # Cap at non-outlier min/max
                    non_outliers = treated_df.loc[~mask, col]
                    lower_cap = non_outliers.min()
                    upper_cap = non_outliers.max()
                    
                    treated_df.loc[mask, col] = np.clip(
                        treated_df.loc[mask, col],
                        lower_cap,
                        upper_cap
                    )
                    
                    self._log_step(
                        "outlier_cap",
                        f"Capped {outlier_count} outliers in '{col}' to range [{lower_cap:.2f}, {upper_cap:.2f}]"
                    )
                    total_treated += outlier_count
                
                elif treatment == "remove":
                    # Remove rows with outliers
                    before_rows = len(treated_df)
                    treated_df = treated_df.loc[~mask]
                    removed = before_rows - len(treated_df)
                    
                    self._log_step(
                        "outlier_remove",
                        f"Removed {removed} rows with outliers in '{col}'"
                    )
                    total_treated += removed
                
                elif treatment == "winsorize":
                    # Winsorize (cap at 5th and 95th percentiles)
                    winsorized = winsorize(treated_df[col], limits=[0.05, 0.05])
                    treated_df[col] = winsorized
                    
                    self._log_step(
                        "outlier_winsorize",
                        f"Winsorized '{col}' at 5th/95th percentiles ({outlier_count} outliers affected)"
                    )
                    total_treated += outlier_count
                
                elif treatment == "impute":
                    # Impute with custom value or median
                    if config.outlier_impute_values and col in config.outlier_impute_values:
                        impute_val = config.outlier_impute_values[col]
                    else:
                        impute_val = treated_df.loc[~mask, col].median()
                    
                    treated_df.loc[mask, col] = impute_val
                    
                    self._log_step(
                        "outlier_impute",
                        f"Imputed {outlier_count} outliers in '{col}' with value: {impute_val:.2f}"
                    )
                    total_treated += outlier_count
                
                elif treatment == "none":
                    self._log_step(
                        "outlier_none",
                        f"Left {outlier_count} outliers unchanged in '{col}'"
                    )
                
                else:
                    logger.warning(f"⚠️ Unknown outlier treatment: {treatment}")
                    self._log_step("outlier_treatment", f"Unknown treatment '{treatment}' for '{col}'")
            
            except Exception as e:
                self._log_step("outlier_treatment_error", f"Failed treating outliers in '{col}': {str(e)}")
                logger.error(f"Outlier treatment error for '{col}': {str(e)}", exc_info=True)
        
        if total_treated > 0:
            self.transformations_applied.append(f"Outlier Treatment - {treatment.title()} ({total_treated} affected)")
            self._log_step("outlier_summary", f"Total outliers treated: {total_treated}")
        
        return treated_df

    # ============================================================
    # ADVANCED MISSING VALUE HANDLING
    # ============================================================
    
    def _handle_missing_values_advanced(
        self,
        df: pd.DataFrame,
        config: CleaningConfig
    ) -> pd.DataFrame:
        """
        Advanced missing value handling with multiple strategies.
        
        Supports:
        - median, mean, mode (statistical)
        - forward, backward (propagation)
        - KNN (K-Nearest Neighbors)
        - MICE (Multiple Imputation by Chained Equations)
        - drop (remove rows)
        - custom fill values
        """
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            self._log_step("missing_values", "No missing values detected")
            return df
        
        logger.info(f"📊 Handling {initial_missing:,} missing values")
        
        # Step 1: Apply custom fill values first
        if config.fill_values:
            for col, val in config.fill_values.items():
                if col in df.columns and df[col].isnull().any():
                    before = df[col].isnull().sum()
                    df[col].fillna(val, inplace=True)
                    self._log_step("custom_fill", f"Filled {before} missing values in '{col}' with custom value: {val}")
        
        # Step 2: Collect columns for batch imputation
        knn_cols = []
        mice_cols = []
        
        # Step 3: Per-column strategies
        for col in df.columns:
            if not df[col].isnull().any():
                continue
            
            # Determine strategy
            strategy = config.missing_strategy
            if config.per_column_missing and col in config.per_column_missing:
                strategy = config.per_column_missing[col]
            
            try:
                if strategy == "drop":
                    before = len(df)
                    df = df[df[col].notnull()]
                    removed = before - len(df)
                    self._log_step("drop_missing", f"Dropped {removed} rows with missing values in '{col}'")
                
                elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                    fill_val = df[col].median()
                    count = df[col].isnull().sum()
                    df[col].fillna(fill_val, inplace=True)
                    self._log_step("median_impute", f"Filled {count} missing values in '{col}' with median: {fill_val:.2f}")
                
                elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                    fill_val = df[col].mean()
                    count = df[col].isnull().sum()
                    df[col].fillna(fill_val, inplace=True)
                    self._log_step("mean_impute", f"Filled {count} missing values in '{col}' with mean: {fill_val:.2f}")
                
                elif strategy == "mode":
                    if not df[col].mode().empty:
                        fill_val = df[col].mode()[0]
                        count = df[col].isnull().sum()
                        df[col].fillna(fill_val, inplace=True)
                        self._log_step("mode_impute", f"Filled {count} missing values in '{col}' with mode: {fill_val}")
                    else:
                        df[col].fillna('Unknown', inplace=True)
                        self._log_step("mode_impute", f"Filled '{col}' with 'Unknown' (no mode found)")
                
                elif strategy == "forward":
                    count = df[col].isnull().sum()
                    df[col].fillna(method='ffill', inplace=True)
                    self._log_step("forward_fill", f"Forward filled {count} missing values in '{col}'")
                
                elif strategy == "backward":
                    count = df[col].isnull().sum()
                    df[col].fillna(method='bfill', inplace=True)
                    self._log_step("backward_fill", f"Backward filled {count} missing values in '{col}'")
                
                elif strategy == "knn":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        knn_cols.append(col)
                
                elif strategy == "mice":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        mice_cols.append(col)
                
            except Exception as e:
                self._log_step("imputation_error", f"Error imputing '{col}' with strategy '{strategy}': {str(e)}")
        
        # Step 4: Batch KNN Imputation
        if knn_cols:
            try:
                imputer = KNNImputer(n_neighbors=config.knn_neighbors)
                df[knn_cols] = imputer.fit_transform(df[knn_cols])
                self._log_step("knn_impute", f"Applied KNN imputation (k={config.knn_neighbors}) to {len(knn_cols)} columns: {knn_cols}")
                self.transformations_applied.append(f"KNN Imputation ({len(knn_cols)} columns)")
            except Exception as e:
                self._log_step("knn_error", f"KNN imputation failed: {str(e)}")
        
        # Step 5: Batch MICE Imputation
        if mice_cols:
            try:
                imputer = IterativeImputer(max_iter=config.mice_max_iter, random_state=42)
                df[mice_cols] = imputer.fit_transform(df[mice_cols])
                self._log_step("mice_impute", f"Applied MICE imputation (max_iter={config.mice_max_iter}) to {len(mice_cols)} columns: {mice_cols}")
                self.transformations_applied.append(f"MICE Imputation ({len(mice_cols)} columns)")
            except Exception as e:
                self._log_step("mice_error", f"MICE imputation failed: {str(e)}")
        
        # Summary
        remaining_missing = df.isnull().sum().sum()
        handled = initial_missing - remaining_missing
        self._log_step("missing_summary", f"Successfully handled {handled:,} missing values. Remaining: {remaining_missing:,}")
        
        return df
    
    # ============================================================
    # DUPLICATE REMOVAL
    # ============================================================
    
    def _remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame
            subset: Optional list of columns to check for duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        duplicates_count = df.duplicated(subset=subset).sum()
        
        if duplicates_count > 0:
            df = df.drop_duplicates(subset=subset, keep='first')
            removed = initial_rows - len(df)
            self._log_step("remove_duplicates", f"Removed {removed:,} duplicate rows (keeping first occurrence)")
            self.transformations_applied.append(f"Duplicate Removal ({removed:,} rows)")
        else:
            self._log_step("remove_duplicates", "No duplicate rows found")
        
        return df
    
    # ============================================================
    # COLUMN DROPPING
    # ============================================================
    
    def _drop_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Drop specified columns.
        
        Args:
            df: DataFrame
            columns: List of column names to drop
            
        Returns:
            DataFrame with columns removed
        """
        existing_cols = [c for c in columns if c in df.columns]
        missing_cols = [c for c in columns if c not in df.columns]
        
        if missing_cols:
            logger.warning(f"⚠️ Columns not found (skipped): {missing_cols}")
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            self._log_step("drop_columns", f"Dropped {len(existing_cols)} columns: {existing_cols}")
            self.transformations_applied.append(f"Column Removal ({len(existing_cols)} columns)")
        
        return df
    
    # ============================================================
    # HELPER METHODS
    # ============================================================
    
    def _read_dataframe(self, file_path: str, file_type: str) -> pd.DataFrame:
        """Read file into DataFrame with error handling."""
        try:
            if file_type == ".csv":
                return pd.read_csv(file_path)
            elif file_type in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            elif file_type == ".json":
                return pd.read_json(file_path)
            elif file_type == ".parquet":
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to read file: {str(e)}"
            ) from e
    
    def _save_cleaned_data(
        self,
        df: pd.DataFrame,
        dataset: Dataset
    ) -> str:
        """Save cleaned DataFrame to file."""
        original_path = Path(dataset.file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cleaned_path = original_path.parent / f"{original_path.stem}_cleaned_{timestamp}{original_path.suffix}"
        
        try:
            if dataset.file_type == ".csv":
                df.to_csv(cleaned_path, index=False)
            elif dataset.file_type in [".xlsx", ".xls"]:
                df.to_excel(cleaned_path, index=False)
            elif dataset.file_type == ".json":
                df.to_json(cleaned_path, orient='records', indent=2)
            elif dataset.file_type == ".parquet":
                df.to_parquet(cleaned_path, index=False)
            
            logger.info(f"💾 Saved cleaned data to {cleaned_path}")
            return str(cleaned_path)
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save cleaned file: {str(e)}"
            ) from e
    
    def _log_step(self, step: str, message: str) -> None:
        """Log cleaning step for audit trail."""
        entry = {
            "step": step,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.cleaning_log.append(entry)
        logger.info(f"[{step}] {message}")

    def clean_dataset(self, dataset_id: int, config: Any) -> Dict[str, Any]:
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"🧹 Starting cleaning for dataset {dataset_id} (Session: {self.session_id})")

        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            logger.error(f"Dataset {dataset_id} not found in the database.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )

        original_file_path = dataset.file_path
        dataset.status = DatasetStatus.CLEANING
        self.db.commit()

        try:
            logger.debug(f"Loading dataset file from: {original_file_path} with type: {dataset.file_type}")

            # load dataframe - support for config-aware and fallback
            try:
                df = self._read_dataframe(dataset.file_path, dataset.file_type, config)
                logger.debug("Dataframe loaded using config-aware _read_dataframe.")
            except TypeError as te:
                logger.warning(f"Config parameter not accepted by _read_dataframe: {te}. Falling back to simple loading.")
                if dataset.file_type == ".csv":
                    df = pd.read_csv(dataset.file_path)
                else:
                    df = pd.read_excel(dataset.file_path)

            logger.info(f"Dataset loaded, shape: {df.shape}")
            original_shape = df.shape
            original_memory = df.memory_usage(deep=True).sum() / 1024**2
            logger.debug(f"Original dataset memory usage: {original_memory:.2f} MB")

            self._log_step("initialization",
                           f"Session {self.session_id}: Loaded dataset {original_shape[0]:,} rows × {original_shape[1]} columns, {original_memory:.2f} MB")

            self.cleaning_log = []
            self.transformations_applied = []
            self.quality_metrics = {}
            self.quality_alerts = []

            remove_duplicates = getattr(config, 'remove_duplicates', True)
            columns_to_drop = getattr(config, 'columns_to_drop', None)
            handle_missing = getattr(config, 'handle_missing', None)
            missing_strategy = getattr(config, 'missing_strategy', 'median')

            logger.debug(f"Cleaning config: remove_duplicates={remove_duplicates}, columns_to_drop={columns_to_drop}, "
                         f"handle_missing={handle_missing}, missing_strategy={missing_strategy}")

            if remove_duplicates:
                initial_rows = len(df)
                df = df.drop_duplicates()
                removed = initial_rows - len(df)
                logger.info(f"Removed {removed} duplicate rows.")
                if removed > 0:
                    self._log_step("remove_duplicates", f"Removed {removed} duplicate rows")
                    self.transformations_applied.append(f"Removed {removed} duplicates")

            if columns_to_drop:
                logger.info(f"Dropping columns: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop, errors='ignore')
                self._log_step("drop_columns", f"Dropped {len(columns_to_drop)} columns")

            if handle_missing or missing_strategy:
                strategy = handle_missing or missing_strategy
                logger.info(f"Handling missing values using strategy: {strategy}")

                if strategy == "drop":
                    initial_rows = len(df)
                    df = df.dropna()
                    removed = initial_rows - len(df)
                    logger.info(f"Dropped {removed} rows due to missing values.")
                    if removed > 0:
                        self._log_step("handle_missing", f"Dropped {removed} rows with missing values")
                        self.transformations_applied.append(f"Dropped {removed} rows with nulls")
                elif strategy in ["fill", "median", "mean", "mode"]:
                    for col in df.columns:
                        if df[col].isnull().any():
                            if pd.api.types.is_numeric_dtype(df[col]):
                                fill_value = df[col].median() if strategy in ["fill", "median"] else df[col].mean()
                                logger.debug(f"Filling missing values in numeric column '{col}' with value {fill_value}.")
                                df[col].fillna(fill_value, inplace=True)
                            else:
                                if not df[col].mode().empty:
                                    fill_value = df[col].mode()[0]
                                    logger.debug(f"Filling missing values in categorical column '{col}' with value '{fill_value}'.")
                                    df[col].fillna(fill_value, inplace=True)

                    self._log_step("handle_missing", f"Filled missing values using {strategy}")
                    self.transformations_applied.append(f"Imputed missing values ({strategy})")

            self.quality_metrics = {
                "overall_score": 85.0,  # Example fixed score
                "quality_grade": "B",
                "completeness": round((1 - df.isnull().sum().sum() / df.size) * 100, 2),
                "uniqueness": round((1 - df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 100,
                "dimensions": {
                    "completeness": {"score": round((1 - df.isnull().sum().sum() / df.size) * 100, 2)},
                    "uniqueness": {"score": round((1 - df.duplicated().sum() / len(df)) * 100, 2) if len(df) > 0 else 100}
                }
            }
            logger.info(f"Quality metrics calculated: {self.quality_metrics}")

            cleaned_dir = Path(settings.UPLOAD_DIR) / "cleaned"
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cleaned_{dataset.id}_{timestamp}.csv"
            cleaned_path = str(cleaned_dir / filename)
            logger.info(f"Saving cleaned data to: {cleaned_path}")

            df.to_csv(cleaned_path, index=False)

            dataset.file_path = cleaned_path
            dataset.row_count = len(df)
            dataset.column_count = len(df.columns)
            dataset.status = DatasetStatus.COMPLETED
            dataset.missing_values_count = int(df.isnull().sum().sum())
            dataset.duplicate_rows_count = 0
            dataset.data_quality_score = self.quality_metrics.get("overall_score", 0)

            self.db.commit()

            self.end_time = datetime.now(timezone.utc)
            processing_time = (self.end_time - self.start_time).total_seconds()
            final_memory = df.memory_usage(deep=True).sum() / 1024**2

            logger.info(f"✅ Cleaning completed for dataset {dataset_id} in {processing_time:.2f}s")

            result = {
                "success": True,
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "session_id": self.session_id,
                "processing_time_seconds": round(processing_time, 2),
                "original_shape": original_shape,
                "cleaned_shape": df.shape,
                "rows_removed": original_shape[0] - df.shape[0],
                "columns_removed": original_shape[1] - df.shape[1],
                "memory_mb_before": round(original_memory, 2),
                "memory_mb_after": round(final_memory, 2),
                "quality_metrics": self.quality_metrics,
                "quality_alerts": self.quality_alerts,
                "transformations_applied": self.transformations_applied,
                "cleaning_log": self.cleaning_log,
                "cleaned_file_path": cleaned_path
            }
            logger.debug(f"Final cleaning result prepared: {result}")
            return result

        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            dataset.processing_error = f"Cleaning failed: {str(e)}"
            self.db.commit()
            logger.error(f"❌ Cleaning failed for dataset {dataset_id}: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data cleaning failed: {str(e)}"
            ) from e
# ============================================================
# DEPENDENCY INJECTION
# ============================================================

def get_cleaning_service(db: Session = Depends(get_db)) -> DataCleaningService:
    """Dependency for DataCleaningService."""
    return DataCleaningService(db)
# ============================================================
# CELERY INTEGRATION (OPTIONAL)
# ============================================================

# Configure Celery for distributed task queue
celery_app = Celery(
    'data_cleaning',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task(name='clean_dataset_celery')
def clean_dataset_celery_task(dataset_id: int, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task for distributed cleaning."""
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        service = DataCleaningService(db)
        config = CleaningConfig(**config_dict)
        result = service.clean_dataset_robust(dataset_id, config)
        return result
    finally:
        db.close()


# ============================================================
# EXAMPLE USAGE PATTERNS
# ============================================================

"""
# Example 1: Simple cleaning with preset
service = DataCleaningService(db)
config = DataCleaningService.get_preset_config(CleaningPreset.ML_READY)
result = service.clean_dataset_robust(dataset_id=1, config=config)

# Example 2: Fluent API pipeline builder
config = (
    service.pipeline()
    .use_preset(CleaningPreset.STANDARD)
    .handle_missing("knn", knn_neighbors=5)
    .detect_outliers("isolation_forest", treatment="cap")
    .scale_features("standard")
    .encode_categorical("onehot")
    .enable_parallel(max_workers=4)
    .with_quality_thresholds(completeness=95, consistency=90)
    .build()
)
result = service.clean_dataset_robust(dataset_id=1, config=config)

# Example 3: Event hooks
def on_completion(data):
    print(f"Cleaning completed: {data}")

service.on_event(CleaningEvent.COMPLETED, on_completion)
result = service.clean_dataset_robust(dataset_id=1, config=config)

# Example 4: Scheduled cleaning
scheduler = DataCleaningService.create_scheduled_cleaner(
    dataset_id=1,
    config=config,
    schedule="0 2 * * *",  # Daily at 2 AM
    db_session_factory=SessionLocal
)
scheduler.start()

# Example 5: Batch processing
dataset_ids = [1, 2, 3, 4, 5]
results = service.clean_multiple_datasets(dataset_ids, config, parallel=True)

# Example 6: Async with webhooks
config.enable_webhooks = True
config.webhook_url = "https://example.com/webhook"
config.run_async = True

await service.clean_dataset_async(dataset_id=1, config=config, background_tasks=background_tasks)
"""
