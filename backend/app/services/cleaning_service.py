"""
Data Cleaning Service.

Handles data preprocessing including missing value imputation,
duplicate removal, outlier detection, data type conversion,
and standardization with comprehensive logging and validation.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from fastapi.params import Depends
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import Dataset, DatasetStatus
from app.schemas.dataset import DatasetCleaningConfig
from app.database import get_db


logger = logging.getLogger(__name__)


class DataCleaningService:
    """
    Data cleaning and preprocessing service.
    
    Implements:
    - Missing value handling (multiple strategies)
    - Duplicate detection and removal
    - Outlier detection and treatment
    - Data type conversion and validation
    - Feature scaling and normalization
    - Data quality reporting
    - Audit trail for transformations
    """
    
    def __init__(self, db: Session):
        """
        Initialize data cleaning service.
        
        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.cleaning_log: List[Dict[str, Any]] = []
    
    # ============================================================
    # MAIN CLEANING PIPELINE
    # ============================================================
    
    def clean_dataset(
        self,
        dataset_id: int,
        config: DatasetCleaningConfig
    ) -> Dict[str, Any]:
        """
        Execute complete data cleaning pipeline.
        
        Args:
            dataset_id: Dataset ID to clean
            config: Cleaning configuration
            
        Returns:
            Dictionary with cleaning results and statistics
            
        Raises:
            HTTPException: If dataset not found or cleaning fails
        """
        # Get dataset
        dataset = self.db.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Dataset not found",
            )
        
        # Update status
        dataset.status = DatasetStatus.CLEANING
        self.db.commit()
        
        try:
            # Load data
            df = self._read_dataframe(dataset.file_path, dataset.file_type)
            original_shape = df.shape
            
            logger.info(f"Starting cleaning for dataset {dataset_id}. Shape: {original_shape}")
            
            # Initialize cleaning log
            self.cleaning_log = []
            self._log_step("initial_load", f"Loaded dataset with shape {original_shape}")
            
            # Step 1: Handle duplicates
            if config.remove_duplicates:
                df = self._remove_duplicates(df)
            
            # Step 2: Drop specified columns
            if config.columns_to_drop:
                df = self._drop_columns(df, config.columns_to_drop)
            
            # Step 3: Handle missing values
            df = self._handle_missing_values(df, config.handle_missing, config.fill_values)
            
            # Step 4: Detect and handle outliers
            if config.outlier_detection:
                df = self._handle_outliers(df)
            
            # Step 5: Convert data types
            df = self._optimize_data_types(df)
            
            # Step 6: Validate cleaned data
            validation_results = self._validate_cleaned_data(df, original_shape)
            
            # Save cleaned data
            cleaned_path = self._save_cleaned_data(df, dataset)
            
            # Update dataset record
            dataset.file_path = cleaned_path
            dataset.row_count = len(df)
            dataset.column_count = len(df.columns)
            dataset.status = DatasetStatus.COMPLETED
            
            # Update quality metrics
            dataset.missing_values_count = int(df.isnull().sum().sum())
            dataset.duplicate_rows_count = 0  # After cleaning
            
            # Recalculate quality score
            total_cells = dataset.row_count * dataset.column_count
            if total_cells > 0:
                missing_ratio = dataset.missing_values_count / total_cells
                quality_score = 100 * (1 - missing_ratio)
                dataset.data_quality_score = round(quality_score, 2)
            
            self.db.commit()
            
            # Prepare results
            results = {
                "dataset_id": dataset_id,
                "original_shape": original_shape,
                "cleaned_shape": df.shape,
                "rows_removed": original_shape[0] - df.shape[0],
                "columns_removed": original_shape[1] - df.shape[1],
                "cleaning_log": self.cleaning_log,
                "validation": validation_results,
                "data_quality_score": dataset.data_quality_score,
            }
            
            logger.info(f"Cleaning completed for dataset {dataset_id}. New shape: {df.shape}")
            
            return results
            
        except Exception as e:
            dataset.status = DatasetStatus.FAILED
            dataset.processing_error = f"Cleaning failed: {str(e)}"
            self.db.commit()
            
            logger.error(f"Cleaning failed for dataset {dataset_id}: {str(e)}")
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Data cleaning failed: {str(e)}",
            ) from e
    
    # ============================================================
    # DUPLICATE HANDLING
    # ============================================================
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        initial_rows = len(df)
        
        # Find duplicates
        duplicates = df.duplicated()
        num_duplicates = duplicates.sum()
        
        if num_duplicates > 0:
            # Remove duplicates, keeping first occurrence
            df = df.drop_duplicates(keep='first')
            
            rows_removed = initial_rows - len(df)
            self._log_step(
                "remove_duplicates",
                f"Removed {rows_removed} duplicate rows ({num_duplicates} found)"
            )
            
            logger.info(f"Removed {rows_removed} duplicate rows")
        else:
            self._log_step("remove_duplicates", "No duplicate rows found")
        
        return df
    
    # ============================================================
    # COLUMN MANAGEMENT
    # ============================================================
    
    def _drop_columns(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Drop specified columns from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of column names to drop
            
        Returns:
            DataFrame with columns removed
        """
        # Filter to existing columns only
        existing_cols = [col for col in columns if col in df.columns]
        missing_cols = [col for col in columns if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Columns not found: {missing_cols}")
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            self._log_step(
                "drop_columns",
                f"Dropped {len(existing_cols)} columns: {existing_cols}"
            )
        
        return df
    
    # ============================================================
    # MISSING VALUE HANDLING
    # ============================================================
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str,
        fill_values: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values using specified strategy.
        
        Args:
            df: Input DataFrame
            strategy: Strategy to use ('drop', 'fill', 'forward', 'backward')
            fill_values: Optional dict of column-specific fill values
            
        Returns:
            DataFrame with missing values handled
        """
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            self._log_step("handle_missing", "No missing values found")
            return df
        
        logger.info(f"Handling {initial_missing} missing values with strategy: {strategy}")
        
        if strategy == "drop":
            df = self._drop_missing_values(df)
        
        elif strategy == "fill":
            df = self._fill_missing_values(df, fill_values)
        
        elif strategy == "forward":
            df = df.fillna(method='ffill')
            self._log_step("handle_missing", "Applied forward fill for missing values")
        
        elif strategy == "backward":
            df = df.fillna(method='bfill')
            self._log_step("handle_missing", "Applied backward fill for missing values")
        
        else:
            logger.warning(f"Unknown strategy '{strategy}', using default 'drop'")
            df = self._drop_missing_values(df)
        
        remaining_missing = df.isnull().sum().sum()
        handled = initial_missing - remaining_missing
        
        self._log_step(
            "missing_summary",
            f"Handled {handled} missing values. Remaining: {remaining_missing}"
        )
        
        return df
    
    def _drop_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing value rows removed
        """
        initial_rows = len(df)
        df = df.dropna()
        rows_removed = initial_rows - len(df)
        
        self._log_step(
            "drop_missing",
            f"Dropped {rows_removed} rows containing missing values"
        )
        
        return df
    
    def _fill_missing_values(
        self,
        df: pd.DataFrame,
        fill_values: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fill missing values using smart imputation strategies.
        
        Args:
            df: Input DataFrame
            fill_values: Optional column-specific fill values
            
        Returns:
            DataFrame with missing values filled
        """
        # Apply custom fill values first
        if fill_values:
            for col, value in fill_values.items():
                if col in df.columns and df[col].isnull().any():
                    df[col] = df[col].fillna(value)
                    self._log_step(
                        "custom_fill",
                        f"Filled missing values in '{col}' with {value}"
                    )
        
        # Smart imputation for remaining missing values
        for col in df.columns:
            if df[col].isnull().any():
                df = self._impute_column(df, col)
        
        return df
    
    def _impute_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Impute missing values in a single column based on data type.
        
        Args:
            df: Input DataFrame
            col: Column name
            
        Returns:
            DataFrame with column imputed
        """
        missing_count = df[col].isnull().sum()
        
        # Numerical columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # Use median for numerical data (more robust to outliers)
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            self._log_step(
                "impute_numerical",
                f"Filled {missing_count} missing values in '{col}' with median: {fill_value:.2f}"
            )
        
        # Categorical columns
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Use mode (most frequent value) for categorical data
            if not df[col].mode().empty:
                fill_value = df[col].mode()[0]
                df[col] = df[col].fillna(fill_value)
                self._log_step(
                    "impute_categorical",
                    f"Filled {missing_count} missing values in '{col}' with mode: {fill_value}"
                )
            else:
                # If no mode, use 'Unknown'
                df[col] = df[col].fillna('Unknown')
                self._log_step(
                    "impute_categorical",
                    f"Filled {missing_count} missing values in '{col}' with 'Unknown'"
                )
        
        # DateTime columns
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Forward fill for time series data
            df[col] = df[col].fillna(method='ffill')
            self._log_step(
                "impute_datetime",
                f"Filled {missing_count} missing values in '{col}' with forward fill"
            )
        
        return df
    
    # ============================================================
    # OUTLIER DETECTION & HANDLING
    # ============================================================
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical columns.
        
        Args:
            df: Input DataFrame
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self._log_step("outlier_detection", "No numerical columns for outlier detection")
            return df
        
        total_outliers = 0
        
        for col in numerical_cols:
            if method == "iqr":
                outlier_mask = self._detect_outliers_iqr(df[col], threshold)
            else:  # zscore
                outlier_mask = self._detect_outliers_zscore(df[col], threshold)
            
            num_outliers = outlier_mask.sum()
            
            if num_outliers > 0:
                # Cap outliers at boundary values instead of removing
                if method == "iqr":
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                else:  # zscore
                    mean = df[col].mean()
                    std = df[col].std()
                    lower_bound = mean - threshold * std
                    upper_bound = mean + threshold * std
                    
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
                total_outliers += num_outliers
                self._log_step(
                    "outlier_capping",
                    f"Capped {num_outliers} outliers in '{col}' using {method} method"
                )
        
        if total_outliers > 0:
            self._log_step(
                "outlier_summary",
                f"Total outliers handled: {total_outliers} across {len(numerical_cols)} columns"
            )
        else:
            self._log_step("outlier_detection", "No outliers detected")
        
        return df
    
    def _detect_outliers_iqr(
        self,
        series: pd.Series,
        threshold: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers using IQR (Interquartile Range) method.
        
        Args:
            series: Pandas Series
            threshold: IQR multiplier (typically 1.5)
            
        Returns:
            Boolean mask of outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(
        self,
        series: pd.Series,
        threshold: float = 3.0
    ) -> pd.Series:
        """
        Detect outliers using Z-score method.
        
        Args:
            series: Pandas Series
            threshold: Z-score threshold (typically 3.0)
            
        Returns:
            Boolean mask of outliers
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    # ============================================================
    # DATA TYPE OPTIMIZATION
    # ============================================================
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types to reduce memory usage.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with optimized data types
        """
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Convert object columns to category if cardinality is low
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            # If less than 50% unique values, convert to category
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
                self._log_step(
                    "optimize_dtype",
                    f"Converted '{col}' to category type ({num_unique} unique values)"
                )
        
        # Downcast numerical columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        final_memory = df.memory_usage(deep=True).sum() / 1024**2  # MB
        memory_saved = initial_memory - final_memory
        
        if memory_saved > 0:
            self._log_step(
                "memory_optimization",
                f"Reduced memory usage by {memory_saved:.2f} MB "
                f"({(memory_saved/initial_memory)*100:.1f}%)"
            )
        
        return df
    
    # ============================================================
    # DATA VALIDATION
    # ============================================================
    
    def _validate_cleaned_data(
        self,
        df: pd.DataFrame,
        original_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """
        Validate cleaned data quality.
        
        Args:
            df: Cleaned DataFrame
            original_shape: Original DataFrame shape
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            "is_valid": True,
            "warnings": [],
            "statistics": {},
        }
        
        # Check if too many rows were removed
        rows_removed_pct = ((original_shape[0] - len(df)) / original_shape[0]) * 100
        if rows_removed_pct > 50:
            validation["warnings"].append(
                f"Warning: {rows_removed_pct:.1f}% of rows were removed during cleaning"
            )
        
        # Check for remaining missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            validation["warnings"].append(
                f"Warning: {missing_count} missing values remain after cleaning"
            )
        
        # Calculate statistics
        validation["statistics"] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": int(missing_count),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "numerical_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns),
        }
        
        return validation
    
    # ============================================================
    # FILE OPERATIONS
    # ============================================================
    
    def _read_dataframe(self, file_path: str, file_type: str) -> pd.DataFrame:
        """
        Read file into pandas DataFrame.
        
        Args:
            file_path: Path to file
            file_type: File extension
            
        Returns:
            DataFrame
        """
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
    
    def _save_cleaned_data(
        self,
        df: pd.DataFrame,
        dataset: Dataset
    ) -> str:
        """
        Save cleaned DataFrame to file.
        
        Args:
            df: Cleaned DataFrame
            dataset: Dataset instance
            
        Returns:
            Path to saved file
        """
        # Generate cleaned file path
        original_path = dataset.file_path
        name, ext = original_path.rsplit('.', 1)
        cleaned_path = f"{name}_cleaned.{ext}"
        
        # Save based on file type
        if dataset.file_type == ".csv":
            df.to_csv(cleaned_path, index=False)
        elif dataset.file_type in [".xlsx", ".xls"]:
            df.to_excel(cleaned_path, index=False)
        elif dataset.file_type == ".json":
            df.to_json(cleaned_path, orient='records')
        elif dataset.file_type == ".parquet":
            df.to_parquet(cleaned_path, index=False)
        
        logger.info(f"Saved cleaned data to {cleaned_path}")
        
        return cleaned_path
    
    # ============================================================
    # LOGGING
    # ============================================================
    
    def _log_step(self, step_name: str, message: str) -> None:
        """
        Log a cleaning step for audit trail.
        
        Args:
            step_name: Name of the step
            message: Log message
        """
        log_entry = {
            "step": step_name,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.cleaning_log.append(log_entry)


# ============================================================
# DEPENDENCY INJECTION HELPER
# ============================================================

def get_cleaning_service(db: Session = Depends(get_db)) -> DataCleaningService:
    """
    Dependency for injecting DataCleaningService.
    
    Args:
        db: Database session
        
    Returns:
        DataCleaningService instance
    """
    return DataCleaningService(db)
