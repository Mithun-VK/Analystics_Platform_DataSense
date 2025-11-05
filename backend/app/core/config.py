# backend/app/core/config.py - COMPLETE WORKING VERSION

"""
Application configuration and settings management.
"""

import secrets
from typing import Any, Optional
from pydantic import (
    EmailStr,
    field_validator,
    ValidationInfo,
    ConfigDict,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        protected_namespaces=(),
        json_schema_extra={
            "BACKEND_CORS_ORIGINS": {
                "description": "Comma-separated CORS origins",
            }
        }
    )
    
    # ============================================================
    # APPLICATION SETTINGS
    # ============================================================
    PROJECT_NAME: str = "DataAnalytics API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    RELOAD: bool = True
    
    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    # ============================================================
    # SECURITY SETTINGS
    # ============================================================
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGITS: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # ============================================================
    # DATABASE SETTINGS
    # ============================================================
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "datainsight"
    
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 3600
    DB_ECHO: bool = False
    
    DATABASE_URL: Optional[str] = None
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info: ValidationInfo) -> str:
        if isinstance(v, str) and v:
            return v
        
        user = info.data.get("POSTGRES_USER", "postgres")
        password = info.data.get("POSTGRES_PASSWORD", "postgres")
        server = info.data.get("POSTGRES_SERVER", "localhost")
        port = info.data.get("POSTGRES_PORT", 5432)
        db = info.data.get("POSTGRES_DB", "datainsight")
        
        return f"postgresql+psycopg2://{user}:{password}@{server}:{port}/{db}"
    
    # ============================================================
    # REDIS SETTINGS
    # ============================================================
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: Optional[str] = None
    
    @field_validator("REDIS_URL", mode="before")
    @classmethod
    def assemble_redis_connection(cls, v: Optional[str], info: ValidationInfo) -> str:
        if isinstance(v, str) and v:
            return v
        
        password = info.data.get("REDIS_PASSWORD")
        auth = f":{password}@" if password else ""
        host = info.data.get("REDIS_HOST", "localhost")
        port = info.data.get("REDIS_PORT", 6379)
        db = info.data.get("REDIS_DB", 0)
        
        return f"redis://{auth}{host}:{port}/{db}"
    
    CACHE_ENABLED: bool = True
    CACHE_DEFAULT_TIMEOUT: int = 300
    
    # ============================================================
    # ✅ CORS SETTINGS - FIXED: Use STRING instead of List[str]
    # ============================================================
    BACKEND_CORS_ORIGINS_STR: str = "http://localhost:5174,http://127.0.0.1:5174,http://192.168.1.7:5174"
    
    @field_validator("BACKEND_CORS_ORIGINS_STR", mode="before")
    @classmethod
    def parse_cors_origins_str(cls, v: Any) -> str:
        """Accept string or return as-is."""
        if isinstance(v, str):
            return v.strip()
        return str(v) if v else ""
    
    @property
    def BACKEND_CORS_ORIGINS(self) -> list:
        """Convert string to list of origins."""
        if not self.BACKEND_CORS_ORIGINS_STR:
            return []
        return [origin.strip() for origin in self.BACKEND_CORS_ORIGINS_STR.split(",")]
    
    # ============================================================
    # FILE UPLOAD SETTINGS
    # ============================================================
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024
    ALLOWED_EXTENSIONS: str = ".csv,.xlsx,.xls,.json,.parquet"
    
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None
    USE_S3: bool = False
    
    # ============================================================
    # EMAIL SETTINGS
    # ============================================================
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    SMTP_PORT: int = 587
    SMTP_HOST: Optional[str] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAILS_FROM_EMAIL: Optional[EmailStr] = None
    EMAILS_FROM_NAME: Optional[str] = None
    EMAIL_ENABLED: bool = False
    
    EMAIL_RESET_TOKEN_EXPIRE_HOURS: int = 24
    EMAIL_VERIFICATION_TOKEN_EXPIRE_HOURS: int = 48
    
    # ============================================================
    # CELERY SETTINGS
    # ============================================================
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_ALWAYS_EAGER: bool = True
    CELERY_TASK_TIME_LIMIT: int = 3600
    CELERY_TASK_SOFT_TIME_LIMIT: int = 3000
    
    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def set_celery_broker(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v:
            return v
        return info.data.get("REDIS_URL", "redis://localhost:6379/0")
    
    @field_validator("CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def set_celery_backend(cls, v: Optional[str], info: ValidationInfo) -> str:
        if v:
            return v
        return info.data.get("REDIS_URL", "redis://localhost:6379/0")
    
    # ============================================================
    # THIRD-PARTY API KEYS
    # ============================================================
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    RAZORPAY_KEY_ID: Optional[str] = None
    RAZORPAY_KEY_SECRET: Optional[str] = None
    RAZORPAY_WEBHOOK_SECRET: Optional[str] = None
    PAYMENT_ENABLED: bool = False
    
    # ============================================================
    # MONITORING & LOGGING
    # ============================================================
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "text"
    LOG_FILE: Optional[str] = None
    
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENABLED: bool = False
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1
    
    # ============================================================
    # FEATURE FLAGS
    # ============================================================
    FEATURE_AI_INSIGHTS: bool = True
    FEATURE_AUTOML: bool = False
    FEATURE_COLLABORATION: bool = False
    FEATURE_ADVANCED_VISUALIZATIONS: bool = True
    
    # ============================================================
    # BUSINESS LOGIC
    # ============================================================
    FREE_TIER_DATASET_LIMIT: int = 5
    FREE_TIER_FILE_SIZE_LIMIT: int = 10 * 1024 * 1024
    PREMIUM_TIER_DATASET_LIMIT: int = 100
    PREMIUM_TIER_FILE_SIZE_LIMIT: int = 500 * 1024 * 1024
    
    MAX_ROWS_FOR_PREVIEW: int = 1000
    EDA_GENERATION_TIMEOUT: int = 300
    AUTOML_TRAINING_TIMEOUT: int = 3600
    
    MAX_PAGE_SIZE: int = 100
    DEFAULT_PAGE_SIZE: int = 20
    
    # ============================================================
    # SUPERUSER SETTINGS
    # ============================================================
    FIRST_SUPERUSER_EMAIL: EmailStr = "admin@dataanalytics.com"
    FIRST_SUPERUSER_PASSWORD: str = "changeme123"
    
    # ============================================================
    # COMPUTED PROPERTIES
    # ============================================================
    
    @property
    def cors_config(self) -> dict[str, Any]:
        """✅ CORS middleware configuration."""
        return {
            "allow_origins": self.BACKEND_CORS_ORIGINS,  # Uses the property above
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
            "expose_headers": ["X-Total-Count", "X-Page-Count"],
            "max_age": 3600,
        }


# ============================================================
# CREATE GLOBAL SETTINGS INSTANCE
# ============================================================

settings = Settings()
