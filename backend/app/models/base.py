"""
Base models and mixins for SQLAlchemy models.
Provides common functionality shared across all models.
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Boolean, func, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, declared_attr


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    Uses SQLAlchemy 2.0+ declarative style.
    """
    
    # Generate __tablename__ automatically from class name
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name (snake_case)."""
        import re
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
        # Add plural 's' if not already present
        return name if name.endswith('s') else f"{name}s"
    
    def dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary."""
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }
    
    def update(self, **kwargs) -> None:
        """Update model attributes from kwargs."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.dict().items()
            if k not in ["created_at", "updated_at"]
        )
        return f"{self.__class__.__name__}({params})"


class TimestampMixin:
    """
    Mixin that adds timestamp fields to models.
    Automatically tracks creation and update times.
    """
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Timestamp when record was created"
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Timestamp when record was last updated"
    )


class SoftDeleteMixin:
    """
    Mixin that adds soft delete functionality.
    Records are marked as deleted instead of being removed from database.
    """
    
    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
        index=True,
        comment="Soft delete flag"
    )
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when record was soft deleted"
    )
    
    def soft_delete(self) -> None:
        """Mark record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self) -> None:
        """Restore soft deleted record."""
        self.is_deleted = False
        self.deleted_at = None
