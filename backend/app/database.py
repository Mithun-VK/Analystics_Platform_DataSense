"""
Database Configuration and Session Management.

Handles SQLAlchemy database engine creation, session management,
connection pooling, and database initialization for the application.
"""

import logging
from typing import Generator, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event, Engine, pool, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.pool import QueuePool, NullPool

from app.core.config import settings


logger = logging.getLogger(__name__)


# ============================================================
# ENGINE CONFIGURATION
# ============================================================
# ⛔ SAFETY BLOCK: Never allow SQLite in production
if settings.is_production and settings.DATABASE_URL.startswith("sqlite"):
    raise RuntimeError(
        "❌ FATAL: SQLite detected in production! "
        "Set DATABASE_URL to your Render PostgreSQL database."
    )

def get_engine_args() -> dict:
    """
    Get SQLAlchemy engine arguments based on environment.
    
    Returns:
        Dictionary of engine configuration arguments
    """
    engine_args = {
        "echo": settings.DB_ECHO,  # Log SQL queries
        "future": True,  # Use SQLAlchemy 2.0 style
        "pool_pre_ping": True,  # Enable connection health checks
    }
    
    # Production-specific configurations
    if settings.is_production:
        engine_args.update({
            "poolclass": QueuePool,
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pool_recycle": settings.DB_POOL_RECYCLE,
            "pool_pre_ping": True,
            "echo_pool": False,
        })
    else:
        # Development configuration
        engine_args.update({
            "poolclass": QueuePool,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 3600,
        })
    
    # SQLite-specific configuration
    if settings.DATABASE_URL.startswith("sqlite"):
        engine_args.update({
            "connect_args": {"check_same_thread": False},
            "poolclass": NullPool,  # SQLite doesn't need pooling
        })
        # Remove pool-related args for SQLite
        for key in ["pool_size", "max_overflow", "pool_timeout", "pool_recycle"]:
            engine_args.pop(key, None)
    
    return engine_args


# Create database engine
engine: Engine = create_engine(
    settings.DATABASE_URL,
    **get_engine_args()
)


# ============================================================
# SESSION CONFIGURATION
# ============================================================

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # Keep objects usable after commit
    class_=Session,
)


# ============================================================
# DATABASE SESSION DEPENDENCY
# ============================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.
    
    Creates a new database session for each request and ensures
    proper cleanup after the request completes.
    
    Yields:
        Database session
        
    Example:
        @app.get("/items/")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Use this for background tasks or non-request contexts.
    
    Yields:
        Database session
        
    Example:
        with get_db_context() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        logger.error(f"Database context error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db() -> None:
    """
    Initialize database by creating all tables.
    
    This should only be used for development/testing.
    For production, use Alembic migrations.
    
    Example:
        from app.database import init_db
        init_db()
    """
    from app.models.base import Base
    
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise


def drop_db() -> None:
    """
    Drop all database tables.
    
    WARNING: This will delete all data!
    Only use for development/testing.
    
    Example:
        from app.database import drop_db
        drop_db()
    """
    from app.models.base import Base
    
    if settings.is_production:
        raise RuntimeError("Cannot drop database in production!")
    
    try:
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {str(e)}")
        raise


def reset_db() -> None:
    """
    Reset database by dropping and recreating all tables.
    
    WARNING: This will delete all data!
    Only use for development/testing.
    
    Example:
        from app.database import reset_db
        reset_db()
    """
    if settings.is_production:
        raise RuntimeError("Cannot reset database in production!")
    
    drop_db()
    init_db()
    logger.info("Database reset complete")


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def check_db_connection() -> bool:
    """
    Check if database connection is healthy.
    
    Returns:
        True if connection is healthy, False otherwise
        
    Example:
        from app.database import check_db_connection
        if check_db_connection():
            print("Database is ready")
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))  # SQLAlchemy 2.0: Use text()
        logger.info("Database connection is healthy")
        return True
    except OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking database: {str(e)}")
        return False


def get_db_info() -> dict:
    """
    Get database connection information.
    
    Returns:
        Dictionary with database details
        
    Example:
        from app.database import get_db_info
        info = get_db_info()
        print(f"Database: {info['database']}")
    """
    url = engine.url
    
    return {
        "database": url.database,
        "drivername": url.drivername,
        "host": url.host,
        "port": url.port,
        "pool_size": engine.pool.size() if hasattr(engine.pool, 'size') else None,
        "pool_checked_out": engine.pool.checked_out() if hasattr(engine.pool, 'checked_out') else None,
        "echo": engine.echo,
        "is_production": settings.is_production,
    }


# ============================================================
# DATABASE EVENT LISTENERS
# ============================================================

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Set SQLite-specific pragmas for better performance.
    
    Only runs for SQLite databases.
    """
    if settings.DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        # Enable foreign key support
        cursor.execute("PRAGMA foreign_keys=ON")
        # Set journal mode for better concurrency
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
        logger.debug("SQLite pragmas set")


@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log when a connection is checked out from the pool."""
    if settings.DB_ECHO:
        logger.debug("Connection checked out from pool")


@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log when a connection is returned to the pool."""
    if settings.DB_ECHO:
        logger.debug("Connection returned to pool")


@event.listens_for(SessionLocal, "before_commit")
def receive_before_commit(session):
    """Hook to run before session commit."""
    if settings.DB_ECHO:
        logger.debug("Session about to commit")


@event.listens_for(SessionLocal, "after_commit")
def receive_after_commit(session):
    """Hook to run after session commit."""
    if settings.DB_ECHO:
        logger.debug("Session committed successfully")


@event.listens_for(SessionLocal, "after_rollback")
def receive_after_rollback(session):
    """Hook to run after session rollback."""
    logger.warning("Session rolled back")


# ============================================================
# CONNECTION POOL MONITORING
# ============================================================

def get_pool_status() -> Optional[dict]:
    """
    Get current connection pool status (SQLAlchemy 2.x safe).
    """
    pool = engine.pool

    try:
        return {
            "pool_size": getattr(pool, "size", lambda: None)(),
            "checked_out": getattr(pool, "checkedout", lambda: None)(),
            "overflow": getattr(pool, "_overflow", None),
        }
    except Exception:
        return None


# ============================================================
# TRANSACTION HELPERS
# ============================================================

@contextmanager
def transaction_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional scope for database operations.
    
    Automatically commits on success or rolls back on failure.
    
    Yields:
        Database session
        
    Example:
        from app.database import transaction_scope
        
        with transaction_scope() as db:
            user = User(email="test@example.com")
            db.add(user)
            # Auto-commits on exit if no exception
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Transaction failed: {str(e)}")
        raise
    finally:
        session.close()


# ============================================================
# STARTUP/SHUTDOWN HANDLERS
# ============================================================

def startup_db() -> None:
    """
    Database startup handler.
    
    Call this when application starts to initialize database
    and verify connection.
    
    Example:
        from app.database import startup_db
        
        @app.on_event("startup")
        async def on_startup():
            startup_db()
    """
    logger.info("Initializing database connection...")
    
    # Check connection
    if not check_db_connection():
        raise RuntimeError("Failed to connect to database")
    
    # Log database info
    db_info = get_db_info()
    logger.info(
    f"✅ Database connected | "
    f"Driver: {db_info['drivername']} | "
    f"Host: {db_info['host']} | "
    f"DB: {db_info['database']}"
)

    # Log pool status if available
    pool_status = get_pool_status()
    if pool_status:
        logger.info(f"Connection pool: size={pool_status['pool_size']}")
    
    logger.info("Database initialized successfully")


def shutdown_db() -> None:
    """
    Database shutdown handler.
    
    Call this when application shuts down to close connections.
    
    Example:
        from app.database import shutdown_db
        
        @app.on_event("shutdown")
        async def on_shutdown():
            shutdown_db()
    """
    logger.info("Closing database connections...")
    
    # Log pool status before closing
    pool_status = get_pool_status()
    if pool_status:
        logger.info(
            f"Pool status before shutdown - "
            f"checked_out={pool_status['checked_out']}, "
            f"overflow={pool_status['overflow']}"
        )
    
    # Dispose of engine
    engine.dispose()
    logger.info("Database connections closed")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def execute_raw_sql(sql: str, params: Optional[dict] = None) -> list:
    """
    Execute raw SQL query.
    
    Use with caution. Prefer ORM queries when possible.
    
    Args:
        sql: SQL query string
        params: Optional query parameters
        
    Returns:
        List of result rows
        
    Example:
        from app.database import execute_raw_sql
        results = execute_raw_sql("SELECT * FROM users WHERE id = :id", {"id": 1})
    """
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})  # SQLAlchemy 2.0: Use text()
        return result.fetchall()


def get_table_names() -> list[str]:
    """
    Get list of all table names in database.
    
    Returns:
        List of table names
        
    Example:
        from app.database import get_table_names
        tables = get_table_names()
        print(f"Tables: {', '.join(tables)}")
    """
    inspector = inspect(engine)
    return inspector.get_table_names()


# ============================================================
# EXPORT
# ============================================================

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "get_db_context",
    "init_db",
    "drop_db",
    "reset_db",
    "check_db_connection",
    "get_db_info",
    "get_pool_status",
    "transaction_scope",
    "startup_db",
    "shutdown_db",
    "execute_raw_sql",
    "get_table_names",
]
