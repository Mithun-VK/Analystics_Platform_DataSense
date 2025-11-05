# backend/app/main.py - COMPLETE FIXED VERSION (OPTIONS IN MAIN.PY)

"""
FastAPI Application Entry Point.

Main application file for the DataAnalytics API.
Configures FastAPI app, middleware, routes, and lifecycle events.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.database import startup_db, shutdown_db, check_db_connection
from app.schemas.response import ErrorResponse

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# LIFESPAN CONTEXT MANAGER
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"   Environment: {settings.ENVIRONMENT}")
    
    try:
        startup_db()
        
        if not check_db_connection():
            logger.error("❌ Database health check failed")
            raise RuntimeError("Database connection failed")
        
        logger.info(f"   Debug mode: {settings.DEBUG}")
        logger.info(f"   API prefix: {settings.API_V1_STR}")
        logger.info(f"   CORS origins: {settings.BACKEND_CORS_ORIGINS}")
        logger.info("✅ Application startup complete\n")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("\n🛑 Shutting down application...")
    try:
        shutdown_db()
        logger.info("✅ Application shutdown complete")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {str(e)}")

# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-powered data analysis and insights platform",
    docs_url=f"{settings.API_V1_STR}/docs" if settings.DEBUG else None,
    redoc_url=f"{settings.API_V1_STR}/redoc" if settings.DEBUG else None,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ============================================================
# MIDDLEWARE CONFIGURATION (ORDER MATTERS!)
# ============================================================

logger.info(f"🔧 Configuring middleware...")

# ✅ 1. CORS Middleware MUST BE FIRST
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
    expose_headers=["X-Total-Count", "X-Page", "X-Page-Size"],
)
logger.info(f"✅ CORS configured for origins: {settings.BACKEND_CORS_ORIGINS}")

# ✅ 2. GZip Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
logger.info("✅ GZip compression configured")

# ✅ 3. Trusted Host Middleware (for production)
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.BACKEND_CORS_ORIGINS
    )
    logger.info("✅ TrustedHost configured")

# ============================================================
# CUSTOM MIDDLEWARE
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses."""
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"📨 Request: {request.method} {request.url.path} from {client_ip}")
    
    try:
        response = await call_next(request)
    except Exception as e:
        logger.error(f"❌ Request failed: {str(e)}")
        raise
    
    duration = time.time() - start_time
    status_code = response.status_code
    
    if 200 <= status_code < 300:
        status_emoji = "✅"
    elif 300 <= status_code < 400:
        status_emoji = "➡️"
    elif 400 <= status_code < 500:
        status_emoji = "⚠️"
    else:
        status_emoji = "❌"
    
    logger.info(
        f"{status_emoji} Response: {status_code} "
        f"for {request.method} {request.url.path} ({duration:.3f}s)"
    )
    
    response.headers["X-Process-Time"] = str(duration)
    response.headers["X-API-Version"] = settings.VERSION
    
    return response

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)
    
    if settings.is_production:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# ============================================================
# ✅ GLOBAL OPTIONS HANDLER (MUST BE HERE - BEFORE ROUTES)
# ============================================================

@app.options("/{full_path:path}", include_in_schema=False)
async def options_handler(full_path: str) -> Response:
    """
    ✅ CRITICAL: Global handler for CORS preflight (OPTIONS) requests.
    
    This catches ALL OPTIONS requests before they reach route handlers.
    Returns 200 OK immediately without validation.
    """
    logger.debug(f"✅ Preflight request: OPTIONS /{full_path}")
    return Response(
        status_code=200,
        headers={
            "Allow": "GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD",
        }
    )

# ============================================================
# EXCEPTION HANDLERS
# ============================================================

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        f"⚠️  HTTP {exc.status_code}: {exc.detail} "
        f"for {request.method} {request.url.path}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=f"HTTP {exc.status_code}",
            message=str(exc.detail),
        ).model_dump()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")
    
    logger.warning(
        f"❌ Validation error for {request.method} {request.url.path}: {errors}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            success=False,
            error="Validation Error",
            message="Invalid request data",
            details=errors,
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"❌ Unexpected error for {request.method} {request.url.path}: {str(exc)}",
        exc_info=True
    )
    
    if settings.is_production:
        message = "An internal server error occurred"
    else:
        message = str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False,
            error="Internal Server Error",
            message=message,
        ).model_dump()
    )

# ============================================================
# ROOT ENDPOINTS
# ============================================================

@app.get("/", tags=["Root"])
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "healthy",
        "docs_url": f"{settings.API_V1_STR}/docs" if settings.DEBUG else None,
    }

@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    db_healthy = check_db_connection()
    
    health_status = {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": {"connected": db_healthy},
        "environment": settings.ENVIRONMENT,
        "version": settings.VERSION,
    }
    
    status_code = status.HTTP_200_OK if db_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    
    return JSONResponse(
        status_code=status_code,
        content=health_status
    )

@app.get("/info", tags=["Info"])
async def info() -> dict[str, Any]:
    """System information endpoint."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "debug": settings.DEBUG,
        "features": {
            "ai_insights": getattr(settings, "FEATURE_AI_INSIGHTS", True),
            "automl": getattr(settings, "FEATURE_AUTOML", True),
            "collaboration": getattr(settings, "FEATURE_COLLABORATION", True),
        },
        "cors_origins": settings.BACKEND_CORS_ORIGINS,
    }

# ============================================================
# API ROUTE REGISTRATION
# ============================================================

logger.info("📋 Registering API routes...")

try:
    from app.api.v1.api import api_router
    
    app.include_router(api_router)
    
    logger.info("✅ API v1 router registered successfully")
    logger.info("   ├── With all endpoint groups")
    logger.info("   ├── Global OPTIONS handler active")
    logger.info("   └── Ready to accept requests\n")

except Exception as e:
    logger.error(f"❌ Failed to register API router: {str(e)}", exc_info=True)
    raise

# ============================================================
# STARTUP EVENT
# ============================================================

@app.on_event("startup")
async def startup_message():
    """Print startup banner."""
    banner = f"""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║   {settings.PROJECT_NAME:^54}   ║
    ║   Version {settings.VERSION:^48}   ║
    ║   Environment: {settings.ENVIRONMENT:^43}   ║
    ║                                                            ║
    ║   API Docs: {f'{settings.API_V1_STR}/docs':^43}   ║
    ║   Server: {f'http://{settings.HOST}:{settings.PORT}':^47}   ║
    ║                                                            ║
    ║   🔧 CORS Enabled | ✅ OPTIONS Handler | 🔒 Secure      ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """
    print(banner)
    logger.info("🎉 Server started successfully!")

# ============================================================
# EXPORT
# ============================================================

if __name__ == "__main__":
    """Run application directly for development."""
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1 if settings.RELOAD else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
