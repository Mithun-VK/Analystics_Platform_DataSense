"""
API v1 Router Aggregator.
Combines all API endpoint routers into a single router.
"""

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

# ============================================================
# CREATE MAIN API ROUTER
# ============================================================

api_router = APIRouter(prefix="/api/v1")

# ============================================================
# ❌ REMOVED OPTIONS HANDLER
# ============================================================
# REASON: CORSMiddleware in main.py handles all OPTIONS requests
# Having OPTIONS handlers at multiple levels causes 405 errors
# FastAPI's CORSMiddleware automatically responds to OPTIONS with 200

# ============================================================
# IMPORT ENDPOINT ROUTERS
# ============================================================

try:
    from app.api.v1.endpoints import auth
    HAS_AUTH = True
    logger.info("✅ Auth endpoints loaded")
except ImportError as e:
    HAS_AUTH = False
    logger.warning(f"⚠️  Auth endpoints not loaded: {e}")
except Exception as e:
    HAS_AUTH = False
    logger.error(f"❌ Error loading auth endpoints: {e}", exc_info=True)

try:
    from app.api.v1.endpoints import users
    HAS_USERS = True
    logger.info("✅ Users endpoints loaded")
except ImportError as e:
    HAS_USERS = False
    logger.warning(f"⚠️  Users endpoints not loaded: {e}")
except Exception as e:
    HAS_USERS = False
    logger.error(f"❌ Error loading users endpoints: {e}", exc_info=True)

try:
    from app.api.v1.endpoints import datasets
    HAS_DATASETS = True
    logger.info("✅ Datasets endpoints loaded")
except ImportError as e:
    HAS_DATASETS = False
    logger.warning(f"⚠️  Datasets endpoints not loaded: {e}")
except Exception as e:
    HAS_DATASETS = False
    logger.error(f"❌ Error loading datasets endpoints: {e}", exc_info=True)

try:
    from app.api.v1.endpoints import cleaning
    HAS_CLEANING = True
    logger.info("✅ Cleaning endpoints loaded")
except ImportError as e:
    HAS_CLEANING = False
    logger.warning(f"⚠️  Cleaning endpoints not loaded: {e}")
except Exception as e:
    HAS_CLEANING = False
    logger.error(f"❌ Error loading cleaning endpoints: {e}", exc_info=True)

try:
    from app.api.v1.endpoints import eda
    HAS_EDA = True
    logger.info("✅ EDA endpoints loaded")
except ImportError as e:
    HAS_EDA = False
    logger.warning(f"⚠️  EDA endpoints not loaded: {e}")
except Exception as e:
    HAS_EDA = False
    logger.error(f"❌ Error loading EDA endpoints: {e}", exc_info=True)

try:
    from app.api.v1.endpoints import insights
    HAS_INSIGHTS = True
    logger.info("✅ Insights endpoints loaded")
except ImportError as e:
    HAS_INSIGHTS = False
    logger.warning(f"⚠️  Insights endpoints not loaded: {e}")
except Exception as e:
    HAS_INSIGHTS = False
    logger.error(f"❌ Error loading insights endpoints: {e}", exc_info=True)

try:
    from app.api.v1.endpoints import visualizations
    HAS_VISUALIZATIONS = True
    logger.info("✅ Visualizations endpoints loaded")
except ImportError as e:
    HAS_VISUALIZATIONS = False
    logger.warning(f"⚠️  Visualizations endpoints not loaded: {e}")
except Exception as e:
    HAS_VISUALIZATIONS = False
    logger.error(f"❌ Error loading visualizations endpoints: {e}", exc_info=True)

# ============================================================
# REGISTER ENDPOINT ROUTERS
# ============================================================

if HAS_AUTH:
    api_router.include_router(
        auth.router,
        prefix="/auth",
        tags=["Authentication"]
    )
    logger.info("✅ Auth routes registered at /api/v1/auth")

if HAS_USERS:
    api_router.include_router(
        users.router,
        prefix="/users",
        tags=["Users"]
    )
    logger.info("✅ Users routes registered at /api/v1/users")

if HAS_DATASETS:
    api_router.include_router(
        datasets.router,
        prefix="/datasets",
        tags=["Datasets"]
    )
    logger.info("✅ Datasets routes registered at /api/v1/datasets")

if HAS_CLEANING:
    api_router.include_router(
        cleaning.router,
        prefix="/cleaning",
        tags=["Data Cleaning"]
    )
    logger.info("✅ Cleaning routes registered at /api/v1/cleaning")

if HAS_EDA:
    api_router.include_router(
        eda.router,
        prefix="/eda",
        tags=["Exploratory Data Analysis"]
    )
    logger.info("✅ EDA routes registered at /api/v1/eda")

if HAS_INSIGHTS:
    api_router.include_router(
        insights.router,
        prefix="/insights",
        tags=["AI Insights"]
    )
    logger.info("✅ Insights routes registered at /api/v1/insights")

if HAS_VISUALIZATIONS:
    api_router.include_router(
        visualizations.router,
        prefix="/visualizations",
        tags=["Visualizations"]
    )
    logger.info("✅ Visualizations routes registered at /api/v1/visualizations")

# ============================================================
# ROUTER SUMMARY
# ============================================================

enabled_count = sum([
    HAS_AUTH,
    HAS_USERS,
    HAS_DATASETS,
    HAS_CLEANING,
    HAS_EDA,
    HAS_INSIGHTS,
    HAS_VISUALIZATIONS
])

logger.info(f"\n📊 API Router Summary: {enabled_count}/7 endpoint groups loaded")
logger.info("✅ All routes registered and ready for requests")
