# backend/app/api/v1/endpoints/__init__.py

"""
API v1 Endpoints Package.

Contains all endpoint route modules for the API v1.
Each module exports a 'router' object that gets registered in api.py
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================
# LAZY IMPORTS - Import on demand
# ============================================================

# These imports are lazy (only when explicitly imported)
# This prevents circular imports and allows for conditional loading

__all__ = [
    "auth",
    "users",
    "datasets",
    "cleaning",
    "eda",
    "insights",
    "visualizations",
]

logger.debug("✅ API v1 endpoints package initialized")
