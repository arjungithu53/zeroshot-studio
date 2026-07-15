"""
FastAPI Main Application for production Service
"""
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any

# Add project root to Python path for infrastructure imports
project_root = Path(__file__).resolve().parents[4]  # Go up to /Users/aiteam2/aishots
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request, Response
# from dotenv import load_dotenv

# DO NOT load .env file - all vars passed via docker-compose.yml
# Loading .env would import CELERY_BROKER_URL/CELERY_RESULT_BACKEND (meant for ai-script/Redis)
# production uses SQS, not Redis, so we must avoid loading Redis-related Celery config
# load_dotenv()  # REMOVED

# Import CORS configuration utility
from backend.shared.utils.cors_config import add_cors_middleware

# Import slowapi for rate limiting (optional dependency)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_ENABLED = True
except ModuleNotFoundError:  # pragma: no cover - fallback path
    logging.warning("slowapi not installed; API rate limiting disabled")

    RATE_LIMITING_ENABLED = False

    class _NoOpLimiter:
        """Fallback limiter that preserves decorator API but performs no limiting."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401
            # Accept arbitrary arguments to mirror slowapi.Limiter signature
            self.limit = self._no_op_decorator

        @staticmethod
        def _no_op_decorator(*_args, **_kwargs):
            def _decorator(func):
                return func

            return _decorator

    def _rate_limit_exceeded_handler(request, exc):  # noqa: D401
        # This handler should never be invoked without real rate limiting,
        # but we keep the signature compatible.
        return Response(
            content="Rate limiting not configured.",
            status_code=503,
        )

    def get_remote_address(request: Request) -> str:  # noqa: D401
        if request.client:
            return request.client.host
        return "anonymous"

    class RateLimitExceeded(Exception):  # noqa: D401
        """Placeholder exception for rate limit errors."""

    Limiter = _NoOpLimiter

# Import routers
from app.api.v1.endpoints import projects, phase1
from app.api.v1.endpoints import phase2, phase3, movies
from app.api.v1.endpoints import master
from app.api.v1.endpoints import phase4

# Import configuration for health checks
from app.config import get_mongo_factory, get_s3_factory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
# Uses remote address as key function (can be changed to user ID for authenticated endpoints)
# Infrastructure variables (unprefixed) - configuration, not service-specific
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "1000/hour")],
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://")
)

# Create FastAPI app
app = FastAPI(
    title="production Service API",
    description="AI-powered image generation pipeline for script-to-video production",
    version="1.0.0"
)

# Add rate limiter to app state
app.state.limiter = limiter

# Add rate limit exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - uses environment-based configuration
add_cors_middleware(app)

# Include routers
app.include_router(projects.router, prefix="/api/v1")
app.include_router(phase1.router, prefix="/api/v1")
app.include_router(phase2.router, prefix="/api/v1")
app.include_router(phase3.router, prefix="/api/v1")
app.include_router(movies.router, prefix="/api/v1")
app.include_router(master.router, prefix="/api/v1")
app.include_router(phase4.router, prefix="/api/v1")
# app.include_router(test_helpers.router, prefix="/api/v1")


@app.get("/")
@limiter.limit("120/minute")
async def root(request: Request) -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "service": "production",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "projects": "/api/v1/projects",
            "movies": "/api/v1/movies",
            "phase1_workflow": "/api/v1/phase1",
            "phase2_workflow": "/api/v1/phase2",
            "phase3_workflow": "/api/v1/phase3",
            "master_pipeline": "/api/v1/master",
            "docs": "/docs"
        }
    }


@app.get("/health")
@limiter.limit("120/minute")
async def health_check(request: Request, response: Response) -> Dict[str, Any]:
    """
    Health check endpoint that verifies MongoDB and S3 connectivity.

    Returns:
        dict: Health status with detailed dependency checks
        - status: Overall health status (healthy/degraded/unhealthy)
        - timestamp: ISO format timestamp of the check
        - dependencies: Status of each critical dependency

    HTTP Status Codes:
        - 200: Service is healthy
        - 503: Service is unhealthy or degraded
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "production",
        "version": "1.0.0",
        "dependencies": {}
    }

    # Check MongoDB connectivity
    mongodb_status = "unknown"
    mongodb_error = None
    try:
        mongo_factory = get_mongo_factory()
        client = mongo_factory.get_client()

        # Ping with a short timeout to prevent hanging
        # serverSelectionTimeoutMS is set in the client config
        client.admin.command('ping')
        mongodb_status = "healthy"
        logger.debug("MongoDB health check passed")
    except Exception as e:
        mongodb_status = "unhealthy"
        mongodb_error = str(e)
        health_status["status"] = "degraded"
        logger.error(f"MongoDB health check failed: {e}")

    health_status["dependencies"]["mongodb"] = {
        "status": mongodb_status,
        "error": mongodb_error
    }

    # Check S3 bucket accessibility
    s3_status = "unknown"
    s3_error = None
    try:
        s3_factory = get_s3_factory()
        bucket_accessible = s3_factory.check_bucket_exists()

        if bucket_accessible:
            s3_status = "healthy"
            logger.debug("S3 health check passed")
        else:
            s3_status = "unhealthy"
            s3_error = "Bucket not accessible"
            health_status["status"] = "degraded"
            logger.warning("S3 bucket not accessible")
    except Exception as e:
        s3_status = "unhealthy"
        s3_error = str(e)
        health_status["status"] = "degraded"
        logger.error(f"S3 health check failed: {e}")

    health_status["dependencies"]["s3"] = {
        "status": s3_status,
        "error": s3_error
    }

    # Determine overall status and HTTP status code
    # If MongoDB is down, mark as unhealthy (MongoDB is critical)
    if mongodb_status == "unhealthy":
        health_status["status"] = "unhealthy"

    # Set HTTP status code based on health status
    if health_status["status"] == "unhealthy":
        response.status_code = 503
    elif health_status["status"] == "degraded":
        response.status_code = 503
    else:
        response.status_code = 200

    return health_status


@app.get("/mongodb/connections")
@limiter.limit("120/minute")
async def mongodb_connections(request: Request) -> Dict[str, Any]:
    """
    Get MongoDB connection statistics.

    Returns detailed information about active MongoDB connections including:
    - Total CLIENT instances created since startup
    - Currently active CLIENT instances
    - ACTUAL pooled connections to MongoDB server
    - Connection pool statistics per client

    Returns:
        dict: Connection statistics and details
    """
    try:
        from infrastructure.mongodb.client import get_connection_stats

        stats = get_connection_stats()

        # Get actual connection pool stats from MongoDB server
        mongo_factory = get_mongo_factory()
        client = mongo_factory.get_client()

        # Query MongoDB for actual server connections
        server_status = None
        current_connections = None
        pool_stats = []

        try:
            # Get server status for connection info
            server_status = client.admin.command('serverStatus')
            current_connections = server_status.get('connections', {})

            # Get connection pool stats from each client
            from infrastructure.mongodb.client import _active_connections
            for conn_id, details in _active_connections.items():
                try:
                    # Get pool stats using topology description
                    pool_info = {
                        'connection_id': conn_id,
                        'pid': details['pid'],
                        'database': details['database'],
                        'configured_pool_size': f"{details['min_pool_size']}-{details['pool_size']}",
                    }
                    pool_stats.append(pool_info)
                except Exception as e:
                    logger.debug(f"Could not get pool stats for connection {conn_id}: {e}")

        except Exception as e:
            logger.warning(f"Could not get MongoDB server stats: {e}")

        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "client_statistics": {
                "total_clients_created": stats['total_connections_created'],
                "currently_active_clients": stats['active_connections_count'],
                "client_details": stats['active_connections']
            },
            "server_statistics": {
                "current_connections": current_connections.get('current') if current_connections else "N/A",
                "available_connections": current_connections.get('available') if current_connections else "N/A",
                "total_created": current_connections.get('totalCreated') if current_connections else "N/A",
                "active_connections": current_connections.get('active') if current_connections else "N/A",
            },
            "pool_statistics": pool_stats
        }
    except Exception as e:
        logger.error(f"Failed to get connection stats: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
