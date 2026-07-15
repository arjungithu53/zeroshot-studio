"""
Service configuration for production - loads production_* environment variables.

Environment Variable Naming Convention:
- production_* - Service-specific configuration (MongoDB, Redis, SQS queues)
- SHARED_* - Resources shared across services (OpenAI, Anthropic, Google APIs)
- Unprefixed - Infrastructure variables (PORT, HOST, ENVIRONMENT, DEBUG)

IMPORTANT: This service does NOT call load_dotenv() because:
1. All required env vars are passed explicitly via docker-compose.yml
2. Loading .env would import CELERY_BROKER_URL/CELERY_RESULT_BACKEND (meant for ai-script/Redis)
3. production uses SQS, not Redis, so we must avoid loading Redis-related Celery config
"""
import os
from typing import Optional, Tuple, Any
# from dotenv import load_dotenv  # COMMENTED OUT - see docstring above

from infrastructure.mongodb.client import MongoConfig, MongoClientFactory
from infrastructure.s3.client import S3Config, S3ClientFactory

# DO NOT load .env file - all vars passed via docker-compose.yml
# load_dotenv()  # REMOVED - would load CELERY_BROKER_URL for Redis (wrong for production)

# ============================================================================
# Configuration Loaders
# ============================================================================

def load_mongo_config() -> MongoConfig:
    """
    Load MongoDB configuration from production_* environment variables.

    Uses service-specific production_ prefix since this MongoDB instance
    is dedicated to the production service.
    """
    uri = os.getenv("production_MONGODB_URI")
    database_name = os.getenv("production_MONGODB_DATABASE_NAME")
    ssl_verify = os.getenv("production_MONGODB_SSL_VERIFY", "false").lower() == "true"

    if not uri:
        raise ValueError(
            "production_MONGODB_URI environment variable is not set. "
            "Please configure it in your .env file."
        )
    if not database_name:
        raise ValueError(
            "production_MONGODB_DATABASE_NAME environment variable is not set. "
            "Please configure it in your .env file."
        )

    return MongoConfig(uri=uri, database_name=database_name, ssl_verify=ssl_verify)


def load_s3_config() -> S3Config:
    """
    Load S3 configuration from production_* environment variables.

    Uses service-specific production_ prefix since this S3 bucket
    is dedicated to the production service.
    """
    access_key = os.getenv('production_AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('production_AWS_SECRET_ACCESS_KEY')
    bucket_name = os.getenv('production_S3_BUCKET_NAME')
    region = os.getenv('production_AWS_REGION')
    endpoint_url = os.getenv('production_AWS_ENDPOINT_URL')

    if not access_key:
        raise ValueError("production_AWS_ACCESS_KEY_ID not set")
    if not secret_key:
        raise ValueError("production_AWS_SECRET_ACCESS_KEY not set")
    if not bucket_name:
        raise ValueError("production_S3_BUCKET_NAME not set")
    if not region:
        raise ValueError("production_AWS_REGION not set")

    return S3Config(
        access_key_id=access_key,
        secret_access_key=secret_key,
        bucket_name=bucket_name,
        region=region,
        endpoint_url=endpoint_url
    )

# ============================================================================
# Singleton Factories (for connection pooling)
# ============================================================================

_mongo_factory: Optional[MongoClientFactory] = None
_s3_factory: Optional[S3ClientFactory] = None
_mongodb_atlas_client: Optional[Any] = None  # MongoDBAtlasClient singleton for Phase 2/3


def get_mongo_factory() -> MongoClientFactory:
    """Get or create MongoDB factory singleton."""
    global _mongo_factory
    if _mongo_factory is None:
        _mongo_factory = MongoClientFactory(load_mongo_config())
    return _mongo_factory


def get_s3_factory() -> S3ClientFactory:
    """Get or create S3 factory singleton."""
    global _s3_factory
    if _s3_factory is None:
        _s3_factory = S3ClientFactory(load_s3_config())
    return _s3_factory


def get_mongodb_atlas_client() -> Any:
    """
    DEPRECATED: Use get_shots_service() instead.

    Get or create ShotsService singleton for Phase 2/3 operations.

    This singleton ensures only ONE MongoDB connection is used across the entire
    application for shots collection operations, preventing connection pool exhaustion.

    Returns:
        ShotsService instance (singleton)

    Raises:
        ValueError: If MongoDB Atlas URI is not configured
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        "get_mongodb_atlas_client() is DEPRECATED. "
        "Please use get_shots_service() instead for better connection management."
    )
    return get_shots_service()


def get_shots_service() -> Any:
    """
    Get or create ShotsService singleton for Phase 2/3 shots operations.

    This singleton ensures only ONE MongoDB connection is used across the entire
    application for shots collection operations, preventing connection pool exhaustion.

    Returns:
        ShotsService instance (singleton)

    Raises:
        ValueError: If MongoDB Atlas URI is not configured
    """
    global _mongodb_atlas_client
    if _mongodb_atlas_client is None:
        # Import here to avoid circular dependency
        from app.services.shots_service import ShotsService

        # Get client and collection from MongoClientFactory singleton
        client, collection = get_shots_collection()

        # Create ShotsService with singleton connection
        _mongodb_atlas_client = ShotsService(client=client, collection=collection)

    return _mongodb_atlas_client

# ============================================================================
# Convenience Functions (for production collections)
# ============================================================================

def get_database() -> Tuple[Any, Any]:
    """Get MongoDB database with its client."""
    factory = get_mongo_factory()
    client = factory.get_client()
    db = factory.get_database()
    return client, db


def get_projects_collection() -> Tuple[Any, Any]:
    """Get production_projects collection with its client."""
    return get_mongo_factory().get_collection("production_projects")


def get_assets_collection() -> Tuple[Any, Any]:
    """Get production_assets collection with its client."""
    return get_mongo_factory().get_collection("production_assets")


def get_pipelines_collection() -> Tuple[Any, Any]:
    """Get production_pipelines collection with its client."""
    return get_mongo_factory().get_collection("production_pipelines")


def get_shots_collection() -> Tuple[Any, Any]:
    """Get shots collection with its client."""
    return get_mongo_factory().get_collection("shots")


def get_idempotency_keys_collection() -> Tuple[Any, Any]:
    """Get idempotency_keys collection with its client."""
    return get_mongo_factory().get_collection("idempotency_keys")


def reset_mongo_connections() -> dict:
    """
    Reset all MongoDB connections.
    
    Closes all active MongoDB connections from the singleton factory
    and resets the singleton instances. New connections will be created
    automatically on the next use.
    
    Returns:
        dict: Summary of reset operation with counts and status
    
    Example:
        >>> from backend.services.production.app.config import reset_mongo_connections
        >>> result = reset_mongo_connections()
        >>> print(result)
    """
    global _mongo_factory, _mongodb_atlas_client
    import logging
    from infrastructure.mongodb.client import reset_all_connections
    
    logger = logging.getLogger(__name__)
    logger.info("🔄 Resetting all MongoDB connections...")
    
    # Get the factory if it exists
    factory = _mongo_factory
    
    # Reset the factory connection
    factories_to_reset = [factory] if factory is not None else []
    result = reset_all_connections(factories_to_reset)
    
    # Reset singleton instances
    _mongo_factory = None
    _mongodb_atlas_client = None
    
    logger.info("✅ MongoDB singleton instances reset")
    
    return {
        **result,
        'singletons_reset': True
    }


def get_s3_client() -> Any:
    """Get S3 client."""
    return get_s3_factory().get_client()


def get_bucket_name() -> str:
    """Get S3 bucket name."""
    return get_s3_factory().get_bucket_name()


# ============================================================================
# SQS Queue Configuration
# ============================================================================

def get_workflow_queue_name() -> str:
    """
    Get the workflow queue name from production_WORKFLOW_QUEUE environment variable.

    Returns:
        str: Queue name for Celery workflow tasks (defaults to 'my-new-queue' if not set)
        Note: The actual SQS queue name will be 'production-{queue_name}' (e.g., 'production-my-new-queue')
    """
    queue_name = os.getenv('production_WORKFLOW_QUEUE')
    if not queue_name:
        raise ValueError(
            "production_WORKFLOW_QUEUE environment variable is not set. "
            "Please configure it in your .env file."
        )
    return queue_name


# ============================================================================
# S3 Helper Functions
# ============================================================================

def upload_file_wrapper(file_path: str, s3_key: Optional[str] = None, make_public: Optional[bool] = None, content_type: Optional[str] = None, use_presigned_url: Optional[bool] = None, presigned_expiration: int = 86400) -> str:
    """
    Upload file to S3 using service configuration.

    This is a wrapper around infrastructure.s3.upload.upload_file that
    automatically provides s3_client, bucket_name, region, and endpoint_url
    from the service configuration.

    Args:
        file_path: Path to the file to upload
        s3_key: Optional S3 key (path) for the file
        make_public: Whether to make the file publicly accessible (deprecated - use use_presigned_url instead)
        content_type: Optional content type for the file
        use_presigned_url: If True, return a pre-signed URL (default: True for security)
        presigned_expiration: Expiration time for pre-signed URL in seconds (default: 86400 = 24 hours)

    Returns:
        str: URL of the uploaded file (pre-signed by default)
    """
    import os
    from infrastructure.s3.upload import upload_file

    factory = get_s3_factory()
    s3_client = factory.get_client()
    bucket_name = factory.get_bucket_name()
    region = factory.config.region
    endpoint_url = factory.config.endpoint_url

    # Default to using pre-signed URLs for security
    if use_presigned_url is None:
        use_presigned_url_env = os.getenv('production_S3_USE_PRESIGNED_URL')
        if use_presigned_url_env is not None:
            use_presigned_url = use_presigned_url_env.lower() == 'true'
        else:
            # Default to True for security (since bucket has public access blocked)
            use_presigned_url = True

    # Get make_public from env if not specified (deprecated - kept for backward compatibility)
    if make_public is None:
        public_read_env = os.getenv('production_S3_PUBLIC_READ')
        if public_read_env is not None:
            make_public = public_read_env.lower() == 'true'
        else:
            make_public = False

    return upload_file(
        file_path=file_path,
        s3_client=s3_client,
        bucket_name=bucket_name,
        s3_key=s3_key,
        make_public=make_public,
        content_type=content_type,
        region=region,
        endpoint_url=endpoint_url,
        use_presigned_url=use_presigned_url,
        presigned_expiration=presigned_expiration
    )
