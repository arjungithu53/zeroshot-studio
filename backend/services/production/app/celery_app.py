"""
Celery Application Configuration for production Service
===================================================

This module configures Celery with Amazon SQS as the message broker for distributed task processing.

Why Celery + SQS?
-----------------
1. **Reliability**: Tasks persist in SQS even if workers crash
2. **Scalability**: Multiple workers can process tasks in parallel
3. **Separation**: Heavy processing separated from API server
4. **Monitoring**: Track task status and failures
5. **Cost-effective**: SQS is extremely cheap (~$0.40 per million tasks)

Architecture:
-------------
FastAPI Endpoint → SQS Queue → Celery Worker → Process Task → Update MongoDB
        ↓                           ↓
    Return job_id              Heavy AI Processing
    immediately               (8 agents, video gen)
"""

import os
from celery import Celery
from kombu.utils.url import safequote
# from dotenv import load_dotenv  # NOT USED - see environment config section below
from pathlib import Path
import sys
from typing import Dict, Any

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

# ============================================================================
# Environment Configuration
# ============================================================================

# NOTE: We do NOT load .env file here anymore!
# Environment variables are passed explicitly via docker-compose.yml
# This prevents loading CELERY_BROKER_URL and CELERY_RESULT_BACKEND from .env
# which are meant for the ai-script service (Redis), not production (SQS)
logger.info("production Celery: Using environment variables from docker-compose (not .env file)")

# ============================================================================
# Service-Specific Configuration (production_* prefix)
# ============================================================================
# AWS Credentials - loaded from environment variables
# Use production_ prefixed variables (service-specific AWS resources)
# Maintains backward compatibility with unprefixed AWS_* variables
AWS_ACCESS_KEY_ID = safequote(os.getenv('production_AWS_ACCESS_KEY_ID') or os.getenv('AWS_ACCESS_KEY_ID', ''))
AWS_SECRET_ACCESS_KEY = safequote(os.getenv('production_AWS_SECRET_ACCESS_KEY') or os.getenv('AWS_SECRET_ACCESS_KEY', ''))
AWS_REGION = os.getenv('production_AWS_REGION') or os.getenv('AWS_REGION', 'eu-north-1')
AWS_ACCOUNT_ID = os.getenv('AWS_ACCOUNT_ID', '')  # Infrastructure-level, unprefixed

# Validate AWS credentials for SQS
if not AWS_ACCOUNT_ID:
    raise ValueError(
        "AWS_ACCOUNT_ID must be set in environment for SQS queue configuration."
    )
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError(
        "production_AWS_ACCESS_KEY_ID and production_AWS_SECRET_ACCESS_KEY must be set."
    )

# MongoDB for storing task results (production-specific database)
# Use production_ prefixed variables (service-specific MongoDB instance)
# Maintains backward compatibility with MONGO_* variables
MONGO_URI = os.getenv('production_MONGODB_URI') or os.getenv('MONGO_URI', '')
MONGO_DB = os.getenv('production_MONGODB_DATABASE_NAME') or os.getenv('MONGO_DB', 'production')
ALLOW_LOCAL_MONGO = os.getenv('production_ALLOW_LOCAL_MONGO', 'false').lower() == 'true'

# CRITICAL VALIDATION: Ensure MongoDB Atlas URI is set (not localhost unless explicitly allowed)
if not MONGO_URI:
    raise ValueError(
        "production_MONGODB_URI or MONGO_URI must be set in environment. "
        "Check the .env file at the project root."
    )

is_local_mongo = MONGO_URI.startswith('mongodb://localhost') or MONGO_URI.startswith('mongodb://127.0.0.1')

if is_local_mongo and not ALLOW_LOCAL_MONGO:
    raise ValueError(
        f"MongoDB URI is set to localhost, but Atlas is required in production. "
        f"Current: {MONGO_URI}\n"
        f"Set production_ALLOW_LOCAL_MONGO=true in your environment only for local testing."
    )

if is_local_mongo and ALLOW_LOCAL_MONGO:
    logger.warning("Using local MongoDB instance for development/testing.")

logger.info(f"MongoDB URI configured: {MONGO_URI[:50]}...")
logger.info(f"MongoDB Database: {MONGO_DB}")

# Fix MongoDB URI for SSL certificate issues with Atlas
# Add tlsAllowInvalidCertificates if not present (for MongoDB Atlas SSL issues)
if not is_local_mongo and ('mongodb+srv://' in MONGO_URI or 'mongodb://' in MONGO_URI):
    if 'tlsAllowInvalidCertificates' not in MONGO_URI:
        separator = '&' if '?' in MONGO_URI else '?'
        MONGO_URI = f"{MONGO_URI}{separator}tlsAllowInvalidCertificates=true"

# SQS Broker URL
# Format: sqs://AWS_ACCESS_KEY:AWS_SECRET_KEY@
BROKER_URL = f"sqs://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}@"

# Result Backend - using file-based storage to avoid MongoDB connection issues
# IMPORTANT: We don't use MongoDB for result backend because:
# 1. Creates additional MongoDB connections that cause SSL handshake errors
# 2. We save all task results directly to MongoDB via application code
# 3. File-based backend is sufficient for task status tracking
# Result files are stored temporarily and cleaned up automatically
import tempfile
import os
_celery_results_dir = os.path.join(tempfile.gettempdir(), 'celery_results')
os.makedirs(_celery_results_dir, exist_ok=True)
RESULT_BACKEND = f'file://{_celery_results_dir}'

# ============================================================================
# Celery Application Instance
# ============================================================================

celery_app = Celery(
    "production_tasks",
    # Include task modules
    include=[
        "app.tasks.phase1_tasks",
        "app.tasks.phase2_tasks",
        "app.tasks.phase3_tasks",
        "app.tasks.phase4_tasks",
        "app.tasks.master_tasks",
    ]
)

# CRITICAL: Set broker and backend AFTER creating Celery app to override env vars
# This must be done via conf.update() to override environment variable detection
celery_app.conf.broker_url = BROKER_URL
celery_app.conf.result_backend = RESULT_BACKEND

print(f"[production CELERY] Broker set to: {BROKER_URL[:50]}...")
print(f"[production CELERY] Backend set to: {RESULT_BACKEND[:50]}...")

# ============================================================================
# Celery Configuration
# ============================================================================
# Workflow queue name is required for SQS routing. Allow a sensible default so
# worker startup doesn't crash, but log loudly when falling back.
# NOTE: Queue name should be the FULL name including any prefix (e.g., 'production-workflow')
_queue_name = (
    os.getenv('production_WORKFLOW_QUEUE')
    or os.getenv('WORKER_QUEUE')
    or 'production-workflow'
)
if not os.getenv('production_WORKFLOW_QUEUE'):
    logger.warning(
        "production_WORKFLOW_QUEUE not set; falling back to "
        f"'{_queue_name}'. Set production_WORKFLOW_QUEUE to control the SQS queue."
    )

logger.info("=" * 80)
logger.info("🚀 production CELERY CONFIGURATION")
logger.info("=" * 80)
logger.info(f"📦 Queue Name:              {_queue_name}")
logger.info(f"🌍 SQS Region:              {AWS_REGION}")
logger.info(f"🔗 SQS Queue URL:           https://sqs.{AWS_REGION}.amazonaws.com/{AWS_ACCOUNT_ID}/{_queue_name}")
logger.info("")
logger.info("📋 Task Routing Configuration:")
logger.info(f"   ✓ Phase 1 tasks → queue='{_queue_name}'")
logger.info(f"   ✓ Phase 2 tasks → queue='{_queue_name}'")
logger.info(f"   ✓ Phase 3 tasks → queue='{_queue_name}'")
logger.info(f"   ✓ Phase 4 tasks → queue='{_queue_name}'")
logger.info(f"   ✓ Default queue → '{_queue_name}'")
logger.info("=" * 80)

celery_app.conf.update(
    # ----- Broker Connection Settings -----
    # CRITICAL: Explicitly set broker_url to prevent environment variable override
    broker_url=BROKER_URL,

    # Celery 6.0+ requires explicit broker connection retry configuration
    broker_connection_retry_on_startup=True,  # Retry broker connections during startup

    # ----- SQS Broker Settings -----
    broker_transport_options={
        'region': AWS_REGION,

        # Visibility timeout: how long a task is invisible to other workers after being picked up
        # Set to 2 hours to handle long video generation tasks
        'visibility_timeout': 7200,  # 2 hours in seconds

        # Polling interval: how often workers check SQS for new messages
        'polling_interval': 1,  # Check every second for responsiveness

        # Queue name prefix - EMPTY since we pass the full queue name in env vars
        'queue_name_prefix': '',

        # Wait time for long polling (reduces API calls and costs)
        'wait_time_seconds': 20,  # Maximum is 20 seconds

        # Single workflow queue for all tasks
        # API controls phase ordering, queue just processes tasks
        # Note: Queue name is the FULL name from environment variable (e.g., 'production-workflow')
        # IMPORTANT: predefined_queues must always be set for SQS transport
        # Key: Full queue name as Celery will look for it
        # URL: Actual SQS queue URL in AWS
        'predefined_queues': {
            _queue_name: {
                'url': f'https://sqs.{AWS_REGION}.amazonaws.com/{AWS_ACCOUNT_ID}/{_queue_name}',
            },
        },
    },

    # ----- Task Serialization -----
    # Use JSON for security and compatibility
    task_serializer='json',
    accept_content=['json'],  # Only accept JSON (prevents pickle exploits)
    result_serializer='json',

    # ----- Result Backend Settings -----
    result_backend=RESULT_BACKEND,
    result_expires=86400,  # Results expire after 24 hours (adjustable)
    result_extended=True,  # Store additional metadata

    # ----- Task Execution Settings -----
    task_track_started=True,  # Track when tasks start (enables progress monitoring)
    task_send_sent_event=True,  # Send events when tasks are sent
    task_ignore_result=False,  # Store results (needed for status checking)

    # Time limits for tasks
    task_time_limit=7200,  # 2 hours hard limit (kills task after this)
    task_soft_time_limit=6900,  # 1h 55min soft limit (raises exception to allow cleanup)

    # ----- Worker Settings -----
    worker_prefetch_multiplier=1,  # Only fetch 1 task at a time (prevents blocking)
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevents memory leaks)
    worker_disable_rate_limits=True,  # No rate limits needed

    # ----- Task Routing -----
    # All phases use the workflow queue
    # Note: queue name without prefix (Celery adds 'production-' automatically)
    task_routes={
        'app.tasks.phase1_tasks.*': {'queue': _queue_name},
        'app.tasks.phase2_tasks.*': {'queue': _queue_name},
        'app.tasks.phase3_tasks.*': {'queue': _queue_name},
        'app.tasks.phase4_tasks.*': {'queue': _queue_name},
    },

    # Default queue for tasks without explicit routing
    task_default_queue=_queue_name,
    task_default_exchange=_queue_name,
    task_default_routing_key=_queue_name,

    # ----- Retry Policy -----
    # Acknowledge immediately on pickup to prevent duplicate processing
    # If task fails, it will be retried automatically (goes back to queue)
    task_acks_late=False,  # Acknowledge immediately when task is picked up
    task_reject_on_worker_lost=True,  # Requeue if worker crashes

    # Retry configuration - failed tasks go back to queue
    task_default_retry_delay=60,  # Wait 1 minute before retry
    task_max_retries=3,  # Maximum 3 retry attempts
    task_acks_on_failure_or_timeout=True,  # Acknowledge even on failure (let retry logic handle it)

    # ----- Timezone -----
    enable_utc=True,
    timezone='UTC',

    # ----- Monitoring & Events -----
    worker_send_task_events=True,  # Send events for monitoring
)

# ============================================================================
# Celery Signals (for logging and debugging)
# ============================================================================

import logging
from celery.signals import after_setup_logger, after_setup_task_logger

class PollTaskFilter(logging.Filter):
    """Filter out noisy polling task logs to keep the console clean."""
    def filter(self, record):
        msg = record.getMessage()
        if "master.poll_phase" in msg or "poll_phase" in msg:
            # We don't filter error logs, just INFO/DEBUG level spam
            if record.levelno <= logging.INFO:
                return False
        return True

@after_setup_logger.connect
def setup_celery_logger(logger, **kwargs):
    logger.addFilter(PollTaskFilter())

@after_setup_task_logger.connect
def setup_celery_task_logger(logger, **kwargs):
    logger.addFilter(PollTaskFilter())

@celery_app.task(bind=True)
def debug_task(self: Any) -> Dict[str, str]:
    """Debug task to test Celery configuration"""
    logger.debug(f'Request: {self.request!r}')
    return {'status': 'success', 'message': 'Celery is working!'}

# ============================================================================
# Export
# ============================================================================

__all__ = ['celery_app']
