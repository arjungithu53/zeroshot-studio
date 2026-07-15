"""Redis-backed quota management for production service.

This module provides quota limiting for costly video generation pipelines.
Uses Redis counters with automatic expiration for daily and weekly limits.

Design:
- Separate Redis database (DB 1) from Celery (DB 0) for isolation
- Pipeline-specific quotas (Phase 1, 2, 3 have different costs)
- Atomic operations using Redis WATCH/MULTI for thread safety
- Auto-expiring keys using TTL (no manual cleanup needed)

Environment Variables:
- REDIS_URL: Redis connection URL (defaults to redis://redis:6379/0)
- production_QUOTA_NAMESPACE: Redis key prefix (defaults to "quota:production")

Redis Key Format:
    quota:production:{user_id}:{period}:{pipeline_name}

Examples:
    quota:production:user123:daily:phase1_workflow = 2
    quota:production:user123:weekly:phase1_workflow = 8
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

import redis
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when a user exceeds their quota limit."""
    pass


@dataclass(frozen=True)
class QuotaLimits:
    """Daily and weekly quota limits for a pipeline."""
    daily: int
    weekly: int


class QuotaManager:
    """Manage per-user quotas for costly production pipelines.

    Video generation is expensive, so we enforce strict limits:
    - Phase 1 (Image generation): 5 daily, 28 weekly
    - Phase 2 (Shot strategy): 3 daily, 15 weekly
    - Phase 3 (Video generation): 2 daily, 10 weekly (most expensive)

    Uses Redis for:
    - Atomic increment operations (thread-safe)
    - Automatic key expiration (no cleanup needed)
    - Fast in-memory lookups (<1ms)
    """

    # Time windows for quota periods
    DAILY_WINDOW_SECONDS = 24 * 60 * 60
    WEEKLY_WINDOW_SECONDS = 7 * 24 * 60 * 60

    # UNIFIED QUOTA SYSTEM
    # All pipelines share the same daily limit to prevent resource exhaustion
    # This ensures users can't bypass limits by using different endpoints
    PIPELINE_LIMITS: Dict[str, QuotaLimits] = {
        # All operations share unified quota (adjust as needed)
        "production_workflow": QuotaLimits(daily=50, weekly=280),
    }

    def __init__(
        self,
        *,
        redis_client: Optional[redis.Redis] = None,
        redis_url: Optional[str] = None,
        namespace: str = "quota:production",
        default_limits: Optional[QuotaLimits] = None,
    ) -> None:
        """Initialize quota manager.

        Args:
            redis_client: Optional pre-configured Redis client
            redis_url: Optional Redis URL (overrides REDIS_URL env var)
            namespace: Redis key prefix for quota keys
            default_limits: Fallback limits if pipeline not in PIPELINE_LIMITS
        """
        # Get Redis URL from args or environment
        raw_url = redis_url or os.getenv("REDIS_URL", "")
        if not raw_url:
            raise ValueError(
                "REDIS_URL environment variable must be set for quota management. "
                "Please configure it in your .env file."
            )

        # Normalize Redis URL to ensure it has the redis:// protocol prefix
        if not raw_url.startswith(("redis://", "rediss://")):
            raw_url = f"redis://{raw_url}"

        # Use Redis database 1 for quotas (Celery uses database 0)
        # This provides isolation between quota data and task queues
        if '/0' in raw_url:
            self.redis_url = raw_url.replace('/0', '/1')
            logger.info("Using Redis DB 1 for quotas (Celery uses DB 0)")
        else:
            self.redis_url = raw_url

        # Create Redis client
        self.redis_client = redis_client or redis.Redis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )

        self.namespace = namespace
        self.default_limits = default_limits or QuotaLimits(daily=50, weekly=280)

        # Test Redis connection on initialization
        try:
            self.redis_client.ping()
            logger.info(f"QuotaManager initialized with namespace '{namespace}'")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _get_limits_for_pipeline(self, pipeline_name: str) -> QuotaLimits:
        """Get quota limits for a specific pipeline.

        Args:
            pipeline_name: Name of the pipeline (e.g., "phase1_workflow")

        Returns:
            QuotaLimits with daily and weekly limits
        """
        limits = self.PIPELINE_LIMITS.get(pipeline_name, self.default_limits)

        if pipeline_name not in self.PIPELINE_LIMITS:
            logger.warning(
                f"No specific quota limits for '{pipeline_name}', "
                f"using defaults: {limits}"
            )

        return limits

    def consume(self, user_id: str, pipeline_name: str) -> None:
        """Consume one unit from user's quota for the given pipeline.

        This should be called BEFORE starting any expensive operations.
        If quota is exceeded, raises HTTPException with status 429.

        Args:
            user_id: User identifier (required)
            pipeline_name: Pipeline identifier (e.g., "phase1_workflow")

        Raises:
            HTTPException: 400 if user_id is missing
            HTTPException: 429 if quota exceeded
            HTTPException: 503 if Redis is unavailable
        """
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required to enforce quotas.",
            )

        # Get limits for this specific pipeline
        limits = self._get_limits_for_pipeline(pipeline_name)

        logger.info(
            f"Checking quota for user={user_id}, pipeline={pipeline_name}, "
            f"limits=(daily={limits.daily}, weekly={limits.weekly})"
        )

        try:
            # Check and increment daily quota
            self._consume_period(
                user_id=user_id,
                period="daily",
                limit=limits.daily,
                ttl_seconds=self.DAILY_WINDOW_SECONDS,
                pipeline_name=pipeline_name,
            )

            # Check and increment weekly quota
            self._consume_period(
                user_id=user_id,
                period="weekly",
                limit=limits.weekly,
                ttl_seconds=self.WEEKLY_WINDOW_SECONDS,
                pipeline_name=pipeline_name,
            )

            logger.info(f"Quota consumed for user={user_id}, pipeline={pipeline_name}")

        except QuotaExceededError as exc:
            logger.warning(
                f"Quota exceeded for user={user_id}, pipeline={pipeline_name}: {exc}"
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
            ) from exc

        except redis.RedisError as exc:
            logger.error(
                f"Redis error during quota check for user={user_id}: {exc}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quota service temporarily unavailable. Please try again later.",
            ) from exc

    def _consume_period(
        self,
        *,
        user_id: str,
        period: str,
        limit: int,
        ttl_seconds: int,
        pipeline_name: str,
    ) -> None:
        """Atomically check and increment quota for a specific time period.

        Uses Redis WATCH/MULTI for optimistic locking to prevent race conditions.

        Args:
            user_id: User identifier
            period: Time period ("daily" or "weekly")
            limit: Maximum allowed count for this period
            ttl_seconds: How long the quota counter should persist
            pipeline_name: Pipeline identifier

        Raises:
            QuotaExceededError: If current count >= limit
        """
        key = f"{self.namespace}:{user_id}:{period}:{pipeline_name}"

        with self.redis_client.pipeline() as pipe:
            while True:
                try:
                    # Watch the key for concurrent modifications
                    pipe.watch(key)

                    # Get current count
                    current_raw = pipe.get(key)
                    current = int(current_raw) if current_raw is not None else 0

                    logger.debug(
                        f"Quota check: key={key}, current={current}, limit={limit}"
                    )

                    # Check if quota would be exceeded
                    if current >= limit:
                        pipe.unwatch()
                        raise QuotaExceededError(
                            f"{period.capitalize()} quota exceeded for {pipeline_name}. "
                            f"Limit: {limit}, Used: {current}"
                        )

                    # Atomically increment and set expiry
                    pipe.multi()
                    pipe.incr(key, 1)

                    # Set TTL only on first increment (when current == 0)
                    if current == 0:
                        pipe.expire(key, ttl_seconds)
                        logger.debug(f"Set TTL={ttl_seconds}s for new quota key: {key}")

                    pipe.execute()
                    break  # Success!

                except redis.WatchError:
                    # Another request modified the key, retry
                    logger.debug(f"Concurrent modification detected for {key}, retrying...")
                    continue

    def get_usage(self, user_id: str, pipeline_name: str) -> Dict[str, int]:
        """Get current quota usage for a user and pipeline.

        Useful for displaying remaining quota to users.

        Args:
            user_id: User identifier
            pipeline_name: Pipeline identifier

        Returns:
            Dict with keys: daily_used, daily_limit, weekly_used, weekly_limit
        """
        limits = self._get_limits_for_pipeline(pipeline_name)

        daily_key = f"{self.namespace}:{user_id}:daily:{pipeline_name}"
        weekly_key = f"{self.namespace}:{user_id}:weekly:{pipeline_name}"

        try:
            daily_used = int(self.redis_client.get(daily_key) or 0)
            weekly_used = int(self.redis_client.get(weekly_key) or 0)

            return {
                "daily_used": daily_used,
                "daily_limit": limits.daily,
                "daily_remaining": max(0, limits.daily - daily_used),
                "weekly_used": weekly_used,
                "weekly_limit": limits.weekly,
                "weekly_remaining": max(0, limits.weekly - weekly_used),
            }
        except redis.RedisError as exc:
            logger.error(f"Failed to get usage for user={user_id}: {exc}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to retrieve quota usage.",
            ) from exc

    def reset_all_quotas(self) -> int:
        """Reset all quota data by deleting all keys in namespace.

        Use with caution! This affects all users and pipelines.

        Returns:
            Number of keys deleted
        """
        try:
            # Find all keys matching the quota namespace pattern
            pattern = f"{self.namespace}:*"
            keys = list(self.redis_client.scan_iter(match=pattern, count=100))

            if not keys:
                logger.info("No quota keys found to reset")
                return 0

            # Delete all matching keys
            deleted_count = self.redis_client.delete(*keys)
            logger.info(f"Reset quota data: deleted {deleted_count} keys")
            return deleted_count

        except redis.RedisError as exc:
            logger.error(f"Failed to reset quota data: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to reset quota data. Redis service unavailable.",
            ) from exc

    def reset_user_quotas(self, user_id: str) -> int:
        """Reset quota data for a specific user across all pipelines.

        Args:
            user_id: User identifier

        Returns:
            Number of keys deleted
        """
        try:
            # Find all keys for this user
            pattern = f"{self.namespace}:{user_id}:*"
            keys = list(self.redis_client.scan_iter(match=pattern, count=100))

            if not keys:
                logger.info(f"No quota keys found for user {user_id}")
                return 0

            # Delete all matching keys
            deleted_count = self.redis_client.delete(*keys)
            logger.info(
                f"Reset quota data for user {user_id}: deleted {deleted_count} keys"
            )
            return deleted_count

        except redis.RedisError as exc:
            logger.error(
                f"Failed to reset quota for user {user_id}: {exc}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to reset user quota. Redis service unavailable.",
            ) from exc

    def reset_pipeline_quotas(self, pipeline_name: str) -> int:
        """Reset quota data for a specific pipeline across all users.

        Args:
            pipeline_name: Pipeline identifier

        Returns:
            Number of keys deleted
        """
        try:
            # Find all keys for this pipeline
            pattern = f"{self.namespace}:*:*:{pipeline_name}"
            keys = list(self.redis_client.scan_iter(match=pattern, count=100))

            if not keys:
                logger.info(f"No quota keys found for pipeline {pipeline_name}")
                return 0

            # Delete all matching keys
            deleted_count = self.redis_client.delete(*keys)
            logger.info(
                f"Reset quota data for pipeline {pipeline_name}: "
                f"deleted {deleted_count} keys"
            )
            return deleted_count

        except redis.RedisError as exc:
            logger.error(
                f"Failed to reset quota for pipeline {pipeline_name}: {exc}",
                exc_info=True
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to reset pipeline quota. Redis service unavailable.",
            ) from exc


# ============================================================================
# Singleton Instance
# ============================================================================

_quota_manager: Optional[QuotaManager] = None


def get_quota_manager() -> QuotaManager:
    """Get or create singleton QuotaManager instance.

    This is used as a FastAPI dependency:
        quota_manager: QuotaManager = Depends(get_quota_manager)

    Returns:
        Singleton QuotaManager instance
    """
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = QuotaManager()
        logger.info("Created singleton QuotaManager instance")
    return _quota_manager
