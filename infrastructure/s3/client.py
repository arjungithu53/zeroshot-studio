"""
S3 client factory for infrastructure layer.
Provides S3Config and S3ClientFactory with boto3 under the hood.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

import boto3

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    region: str
    endpoint_url: Optional[str] = None


class S3ClientFactory:
    """
    Thread-safe singleton-friendly S3 client factory.
    """

    def __init__(self, config: S3Config) -> None:
        self.config = config
        self._client: Optional[Any] = None
        self._lock = threading.Lock()

    def _build_client(self) -> Any:
        kwargs: dict = {
            "service_name": "s3",
            "region_name": self.config.region,
            "aws_access_key_id": self.config.access_key_id,
            "aws_secret_access_key": self.config.secret_access_key,
            # Use regional endpoint so presigned URLs don't break on redirect
            "endpoint_url": self.config.endpoint_url or f"https://s3.{self.config.region}.amazonaws.com",
        }

        client = boto3.client(**kwargs)
        logger.info(
            "S3 client created for bucket '%s' in region '%s'",
            self.config.bucket_name,
            self.config.region,
        )
        return client

    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._build_client()
        return self._client

    def get_client(self) -> Any:
        return self._ensure_client()

    def get_bucket_name(self) -> str:
        return self.config.bucket_name
