"""
S3 streaming utilities.

Provides:
- parse_s3_url()          — extract (bucket, key) from an S3 URL or presigned URL
- stream_bytes_from_s3()  — download object bytes from S3
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def parse_s3_url(
    url: str,
    default_bucket: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parse an S3 URL into (bucket, key).

    Supports:
    - s3://bucket/key
    - https://bucket.s3.region.amazonaws.com/key
    - https://s3.region.amazonaws.com/bucket/key
    - Presigned URLs (https://bucket.s3.*.amazonaws.com/key?X-Amz-*)

    Args:
        url:            S3 URL or presigned URL string.
        default_bucket: Fallback bucket name if it cannot be parsed from the URL.

    Returns:
        (bucket, key) tuple.

    Raises:
        ValueError: If neither the URL nor default_bucket provides a bucket name.
    """
    if url.startswith("s3://"):
        parsed = urlparse(url)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    parsed = urlparse(url)
    host = parsed.netloc  # e.g. "mybucket.s3.eu-north-1.amazonaws.com"

    # Pattern: <bucket>.s3[.region].amazonaws.com
    virtual_hosted = re.match(r"^(.+?)\.s3[.\-]", host)
    if virtual_hosted:
        bucket = virtual_hosted.group(1)
        key = parsed.path.lstrip("/")
        return bucket, key

    # Pattern: s3[.region].amazonaws.com/<bucket>/key
    path_parts = parsed.path.lstrip("/").split("/", 1)
    if len(path_parts) == 2:
        return path_parts[0], path_parts[1]
    if len(path_parts) == 1 and path_parts[0]:
        if default_bucket:
            return default_bucket, path_parts[0]
        return path_parts[0], ""

    if default_bucket:
        return default_bucket, parsed.path.lstrip("/")

    raise ValueError(f"Cannot parse bucket from URL: {url}")


def stream_bytes_from_s3(
    url: str,
    s3_client: Any,
    default_bucket: Optional[str] = None,
) -> bytes:
    """
    Download an S3 object and return its bytes.

    Args:
        url:            S3 URL or presigned URL.
        s3_client:      Initialised boto3 S3 client.
        default_bucket: Fallback bucket when it cannot be parsed from URL.

    Returns:
        Raw bytes of the object body.
    """
    bucket, key = parse_s3_url(url, default_bucket=default_bucket)
    logger.debug("Streaming s3://%s/%s", bucket, key)
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()
