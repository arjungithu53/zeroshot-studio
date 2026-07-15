"""Image fetching helpers for Phase 2 agents."""

import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import unquote, urlparse

import requests

logger = logging.getLogger(__name__)


def _parse_s3_http_url(url: str) -> Optional[Tuple[str, str]]:
    """Return (bucket, key) for common S3 HTTP URL formats."""
    parsed = urlparse(url)
    host_parts = parsed.netloc.split(".")
    path = unquote(parsed.path.lstrip("/"))

    if not path:
        return None

    if len(host_parts) >= 4 and host_parts[1] == "s3":
        return host_parts[0], path

    if parsed.netloc.startswith("s3.") and len(path.split("/", 1)) == 2:
        bucket, key = path.split("/", 1)
        return bucket, key

    return None


def fetch_image_bytes(url_or_path: str, *, timeout: int = 60) -> Optional[bytes]:
    """Fetch raw image bytes from a local path, HTTP URL, or private S3 URL."""
    try:
        if url_or_path.startswith(("http://", "https://")):
            s3_location = _parse_s3_http_url(url_or_path)
            if s3_location:
                try:
                    from app.config import get_s3_client

                    bucket, key = s3_location
                    response = get_s3_client().get_object(Bucket=bucket, Key=key)
                    return response["Body"].read()
                except Exception as s3_error:
                    logger.error(
                        "S3 API fetch failed (bucket=%s key=%s): %s — NOT falling back to HTTP for private bucket",
                        bucket,
                        key,
                        s3_error,
                    )
                    return None

            response = requests.get(url_or_path, timeout=timeout)
            response.raise_for_status()
            return response.content

        return Path(url_or_path).read_bytes()
    except Exception as error:
        logger.error("Failed to fetch image from %s: %s", url_or_path, error)
        return None
