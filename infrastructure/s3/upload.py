"""
S3 upload utilities.

Provides:
- upload_file()        — low-level single-file upload used by production config wrapper
- S3ImageUploader      — class-based uploader used by Phase-2 agent_7
"""
from __future__ import annotations

import logging
import mimetypes
import os
import uuid
from typing import Any, Optional

import boto3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Functional upload_file helper
# ---------------------------------------------------------------------------

def upload_file(
    file_path: str,
    s3_client: Any,
    bucket_name: str,
    s3_key: Optional[str] = None,
    make_public: bool = False,
    content_type: Optional[str] = None,
    region: str = "eu-north-1",
    endpoint_url: Optional[str] = None,
    use_presigned_url: bool = True,
    presigned_expiration: int = 86400,
) -> str:
    """
    Upload a local file to S3 and return either a presigned URL or a public URL.

    Args:
        file_path:            Path to local file.
        s3_client:            Initialised boto3 S3 client.
        bucket_name:          Target bucket.
        s3_key:               Destination key; defaults to basename of file_path.
        make_public:          Set ACL to public-read (ignored when use_presigned_url=True).
        content_type:         Explicit MIME type; auto-detected when omitted.
        region:               AWS region (used to build public URL).
        endpoint_url:         Custom endpoint for S3-compatible services.
        use_presigned_url:    Return a time-limited presigned URL (recommended).
        presigned_expiration: Lifetime of the presigned URL in seconds.

    Returns:
        URL string for the uploaded object.
    """
    if s3_key is None:
        s3_key = os.path.basename(file_path)

    if content_type is None:
        guessed, _ = mimetypes.guess_type(file_path)
        content_type = guessed or "application/octet-stream"

    extra_args: dict = {"ContentType": content_type}
    if make_public and not use_presigned_url:
        extra_args["ACL"] = "public-read"

    s3_client.upload_file(file_path, bucket_name, s3_key, ExtraArgs=extra_args)
    logger.info("Uploaded '%s' → s3://%s/%s", file_path, bucket_name, s3_key)

    if use_presigned_url:
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": s3_key},
            ExpiresIn=presigned_expiration,
        )
    elif endpoint_url:
        url = f"{endpoint_url.rstrip('/')}/{bucket_name}/{s3_key}"
    else:
        url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{s3_key}"

    return url


# ---------------------------------------------------------------------------
# Class-based uploader (used by Phase-2 agent_7_shot_editor)
# ---------------------------------------------------------------------------

class S3ImageUploader:
    """
    Convenience wrapper around boto3 for uploading images to a fixed bucket.

    Usage (mirrors original interface expected by agent_7):
        uploader = S3ImageUploader(bucket_name="my-bucket")
        url = uploader.upload_image(local_path, s3_key="shots/v1/shot_001.jpg")
    """

    def __init__(
        self,
        bucket_name: str,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.region = region or os.getenv("production_AWS_REGION", "eu-north-1")
        _access_key = access_key_id or os.getenv("production_AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID", "")
        _secret_key = secret_access_key or os.getenv("production_AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        _endpoint = endpoint_url or os.getenv("production_AWS_ENDPOINT_URL")

        kwargs: dict = {
            "service_name": "s3",
            "region_name": self.region,
            "aws_access_key_id": _access_key,
            "aws_secret_access_key": _secret_key,
        }
        if _endpoint:
            kwargs["endpoint_url"] = _endpoint

        self._client = boto3.client(**kwargs)
        logger.info("S3ImageUploader initialised for bucket '%s'", bucket_name)

    def upload_image(
        self,
        file_path: str,
        s3_key: Optional[str] = None,
        content_type: Optional[str] = None,
        use_presigned_url: bool = True,
        presigned_expiration: int = 86400,
    ) -> str:
        """Upload image file and return URL."""
        if s3_key is None:
            ext = os.path.splitext(file_path)[1] or ".jpg"
            s3_key = f"images/{uuid.uuid4().hex}{ext}"

        return upload_file(
            file_path=file_path,
            s3_client=self._client,
            bucket_name=self.bucket_name,
            s3_key=s3_key,
            content_type=content_type,
            region=self.region,
            use_presigned_url=use_presigned_url,
            presigned_expiration=presigned_expiration,
        )

    def upload_bytes(
        self,
        data: bytes,
        s3_key: str,
        content_type: str = "image/jpeg",
        use_presigned_url: bool = True,
        presigned_expiration: int = 86400,
    ) -> str:
        """Upload raw bytes directly (no temp file required)."""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_key)[1] or ".bin") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            return self.upload_image(
                file_path=tmp_path,
                s3_key=s3_key,
                content_type=content_type,
                use_presigned_url=use_presigned_url,
                presigned_expiration=presigned_expiration,
            )
        finally:
            os.unlink(tmp_path)
