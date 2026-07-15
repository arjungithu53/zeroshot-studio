#!/usr/bin/env python3
"""
CLI Tool for Testing Asset Retry Functionality
===============================================
Test the retry mechanism for failed asset generations without a frontend.

Usage:
    python retry_cli.py list <job_id>                    # List failed assets
    python retry_cli.py retry <job_id> <asset_name> <asset_type>  # Retry a specific asset
    python retry_cli.py retry-all <job_id>               # Retry all failed assets
"""

import sys
import requests
import json
from typing import Dict, Any, List

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)



class RetryClient:
    """Client for testing retry functionality via API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize retry client

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
        self.api_prefix = "/api/v1/phase1"

    def list_failed_assets(self, job_id: str) -> Dict[str, Any]:
        """
        List all failed asset generations for a job

        Args:
            job_id: The pipeline job ID

        Returns:
            Dictionary with failed assets information
        """
        url = f"{self.base_url}{self.api_prefix}/failed-assets/{job_id}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching failed assets: {e}")
            if hasattr(e.response, 'text'):
                logger.info(f"   Response: {e.response.text}")
            return None

    def retry_asset(self, job_id: str, asset_name: str, asset_type: str) -> Dict[str, Any]:
        """
        Retry a single failed asset

        Args:
            job_id: The pipeline job ID
            asset_name: Name of the asset to retry
            asset_type: Type of asset (character/location/prop)

        Returns:
            Dictionary with retry result
        """
        url = f"{self.base_url}{self.api_prefix}/retry-asset/{job_id}"
        params = {
            "asset_name": asset_name,
            "asset_type": asset_type
        }

        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrying asset: {e}")
            if hasattr(e.response, 'text'):
                logger.info(f"   Response: {e.response.text}")
            return None

    def retry_all_assets(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Retry all failed assets for a job

        Args:
            job_id: The pipeline job ID

        Returns:
            List of retry results
        """
        # First, get list of failed assets
        failed_data = self.list_failed_assets(job_id)

        if not failed_data or failed_data['failed_count'] == 0:
            logger.error("✓ No failed assets to retry")
            return []

        failed_assets = failed_data['failed_assets']
        logger.error(f"\nRetrying {len(failed_assets)} failed asset(s)...\n")

        results = []
        for asset in failed_assets:
            asset_name = asset['asset_name']
            asset_type = asset['asset_type']

            logger.info(f"   Retrying: {asset_name} ({asset_type})...")
            result = self.retry_asset(job_id, asset_name, asset_type)

            if result:
                if result.get('success'):
                    logger.info(f"   Success!")
                else:
                    logger.error(f"   Failed: {result.get('error')}")

            results.append(result)

        return results

    def print_failed_assets(self, failed_data: Dict[str, Any]) -> None:
        """
        Pretty print failed assets information

        Args:
            failed_data: Dictionary with failed assets info
        """
        if not failed_data:
            return

        logger.info("\n" + "="*60)
        logger.info("FAILED ASSET GENERATIONS")
        logger.info("="*60)

        failed_count = failed_data.get('failed_count', 0)

        if failed_count == 0:
            logger.error("✓ No failed assets!")
            return

        logger.error(f"\nTotal Failed: {failed_count}\n")

        failed_assets = failed_data.get('failed_assets', [])

        # Group by type
        by_type = {
            'character': [],
            'location': [],
            'prop': []
        }

        for asset in failed_assets:
            asset_type = asset.get('asset_type')
            if asset_type in by_type:
                by_type[asset_type].append(asset)

        # Print by type
        for asset_type, assets in by_type.items():
            if not assets:
                continue

            logger.info(f"{asset_type.upper()}S ({len(assets)}):")
            for asset in assets:
                name = asset.get('asset_name')
                reason = asset.get('reason', 'Unknown')
                task_id = asset.get('task_id', 'N/A')
                logger.info(f"  • {name}")
                logger.info(f"    Reason: {reason}")
                if task_id != 'N/A':
                    logger.info(f"    Task ID: {task_id}")
            print()

        logger.info(f"\nTo retry a specific asset:")
        logger.info(f"  python retry_cli.py retry <job_id> \"<asset_name>\" <asset_type>")
        logger.error(f"\nTo retry all failed assets:")
        logger.info(f"  python retry_cli.py retry-all <job_id>")
        logger.info("="*60 + "\n")

    def print_retry_result(self, result: Dict[str, Any]) -> None:
        """
        Pretty print retry result

        Args:
            result: Retry result dictionary
        """
        if not result:
            return

        logger.info("\n" + "="*60)
        logger.info("RETRY RESULT")
        logger.info("="*60)

        success = result.get('success', False)
        asset_name = result.get('asset_name')
        asset_type = result.get('asset_type')

        logger.info(f"\nAsset: {asset_name} ({asset_type})")
        logger.error(f"Status: {'SUCCESS' if success else 'FAILED'}")

        if success:
            images = result.get('images', [])
            task_id = result.get('task_id', 'N/A')
            logger.info(f"Task ID: {task_id}")
            logger.info(f"Images Generated: {len(images)}")

            if images:
                logger.info("\nGenerated Images:")
                for img in images:
                    logger.info(f"  • {img.get('filename')}")
                    logger.info(f"    Path: {img.get('local_path')}")
                    if 's3_url' in img:
                        logger.info(f"    S3: {img.get('s3_url')}")
        else:
            error = result.get('error', 'Unknown error')
            logger.info(f"Error: {error}")

        logger.info("="*60 + "\n")


def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        logger.info(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    client = RetryClient()

    if command == "list":
        if len(sys.argv) < 3:
            logger.info("Usage: python retry_cli.py list <job_id>")
            sys.exit(1)

        job_id = sys.argv[2]
        logger.error(f"\nFetching failed assets for job: {job_id}")

        failed_data = client.list_failed_assets(job_id)
        client.print_failed_assets(failed_data)

    elif command == "retry":
        if len(sys.argv) < 5:
            logger.info("Usage: python retry_cli.py retry <job_id> <asset_name> <asset_type>")
            logger.info("Example: python retry_cli.py retry abc123 \"BLACK LAB PUPPY\" character")
            sys.exit(1)

        job_id = sys.argv[2]
        asset_name = sys.argv[3]
        asset_type = sys.argv[4]

        logger.info(f"\nRetrying asset: {asset_name} ({asset_type})")
        logger.info(f"   Job ID: {job_id}\n")

        result = client.retry_asset(job_id, asset_name, asset_type)
        client.print_retry_result(result)

    elif command == "retry-all":
        if len(sys.argv) < 3:
            logger.info("Usage: python retry_cli.py retry-all <job_id>")
            sys.exit(1)

        job_id = sys.argv[2]
        logger.error(f"\nRetrying all failed assets for job: {job_id}")

        results = client.retry_all_assets(job_id)

        if results:
            logger.info("\n" + "="*60)
            logger.info("SUMMARY")
            logger.info("="*60)

            success_count = sum(1 for r in results if r and r.get('success'))
            failed_count = len(results) - success_count

            logger.info(f"\nTotal Retries: {len(results)}")
            logger.info(f"Successful: {success_count}")
            logger.error(f"Failed: {failed_count}")
            logger.info("="*60 + "\n")

    else:
        logger.info(f"Unknown command: {command}")
        logger.info(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
