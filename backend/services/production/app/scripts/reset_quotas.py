#!/usr/bin/env python3
"""Script to reset Redis quota data for production service.

Usage:
    python3 -m backend.services.production.app.scripts.reset_quotas
    OR
    cd backend/services/production/app && python3 scripts/reset_quotas.py

This script resets all quota counters in Redis for the production service.
Use with caution as this will allow all users to start fresh with their quotas.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Load environment variables from .env file
from dotenv import load_dotenv

# Find .env file in project root
env_path = backend_dir.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")
else:
    # Try loading from current directory or parent directories
    load_dotenv()
    print("Loaded environment from .env file (if found)")

# Now import quota manager
from services.production.app.core.quota import get_quota_manager


def main():
    """Reset all quota data in Redis for production service."""
    print("=" * 60)
    print("production Quota Reset Tool")
    print("=" * 60)
    print()
    print("This will reset all quota counters for the production service.")
    print("All users will be able to start fresh with their quotas.")
    print()

    # Confirm action
    response = input("Are you sure you want to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return 0

    print()
    print("Resetting all quota data in Redis...")

    try:
        quota_manager = get_quota_manager()
        deleted_count = quota_manager.reset_all_quotas()

        print()
        print("=" * 60)
        print(f"✅ Successfully reset quota data")
        print(f"   Keys deleted: {deleted_count}")
        print("=" * 60)
        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Error resetting quota data:")
        print(f"   {e}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
