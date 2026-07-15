"""S3 helper utilities for uploading scene scripts and shotlist JSONs."""

import json
import tempfile
from typing import Dict, Any, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from app.config import upload_file_wrapper
from app.utils.csv_parser import ShotData, generate_shot_id, parse_shot_number

logger = get_logger(__name__)


def upload_scene_script_to_s3(
    movie_id: str,
    scene_number: int,
    script_text: str
) -> str:
    """
    Upload scene script as a text file to S3.

    Args:
        movie_id: Movie ID
        scene_number: Scene number
        script_text: Full script text for the scene

    Returns:
        S3 URL (or presigned URL) of the uploaded script

    Raises:
        Exception: If upload fails
    """
    try:
        # Create temporary file with script content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(script_text)
            temp_file_path = temp_file.name

        # Generate S3 key
        s3_key = f"movies/{movie_id}/scenes/scene_{scene_number:02d}_script.txt"

        # Upload to S3
        url = upload_file_wrapper(
            file_path=temp_file_path,
            s3_key=s3_key,
            content_type='text/plain',
            use_presigned_url=False  # Store permanent S3 URL
        )

        logger.info(f"Uploaded scene {scene_number} script to S3: {s3_key}")

        # Clean up temp file
        Path(temp_file_path).unlink()

        return url

    except Exception as e:
        logger.error(f"Failed to upload scene script to S3: {e}")
        raise Exception(f"Failed to upload scene script to S3: {str(e)}")


def create_shotlist_json(
    movie_id: str,
    scene_number: int,
    scene_name: str,
    scene_script: str,
    shots_data: List[ShotData],
    project_id: str = ""
) -> Dict[str, Any]:
    """
    Create shotlist JSON structure from shot data.

    Args:
        movie_id: Movie ID
        scene_number: Scene number
        scene_name: Scene name
        scene_script: Full scene script
        shots_data: List of ShotData objects for this scene
        project_id: production project ID (can be empty initially)

    Returns:
        Dictionary with shotlist structure matching the required format
    """
    # Create shots array with generated shot IDs
    shots_array = []
    for shot in shots_data:
        try:
            # Parse shot number to get sequence
            _, sequence_num, _ = parse_shot_number(shot.shot_number)

            # Generate shot ID
            shot_id = generate_shot_id(scene_number, shot.shot_number)

            shot_dict = {
                "shot_id": shot_id,
                "description": shot.description,
                "scene_number": scene_number,
                "sequence_number": sequence_num,
                "shot_style": shot.shot_type,
                "camera_movement": shot.camera_movement,
                "characters": shot.characters,
                "locations": shot.locations,
                "product_present": shot.product_present,
                "duration": None,
                "source_type": None,
                "optimized_ai_notes": None
            }
            shots_array.append(shot_dict)

        except Exception as e:
            logger.error(f"Error processing shot {shot.shot_number}: {e}")
            raise

    # Create the full shotlist JSON structure
    shotlist_json = {
        "shot_list": {
            "episode_id": f"E{scene_number:02d}",
            "title": scene_name,
            "scene_description": scene_script,
            "shots": shots_array
        },
        "show_id": movie_id,
        "episode_number": scene_number,
        "project_id": project_id,
        "scene_description": scene_name
    }

    return shotlist_json


def upload_shotlist_json_to_s3(
    movie_id: str,
    scene_number: int,
    shotlist_json: Dict[str, Any]
) -> str:
    """
    Upload shotlist JSON to S3.

    Args:
        movie_id: Movie ID
        scene_number: Scene number
        shotlist_json: Shotlist JSON dictionary

    Returns:
        S3 URL (or presigned URL) of the uploaded JSON

    Raises:
        Exception: If upload fails
    """
    try:
        # Create temporary file with JSON content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(shotlist_json, temp_file, indent=2)
            temp_file_path = temp_file.name

        # Generate S3 key
        s3_key = f"movies/{movie_id}/scenes/scene_{scene_number:02d}_shotlist.json"

        # Upload to S3
        url = upload_file_wrapper(
            file_path=temp_file_path,
            s3_key=s3_key,
            content_type='application/json',
            use_presigned_url=False  # Store permanent S3 URL
        )

        logger.info(f"Uploaded scene {scene_number} shotlist to S3: {s3_key}")

        # Clean up temp file
        Path(temp_file_path).unlink()

        return url

    except Exception as e:
        logger.error(f"Failed to upload shotlist JSON to S3: {e}")
        raise Exception(f"Failed to upload shotlist JSON to S3: {str(e)}")
