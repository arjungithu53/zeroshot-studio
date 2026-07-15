"""CSV parser utility for movie scene uploads."""

import csv
import io
import re
from typing import List, Dict, Any, Tuple, Optional
from fastapi import UploadFile, HTTPException

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.services.production.app.utils.name_normalization import normalize_asset_name, normalize_list, parse_location_with_variation

# Initialize logger for this module
logger = get_logger(__name__)


class SceneData:
    """Data class for scene information."""
    def __init__(self, scene_number: int, scene_name: str, script: str, shotlist: str = ""):
        self.scene_number = scene_number
        self.scene_name = scene_name
        self.script = script
        self.shotlist = shotlist

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_number": self.scene_number,
            "scene_name": self.scene_name,
            "script": self.script,
            "shotlist": self.shotlist
        }


async def parse_movie_csv(csv_file: UploadFile) -> List[SceneData]:
    """
    Parse uploaded CSV file containing movie scenes.

    Expected CSV format:
    - Column 1: scene_number (integer)
    - Column 2: scene_name (string)
    - Column 3: scene_script (string)
    - Column 4: shotlist (string, optional)

    Args:
        csv_file: Uploaded CSV file

    Returns:
        List of SceneData objects

    Raises:
        HTTPException: If CSV format is invalid
    """
    try:
        # Read file content
        content = await csv_file.read()
        content_str = content.decode('utf-8')

        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(content_str))

        # Validate headers
        required_headers = {'scene_number', 'scene_name', 'scene_script'}
        headers = set(csv_reader.fieldnames or [])

        if not required_headers.issubset(headers):
            missing = required_headers - headers
            raise HTTPException(
                status_code=400,
                detail=f"CSV missing required columns: {', '.join(missing)}. Required: scene_number, scene_name, scene_script"
            )

        scenes = []
        for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 (1 is header)
            try:
                # Parse scene number
                scene_number = int(row['scene_number'])
                if scene_number <= 0:
                    raise ValueError("Scene number must be positive")

                # Parse scene name
                scene_name = row['scene_name'].strip()
                if not scene_name:
                    raise ValueError("Scene name cannot be empty")

                # Parse script
                script = row['scene_script'].strip()
                if not script:
                    raise ValueError("Scene script cannot be empty")

                # Parse shotlist (optional)
                shotlist = row.get('shotlist', '').strip()

                # Create scene data
                scene = SceneData(
                    scene_number=scene_number,
                    scene_name=scene_name,
                    script=script,
                    shotlist=shotlist
                )
                scenes.append(scene)

            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid data in row {row_num}: {str(e)}"
                )

        # Validate at least one scene
        if not scenes:
            raise HTTPException(
                status_code=400,
                detail="CSV file must contain at least one scene"
            )

        # Validate scene numbers are unique and sequential (optional, but recommended)
        scene_numbers = [s.scene_number for s in scenes]
        if len(scene_numbers) != len(set(scene_numbers)):
            raise HTTPException(
                status_code=400,
                detail="Scene numbers must be unique"
            )

        # Sort by scene number
        scenes.sort(key=lambda s: s.scene_number)

        logger.info(f"Successfully parsed {len(scenes)} scenes from CSV")
        return scenes

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid file encoding. Please use UTF-8 encoded CSV file"
        )
    except csv.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV parsing error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing CSV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse CSV file: {str(e)}"
        )


def combine_scripts_for_phase1(scenes: List[SceneData]) -> str:
    """
    Combine all scene scripts into a single string for Phase 1 processing.

    Args:
        scenes: List of SceneData objects

    Returns:
        Combined script text with scene markers
    """
    combined_script = ""

    for scene in scenes:
        combined_script += f"=== SCENE {scene.scene_number}: {scene.scene_name} ===\n\n"
        combined_script += scene.script
        combined_script += "\n\n"

    return combined_script.strip()


def validate_csv_format(csv_file: UploadFile) -> None:
    """
    Quick validation of CSV format before processing.

    Args:
        csv_file: Uploaded CSV file

    Raises:
        HTTPException: If file format is invalid
    """
    # Check file extension
    if not csv_file.filename or not csv_file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file (.csv extension)"
        )

    # Check file size (max 10MB)
    if hasattr(csv_file, 'size') and csv_file.size:
        max_size = 10 * 1024 * 1024  # 10MB
        if csv_file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size of 10MB"
            )


class ShotData:
    """Data class for shot information from shotlist CSV."""
    def __init__(self, scene_number: int, shot_number: str, shot_type: str,
                 camera_movement: str, description: str, characters: List[str] = None,
                 locations: str = "", product_present: bool = False):
        self.scene_number = scene_number
        self.shot_number = shot_number
        self.shot_type = shot_type
        self.camera_movement = camera_movement
        self.description = description
        self.characters = characters if characters is not None else []
        self.locations = locations
        self.product_present = product_present

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scene_number": self.scene_number,
            "shot_number": self.shot_number,
            "shot_type": self.shot_type,
            "camera_movement": self.camera_movement,
            "description": self.description,
            "characters": self.characters,
            "locations": self.locations,
            "product_present": self.product_present
        }


def generate_shot_id(scene_number: int, shot_number: str) -> str:
    """
    Generate shot ID from scene and shot numbers.
    Examples: 1.1 -> S01E01_001, 1.2 -> S01E01_002, 2.5 -> S01E02_005
    1.1A -> S01E01_001A, 1.1B -> S01E01_001B

    Args:
        scene_number: Scene number (e.g., 1, 2, 3)
        shot_number: Shot number (e.g., "1.1", "1.2", "1.1A")

    Returns:
        Formatted shot ID
    """
    # Parse shot number (e.g., "1.1", "1.1A", "2.5B")
    match = re.match(r'^(\d+)\.(\d+)([A-Z]?)$', shot_number)
    if not match:
        raise ValueError(f"Invalid shot number format: {shot_number}")

    shot_scene = int(match.group(1))
    shot_seq = int(match.group(2))
    shot_suffix = match.group(3)

    # Validate scene number matches
    if shot_scene != scene_number:
        raise ValueError(
            f"Shot number {shot_number} does not match scene number {scene_number}"
        )

    # Format: S01E{scene:02d}_{seq:03d}{suffix}
    shot_id = f"S01E{scene_number:02d}_{shot_seq:03d}{shot_suffix}"
    return shot_id


def parse_shot_number(shot_number: str) -> Tuple[int, int, str]:
    """
    Parse shot number into components.

    Args:
        shot_number: Shot number string (e.g., "1.1", "1.2", "1.1A")

    Returns:
        Tuple of (scene_num, sequence_num, suffix)

    Raises:
        ValueError: If format is invalid
    """
    match = re.match(r'^(\d+)\.(\d+)([A-Z]?)$', shot_number)
    if not match:
        raise ValueError(f"Invalid shot number format: {shot_number}")

    scene_num = int(match.group(1))
    seq_num = int(match.group(2))
    suffix = match.group(3)

    return scene_num, seq_num, suffix


def validate_scene_sequence(scene_numbers: List[int]) -> Optional[str]:
    """
    Validate that scene numbers are sequential (1, 2, 3...) with no gaps.

    Args:
        scene_numbers: List of scene numbers

    Returns:
        Error message if invalid, None if valid
    """
    if not scene_numbers:
        return "No scenes found"

    sorted_scenes = sorted(scene_numbers)

    # Check if starts at 1
    if sorted_scenes[0] != 1:
        return f"Scene numbers must start at 1 (found: {sorted_scenes[0]})"

    # Check for sequential numbering (no gaps)
    for i in range(len(sorted_scenes)):
        expected = i + 1
        if sorted_scenes[i] != expected:
            return f"Scene numbers must be sequential - expected {expected}, found {sorted_scenes[i]}"

    return None


def validate_shot_sequence(shots: List[ShotData]) -> Optional[str]:
    """
    Validate that shot numbers are sequential within each scene.
    Supports formats like: 1.1, 1.2, 1.1A, 1.1B, etc.

    Args:
        shots: List of ShotData objects

    Returns:
        Error message if invalid, None if valid
    """
    if not shots:
        return "No shots found"

    # Group shots by scene
    shots_by_scene: Dict[int, List[ShotData]] = {}
    for shot in shots:
        if shot.scene_number not in shots_by_scene:
            shots_by_scene[shot.scene_number] = []
        shots_by_scene[shot.scene_number].append(shot)

    # Validate each scene's shots
    for scene_num, scene_shots in shots_by_scene.items():
        # Sort by shot number for validation
        try:
            scene_shots.sort(key=lambda s: (
                parse_shot_number(s.shot_number)[1],  # sequence number
                parse_shot_number(s.shot_number)[2]   # suffix (A, B, etc.)
            ))
        except ValueError as e:
            return f"Scene {scene_num}: {str(e)}"

        prev_seq = 0
        prev_suffix = ""

        for shot in scene_shots:
            try:
                shot_scene, shot_seq, suffix = parse_shot_number(shot.shot_number)

                # Validate scene number matches
                if shot_scene != scene_num:
                    return f"Shot {shot.shot_number} has mismatched scene number (expected {scene_num})"

                # Validate sequence is increasing or same with suffix
                if shot_seq < prev_seq:
                    return f"Scene {scene_num}: Shot numbers not in order ({shot.shot_number} after {scene_shots[scene_shots.index(shot)-1].shot_number})"

                # If same sequence number, must have suffix progression
                if shot_seq == prev_seq:
                    if not suffix:
                        return f"Scene {scene_num}: Shot {shot.shot_number} has same sequence as previous but no suffix"
                    if prev_suffix and suffix <= prev_suffix:
                        return f"Scene {scene_num}: Shot suffixes not in order ({suffix} after {prev_suffix})"

                # If new sequence number, it should be prev + 1 (strict sequential)
                if shot_seq > prev_seq and not prev_suffix:
                    if shot_seq != prev_seq + 1:
                        return f"Scene {scene_num}: Gap in shot sequence - expected {scene_num}.{prev_seq + 1}, found {shot.shot_number}"

                prev_seq = shot_seq
                prev_suffix = suffix if shot_seq == prev_seq else ""

            except ValueError as e:
                return f"Scene {scene_num}: {str(e)}"

    return None


def validate_required_fields(shots: List[ShotData]) -> Optional[str]:
    """
    Validate that all required fields are present and non-empty.

    Args:
        shots: List of ShotData objects

    Returns:
        Error message if invalid, None if valid
    """
    for i, shot in enumerate(shots, start=2):  # Start at 2 (row 1 is header)
        if not shot.shot_type or not shot.shot_type.strip():
            return f"Row {i}: Missing shot_type for shot {shot.shot_number}"

        if not shot.camera_movement or not shot.camera_movement.strip():
            return f"Row {i}: Missing camera_movement for shot {shot.shot_number}"

        if not shot.description or not shot.description.strip():
            return f"Row {i}: Missing description for shot {shot.shot_number}"
        
        # Note: characters and locations are optional now, so we don't validate them here

    return None


def validate_scene_alignment(script_scenes: List[SceneData], shotlist_shots: List[ShotData]) -> Optional[str]:
    """
    Validate that all scenes in shotlist exist in script.

    Args:
        script_scenes: List of SceneData from script CSV
        shotlist_shots: List of ShotData from shotlist CSV

    Returns:
        Error message if invalid, None if valid
    """
    script_scene_numbers = {s.scene_number for s in script_scenes}
    shotlist_scene_numbers = {s.scene_number for s in shotlist_shots}

    # Check if shotlist has scenes not in script
    missing_in_script = shotlist_scene_numbers - script_scene_numbers
    if missing_in_script:
        missing_sorted = sorted(missing_in_script)
        return f"Shotlist contains scenes not in script: {', '.join(map(str, missing_sorted))}"

    return None


async def parse_shotlist_csv(csv_file: UploadFile) -> List[ShotData]:
    """
    Parse uploaded shotlist CSV file.

    Expected CSV format:
    - Column 1: scene_number (integer)
    - Column 2: shot_number (string, e.g., "1.1", "1.2", "1.1A")
    - Column 3: shot_type (string)
    - Column 4: camera_movement (string)
    - Column 5: description (string)
    - Column 6: characters (string, optional)
    - Column 7: locations (string, optional)

    Args:
        csv_file: Uploaded CSV file

    Returns:
        List of ShotData objects

    Raises:
        HTTPException: If CSV format is invalid
    """
    try:
        # Read file content
        content = await csv_file.read()
        content_str = content.decode('utf-8')

        # Parse CSV
        csv_reader = csv.DictReader(io.StringIO(content_str))

        # Validate headers
        required_headers = {'scene_number', 'shot_number', 'shot_type', 'camera_movement', 'description'}
        # Optional headers: characters, locations
        
        headers = set(csv_reader.fieldnames or [])

        if not required_headers.issubset(headers):
            missing = required_headers - headers
            raise HTTPException(
                status_code=400,
                detail=f"Shotlist CSV missing required columns: {', '.join(missing)}. Required: scene_number, shot_number, shot_type, camera_movement, description"
            )

        shots = []
        for row_num, row in enumerate(csv_reader, start=2):  # Start at 2 (1 is header)
            try:
                # Parse scene number
                scene_number = int(row['scene_number'])
                if scene_number <= 0:
                    raise ValueError("Scene number must be positive")

                # Parse shot number
                shot_number = row['shot_number'].strip()
                if not shot_number:
                    raise ValueError("Shot number cannot be empty")

                # Validate shot number format
                try:
                    parse_shot_number(shot_number)
                except ValueError as e:
                    raise ValueError(f"Invalid shot number format '{shot_number}': {str(e)}")

                # Parse shot type
                shot_type = row['shot_type'].strip()
                if not shot_type:
                    raise ValueError("Shot type cannot be empty")

                # Parse camera movement
                camera_movement = row['camera_movement'].strip()
                if not camera_movement:
                    raise ValueError("Camera movement cannot be empty")

                # Parse description
                description = row['description'].strip()
                if not description:
                    raise ValueError("Description cannot be empty")
                    
                # Parse optional fields
                characters_str = row.get('characters', '').strip()
                if characters_str:
                    characters = [c.strip() for c in characters_str.split(',') if c.strip()]
                else:
                    characters = []
                    
                locations = row.get('locations', '').strip()

                # Parse optional product_present field (new CSV format; defaults False if absent)
                product_present_str = row.get('product_present', '').strip().lower()
                product_present = product_present_str == 'yes'

                # Create shot data
                shot = ShotData(
                    scene_number=scene_number,
                    shot_number=shot_number,
                    shot_type=shot_type,
                    camera_movement=camera_movement,
                    description=description,
                    characters=characters,
                    locations=locations,
                    product_present=product_present
                )
                shots.append(shot)

            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid data in row {row_num}: {str(e)}"
                )

        # Validate at least one shot
        if not shots:
            raise HTTPException(
                status_code=400,
                detail="Shotlist CSV must contain at least one shot"
            )

        logger.info(f"Successfully parsed {len(shots)} shots from shotlist CSV")
        return shots

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="Invalid file encoding. Please use UTF-8 encoded shotlist CSV file"
        )
    except csv.Error as e:
        raise HTTPException(
            status_code=400,
            detail=f"Shotlist CSV parsing error: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing shotlist CSV: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse shotlist CSV file: {str(e)}"
        )


def extract_unique_entities_from_shotlist(shots: List[ShotData]) -> Dict[str, Any]:
    """
    Extract unique characters and locations from shotlist CSV.

    This function pre-processes the shotlist to identify all unique characters and
    locations before Phase 1 asset generation runs. This ensures Agent 1 can use
    CSV entity names as the source of truth.

    Args:
        shots: List of ShotData objects from parsed shotlist CSV

    Returns:
        Dictionary containing:
        - unique_characters: List[str] - All unique character names (normalized to UPPERCASE_WITH_UNDERSCORES)
        - unique_locations: List[str] - All unique location names (normalized to UPPERCASE_WITH_UNDERSCORES)
        - character_shots: Dict[str, List[str]] - Maps each character to list of shot numbers where they appear
        - location_shots: Dict[str, List[str]] - Maps each location to list of shot numbers where it appears
        - has_entity_data: bool - True if CSV contains character/location data, False for backward compatibility

    Example:
        >>> shots = [
        ...     ShotData(1, "1.1", "close_up", "static", "Alex walks", characters=["Alex"], locations="Jungle"),
        ...     ShotData(1, "1.2", "wide", "pan", "Sarah runs", characters=["Sarah", "Alex"], locations="Jungle")
        ... ]
        >>> result = extract_unique_entities_from_shotlist(shots)
        >>> result['unique_characters']
        ['ALEX', 'SARAH']
        >>> result['unique_locations']
        ['JUNGLE']
        >>> result['character_shots']['ALEX']
        ['1.1', '1.2']
    """
    if not shots:
        return {
            "unique_characters": [],
            "unique_locations": [],
            "character_shots": {},
            "location_shots": {},
            "has_entity_data": False,
            "has_product_shots": False,
            "product_shot_numbers": []
        }

    characters_map = {}  # Maps normalized character name to list of shot numbers
    locations_map = {}   # Maps normalized location name to list of shot numbers
    has_any_entity_data = False
    product_shot_numbers = []

    for shot in shots:
        shot_number = shot.shot_number

        # Process characters (list of names)
        if shot.characters:
            has_any_entity_data = True
            for character in shot.characters:
                if character and character.strip():
                    normalized_name = normalize_asset_name(character)
                    if normalized_name not in characters_map:
                        characters_map[normalized_name] = []
                    characters_map[normalized_name].append(shot_number)

        # Process location (single name)
        if shot.locations and shot.locations.strip():
            has_any_entity_data = True
            # Parse location to extract base name (e.g., "JUNGLE_NORTH" -> "JUNGLE")
            base_location, variation_angle = parse_location_with_variation(shot.locations)
            if base_location and base_location not in locations_map:
                locations_map[base_location] = []
            if base_location:
                locations_map[base_location].append(shot_number)

        # Track shots that feature the product
        if shot.product_present:
            product_shot_numbers.append(shot_number)

    # Sort for consistent ordering
    unique_characters = sorted(characters_map.keys())
    unique_locations = sorted(locations_map.keys())

    logger.info(f"Extracted CSV entities: {len(unique_characters)} characters, {len(unique_locations)} locations")
    if has_any_entity_data:
        logger.info(f"  Characters: {', '.join(unique_characters)}")
        logger.info(f"  Locations: {', '.join(unique_locations)}")
    if product_shot_numbers:
        logger.info(f"  Product shots ({len(product_shot_numbers)}): {', '.join(product_shot_numbers)}")

    return {
        "unique_characters": unique_characters,
        "unique_locations": unique_locations,
        "character_shots": characters_map,
        "location_shots": locations_map,
        "has_entity_data": has_any_entity_data,
        "has_product_shots": len(product_shot_numbers) > 0,
        "product_shot_numbers": product_shot_numbers
    }
