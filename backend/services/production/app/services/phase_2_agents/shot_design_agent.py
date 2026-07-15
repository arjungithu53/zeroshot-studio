"""
Agent 12: Shot Design Agent
A critical AI "Cinematographer" that analyzes shot requirements and selects
the correct assets for composition, ensuring scene consistency and technical feasibility.
"""


import sys
from pathlib import Path
import os

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import google.generativeai as genai

from .helpers.asset_library import AssetLibrary, AssetInfo, ANGLE_FALLBACKS
from .helpers.prompt_feasibility_checker import PromptFeasibilityChecker
from .helpers.consistency_checker import ConsistencyChecker
from .helpers.gemini_client import GeminiClient
from backend.services.production.app.utils.name_normalization import normalize_asset_name

GOOGLE_API_KEY = os.getenv("SHARED_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(
        api_key=GOOGLE_API_KEY,
        transport="rest"  # Avoid gRPC entirely
    )
else:
    logger.warning("GOOGLE_API_KEY not found; google.generativeai not configured for ShotDesignAgent.")



@dataclass
class ShotDesignOutput:
    """Output from Agent 12 for a single shot"""
    shot_id: str
    generation_strategy: str
    feasibility_score: float
    selected_assets: List[Dict]
    model_recommendation: str
    prompt: str  # Prompt from Agent 11 (not modified by Agent 12)
    composition_strategy: Dict
    warnings: List[str]
    metadata: Dict


class ShotDesignAgent:
    """Agent 12: Cinematographer for shot composition"""

    def __init__(self, asset_library: AssetLibrary, use_feasibility_check: bool = True):
        self.asset_library = asset_library
        self.shot_history: List[ShotDesignOutput] = []
        self.use_feasibility_check = use_feasibility_check
        self.scene_baseline = None  # Store scene baseline for consistency
        self.gemini_client = GeminiClient() if self.use_feasibility_check else None

        # Initialize checkers if enabled
        if self.use_feasibility_check:
            self.feasibility_checker = PromptFeasibilityChecker(
                gemini_client=self.gemini_client
            )
            self.consistency_checker = ConsistencyChecker(
                gemini_client=self.gemini_client
            )
        else:
            self.feasibility_checker = None
            self.consistency_checker = None

    def _get_assets_for_shot(self, shot) -> Dict[str, List[Dict]]:
        """
        Extract and fetch assets from AssetLibrary for a given shot.

        Uses characters and locations from CSV fields for targeted asset fetching,
        while props are still detected from the description text.

        Args:
            shot: Shot object with characters, locations, and description fields

        Returns:
            Dictionary with 'characters', 'locations', and 'props' lists containing asset info
        """
        if not self.asset_library:
            logger.warning("No AssetLibrary available - cannot fetch assets")
            return {'characters': [], 'locations': [], 'props': []}

        assets_info = {
            'characters': [],
            'locations': [],
            'props': []
        }

        # 1. CHARACTERS: Use CSV field (shot.characters)
        if hasattr(shot, 'characters') and shot.characters:
            logger.info(f"Fetching assets for characters from CSV: {shot.characters}")
            for char_name in shot.characters:
                # Normalize the name to match asset library naming (UPPERCASE_WITH_UNDERSCORES)
                normalized_name = normalize_asset_name(char_name)

                # Get all assets for this character
                char_assets = self.asset_library.get_all_assets_for_character(normalized_name)
                if char_assets:
                    assets_info['characters'].append({
                        'name': normalized_name,
                        'assets': [
                            {
                                'angle': asset.angle,
                                'local_path': asset.local_path,
                                'url': asset.url,
                                'prompt': asset.prompt,
                                'confidence': asset.confidence,
                                'type': 'character'
                            }
                            for asset in char_assets
                        ]
                    })
                else:
                    logger.warning(f"No assets found for character: {normalized_name}")

        # 2. LOCATIONS: Use CSV field (shot.locations)
        if hasattr(shot, 'locations') and shot.locations:
            logger.info(f"Fetching assets for location from CSV: {shot.locations}")

            # Check if location has direction suffix (e.g., "JUNGLE_NORTH")
            location_parts = shot.locations.split('_')
            direction_keywords = ['north', 'south', 'east', 'west']

            # Check if the last part is a direction
            if len(location_parts) > 1 and location_parts[-1].lower() in direction_keywords:
                # Location has direction
                direction = location_parts[-1]
                base_location = '_'.join(location_parts[:-1])  # Everything except the last part
                normalized_name = normalize_asset_name(base_location)

                logger.info(f"Location with direction detected: {base_location} in direction {direction}")

                # Get specific direction asset
                loc_asset = self.asset_library.get_location_asset_by_direction(normalized_name, direction)
                if loc_asset:
                    assets_info['locations'].append({
                        'name': normalized_name,
                        'direction': direction,
                        'assets': [
                            {
                                'angle': loc_asset.angle,
                                'local_path': loc_asset.local_path,
                                'url': loc_asset.url,
                                'prompt': loc_asset.prompt,
                                'confidence': loc_asset.confidence,
                                'type': 'location'
                            }
                        ]
                    })
                    logger.info(f"Found asset for location {normalized_name} in direction {direction}: {loc_asset.url}")
                else:
                    logger.warning(f"No asset found for location {normalized_name} with direction {direction}")
            else:
                # No direction specified, use original behavior (get all assets)
                normalized_name = normalize_asset_name(shot.locations)

                # Get all assets for this location
                loc_assets = self.asset_library.get_all_assets_for_character(normalized_name)  # Uses same method
                if loc_assets:
                    assets_info['locations'].append({
                        'name': normalized_name,
                        'assets': [
                            {
                                'angle': asset.angle,
                                'local_path': asset.local_path,
                                'url': asset.url,
                                'prompt': asset.prompt,
                                'confidence': asset.confidence,
                                'type': 'location'
                            }
                            for asset in loc_assets
                        ]
                    })
                else:
                    logger.warning(f"No assets found for location: {normalized_name}")

        # 3. PROPS: Search through description text (NOT from CSV)
        # Props are not in CSV, so we still detect them from description
        description = shot.description.lower()
        available_props = self.asset_library.get_available_props()

        for prop_name in available_props:
            prop_lower = prop_name.lower().replace('_', ' ')
            if prop_lower in description:
                # Get all assets for this prop
                prop_assets = self.asset_library.get_all_assets_for_character(prop_name)  # Uses same method
                if prop_assets:
                    assets_info['props'].append({
                        'name': prop_name,
                        'assets': [
                            {
                                'angle': asset.angle,
                                'local_path': asset.local_path,
                                'url': asset.url,
                                'prompt': asset.prompt,
                                'confidence': asset.confidence,
                                'type': 'prop'
                            }
                            for asset in prop_assets
                        ]
                    })

        logger.info(f"Found assets for shot {shot.shot_id}: "
                   f"{len(assets_info['characters'])} characters, "
                   f"{len(assets_info['locations'])} locations, "
                   f"{len(assets_info['props'])} props")

        return assets_info

    def analyze_shot(self, shot, previous_shot: Optional[ShotDesignOutput] = None) -> ShotDesignOutput:
        """
        Main method to analyze a shot and determine composition strategy

        Args:
            shot: AnnotatedShotItem containing shot information (with characters and locations from CSV)
            previous_shot: Previous shot's design output

        Returns:
            ShotDesignOutput with design analysis and asset selection
        """

        logger.info(f"Analyzing shot {shot.shot_id} with strategy: {shot.generation_strategy}")

        # Determine generation strategy handling
        if shot.generation_strategy == 'generate_new':
            return self._handle_generate_new(shot)
        elif shot.generation_strategy == 'last_frame_seed':
            return self._handle_last_seed(shot, previous_shot)
        elif shot.generation_strategy == 'multi_shot':
            return self._handle_multi_shot(shot)
        else:
            # Default to generate_new
            return self._handle_generate_new(shot)

    def _handle_generate_new(self, shot) -> ShotDesignOutput:
        """Handle shots that need fresh generation (keyframes)"""
        warnings = []
        selected_assets = []

        # Fetch assets using CSV fields for characters/locations and description for props
        assets_info = self._get_assets_for_shot(shot)

        # DEBUG: Log what was extracted
        logger.info(f"[Agent 12 DEBUG] Shot {shot.shot_id}:")
        logger.info(f"  Description: {shot.description}")
        logger.info(f"  Characters from CSV: {getattr(shot, 'characters', None)}")
        logger.info(f"  Locations from CSV: {getattr(shot, 'locations', None)}")
        logger.info(f"  Fetched assets: {len(assets_info['characters'])} characters, "
                   f"{len(assets_info['locations'])} locations, {len(assets_info['props'])} props")
        logger.info(f"  Available assets in library: {list(self.asset_library.assets.keys())}")
        logger.info(f"  Total assets in library: {len(self.asset_library.assets)}")

        # Determine required camera angle from shot style (AnnotatedShotItem uses shot_style)
        required_angle = self._map_shot_type_to_angle(shot.shot_style or 'wide_shot', shot.description)
        logger.info(f"  Required angle: {required_angle}")

        # Process character assets - select the appropriate angle from fetched assets
        for char_info in assets_info.get('characters', []):
            char_name = char_info['name']
            logger.info(f"  [Asset Search] Looking for character '{char_name}' with angle '{required_angle}'")

            # Find the asset with the required angle from the already-fetched list
            suitable_asset = next(
                (asset for asset in char_info['assets'] if asset['angle'] == required_angle),
                None
            )

            # If not found, try fallback angles
            if not suitable_asset:
                for fallback_angle in ANGLE_FALLBACKS.get(required_angle, []):
                    suitable_asset = next(
                        (asset for asset in char_info['assets'] if asset['angle'] == fallback_angle),
                        None
                    )
                    if suitable_asset:
                        break

            if suitable_asset:
                logger.info(f"  [Asset Found] Character '{char_name}': angle={suitable_asset['angle']}, "
                           f"confidence={suitable_asset['confidence']}, path={suitable_asset['local_path']}")
                selected_assets.append({
                    'character': char_name,
                    'angle': suitable_asset['angle'],
                    'local_path': suitable_asset['local_path'],
                    'url': suitable_asset['url'],
                    'confidence': suitable_asset['confidence'],
                    'type': 'character'
                })

                if suitable_asset['confidence'] < 1.0 or suitable_asset['angle'] != required_angle:
                    warnings.append(
                        f"Using fallback angle '{suitable_asset['angle']}' for {char_name} "
                        f"(preferred: '{required_angle}')"
                    )
            else:
                logger.warning(f"  [Asset NOT Found] No suitable asset for character '{char_name}' with angle '{required_angle}'")
                warnings.append(f"No suitable asset found for {char_name} with angle '{required_angle}'")

        # Process location assets
        for loc_info in assets_info.get('locations', []):
            loc_name = loc_info['name']
            suitable_asset = self._select_location_asset(loc_info)

            if suitable_asset:
                logger.info(f"  [Asset Found] Location '{loc_name}': angle={suitable_asset['angle']}, "
                           f"confidence={suitable_asset['confidence']}, path={suitable_asset['local_path']}")
                selected_assets.append({
                    'location': loc_name,
                    'angle': suitable_asset['angle'],
                    'local_path': suitable_asset['local_path'],
                    'url': suitable_asset['url'],
                    'confidence': suitable_asset['confidence'],
                    'type': 'location'
                })
            else:
                logger.warning(f"  [Asset NOT Found] No suitable asset for location '{loc_name}'")
                warnings.append(f"No suitable asset found for location {loc_name}")

        # Process prop assets - typically use master angle
        for prop_info in assets_info.get('props', []):
            prop_name = prop_info['name']
            logger.info(f"  [Asset Search] Looking for prop '{prop_name}' with angle 'master'")

            suitable_asset = next(
                (asset for asset in prop_info['assets'] if asset['angle'] == 'master'),
                None
            )
            if not suitable_asset and prop_info['assets']:
                suitable_asset = prop_info['assets'][0]

            if suitable_asset:
                logger.info(f"  [Asset Found] Prop '{prop_name}': angle={suitable_asset['angle']}, "
                           f"confidence={suitable_asset['confidence']}, path={suitable_asset['local_path']}")
                selected_assets.append({
                    'prop': prop_name,
                    'angle': suitable_asset['angle'],
                    'local_path': suitable_asset['local_path'],
                    'url': suitable_asset['url'],
                    'confidence': suitable_asset['confidence'],
                    'type': 'prop'
                })
            else:
                logger.warning(f"  [Asset NOT Found] No suitable asset for prop '{prop_name}'")
                warnings.append(f"No suitable asset found for prop {prop_name}")

        # Calculate initial feasibility
        feasibility = self._calculate_feasibility(selected_assets, warnings)

        # DEBUG: Log final results
        logger.info(f"  [Final Results] Selected {len(selected_assets)} assets")
        logger.info(f"  [Final Results] Warnings: {len(warnings)}")
        if selected_assets:
            for asset in selected_assets:
                logger.info(f"    - {asset.get('type')} '{asset.get('character') or asset.get('location') or asset.get('prop')}': {asset.get('angle')}")
        else:
            logger.warning(f"  [PROBLEM] No assets were selected for shot {shot.shot_id}!")

        # Determine model and composition strategy
        model = self._recommend_model(selected_assets)
        composition = self._determine_composition_strategy(selected_assets)

        # Check prompt feasibility with Gemini if enabled
        spatial_details = None
        consistency_report = None

        if self.use_feasibility_check and self.feasibility_checker:
            # Step 1: Analyze spatial placement
            # Use optimized_ai_notes as the prompt (Agent 11's output)
            prompt_text = (
                (shot.image or {}).get("v1", {}).get("updated_prompt")
                or (shot.image or {}).get("v0", {}).get("updated_prompt")
                or shot.optimized_ai_notes
                or shot.description
            )
            spatial_result = self.feasibility_checker.enhance_spatial_placement(
                prompt=prompt_text,
                shot_type=shot.shot_style or 'wide_shot',
                scene_context=None
            )

            # Extract spatial details for metadata
            spatial_details = spatial_result.get('placement_details', {})

            # Step 2: Check scene consistency
            if self.consistency_checker:
                # Build previous shots for consistency check
                previous_shots_data = [
                    {
                        'shot_id': s.shot_id,
                        'description': s.metadata.get('original_description', ''),
                        'prompt': s.prompt  # This comes from the ShotDesignOutput which has prompt
                    }
                    for s in self.shot_history
                ]

                # Establish baseline from first shot
                if len(self.shot_history) == 0 and not self.scene_baseline:
                    self.scene_baseline = self.consistency_checker.establish_scene_baseline({
                        'shot_id': shot.shot_id,
                        'description': shot.description,
                        'prompt': prompt_text
                    })
                    warnings.append("[Consistency] Scene baseline established from first shot")

                # Check consistency with previous shots
                if len(previous_shots_data) > 0:
                    consistency_report = self.consistency_checker.check_shot_consistency(
                        current_shot={
                            'shot_id': shot.shot_id,
                            'description': shot.description,
                            'prompt': prompt_text
                        },
                        previous_shots=previous_shots_data,
                        scene_context=self.scene_baseline
                    )

                    # Add consistency issues to warnings
                    for issue in consistency_report.get('inconsistencies', []):
                        severity = issue.get('severity', 'warning')
                        warnings.append(f"[Consistency-{severity.upper()}] {issue.get('description')}")

            # Step 3: Check feasibility of the prompt
            feasibility_result = self.feasibility_checker.check_feasibility(
                prompt=prompt_text,
                shot_type=shot.shot_style or 'wide_shot',
                available_assets=selected_assets,
                model_type=model
            )

            # Update feasibility score based on AI check and consistency
            ai_confidence = feasibility_result.get('confidence', 1.0)
            consistency_confidence = consistency_report.get('confidence', 1.0) if consistency_report else 1.0
            feasibility = (feasibility + ai_confidence + consistency_confidence) / 3

            # Add AI-identified issues to warnings
            if feasibility_result.get('issues'):
                warnings.extend([f"[AI Check] {issue}" for issue in feasibility_result['issues']])

        # Build metadata
        metadata = {
            'required_angle': required_angle,
            'characters_found': [a['character'] for a in selected_assets if a.get('type') == 'character'],
            'locations_found': [a['location'] for a in selected_assets if a.get('type') == 'location'],
            'props_found': [a['prop'] for a in selected_assets if a.get('type') == 'prop'],
            'original_description': shot.description,
            'characters_from_csv': getattr(shot, 'characters', None),
            'locations_from_csv': getattr(shot, 'locations', None)
        }
        if spatial_details:
            metadata['spatial_placement'] = spatial_details
        if consistency_report:
            metadata['consistency_check'] = {
                'is_consistent': consistency_report.get('is_consistent'),
                'confidence': consistency_report.get('confidence'),
                'issues_found': len(consistency_report.get('inconsistencies', []))
            }

        return ShotDesignOutput(
            shot_id=shot.shot_id,
            generation_strategy='generate_new',
            feasibility_score=feasibility,
            selected_assets=selected_assets,
            model_recommendation=model,
            prompt=(shot.image or {}).get("v1", {}).get("updated_prompt") or (shot.image or {}).get("v0", {}).get("updated_prompt") or shot.optimized_ai_notes or shot.description,
            composition_strategy=composition,
            warnings=warnings,
            metadata=metadata
        )

    def _handle_last_seed(self, shot, previous_shot: Optional[ShotDesignOutput]) -> ShotDesignOutput:
        """Handle shots using last frame seed from previous shot"""
        warnings = []
        selected_assets = []

        if not previous_shot:
            warnings.append("No previous shot available for last_seed strategy, falling back to generate_new")
            return self._handle_generate_new(shot)

        # Fetch assets using CSV fields for characters/locations and description for props
        assets_info = self._get_assets_for_shot(shot)
        required_angle = self._map_shot_type_to_angle(shot.shot_style or 'wide_shot', shot.description)

        # Process character assets
        for char_info in assets_info.get('characters', []):
            char_name = char_info['name']

            # First try to reuse from previous shot if same angle
            prev_asset = next((a for a in previous_shot.selected_assets if a.get('character') == char_name), None)

            if prev_asset and self._is_angle_compatible(prev_asset['angle'], required_angle):
                selected_assets.append(prev_asset)
            else:
                # Need to find new asset for different angle
                suitable_asset = next(
                    (asset for asset in char_info['assets'] if asset['angle'] == required_angle),
                    None
                )

                # If not found, try fallback angles
                if not suitable_asset:
                    for fallback_angle in ANGLE_FALLBACKS.get(required_angle, []):
                        suitable_asset = next(
                            (asset for asset in char_info['assets'] if asset['angle'] == fallback_angle),
                            None
                        )
                        if suitable_asset:
                            break

                if suitable_asset:
                    selected_assets.append({
                        'character': char_name,
                        'angle': suitable_asset['angle'],
                        'local_path': suitable_asset['local_path'],
                        'url': suitable_asset['url'],
                        'confidence': suitable_asset['confidence'],
                        'type': 'character'
                    })
                    if prev_asset:
                        warnings.append(f"Angle changed from '{prev_asset['angle']}' to '{suitable_asset['angle']}' for {char_name}")
                else:
                    warnings.append(f"No suitable asset found for {char_name} with angle '{required_angle}'")

        # Process location assets
        for loc_info in assets_info.get('locations', []):
            loc_name = loc_info['name']
            prev_asset = next((a for a in previous_shot.selected_assets if a.get('location') == loc_name), None)

            if prev_asset:
                selected_assets.append(prev_asset)
            else:
                suitable_asset = self._select_location_asset(loc_info)

                if suitable_asset:
                    selected_assets.append({
                        'location': loc_name,
                        'angle': suitable_asset['angle'],
                        'local_path': suitable_asset['local_path'],
                        'url': suitable_asset['url'],
                        'confidence': suitable_asset['confidence'],
                        'type': 'location'
                    })
                else:
                    warnings.append(f"No suitable asset found for location {loc_name}")

        # Process prop assets
        for prop_info in assets_info.get('props', []):
            prop_name = prop_info['name']
            prev_asset = next((a for a in previous_shot.selected_assets if a.get('prop') == prop_name), None)

            if prev_asset:
                selected_assets.append(prev_asset)
            else:
                suitable_asset = next(
                    (asset for asset in prop_info['assets'] if asset['angle'] == 'master'),
                    None
                )
                if not suitable_asset and prop_info['assets']:
                    suitable_asset = prop_info['assets'][0]

                if suitable_asset:
                    selected_assets.append({
                        'prop': prop_name,
                        'angle': suitable_asset['angle'],
                        'local_path': suitable_asset['local_path'],
                        'url': suitable_asset['url'],
                        'confidence': suitable_asset['confidence'],
                        'type': 'prop'
                    })
                else:
                    warnings.append(f"No suitable asset found for prop {prop_name}")

        feasibility = self._calculate_feasibility(selected_assets, warnings)
        model = self._recommend_model(selected_assets)
        composition = {
            'primary_technique': 'img2img_with_seed',
            'seed_source': previous_shot.shot_id,
            'reference_assets': selected_assets
        }

        return ShotDesignOutput(
            shot_id=shot.shot_id,
            generation_strategy='last_frame_seed',
            feasibility_score=feasibility,
            selected_assets=selected_assets,
            model_recommendation=model,
            prompt=(shot.image or {}).get("v1", {}).get("updated_prompt") or (shot.image or {}).get("v0", {}).get("updated_prompt") or shot.optimized_ai_notes or shot.description,
            composition_strategy=composition,
            warnings=warnings,
            metadata={
                'previous_shot': previous_shot.shot_id,
                'required_angle': required_angle
            }
        )

    def _handle_multi_shot(self, shot) -> ShotDesignOutput:
        """Handle multi-shot sequences (one image for multiple cuts)"""
        warnings = []
        selected_assets = []

        # Fetch assets using CSV fields for characters/locations and description for props
        assets_info = self._get_assets_for_shot(shot)
        required_angle = self._map_shot_type_to_angle(shot.shot_style or 'wide_shot', shot.description)

        # Process character assets
        for char_info in assets_info.get('characters', []):
            char_name = char_info['name']

            # Find the asset with the required angle
            suitable_asset = next(
                (asset for asset in char_info['assets'] if asset['angle'] == required_angle),
                None
            )

            # If not found, try fallback angles
            if not suitable_asset:
                for fallback_angle in ANGLE_FALLBACKS.get(required_angle, []):
                    suitable_asset = next(
                        (asset for asset in char_info['assets'] if asset['angle'] == fallback_angle),
                        None
                    )
                    if suitable_asset:
                        break

            if suitable_asset:
                selected_assets.append({
                    'character': char_name,
                    'angle': suitable_asset['angle'],
                    'local_path': suitable_asset['local_path'],
                    'url': suitable_asset['url'],
                    'confidence': suitable_asset['confidence'],
                    'type': 'character'
                })
            else:
                warnings.append(f"No suitable asset found for {char_name} with angle '{required_angle}'")

        # Process location assets
        for loc_info in assets_info.get('locations', []):
            loc_name = loc_info['name']

            suitable_asset = self._select_location_asset(loc_info)

            if suitable_asset:
                selected_assets.append({
                    'location': loc_name,
                    'angle': suitable_asset['angle'],
                    'local_path': suitable_asset['local_path'],
                    'url': suitable_asset['url'],
                    'confidence': suitable_asset['confidence'],
                    'type': 'location'
                })
            else:
                warnings.append(f"No suitable asset found for location {loc_name}")

        # Process prop assets
        for prop_info in assets_info.get('props', []):
            prop_name = prop_info['name']

            # Find the master angle asset
            suitable_asset = next(
                (asset for asset in prop_info['assets'] if asset['angle'] == 'master'),
                None
            )
            if not suitable_asset and prop_info['assets']:
                suitable_asset = prop_info['assets'][0]

            if suitable_asset:
                selected_assets.append({
                    'prop': prop_name,
                    'angle': suitable_asset['angle'],
                    'local_path': suitable_asset['local_path'],
                    'url': suitable_asset['url'],
                    'confidence': suitable_asset['confidence'],
                    'type': 'prop'
                })
            else:
                warnings.append(f"No suitable asset found for prop {prop_name}")

        feasibility = self._calculate_feasibility(selected_assets, warnings)
        model = self._recommend_model(selected_assets)
        composition = {
            'primary_technique': 'multi_shot_single_image',
            'reference_assets': selected_assets,
            'note': 'Single image will be used for multiple video cuts'
        }

        return ShotDesignOutput(
            shot_id=shot.shot_id,
            generation_strategy='multi_shot',
            feasibility_score=feasibility,
            selected_assets=selected_assets,
            model_recommendation=model,
            prompt=(shot.image or {}).get("v1", {}).get("updated_prompt") or (shot.image or {}).get("v0", {}).get("updated_prompt") or shot.optimized_ai_notes or shot.description,
            composition_strategy=composition,
            warnings=warnings,
            metadata={'required_angle': required_angle}
        )


    def _select_location_asset(self, loc_info: Dict) -> Optional[Dict]:
        """
        Pick the best location asset from a pre-fetched list.

        When the CSV has a direction suffix (e.g. CAFE_NORTH) the caller already
        provides only that one directional asset, so we just return it.

        When the CSV has no direction we prefer a ground-level directional view
        (north → south → east → west) over the aerial master image, because Agent 14
        receives this as a compositing reference and a top-down aerial is the wrong
        framing for interior / street-level shots.
        """
        assets = loc_info.get('assets', [])
        if not assets:
            return None

        # If there is only one asset (directional already selected upstream), return it.
        if len(assets) == 1:
            return assets[0]

        # Prefer ground-level directional views over the aerial master.
        for preferred in ('north', 'south', 'east', 'west'):
            match = next((a for a in assets if a.get('angle', '').lower() == preferred), None)
            if match:
                return match

        # Fall back to master (aerial) if no directional view exists.
        return next((a for a in assets if a.get('angle', '').lower() == 'master'), assets[0])

    def _map_shot_type_to_angle(self, shot_type: str, description: str) -> str:
        """Map shot type and description to required asset angle"""
        shot_type_lower = shot_type.lower()
        desc_lower = description.lower()

        # Check description for specific angle mentions
        if 'back' in desc_lower or 'rear' in desc_lower:
            return 'back_shot'
        if 'profile' in desc_lower or 'side' in desc_lower:
            if 'left' in desc_lower:
                return 'profile_left'
            elif 'right' in desc_lower:
                return 'profile_right'
            return 'profile_left'

        # Map shot types to angles
        if 'cu' in shot_type_lower or 'close' in shot_type_lower or 'mcu' in shot_type_lower:
            return 'close_up'
        elif 'wide' in shot_type_lower or 'ws' in shot_type_lower:
            return 'wide_shot'
        elif 'ms' in shot_type_lower or 'medium' in shot_type_lower:
            return 'wide_shot'

        return 'close_up'  # Default

    def _is_angle_compatible(self, angle1: str, angle2: str) -> bool:
        """Check if two angles are compatible for reuse"""
        if angle1 == angle2:
            return True
        if {angle1, angle2} == {'close_up', 'master'}:
            return True
        return False

    def _calculate_feasibility(self, assets: List[Dict], warnings: List[str]) -> float:
        """Calculate feasibility score for shot composition"""
        if not assets:
            return 0.0

        # Base score from asset availability
        avg_confidence = sum(a['confidence'] for a in assets) / len(assets)

        # Penalty for warnings
        warning_penalty = len(warnings) * 0.1

        return max(0.0, min(1.0, avg_confidence - warning_penalty))

    def _recommend_model(self, assets: List[Dict]) -> str:
        """Recommend image generation model based on asset count"""
        asset_count = len(assets)

        if asset_count == 0:
            return 'flux_schnell'
        elif asset_count == 1:
            return 'flux_with_ip_adapter'
        else:
            return 'flux_with_multi_ip_adapter'

    def _determine_composition_strategy(self, assets: List[Dict]) -> Dict:
        """Determine how to compose the shot"""
        if len(assets) == 0:
            return {'primary_technique': 'text_to_image'}
        elif len(assets) == 1:
            return {
                'primary_technique': 'ip_adapter_single_reference',
                'reference_weights': [1.0]
            }
        else:
            weights = [0.7] + [0.3 / (len(assets) - 1)] * (len(assets) - 1)
            return {
                'primary_technique': 'ip_adapter_multi_reference',
                'reference_weights': weights
            }


def process_shot_list(
    shot_list: List,  # List of AnnotatedShotItem objects
    asset_library: AssetLibrary,
    use_feasibility_check: bool = True
) -> List[ShotDesignOutput]:
    """
    Process entire shot list

    Args:
        shot_list: List of AnnotatedShotItem objects (with characters and locations from CSV)
        asset_library: AssetLibrary instance
        use_feasibility_check: Whether to run feasibility checks

    Returns:
        List of ShotDesignOutput objects
    """
    agent = ShotDesignAgent(asset_library, use_feasibility_check=use_feasibility_check)
    results = []
    previous_shot = None

    for shot in shot_list:
        output = agent.analyze_shot(shot, previous_shot)
        results.append(output)
        previous_shot = output
        agent.shot_history.append(output)

    return results


def save_results(results: List[ShotDesignOutput], output_dir: str = "phase_2_agents/outputs/agent_shot_design"):
    """Save agent 12 results to JSON"""
    import json
    from pathlib import Path
    from datetime import datetime
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/agent12_shot_design_{timestamp}.json"

    output_data = {
        'agent': 'Agent 12: Shot Design Agent',
        'timestamp': datetime.now().isoformat(),
        'shot_designs': [asdict(r) for r in results],
        'statistics': {
            'total_shots': len(results),
            'avg_feasibility': sum(r.feasibility_score for r in results) / len(results) if results else 0,
            'strategies_used': {
                'generate_new': sum(1 for r in results if r.generation_strategy == 'generate_new'),
                'last_frame_seed': sum(1 for r in results if r.generation_strategy == 'last_frame_seed'),
                'multi_shot': sum(1 for r in results if r.generation_strategy == 'multi_shot')
            }
        }
    }

    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {filename}")
    return filename

