"""
Asset Library Manager for Phase 2
Handles asset storage and retrieval for shot composition from MongoDB/Phase 1 outputs
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from bson import ObjectId
from backend.shared.utils.mongodb_validators import validate_object_id

logger = logging.getLogger(__name__)


@dataclass
class AssetInfo:
    """Information about a single asset"""
    name: str
    type: str  # character, location, prop
    angle: str  # back_shot, close_up, profile_right, profile_left, wide_shot, master
    local_path: str
    url: Optional[str] = None
    confidence: float = 1.0
    prompt: str = ''  # Image generation prompt
    technical: str = ''  # Technical specifications
    framing: str = ''  # Framing details


class AssetLibrary:
    """Manages asset storage and retrieval from Phase 1 MongoDB (production_projects.agent_outputs.agent8)"""

    def __init__(self, mongodb_client=None, show_id: Optional[str] = None, project_id: Optional[str] = None, episode_number: Optional[int] = None):
        """
        Initialize Asset Library
        
        Args:
            mongodb_client: MongoDB client to fetch assets
            show_id: Show identifier (matches _id in production_projects)
            project_id: Project ID (alternative to show_id)
            episode_number: Episode number (for backward compatibility)
        """
        self.mongodb_client = mongodb_client
        self.show_id = show_id
        self.project_id = project_id
        self.episode_number = episode_number
        self.assets: Dict[str, List[AssetInfo]] = {}
        
        # Load assets from MongoDB
        if mongodb_client and (show_id or project_id):
            # Prefer show_id if provided, otherwise use project_id
            identifier = show_id if show_id else project_id
            self._load_from_agent8_output(identifier)
        else:
            logger.warning("No MongoDB client or show_id/project_id provided. Asset library will be empty.")

    def _load_from_agent8_output(self, show_id: str):
        """
        Load assets from MongoDB production_projects collection by _id (show_id)
        - Characters from agent8.output.variation_images
        - Locations and props from agent5.output.generated_images
        
        Args:
            show_id: Show identifier (matches _id in production_projects)
        """
        try:
            logger.info(f"Loading assets from MongoDB for show_id: {show_id}")
            
            # Get database from client - handle different client types
            if not self.mongodb_client:
                logger.error("MongoDB client is None")
                return
                
            # Check if it's a MongoDBAtlasClient with database attribute
            if hasattr(self.mongodb_client, 'database') and self.mongodb_client.database is not None:
                db = self.mongodb_client.database
            elif hasattr(self.mongodb_client, 'client'):
                # It's a MongoDBAtlasClient, get database from client
                if hasattr(self.mongodb_client, 'database_name'):
                    db = self.mongodb_client.client[self.mongodb_client.database_name]
                else:
                    # Default database name
                    db = self.mongodb_client.client.get_database("production")
            elif hasattr(self.mongodb_client, 'get_database'):
                # It's a MongoClient directly
                db = self.mongodb_client.get_database("production")
            else:
                logger.error(f"Unsupported MongoDB client type: {type(self.mongodb_client)}")
                return
                
            projects_collection = db['production_projects']
            assets_collections = db['assets_collections']
            
            # Convert string show_id to ObjectId if needed
            if isinstance(show_id, str):
                try:
                    from fastapi import HTTPException
                    show_id_obj = validate_object_id(show_id)
                except (ValueError, HTTPException) as e:
                    logger.error(f"Invalid show_id format: {e}")
                    logger.warning(f"Attempting to use show_id as string: {show_id}")
                    show_id_obj = show_id
            else:
                show_id_obj = show_id

            # Query project by _id
            query = {"_id": show_id_obj}
            project_doc = projects_collection.find_one(query)
            
            if not project_doc:
                logger.warning(f"No project found for show_id: {show_id}")
                return
            
            logger.info(f"Found project: {project_doc.get('name', 'Unknown')}")
            
            # Attempt to load agent outputs from assets_collections (Phase 1 movie workflow)
            agent_outputs_source = "project.agent_outputs"
            agent_outputs = project_doc.get('agent_outputs', {})
            assets_collection_id = project_doc.get('assets_collection_id')
            assets_collection_doc = None
            
            if assets_collection_id:
                # assets_collection_id might already be an ObjectId
                assets_collection_id_str = str(assets_collection_id)
                try:
                    from fastapi import HTTPException
                    assets_collection_obj_id = (
                        assets_collection_id
                        if isinstance(assets_collection_id, ObjectId)
                        else validate_object_id(assets_collection_id_str)
                    )
                    
                    assets_collection_doc = assets_collections.find_one({"_id": assets_collection_obj_id})
                except (ValueError, HTTPException) as e:
                    logger.error(f"Invalid assets_collection_id format ({assets_collection_id_str}): {e}")
                except Exception as e:
                    logger.error(f"Failed to fetch assets collection {assets_collection_id_str}: {e}")
            
            if assets_collection_doc:
                logger.info(f"Loading assets from assets_collections using ID {assets_collection_id_str}")
                agent_outputs = {
                    'agent8': assets_collection_doc.get('agent8_output', {}),
                    'agent5': assets_collection_doc.get('agent5_output', {})
                }
                agent_outputs_source = "assets_collections"
            elif assets_collection_id:
                logger.warning(
                    f"Assets collection {assets_collection_id_str} not found. "
                    "Falling back to project agent_outputs."
                )
            else:
                logger.warning(
                    "Project does not have assets_collection_id. "
                    "Using agent_outputs embedded in project document."
                )
            
            logger.info(f"Using agent outputs from {agent_outputs_source}")
            
            # Load characters and locations from agent8
            agent8_data = agent_outputs.get('agent8', {})
            if agent8_data.get('status') == 'completed':
                output = agent8_data.get('output', {})
                variation_images = output.get('variation_images', {})

                # Load characters from agent8
                characters = variation_images.get('characters', {})
                logger.info(f"Characters data type: {type(characters)}")
                if isinstance(characters, list):
                    logger.info(f"Characters list length: {len(characters)}")
                    if characters:
                        logger.info(f"First character item type: {type(characters[0])}")
                elif isinstance(characters, dict):
                    logger.info(f"Characters dict keys: {list(characters.keys())}")
                self._load_characters_from_agent8(characters)

                # Load locations from agent8 (with directional variations)
                locations_agent8 = variation_images.get('locations', {})
                logger.info(f"Locations from agent8 data type: {type(locations_agent8)}")
                if isinstance(locations_agent8, list):
                    logger.info(f"Locations from agent8 list length: {len(locations_agent8)}")
                    if locations_agent8:
                        logger.info(f"First location from agent8 item type: {type(locations_agent8[0])}")
                elif isinstance(locations_agent8, dict):
                    logger.info(f"Locations from agent8 dict keys: {list(locations_agent8.keys())}")
                if locations_agent8:
                    self._load_locations_from_agent8(locations_agent8)
                    logger.info(f"Loaded {len(locations_agent8)} locations from agent8 with directional variations")
            else:
                logger.warning(f"Agent 8 status is not 'completed': {agent8_data.get('status')}")
            
            # Load locations and props from agent5
            agent5_data = agent_outputs.get('agent5', {})
            if agent5_data.get('status') == 'completed':
                output = agent5_data.get('output', {})
                generated_images = output.get('generated_images', {})
                
                # Load locations from agent5
                locations = generated_images.get('locations', {})
                logger.info(f"Locations data type: {type(locations)}")
                if isinstance(locations, list):
                    logger.info(f"Locations list length: {len(locations)}")
                    if locations:
                        logger.info(f"First location item type: {type(locations[0])}")
                elif isinstance(locations, dict):
                    logger.info(f"Locations dict keys: {list(locations.keys())}")
                if locations:
                    self._load_locations_from_agent5(locations)
                
                # Load props from agent5
                props = generated_images.get('props', {})
                logger.info(f"Props data type: {type(props)}")
                if isinstance(props, list):
                    logger.info(f"Props list length: {len(props)}")
                    if props:
                        logger.info(f"First prop item type: {type(props[0])}")
                elif isinstance(props, dict):
                    logger.info(f"Props dict keys: {list(props.keys())}")
                if props:
                    self._load_props_from_agent5(props)

                # Load characters from agent5 as fallback when agent8 produced none
                existing_chars = [
                    name for name, assets in self.assets.items()
                    if assets and assets[0].type == 'character'
                ]
                if not existing_chars:
                    characters_agent5 = generated_images.get('characters', {})
                    if characters_agent5:
                        logger.info("No characters loaded from agent8 — falling back to agent5 character images")
                        self._load_characters_from_agent5(characters_agent5)
                    else:
                        logger.warning("No character images found in agent5 output either")
                else:
                    logger.info(f"Characters already loaded from agent8 ({len(existing_chars)}), skipping agent5 fallback")
            else:
                logger.warning(f"Agent 5 status is not 'completed': {agent5_data.get('status')}")
            
            logger.info(f"Successfully loaded {len(self.assets)} unique assets from agent8 and agent5 output")
            
            # Log asset summary
            for asset_name, asset_list in self.assets.items():
                angles = [a.angle for a in asset_list]
                logger.info(f"  {asset_name} ({asset_list[0].type}): {', '.join(angles)}")
            
        except Exception as e:
            logger.error(f"Failed to load assets from MongoDB: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _load_characters_from_agent8(self, characters):
        """Load character assets from agent8 variation_images.characters"""
        # Handle both dict and list formats
        if isinstance(characters, list):
            # Convert list to dict format if needed
            characters_dict = {}
            for i, char_data in enumerate(characters):
                if isinstance(char_data, dict):
                    # Try to find a name field or use index as fallback
                    char_name = char_data.get('name') or char_data.get('character_name') or f"character_{i}"
                    characters_dict[char_name] = char_data
                else:
                    logger.warning(f"Unexpected character data format in list at index {i}: {char_data}")
                    continue
            characters = characters_dict
            logger.info(f"Converted characters list to dict with {len(characters)} items")
        elif not isinstance(characters, dict):
            logger.error(f"Characters data is neither dict nor list: {type(characters)}")
            return
            
        for char_name, char_data in characters.items():
            normalized_name = char_name.upper().replace(' ', '_')
            if normalized_name not in self.assets:
                self.assets[normalized_name] = []
            
            # Add master image if exists
            master_image = char_data.get('master_image')
            if master_image:
                # Extract prompt and technical details from master image data
                prompt = char_data.get('prompt', '')
                technical = char_data.get('technical', '')
                framing = char_data.get('framing', '')

                # master_image may be a presigned S3 URL — store it in url, not local_path
                is_url = isinstance(master_image, str) and master_image.startswith('http')
                self.assets[normalized_name].append(AssetInfo(
                    name=normalized_name,
                    type='character',
                    angle='master',
                    local_path='' if is_url else master_image,
                    url=master_image if is_url else None,
                    confidence=1.0,
                    prompt=prompt,
                    technical=technical,
                    framing=framing
                ))
            
            # Add variation images
            variations = char_data.get('variations', {})
            for angle, angle_data in variations.items():
                images = angle_data.get('images', [])
                if images:
                    img = images[-1]
                    # Extract prompt and technical details from variation data
                    prompt = angle_data.get('prompt', '')
                    technical = angle_data.get('technical', '')
                    framing = angle_data.get('framing', '')

                    self.assets[normalized_name].append(AssetInfo(
                        name=normalized_name,
                        type='character',
                        angle=angle,
                        local_path=img.get('local_path', ''),
                        url=img.get('url'),
                        confidence=1.0,
                        prompt=prompt,
                        technical=technical,
                        framing=framing
                    ))
    
    def _load_locations_from_agent8(self, locations):
        """Load location assets from agent8 variation_images.locations"""
        # Handle both dict and list formats
        if isinstance(locations, list):
            # Convert list to dict format if needed
            locations_dict = {}
            for i, loc_data in enumerate(locations):
                if isinstance(loc_data, dict):
                    # Try to find a name field or use index as fallback
                    loc_name = loc_data.get('name') or loc_data.get('location_name') or f"location_{i}"
                    locations_dict[loc_name] = loc_data
                else:
                    logger.warning(f"Unexpected location data format in list at index {i}: {loc_data}")
                    continue
            locations = locations_dict
            logger.info(f"Converted locations list to dict with {len(locations)} items")
        elif not isinstance(locations, dict):
            logger.error(f"Locations data is neither dict nor list: {type(locations)}")
            return

        for loc_name, loc_data in locations.items():
            normalized_name = loc_name.upper().replace(' ', '_')
            if normalized_name not in self.assets:
                self.assets[normalized_name] = []

            # Add master image if exists
            master_image = loc_data.get('master_image')
            if master_image:
                # Extract prompt and technical details from master image data
                prompt = loc_data.get('prompt', '')
                technical = loc_data.get('technical', '')
                framing = loc_data.get('framing', '')

                # master_image from agent8 is a presigned S3 URL — store it in url, not local_path
                is_url = isinstance(master_image, str) and master_image.startswith('http')
                self.assets[normalized_name].append(AssetInfo(
                    name=normalized_name,
                    type='location',
                    angle='master',
                    local_path='' if is_url else master_image,
                    url=master_image if is_url else None,
                    confidence=1.0,
                    prompt=prompt,
                    technical=technical,
                    framing=framing
                ))

            # Add variation images if they exist
            variations = loc_data.get('variations', {})
            for angle, angle_data in variations.items():
                images = angle_data.get('images', [])
                if images:
                    img = images[-1]
                    # Extract prompt and technical details from variation data
                    prompt = angle_data.get('prompt', '')
                    technical = angle_data.get('technical', '')
                    framing = angle_data.get('framing', '')

                    # Extract s3_url from images[-1]
                    s3_url = img.get('s3_url') or img.get('url')

                    self.assets[normalized_name].append(AssetInfo(
                        name=normalized_name,
                        type='location',
                        angle=angle,
                        local_path=img.get('local_path', ''),
                        url=s3_url,
                        confidence=1.0,
                        prompt=prompt,
                        technical=technical,
                        framing=framing
                    ))
    
    def _load_props_from_agent8(self, props):
        """Load prop assets from agent8 variation_images.props"""
        # Handle both dict and list formats
        if isinstance(props, list):
            # Convert list to dict format if needed
            props_dict = {}
            for i, prop_data in enumerate(props):
                if isinstance(prop_data, dict):
                    # Try to find a name field or use index as fallback
                    prop_name = prop_data.get('name') or prop_data.get('prop_name') or f"prop_{i}"
                    props_dict[prop_name] = prop_data
                else:
                    logger.warning(f"Unexpected prop data format in list at index {i}: {prop_data}")
                    continue
            props = props_dict
            logger.info(f"Converted props list to dict with {len(props)} items")
        elif not isinstance(props, dict):
            logger.error(f"Props data is neither dict nor list: {type(props)}")
            return
            
        for prop_name, prop_data in props.items():
            normalized_name = prop_name.upper().replace(' ', '_')
            if normalized_name not in self.assets:
                self.assets[normalized_name] = []
            
            # Add master image if exists
            master_image = prop_data.get('master_image')
            if master_image:
                # Extract prompt and technical details from master image data
                prompt = prop_data.get('prompt', '')
                technical = prop_data.get('technical', '')
                framing = prop_data.get('framing', '')
                
                self.assets[normalized_name].append(AssetInfo(
                    name=normalized_name,
                    type='prop',
                    angle='master',
                    local_path=master_image,
                    url=None,
                    confidence=1.0,
                    prompt=prompt,
                    technical=technical,
                    framing=framing
                ))
            
            # Add variation images if they exist
            variations = prop_data.get('variations', {})
            for angle, angle_data in variations.items():
                images = angle_data.get('images', [])
                if images:
                    img = images[-1]
                    # Extract prompt and technical details from variation data
                    prompt = angle_data.get('prompt', '')
                    technical = angle_data.get('technical', '')
                    framing = angle_data.get('framing', '')

                    self.assets[normalized_name].append(AssetInfo(
                        name=normalized_name,
                        type='prop',
                        angle=angle,
                        local_path=img.get('local_path', ''),
                        url=img.get('url'),
                        confidence=1.0,
                        prompt=prompt,
                        technical=technical,
                        framing=framing
                    ))
    
    def _load_locations_from_agent5(self, locations):
        """Load location assets from agent5 generated_images.locations"""
        # Handle both dict and list formats
        if isinstance(locations, list):
            # Convert list to dict format if needed
            locations_dict = {}
            for i, loc_data in enumerate(locations):
                if isinstance(loc_data, dict):
                    # Try to find a name field or use index as fallback
                    loc_name = loc_data.get('name') or loc_data.get('location_name') or f"location_{i}"
                    locations_dict[loc_name] = loc_data
                else:
                    logger.warning(f"Unexpected location data format in list at index {i}: {loc_data}")
                    continue
            locations = locations_dict
            logger.info(f"Converted locations list to dict with {len(locations)} items")
        elif not isinstance(locations, dict):
            logger.error(f"Locations data is neither dict nor list: {type(locations)}")
            return
            
        for loc_name, loc_data in locations.items():
            normalized_name = loc_name.upper().replace(' ', '_')
            if normalized_name not in self.assets:
                self.assets[normalized_name] = []
            
            # Extract data from agent5 structure
            prompt = loc_data.get('prompt', '')
            technical_specs = loc_data.get('technical_specs', {})
            images = loc_data.get('images', [])
            
            # Add master image if exists
            if images:
                master_img = images[-1]
                self.assets[normalized_name].append(AssetInfo(
                    name=normalized_name,
                    type='location',
                    angle='master',
                    local_path=master_img.get('local_path', ''),
                    url=master_img.get('url'),
                    confidence=1.0,
                    prompt=prompt,
                    technical=str(technical_specs),
                    framing=''
                ))
    
    def _load_characters_from_agent5(self, characters):
        """Load character assets from agent5 generated_images.characters (fallback when agent8 has none)"""
        if isinstance(characters, list):
            characters_dict = {}
            for i, char_data in enumerate(characters):
                if isinstance(char_data, dict):
                    char_name = char_data.get('name') or char_data.get('character_name') or f"character_{i}"
                    characters_dict[char_name] = char_data
                else:
                    logger.warning(f"Unexpected character data in agent5 list at index {i}: {char_data}")
                    continue
            characters = characters_dict
            logger.info(f"Converted agent5 characters list to dict with {len(characters)} items")
        elif not isinstance(characters, dict):
            logger.error(f"Agent5 characters data is neither dict nor list: {type(characters)}")
            return

        for char_name, char_data in characters.items():
            normalized_name = char_name.upper().replace(' ', '_')
            if normalized_name not in self.assets:
                self.assets[normalized_name] = []

            prompt = char_data.get('prompt', '')
            technical_specs = char_data.get('technical_specs', {})
            images = char_data.get('images', [])

            if images:
                master_img = images[-1]
                s3_url = master_img.get('url') or master_img.get('s3_url')
                is_url = isinstance(s3_url, str) and s3_url.startswith('http')
                self.assets[normalized_name].append(AssetInfo(
                    name=normalized_name,
                    type='character',
                    angle='master',
                    local_path='' if is_url else master_img.get('local_path', ''),
                    url=s3_url if is_url else None,
                    confidence=1.0,
                    prompt=prompt,
                    technical=str(technical_specs),
                    framing=''
                ))
                logger.info(f"Loaded character {normalized_name} from agent5 (master image)")

    def _load_props_from_agent5(self, props):
        """Load prop assets from agent5 generated_images.props"""
        # Handle both dict and list formats
        if isinstance(props, list):
            # Convert list to dict format if needed
            props_dict = {}
            for i, prop_data in enumerate(props):
                if isinstance(prop_data, dict):
                    # Try to find a name field or use index as fallback
                    prop_name = prop_data.get('name') or prop_data.get('prop_name') or f"prop_{i}"
                    props_dict[prop_name] = prop_data
                else:
                    logger.warning(f"Unexpected prop data format in list at index {i}: {prop_data}")
                    continue
            props = props_dict
            logger.info(f"Converted props list to dict with {len(props)} items")
        elif not isinstance(props, dict):
            logger.error(f"Props data is neither dict nor list: {type(props)}")
            return
            
        for prop_name, prop_data in props.items():
            normalized_name = prop_name.upper().replace(' ', '_')
            if normalized_name not in self.assets:
                self.assets[normalized_name] = []
            
            # Extract data from agent5 structure
            prompt = prop_data.get('prompt', '')
            technical_specs = prop_data.get('technical_specs', {})
            images = prop_data.get('images', [])
            
            # Add master image if exists
            if images:
                master_img = images[-1]
                # Extract S3 URL from the images field
                s3_url = master_img.get('url')  # This should contain the S3 URL
                
                self.assets[normalized_name].append(AssetInfo(
                    name=normalized_name,
                    type='prop',
                    angle='master',
                    local_path=master_img.get('local_path', ''),
                    url=s3_url,  # Use the S3 URL from the images field
                    confidence=1.0,
                    prompt=prompt,
                    technical=str(technical_specs),
                    framing=''
                ))
    


    def find_asset(
        self,
        asset_name: str,
        preferred_angle: str,
        fallback_angles: Optional[List[str]] = None
    ) -> Optional[AssetInfo]:
        """Find the best matching asset for a character, location, or prop and angle"""
        normalized_name = asset_name.replace(' ', '_').upper()

        if normalized_name not in self.assets:
            logger.warning(f"No assets found for asset: {normalized_name}")
            return None

        # Try exact match first
        for asset in self.assets[normalized_name]:
            if asset.angle == preferred_angle:
                return asset

        # Try fallback angles
        if fallback_angles:
            for fallback in fallback_angles:
                for asset in self.assets[normalized_name]:
                    if asset.angle == fallback:
                        asset.confidence = 0.8  # Reduced confidence for fallback
                        return asset

        # Return master as last resort
        for asset in self.assets[normalized_name]:
            if asset.angle == 'master':
                asset.confidence = 0.6
                return asset

        return None

    def get_all_assets_for_character(self, character_name: str) -> List[AssetInfo]:
        """Get all available assets for a character"""
        normalized_name = character_name.replace(' ', '_').upper()
        return self.assets.get(normalized_name, [])

    def get_location_asset_by_direction(self, location_name: str, direction: str) -> Optional[AssetInfo]:
        """
        Get location asset for a specific direction (north, south, east, west)

        Args:
            location_name: Base location name (e.g., "JUNGLE")
            direction: Direction (e.g., "north", "south", "east", "west")

        Returns:
            AssetInfo for the specific direction, or None if not found
        """
        normalized_name = location_name.replace(' ', '_').upper()
        direction_lower = direction.lower()

        if normalized_name not in self.assets:
            logger.warning(f"No assets found for location: {normalized_name}")
            return None

        # Search for the asset with the matching direction as angle
        for asset in self.assets[normalized_name]:
            if asset.angle.lower() == direction_lower:
                logger.info(f"Found location asset for {normalized_name} in direction {direction}: {asset.url}")
                return asset

        logger.warning(f"No asset found for location {normalized_name} with direction {direction}")
        return None

    def list_available_angles(self, character_name: str) -> List[str]:
        """List all available angles for a character"""
        assets = self.get_all_assets_for_character(character_name)
        return [asset.angle for asset in assets]

    def get_available_characters(self) -> List[str]:
        """Get list of all available character names"""
        return [name for name, assets in self.assets.items() 
                if assets and assets[0].type == 'character']

    def get_available_locations(self) -> List[str]:
        """Get list of all available location names"""
        return [name for name, assets in self.assets.items() 
                if assets and assets[0].type == 'location']

    def get_available_props(self) -> List[str]:
        """Get list of all available prop names"""
        return [name for name, assets in self.assets.items() 
                if assets and assets[0].type == 'prop']

    def get_asset_prompt(self, asset_name: str, angle: str = 'master') -> Optional[str]:
        """Get the actual prompt text for an asset"""
        normalized_name = asset_name.replace(' ', '_').upper()
        
        if normalized_name not in self.assets:
            return None
            
        for asset in self.assets[normalized_name]:
            if asset.angle == angle:
                return asset.prompt
                
        # If specific angle not found, try master
        for asset in self.assets[normalized_name]:
            if asset.angle == 'master':
                return asset.prompt
                
        return None

    def search_assets(self, search_term: str) -> Dict:
        """Search for assets containing the search term"""
        results = {
            'characters': [],
            'locations': [],
            'props': []
        }
        
        search_lower = search_term.lower()
        
        for name, assets in self.assets.items():
            if not assets:
                continue
                
            asset_type = assets[0].type
            name_lower = name.lower()
            
            # Check if search term matches asset name
            if search_lower in name_lower:
                if asset_type == 'character':
                    results['characters'].append(name)
                elif asset_type == 'location':
                    results['locations'].append(name)
                elif asset_type == 'prop':
                    results['props'].append(name)
            
            # Also check prompts for matches
            for asset in assets:
                if search_lower in asset.prompt.lower():
                    if asset_type == 'character' and name not in results['characters']:
                        results['characters'].append(name)
                    elif asset_type == 'location' and name not in results['locations']:
                        results['locations'].append(name)
                    elif asset_type == 'prop' and name not in results['props']:
                        results['props'].append(name)
        
        return results

    def generate_asset_summary(self) -> Dict:
        """Generate summary of all assets in library"""
        summary = {
            'total_characters': len([name for name, assets in self.assets.items() 
                                   if assets and assets[0].type == 'character']),
            'total_locations': len([name for name, assets in self.assets.items() 
                                   if assets and assets[0].type == 'location']),
            'total_props': len([name for name, assets in self.assets.items() 
                               if assets and assets[0].type == 'prop']),
            'characters': {},
            'locations': {},
            'props': {}
        }
        
        # Character details
        for name, assets in self.assets.items():
            if not assets or assets[0].type != 'character':
                continue
                
            angles = [asset.angle for asset in assets]
            prompts = [asset.prompt for asset in assets if asset.prompt]
            
            summary['characters'][name] = {
                'available_angles': angles,
                'angle_count': len(angles),
                'has_prompts': len(prompts) > 0,
                'prompt_count': len(prompts)
            }
        
        # Location details
        for name, assets in self.assets.items():
            if not assets or assets[0].type != 'location':
                continue
                
            angles = [asset.angle for asset in assets]
            prompts = [asset.prompt for asset in assets if asset.prompt]
            
            summary['locations'][name] = {
                'available_angles': angles,
                'angle_count': len(angles),
                'has_prompts': len(prompts) > 0,
                'prompt_count': len(prompts)
            }
        
        # Prop details
        for name, assets in self.assets.items():
            if not assets or assets[0].type != 'prop':
                continue
                
            angles = [asset.angle for asset in assets]
            prompts = [asset.prompt for asset in assets if asset.prompt]
            
            summary['props'][name] = {
                'available_angles': angles,
                'angle_count': len(angles),
                'has_prompts': len(prompts) > 0,
                'prompt_count': len(prompts)
            }
        
        return summary

    def validate_library(self) -> Dict:
        """Validate library structure and completeness"""
        validation_report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check if we have any assets
        if not self.assets:
            validation_report['warnings'].append("No assets loaded in library")
            return validation_report
        
        # Check each asset type
        character_count = 0
        location_count = 0
        prop_count = 0
        
        for name, assets in self.assets.items():
            if not assets:
                validation_report['warnings'].append(f"Asset '{name}' has no variations")
                continue
                
            asset_type = assets[0].type
            
            if asset_type == 'character':
                character_count += 1
            elif asset_type == 'location':
                location_count += 1
            elif asset_type == 'prop':
                prop_count += 1
            
            # Check for missing prompts
            has_prompts = any(asset.prompt for asset in assets)
            if not has_prompts:
                validation_report['warnings'].append(f"Asset '{name}' has no prompts")
            
            # Check for missing technical specs
            has_technical = any(asset.technical for asset in assets)
            if not has_technical:
                validation_report['warnings'].append(f"Asset '{name}' has no technical specifications")
        
        # Recommendations
        if character_count == 0:
            validation_report['recommendations'].append("Consider adding character assets")
        if location_count == 0:
            validation_report['recommendations'].append("Consider adding location assets")
        if prop_count == 0:
            validation_report['recommendations'].append("Consider adding prop assets")
        
        return validation_report


# Available angles from Phase 1
AVAILABLE_ANGLES = ['back_shot', 'close_up', 'profile_right', 'profile_left', 'wide_shot', 'master']

# Angle similarity mappings for fallback logic
ANGLE_FALLBACKS = {
    'close_up': ['wide_shot', 'master'],
    'medium_shot': ['wide_shot', 'close_up', 'master'],
    'wide_shot': ['master'],
    'profile_left': ['profile_right', 'master'],
    'profile_right': ['profile_left', 'master'],
    'back_shot': ['master'],
    'front': ['close_up', 'master'],
    'master': []
}

