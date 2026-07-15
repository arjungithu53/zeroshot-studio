#!/usr/bin/env python3
"""
Agent 1: Asset Generator
========================
Parses the script and extracts assets (characters, locations, props).
Includes human intervention points for quality assurance.

Flow:
1. Parse script content
2. Extract characters, locations, and props using Gemini with structured output
3. Human intervention: Review and refine extracted assets
4. Save structured asset records
"""

from google import genai
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Import Pydantic models from models.py
from backend.services.production.app.services.phase_1_agents.models import (
    Character,
    Location,
    Prop,
    ExtractedAssets
)

# Import prompts from prompts.py
from backend.services.production.app.services.phase_1_agents.prompts import Agent1Prompts

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger
from backend.services.production.app.utils.name_normalization import normalize_asset_name

# Initialize logger for this module
logger = get_logger(__name__)



class AssetType(str, Enum):
    """Types of assets that can be extracted"""
    CHARACTER = "character"
    LOCATION = "location"
    PROP = "prop"


class HumanFeedbackType(str, Enum):
    """Types of human feedback"""
    APPROVE = "approve"
    MODIFY = "modify"
    ADD_MISSING = "add_missing"
    REMOVE_DUPLICATE = "remove_duplicate"


class AssetGeneratorAgent:
    """
    Agent 1: Parses script and generates asset records

    This agent extracts characters, locations, and props from the script
    with human intervention checkpoints to ensure accuracy and completeness.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-3.1-pro-preview", csv_entity_mapping: Optional[Dict[str, Any]] = None, product_image_available: bool = False):
        """
        Initialize Asset Generator Agent

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
            csv_entity_mapping: Optional CSV entity mapping from shotlist (characters and locations)
            product_image_available: True when an uploaded product image exists for this project
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.script_content = ""
        self.csv_entity_mapping = csv_entity_mapping
        self.product_image_available = product_image_available
        self.extracted_assets = {
            "characters": [],
            "locations": [],
            "props": []
        }
        self.human_feedback_log = []

    def load_script(self, script_path: str = None, script_content: str = None) -> None:
        """
        Load script from file or direct content

        Args:
            script_path: Path to script file
            script_content: Direct script content
        """
        if script_path:
            with open(script_path, 'r', encoding='utf-8') as f:
                self.script_content = f.read()
            logger.info(f"✓ Script loaded from: {script_path}")
        elif script_content:
            self.script_content = script_content
            logger.info(f"✓ Script loaded from direct content ({len(script_content)} characters)")
        else:
            raise ValueError("Either script_path or script_content must be provided")

    def _create_asset_extraction_prompt(self) -> str:
        """Create comprehensive prompt for asset extraction using centralized prompts"""
        return Agent1Prompts.asset_extraction(
            self.script_content,
            self.csv_entity_mapping,
            product_image_available=self.product_image_available
        )

    def extract_assets(self) -> Dict[str, List[Dict]]:
        """
        Extract all assets from the script using Gemini with structured output

        Returns:
            Dictionary with characters, locations, and props
        """
        if not self.script_content:
            raise ValueError("No script content loaded. Call load_script() first.")

        prompt = self._create_asset_extraction_prompt()

        try:
            # Use structured output with Pydantic schema
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExtractedAssets,
                }
            )

            # Get the parsed objects directly
            parsed_assets: ExtractedAssets = response.parsed

            # Convert Pydantic models to dictionaries for backward compatibility
            self.extracted_assets = {
                "characters": [char.model_dump() for char in parsed_assets.characters],
                "locations": [loc.model_dump() for loc in parsed_assets.locations],
                "props": [prop.model_dump() for prop in parsed_assets.props]
            }

            # Add csv_name field if CSV mapping was used
            if self.csv_entity_mapping and self.csv_entity_mapping.get('has_entity_data'):
                # Add csv_name to characters (same as name since Agent 1 uses CSV names)
                for char in self.extracted_assets["characters"]:
                    char["csv_name"] = char["name"]

                # Add csv_name to locations (same as name since Agent 1 uses CSV names)
                for loc in self.extracted_assets["locations"]:
                    loc["csv_name"] = loc["name"]

            # Store CSV mapping if provided
            if self.csv_entity_mapping:
                self.extracted_assets["csv_entity_mapping"] = self.csv_entity_mapping

            print(f"✓ Extracted {len(self.extracted_assets.get('characters', []))} characters, "
                  f"{len(self.extracted_assets.get('locations', []))} locations, "
                  f"{len(self.extracted_assets.get('props', []))} props")

            return self.extracted_assets

        except Exception as e:
            logger.error(f"Asset extraction error: {e}")
            raise

    def validate_csv_mapping(self) -> Dict[str, Any]:
        """
        Validate that extracted assets match CSV entities.

        This method checks if the extracted characters and locations match
        the entities defined in the CSV shotlist. It identifies:
        - Missing entities: In CSV but not extracted
        - Extra entities: Extracted but not in CSV

        Returns:
            Dictionary with:
            - missing_csv_entities: Entities in CSV but not extracted
            - extra_entities: Entities extracted but not in CSV
            - validation_passed: bool indicating if validation passed
            - reason: Explanation if validation skipped
        """
        if not self.csv_entity_mapping or not self.csv_entity_mapping.get('has_entity_data'):
            return {
                "validation_passed": True,
                "reason": "No CSV entities to validate (backward compatibility mode)"
            }

        csv_chars = set(self.csv_entity_mapping.get('unique_characters', []))
        csv_locs = set(self.csv_entity_mapping.get('unique_locations', []))

        # Normalize extracted asset names for comparison
        extracted_chars = set([
            normalize_asset_name(c['name'])
            for c in self.extracted_assets.get('characters', [])
        ])
        extracted_locs = set([
            normalize_asset_name(l['name'])
            for l in self.extracted_assets.get('locations', [])
        ])

        missing_chars = csv_chars - extracted_chars
        missing_locs = csv_locs - extracted_locs
        extra_chars = extracted_chars - csv_chars
        extra_locs = extracted_locs - csv_locs

        validation_passed = (
            len(missing_chars) == 0 and
            len(missing_locs) == 0 and
            len(extra_chars) == 0 and
            len(extra_locs) == 0
        )

        result = {
            "missing_csv_entities": {
                "characters": list(missing_chars),
                "locations": list(missing_locs)
            },
            "extra_entities": {
                "characters": list(extra_chars),
                "locations": list(extra_locs)
            },
            "validation_passed": validation_passed
        }

        # Log validation results
        if not validation_passed:
            if missing_chars or missing_locs:
                logger.warning(f"Missing CSV entities - Characters: {missing_chars}, Locations: {missing_locs}")
            if extra_chars or extra_locs:
                logger.warning(f"Extra entities not in CSV - Characters: {extra_chars}, Locations: {extra_locs}")
        else:
            logger.info("✓ CSV entity mapping validation passed")

        return result

    def display_assets_for_review(self) -> None:
        """Display extracted assets in a readable format for human review"""

        logger.info("\n" + "="*60)
        logger.info("EXTRACTED ASSETS - HUMAN REVIEW REQUIRED")
        logger.info("="*60)

        # Display Characters
        logger.info("\n" + "─"*60)
        logger.info("🎭 CHARACTERS:")
        logger.info("─"*60)
        for i, char in enumerate(self.extracted_assets.get('characters', []), 1):
            logger.info(f"\n{i}. {char.get('name', 'Unknown')}")
            logger.info(f"   Description: {char.get('description', 'N/A')}")
            logger.info(f"   Role: {char.get('role', 'N/A')}")
            logger.info(f"   Importance: {char.get('importance', 'N/A')}")
            logger.info(f"   Scenes: {', '.join(char.get('scenes', ['N/A']))}")

        # Display Locations
        logger.info("\n" + "─"*60)
        logger.info("🗺LOCATIONS:")
        logger.info("─"*60)
        for i, loc in enumerate(self.extracted_assets.get('locations', []), 1):
            logger.info(f"\n{i}. {loc.get('name', 'Unknown')}")
            logger.info(f"   Description: {loc.get('description', 'N/A')}")
            logger.info(f"   Type: {loc.get('setting_type', 'N/A')}")
            logger.info(f"   Importance: {loc.get('importance', 'N/A')}")
            logger.info(f"   Scenes: {', '.join(loc.get('scenes', ['N/A']))}")

        # Display Props
        logger.info("\n" + "─"*60)
        logger.info("PROPS:")
        logger.info("─"*60)
        for i, prop in enumerate(self.extracted_assets.get('props', []), 1):
            logger.info(f"\n{i}. {prop.get('name', 'Unknown')}")
            logger.info(f"   Description: {prop.get('description', 'N/A')}")
            logger.info(f"   Material: {prop.get('material', 'N/A')}")
            logger.info(f"   Importance: {prop.get('importance', 'N/A')}")
            logger.info(f"   Scenes: {', '.join(prop.get('scenes', ['N/A']))}")

        logger.info("\n" + "="*60)

    def request_human_feedback(self) -> Dict[str, Any]:
        """
        Request human feedback on extracted assets

        Returns:
            Dictionary containing human feedback and modifications
        """
        logger.info("\n" + "🤔 "*30)
        logger.info("HUMAN INTERVENTION CHECKPOINT - AGENT 1")
        logger.info("🤔 "*30)

        logger.info("\nPlease review the extracted assets above and provide feedback:")
        logger.info("\nHuman should provide (via JSON file or dict):")
        logger.info("1. Are all characters present? Any missing?")
        logger.info("2. Are all locations identified? Any missing?")
        logger.info("3. Are all important props listed? Any missing?")
        logger.info("4. Are there any duplicates?")
        logger.info("5. Are descriptions accurate and detailed enough?")
        logger.info("6. Any modifications needed to existing assets?")

        logger.info("\nEXPECTED FEEDBACK FORMAT:")
        print("""
{
    "feedback_type": "approve/modify",
    "comments": "General comments about the extraction",
    "missing_assets": {
        "characters": [...],  // Add missing characters
        "locations": [...],   // Add missing locations
        "props": [...]        // Add missing props
    },
    "duplicates_to_remove": {
        "characters": ["char1", "char2"],  // Names to remove
        "locations": ["loc1"],
        "props": []
    },
    "modifications": {
        "characters": {
            "Character Name": {
                "field": "new_value",  // e.g., "description": "Better description"
            }
        },
        "locations": {...},
        "props": {...}
    }
}
        """)

        logger.info("\n" + "="*60)
        logger.info("⏸AGENT PAUSED - Waiting for human feedback...")
        logger.info("="*60)

        # Return placeholder - in real implementation, this would wait for actual human input
        return {
            "feedback_type": "pending",
            "message": "Human feedback required before proceeding to Agent 2"
        }

    def apply_human_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Apply human feedback to modify extracted assets

        Args:
            feedback: Dictionary containing human feedback
        """
        feedback_type = feedback.get('feedback_type', 'modify')

        # Log feedback
        self.human_feedback_log.append({
            "timestamp": datetime.now().isoformat(),
            "agent": "Agent 1: Asset Generator",
            "feedback": feedback
        })

        if feedback_type == "approve":
            return

        # Add missing assets
        missing = feedback.get('missing_assets', {})
        for asset_type in ['characters', 'locations', 'props']:
            if missing.get(asset_type):
                self.extracted_assets[asset_type].extend(missing[asset_type])

        # Remove duplicates
        duplicates = feedback.get('duplicates_to_remove', {})
        for asset_type in ['characters', 'locations', 'props']:
            if duplicates.get(asset_type):
                for dup_name in duplicates[asset_type]:
                    self.extracted_assets[asset_type] = [
                        asset for asset in self.extracted_assets[asset_type]
                        if asset.get('name') != dup_name
                    ]

        # Apply modifications
        modifications = feedback.get('modifications', {})
        for asset_type in ['characters', 'locations', 'props']:
            if modifications.get(asset_type):
                for asset_name, changes in modifications[asset_type].items():
                    for asset in self.extracted_assets[asset_type]:
                        if asset.get('name') == asset_name:
                            asset.update(changes)


    def run_full_pipeline(self, script_path: str = None, script_content: str = None) -> Dict[str, Any]:
        """
        Run the complete Agent 1 pipeline

        Args:
            script_path: Path to script file
            script_content: Direct script content

        Returns:
            Dictionary with extracted assets and status
        """
        # Step 1: Load script
        self.load_script(script_path=script_path, script_content=script_content)

        # Step 2: Extract assets
        self.extract_assets()

        # Step 3: Display for human review
        self.display_assets_for_review()

        # Step 4: Request human feedback
        feedback_info = self.request_human_feedback()

        return {
            "status": "pending_human_review",
            "extracted_assets": self.extracted_assets,
            "feedback_request": feedback_info,
            "next_step": "Provide human feedback via apply_human_feedback() method"
        }


def main():
    """Example usage of Agent 1"""

    # Initialize agent
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set")
    agent = AssetGeneratorAgent(api_key=api_key)

    # Example: Load script
    script_content = """
    FADE IN:

    INT. SPACE STATION - COMMAND CENTER - DAY

    CAPTAIN MAYA CHEN (30s, Asian-American, military uniform) stands at the main console,
    staring at the warning lights flashing across the screen.

    MAYA
    (into comms)
    This is Captain Chen. We have a situation.

    The doors slide open. DR. JAMES WRIGHT (40s, British, lab coat) rushes in,
    carrying a data pad.

    WRIGHT
    The containment field is failing. We have maybe ten minutes.

    MAYA
    (determined)
    Then we don't waste a second.

    She grabs her helmet from the equipment rack and heads for the airlock.

    EXT. SPACE STATION - EXTERIOR - CONTINUOUS

    Maya floats in zero gravity, her magnetic boots keeping her anchored to the hull.
    Earth gleams blue and beautiful in the background.

    FADE OUT.
    """

    # Run pipeline
    result = agent.run_full_pipeline(script_content=script_content)

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline Status: {result['status']}")
    logger.info(f"Next Step: {result['next_step']}")
    logger.info(f"{'='*60}")

    # Example: Apply human feedback
    logger.info("\n\n" + "="*60)
    logger.info("EXAMPLE: Applying Human Feedback")
    logger.info("="*60)

    example_feedback = {
        "feedback_type": "modify",
        "comments": "Added missing spacecraft prop and refined Maya's description",
        "missing_assets": {
            "props": [
                {
                    "name": "Space Station",
                    "description": "Large orbital research facility with rotating habitat ring",
                    "material": "titanium and reinforced glass",
                    "size": "large",
                    "condition": "functional but showing wear",
                    "usage": "Main setting for the story",
                    "scenes": ["Scene 1", "Scene 2"],
                    "importance": "critical"
                }
            ]
        },
        "duplicates_to_remove": {},
        "modifications": {
            "characters": {
                "MAYA CHEN": {
                    "description": "Asian-American woman in her early 30s, athletic build, determined expression, wearing a crisp military-style uniform with captain's insignia"
                }
            }
        }
    }

    agent.apply_human_feedback(example_feedback)

    # Save final results

    logger.info("\nAGENT 1 COMPLETE - Ready for Agent 2")


if __name__ == "__main__":
    main()
