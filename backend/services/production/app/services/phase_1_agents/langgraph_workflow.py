#!/usr/bin/env python3
"""
LangGraph Workflow Orchestrator for Phase 1 Agents
===================================================
Coordinates the 4-agent pipeline using LangGraph state management.

Flow:
Script → Agent 1 (Extract) → Agent 2 (Review) → Agent 3 (Generate) → Agent 4 (Optimize) → Final Output
"""


import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))
from backend.shared.utils.logger import get_logger

# Initialize logger for this module
logger = get_logger(__name__)

import os
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from backend.services.production.app.services.phase_1_agents.workflow_state import Phase1State
from backend.services.production.app.services.project_service import ProjectService
from backend.services.production.app.services.assets_collection_service import AssetsCollectionService
# PipelineService is imported inside _update_job_agent_status to avoid MongoDB fork issues

# Import all agents
from backend.services.production.app.services.phase_1_agents.agent_1_asset_generator import AssetGeneratorAgent
from backend.services.production.app.services.phase_1_agents.agent_2_asset_reviewer import AssetReviewerAgent
from backend.services.production.app.services.phase_1_agents.agent_3_prompt_generator import PromptGeneratorAgent
from backend.services.production.app.services.phase_1_agents.agent_4_prompt_optimizer import PromptOptimizerAgent
from backend.services.production.app.services.phase_1_agents.agent_5_image_generator import ImageGeneratorAgent
from backend.services.production.app.services.phase_1_agents.agent_6_image_reviewer import ImageReviewerAgent
from backend.services.production.app.services.phase_1_agents.agent_7_image_editor import ImageEditAgent
from backend.services.production.app.services.phase_1_agents.agent_8_variation_generator import VariationGeneratorAgent

# Initialize services for saving agent outputs
# Note: PipelineService is created inside _update_job_agent_status to avoid MongoDB fork issues
project_service = ProjectService()
assets_collection_service = AssetsCollectionService()


# ============================================================================
# HELPER TYPES / EXCEPTIONS
# ============================================================================


class AssetsCollectionNotFoundError(Exception):
    """Raised when the assets collection expected for movie workflows is missing."""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _merge_generated_images_with_versioning(
    existing_images: Dict[str, list],
    new_images: Dict[str, list],
    assets_regenerated: list
) -> Dict[str, list]:
    """
    Merge newly regenerated images with existing ones, preserving old versions.

    Args:
        existing_images: Existing generated images from state
        new_images: Newly generated images from agent
        assets_regenerated: List of asset IDs that were regenerated (format: "asset_type:asset_id")

    Returns:
        Merged images with versioning
    """
    from datetime import datetime
    from copy import deepcopy

    # Deep copy to avoid mutation
    merged = deepcopy(existing_images)

    # Parse assets_regenerated into a set for fast lookup
    regenerated_set = set()
    for asset_key in assets_regenerated:
        # Format: "characters:char-001" or "locations:loc-001"
        if ':' in asset_key:
            asset_type_plural, asset_id = asset_key.split(':', 1)
            regenerated_set.add((asset_type_plural, asset_id))

    # Process each asset type
    for asset_type in ["characters", "locations", "props"]:
        # Get newly generated images for this type
        new_assets = new_images.get(asset_type, [])

        for new_asset in new_assets:
            asset_id = new_asset.get('id')
            asset_name = new_asset.get('name')

            # Check if this asset was regenerated
            if (asset_type, asset_id) not in regenerated_set:
                continue

            # Find existing asset in merged images
            existing_asset_idx = None
            for idx, existing_asset in enumerate(merged.get(asset_type, [])):
                if existing_asset.get('id') == asset_id:
                    existing_asset_idx = idx
                    break

            if existing_asset_idx is not None:
                # Asset exists - create versioned backup and replace
                existing_asset = merged[asset_type][existing_asset_idx]

                # Create versions array if it doesn't exist
                if 'versions' not in existing_asset:
                    existing_asset['versions'] = []

                # Move current images to versions array
                current_images = existing_asset.get('images', [])
                if current_images:
                    version_entry = {
                        'version': len(existing_asset['versions']),
                        'images': current_images,
                        'archived_at': datetime.now().isoformat(),
                        'prompt': existing_asset.get('prompt', ''),
                        'negative_prompt': existing_asset.get('negative_prompt', ''),
                        'technical_specs': existing_asset.get('technical_specs', {})
                    }
                    existing_asset['versions'].append(version_entry)
                    logger.info(f"   📦 Archived v{version_entry['version']} for {asset_name}")

                # Replace with new images
                existing_asset['images'] = new_asset.get('images', [])
                existing_asset['prompt'] = new_asset.get('prompt', '')
                existing_asset['negative_prompt'] = new_asset.get('negative_prompt', '')
                existing_asset['technical_specs'] = new_asset.get('technical_specs', {})
                existing_asset['regenerated_at'] = datetime.now().isoformat()
                existing_asset['current_version'] = len(existing_asset['versions'])

                logger.info(f"   ✓ Updated {asset_name} (now v{existing_asset['current_version']})")
            else:
                # New asset - just add it
                merged[asset_type].append(new_asset)
                logger.info(f"   ✓ Added new asset {asset_name}")

    return merged


def _save_agent_output(state: Phase1State, agent_number: int, status: str, output: Optional[Dict] = None, error: Optional[str] = None) -> None:
    """
    Save agent output to the appropriate service based on workflow mode.

    Supports two modes:
    - Legacy mode: project_id is set → save to project_service
    - Movie mode: assets_collection_id is set → save to assets_collection_service

    Args:
        state: Workflow state
        agent_number: Agent number (1-8)
        status: Agent status (pending, running, completed, failed)
        output: Optional agent output data
        error: Optional error message
    """
    try:
        # Movie workflow mode - save to assets_collection
        if state.get("assets_collection_id"):
            assets_collection_service.update_agent_output(
                assets_collection_id=state["assets_collection_id"],
                agent_number=agent_number,
                status=status,
                output=output,
                error=error
            )
            logger.info(f"Agent {agent_number} output saved to assets collection {state['assets_collection_id']}")

        # Legacy workflow mode - save to project
        elif state.get("project_id"):
            project_service.update_agent_output(
                project_id=state["project_id"],
                agent_number=agent_number,
                status=status,
                output=output,
                error=error
            )
            logger.info(f"Agent {agent_number} output saved to project {state['project_id']}")

        else:
            logger.warning(f"No project_id or assets_collection_id in state - cannot save Agent {agent_number} output")

    except Exception as e:
        logger.error(f"Failed to save Agent {agent_number} output: {e}")


def _update_job_agent_status(state: Phase1State, agent_number: int, status: str) -> None:
    """
    Update the job record with the current agent status.

    This allows the /movies/{movie_id}/status endpoint to reflect real-time progress.

    Args:
        state: Workflow state containing job_id
        agent_number: Agent number (1-8)
        status: Agent status (pending, running, completed, failed)
    """
    try:
        job_id = state.get("job_id")
        if not job_id:
            logger.warning("No job_id in state - cannot update job agent status")
            return

        # Create a fresh PipelineService instance to avoid MongoDB fork issues
        # This is necessary because Celery workers fork the process
        from backend.services.production.app.services.pipeline_service import PipelineService
        _pipeline_service = PipelineService()

        # Build update dict with the specific agent status
        update_data = {
            f"agent{agent_number}_status": status,
            "current_agent": f"agent_{agent_number}" if status == "running" else state.get("current_agent", ""),
        }

        # If agent completed, set current_agent to next agent
        if status == "completed" and agent_number < 8:
            update_data["current_agent"] = f"agent_{agent_number + 1}"

        _pipeline_service.update_job_status(job_id, status="running", **update_data)
        logger.info(f"Job {job_id} updated: agent{agent_number}_status={status}")

    except Exception as e:
        logger.error(f"Failed to update job agent status: {e}")


def prepare_csv_entity_mapping(shotlist_shots) -> Optional[Dict[str, Any]]:
    """
    Pre-extract unique entities from shotlist CSV before Agent 1 runs.

    This function processes the shotlist to identify all unique characters
    and locations that should be used by Agent 1 as the source of truth.

    Args:
        shotlist_shots: List of ShotData objects from parsed shotlist CSV

    Returns:
        CSV entity mapping dictionary or None if no shotlist
    """
    if not shotlist_shots:
        logger.info("No shotlist provided - Agent 1 will run in legacy mode")
        return None

    try:
        from backend.services.production.app.utils.csv_parser import extract_unique_entities_from_shotlist

        mapping = extract_unique_entities_from_shotlist(shotlist_shots)

        if mapping['has_entity_data']:
            logger.info(f"✓ CSV entity mapping extracted:")
            logger.info(f"  - Characters ({len(mapping['unique_characters'])}): {', '.join(mapping['unique_characters'][:5])}{' ...' if len(mapping['unique_characters']) > 5 else ''}")
            logger.info(f"  - Locations ({len(mapping['unique_locations'])}): {', '.join(mapping['unique_locations'][:5])}{' ...' if len(mapping['unique_locations']) > 5 else ''}")
        else:
            logger.info("No character/location data in CSV - Agent 1 will extract from script only")

        return mapping

    except Exception as e:
        logger.error(f"Failed to extract CSV entity mapping: {e}")
        return None


# ============================================================================
# NODE FUNCTIONS - Each agent as a LangGraph node
# ============================================================================

def agent_1_node(state: Phase1State) -> Phase1State:
    """
    Agent 1: Asset Generator Node
    Extracts characters, locations, and props from script
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 1: ASSET GENERATOR")
    logger.info("="*60)

    try:
        # Initialize agent with CSV entity mapping and product image flag
        api_key = os.getenv("GEMINI_API_KEY")
        csv_mapping = state.get("csv_entity_mapping")
        product_image_s3_url = state.get("product_image_s3_url")
        agent = AssetGeneratorAgent(
            api_key=api_key,
            csv_entity_mapping=csv_mapping,
            product_image_available=bool(product_image_s3_url)
        )

        # Load script
        agent.load_script(script_content=state["script_content"])

        # Extract assets
        extracted_assets = agent.extract_assets()

        # Attach uploaded product image URL to the correct prop
        if product_image_s3_url:
            # Check if Agent 1 flagged a prop as the product
            product_prop_from_agent1 = next(
                (p for p in extracted_assets.get("props", []) if p.get("is_product")),
                None
            )
            if product_prop_from_agent1:
                product_prop_from_agent1["pre_generated_image_url"] = product_image_s3_url
                logger.info(f"✓ Uploaded product image attached to prop '{product_prop_from_agent1['name']}'")
            else:
                # Agent 1 didn't flag any prop — inject a fallback PRODUCT prop
                product_shot_numbers = (state.get("csv_entity_mapping") or {}).get("product_shot_numbers", [])
                product_scenes = sorted(set(
                    f"Scene {sn.split('.')[0]}" for sn in product_shot_numbers if '.' in sn
                ))
                fallback_prop = {
                    "id": "product-prop-001",
                    "name": "PRODUCT",
                    "description": "The featured product for this video ad",
                    "material": "N/A",
                    "size": "medium",
                    "condition": "new",
                    "usage": "Featured product appearing in designated shots",
                    "scenes": product_scenes,
                    "importance": "critical",
                    "is_product": True,
                    "pre_generated_image_url": product_image_s3_url,
                }
                extracted_assets.setdefault("props", []).append(fallback_prop)
                logger.info("✓ Fallback PRODUCT prop injected (Agent 1 did not flag one)")
        elif state.get("product_prop"):
            # Legacy path: product_prop pre-built in state (no uploaded image)
            extracted_assets.setdefault("props", [])
            extracted_assets["props"].append(state["product_prop"])
            logger.info("✓ Product prop injected into extracted_assets (legacy path)")

        # Validate CSV mapping if applicable
        validation = agent.validate_csv_mapping()
        if not validation.get('validation_passed'):
            logger.warning(f"CSV validation issues found: {validation}")

        # Apply human feedback if provided
        if state.get("agent1_human_feedback"):
            agent.apply_human_feedback(state["agent1_human_feedback"])
        else:
            # Auto-approve for demo
            agent.apply_human_feedback({
                "feedback_type": "approve",
                "comments": "Auto-approved"
            })

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=1,
            status="completed",
            output={"extracted_assets": agent.extracted_assets}
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=1, status="completed")

        # Update state
        return {
            **state,
            "extracted_assets": agent.extracted_assets,
            "agent1_status": "completed",
            "current_agent": "agent_2",
        }

    except Exception as e:
        logger.error(f"Agent 1 failed: {e}")

        # Save error to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=1,
            status="failed",
            error=str(e)
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=1, status="failed")

        return {
            **state,
            "agent1_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_2_node(state: Phase1State) -> Phase1State:
    """
    Agent 2: Asset Reviewer Node
    Reviews and enhances assets from Agent 1
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 2: ASSET REVIEWER")
    logger.info("="*60)

    try:
        # Initialize agent
        api_key = os.getenv("GEMINI_API_KEY")
        agent = AssetReviewerAgent(api_key=api_key)

        # Set assets and script from state
        agent.original_assets = state["extracted_assets"]
        agent.script_content = state["script_content"]

        # Review assets
        review_results = agent.review_assets()

        # Apply enhancements
        feedback = state.get("agent2_human_feedback")
        if not feedback:
            # Auto-approve all enhancements
            completeness = review_results.get('completeness_check', {})
            feedback = {
                "approve_enhancements": True,
                "approve_missing_additions": {
                    "characters": [item['name'] for item in completeness.get('missing_characters', [])],
                    "locations": [item['name'] for item in completeness.get('missing_locations', [])],
                    "props": [item['name'] for item in completeness.get('missing_props', [])],
                }
            }

        agent.apply_enhancements(feedback)

        # Strip any characters/locations Agent 2 added that are not in the CSV.
        # Agent 2 has no CSV context, so its completeness check can invent phantom
        # entities (e.g. "Second Friend"). Only props are unconstrained by the CSV.
        csv_entity_mapping = state.get("csv_entity_mapping") or {}
        if csv_entity_mapping.get("has_entity_data"):
            allowed_characters = {c.upper() for c in csv_entity_mapping.get("unique_characters", [])}
            allowed_locations = {l.upper() for l in csv_entity_mapping.get("unique_locations", [])}

            enhanced = agent.enhanced_assets or {}

            original_char_count = len(enhanced.get("characters", []))
            original_loc_count = len(enhanced.get("locations", []))

            enhanced["characters"] = [
                c for c in enhanced.get("characters", [])
                if c.get("name", "").upper() in allowed_characters
            ]
            enhanced["locations"] = [
                l for l in enhanced.get("locations", [])
                if l.get("name", "").upper() in allowed_locations
            ]

            removed_chars = original_char_count - len(enhanced["characters"])
            removed_locs = original_loc_count - len(enhanced["locations"])
            if removed_chars or removed_locs:
                logger.warning(
                    f"CSV filter removed {removed_chars} phantom character(s) and "
                    f"{removed_locs} phantom location(s) added by Agent 2"
                )
            else:
                logger.info("CSV filter: all Agent 2 characters/locations are within CSV bounds")

            agent.enhanced_assets = enhanced

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=2,
            status="completed",
            output={"review_results": review_results, "enhanced_assets": agent.enhanced_assets}
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=2, status="completed")

        # Update state
        return {
            **state,
            "review_results": review_results,
            "enhanced_assets": agent.enhanced_assets,
            "agent2_status": "completed",
            "current_agent": "agent_3",
        }

    except Exception as e:
        logger.error(f"Agent 2 failed: {e}")
        _save_agent_output(
            state=state,
            agent_number=2,
            status="failed",
            error=str(e)
        )
        _update_job_agent_status(state, agent_number=2, status="failed")
        return {
            **state,
            "agent2_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_3_node(state: Phase1State) -> Phase1State:
    """
    Agent 3: Prompt Generator Node
    Generates image generation prompts for each asset
    """
    logger.info("\n" + "="*60)
    logger.info("✨ AGENT 3: PROMPT GENERATOR")
    logger.info("="*60)

    try:
        # Initialize agent
        api_key = os.getenv("GEMINI_API_KEY")
        visual_style = state.get("visual_style", "realistic")
        agent = PromptGeneratorAgent(api_key=api_key, visual_style=visual_style)

        # Set enhanced assets from state
        agent.enhanced_assets = state["enhanced_assets"]

        # Generate prompts
        generated_prompts = agent.generate_prompts()

        # Ensure prompts were generated
        if not generated_prompts or not agent.generated_prompts:
            raise ValueError("Prompt generation returned None or empty result")

        # Apply human feedback if provided
        feedback = state.get("agent3_human_feedback", {"approve_all": True})
        agent.apply_human_feedback(feedback)

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=3,
            status="completed",
            output={"generated_prompts": agent.generated_prompts}
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=3, status="completed")

        # Update state
        return {
            **state,
            "generated_prompts": agent.generated_prompts,
            "agent3_status": "completed",
            "current_agent": "agent_4",
        }

    except Exception as e:
        logger.error(f"Agent 3 failed: {e}")
        import traceback
        traceback.print_exc()
        _save_agent_output(
            state=state,
            agent_number=3,
            status="failed",
            error=str(e)
        )
        _update_job_agent_status(state, agent_number=3, status="failed")
        return {
            **state,
            "agent3_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_4_node(state: Phase1State) -> Phase1State:
    """
    Agent 4: Prompt Optimizer Node
    Optimizes and finalizes prompts from Agent 3
    """
    logger.info("\n" + "="*60)
    logger.info("🔧 AGENT 4: PROMPT OPTIMIZER")
    logger.info("="*60)

    try:
        # Initialize agent
        api_key = os.getenv("GEMINI_API_KEY")
        visual_style = state.get("visual_style", "realistic")
        agent = PromptOptimizerAgent(api_key=api_key, visual_style=visual_style)

        # Set initial prompts from state
        agent.initial_prompts = state["generated_prompts"]

        # Optimize prompts
        optimized_prompts = agent.optimize_prompts()

        # Apply human feedback if provided
        feedback = state.get("agent4_human_feedback", {"approve_all": True})
        agent.apply_human_feedback(feedback)

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=4,
            status="completed",
            output={"optimized_prompts": agent.optimized_prompts}
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=4, status="completed")

        # Update state
        return {
            **state,
            "optimized_prompts": agent.optimized_prompts,
            "agent4_status": "completed",
            "current_agent": "agent_5",
        }

    except Exception as e:
        logger.error(f"Agent 4 failed: {e}")
        _save_agent_output(
            state=state,
            agent_number=4,
            status="failed",
            error=str(e)
        )
        _update_job_agent_status(state, agent_number=4, status="failed")
        return {
            **state,
            "agent4_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_5_node(state: Phase1State) -> Phase1State:
    """
    Agent 5: Image Generator Node
    Generates images using Google Imagen 4.0 API
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 5: IMAGE GENERATOR")
    logger.info("="*60)

    try:
        # Initialize agent
        api_key = os.getenv("GOOGLE_API_KEY")
        agent = ImageGeneratorAgent(api_key=api_key)

        # Set final prompts from state
        agent.final_prompts = state["optimized_prompts"]

        # Check if we're in regeneration mode
        assets_to_regenerate = state.get("needs_regeneration_assets")

        # Load existing generated_images from state if in regeneration mode
        existing_generated_images = None
        if assets_to_regenerate:
            existing_generated_images = state.get("generated_images", {
                "characters": [],
                "locations": [],
                "props": []
            })
            logger.info(f"\n📦 Loading existing generated images for merge")
            logger.info(f"   Characters: {len(existing_generated_images.get('characters', []))}")
            logger.info(f"   Locations: {len(existing_generated_images.get('locations', []))}")
            logger.info(f"   Props: {len(existing_generated_images.get('props', []))}")

        # Exclude all product props from Imagen — they have an uploaded image (pre_generated_image_url)
        if "props" in agent.final_prompts:
            before_count = len(agent.final_prompts["props"])
            agent.final_prompts["props"] = [
                p for p in agent.final_prompts["props"]
                if not p.get("is_product")
            ]
            excluded = before_count - len(agent.final_prompts["props"])
            if excluded:
                logger.info(f"✓ {excluded} product prop(s) excluded from Imagen generation")

        # Generate images (all assets or only specific ones if regenerating)
        agent.generate_images(assets_to_regenerate=assets_to_regenerate)

        # Inject uploaded product images directly into results (bypassing Imagen)
        from datetime import datetime as _dt
        all_props = state.get("extracted_assets", {}).get("props", [])
        for prop in all_props:
            if prop.get("is_product") and prop.get("pre_generated_image_url"):
                product_image_entry = {
                    "id": prop["id"],
                    "name": prop["name"],
                    "images": [{
                        "index": 1,
                        "url": prop["pre_generated_image_url"],
                        "s3_url": prop["pre_generated_image_url"],
                        "filename": f"product_{prop['id']}.png",
                        "source": "uploaded_product_image"
                    }],
                    "prompt": "Uploaded product image — not AI-generated",
                    "is_product": True,
                    "generation_timestamp": _dt.now().isoformat()
                }
                agent.generated_images.setdefault("props", [])
                agent.generated_images["props"].append(product_image_entry)
                logger.info(f"✓ Uploaded product image injected for prop '{prop['name']}'")

        # Print failed generations summary
        if agent.failed_generations:
            logger.warning(f"\n{len(agent.failed_generations)} asset(s) failed to generate:")
            for failed in agent.failed_generations:
                logger.error(f"   • {failed['asset_type']}: {failed['asset_name']} - {failed['reason']}")
            logger.error(f"\n💡 Use the retry endpoint to regenerate failed assets")

        # Merge newly generated images with existing ones (in regeneration mode)
        final_generated_images = agent.generated_images
        if existing_generated_images:
            logger.info(f"\n🔀 Merging newly regenerated images with existing images...")
            final_generated_images = _merge_generated_images_with_versioning(
                existing_images=existing_generated_images,
                new_images=agent.generated_images,
                assets_regenerated=assets_to_regenerate
            )
            logger.info(f"   ✓ Merge complete")
            logger.info(f"   Total characters: {len(final_generated_images.get('characters', []))}")
            logger.info(f"   Total locations: {len(final_generated_images.get('locations', []))}")
            logger.info(f"   Total props: {len(final_generated_images.get('props', []))}")

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=5,
            status="completed",
            output={
                "generated_images": final_generated_images,
                "failed_generations": agent.failed_generations,
            }
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=5, status="completed")

        # Update state
        return {
            **state,
            "generated_images": final_generated_images,  # Use merged images
            "failed_generations": agent.failed_generations,
            "agent5_status": "completed",
            "current_agent": "agent_6",
            # Clear needs_regeneration_assets after regeneration is complete
            "needs_regeneration_assets": [],
        }

    except Exception as e:
        logger.error(f"Agent 5 failed: {e}")
        import traceback
        traceback.print_exc()
        _save_agent_output(
            state=state,
            agent_number=5,
            status="failed",
            error=str(e)
        )
        _update_job_agent_status(state, agent_number=5, status="failed")
        return {
            **state,
            "agent5_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_6_node(state: Phase1State) -> Phase1State:
    """
    Agent 6: Image Reviewer Node
    Reviews generated images using AI (Gemini Vision)
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 6: IMAGE REVIEWER (AI CRITIC)")
    logger.info("="*60)

    try:
        # Initialize agent
        api_key = os.getenv("GEMINI_API_KEY")
        agent = ImageReviewerAgent(api_key=api_key)

        # Set data from state
        logger.info(f"\nDEBUG: Type of state['generated_images']: {type(state['generated_images'])}")
        logger.info(f"DEBUG: Type of state['optimized_prompts']: {type(state['optimized_prompts'])}")

        agent.final_prompts = state["optimized_prompts"]

        # Check if we should do selective review (only re-review recently edited assets)
        recently_edited_assets = state.get("recently_edited_asset_ids")

        # Bug fix: When re-reviewing after Agent 7 edits, overlay the edited image local paths
        # onto generated_images so Agent 6 actually reviews the edited files, not the originals.
        edited_images_in_state = state.get("edited_images", {})
        if edited_images_in_state and recently_edited_assets:
            import copy as _copy
            merged_images = _copy.deepcopy(state["generated_images"])
            for _edit_key, _edit_data in edited_images_in_state.items():
                if _edit_data.get('edit_skipped'):
                    continue
                _asset_id = _edit_data.get('asset_id')
                _asset_type = _edit_data.get('asset_type')
                _edited_imgs = _edit_data.get('edited_images', [])
                if not _asset_id or not _asset_type or not _edited_imgs:
                    continue
                for _asset in merged_images.get(_asset_type, []):
                    if _asset.get('id') == _asset_id:
                        for _edited_img in _edited_imgs:
                            _img_index = _edited_img.get('index', 1)
                            _local_path = _edited_img.get('local_path')
                            if _local_path and os.path.exists(_local_path):
                                for _gen_img in _asset.get('images', []):
                                    if _gen_img.get('index') == _img_index:
                                        # Point Agent 6 to the edited local file
                                        _gen_img['s3_url'] = _local_path
                                        _gen_img['url'] = _local_path
                                        _gen_img['local_path'] = _local_path
                                        logger.info(f"   ✓ Overlaying edited image for {_asset.get('name')}: {_local_path}")
                                        break
                        break
            agent.generated_images = merged_images
        else:
            agent.generated_images = state["generated_images"]

        logger.info(f"DEBUG: After assignment - Type of agent.generated_images: {type(agent.generated_images)}")
        logger.info(f"DEBUG: After assignment - Type of agent.final_prompts: {type(agent.final_prompts)}")
        logger.info(f"DEBUG: After assignment - Type of agent.review_results: {type(agent.review_results)}")

        # Get previous review results if doing selective review
        previous_reviews = state.get("image_reviews") if recently_edited_assets else None

        # Review images (all or selective based on context)
        result = agent.review_all_images(
            assets_to_review=recently_edited_assets,
            previous_reviews=previous_reviews
        )

        logger.info(f"\nDEBUG: After review_all_images()")
        logger.info(f"DEBUG: Return value type: {type(result)}")
        logger.info(f"DEBUG: agent.review_results type: {type(agent.review_results)}")
        logger.info(f"DEBUG: Are they the same object? {result is agent.review_results}")

        # Determine which assets need editing or regeneration
        needs_editing = []
        needs_regeneration = []

        for asset_type in ["characters", "locations", "props"]:
            for asset_data in agent.review_results.get(asset_type, []):
                asset_id = asset_data.get('id')
                reviews = asset_data.get('reviews', [])
                # reviews is a list of review dicts (one per image variation)
                if isinstance(reviews, list):
                    for review in reviews:
                        decision = review.get("decision")
                        if decision == "needs_edit":
                            needs_editing.append(f"{asset_type}:{asset_id}")
                            break  # Only add once per asset
                        elif decision == "regenerate":
                            needs_regeneration.append(f"{asset_type}:{asset_id}")
                            break  # Only add once per asset
                else:
                    # Fallback for single review dict
                    decision = reviews.get("decision")
                    if decision == "needs_edit":
                        needs_editing.append(f"{asset_type}:{asset_id}")
                    elif decision == "regenerate":
                        needs_regeneration.append(f"{asset_type}:{asset_id}")

        # Handle regeneration - rewrite prompts if needed
        regenerated_prompts = None
        if needs_regeneration:
            logger.info(f"\n{len(needs_regeneration)} asset(s) need regeneration")
            try:
                regenerated_prompts = agent.rewrite_prompts_for_regeneration(state["optimized_prompts"])
            except Exception as e:
                logger.error(f"Failed to rewrite prompts: {e}")
                import traceback
                traceback.print_exc()

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=6,
            status="completed",
            output={
                "image_reviews": agent.review_results,
                "needs_editing_assets": needs_editing,
                "needs_regeneration_assets": needs_regeneration,
            }
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=6, status="completed")

        # If no editing and no regeneration needed, mark Agent 7 as skipped
        if not needs_editing and not needs_regeneration:
            _save_agent_output(
                state=state,
                agent_number=7,
                status="skipped",
                output={"reason": "No images need editing - all images approved by Agent 6"}
            )
            _update_job_agent_status(state, agent_number=7, status="skipped")
            logger.info("Agent 7 marked as skipped (no editing needed)")

        # Determine next agent based on review decisions
        # Priority: regeneration > editing > human checkpoint
        if needs_regeneration:
            next_agent = "regeneration_router"
            logger.info(f"   → Routing to regeneration (will go back to Agent 5)")
        elif needs_editing:
            next_agent = "agent_7"
            logger.info(f"   → Routing to Agent 7 for editing")
        else:
            next_agent = "agent_8"
            logger.info(f"   → All images approved, routing directly to Agent 8")

        # Update state
        return {
            **state,
            "image_reviews": agent.review_results,
            "agent6_status": "completed",
            "agent7_status": "skipped" if not needs_editing else "pending",
            "needs_editing_assets": needs_editing,
            "needs_regeneration_assets": needs_regeneration,
            "regenerated_prompts": regenerated_prompts,
            "current_agent": next_agent,
            "recently_edited_asset_ids": None,  # Clear the selective review flag after processing
        }

    except Exception as e:
        logger.error(f"Agent 6 failed: {e}")
        import traceback
        traceback.print_exc()
        _save_agent_output(
            state=state,
            agent_number=6,
            status="failed",
            error=str(e)
        )
        _update_job_agent_status(state, agent_number=6, status="failed")
        return {
            **state,
            "agent6_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def agent_7_node(state: Phase1State) -> Phase1State:
    """
    Agent 7: Image Editor Node
    Edits images that need improvements using SeeDream 4 Edit
    """
    logger.info("\n" + "="*60)
    logger.info("✏AGENT 7: IMAGE EDITOR (AUTO-FIX)")
    logger.info("="*60)

    try:
        # Initialize agent
        gemini_key = os.getenv("GEMINI_API_KEY")
        agent = ImageEditAgent(gemini_api_key=gemini_key)

        # Set data from state
        agent.review_results = state["image_reviews"]
        agent.generated_images = state["generated_images"]
        agent.final_prompts = state["optimized_prompts"]

        # Build edit instructions from needs_editing_assets
        agent.edit_instructions = {}
        for asset_key in state.get("needs_editing_assets", []):
            # asset_key format: "characters:asset_id"
            parts = asset_key.split(":")
            if len(parts) == 2:
                asset_type, asset_id = parts

                # Get the list of reviews for this asset by searching through the list
                reviews_list = []
                for asset_data in agent.review_results.get(asset_type, []):
                    if asset_data.get('id') == asset_id or asset_data.get('name') == asset_id:
                        reviews_list = asset_data.get('reviews', [])
                        break

                # Get the images for this asset by searching through the list
                asset_images = []
                for asset_data in agent.generated_images.get(asset_type, []):
                    if asset_data.get('id') == asset_id or asset_data.get('name') == asset_id:
                        asset_images = asset_data.get('images', [])
                        break

                # Process each review that needs editing
                for review in reviews_list:
                    if isinstance(review, dict) and review.get('decision') == 'needs_edit':
                        image_index = review.get('image_index', 1)

                        # Find the corresponding image path (S3 URL or local path)
                        image_path = None
                        for img_data in asset_images:
                            if img_data.get('index') == image_index:
                                image_path = (
                                    img_data.get('local_path') or
                                    img_data.get('url') or
                                    img_data.get('s3_url')
                                )
                                break

                        if image_path:
                            # Create a unique key for this specific image
                            edit_key = f"{asset_key}:image_{image_index}"

                            # Get the original generation prompt for this asset
                            original_prompt = ''
                            for p_data in agent.final_prompts.get(asset_type, []):
                                if p_data.get('id') == asset_id:
                                    original_prompt = p_data.get('final_prompt', {}).get('prompt', '')
                                    break

                            # Check if this is the product prop (fidelity lock in Agent 7)
                            is_product = False
                            for gen_asset in agent.generated_images.get(asset_type, []):
                                if gen_asset.get('id') == asset_id:
                                    is_product = gen_asset.get('is_product', False)
                                    break

                            # Build edit_data in the flat format expected by generate_edit_prompt.
                            # The raw review dict nests these under 'feedback' and 'assessment',
                            # but Agent 7 looks for flat keys: edit_instructions, issues, original_prompt.
                            edit_data = {
                                'asset_id': asset_id,
                                'asset_name': review.get('asset_name', ''),
                                'asset_type': asset_type,
                                'image_path': image_path,
                                'original_prompt': original_prompt,
                                'edit_instructions': review.get('feedback', {}).get('for_edit', ''),
                                'issues': review.get('assessment', {}).get('issues', []),
                                'is_product': is_product,
                            }

                            agent.edit_instructions[edit_key] = edit_data

        # Edit images that need editing
        agent.edit_images()

        # Build two lists from actual edit results (not from the input request list):
        # 1. actually_edited_ids: assets where a real edit was applied (for Agent 6 re-review)
        # 2. edited_asset_ids: all assets that were processed (for logging / MongoDB)
        actually_edited_ids = []
        edited_asset_ids = []
        for _edit_key, _edit_data in agent.edited_images.items():
            _asset_id = _edit_data.get('asset_id')
            if not _asset_id:
                continue
            if _asset_id not in edited_asset_ids:
                edited_asset_ids.append(_asset_id)
            # Only count as "actually edited" when a real edit was applied (not skipped)
            if not _edit_data.get('edit_skipped') and _edit_data.get('edited_images'):
                if _asset_id not in actually_edited_ids:
                    actually_edited_ids.append(_asset_id)

        logger.info(f"\nActually edited (real edits applied): {len(actually_edited_ids)} asset(s): {actually_edited_ids}")
        logger.info(f"Processed total (including skipped): {len(edited_asset_ids)} asset(s): {edited_asset_ids}")

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=7,
            status="completed",
            output={"edited_images": agent.edited_images, "edited_asset_ids": edited_asset_ids}
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=7, status="completed")

        # Determine next agent based on context
        # If this is a human-requested edit (edit_prompt decision), return to human checkpoint
        # Otherwise, loop back to Agent 6 for auto re-review
        human_decision = state.get("human_approval_decision")
        
        # Get edit count
        auto_edit_count = state.get("auto_edit_count", 0)
        max_auto_edits = 3  # Prevent recursion limit error by capping loops

        if human_decision == "edit_prompt":
            next_agent = "agent_6"
            logger.info("   → Edit applied, routing to Agent 6 for re-review")
        elif not actually_edited_ids:
            # No real edits were applied (all skipped) — looping back to Agent 6 would just
            # re-review the same unchanged images and reach the same NEEDS_EDIT verdict.
            # Proceed to Agent 8 directly to avoid a pointless spin.
            next_agent = "agent_8"
            logger.warning("   → No actual edits applied (all skipped). Proceeding to Agent 8.")
        elif auto_edit_count >= max_auto_edits:
            next_agent = "agent_8"
            logger.warning(f"   → Max auto-edit loops ({max_auto_edits}) reached. Proceeding to Agent 8.")
        else:
            next_agent = "agent_6"
            auto_edit_count += 1
            logger.info(f"   → Looping back to Agent 6 for re-review (auto-fix loop {auto_edit_count}/{max_auto_edits})")

        # Update state — use actually_edited_ids so Agent 6 only re-reviews images
        # that were genuinely changed, not skipped ones (which would score identically).
        return {
            **state,
            "edited_images": agent.edited_images,
            "agent7_status": "completed",
            "current_agent": next_agent,
            "recently_edited_asset_ids": actually_edited_ids,
            "auto_edit_count": auto_edit_count,
        }

    except Exception as e:
        logger.error(f"Agent 7 failed: {e}")
        import traceback
        traceback.print_exc()
        _save_agent_output(
            state=state,
            agent_number=7,
            status="failed",
            error=str(e)
        )
        _update_job_agent_status(state, agent_number=7, status="failed")
        return {
            **state,
            "agent7_status": "failed",
            "pipeline_status": "failed",
            "error_message": str(e),
        }


def human_approval_checkpoint(state: Phase1State) -> Phase1State:
    """
    Human Approval Checkpoint Node
    Pauses workflow and waits for human decision
    """
    logger.info("\n" + "="*60)
    logger.info("🧑 HUMAN APPROVAL CHECKPOINT")
    logger.info("="*60)
    logger.info("⏸Workflow paused. Awaiting human approval...")
    logger.info("   - All images have been reviewed by AI")
    logger.info("   - Human can now review images and prompts")
    logger.info("   - Decisions: 'approve' → Agent 8, 'edit_prompts' → Agent 5")

    # Check if human has already provided a decision
    if state.get("human_approval_decision"):
        decision = state["human_approval_decision"]
        logger.info(f"   ✓ Human decision received: {decision}")

        if decision == "approve":
            return {
                **state,
                "pipeline_status": "generating_variations",
                "current_agent": "agent_8",
            }
        elif decision == "edit_prompts":
            # Check regeneration limit
            regen_count = state.get("regeneration_count", 0)
            max_regen = state.get("max_regenerations", 5)

            if regen_count >= max_regen:
                logger.error(f"   Maximum regeneration attempts reached ({max_regen})")
                return {
                    **state,
                    "pipeline_status": "failed",
                    "error_message": f"Maximum regeneration attempts ({max_regen}) exceeded",
                }

            # Increment regeneration count and go back to Agent 5
            return {
                **state,
                "pipeline_status": "regenerating_images",
                "current_agent": "agent_5",
                "regeneration_count": regen_count + 1,
                "human_approval_decision": None,  # Reset for next round
            }

    # No decision yet - pause workflow
    return {
        **state,
        "pipeline_status": "waiting_for_human_approval",
        "current_agent": "human_checkpoint",
    }


def agent_8_node(state: Phase1State) -> Phase1State:
    """
    Agent 8: Variation Generator Node
    Generates camera angle variations for approved character images
    Only processes assets in approved_asset_ids list
    """
    logger.info("\n" + "="*60)
    logger.info("AGENT 8: VARIATION GENERATOR")
    logger.info("="*60)

    # Check if this is a retry attempt
    retry_count = state.get("agent8_retry_count", 0)
    if retry_count > 0:
        logger.info(f"🔄 This is retry attempt {retry_count}/3")
        logger.info("   Starting fresh with clean agent instance...")

    try:
        # Initialize agent (fresh instance for each attempt)
        agent = VariationGeneratorAgent()

        # Get approved asset IDs (if provided)
        approved_asset_ids = state.get("approved_asset_ids")

        # Normalize approved_asset_ids: convert hyphens to underscores to match Agent 5 format
        # Frontend sends: char-001, Agent 5 has: char_001
        normalized_approved_ids = None
        if approved_asset_ids:
            normalized_approved_ids = [aid.replace('-', '_') for aid in approved_asset_ids]
            logger.info(f"Processing {len(approved_asset_ids)} approved assets only")
            logger.info(f"   Approved IDs (original): {approved_asset_ids}")
            logger.info(f"   Normalized IDs (for matching): {normalized_approved_ids}")
        else:
            logger.info("📋 No approved_asset_ids filter - processing all assets")

        # Transform image data structure from Agent 5/7 format to Agent 8 format
        # Agent 5/7 can have TWO structures:
        #   1. Dict format: {characters: {"AssetName": {images: [{local_path}]}}}
        #   2. List format: {characters: [{id, name, images: [{local_path}]}]}
        # Agent 8 expects: {characters: [{id, name, master_image: "path"}]}

        approved_images = {"characters": [], "locations": [], "props": []}

        # Use edited images if available, otherwise use generated images
        edited_imgs = state.get("edited_images")
        generated_imgs = state.get("generated_images", {})

        # If image data is missing (happens with minimal state for retry), load from database
        if (not edited_imgs or (isinstance(edited_imgs, dict) and not any(edited_imgs.values()))) and \
           (not generated_imgs or (isinstance(generated_imgs, dict) and not any(generated_imgs.values()))):
            logger.warning("Image data missing from state - loading from database...")

            # Try loading from assets_collection (for movies) or project (for projects)
            assets_collection_id = state.get("assets_collection_id")
            project_id = state.get("project_id")

            if assets_collection_id:
                # Load from assets_collection (movie workflow)
                from app.services.assets_collection_service import AssetsCollectionService
                assets_service = AssetsCollectionService()
                assets_collection = assets_service.get_assets_collection(assets_collection_id)

                if assets_collection:
                    logger.info(f"✓ Found assets collection {assets_collection_id}")

                    # Check what agent outputs exist
                    agent7_output = assets_collection.get("agent7_output", {})
                    agent5_output = assets_collection.get("agent5_output", {})

                    logger.info(f"   Agent 7 output exists: {bool(agent7_output)}, type: {type(agent7_output)}")
                    logger.info(f"   Agent 5 output exists: {bool(agent5_output)}, type: {type(agent5_output)}")

                    if isinstance(agent7_output, dict):
                        logger.info(f"   Agent 7 keys: {list(agent7_output.keys())}")
                    if isinstance(agent5_output, dict):
                        logger.info(f"   Agent 5 keys: {list(agent5_output.keys())}")

                    # Extract the actual output data from the nested structure
                    # Handle cases where output field might be None
                    agent7_output_data = agent7_output.get("output") if isinstance(agent7_output, dict) else None
                    agent5_output_data = agent5_output.get("output") if isinstance(agent5_output, dict) else None

                    agent7_data = agent7_output_data if isinstance(agent7_output_data, dict) else {}
                    agent5_data = agent5_output_data if isinstance(agent5_output_data, dict) else {}

                    logger.info(f"   Agent 7 data exists: {bool(agent7_data)}, keys: {list(agent7_data.keys()) if isinstance(agent7_data, dict) else 'N/A'}")
                    logger.info(f"   Agent 5 data exists: {bool(agent5_data)}, keys: {list(agent5_data.keys()) if isinstance(agent5_data, dict) else 'N/A'}")

                    edited_imgs = agent7_data.get("edited_images", {}) if isinstance(agent7_data, dict) else {}
                    generated_imgs = agent5_data.get("generated_images", {}) if isinstance(agent5_data, dict) else {}

                    logger.info(f"✓ Loaded image data from assets collection")
                    logger.info(f"   edited_images: {type(edited_imgs)}, has data: {bool(edited_imgs)}")
                    logger.info(f"   generated_images: {type(generated_imgs)}, has data: {bool(generated_imgs)}")
                else:
                    error_message = (
                        f"Assets collection {assets_collection_id} not found. "
                        "Agent 8 cannot load generated/edited images for retry. "
                        "Please recreate the assets collection or update the job with a valid assets_collection_id."
                    )
                    logger.error(error_message)
                    raise AssetsCollectionNotFoundError(error_message)
            elif project_id:
                # Load from project (project workflow)
                from app.services.project_service import ProjectService
                project_service = ProjectService()
                project = project_service.get_project(project_id)

                if project:
                    logger.info(f"✓ Found project {project_id}")

                    # Get agent_outputs from project
                    agent_outputs = project.get("agent_outputs", {})
                    agent7_output = agent_outputs.get("agent7", {})
                    agent5_output = agent_outputs.get("agent5", {})

                    logger.info(f"   Agent 7 output exists: {bool(agent7_output)}, type: {type(agent7_output)}")
                    logger.info(f"   Agent 5 output exists: {bool(agent5_output)}, type: {type(agent5_output)}")

                    # Extract output data
                    agent7_output_data = agent7_output.get("output") if isinstance(agent7_output, dict) else None
                    agent5_output_data = agent5_output.get("output") if isinstance(agent5_output, dict) else None

                    agent7_data = agent7_output_data if isinstance(agent7_output_data, dict) else {}
                    agent5_data = agent5_output_data if isinstance(agent5_output_data, dict) else {}

                    logger.info(f"   Agent 7 data exists: {bool(agent7_data)}, keys: {list(agent7_data.keys()) if isinstance(agent7_data, dict) else 'N/A'}")
                    logger.info(f"   Agent 5 data exists: {bool(agent5_data)}, keys: {list(agent5_data.keys()) if isinstance(agent5_data, dict) else 'N/A'}")

                    edited_imgs = agent7_data.get("edited_images", {}) if isinstance(agent7_data, dict) else {}
                    generated_imgs = agent5_data.get("generated_images", {}) if isinstance(agent5_data, dict) else {}

                    logger.info(f"✓ Loaded image data from project")
                    logger.info(f"   edited_images: {type(edited_imgs)}, has data: {bool(edited_imgs)}")
                    logger.info(f"   generated_images: {type(generated_imgs)}, has data: {bool(generated_imgs)}")
                else:
                    error_message = (
                        f"Project {project_id} not found. "
                        "Agent 8 cannot load generated/edited images for retry. "
                        "Please ensure the project exists."
                    )
                    logger.error(error_message)
                    raise AssetsCollectionNotFoundError(error_message)
            else:
                logger.error("No assets_collection_id or project_id in state")
                raise AssetsCollectionNotFoundError(
                    "State is missing assets_collection_id or project_id; Agent 8 cannot load generated/edited images."
                )

        logger.info(f"DEBUG: edited_images type: {type(edited_imgs)}, keys: {list(edited_imgs.keys()) if isinstance(edited_imgs, dict) else 'N/A'}")
        logger.info(f"DEBUG: generated_images type: {type(generated_imgs)}, keys: {list(generated_imgs.keys()) if isinstance(generated_imgs, dict) else 'N/A'}")

        if isinstance(edited_imgs, dict):
            for k, v in edited_imgs.items():
                logger.info(f"   edited_images['{k}']: {type(v)} with {len(v) if isinstance(v, (list, dict)) else 'N/A'} items")
        if isinstance(generated_imgs, dict):
            for k, v in generated_imgs.items():
                logger.info(f"   generated_images['{k}']: {type(v)} with {len(v) if isinstance(v, (list, dict)) else 'N/A'} items")

        source_images = edited_imgs or generated_imgs

        logger.info(f"📋 Source images available: {list(source_images.keys()) if isinstance(source_images, dict) else 'N/A'}")

        # Transform Agent 7's flat dictionary format if needed
        # Agent 7 uses keys like "characters:char-001:image_1"
        # We need to restructure it to {characters: [...], locations: [...], props: [...]}
        if source_images and not any(k in source_images for k in ["characters", "locations", "props"]):
            logger.info("   Transforming Agent 7's flat dictionary format...")
            restructured_images = {"characters": [], "locations": [], "props": []}
            generated_images = state.get("generated_images", {})

            # Group edited images by asset type
            for edit_key, edit_data in source_images.items():
                # Parse the key format: "asset_type:asset_id:image_index"
                parts = edit_key.split(":")
                if len(parts) >= 2:
                    asset_type_singular = parts[0]  # e.g., "characters"
                    asset_id = parts[1]  # e.g., "char-001"

                    # Find the asset in generated_images to get its metadata
                    for asset in generated_images.get(asset_type_singular, []):
                        if asset.get("id") == asset_id:
                            # Create a new asset entry with edited image
                            edited_image_path = None

                            # Check if edit was applied or skipped
                            if edit_data.get("edit_skipped"):
                                # Use original image
                                edited_image_path = edit_data.get("original_image")
                            elif edit_data.get("edited_images"):
                                # Use first edited image
                                edited_imgs = edit_data.get("edited_images", [])
                                if edited_imgs:
                                    edited_image_path = edited_imgs[0].get("local_path")

                            if edited_image_path:
                                restructured_images[asset_type_singular].append({
                                    "id": asset_id,
                                    "name": asset.get("name", asset_id),
                                    "images": [{"local_path": edited_image_path, "index": 1}]
                                })
                            break

            source_images = restructured_images
            logger.info(f"   Restructured: {len(restructured_images.get('characters', []))} characters, {len(restructured_images.get('locations', []))} locations, {len(restructured_images.get('props', []))} props")

        # If source_images is still empty or doesn't have the expected structure, fall back to generated_images
        # Also fall back if source_images has the keys but all arrays are empty
        has_structure = any(k in source_images for k in ["characters", "locations", "props"])
        has_data = False
        if has_structure:
            for asset_type in ["characters", "locations", "props"]:
                assets_data = source_images.get(asset_type, [])
                if isinstance(assets_data, list) and len(assets_data) > 0:
                    has_data = True
                    break
                elif isinstance(assets_data, dict) and len(assets_data) > 0:
                    has_data = True
                    break

        if not source_images or not has_structure or not has_data:
            if has_structure and not has_data:
                logger.warning("   Source images has structure but no data, falling back to generated_images...")
            else:
                logger.warning("   Falling back to generated_images...")
            source_images = state.get("generated_images", {})

            # Check again if we have data after fallback
            has_data_after_fallback = False
            for asset_type in ["characters", "locations", "props"]:
                assets_data = source_images.get(asset_type, [])
                if isinstance(assets_data, list) and len(assets_data) > 0:
                    has_data_after_fallback = True
                    break
                elif isinstance(assets_data, dict) and len(assets_data) > 0:
                    has_data_after_fallback = True
                    break

            if not has_data_after_fallback:
                # Try to load from Agent 5's output files as a last resort
                logger.info("   Attempting to load from Agent 5's output files...")
                output_files = state.get("output_files", [])
                agent5_file = None
                for file in output_files:
                    if "agent5" in file and file.endswith(".json"):
                        agent5_file = file
                        break

                if agent5_file and os.path.exists(agent5_file):
                    try:
                        import json
                        with open(agent5_file, 'r') as f:
                            agent5_data = json.load(f)
                        source_images = agent5_data.get("generated_images", {})
                        logger.info(f"   Loaded generated_images from {agent5_file}")
                        logger.info(f"   Found: {len(source_images.get('characters', []))} characters, {len(source_images.get('locations', []))} locations, {len(source_images.get('props', []))} props")

                        # Check if we now have data
                        for asset_type in ["characters", "locations", "props"]:
                            assets_data = source_images.get(asset_type, [])
                            if isinstance(assets_data, list) and len(assets_data) > 0:
                                has_data_after_fallback = True
                                break
                            elif isinstance(assets_data, dict) and len(assets_data) > 0:
                                has_data_after_fallback = True
                                break
                    except Exception as e:
                        logger.warning(f"   Failed to load from Agent 5 output file: {e}")

                if not has_data_after_fallback:
                    error_msg = (
                        "No image data found in state. Both edited_images and generated_images are empty. "
                        "This usually happens when resuming a job but the image data was not properly saved to MongoDB. "
                        f"Available state keys: {list(state.keys())}\n"
                        f"Agent 5 output file: {agent5_file if agent5_file else 'Not found'}"
                    )
                    logger.error(f"{error_msg}")
                    raise ValueError(error_msg)

        for asset_type in ["characters", "locations", "props"]:
            assets_data = source_images.get(asset_type, {})

            # Handle both dict and list formats
            if isinstance(assets_data, dict):
                # Dict format: keyed by asset name
                logger.info(f"   {asset_type}: Processing dict format with {len(assets_data)} items")

                for asset_name, asset_data in assets_data.items():
                    if not isinstance(asset_data, dict):
                        logger.warning(f"   Skipping {asset_name}: invalid data type")
                        continue

                    # Extract master image path
                    master_image_path = None
                    asset_id = asset_data.get("id", asset_name)  # Use name as fallback ID

                    # Filter by approved_asset_ids if provided
                    # Normalize the asset_id for comparison (handle both hyphens and underscores)
                    normalized_asset_id = asset_id.replace('-', '_') if asset_id else None
                    if normalized_approved_ids and normalized_asset_id not in normalized_approved_ids:
                        logger.info(f"   ⏭Skipping {asset_name} (not in approved list)")
                        continue

                    # Check for images array
                    if isinstance(asset_data.get("images"), list) and asset_data["images"]:
                        first_image = asset_data["images"][0]
                        # Try local_path first, then s3_url, then url
                        master_image_path = (
                            first_image.get("local_path") or
                            first_image.get("s3_url") or
                            first_image.get("url")
                        )

                    # Check for edited_images
                    elif isinstance(asset_data.get("edited_images"), dict):
                        for key, img_data in asset_data.get("edited_images", {}).items():
                            if isinstance(img_data, dict) and img_data.get("edited_images"):
                                first_edit = img_data["edited_images"][0]
                                master_image_path = (
                                    first_edit.get("local_path") or
                                    first_edit.get("s3_url") or
                                    first_edit.get("url")
                                )
                                break

                    # Validate path: check if it's a local file or an S3 URL
                    is_valid = False
                    if master_image_path:
                        if master_image_path.startswith(('http://', 'https://', 's3://')):
                            # S3 URL or HTTP URL - assume valid
                            is_valid = True
                            logger.info(f"   ✓ {asset_name} -> {master_image_path[:80]}...")
                        elif os.path.exists(master_image_path):
                            # Local file path - verify it exists
                            is_valid = True
                            logger.info(f"   ✓ {asset_name} -> {os.path.basename(master_image_path)}")
                        else:
                            logger.warning(f"   {asset_name} - Path found but doesn't exist: {master_image_path}")

                    if is_valid:
                        approved_images[asset_type].append({
                            "id": asset_id,
                            "name": asset_name,
                            "master_image": master_image_path
                        })
                    else:
                        if not master_image_path:
                            logger.warning(f"   {asset_name} - No valid image path found")

            elif isinstance(assets_data, list):
                # List format: array of asset objects
                logger.info(f"   {asset_type}: Processing list format with {len(assets_data)} items")
                if assets_data and normalized_approved_ids:
                    asset_ids_in_data = [a.get('id') for a in assets_data if isinstance(a, dict)]
                    logger.info(f"      Asset IDs in data: {asset_ids_in_data}")
                    # Normalize asset IDs from data for comparison (handle both hyphens and underscores)
                    normalized_asset_ids = [aid.replace('-', '_') for aid in asset_ids_in_data if aid]
                    matching = [aid for aid in normalized_approved_ids if aid in normalized_asset_ids]
                    logger.info(f"      Matching approved IDs: {matching}")

                for asset in assets_data:
                    if not isinstance(asset, dict):
                        continue

                    asset_id = asset.get("id")
                    asset_name = asset.get("name", "Unknown")
                    master_image_path = None

                    # Filter by approved_asset_ids if provided
                    # Normalize the asset_id for comparison (handle both hyphens and underscores)
                    normalized_asset_id = asset_id.replace('-', '_') if asset_id else None
                    if normalized_approved_ids and normalized_asset_id not in normalized_approved_ids:
                        logger.info(f"   ⏭Skipping {asset_name} (not in approved list)")
                        continue

                    # DEBUG: Log asset structure
                    logger.info(f"   🔍 Asset {asset_name} ({asset_id}) structure: {list(asset.keys())}")

                    # Check for images array
                    if isinstance(asset.get("images"), list) and asset["images"]:
                        first_image = asset["images"][0]
                        # Try local_path first, then s3_url, then url
                        master_image_path = (
                            first_image.get("local_path") or
                            first_image.get("s3_url") or
                            first_image.get("url")
                        )
                        logger.info(f"      Found images array, path: {master_image_path}")

                    # Check for edited_images
                    elif isinstance(asset.get("edited_images"), dict):
                        logger.debug(f"      Found edited_images dict with keys: {list(asset['edited_images'].keys())}")
                        for key, img_data in asset.get("edited_images", {}).items():
                            if isinstance(img_data, dict) and img_data.get("edited_images"):
                                master_image_path = img_data["edited_images"][0].get("local_path")
                                logger.debug(f"      Found edited path: {master_image_path}")
                                break

                    # Check for master_image_path directly in asset
                    elif asset.get("master_image_path"):
                        master_image_path = asset.get("master_image_path")
                        logger.debug(f"      Found master_image_path directly: {master_image_path}")

                    # Check for image_url (might be local path)
                    elif asset.get("image_url"):
                        master_image_path = asset.get("image_url")
                        logger.debug(f"      Found image_url: {master_image_path}")

                    # Check for local_path directly
                    elif asset.get("local_path"):
                        master_image_path = asset.get("local_path")
                        logger.debug(f"      Found local_path directly: {master_image_path}")

                    # Validate path: check if it's a local file or an S3 URL
                    is_valid = False
                    if master_image_path:
                        if master_image_path.startswith(('http://', 'https://', 's3://')):
                            # S3 URL or HTTP URL - assume valid
                            is_valid = True
                            logger.info(f"   ✓ {asset_name} -> {master_image_path[:80]}...")
                        elif os.path.exists(master_image_path):
                            # Local file path - verify it exists
                            is_valid = True
                            logger.info(f"   ✓ {asset_name} -> {os.path.basename(master_image_path)}")
                        else:
                            logger.warning(f"   {asset_name} - Path found but doesn't exist: {master_image_path}")

                    if is_valid:
                        approved_images[asset_type].append({
                            "id": asset_id,
                            "name": asset_name,
                            "master_image": master_image_path
                        })
                    else:
                        if not master_image_path:
                            logger.warning(f"   {asset_name} - No valid image path found in asset keys: {list(asset.keys())}")

        # Set the transformed data
        agent.approved_images = approved_images

        total_loaded = len(approved_images['characters']) + len(approved_images['locations']) + len(approved_images['props'])
        logger.info(f"\nLoaded {len(approved_images['characters'])} characters, {len(approved_images['locations'])} locations, {len(approved_images['props'])} props")

        # CRITICAL FIX: If we have approved_asset_ids but loaded 0 images, MongoDB data is incomplete
        # Fall back to loading directly from Agent 5's output file
        if normalized_approved_ids and total_loaded == 0:
            logger.info("\n" + "="*60)
            logger.warning("FALLBACK: MongoDB data incomplete, loading from output files")
            logger.info("="*60)

            # Find Agent 5 output file
            output_files = state.get("output_files", [])
            agent5_file = None
            for file in output_files:
                if "agent5" in file and file.endswith(".json"):
                    agent5_file = file
                    break

            if agent5_file and os.path.exists(agent5_file):
                try:
                    import json
                    with open(agent5_file, 'r') as f:
                        agent5_data = json.load(f)

                    file_generated_images = agent5_data.get("generated_images", {})
                    logger.info(f"📁 Loading from: {agent5_file}")
                    logger.info(f"   File contains: {len(file_generated_images.get('characters', []))} chars, {len(file_generated_images.get('locations', []))} locs, {len(file_generated_images.get('props', []))} props")

                    # Check if we need to use edited images from Agent 7
                    agent7_edited = state.get("edited_images", {})
                    has_edits = agent7_edited and len(agent7_edited) > 0

                    # Build mapping of edited images if Agent 7 ran
                    edited_mapping = {}
                    if has_edits and isinstance(agent7_edited, dict):
                        # Handle flat dictionary format from Agent 7
                        if not any(k in agent7_edited for k in ["characters", "locations", "props"]):
                            logger.info(f"   Agent 7 edits found (flat format): {len(agent7_edited)} keys")
                            for edit_key, edit_data in agent7_edited.items():
                                # Parse key: "characters:char-001:image_1"
                                parts = edit_key.split(":")
                                if len(parts) >= 2:
                                    asset_type = parts[0]
                                    asset_id = parts[1]

                                    # Use edited image if available, otherwise original
                                    if edit_data.get("edit_skipped"):
                                        edited_mapping[f"{asset_type}:{asset_id}"] = edit_data.get("original_image")
                                    elif edit_data.get("edited_images"):
                                        edited_imgs = edit_data.get("edited_images", [])
                                        if edited_imgs:
                                            edited_mapping[f"{asset_type}:{asset_id}"] = edited_imgs[0].get("local_path")

                    # Rebuild approved_images from file data
                    approved_images = {"characters": [], "locations": [], "props": []}

                    for asset_type in ["characters", "locations", "props"]:
                        assets_data = file_generated_images.get(asset_type, [])
                        logger.info(f"\n   Processing {asset_type} from file:")

                        for asset in assets_data:
                            if not isinstance(asset, dict):
                                continue

                            asset_id = asset.get("id")
                            asset_name = asset.get("name", "Unknown")

                            # Filter by approved_asset_ids
                            if asset_id not in normalized_approved_ids:
                                logger.info(f"      ⏭Skipping {asset_name} (id={asset_id})")
                                continue

                            # Check if we have an edited version
                            edit_key = f"{asset_type}:{asset_id}"
                            if edit_key in edited_mapping:
                                master_image_path = edited_mapping[edit_key]
                                logger.info(f"      {asset_name} (id={asset_id}) [EDITED] -> {os.path.basename(master_image_path)}")
                            else:
                                # Use original from Agent 5
                                images_list = asset.get("images", [])
                                if images_list:
                                    master_image_path = images_list[0].get("local_path")
                                    logger.info(f"      {asset_name} (id={asset_id}) [ORIGINAL] -> {os.path.basename(master_image_path)}")
                                else:
                                    logger.warning(f"      {asset_name} - No images found")
                                    continue

                            if master_image_path and os.path.exists(master_image_path):
                                approved_images[asset_type].append({
                                    "id": asset_id,
                                    "name": asset_name,
                                    "master_image": master_image_path
                                })
                            else:
                                logger.error(f"      {asset_name} - Image file not found: {master_image_path}")

                    agent.approved_images = approved_images
                    total_loaded = len(approved_images['characters']) + len(approved_images['locations']) + len(approved_images['props'])
                    logger.info(f"\nFALLBACK SUCCESS: Loaded {total_loaded} approved images from file")

                except Exception as e:
                    logger.error(f"\nFALLBACK FAILED: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.error(f"\nAgent 5 output file not found or doesn't exist")
                logger.info(f"   Looking for: {agent5_file}")
                logger.info(f"   Available: {output_files}")

        # Generate variations for characters only
        agent.generate_variations()

        # Save agent output to MongoDB (supports both project and movie workflows)
        _save_agent_output(
            state=state,
            agent_number=8,
            status="completed",
            output={"variation_images": agent.generated_variations}
        )

        # Update job status in database for polling
        _update_job_agent_status(state, agent_number=8, status="completed")

        # Update state (clear retry counter on success)
        return {
            **state,
            "variation_images": agent.generated_variations,
            "agent8_status": "completed",
            "agent8_retry_count": 0,  # Reset retry counter on success
            "current_agent": "completed",
            "pipeline_status": "completed",
            "error_message": None,  # Clear any previous error
        }

    except Exception as e:
        logger.error(f"Agent 8 failed: {e}")
        import traceback
        traceback.print_exc()

        if isinstance(e, AssetsCollectionNotFoundError):
            logger.error("Agent 8 encountered a fatal error (missing assets collection). Skipping retries.")
            _save_agent_output(
                state=state,
                agent_number=8,
                status="failed",
                error=str(e)
            )
            return {
                **state,
                "agent8_status": "failed",
                "pipeline_status": "failed",
                "error_message": str(e),
            }

        # Get current retry count for Agent 8
        agent8_retry_count = state.get("agent8_retry_count", 0)
        max_agent8_retries = 3  # Maximum number of retries for Agent 8

        # Increment retry count
        agent8_retry_count += 1

        if agent8_retry_count <= max_agent8_retries:
            logger.warning(f"Agent 8 retry {agent8_retry_count}/{max_agent8_retries} - will retry...")

            # Add exponential backoff delay before retry (2^retry seconds)
            import time
            delay_seconds = 2 ** agent8_retry_count  # 2, 4, 8 seconds
            logger.info(f"Waiting {delay_seconds} seconds before retry...")
            time.sleep(delay_seconds)

            _save_agent_output(
                state=state,
                agent_number=8,
                status="retrying",
                error=f"Attempt {agent8_retry_count} failed: {str(e)}"
            )
            _update_job_agent_status(state, agent_number=8, status="retrying")
            return {
                **state,
                "agent8_status": "retrying",
                "agent8_retry_count": agent8_retry_count,
                "pipeline_status": "generating_variations",  # Keep in generating state to retry
                "current_agent": "agent_8",  # Stay on agent 8
                "error_message": f"Agent 8 attempt {agent8_retry_count} failed, retrying in {delay_seconds}s... Error: {str(e)}",
            }
        else:
            logger.error(f"Agent 8 failed after {max_agent8_retries} retries")
            _save_agent_output(
                state=state,
                agent_number=8,
                status="failed",
                error=f"Failed after {max_agent8_retries} retries. Last error: {str(e)}"
            )
            _update_job_agent_status(state, agent_number=8, status="failed")
            return {
                **state,
                "agent8_status": "failed",
                "agent8_retry_count": agent8_retry_count,
                "pipeline_status": "failed",
                "error_message": f"Agent 8 failed after {max_agent8_retries} retries. Last error: {str(e)}",
            }


def regeneration_router_node(state: Phase1State) -> Phase1State:
    """
    Regeneration Router Node
    Handles regeneration logic with counter and max limit of 3
    """
    logger.info("\n" + "="*60)
    logger.info("REGENERATION ROUTER")
    logger.info("="*60)

    # Get current regeneration count
    auto_regen_count = state.get("auto_regeneration_count", 0)
    max_auto_regen = 3

    # Check if we've hit the limit BEFORE printing the attempt number
    if auto_regen_count >= max_auto_regen:
        logger.error(f"   Maximum auto-regeneration attempts reached ({max_auto_regen})")
        logger.info(f"   → Proceeding to Agent 8 with available images")

        return {
            **state,
            "current_agent": "agent_8",
            "pipeline_status": "generating_variations",
        }

    # Use regenerated prompts for next attempt
    prompts_to_use = state.get("regenerated_prompts") or state.get("optimized_prompts")

    # Extract asset IDs from needs_regeneration_assets for selective review
    needs_regeneration_assets = state.get("needs_regeneration_assets", [])
    regenerated_asset_ids = []
    for asset_ref in needs_regeneration_assets:
        # Format is "asset_type:asset_id" (e.g., "props:prop_shippinglabels")
        if ":" in asset_ref:
            regenerated_asset_ids.append(asset_ref.split(":", 1)[1])

    logger.info(f"   Current auto-regeneration attempt: {auto_regen_count + 1}/{max_auto_regen}")
    logger.info(f"   Proceeding with regeneration attempt {auto_regen_count + 1}")
    logger.info(f"   → Routing back to Agent 5 with rewritten prompts")
    logger.info(f"   → Will re-review {len(regenerated_asset_ids)} regenerated asset(s)")

    return {
        **state,
        "optimized_prompts": prompts_to_use,  # Use rewritten prompts
        "auto_regeneration_count": auto_regen_count + 1,
        "current_agent": "agent_5",
        "pipeline_status": "regenerating_images",
        "recently_edited_asset_ids": regenerated_asset_ids,  # Tell Agent 6 which assets to re-review
        # Note: Don't clear needs_regeneration_assets here - Agent 5 needs it!
        # It will be cleared after Agent 5 completes
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue(state: Phase1State) -> Literal["agent_2", "agent_3", "agent_4", "agent_5", "end", "failed"]:
    """
    Determines next step based on current agent status (Agents 1-4)
    """
    current = state.get("current_agent", "")
    pipeline_status = state.get("pipeline_status", "running")

    # Check for failures
    if pipeline_status == "failed":
        return "failed"

    # Check for completion
    if pipeline_status == "completed":
        return "end"

    # Route to next agent
    if current == "agent_2":
        return "agent_2"
    elif current == "agent_3":
        return "agent_3"
    elif current == "agent_4":
        return "agent_4"
    elif current == "agent_5":
        return "agent_5"
    elif current == "completed":
        return "end"

    return "failed"


def route_after_agent6(state: Phase1State) -> Literal["agent_7", "agent_8", "regeneration_router", "failed"]:
    """
    After Agent 6: Route based on review decisions
    Priority: regeneration > editing > agent_8
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "failed"

    needs_regeneration = state.get("needs_regeneration_assets", [])
    needs_editing = state.get("needs_editing_assets", [])

    # Priority: regeneration first
    if needs_regeneration:
        logger.info(f"   → Routing to regeneration ({len(needs_regeneration)} assets need regeneration)")
        return "regeneration_router"
    elif needs_editing:
        logger.info(f"   → Routing to Agent 7 ({len(needs_editing)} assets need editing)")
        return "agent_7"
    else:
        logger.info("   → All images approved by AI, routing directly to Agent 8")
        return "agent_8"


def route_after_agent7(state: Phase1State) -> Literal["agent_6", "agent_8", "failed"]:
    """
    After Agent 7:
    - If max auto-edit loops reached, proceed to Agent 8
    - Otherwise, loop back to Agent 6 for auto re-review
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "failed"

    current_agent = state.get("current_agent")
    if current_agent == "agent_8":
        logger.info("   → Max edit loops reached, proceeding to Agent 8")
        return "agent_8"
    else:
        logger.info("   → Routing back to Agent 6 for re-review of edited images")
        return "agent_6"


def route_from_regeneration_router(state: Phase1State) -> Literal["agent_5", "agent_8"]:
    """
    After Regeneration Router: Route to Agent 5 or Agent 8 based on counter
    """
    current_agent = state.get("current_agent", "")

    if current_agent == "agent_5":
        logger.info("   → Routing to Agent 5 for regeneration")
        return "agent_5"
    else:
        logger.info("   → Max regenerations reached, proceeding to Agent 8 with available images")
        return "agent_8"


def route_after_human_checkpoint(state: Phase1State) -> Literal["agent_5", "agent_8", "wait", "failed"]:
    """
    After Human Checkpoint: Route based on human decision
    """
    pipeline_status = state.get("pipeline_status", "")

    if pipeline_status == "failed":
        return "failed"

    # If still waiting for human approval, stay in wait state
    if pipeline_status == "waiting_for_human_approval":
        return "wait"

    # Check human decision
    current_agent = state.get("current_agent", "")

    if current_agent == "agent_8":
        logger.info("   → Human approved! Routing to Agent 8 (Variation Generator)")
        return "agent_8"
    elif current_agent == "agent_5":
        logger.info("   → Human requested prompt edits. Routing back to Agent 5 (Regeneration)")
        return "agent_5"

    # Default: wait
    return "wait"


# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def router_entry_point(state: Phase1State) -> Literal["agent_1", "agent_5", "agent_7", "agent_8", "human_checkpoint"]:
    """
    Dynamic entry point router - determines where to start based on current_agent in state.
    This allows resuming workflows from checkpoints instead of always starting at agent_1.
    """
    current_agent = state.get("current_agent", "agent_1")

    logger.info(f"\n🔀 ENTRY ROUTER: Current agent in state = {current_agent}")

    # Map current_agent to the actual node to resume from
    if current_agent == "agent_8":
        logger.info("   → Resuming from Agent 8 (Variation Generator)")
        return "agent_8"
    elif current_agent == "agent_7":
        logger.info("   → Resuming from Agent 7 (Image Editor - Single Asset)")
        return "agent_7"
    elif current_agent == "agent_5":
        logger.info("   → Resuming from Agent 5 (Image Generator - Regeneration)")
        return "agent_5"
    elif current_agent == "human_checkpoint":
        logger.info("   → Resuming at Human Checkpoint")
        return "human_checkpoint"
    else:
        # Default: start from agent_1 for new workflows
        logger.info("   → Starting from Agent 1 (new workflow)")
        return "agent_1"


def create_phase1_workflow() -> CompiledStateGraph:
    """
    Creates the LangGraph workflow for all 8 agents with human checkpoint
    """
    # Create state graph
    workflow = StateGraph(Phase1State)

    # Add a router node to determine entry point dynamically
    workflow.add_node("router", lambda state: state)  # Pass-through node

    # Add all nodes
    workflow.add_node("agent_1", agent_1_node)
    workflow.add_node("agent_2", agent_2_node)
    workflow.add_node("agent_3", agent_3_node)
    workflow.add_node("agent_4", agent_4_node)
    workflow.add_node("agent_5", agent_5_node)
    workflow.add_node("agent_6", agent_6_node)
    workflow.add_node("agent_7", agent_7_node)
    workflow.add_node("regeneration_router", regeneration_router_node)
    workflow.add_node("human_checkpoint", human_approval_checkpoint)
    workflow.add_node("agent_8", agent_8_node)

    # Set entry point to router (which will determine where to actually start)
    workflow.set_entry_point("router")

    # Router determines where to go based on current_agent in state
    workflow.add_conditional_edges(
        "router",
        router_entry_point,
        {
            "agent_1": "agent_1",
            "agent_5": "agent_5",
            "agent_7": "agent_7",
            "agent_8": "agent_8",
            "human_checkpoint": "human_checkpoint",
        }
    )

    # Add conditional edges for Agents 1-4 (sequential)
    workflow.add_conditional_edges(
        "agent_1",
        should_continue,
        {
            "agent_2": "agent_2",
            "failed": END,
        }
    )

    workflow.add_conditional_edges(
        "agent_2",
        should_continue,
        {
            "agent_3": "agent_3",
            "failed": END,
        }
    )

    workflow.add_conditional_edges(
        "agent_3",
        should_continue,
        {
            "agent_4": "agent_4",
            "failed": END,
        }
    )

    workflow.add_conditional_edges(
        "agent_4",
        should_continue,
        {
            "agent_5": "agent_5",
            "failed": END,
        }
    )

    # Agent 5 → Agent 6 (always - direct edge)
    workflow.add_edge("agent_5", "agent_6")

    # Agent 6 → Regeneration Router (if needs regeneration) OR Agent 7 (if needs edits) OR Agent 8 (if all approved)
    workflow.add_conditional_edges(
        "agent_6",
        route_after_agent6,
        {
            "regeneration_router": "regeneration_router",
            "agent_7": "agent_7",
            "agent_8": "agent_8",
            "failed": END,
        }
    )

    # Regeneration Router → Agent 5 (retry) OR Agent 8 (max attempts reached)
    workflow.add_conditional_edges(
        "regeneration_router",
        route_from_regeneration_router,
        {
            "agent_5": "agent_5",
            "agent_8": "agent_8",
        }
    )

    # Agent 7 → Agent 6 (loop back for re-review) OR Agent 8 (if max edit loops reached)
    workflow.add_conditional_edges(
        "agent_7",
        route_after_agent7,
        {
            "agent_6": "agent_6",
            "agent_8": "agent_8",
            "failed": END,
        }
    )

    # Human Checkpoint → Agent 5 (regenerate) OR Agent 8 (approved) OR Wait
    workflow.add_conditional_edges(
        "human_checkpoint",
        route_after_human_checkpoint,
        {
            "agent_5": "agent_5",
            "agent_8": "agent_8",
            "wait": END,  # Pause workflow
            "failed": END,
        }
    )

    # Agent 8 → END (completion) or retry on failure
    def agent_8_router(state: Phase1State) -> Literal["end", "failed", "retry"]:
        """Route agent 8 to retry, end, or failed"""
        agent8_status = state.get("agent8_status", "")
        pipeline_status = state.get("pipeline_status", "")

        # If retrying, go back to agent_8
        if agent8_status == "retrying":
            logger.info("   → Agent 8 is retrying...")
            return "retry"

        # If failed (after max retries), end
        if agent8_status == "failed" or pipeline_status == "failed":
            logger.info("   → Agent 8 failed, ending workflow")
            return "failed"

        # If completed, end successfully
        logger.info("   → Agent 8 completed, ending workflow")
        return "end"

    workflow.add_conditional_edges(
        "agent_8",
        agent_8_router,
        {
            "end": END,
            "failed": END,
            "retry": "agent_8",  # Loop back to agent_8 for retry
        }
    )

    # Compile the graph
    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_phase1_pipeline(
    script_path: Optional[str] = None,
    script_content: Optional[str] = None,
    project_id: Optional[str] = None,
    movie_id: Optional[str] = None,
    assets_collection_id: Optional[str] = None,
    job_id: Optional[str] = None,
    visual_style: Optional[str] = None,
    shotlist_shots: Optional[list] = None,
    agent1_feedback: Optional[Dict[str, Any]] = None,
    agent2_feedback: Optional[Dict[str, Any]] = None,
    agent3_feedback: Optional[Dict[str, Any]] = None,
    agent4_feedback: Optional[Dict[str, Any]] = None,
    v1_project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the complete Phase 1 pipeline using LangGraph

    Supports two modes:
    - Movie mode: movie_id + assets_collection_id (saves to assets_collections)
    - Legacy mode: project_id (saves to production_projects)

    Args:
        script_path: Path to script file
        script_content: Direct script content
        project_id: MongoDB project ID for saving agent outputs (legacy mode)
        movie_id: MongoDB movie ID (movie mode)
        assets_collection_id: MongoDB assets collection ID (movie mode)
        job_id: Pipeline job ID for status tracking
        visual_style: Visual style for image generation (realistic, pixar, etc.)
        shotlist_shots: Optional list of ShotData objects from parsed shotlist CSV
        agent1_feedback: Optional human feedback for Agent 1
        agent2_feedback: Optional human feedback for Agent 2
        agent3_feedback: Optional human feedback for Agent 3
        agent4_feedback: Optional human feedback for Agent 4

    Returns:
        Final state dictionary with all results
    """
    logger.info("\n" + "🎉 "*40)
    logger.info("PHASE 1 LANGGRAPH PIPELINE STARTING")
    logger.info("🎉 "*40 + "\n")

    # Load script
    if script_path:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    elif script_content:
        content = script_content
    else:
        raise ValueError("Either script_path or script_content required")

    # Fetch visual_style from movie if not provided and movie_id exists
    if not visual_style and movie_id:
        try:
            from backend.services.production.app.services.movie_service import MovieService
            movie_service = MovieService()
            movie = movie_service.get_movie_by_id(movie_id)
            if movie and movie.get("global_settings"):
                visual_style = movie["global_settings"].get("visual_style", "realistic")
                logger.info(f"✓ Fetched visual_style from movie: {visual_style}")
        except Exception as e:
            logger.warning(f"Failed to fetch visual_style from movie: {e}")
            visual_style = "realistic"

    # Default to realistic if still not set
    if not visual_style:
        visual_style = "realistic"
        logger.info(f"Using default visual_style: {visual_style}")

    # Pre-extract CSV entity mapping from shotlist
    csv_entity_mapping = prepare_csv_entity_mapping(shotlist_shots)

    # Fetch product image URL from production MongoDB (this project's own document)
    product_prop = None
    product_image_s3_url = None
    if csv_entity_mapping and csv_entity_mapping.get("has_product_shots"):
        if project_id:
            try:
                proj_doc = project_service.get_project(project_id)
                product_image_s3_url = proj_doc.get("product_image_s3_url") if proj_doc else None
                if product_image_s3_url:
                    logger.info(f"✓ Phase 1: product image URL found in production project {project_id}")
                else:
                    logger.warning(
                        f"Phase 1: product_present shots found but project {project_id} "
                        "has no product_image_s3_url — upload a product image when creating the project"
                    )
            except Exception as exc:
                logger.warning(f"Phase 1: failed to fetch product_image_s3_url from project: {exc}")
        elif movie_id:
            try:
                from app.services.movie_service import MovieService
                movie_svc = MovieService()
                movie_doc = movie_svc.get_movie(movie_id)
                if movie_doc and movie_doc.get("project_ids"):
                    for pid in movie_doc["project_ids"]:
                        proj_doc = project_service.get_project(pid)
                        if proj_doc and proj_doc.get("product_image_s3_url"):
                            product_image_s3_url = proj_doc.get("product_image_s3_url")
                            logger.info(f"✓ Phase 1: product image URL found in scene project {pid} of movie {movie_id}")
                            break
                if not product_image_s3_url:
                    logger.warning(
                        f"Phase 1: product_present shots found but no scene projects for movie {movie_id} "
                        "have product_image_s3_url"
                    )
            except Exception as exc:
                logger.warning(f"Phase 1: failed to fetch product_image_s3_url from movie projects: {exc}")
        else:
            logger.warning(
                "Phase 1: product_present shots found but neither project_id nor movie_id are available — "
                "cannot load product image"
            )

    # Initialize state
    initial_state: Phase1State = {
        "script_path": script_path or "",
        "script_content": content,
        "project_id": project_id,
        "movie_id": movie_id,
        "assets_collection_id": assets_collection_id,
        "job_id": job_id,
        "visual_style": visual_style,
        "csv_entity_mapping": csv_entity_mapping,
        "product_prop": product_prop,
        "product_image_s3_url": product_image_s3_url,
        "extracted_assets": {},
        "agent1_status": "pending",
        "agent1_human_feedback": agent1_feedback,
        "review_results": {},
        "enhanced_assets": {},
        "agent2_status": "pending",
        "agent2_human_feedback": agent2_feedback,
        "generated_prompts": {},
        "agent3_status": "pending",
        "agent3_human_feedback": agent3_feedback,
        "optimized_prompts": {},
        "agent4_status": "pending",
        "agent4_human_feedback": agent4_feedback,
        "generated_images": {},
        "agent5_status": "pending",
        "image_reviews": {},
        "agent6_status": "pending",
        "edited_images": {},
        "agent7_status": "pending",
        "needs_editing_assets": [],
        "needs_regeneration_assets": [],
        "regenerated_prompts": None,
        "auto_regeneration_count": 0,
        "auto_edit_count": 0,
        "variation_images": {},
        "agent8_status": "pending",
        "human_approval_decision": None,
        "human_approval_feedback": None,
        "regeneration_count": 0,
        "max_regenerations": 5,
        "current_agent": "agent_1",
        "pipeline_status": "running",
        "error_message": None,
        "requires_human_feedback": False,
        "feedback_agent": None,
    }

    # Create and run workflow
    app = create_phase1_workflow()

    # Execute the workflow
    final_state = app.invoke(initial_state)

    # Print results
    logger.info("\n" + "🎉 "*40)
    logger.info("PHASE 1 PIPELINE COMPLETED")
    logger.info("🎉 "*40)

    logger.info(f"\nPIPELINE STATUS: {final_state['pipeline_status']}")

    if final_state["pipeline_status"] == "failed":
        logger.error(f"\nERROR: {final_state.get('error_message', 'Unknown error')}")

    return final_state


if __name__ == "__main__":
    # Example usage
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # Detect script path
    if os.path.basename(os.getcwd()) == "phase_1_agents":
        script_path = "test_script_lakeside.txt"
    else:
        script_path = "phase_1_agents/test_script_lakeside.txt"

    if not os.path.exists(script_path):
        logger.error(f"Script file not found: {script_path}")
        sys.exit(1)

    # Run pipeline
    result = run_phase1_pipeline(script_path=script_path)

    logger.info("\nPipeline execution complete!")
