#!/usr/bin/env python3
"""
Pydantic Models for Phase 1 Agents
===================================
Defines structured output models for all Phase 1 agents using Pydantic.
These models are used with Gemini's structured output feature to ensure
type-safe and validated responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import uuid


# ============================================================================
# CSV ENTITY MAPPING MODELS
# ============================================================================

class CSVEntityMapping(BaseModel):
    """CSV entity mapping from shotlist"""
    unique_characters: List[str] = Field(default_factory=list, description="Unique character names from CSV (normalized to UPPERCASE_WITH_UNDERSCORES)")
    unique_locations: List[str] = Field(default_factory=list, description="Unique location names from CSV (normalized to UPPERCASE_WITH_UNDERSCORES)")
    character_shots: Dict[str, List[str]] = Field(default_factory=dict, description="Maps character to shot numbers where they appear")
    location_shots: Dict[str, List[str]] = Field(default_factory=dict, description="Maps location to shot numbers where they appear")
    has_entity_data: bool = Field(default=False, description="Whether CSV contains entity fields (backward compatibility)")


# ============================================================================
# AGENT 1: ASSET GENERATOR MODELS
# ============================================================================

class Character(BaseModel):
    """Character asset model for structured output"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this character")
    name: str = Field(description="Character name as it appears in the script")
    description: str = Field(description="Detailed visual description of the character")
    age_range: Optional[str] = Field(default=None, description="Age range (e.g., '20-30')")
    gender: Optional[str] = Field(default=None, description="Gender: male/female/non-binary/unspecified")
    key_features: List[str] = Field(default_factory=list, description="List of distinctive visual features")
    clothing_style: Optional[str] = Field(default=None, description="Description of typical outfit")
    role: str = Field(description="Character role: protagonist/antagonist/supporting")
    scenes: List[str] = Field(default_factory=list, description="List of scenes where character appears")
    importance: str = Field(description="Importance level: critical/important/background")
    # Note: csv_name is added after extraction in agent code, not part of Gemini's structured output


class Location(BaseModel):
    """Location asset model for structured output"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this location")
    name: str = Field(description="Location name as it appears in the script")
    description: str = Field(description="Detailed visual description of the location")
    setting_type: str = Field(description="Setting type: interior/exterior")
    time_of_day: Optional[str] = Field(default=None, description="Time of day: morning/afternoon/evening/night")
    weather: Optional[str] = Field(default=None, description="Weather conditions if applicable")
    lighting: Optional[str] = Field(default=None, description="Lighting conditions: natural/artificial/dim/bright")
    atmosphere: Optional[str] = Field(default=None, description="Atmosphere or mood of the location")
    key_visual_elements: List[str] = Field(default_factory=list, description="Key visual elements in the location")
    scenes: List[str] = Field(default_factory=list, description="List of scenes at this location")
    importance: str = Field(description="Importance level: critical/important/background")
    # Note: csv_name is added after extraction in agent code, not part of Gemini's structured output


class Prop(BaseModel):
    """Prop asset model for structured output"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this prop")
    name: str = Field(description="Prop name as it appears in the script")
    description: str = Field(description="Detailed visual description of the prop")
    material: Optional[str] = Field(default=None, description="Material composition: metal/wood/fabric/etc")
    size: Optional[str] = Field(default=None, description="Size: small/medium/large")
    condition: Optional[str] = Field(default=None, description="Condition: new/worn/damaged/etc")
    usage: str = Field(description="How the prop is used in the story")
    scenes: List[str] = Field(default_factory=list, description="List of scenes where prop appears")
    importance: str = Field(description="Importance level: critical/important/background")


class ExtractedAssets(BaseModel):
    """Complete asset extraction result from Agent 1"""
    characters: List[Character] = Field(default_factory=list, description="All extracted characters")
    locations: List[Location] = Field(default_factory=list, description="All extracted locations")
    props: List[Prop] = Field(default_factory=list, description="All extracted props")
    # Note: csv_entity_mapping is not included here because it's added after extraction
    # and the Gemini API doesn't support Optional fields in structured output


# ============================================================================
# AGENT 2: ASSET REVIEWER MODELS
# ============================================================================

class MissingAsset(BaseModel):
    """Model for a missing asset"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this asset")
    name: str = Field(description="Name of the missing asset")
    reason: str = Field(description="Why this asset should be included")
    description: str = Field(description="Description of the missing asset")
    importance: str = Field(description="Importance level: critical/important/background")


class DuplicateAsset(BaseModel):
    """Model for duplicate assets"""
    duplicate_names: List[str] = Field(description="Names that are duplicates")
    reason: str = Field(description="Why they are duplicates")
    merged_name: str = Field(description="Suggested merged name")
    merged_description: str = Field(description="Merged description")


class DescriptionEnhancement(BaseModel):
    """Model for description enhancement"""
    asset_name: str = Field(description="Name of the asset being enhanced")
    original_description: str = Field(description="Original description")
    enhanced_description: str = Field(description="Enhanced description")
    improvements_made: List[str] = Field(default_factory=list, description="List of improvements")


class AccuracyIssue(BaseModel):
    """Model for accuracy issue"""
    asset_type: str = Field(description="Type: character/location/prop")
    asset_name: str = Field(description="Name of the asset")
    issue: str = Field(description="Description of the problem")
    suggested_fix: str = Field(description="How to fix it")


class EdgeCase(BaseModel):
    """Model for edge case"""
    asset_name: str = Field(description="Name of the asset")
    case_type: str = Field(description="Type: multiple_forms/time_sensitive/state_change")
    description: str = Field(description="Description of edge case")
    recommendation: str = Field(description="How to handle this")


class QualityScores(BaseModel):
    """Quality scores model"""
    completeness: int = Field(ge=0, le=100, description="Completeness score (0-100)")
    accuracy: int = Field(ge=0, le=100, description="Accuracy score (0-100)")
    detail_level: int = Field(ge=0, le=100, description="Detail level score (0-100)")
    production_readiness: int = Field(ge=0, le=100, description="Production readiness score (0-100)")


class CompletenessCheck(BaseModel):
    """Completeness check results"""
    missing_characters: List[MissingAsset] = Field(default_factory=list, description="Missing characters")
    missing_locations: List[MissingAsset] = Field(default_factory=list, description="Missing locations")
    missing_props: List[MissingAsset] = Field(default_factory=list, description="Missing props")


class DuplicatesDetected(BaseModel):
    """Duplicates detection results"""
    characters: List[DuplicateAsset] = Field(default_factory=list, description="Duplicate characters")
    locations: List[DuplicateAsset] = Field(default_factory=list, description="Duplicate locations")
    props: List[DuplicateAsset] = Field(default_factory=list, description="Duplicate props")


class DescriptionEnhancements(BaseModel):
    """Description enhancements for all asset types"""
    characters: List[DescriptionEnhancement] = Field(default_factory=list, description="Character enhancements")
    locations: List[DescriptionEnhancement] = Field(default_factory=list, description="Location enhancements")
    props: List[DescriptionEnhancement] = Field(default_factory=list, description="Prop enhancements")


class AssetReviewReport(BaseModel):
    """Complete asset review report from Agent 2"""
    completeness_check: CompletenessCheck = Field(description="Completeness check results")
    duplicates_detected: DuplicatesDetected = Field(description="Duplicate detection results")
    description_enhancements: DescriptionEnhancements = Field(description="Description enhancements")
    accuracy_issues: List[AccuracyIssue] = Field(default_factory=list, description="Accuracy issues found")
    edge_cases: List[EdgeCase] = Field(default_factory=list, description="Edge cases identified")
    overall_quality_score: QualityScores = Field(description="Overall quality scores")
    recommendations: List[str] = Field(default_factory=list, description="Overall recommendations")


# ============================================================================
# AGENT 3: PROMPT GENERATOR MODELS
# ============================================================================

class TechnicalSpecs(BaseModel):
    """Technical specifications for image generation"""
    aspect_ratio: str = Field(description="Recommended aspect ratio (e.g., '3:4', '16:9')")
    camera_angle: str = Field(description="Optimal camera angle")
    framing: str = Field(description="How subject should be framed")
    lighting: str = Field(description="Lighting setup description")
    style_keywords: List[str] = Field(default_factory=list, description="Style keywords")


class RecommendedSettings(BaseModel):
    """Recommended generation settings"""
    model: str = Field(description="Best AI model for this asset")
    steps: str = Field(description="Recommended inference steps")
    guidance_scale: str = Field(description="Recommended CFG/guidance")


class MasterPrompt(BaseModel):
    """Master prompt for an asset"""
    initial_prompt: str = Field(description="Comprehensive prompt for master image (150-300 words)")
    negative_prompt: str = Field(description="Things to avoid")
    technical_specs: TechnicalSpecs = Field(description="Technical specifications")
    recommended_settings: RecommendedSettings = Field(description="Recommended generation settings")


class CharacterPromptData(BaseModel):
    """Prompt data for a character"""
    character_name: str = Field(description="Name of the character")
    master_prompt: MasterPrompt = Field(description="Master prompt for the character")


class LocationPromptData(BaseModel):
    """Prompt data for a location"""
    location_name: str = Field(description="Name of the location")
    master_prompt: MasterPrompt = Field(description="Master prompt for the location")


class PropPromptData(BaseModel):
    """Prompt data for a prop"""
    prop_name: str = Field(description="Name of the prop")
    master_prompt: MasterPrompt = Field(description="Master prompt for the prop")


# ============================================================================
# AGENT 4: PROMPT OPTIMIZER MODELS
# ============================================================================

class OptimizationAnalysis(BaseModel):
    """Analysis of the optimization process"""
    strengths: List[str] = Field(default_factory=list, description="What works well in the initial prompt")
    improvements_needed: List[str] = Field(default_factory=list, description="What could be enhanced")
    added_elements: List[str] = Field(default_factory=list, description="New details or refinements added")


class FinalPrompt(BaseModel):
    """Final optimized prompt"""
    prompt: str = Field(description="Optimized final prompt (200-350 words)")
    negative_prompt: str = Field(description="Enhanced negative prompt")
    technical_specs: TechnicalSpecs = Field(description="Technical specifications")
    recommended_settings: RecommendedSettings = Field(description="Recommended settings")


class PromptComparison(BaseModel):
    """Comparison between initial and final prompts"""
    initial_word_count: str = Field(description="Word count of initial prompt")
    final_word_count: str = Field(description="Word count of final prompt")
    detail_level_improvement: str = Field(description="Percentage or description of improvement")
    key_changes: List[str] = Field(default_factory=list, description="Major changes made")


class OptimizedPromptData(BaseModel):
    """Complete optimized prompt data from Agent 4"""
    asset_name: str = Field(description="Name of the asset")
    asset_type: str = Field(description="Type of asset: character/location/prop")
    optimization_analysis: OptimizationAnalysis = Field(description="Analysis of the optimization")
    final_prompt: FinalPrompt = Field(description="Final optimized prompt")
    comparison: PromptComparison = Field(description="Comparison with initial prompt")


# Legacy models for backward compatibility
class RefinedPrompt(BaseModel):
    """Refined prompt model"""
    asset_name: str = Field(description="Name of the asset")
    asset_type: str = Field(description="Type of asset: character/location/prop")
    original_description: str = Field(description="Original asset description")
    refined_prompt: str = Field(description="Refined prompt optimized for image generation")
    enhancement_notes: List[str] = Field(default_factory=list, description="Notes on what was enhanced")
    style_tags: List[str] = Field(default_factory=list, description="Style tags added to the prompt")
    technical_parameters: Dict[str, Any] = Field(default_factory=dict, description="Technical parameters for generation")


class PromptRefinementBatch(BaseModel):
    """Batch of refined prompts from Agent 4"""
    refined_prompts: List[RefinedPrompt] = Field(default_factory=list, description="All refined prompts")
    global_style_guide: str = Field(description="Global style guide applied to all prompts")
    consistency_notes: List[str] = Field(default_factory=list, description="Notes on maintaining consistency")


# ============================================================================
# AGENT 5: IMAGE EDITOR MODELS
# ============================================================================

class EditOperation(str, Enum):
    """Types of edit operations"""
    MASK_EDIT = "mask_edit"
    INPAINT = "inpaint"
    STYLE_TRANSFER = "style_transfer"
    COLOR_CORRECTION = "color_correction"
    RESIZE = "resize"


class ImageEdit(BaseModel):
    """Model for an image edit operation"""
    original_image_path: str = Field(description="Path to the original image")
    edited_image_path: str = Field(description="Path to the edited image")
    operation: EditOperation = Field(description="Type of edit operation performed")
    edit_prompt: Optional[str] = Field(default=None, description="Prompt used for the edit")
    mask_path: Optional[str] = Field(default=None, description="Path to mask if used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Edit parameters")
    success: bool = Field(description="Whether the edit was successful")
    notes: Optional[str] = Field(default=None, description="Additional notes about the edit")


class ImageEditBatch(BaseModel):
    """Batch of image edits from Agent 5"""
    edits: List[ImageEdit] = Field(default_factory=list, description="All edit operations")
    total_edits: int = Field(description="Total number of edits performed")
    successful_edits: int = Field(description="Number of successful edits")
    failed_edits: int = Field(description="Number of failed edits")


# ============================================================================
# AGENT 6: IMAGE REVIEWER MODELS
# ============================================================================

class ImageReviewScores(BaseModel):
    """Detailed scores for image review"""
    prompt_accuracy: int = Field(ge=0, le=40, description="Prompt accuracy score (0-40)")
    background_compliance: int = Field(ge=0, le=30, description="Background compliance score (0-30)")
    technical_quality: int = Field(ge=0, le=20, description="Technical quality score (0-20)")
    production_readiness: int = Field(ge=0, le=10, description="Production readiness score (0-10)")


class ImageAssessment(BaseModel):
    """Assessment details for an image"""
    strengths: List[str] = Field(default_factory=list, description="What works well in this image")
    issues: List[str] = Field(default_factory=list, description="Problems or concerns identified")
    missing_elements: List[str] = Field(default_factory=list, description="Elements from prompt that are missing")
    ai_artifacts: List[str] = Field(default_factory=list, description="AI generation artifacts detected")


class ImageReviewFeedback(BaseModel):
    """Feedback for image improvements"""
    for_edit: str = Field(description="Specific edits required if needs_edit")
    for_regeneration: str = Field(description="How to modify prompt if regenerate")
    general_notes: str = Field(description="Additional observations")


class ProductionNotes(BaseModel):
    """Production workflow notes"""
    compositing_ready: bool = Field(description="Whether ready for compositing")
    concerns: List[str] = Field(default_factory=list, description="Production workflow concerns")
    recommendations: List[str] = Field(default_factory=list, description="Suggestions for improvement")


class ImageReviewResult(BaseModel):
    """Complete image review result from Agent 6"""
    asset_name: str = Field(description="Name of the asset")
    asset_type: str = Field(description="Type of asset: character/location/prop")
    image_index: int = Field(description="Image iteration number")
    decision: str = Field(description="Decision: approved/needs_edit/regenerate")
    overall_score: int = Field(ge=0, le=100, description="Overall score (0-100)")
    scores: ImageReviewScores = Field(description="Detailed scores")
    assessment: ImageAssessment = Field(description="Assessment details")
    feedback: ImageReviewFeedback = Field(description="Feedback for improvements")
    production_notes: ProductionNotes = Field(description="Production notes")


# Legacy models for backward compatibility
class ImageQualityScore(BaseModel):
    """Quality scores for an image"""
    technical_quality: int = Field(ge=1, le=10, description="Technical quality score (1-10)")
    visual_accuracy: int = Field(ge=1, le=10, description="Visual accuracy to description (1-10)")
    artistic_quality: int = Field(ge=1, le=10, description="Artistic quality score (1-10)")
    consistency: int = Field(ge=1, le=10, description="Consistency with other assets (1-10)")
    overall_score: int = Field(ge=1, le=10, description="Overall quality score (1-10)")


class ImageReview(BaseModel):
    """Review of a generated image"""
    image_path: str = Field(description="Path to the image being reviewed")
    asset_name: str = Field(description="Name of the asset")
    asset_type: str = Field(description="Type of asset: character/location/prop")
    quality_scores: ImageQualityScore = Field(description="Quality scores")
    issues_found: List[str] = Field(default_factory=list, description="List of issues found")
    strengths: List[str] = Field(default_factory=list, description="List of strengths")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    approved: bool = Field(description="Whether the image is approved")
    requires_regeneration: bool = Field(description="Whether regeneration is needed")
    requires_editing: bool = Field(description="Whether editing is needed")


class ImageReviewBatch(BaseModel):
    """Batch review of images from Agent 6"""
    reviews: List[ImageReview] = Field(default_factory=list, description="All image reviews")
    total_reviewed: int = Field(description="Total number of images reviewed")
    total_approved: int = Field(description="Number of approved images")
    total_needs_work: int = Field(description="Number of images needing work")
    overall_batch_quality: float = Field(description="Average quality score for the batch")
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues across all images")
    batch_approved: bool = Field(description="Whether the entire batch is approved")


# ============================================================================
# AGENT 7: VARIATION GENERATOR MODELS
# ============================================================================

class ImageVariation(BaseModel):
    """Model for an image variation"""
    original_image_path: str = Field(description="Path to the original image")
    variation_image_path: str = Field(description="Path to the variation image")
    variation_number: int = Field(description="Variation number")
    variation_description: str = Field(description="Description of what was varied")
    seed_used: Optional[int] = Field(default=None, description="Seed used for generation")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Generation parameters")
    success: bool = Field(description="Whether variation generation was successful")


class VariationBatch(BaseModel):
    """Batch of variations from Agent 7"""
    variations: List[ImageVariation] = Field(default_factory=list, description="All generated variations")
    base_asset_name: str = Field(description="Name of the base asset")
    total_variations_requested: int = Field(description="Total variations requested")
    successful_variations: int = Field(description="Number of successful variations")
    failed_variations: int = Field(description="Number of failed variations")
