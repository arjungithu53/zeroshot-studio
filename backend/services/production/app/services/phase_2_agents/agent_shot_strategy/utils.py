"""
Utility functions for processing shot lists and annotating output.

Contains helper functions for shot list analysis, continuity detection,
and output formatting.
"""

from typing import List, Dict, Tuple, Optional
import re
from backend.services.production.app.models.mongodb.shots import ShotItem, AnnotatedShotItem, ShotList, AnnotatedShotList
from .data_schema import GenerationStrategy


def analyze_shot_continuity(shots: List[ShotItem]) -> List[Dict[str, any]]:
    """
    Analyze visual and action continuity between consecutive shots.
    
    Args:
        shots: List of shot items to analyze
        
    Returns:
        List of dictionaries containing continuity analysis for each shot
    """
    continuity_analysis = []
    
    for i, shot in enumerate(shots):
        analysis = {
            "shot_id": shot.shot_id,
            "shot": shot,
            "is_first_shot": i == 0,
            "is_last_shot": i == len(shots) - 1,
            "previous_shot": shots[i-1] if i > 0 else None,
            "next_shot": shots[i+1] if i < len(shots) - 1 else None,
            "continuity_indicators": []
        }
        
        # Analyze continuity with previous shot
        if i > 0:
            prev_shot = shots[i-1]
            continuity_indicators = _detect_continuity_indicators(prev_shot, shot)
            analysis["continuity_indicators"] = continuity_indicators
        
        continuity_analysis.append(analysis)
    
    return continuity_analysis


def _detect_continuity_indicators(prev_shot: ShotItem, current_shot: ShotItem) -> List[str]:
    """
    Detect continuity indicators between two consecutive shots.
    
    Args:
        prev_shot: Previous shot item
        current_shot: Current shot item
        
    Returns:
        List of continuity indicator strings
    """
    indicators = []
    
    # Check for character continuity
    if _has_character_overlap(prev_shot.description, current_shot.description):
        indicators.append("character_continuity")
    
    # Check for location continuity
    if _has_location_overlap(prev_shot.description, current_shot.description):
        indicators.append("location_continuity")
    
    # Check for action continuity
    if _has_action_continuity(prev_shot.description, current_shot.description):
        indicators.append("action_continuity")
    
    # Check for camera movement continuity
    if _has_camera_continuity(prev_shot.description, current_shot.description):
        indicators.append("camera_continuity")
    
    return indicators


def _has_character_overlap(desc1: str, desc2: str) -> bool:
    """Check if two shot descriptions share characters."""
    # Simple keyword-based approach - could be enhanced with NLP
    character_keywords = ["protagonist", "character", "person", "man", "woman", "actor"]
    desc1_lower = desc1.lower()
    desc2_lower = desc2.lower()
    
    for keyword in character_keywords:
        if keyword in desc1_lower and keyword in desc2_lower:
            return True
    return False


def _has_location_overlap(desc1: str, desc2: str) -> bool:
    """Check if two shot descriptions share locations."""
    location_keywords = ["room", "house", "street", "forest", "office", "car", "building"]
    desc1_lower = desc1.lower()
    desc2_lower = desc2.lower()
    
    for keyword in location_keywords:
        if keyword in desc1_lower and keyword in desc2_lower:
            return True
    return False


def _has_action_continuity(desc1: str, desc2: str) -> bool:
    """Check if two shot descriptions have action continuity."""
    action_keywords = ["walking", "running", "talking", "looking", "moving", "gesturing"]
    desc1_lower = desc1.lower()
    desc2_lower = desc2.lower()
    
    for keyword in action_keywords:
        if keyword in desc1_lower and keyword in desc2_lower:
            return True
    return False


def _has_camera_continuity(desc1: str, desc2: str) -> bool:
    """Check if two shot descriptions have camera movement continuity."""
    camera_keywords = ["pan", "zoom", "track", "dolly", "close-up", "wide", "medium"]
    desc1_lower = desc1.lower()
    desc2_lower = desc2.lower()
    
    for keyword in camera_keywords:
        if keyword in desc1_lower and keyword in desc2_lower:
            return True
    return False


def suggest_generation_strategy(
    shot: ShotItem, 
    continuity_analysis: Dict[str, any],
    context: Optional[str] = None
) -> Tuple[GenerationStrategy, str, float]:
    """
    Suggest generation strategy based on shot analysis and context.
    
    Args:
        shot: Current shot item
        continuity_analysis: Continuity analysis for the shot
        context: Additional context string
        
    Returns:
        Tuple of (strategy, reasoning, confidence_score)
    """
    # Default strategy and reasoning
    strategy = "generate_new"
    reasoning = "No specific continuity detected, generating new content"
    confidence = 0.7
    
    # Check for strong continuity indicators
    if continuity_analysis.get("continuity_indicators"):
        indicators = continuity_analysis["continuity_indicators"]
        
        if "character_continuity" in indicators and "action_continuity" in indicators:
            strategy = "last_frame_seed"
            reasoning = "Strong character and action continuity detected, using last frame as seed"
            confidence = 0.9
        elif "location_continuity" in indicators and "character_continuity" in indicators:
            strategy = "last_frame_seed"
            reasoning = "Same character and location with static poses, can reuse previous frame with minor viewpoint changes"
            confidence = 0.8
        elif "character_continuity" in indicators:
            strategy = "last_frame_seed"
            reasoning = "Character continuity detected, using last frame as seed"
            confidence = 0.8
    
    # Check if it's the first shot
    if continuity_analysis.get("is_first_shot", False):
        strategy = "generate_new"
        reasoning = "First shot of sequence, no previous context available"
        confidence = 0.95
    
    # Check for scene changes
    if _is_scene_change(shot, continuity_analysis):
        strategy = "generate_new"
        reasoning = "Scene change detected, generating new content"
        confidence = 0.9
    
    return strategy, reasoning, confidence


def _is_scene_change(shot: ShotItem, continuity_analysis: Dict[str, any]) -> bool:
    """Check if this shot represents a scene change."""
    # Check if scene number changed
    if continuity_analysis.get("previous_shot"):
        prev_scene = continuity_analysis["previous_shot"].scene_number
        current_scene = shot.scene_number
        
        if prev_scene and current_scene and prev_scene != current_scene:
            return True
    
    # Check for location change keywords
    location_change_keywords = ["new scene", "different location", "cut to", "meanwhile"]
    shot_desc_lower = shot.description.lower()
    
    for keyword in location_change_keywords:
        if keyword in shot_desc_lower:
            return True
    
    return False


def format_shot_list_for_llm(shot_list: ShotList) -> str:
    """
    Format shot list for LLM processing with clear structure.
    
    Args:
        shot_list: Input shot list
        
    Returns:
        Formatted string representation
    """
    formatted = f"Episode: {shot_list.episode_id}"
    if shot_list.title:
        formatted += f" - {shot_list.title}"
    formatted += "\n\n"
    
    formatted += "Shot List:\n"
    for i, shot in enumerate(shot_list.shots, 1):
        formatted += f"{i}. Shot ID: {shot.shot_id}\n"
        formatted += f"   Description: {shot.description}\n"
        if shot.duration:
            formatted += f"   Duration: {shot.duration}s\n"
        if shot.scene_number:
            formatted += f"   Scene: {shot.scene_number}\n"
        if shot.sequence_number:
            formatted += f"   Sequence: {shot.sequence_number}\n"
        if shot.optimized_ai_notes:
            formatted += f"   Optimized AI Notes: {shot.optimized_ai_notes}\n"
        formatted += "\n"
    
    return formatted


def create_strategy_summary(annotated_shots: List[AnnotatedShotItem]) -> Dict[str, int]:
    """
    Create summary of strategy distribution across all shots.
    
    Args:
        annotated_shots: List of annotated shot items
        
    Returns:
        Dictionary with strategy counts
    """
    summary = {
        "generate_new": 0,
        "multi_shot": 0,
        "last_frame_seed": 0
    }
    
    for shot in annotated_shots:
        strategy = shot.generation_strategy
        if strategy in summary:
            summary[strategy] += 1
    
    return summary


def validate_annotated_output(annotated_list: AnnotatedShotList) -> List[str]:
    """
    Validate annotated shot list output for consistency and completeness.
    
    Args:
        annotated_list: Annotated shot list to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check that all shots have strategies
    for shot in annotated_list.annotated_shots:
        if not shot.generation_strategy:
            errors.append(f"Shot {shot.shot_id} missing generation strategy")
        
        # Check for reasoning in versioned structure (v0) or legacy field
        has_reasoning = False
        if hasattr(shot, 'image') and shot.image and 'v0' in shot.image:
            has_reasoning = bool(shot.image['v0'].get('reasoning'))
        elif hasattr(shot, 'reasoning') and shot.reasoning:
            has_reasoning = True
        
        if not has_reasoning:
            errors.append(f"Shot {shot.shot_id} missing reasoning in versioned structure")
        
        if shot.confidence_score < 0.0 or shot.confidence_score > 1.0:
            errors.append(f"Shot {shot.shot_id} has invalid confidence score: {shot.confidence_score}")
    
    # Check strategy summary consistency
    expected_summary = create_strategy_summary(annotated_list.annotated_shots)
    if annotated_list.strategy_summary != expected_summary:
        errors.append("Strategy summary does not match actual shot strategies")
    
    return errors


def extract_continuity_notes(continuity_analysis: List[Dict[str, any]]) -> str:
    """
    Extract high-level continuity notes from analysis.
    
    Args:
        continuity_analysis: List of continuity analysis results
        
    Returns:
        Formatted continuity notes string
    """
    notes = []
    
    # Count continuity indicators
    total_continuity = 0
    for analysis in continuity_analysis:
        if analysis.get("continuity_indicators"):
            total_continuity += len(analysis["continuity_indicators"])
    
    if total_continuity > 0:
        notes.append(f"Strong continuity detected across {total_continuity} shot transitions")
    else:
        notes.append("Limited continuity detected, mostly independent shots")
    
    # Check for scene changes
    scene_changes = sum(
        1 for analysis in continuity_analysis
        if analysis.get("shot") and _is_scene_change(analysis["shot"], analysis)
    )
    
    if scene_changes > 0:
        notes.append(f"{scene_changes} scene changes detected")
    
    return "; ".join(notes) if notes else "No significant continuity patterns detected"
