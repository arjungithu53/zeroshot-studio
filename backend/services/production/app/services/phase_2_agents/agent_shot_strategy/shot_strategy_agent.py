"""
Main LangChain agent implementation for shot strategy selection.

This module contains the ShotStrategyAgent class that analyzes episode shot lists
and determines optimal generation strategies for each shot.
"""

import json
import logging
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from backend.services.production.app.models.mongodb.shots import ShotList, AnnotatedShotList, AnnotatedShotItem
from .data_schema import GenerationStrategy, ShotStrategyResponse, AnnotatedShotStrategy
from .utils import (
    analyze_shot_continuity,
    format_shot_list_for_llm,
    create_strategy_summary,
    validate_annotated_output,
    extract_continuity_notes
)
from .prompts import (
    get_shot_strategy_prompt,
    get_continuity_analysis_prompt,
    get_strategy_validation_prompt
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_json(text):
    """
    Extract valid JSON from LLM response text that may contain explanations around JSON.
    Tries the following in order:
      1) Parse any fenced code blocks (```json ...``` or ``` ...```)
      2) Parse the largest JSON-looking object {...}
      3) Parse the largest JSON-looking array [...]

    Args:
        text: Raw text response from LLM that may contain JSON

    Returns:
        Parsed JSON object

    Raises:
        ValueError: If no valid JSON is found in the response
    """
    if text is None:
        raise ValueError("No JSON found in LLM response.")

    # 1) Look for fenced code blocks, prioritizing ```json
    # More specific pattern to capture content between fences
    code_blocks = re.findall(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE)
    for block in code_blocks:
        candidate = block.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            logger.info("Successfully extracted JSON from code block")
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Found code block but JSON parsing failed: {e}")
            # Check if the JSON appears truncated
            if candidate.rstrip().endswith((',', ':', '"', '{')):
                raise ValueError(
                    "JSON in code block appears truncated (ends with incomplete syntax). "
                    "The LLM response may have hit token limits."
                )
            continue

    # 2) Also try to find complete JSON objects/arrays without fences
    # Look for complete JSON structures
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'  # Nested arrays
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # 3) Fallback to simple object/array extraction
    candidates = []
    # Objects - try both greedy and non-greedy
    candidates += re.findall(r"\{[\s\S]*\}", text)
    candidates += re.findall(r"\{[\s\S]*?\}", text)
    # Arrays
    candidates += re.findall(r"\[[\s\S]*\]", text)
    candidates += re.findall(r"\[[\s\S]*?\]", text)

    # De-duplicate and sort by length descending to try the most complete first
    unique_candidates = sorted(set(candidates), key=lambda s: len(s), reverse=True)
    for cand in unique_candidates:
        candidate = cand.strip()
        # Strip surrounding code fencing if accidentally captured
        if candidate.startswith("```") and candidate.endswith("```"):
            candidate = candidate.strip("`")
        try:
            parsed = json.loads(candidate)
            logger.info("Successfully extracted JSON from text pattern matching")
            return parsed
        except json.JSONDecodeError:
            continue

    # Provide more detailed error message
    logger.error(f"Failed to extract valid JSON from LLM response. Response preview: {text[:500]}...")
    raise ValueError(
        "No valid JSON found in LLM response. The LLM may have returned plain text instead of JSON, "
        "or the response was truncated. Ensure the LLM is following the JSON format instructions."
    )


def save_strategies_to_file(annotated_list: AnnotatedShotList, output_dir: str = "phase_2_agents/outputs/agent_shot_strategy") -> str:
    """
    Save generated strategies to a JSON file with timestamp.
    
    Args:
        annotated_list: Annotated shot list to save
        output_dir: Directory to save the file (default: "phase_2_agents/outputs/agent_shot_strategy")
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"strategies_{annotated_list.episode_id}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to dictionary for JSON serialization
    strategies_data = {
        "episode_id": annotated_list.episode_id,
        "title": annotated_list.title,
        "generated_at": datetime.now().isoformat(),
        "total_shots": len(annotated_list.annotated_shots),
        "strategy_summary": annotated_list.strategy_summary,
        "overall_continuity_notes": annotated_list.overall_continuity_notes,
        "annotated_shots": []
    }
    
    # Convert each annotated shot to dictionary
    for shot in annotated_list.annotated_shots:
        shot_dict = {
            "shot_id": shot.shot_id,
            "description": shot.description,
            "duration": shot.duration,
            "scene_number": shot.scene_number,
            "sequence_number": shot.sequence_number,
            "shot_style": shot.shot_style,
            "camera_movement": shot.camera_movement,
            "source_type": shot.source_type,
            "uploaded_image_id": shot.uploaded_image_id,
            "generated_image_id": shot.generated_image_id,
            "generated_video_id": shot.generated_video_id,
            "generation_strategy": shot.generation_strategy,
            "reasoning": shot.image["v0"]["reasoning"] if shot.image and "v0" in shot.image else (shot.reasoning if hasattr(shot, 'reasoning') else None),
            "continuity_notes": shot.continuity_notes,
            "confidence_score": shot.confidence_score,
            "seed_shot_id": shot.seed_shot_id
        }
        strategies_data["annotated_shots"].append(shot_dict)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(strategies_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Strategies saved to: {filepath}")
    return filepath


class ShotStrategyAgent:
    """
    LangChain-based agent for analyzing shot lists and determining generation strategies.
    
    This agent takes episode shot lists as input and outputs annotated shot lists
    with recommended generation strategies:
    - generate_new: Create completely new content from scratch
    - last_frame_seed: Use the last frame of the previous shot as a seed for continuity
    - multi_shot: Reuse a single generated image across multiple shots with shared environment/characters
    for each shot.
    """
    
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        model_name: str = "gemini-3.1-pro-preview",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        enable_validation: bool = True,
        enable_saving: bool = True,
        output_dir: str = "phase_2_agents/outputs/agent_shot_strategy",
        api_key: Optional[str] = None
    ):
        """
        Initialize the ShotStrategyAgent.
        
        Args:
            llm: LangChain LLM instance for strategy analysis (optional, will create Gemini if not provided)
            model_name: Gemini model name (default: gemini-3.1-pro-preview)
            temperature: LLM temperature for generation (default: 0.1 for consistency)
            max_tokens: Maximum tokens for LLM output
            enable_validation: Whether to validate output before returning
            enable_saving: Whether to save generated strategies to files (default: True)
            output_dir: Directory to save strategy files (default: "phase_2_agents/outputs/agent_shot_strategy")
            api_key: Google API key (optional, will use environment variable if not provided)
        """
        import os
        
        # Initialize Gemini LLM if not provided
        if llm is None:
            api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
            
            # Create base LLM with structured output support
            base_llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key,
                max_output_tokens=max_tokens or 8192,  # Increased from 2048 to handle larger shot lists
                transport="rest"
            )
            # Bind structured output schema for Gemini
            self.llm = base_llm.with_structured_output(ShotStrategyResponse)
        else:
            # If LLM is provided, check if it already has structured output
            # If not, bind the schema
            if not hasattr(llm, '_structured_output'):
                self.llm = llm.with_structured_output(ShotStrategyResponse) if hasattr(llm, 'with_structured_output') else llm
            else:
                self.llm = llm
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_validation = enable_validation
        self.enable_saving = enable_saving
        self.output_dir = output_dir
        
        logger.info(f"Initialized ShotStrategyAgent with Gemini model: {self.model_name}")
    
    def analyze_shot_list(self, shot_list: ShotList) -> AnnotatedShotList:
        """
        Analyze a shot list and determine generation strategies for each shot.
        
        Args:
            shot_list: Input shot list to analyze
            
        Returns:
            AnnotatedShotList with generation strategies and reasoning
            
        Raises:
            ValueError: If shot list is invalid or analysis fails
        """
        logger.info(f"Analyzing shot list for episode: {shot_list.episode_id}")
        
        try:
            # Step 1: Analyze continuity between shots
            continuity_analysis = analyze_shot_continuity(shot_list.shots)
            logger.debug(f"Continuity analysis completed for {len(shot_list.shots)} shots")
            
            # Step 2: Format input for LLM
            shot_list_text = format_shot_list_for_llm(shot_list)
            continuity_text = self._format_continuity_analysis(continuity_analysis)
            
            # Step 3: Generate strategy recommendations using LLM with structured output
            strategy_response = self._get_strategy_recommendations(
                shot_list_text, 
                continuity_text
            )
            
            # Step 4: Parse and validate response (now receives Pydantic model)
            annotated_shots = self._parse_strategy_response(strategy_response, shot_list.shots)
            
            # Step 5: Create annotated shot list
            annotated_list = AnnotatedShotList(
                episode_id=shot_list.episode_id,
                title=shot_list.title,
                annotated_shots=annotated_shots,
                overall_continuity_notes=extract_continuity_notes(continuity_analysis),
                strategy_summary=create_strategy_summary(annotated_shots)
            )
            
            # Step 6: Validate output if enabled
            if self.enable_validation:
                validation_errors = validate_annotated_output(annotated_list)
                if validation_errors:
                    logger.warning(f"Validation errors found: {validation_errors}")
                    # Continue with warnings rather than failing
            
            # Step 7: Save strategies to file if enabled
            if self.enable_saving:
                try:
                    saved_file = save_strategies_to_file(annotated_list, self.output_dir)
                    logger.info(f"Strategies saved to: {saved_file}")
                except Exception as e:
                    logger.warning(f"Failed to save strategies to file: {e}")
                    # Continue without failing the analysis
            
            logger.info(f"Successfully analyzed {len(annotated_shots)} shots")
            return annotated_list
            
        except Exception as e:
            logger.error(f"Error analyzing shot list: {str(e)}")
            raise ValueError(f"Failed to analyze shot list: {str(e)}")
    
    def _format_continuity_analysis(self, continuity_analysis: List[Dict[str, Any]]) -> str:
        """Format continuity analysis for LLM input."""
        formatted = "Continuity Analysis:\n\n"
        
        for analysis in continuity_analysis:
            formatted += f"Shot {analysis['shot_id']}:\n"
            formatted += f"  First shot: {analysis['is_first_shot']}\n"
            formatted += f"  Last shot: {analysis['is_last_shot']}\n"
            
            if analysis.get('continuity_indicators'):
                formatted += f"  Continuity indicators: {', '.join(analysis['continuity_indicators'])}\n"
            else:
                formatted += f"  Continuity indicators: None\n"
            
            formatted += "\n"
        
        return formatted
    
    def _get_strategy_recommendations(self, shot_list_text: str, continuity_text: str) -> ShotStrategyResponse:
        """Get strategy recommendations from LLM using structured output."""
        try:
            # Create the prompt
            prompt = get_shot_strategy_prompt(shot_list_text, continuity_text)
            
            # Generate response with structured output
            # When using with_structured_output(), the LLM returns the Pydantic model directly
            if isinstance(self.llm, BaseChatModel) or hasattr(self.llm, 'with_structured_output'):
                messages = [
                    SystemMessage(content="You are an expert film production assistant."),
                    SystemMessage(content=(
                        "CRITICAL INSTRUCTIONS:\n"
                        "1. You MUST analyze ALL shots provided and return a complete response\n"
                        "2. Keep 'reasoning' fields concise (1-2 sentences max) to avoid token limits\n"
                        "3. Each shot must include: shot_id, generation_strategy, reasoning (brief!), "
                        "confidence_score (0.0-1.0), continuity_notes (optional, brief), seed_shot_id (optional)\n"
                        "4. Ensure you provide strategies for ALL shots in the input list"
                    )),
                    HumanMessage(content=prompt)
                ]
                # With structured output, invoke returns the Pydantic model directly
                response = self.llm.invoke(messages)
                
                # If response is already a Pydantic model, return it
                if isinstance(response, ShotStrategyResponse):
                    logger.info(f"Received structured response with {len(response.annotated_shots)} shots")
                    return response
                # Fallback: if response has content attribute, try to parse it
                elif hasattr(response, "content"):
                    # This shouldn't happen with structured output, but handle gracefully
                    logger.warning("Received response with content attribute instead of structured output")
                    return ShotStrategyResponse.model_validate_json(response.content)
                else:
                    # Try to parse as JSON string
                    return ShotStrategyResponse.model_validate_json(str(response))
            else:
                # Fallback for non-chat models (shouldn't happen with structured output)
                response = self.llm.invoke(prompt)
                if isinstance(response, ShotStrategyResponse):
                    return response
                return ShotStrategyResponse.model_validate_json(str(response))
                
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {str(e)}")
            raise ValueError(f"LLM analysis failed: {str(e)}")
    
    def _parse_strategy_response(self, response: ShotStrategyResponse, original_shots: List) -> List[AnnotatedShotItem]:
        """Parse structured Pydantic response and create annotated shot items."""
        try:
            # Response is already a validated Pydantic model
            if not isinstance(response, ShotStrategyResponse):
                # Fallback: try to parse if it's a dict or string
                if isinstance(response, dict):
                    response = ShotStrategyResponse.model_validate(response)
                elif isinstance(response, str):
                    response = ShotStrategyResponse.model_validate_json(response)
                else:
                    raise ValueError(f"Unexpected response type: {type(response)}")
            
            # Validate completeness: check if we got responses for all shots
            annotated_shot_data = response.annotated_shots
            if len(annotated_shot_data) < len(original_shots):
                logger.error(
                    f"INCOMPLETE RESPONSE: Expected {len(original_shots)} shots, "
                    f"but only received {len(annotated_shot_data)}. Response may be truncated."
                )
                raise ValueError(
                    f"Incomplete structured response: received {len(annotated_shot_data)}/{len(original_shots)} shots. "
                    "The LLM response was likely truncated. Try reducing reasoning length or increasing max_output_tokens."
                )
            
            # Parse annotated shots from Pydantic model
            annotated_shots = []
            for shot_strategy in annotated_shot_data:
                # Force fallback if the LLM hallucinated multi_shot
                if shot_strategy.generation_strategy == "multi_shot":
                    logger.warning(f"LLM hallucinated disabled strategy 'multi_shot' for shot {shot_strategy.shot_id}, falling back to 'last_frame_seed'")
                    shot_strategy.generation_strategy = "last_frame_seed"

                # Find corresponding original shot
                original_shot = next(
                    (s for s in original_shots if s.shot_id == shot_strategy.shot_id), 
                    None
                )
                
                if not original_shot:
                    logger.warning(f"Could not find original shot for {shot_strategy.shot_id}")
                    continue
                
                # Create annotated shot item
                # Store reasoning in versioned image structure as expected by validation
                image_data = {
                    'v0': {
                        'reasoning': shot_strategy.reasoning,
                        'generation_strategy': shot_strategy.generation_strategy,
                        'confidence_score': float(shot_strategy.confidence_score)
                    }
                }

                annotated_shot = AnnotatedShotItem(
                    shot_id=original_shot.shot_id,
                    description=original_shot.description,
                    duration=original_shot.duration,
                    scene_number=original_shot.scene_number,
                    sequence_number=original_shot.sequence_number,
                    shot_style=original_shot.shot_style,
                    camera_movement=original_shot.camera_movement,
                    source_type=original_shot.source_type,
                    uploaded_image_id=original_shot.uploaded_image_id,
                    generated_image_id=original_shot.generated_image_id,
                    generated_video_id=original_shot.generated_video_id,
                    optimized_ai_notes=original_shot.optimized_ai_notes,
                    characters=original_shot.characters,
                    locations=original_shot.locations,
                    generation_strategy=shot_strategy.generation_strategy,
                    continuity_notes=shot_strategy.continuity_notes or '',
                    confidence_score=float(shot_strategy.confidence_score),
                    seed_shot_id=shot_strategy.seed_shot_id,
                    image=image_data
                )
                
                annotated_shots.append(annotated_shot)
            
            return annotated_shots
            
        except Exception as e:
            logger.error(f"Error parsing strategy response: {e}")
            raise RuntimeError(f"Failed to parse strategy response: {e}")
    
    def batch_analyze(self, shot_lists: List[ShotList]) -> List[AnnotatedShotList]:
        """
        Analyze multiple shot lists in batch.
        
        Args:
            shot_lists: List of shot lists to analyze
            
        Returns:
            List of annotated shot lists
        """
        logger.info(f"Batch analyzing {len(shot_lists)} shot lists")
        
        results = []
        for i, shot_list in enumerate(shot_lists):
            try:
                logger.info(f"Processing shot list {i+1}/{len(shot_lists)}: {shot_list.episode_id}")
                annotated_list = self.analyze_shot_list(shot_list)
                results.append(annotated_list)
            except Exception as e:
                logger.error(f"Error processing shot list {shot_list.episode_id}: {str(e)}")
                # Continue with other shot lists rather than failing completely
                continue
        
        logger.info(f"Batch analysis completed: {len(results)}/{len(shot_lists)} successful")
        return results
    
    def get_strategy_summary(self, annotated_list: AnnotatedShotList) -> Dict[str, Any]:
        """
        Get detailed summary of strategy analysis.
        
        Args:
            annotated_list: Annotated shot list
            
        Returns:
            Dictionary with strategy summary and insights
        """
        total_shots = len(annotated_list.annotated_shots)
        average_confidence = 0.0
        if total_shots > 0:
            average_confidence = sum(s.confidence_score for s in annotated_list.annotated_shots) / total_shots
        
        summary = {
            "episode_id": annotated_list.episode_id,
            "total_shots": total_shots,
            "strategy_distribution": annotated_list.strategy_summary,
            "average_confidence": average_confidence,
            "high_confidence_shots": len([s for s in annotated_list.annotated_shots if s.confidence_score >= 0.8]),
            "continuity_notes": annotated_list.overall_continuity_notes
        }
        
        return summary
    
    def validate_strategies(self, annotated_list: AnnotatedShotList) -> List[str]:
        """
        Validate strategy choices for consistency and logic.
        
        Args:
            annotated_list: Annotated shot list to validate
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check for strategy consistency
        strategies = [shot.generation_strategy for shot in annotated_list.annotated_shots]
        
        # Check if first shot is generate_new (should be)
        first_shot = annotated_list.annotated_shots[0] if annotated_list.annotated_shots else None
        if first_shot and first_shot.generation_strategy != "generate_new":
            warnings.append("First shot should typically use 'generate_new' strategy")
        
        # Check for unusual patterns
        if strategies.count("generate_new") > len(strategies) * 0.7:
            warnings.append("High proportion of 'generate_new' strategies - check for continuity issues")
        
        if strategies.count("multi_shot") > len(strategies) * 0.5:
            warnings.append("High proportion of 'multi_shot' strategies - may indicate many static scenes with shared environments")
        
        # Check confidence scores
        low_confidence = [s for s in annotated_list.annotated_shots if s.confidence_score < 0.6]
        if low_confidence:
            warnings.append(f"{len(low_confidence)} shots have low confidence scores (< 0.6)")
        
        return warnings
    
    def to_mongodb_collection(
        self, 
        annotated_list: AnnotatedShotList, 
        show_id: str, 
        episode_number: int
    ) -> List[dict]:
        """
        Convert annotated shot list to MongoDB collection format.
        
        Args:
            annotated_list: Annotated shot list
            show_id: Show ID for MongoDB reference
            episode_number: Episode number
            
        Returns:
            List of MongoDB collection documents
        """
        from .data_schema import shot_item_to_mongodb
        
        mongodb_docs = []
        
        for shot in annotated_list.annotated_shots:
            # Start with base MongoDB format
            doc = shot_item_to_mongodb(shot, show_id, episode_number)
            
            # Update with strategy information
            doc.update({
                "generation_strategy": shot.generation_strategy,
                "seed_shot_id": shot.seed_shot_id,
                "image": shot.image if hasattr(shot, 'image') and shot.image else None
            })
            
            mongodb_docs.append(doc)
        
        return mongodb_docs


# Example usage and testing functions
def create_example_shot_list() -> ShotList:
    """Create an example shot list for testing."""
    from app.models.mongodb.shots import ShotItem
    
    shots = [
        ShotItem(
            shot_id="S01E01_001",
            description="Wide establishing shot of a modern office building at sunset",
            duration=4.0,
            scene_number=1,
            sequence_number=1
        ),
        ShotItem(
            shot_id="S01E01_002", 
            description="Close-up of protagonist's face as they look out the window",
            duration=3.0,
            scene_number=1,
            sequence_number=2
        ),
        ShotItem(
            shot_id="S01E01_003",
            description="Medium shot of protagonist walking to their desk",
            duration=2.5,
            scene_number=1,
            sequence_number=3
        ),
        ShotItem(
            shot_id="S01E01_004",
            description="Close-up of protagonist's hands typing on keyboard",
            duration=2.0,
            scene_number=1,
            sequence_number=4
        )
    ]
    
    return ShotList(
        episode_id="E01",
        title="The Beginning",
        shots=shots
    )


def test_agent_with_example():
    """Test the agent with an example shot list."""
    # This would be used for testing with a real LLM
    # For now, just return the example shot list
    return create_example_shot_list()
