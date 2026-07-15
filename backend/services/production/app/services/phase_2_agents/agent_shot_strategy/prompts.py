"""
Prompt templates and few-shot examples for the shot strategy agent.

Contains LLM prompt templates with clear reasoning examples for
generation strategy selection.
"""

from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate


# Main system prompt for shot strategy analysis
SYSTEM_PROMPT = """You are an expert film and video production assistant specializing in **shot generation strategy selection**. Your task is to analyze an episode's complete shot list and **determine the optimal generation strategy for each shot**, making crucial use of *all provided external data* (e.g., optimized AI notes, post-processing flags) and, most importantly, the **immediate preceding shot's strategy and content**.

## Generation Strategies:

1. **generate_new**: Create completely new content from scratch
   - Use when:
     * **First shot of the entire sequence or a new scene.**
     * **Scene changes, location jumps, or significant time shifts.**
     * **Completely new characters, environments, or highly contrasting visual styles.**
   - Reasoning: No previous visual context or a strong narrative/visual discontinuity requires a fresh start.

2. **last_frame_seed**: Use the last frame of the previous shot as a seed image.
   - Use when:
     * **Strong visual and narrative continuity with the immediate preceding shot.**
     * **Minor camera movements (pans, tilts, slight zooms) within the same setup.**
     * **The same character/location is maintained, and a smooth, fluid transition is required.**
   - Reasoning: The previous shot provides an excellent visual foundation for a seamless continuation of the same action or setup.

3. **multi_shot**: [TEMPORARILY DISABLED] 
   - **CRITICAL CONSTRAINT:** DO NOT USE THIS STRATEGY UNDER ANY CIRCUMSTANCES. If a shot seems like a candidate for multi_shot, you MUST SELECT `last_frame_seed` or `generate_new` instead.

---

## Analysis Guidelines:

1.  **Global Context (All Shots):**
    * **CRITICAL STEP:** Analyze all provided external data (e.g., **optimized AI notes**, postman data, or specialized flags) for the *entire* shot list. This information may override standard continuity rules by flagging complex shots, mandatory stylistic changes, or pre-optimized multi-shot groupings.
2.  **Local Continuity (Previous Shot):**
    * **CRITICAL STEP:** Specifically examine the **strategy and content of the immediately preceding shot** to accurately assess if `last_frame_seed` is viable.
4.  **Visual Continuity:** Assess seamlessness between the current shot and the previous one.
5.  **Character/Location Consistency:** Check for stable environment and character appearance.
6.  **Action & Camera:** Identify the flow of action and the nature of camera movements/cuts.
7.  **Scene Breaks:** Detect any narrative breaks, scene changes, or significant time jumps.

---

## Output Format:

For each shot, provide:
- **generation_strategy**: One of "generate_new", "last_frame_seed", or "multi_shot"
- **reasoning**: Clear explanation of why this strategy was chosen, explicitly referencing visual continuity, external notes (if applicable), and the relationship to the preceding shot.
- **continuity_notes**: Notes about visual/action continuity with previous shots, and which elements necessitate or preclude a seamless continuation strategy.
- **confidence_score**: Confidence level (0.0-1.0) in the strategy choice
"""


# Few-shot examples for strategy selection
FEW_SHOT_EXAMPLES = [
    {
        "input": {
            "episode_id": "E01",
            "shots": [
                {
                    "shot_id": "S01E01_001",
                    "description": "Opening shot: Wide establishing shot of a bustling city street at dawn",
                    "scene_number": 1,
                    "sequence_number": 1
                },
                {
                    "shot_id": "S01E01_002", 
                    "description": "Close-up of protagonist's face as they walk through the crowd",
                    "scene_number": 1,
                    "sequence_number": 2
                },
                {
                    "shot_id": "S01E01_003",
                    "description": "Medium shot of protagonist entering a coffee shop",
                    "scene_number": 1,
                    "sequence_number": 3
                },
                {
                    "shot_id": "S01E01_004",
                    "description": "Interior shot: Protagonist ordering coffee at the counter",
                    "scene_number": 2,
                    "sequence_number": 1
                }
            ]
        },
        "output": {
            "annotated_shots": [
                {
                    "shot_id": "S01E01_001",
                    "generation_strategy": "generate_new",
                    "reasoning": "First shot of the episode with no previous context, requires completely new generation",
                    "continuity_notes": "This is the opening shot of the episode",
                    "confidence_score": 0.95,
                    "seed_shot_id": None
                },
                {
                    "shot_id": "S01E01_002",
                    "generation_strategy": "last_frame_seed",
                    "reasoning": "Strong continuity with previous shot - same character, same location, smooth transition from wide to close-up",
                    "continuity_notes": "Character continuity with previous shot, camera movement from wide to close-up",
                    "confidence_score": 0.9,
                    "seed_shot_id": "S01E01_001"
                },
                {
                    "shot_id": "S01E01_003",
                    "generation_strategy": "last_frame_seed",
                    "reasoning": "Character continuity maintained, natural progression of protagonist's movement",
                    "continuity_notes": "Character continuity, action progression from walking to entering",
                    "confidence_score": 0.85,
                    "seed_shot_id": "S01E01_002"
                },
                {
                    "shot_id": "S01E01_004",
                    "generation_strategy": "generate_new",
                    "reasoning": "Scene change from exterior to interior, new location context required",
                    "continuity_notes": "Scene change from exterior street to interior coffee shop",
                    "confidence_score": 0.9,
                    "seed_shot_id": None
                }
            ]
        }
    },
    {
        "input": {
            "episode_id": "E02",
            "shots": [
                {
                    "shot_id": "S01E02_001",
                    "description": "Medium shot of protagonist sitting at desk in office, typing on computer",
                    "scene_number": 1,
                    "sequence_number": 1
                },
                {
                    "shot_id": "S01E02_002",
                    "description": "Close-up of protagonist's hands on keyboard, same office background",
                    "scene_number": 1, 
                    "sequence_number": 2
                },
                {
                    "shot_id": "S01E02_003",
                    "description": "Wide shot of protagonist at desk, same office setting",
                    "scene_number": 1,
                    "sequence_number": 3
                }
            ]
        },
        "output": {
            "annotated_shots": [
                {
                    "shot_id": "S01E02_001",
                    "generation_strategy": "generate_new",
                    "reasoning": "First shot of the sequence, no previous context available",
                    "continuity_notes": "This is the first shot of the sequence",
                    "confidence_score": 0.95,
                    "seed_shot_id": None
                },
                {
                    "shot_id": "S01E02_002",
                    "generation_strategy": "last_frame_seed",
                    "reasoning": "Same character, same office environment, view changes from medium to close-up",
                    "continuity_notes": "Same character and location, minor viewpoint change",
                    "confidence_score": 0.9,
                    "seed_shot_id": "S01E02_001"
                },
                {
                    "shot_id": "S01E02_003",
                    "generation_strategy": "last_frame_seed",
                    "reasoning": "Same character and office environment, camera angle change from close-up to wide",
                    "continuity_notes": "Same character and location, camera movement from close-up to wide shot",
                    "confidence_score": 0.85,
                    "seed_shot_id": "S01E02_001"
                }
            ]
        }
    },
    {
        "input": {
            "episode_id": "E03",
            "shots": [
                {
                    "shot_id": "S01E03_001",
                    "description": "Close-up of a bird perched on a tree branch",
                    "scene_number": 1,
                    "sequence_number": 1
                },
                {
                    "shot_id": "S01E03_002",
                    "description": "Character's POV looking up at the bird on the branch",
                    "scene_number": 1,
                    "sequence_number": 2
                },
                {
                    "shot_id": "S01E03_003",
                    "description": "Close-up of character's face smiling",
                    "scene_number": 1,
                    "sequence_number": 3
                }
            ]
        },
        "output": {
            "annotated_shots": [
                {
                    "shot_id": "S01E03_001",
                    "generation_strategy": "generate_new",
                    "reasoning": "First shot of sequence, no previous context available",
                    "continuity_notes": "Establishes the bird and environment",
                    "confidence_score": 0.95,
                    "seed_shot_id": None
                },
                {
                    "shot_id": "S01E03_002",
                    "generation_strategy": "last_frame_seed",
                    "reasoning": "Same bird and environment from previous shot, perspective changes to POV angle",
                    "continuity_notes": "Continuous bird from previous shot with POV perspective change",
                    "confidence_score": 0.9,
                    "seed_shot_id": "S01E03_001"
                },
                {
                    "shot_id": "S01E03_003",
                    "generation_strategy": "generate_new",
                    "reasoning": "New subject (character) introduced, requires fresh generation",
                    "continuity_notes": "Cuts to new subject, maintains scene environment",
                    "confidence_score": 0.9,
                    "seed_shot_id": None
                }
            ]
        }
    }
]


def _format_shot_list(input_data: Dict[str, Any]) -> str:
    """Format shot list input for display in examples."""
    formatted = f"Episode: {input_data['episode_id']}\n\n"
    formatted += "Shots:\n"

    for shot in input_data['shots']:
        formatted += f"- {shot['shot_id']}: {shot['description']}\n"
        if shot.get('scene_number'):
            formatted += f"  Scene: {shot['scene_number']}\n"
        if shot.get('sequence_number'):
            formatted += f"  Sequence: {shot['sequence_number']}\n"

    return formatted


def _format_annotated_output(output_data: Dict[str, Any]) -> str:
    """Format annotated output for display in examples."""
    formatted = "Strategy Analysis:\n\n"

    for shot in output_data['annotated_shots']:
        formatted += f"- {shot['shot_id']}:\n"
        formatted += f"  Strategy: {shot['generation_strategy']}\n"
        formatted += f"  Reasoning: {shot['reasoning']}\n"
        formatted += f"  Confidence: {shot['confidence_score']}\n\n"

    return formatted


def _format_few_shot_examples() -> str:
    """Format few-shot examples for inclusion in the prompt."""
    examples_text = ""

    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"\n### Example {i}:\n\n"
        examples_text += f"**Input:**\n{_format_shot_list(example['input'])}\n\n"
        examples_text += f"**Output:**\n{_format_annotated_output(example['output'])}\n"

    return examples_text


# Main prompt template for shot strategy analysis
SHOT_STRATEGY_PROMPT = PromptTemplate(
    input_variables=["shot_list_text", "continuity_analysis"],
    template="""{system_prompt}

## Current Episode Analysis:

{shot_list_text}

## Continuity Analysis:
{continuity_analysis}

## Task:
Analyze EVERY shot in the list and determine the optimal generation strategy. Consider:
- Visual continuity with previous shots
- Character and location consistency
- Action sequences and camera movements
- Scene changes and narrative breaks
- Complexity of generation requirements
- Optimized AI notes for generation guidance

IMPORTANT: Keep all 'reasoning' and 'continuity_notes' fields CONCISE (1-2 sentences max).

Provide your analysis as a structured JSON object matching the schema below.

You MUST include ALL shots from the input list in your response. Keep reasoning brief to avoid truncation.

## Few-Shot Examples:

{examples}

Now analyze the provided shot list and provide your strategy recommendations.""",
    partial_variables={
        "system_prompt": SYSTEM_PROMPT,
        "examples": _format_few_shot_examples()
    }
)


# Prompt for continuity analysis
CONTINUITY_ANALYSIS_PROMPT = PromptTemplate(
    input_variables=["shot_list_text"],
    template="""Analyze the visual and narrative continuity between consecutive shots in this episode.

Shot List:
{shot_list_text}

For each shot, identify:
1. Visual continuity indicators (characters, locations, camera movements)
2. Action continuity (smooth transitions vs. breaks)
3. Scene changes and narrative breaks
4. Complexity factors that might affect generation

Provide a structured analysis of continuity patterns and potential generation challenges."""
)


# Prompt for strategy validation
STRATEGY_VALIDATION_PROMPT = PromptTemplate(
    input_variables=["annotated_shots", "original_shots"],
    template="""Review the generated shot strategies for consistency and accuracy.

Original Shots:
{original_shots}

Generated Strategies:
{annotated_shots}

Check for:
1. Logical consistency in strategy choices
2. Appropriate reasoning for each decision
3. Confidence scores that match the complexity
4. Overall strategy distribution makes sense

Identify any inconsistencies or improvements needed."""
)


def get_continuity_analysis_prompt(shot_list_text: str) -> str:
    """Get formatted continuity analysis prompt."""
    return CONTINUITY_ANALYSIS_PROMPT.format(shot_list_text=shot_list_text)


def get_strategy_validation_prompt(annotated_shots: str, original_shots: str) -> str:
    """Get formatted strategy validation prompt."""
    return STRATEGY_VALIDATION_PROMPT.format(
        annotated_shots=annotated_shots,
        original_shots=original_shots
    )


def get_shot_strategy_prompt(shot_list_text: str, continuity_analysis: str) -> str:
    """Get formatted shot strategy analysis prompt."""
    return SHOT_STRATEGY_PROMPT.format(
        shot_list_text=shot_list_text,
        continuity_analysis=continuity_analysis
    )
