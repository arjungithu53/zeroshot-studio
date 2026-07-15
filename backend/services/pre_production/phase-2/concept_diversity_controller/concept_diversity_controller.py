import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from bson import ObjectId
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "60000"))

PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")
IDEATION_COLLECTION = os.getenv("COLLECTION_IDEATION", "ideation")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline_logs")

logger = logging.getLogger("zeroshot.phase2.concept_diversity_controller")


class DiversityAuditRow(BaseModel):
    concept_id: str = Field(description="Concept identifier.")
    emotional_axis: int = Field(description="1=functional/rational, 5=deeply emotional/visceral.")
    hook_physics: int = Field(description="1=slow/earned, 5=immediate/disruptive.")
    irony_sincerity: int = Field(description="1=fully sincere, 5=fully ironic/satirical.")
    proof_intensity: int = Field(description="1=purely narrative, 5=explicit product demonstration.")
    protagonist_presence: int = Field(description="1=no human subject, 5=character-driven narrative.")


class ClusterPair(BaseModel):
    concept_a: str = Field(description="First concept_id in clustered pair.")
    concept_b: str = Field(description="Second concept_id in clustered pair.")
    shared_dimensions: List[str] = Field(
        default_factory=list,
        description="Dimensions where concepts are too similar (within 1 point).",
    )


class MutationPromptIssued(BaseModel):
    concept_id: str = Field(description="Concept selected for mutation.")
    dimension_to_change: str = Field(description="Specific creative dimension to move.")
    direction: str = Field(description="Direction for the mutation (increase/decrease + intent).")


class ConceptDiversityControllerResult(BaseModel):
    status: Literal["completed", "skipped", "error"] = Field(default="completed")
    reason: Optional[str] = Field(default=None, description="Reason when skipped or errored.")
    reasoning: str = Field(default="", description="Reasoning behind diversity audit decisions.")
    diversity_audit_matrix: List[DiversityAuditRow] = Field(default_factory=list)
    cluster_pairs: List[ClusterPair] = Field(default_factory=list)
    mutation_prompts_issued: List[MutationPromptIssued] = Field(default_factory=list)
    experimental_concept_flag: str = Field(
        default="",
        description="Concept id flagged as the high-risk/high-reward experimental concept.",
    )


def _clean_json_string(json_str: str) -> str:
    cleaned = json_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_seed_lists_from_pipeline(agent_logs: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    pattern_interrupt_seeds: List[str] = []
    mental_model_seeds: List[str] = []
    intergalactic_seeds: List[str] = []

    for log_entry in agent_logs:
        name = str(log_entry.get("agent_name", ""))
        agent_id = str(log_entry.get("agent_id", ""))
        payload = log_entry.get("output") or log_entry.get("data") or log_entry

        if ("pattern_interrupt" in name) or ("pattern_interrupt" in agent_id):
            pattern_interrupt_seeds = payload.get("seed_list", []) or []
        elif ("mental_model" in name) or ("mental_model" in agent_id):
            mental_model_seeds = payload.get("seed_list", []) or []
        elif ("intergalactic" in name) or ("intergalactic" in agent_id):
            intergalactic_seeds = payload.get("seed_list", []) or []

    return pattern_interrupt_seeds, mental_model_seeds, intergalactic_seeds


async def run_concept_diversity_controller_agent(project_id: str, db: Any) -> ConceptDiversityControllerResult:
    logger.info("Initializing Agent 23 (Concept Diversity Controller) for project_id=%s", project_id)
    start_time = time.time()

    try:
        project_obj_id = ObjectId(project_id)
    except Exception as exc:
        logger.error("Agent 23: Invalid project_id format for ObjectId conversion. project_id=%s error=%s", project_id, exc)
        raise ValueError(f"Invalid project_id {project_id}")

    logger.info("Agent 23: Fetching data for project_id=%s", project_id)
    project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": project_obj_id})
    if not project_doc:
        logger.error("Agent 23: Project document not found in %s for project_id=%s", PROJECTS_COLLECTION, project_id)
        raise ValueError(f"Project not found in {PROJECTS_COLLECTION}")

    ideation_doc = await db[IDEATION_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": str(project_id)}) or {}
    pipeline_doc = await db[PIPELINE_COLLECTION].find_one({"project_id": str(project_id)}) or {}

    concept_portfolio = ideation_doc.get("concept_portfolio", [])
    agent_logs = pipeline_doc.get("agent_logs", []) if isinstance(pipeline_doc.get("agent_logs", []), list) else []
    pattern_interrupt_seeds, mental_model_seeds, intergalactic_seeds = _extract_seed_lists_from_pipeline(agent_logs)

    logger.info("Agent 23: Extracted key inputs from DB docs.")
    logger.info(
        "Agent 23: Input summary | concept_count=%s strategy_agents_present=%s pipeline_logs_count=%s pattern_interrupt_seeds=%s mental_model_seeds=%s intergalactic_seeds=%s",
        len(concept_portfolio) if isinstance(concept_portfolio, list) else 0,
        isinstance(strategy_doc.get("agents", {}), dict) and bool(strategy_doc.get("agents", {})),
        len(agent_logs),
        len(pattern_interrupt_seeds),
        len(mental_model_seeds),
        len(intergalactic_seeds),
    )

    if not isinstance(concept_portfolio, list) or not concept_portfolio:
        logger.error("Agent 23: concept_portfolio missing or empty in %s for project_id=%s", IDEATION_COLLECTION, project_id)
        raise ValueError("concept_portfolio is missing in ideation document")

    prompt_payload = {
        "concept_portfolio": concept_portfolio,
        "pattern_interrupt_seeds": pattern_interrupt_seeds,
        "mental_model_seeds": mental_model_seeds,
        "intergalactic_seeds": intergalactic_seeds,
    }

    prompt = f"""
You are
 concept_diversity_controller.

RUN CONDITION: ALWAYS

Source: Created - Anti-clustering agent

Purpose:
Project concept portfolio into a 5-dimensional creative space:
1) emotional_axis (1-5)
2) hook_physics (1-5)
3) irony_sincerity (1-5)
4) proof_intensity (1-5)
5) protagonist_presence (1-5)

Then identify dangerous clustering where multiple concepts are structurally similar despite surface differences.
Clustering means the campaign tests the same hypothesis repeatedly and reduces A/B learning value.

Input Schema:
- concept_portfolio: [object]
- pattern_interrupt_seeds: [string]
- mental_model_seeds: [string]
- intergalactic_seeds: [string]

Output requirements:
- reasoning: string
- diversity_audit_matrix: [
    {{
      "concept_id": string,
      "emotional_axis": int,
      "hook_physics": int,
      "irony_sincerity": int,
      "proof_intensity": int,
      "protagonist_presence": int
    }}
  ]
- cluster_pairs: [
    {{ "concept_a": string, "concept_b": string, "shared_dimensions": [string] }}
  ]
- mutation_prompts_issued: [
    {{ "concept_id": string, "dimension_to_change": string, "direction": string }}
  ]
- experimental_concept_flag: string

Prompt Logic:
Step 1 - Score each concept 1-5 on each of the five creative dimensions:
- Emotional axis: 1 = functional/rational, 5 = deeply emotional/visceral.
- Hook physics: 1 = slow/earned, 5 = immediate/disruptive.
- Irony/sincerity: 1 = fully sincere, 5 = fully ironic/satirical.
- Proof intensity: 1 = purely narrative, 5 = explicit product demonstration.
- Protagonist presence: 1 = no human subject, 5 = character-driven narrative.

Step 2 - Identify clustering:
If any two concepts score within 1 point on at least 4 of 5 dimensions, they are clustered.
For each clustered pair, issue a targeted mutation prompt for the lower-scoring concept.
Mutation prompts must specify the exact dimension_to_change and explicit direction.
Avoid vague instructions like "be more creative".

Step 3 - Enforce high-risk/high-reward rule:
At least one concept must score >=4 on both hook_physics and irony_sincerity.
Select one such concept and log it in experimental_concept_flag.
If none naturally qualify, choose the best candidate and describe the needed directional mutation in mutation_prompts_issued.

Return JSON that exactly matches the response schema and covers all concepts in concept_portfolio.

INPUT JSON:
{json.dumps(prompt_payload, indent=2)}
"""

    invoke_start = time.time()
    logger.info(f"Agent 23: Preparing to call Gemini model={GEMINI_MODEL}...")
    try:
        client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"api_version": "v1alpha", "timeout": GEMINI_TIMEOUT_MS},
        )
        logger.info(f"Agent 23: Gemini Client instantiated. Sending prompt...")

        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": ConceptDiversityControllerResult.model_json_schema(),
                "automatic_function_calling": {"disable": True},
            }
        )

        api_duration = time.time() - invoke_start
        logger.info(f"Agent 23: Gemini API call completed in {api_duration:.2f}s")
        logger.debug(f"Agent 23: Raw response length={len(response.text)} chars")

        cleaned_json = _clean_json_string(response.text)
        parsed_data = json.loads(cleaned_json)
        logger.info(f"Agent 23: Successfully parsed JSON response.")

        result = ConceptDiversityControllerResult(**parsed_data)
        result.status = "completed"
        logger.info(f"Agent 23: Successfully validated structured output with Pydantic.")
    except Exception as exc:
        logger.error("Agent 23: Error during Gemini inference or JSON parsing for project_id=%s error=%s", project_id, exc)
        raise

    logger.info("Agent 23: Updating IDEATION and PIPELINE collections...")
    try:
        await db[IDEATION_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {
                "$set": {
                    "status.concept_diversity_controller": "completed",
                    "concept_diversity_audit": {
                        "diversity_audit_matrix": [row.model_dump() for row in result.diversity_audit_matrix],
                        "cluster_pairs": [pair.model_dump() for pair in result.cluster_pairs],
                        "mutation_prompts_issued": [item.model_dump() for item in result.mutation_prompts_issued],
                        "experimental_concept_flag": result.experimental_concept_flag,
                        "reasoning": result.reasoning,
                    },
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )

        total_duration = time.time() - start_time
        pipeline_log = {
            "agent_id": 23,
            "agent_name": "concept_diversity_controller",
            "status": "completed",
            "timestamp": time.time(),
            "execution_time_sec": round(total_duration, 2),
            "duration_sec": round(total_duration, 2),
            "reasoning": result.reasoning,
            "output": {
                "diversity_audit_matrix": [row.model_dump() for row in result.diversity_audit_matrix],
                "cluster_pairs": [pair.model_dump() for pair in result.cluster_pairs],
                "mutation_prompts_issued": [item.model_dump() for item in result.mutation_prompts_issued],
                "experimental_concept_flag": result.experimental_concept_flag,
            },
        }

        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": pipeline_log}},
            upsert=True,
        )
        logger.info("Agent 23: Successfully updated IDEATION and PIPELINE collections.")
    except Exception as exc:
        logger.error("Agent 23: Error writing DB updates for project_id=%s error=%s", project_id, exc)
        raise

    total_duration = time.time() - start_time
    logger.info("Agent 23: Execution completed in %.2fs for project_id=%s", total_duration, project_id)
    return result
