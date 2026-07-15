import os
import json
import logging
from datetime import datetime, timezone
from bson import ObjectId
from google import genai
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
PIPELINE_COLLECTION = os.getenv("COLLECTION_PIPELINE", "pipeline")
STRATEGY_COLLECTION = os.getenv("COLLECTION_STRATEGY", "strategy")
PROJECTS_COLLECTION = os.getenv("COLLECTION_PROJECTS", "projects")

logger = logging.getLogger("zeroshot.positioning_alignment")

class Contradiction(BaseModel):
    issue: str = Field(description="Description of the contradiction")
    agents_affected: List[str] = Field(description="Names of the agents affected by this contradiction")
    resolution: str = Field(description="Specific resolution suggestion")

class PositioningAlignmentResult(BaseModel):
    alignment_score: int = Field(description="An alignment score from 0 to 100")
    is_aligned: bool = Field(description="Whether the strategy is overall aligned")
    contradictions: List[Contradiction] = Field(description="Any contradictions found")
    positioning_statement: str = Field(description="The formal positioning statement")

def clean_json_string(raw_str: str) -> str:
    """Helper to strip markdown fences from Gemini JSON response."""
    cleaned = raw_str.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

async def run_positioning_alignment_agent(project_id: str, db):
    agent_key = "positioning_alignment"
    timestamp = datetime.now(timezone.utc)
    
    logger.info(f"[{agent_key}] Starting agent. project_id={project_id}")
    
    try:
        # 1. Fetch data from DB
        logger.info(f"[{agent_key}] Fetching strategy document.")
        strategy_doc = await db[STRATEGY_COLLECTION].find_one({"project_id": project_id})
        if not strategy_doc:
            raise ValueError(f"Strategy document for project {project_id} not found.")
            
        strategy_id = strategy_doc["_id"]

        logger.info(f"[{agent_key}] Fetching project document.")
        project_doc = await db[PROJECTS_COLLECTION].find_one({"_id": ObjectId(project_id)})
        if not project_doc:
            raise ValueError(f"Project document {project_id} not found.")

        # Extract fields
        company_research = strategy_doc.get("company_research", {})
        raw_text = company_research.get("raw_text", "")
        
        agents_data = strategy_doc.get("agents", {})
        
        # 2. Extract brand guidelines and url
        brand_guidelines = project_doc.get("brand_guidelines", "")
        company_url = project_doc.get("company_url", "")

        # Target audience
        target_audience = project_doc.get("target_audience", "")
        
        # We need ALL 10 agent outputs
        # Usually, missing ones should be checked, but for now we take what's in agents_data
        
        logger.info(f"[{agent_key}] Successfully loaded all required inputs.")

        # 3. Call Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")

        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"}
        )
        
        prompt = f"""[CONTEXT AND SOURCE MATERIAL]
Company URL:
{company_url}

Target Audience (from user):
{target_audience}

Brand Guidelines (from user):
{brand_guidelines}

Raw Company Research:
{raw_text}

All 10 Previous Strategy Agent Outputs:
{json.dumps(agents_data, indent=2)}

[MAIN TASK INSTRUCTIONS]
You are the Strategy Alignment Director. Your objective is to review the complete strategy built across all previous agents for strict internal consistency, brand alignment, and market fit. 
Analyze the entire context provided above, using logical deduction to evaluate how well the different strategic components integrate.

[CONSTRAINTS AND RULES]
1. Product Context & Permitted Newness: The product for this campaign could be a new product launch, a rebrand, or a brand refresh. You must extract the product's details strictly from the "All 10 Previous Strategy Agent Outputs".
   -> CRITICAL EXCEPTION: Do NOT flag the new product's specific features, ingredients, flavors, or name as a contradiction simply because they do not appear in the "Raw Company Research". A new product naturally introduces new elements. Assume the product details defined in the agent outputs are factual for this specific campaign.
2. Hierarchy of Truth (Conflict Resolution): If you detect a conflict between the core inputs (Brand Guidelines, core product information, or Target Audience) and an Agent's output, the core inputs take absolute precedence. You must prioritize the brand and product information over the Agent's output. Flag the Agent's output as incorrect, and ensure your resolution suggestion forces the agent to align back to the core brand/product rules.
3. Internal Consistency Checks: You must explicitly verify the following relationships within the proposed strategy:
   - Does the campaign platform contradict the brand adjective?
   - Does the persona conflict with the target_audience input?
   - Does the value prop align logically with the offer hook?
4. Market Fit Check: Evaluate whether the positioning accurately occupies the identified whitespace from Agent 3 (Competitive Landscape).
5. Brand Compliance Check: Rigorously evaluate the entire strategy against the provided brand_guidelines. (Note: Only flag new product features if they explicitly violate fundamental rules in the Brand Guidelines, not just because they are absent from the historical Raw Company Research).
6. Rigor and Specificity: Be highly rigorous and specific in identifying actual contradictions. Do not provide generalized feedback.[OUTPUT FORMAT]
Based on your analysis, output your response including:
- An alignment score from 0 to 100.
- Any contradictions flagged (be specific).
- Specific resolution suggestions for each flagged contradiction.
[OUTPUT FORMAT]
Based on your analysis, output your response including:
- An alignment score from 0 to 100.
- Any contradictions flagged (be specific).
- Specific resolution suggestions for each flagged contradiction.
"""

        logger.info(f"[{agent_key}] Calling Gemini model {GEMINI_MODEL}")
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "tools": [
                    {"google_search": {}},
                    {"url_context": {}},
                ],
                "response_mime_type": "application/json",
                "response_json_schema": PositioningAlignmentResult.model_json_schema(),
                "temperature": 0.2
            }
        )

        response_text = clean_json_string(response.text)
        logger.info(f"[{agent_key}] Received response from Gemini.")

        # Parse JSON to ensure it matches
        try:
            parsed_data = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON output: {e}")
            raise ValueError("Agent failed to return valid JSON.")

        # Validate with pydantic
        final_result = PositioningAlignmentResult(**parsed_data)

        # 4. Save to DB
        logger.info(f"[{agent_key}] Updating strategy DB.")
        await db[STRATEGY_COLLECTION].update_one(
            {"_id": strategy_id},
            {"$set": {
                f"agents.{agent_key}": final_result.model_dump(),
                "updated_at": timestamp
            }}
        )

        # Log to pipeline
        log_entry = {
            "agent_key": agent_key,
            "status": "completed",
            "timestamp": timestamp,
            "reasoning": f"Alignment score: {final_result.alignment_score}, Contradictions found: {len(final_result.contradictions)}"
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": log_entry}},
            upsert=True
        )

        logger.info(f"[{agent_key}] Completed successfully.")
        return final_result

    except Exception as e:
        logger.error(f"[{agent_key}] Failed: {str(e)}")
        error_log = {
            "agent_key": agent_key,
            "status": "failed",
            "timestamp": timestamp,
            "error_msg": str(e)
        }
        await db[PIPELINE_COLLECTION].update_one(
            {"project_id": str(project_id)},
            {"$push": {"agent_logs": error_log}},
            upsert=True
        )
        raise e
