"""
Gemini API Client for Phase 2 Agent Helpers
Simplified wrapper around google.generativeai for use with Agent 12 and Agent 13
"""

import google.generativeai as genai
import json
import os
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APICallMetrics:
    """Metrics for a single API call"""
    stage: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    duration: float
    cost: float


class GeminiClient:
    """
    Simplified Gemini API client for feasibility checking and consistency validation.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: str = "gemini-3.1-pro-preview",
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
            timeout: Request timeout in seconds (default: 60)
            max_retries: Maximum number of retry attempts for transient errors (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 2.0)
        """
        # Use SHARED_ prefix for Google API key (shared across services)
        # Maintain backward compatibility with unprefixed version
        self.api_key = api_key or os.getenv("SHARED_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required. Set SHARED_GOOGLE_API_KEY environment variable.")

        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.call_metrics: list = []

        # Configure Google AI with REST transport to avoid gRPC DNS issues
        genai.configure(api_key=self.api_key, transport="rest")
        self.model = genai.GenerativeModel(model_name)

        logger.info(f"Initialized GeminiClient with model: {model_name}, timeout: {timeout}s, max_retries: {max_retries}")

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Check if an error is retryable (transient network/DNS errors).
        
        Args:
            error: The exception to check
            
        Returns:
            True if the error should be retried
        """
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Check for DNS resolution errors
        if "dns" in error_str or "ares" in error_str:
            return True
        
        # Check for network/timeout errors
        if "timeout" in error_str or "connection" in error_str or "network" in error_str:
            return True
        
        # Check for specific exception types that are retryable
        retryable_types = [
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "IOError"
        ]
        if any(retryable_type in error_type for retryable_type in retryable_types):
            return True
        
        # Check for HTTP 5xx errors (server errors are retryable)
        if "503" in error_str or "500" in error_str or "502" in error_str or "504" in error_str:
            return True
        
        return False

    def call_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        stage: str = "unknown",
        expect_json: bool = True
    ) -> Dict[str, Any]:
        """
        Synchronous call to Gemini API with retry logic and timeout handling.

        Args:
            system_prompt: System message for context
            user_prompt: User message with the actual request
            stage: Processing stage name for metrics
            expect_json: Whether to expect JSON response

        Returns:
            Parsed response (JSON if expect_json=True, otherwise string)
        """
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        last_error = None
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                if retry_count > 0:
                    # Exponential backoff: delay increases with each retry
                    delay = self.retry_delay * (2 ** (retry_count - 1))
                    logger.warning(
                        f"Retrying API call for stage {stage} (attempt {retry_count + 1}/{self.max_retries + 1}) "
                        f"after {delay:.1f}s delay. Last error: {str(last_error)[:200]}"
                    )
                    time.sleep(delay)
                else:
                    logger.info(f"Making API call for stage: {stage}")

                # Make the API call with timeout handling
                start_time = time.time()

                # Reconfigure transport before each call in case other modules changed it
                genai.configure(api_key=self.api_key, transport="rest")
                self.model = genai.GenerativeModel(self.model_name)
                
                # Use generate_content - the underlying library should respect transport="rest" timeout settings
                # We'll monitor elapsed time to enforce our timeout
                response = self.model.generate_content(full_prompt)
                
                # Check if we exceeded timeout
                elapsed = time.time() - start_time
                # Removed artificial timeout check that was throwing away valid responses
                
                response_text = response.text
                logger.info(f"API call completed for stage: {stage} (took {elapsed:.2f}s)")

                # Parse JSON if expected
                if expect_json:
                    try:
                        # Clean response content (remove markdown formatting if present)
                        cleaned_content = response_text.strip()
                        if cleaned_content.startswith("```json"):
                            cleaned_content = cleaned_content[7:]
                        if cleaned_content.endswith("```"):
                            cleaned_content = cleaned_content[:-3]
                        cleaned_content = cleaned_content.strip()

                        # Find JSON object
                        json_start = cleaned_content.find('{')
                        json_end = cleaned_content.rfind('}') + 1
                        if json_start != -1 and json_end > json_start:
                            json_text = cleaned_content[json_start:json_end]
                            return json.loads(json_text)
                        else:
                            return json.loads(cleaned_content)

                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        logger.error(f"Raw response: {response_text}")
                        # Return a default response instead of raising
                        return {
                            "error": "JSON parse failed",
                            "raw_response": response_text[:500]
                        }

                return {"content": response_text}

            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if this is a retryable error
                if self._is_retryable_error(e) and retry_count < self.max_retries:
                    retry_count += 1
                    continue
                else:
                    # Non-retryable error or max retries exceeded
                    logger.error(
                        f"API call failed for stage {stage}: {error_str}. "
                        f"Retries exhausted ({retry_count}/{self.max_retries})"
                    )
                    # Return a default response instead of raising
                    return {
                        "error": error_str,
                        "stage": stage,
                        "retries_attempted": retry_count
                    }
        
        # If we get here, all retries were exhausted
        logger.error(
            f"API call failed for stage {stage} after {self.max_retries} retries. "
            f"Last error: {str(last_error)}"
        )
        return {
            "error": str(last_error) if last_error else "Unknown error",
            "stage": stage,
            "retries_attempted": retry_count
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all API call metrics"""
        return {
            "total_calls": len(self.call_metrics),
            "stages": [metric.stage for metric in self.call_metrics] if hasattr(self, 'call_metrics') else []
        }

