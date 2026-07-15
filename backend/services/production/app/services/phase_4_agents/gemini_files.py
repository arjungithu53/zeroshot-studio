import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


def wait_for_gemini_file_active(
    genai_client: Any,
    uploaded_file: Any,
    *,
    timeout_sec: int = 90,
    poll_sec: float = 2.0,
) -> Any:
    """Poll Gemini Files API until an uploaded file can be used in prompts."""
    name = getattr(uploaded_file, "name", None)
    if not name:
        return uploaded_file

    deadline = time.time() + timeout_sec
    last_state = None

    while time.time() < deadline:
        file_obj = genai_client.files.get(name=name)
        state = getattr(file_obj, "state", None)
        state_name = getattr(state, "name", str(state))
        last_state = state_name

        if state_name == "ACTIVE":
            return file_obj
        if state_name == "FAILED":
            raise RuntimeError(f"Gemini file {name} failed processing")

        logger.info(f"Waiting for Gemini file {name} to become ACTIVE (state={state_name})")
        time.sleep(poll_sec)

    raise TimeoutError(f"Gemini file {name} did not become ACTIVE within {timeout_sec}s (last_state={last_state})")
