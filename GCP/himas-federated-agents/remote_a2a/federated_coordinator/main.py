"""
HIMAS Federated Coordinator - A2A Cloud Run Entry Point
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types

# ABSOLUTE IMPORTS (no dots) - required for Cloud Run
from prompt import COORDINATOR_INSTRUCTION
from tools.capability_tools import query_hospital_capabilities
from tools.consultation_tools import query_similar_cases
from tools.transfer_tools import initiate_transfer, get_transfer_status
from tools.statistics_tools import get_network_statistics
from tools.privacy_tools import anonymize_patient_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load environment variables
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# Configuration - Cloud Run uses PORT env var (default 8080)
PORT = int(os.getenv("COORDINATOR_PORT", "8080"))

# ============================================================================
# ROOT AGENT DEFINITION
# ============================================================================

root_agent = Agent(
    model="gemini-2.5-flash",
    name="federated_coordinator",
    description=(
        "HIMAS Federated Coordinator for cross-hospital queries. "
        "Handles capability matching, privacy-preserved case consultation, "
        "patient transfer coordination, and network statistics. "
        "All queries implement k-anonymity and differential privacy."
    ),
    instruction=COORDINATOR_INSTRUCTION,
    tools=[
        query_hospital_capabilities,
        query_similar_cases,
        initiate_transfer,
        get_transfer_status,
        get_network_statistics,
        anonymize_patient_data,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)

# ============================================================================
# A2A APPLICATION - Cloud Run expects 'app' variable
# ============================================================================

app = to_a2a(root_agent, port=PORT)

logger.info(f"Federated Coordinator A2A app initialized on port {PORT}")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)