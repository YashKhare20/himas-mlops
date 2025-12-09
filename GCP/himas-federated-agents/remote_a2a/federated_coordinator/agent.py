"""
HIMAS Federated Coordinator - Remote A2A Agent

Central coordinator for the federated healthcare network.
Exposed via A2A Protocol using to_a2a().

Run with: uvicorn remote_a2a.federated_coordinator.agent:a2a_app --host 0.0.0.0 --port 8001
Verify:   curl http://localhost:8001/.well-known/agent-card.json
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import logging

from google.adk.agents import Agent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.genai import types

from .prompt import COORDINATOR_INSTRUCTION
from .tools.capability_tools import query_hospital_capabilities
from .tools.consultation_tools import query_similar_cases
from .tools.transfer_tools import initiate_transfer, get_transfer_status
from .tools.statistics_tools import get_network_statistics
from .tools.privacy_tools import anonymize_patient_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8001"))
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")


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
# A2A APPLICATION
# ============================================================================

# Create A2A-compatible application
# This auto-generates an agent card from the agent definition
a2a_app = to_a2a(root_agent, port=COORDINATOR_PORT)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"""
    ═══════════════════════════════════════════════════════════════════
    HIMAS Federated Coordinator - A2A Server
    ═══════════════════════════════════════════════════════════════════
    
    Port: {COORDINATOR_PORT}
    Agent Card: http://localhost:{COORDINATOR_PORT}/.well-known/agent-card.json
    
    ═══════════════════════════════════════════════════════════════════
    """)

    uvicorn.run(a2a_app, host="0.0.0.0", port=COORDINATOR_PORT)
