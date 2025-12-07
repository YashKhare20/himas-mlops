# Copyright 2025 HIMAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
HIMAS Hospital A - Root Agent with A2A Support

Coordinates all local agents for Hospital A clinical operations.
Exposed via A2A Protocol using to_a2a().

Run with: uvicorn hospital_a.agent:a2a_app --host 0.0.0.0 --port 8002
Verify:   curl http://localhost:8002/.well-known/agent-card.json
"""

import logging

from google.adk.agents import Agent
from google.genai import types
from google.adk.tools.agent_tool import AgentTool
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from .subagents.clinical_decision import clinical_decision_agent
from .subagents.treatment_optimization import treatment_optimization_agent
from .subagents.case_consultation import case_consultation_agent
from .subagents.resource_allocation import resource_allocation_agent
from .subagents.privacy_guardian import privacy_guardian_agent
from .prompt import HOSPITAL_AGENT_INSTRUCTION
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ============================================================================
# ROOT AGENT DEFINITION
# ============================================================================

root_agent = Agent(
    model=config.MODEL_NAME,
    name=f'{config.HOSPITAL_ID}_root_agent',
    description=f"""
    I am the clinical AI assistant for {config.HOSPITAL_NAME}, accessible via A2A protocol.
    
    I help clinicians with:
    - ICU mortality risk prediction using our federated AI model
    - Treatment plan optimization based on local resources
    - Finding similar cases across our healthcare network (privacy-preserved)
    - ICU bed and resource allocation
    - Patient transfer coordination when specialized care is needed
    
    All operations are HIPAA-compliant with complete audit trails.
    
    A2A Capabilities:
    - Can be invoked by the Federated Coordinator for network-wide queries
    - Participates in privacy-preserved case consultations
    - Reports resource availability to the federated network
    """,
    instruction=HOSPITAL_AGENT_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        top_p=0.95,
        top_k=40,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
    tools=[
        AgentTool(agent=clinical_decision_agent),
        AgentTool(agent=treatment_optimization_agent),
        AgentTool(agent=case_consultation_agent),
        AgentTool(agent=resource_allocation_agent),
        AgentTool(agent=privacy_guardian_agent),
    ],
)


# ============================================================================
# A2A APPLICATION
# ============================================================================

# Create A2A-compatible application
# This auto-generates an agent card from the agent definition
a2a_app = to_a2a(root_agent, port=config.A2A_PORT)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print(f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║  HIMAS {config.HOSPITAL_NAME} - A2A Clinical AI Agent                          ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║                                                                      ║
    ║  Port: {config.A2A_PORT}                                                        ║
    ║  Agent Card: http://localhost:{config.A2A_PORT}/.well-known/agent-card.json    ║
    ║                                                                      ║
    ║  Capabilities:                                                       ║
    ║  • ICU Mortality Risk Prediction (Federated Model)                   ║
    ║  • Treatment Optimization                                            ║
    ║  • Privacy-Preserved Case Consultation                               ║
    ║  • Resource Allocation & Bed Management                              ║
    ║  • Cross-Hospital Transfer Coordination                              ║
    ║                                                                      ║
    ║  Privacy Guarantees:                                                 ║
    ║  • Differential Privacy (ε={config.DIFFERENTIAL_PRIVACY_EPSILON})                                    ║
    ║  • K-Anonymity (k={config.K_ANONYMITY_THRESHOLD})                                             ║
    ║  • HIPAA Compliant                                                   ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(a2a_app, host="0.0.0.0", port=config.A2A_PORT)