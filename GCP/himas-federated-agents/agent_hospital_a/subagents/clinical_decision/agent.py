"""
Clinical Decision Support Agent
Predicts ICU mortality risk using global federated model.

Uses data_mappings module for standardizing input parameters.
"""

from google.genai import types
from google.adk import Agent

from ...config import config
from .prompt import CLINICAL_DECISION_INSTRUCTION
from .tools.prediction import predict_mortality_risk


clinical_decision_agent = Agent(
    model=config.MODEL_NAME,
    name="clinical_decision_agent",
    description="""
    Predicts ICU mortality risk for newly admitted patients using the global 
    federated learning model. Uses only admission-time features (available 
    when patient enters ICU).
    
    Data Mapping Features:
    - Accepts user-friendly admission types (e.g., "emergency" → "EW EMER.")
    - Validates and normalizes all input parameters
    - Early ICU score auto-validated to 0-3 range
    - Gender normalized to M/F
    
    Supported User-Friendly Terms:
    - Admission: emergency, urgent, elective, observation, same day
    - ICU Type: cardiac, medical, surgical, neuro, mixed
    """,
    instruction=CLINICAL_DECISION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
    tools=[predict_mortality_risk],
)