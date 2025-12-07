"""
Treatment Optimization Agent
Finds optimal treatment plan based on Hospital A's resources.
"""

from google.genai import types
from google.adk import Agent

from .prompt import TREATMENT_OPTIMIZATION_INSTRUCTION
from .tools.bigquery_tools import (
    query_similar_cases,
    query_treatment_protocols,
    query_outcomes_by_admission_type,
    query_outcomes_by_icu_type
)
from .tools.resource_check import (
    check_hospital_resources,
    check_specific_capability,
    get_icu_capacity_by_type
)
from ...config import config

treatment_optimization_agent = Agent(
    model=config.MODEL_NAME,
    name="treatment_optimization_agent",
    description="""
    Optimizes treatment plans based on Hospital A's available resources 
    (queried from BigQuery) and historical case outcomes.
    
    Features:
    - Automatic mapping of user-friendly terms (e.g., "emergency" → "EW EMER.")
    - Historical outcome queries with privacy-preserved statistics
    - Real-time resource availability from BigQuery
    - ICU type-specific outcome analysis
    
    Supported User-Friendly Terms:
    - Admission Types: emergency, urgent, elective, observation, same day surgery
    - ICU Types: cardiac, medical, surgical, neuro, mixed
    """,
    instruction=TREATMENT_OPTIMIZATION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
    tools=[
        # Resource checks
        check_hospital_resources,
        check_specific_capability,
        get_icu_capacity_by_type,
        # Historical case queries
        query_similar_cases,
        query_treatment_protocols,
        query_outcomes_by_admission_type,
        query_outcomes_by_icu_type
    ],
)
