"""
Resource Allocation Agent
Manages ICU beds, equipment, and staff allocation within Hospital A.

Uses data_mappings module for ICU type validation.
"""

from google.genai import types
from google.adk import Agent

from .prompt import RESOURCE_ALLOCATION_INSTRUCTION
from .tools.allocation_tools import (
    check_bed_availability,
    allocate_icu_bed
)
from ...config import config


resource_allocation_agent = Agent(
    model=config.MODEL_NAME,
    name="resource_allocation_agent",
    description="""
    Manages Hospital A's ICU resources including bed allocation, equipment 
    assignment, and staff scheduling based on predicted patient risk.
    
    Features:
    - Real-time bed availability from BigQuery
    - Risk-based prioritization for allocation
    - ICU type validation via data_mappings
    
    Supported ICU Types:
    - Cardiac ICU, Medical ICU, Surgical ICU, Neuro ICU
    - Mixed ICU (2 Units), Mixed ICU (3+ Units), Other ICU
    
    User-Friendly Terms Accepted:
    - "cardiac", "medical", "surgical", "neuro", "mixed"
    """,
    instruction=RESOURCE_ALLOCATION_INSTRUCTION,
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
    tools=[
        check_bed_availability,
        allocate_icu_bed,
    ],
)