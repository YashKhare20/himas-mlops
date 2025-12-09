# HIMAS Federated Agents - Recommended Structure

## Current vs Recommended Structure

### Option 1: Separate Hospital Folders (Simple, some duplication)

```
himas-federated-agents/
├── remote_a2a/                              # 🆕 Remote A2A agents
│   └── federated_coordinator/
│       ├── __init__.py
│       ├── agent.py                         # Exposed via to_a2a()
│       ├── agent.json                       # Agent card for adk api_server --a2a
│       ├── prompt.py
│       └── tools/
│           ├── capability_tools.py
│           ├── consultation_tools.py
│           ├── transfer_tools.py
│           └── statistics_tools.py
│
├── agent_hospital_a/                        # ✅ Already exists
│   ├── __init__.py
│   ├── agent.py                             # Uses RemoteA2aAgent for coordinator
│   ├── config.py                            # HOSPITAL_ID = "hospital_a"
│   ├── prompt.py
│   ├── subagents/
│   │   ├── case_consultation/
│   │   ├── clinical_decision/
│   │   ├── privacy_guardian/
│   │   ├── resource_allocation/
│   │   └── treatment_optimization/
│   └── utils/
│
├── agent_hospital_b/                        # 🆕 Copy structure from A
│   ├── __init__.py
│   ├── agent.py                             # Uses RemoteA2aAgent for coordinator
│   ├── config.py                            # HOSPITAL_ID = "hospital_b"
│   ├── prompt.py
│   ├── subagents/
│   │   ├── case_consultation/
│   │   ├── clinical_decision/
│   │   ├── privacy_guardian/
│   │   ├── resource_allocation/
│   │   └── treatment_optimization/
│   └── utils/
│
├── agent_hospital_c/                        # 🆕 Copy structure from A
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py                            # HOSPITAL_ID = "hospital_c"
│   ├── prompt.py
│   ├── subagents/
│   │   └── ... (same as A and B)
│   └── utils/
│
├── shared/                                  # 🆕 Shared utilities
│   ├── __init__.py
│   ├── bigquery_client.py
│   ├── model_loader.py
│   └── constants.py
│
├── deployment/
├── eval/
├── tests/
├── pyproject.toml
└── README.md
```

### Option 2: Shared Subagents with Hospital Config (DRY, recommended)

```
himas-federated-agents/
├── remote_a2a/                              # Remote A2A agents (served separately)
│   └── federated_coordinator/
│       ├── __init__.py
│       ├── agent.py                         # to_a2a(root_agent, port=8001)
│       ├── agent.json
│       ├── prompt.py
│       └── tools/
│           ├── __init__.py
│           ├── capability_tools.py
│           ├── consultation_tools.py
│           ├── transfer_tools.py
│           └── statistics_tools.py
│
├── hospital_agents/                         # All hospital agents share structure
│   ├── __init__.py
│   ├── shared/                              # Shared subagent implementations
│   │   ├── __init__.py
│   │   ├── subagents/
│   │   │   ├── __init__.py
│   │   │   ├── case_consultation/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py               # Takes hospital_config as parameter
│   │   │   │   ├── prompt.py
│   │   │   │   └── tools/
│   │   │   ├── clinical_decision/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── agent.py
│   │   │   │   ├── prompt.py
│   │   │   │   └── tools/
│   │   │   ├── privacy_guardian/
│   │   │   ├── resource_allocation/
│   │   │   └── treatment_optimization/
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── bigquery_client.py
│   │       ├── data_preprocessor.py
│   │       └── feature_extractor.py
│   │
│   ├── hospital_a/                          # Hospital A entry point
│   │   ├── __init__.py
│   │   ├── agent.py                         # Imports shared subagents + config
│   │   ├── config.py                        # Hospital A specific config
│   │   └── prompt.py                        # Hospital A specific prompts (optional)
│   │
│   ├── hospital_b/                          # Hospital B entry point
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── config.py                        # Hospital B specific config (Tertiary)
│   │   └── prompt.py
│   │
│   └── hospital_c/                          # Hospital C entry point
│       ├── __init__.py
│       ├── agent.py
│       ├── config.py                        # Hospital C specific config (Rural)
│       └── prompt.py
│
├── deployment/
├── eval/
├── tests/
├── pyproject.toml
└── README.md
```

## Running the System

### Step 1: Start Federated Coordinator (Required First)

```bash
# Terminal 1: Start the coordinator on port 8001
uvicorn remote_a2a.federated_coordinator.agent:a2a_app --host 0.0.0.0 --port 8001

# Verify it's running
curl http://localhost:8001/.well-known/agent-card.json
```

### Step 2: Start Hospital Agents (Choose One)

```bash
# Terminal 2: Start Hospital A
cd hospital_agents/hospital_a
adk web .

# OR Terminal 3: Start Hospital B (different port)
cd hospital_agents/hospital_b
adk web . --port 8002

# OR Terminal 4: Start Hospital C
cd hospital_agents/hospital_c
adk web . --port 8003
```

### Alternative: Run All via adk api_server

```bash
# Start all hospital agents from parent folder
adk api_server hospital_agents --port 8000

# Access via:
# http://localhost:8000/hospital_a
# http://localhost:8000/hospital_b
# http://localhost:8000/hospital_c
```

## Key Files to Create/Modify

### 1. remote_a2a/federated_coordinator/agent.py

```python
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.llm_agent import Agent

root_agent = Agent(
    model="gemini-2.0-flash",
    name="federated_coordinator",
    description="Cross-hospital coordination with privacy guarantees",
    tools=[...],
)

a2a_app = to_a2a(root_agent, port=8001)
```

### 2. hospital_agents/hospital_a/agent.py

```python
from google.adk.agents.llm_agent import Agent
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent, AGENT_CARD_WELL_KNOWN_PATH

from .config import HOSPITAL_CONFIG
from ..shared.subagents.clinical_decision import create_clinical_agent
from ..shared.subagents.resource_allocation import create_resource_agent

# Remote coordinator via A2A
federated_coordinator = RemoteA2aAgent(
    name="federated_coordinator",
    description="Cross-hospital queries and transfer coordination",
    agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
)

# Local subagents (configured for this hospital)
clinical_agent = create_clinical_agent(HOSPITAL_CONFIG)
resource_agent = create_resource_agent(HOSPITAL_CONFIG)

root_agent = Agent(
    model="gemini-2.0-flash",
    name=f"hospital_{HOSPITAL_CONFIG['id']}_agent",
    sub_agents=[clinical_agent, resource_agent, federated_coordinator],
)
```

### 3. hospital_agents/hospital_a/config.py

```python
HOSPITAL_CONFIG = {
    "id": "hospital_a",
    "name": "Community Hospital A",
    "tier": "Community Hospital",
    "capabilities": {
        "advanced_cardiac_care": False,
        "ecmo": False,
        "cardiac_surgery": True,
    },
    "coordinator_url": "http://localhost:8001",
}
```

### 4. hospital_agents/hospital_b/config.py

```python
HOSPITAL_CONFIG = {
    "id": "hospital_b",
    "name": "Tertiary Medical Center B",
    "tier": "Tertiary Care Center",
    "capabilities": {
        "advanced_cardiac_care": True,
        "ecmo": True,
        "cardiac_surgery": True,
        "infectious_disease": True,
    },
    "coordinator_url": "http://localhost:8001",
}
```

### 5. hospital_agents/hospital_c/config.py

```python
HOSPITAL_CONFIG = {
    "id": "hospital_c",
    "name": "Rural Hospital C",
    "tier": "Rural Hospital",
    "capabilities": {
        "advanced_cardiac_care": False,
        "ecmo": False,
        "cardiac_surgery": False,
    },
    "coordinator_url": "http://localhost:8001",
}
```