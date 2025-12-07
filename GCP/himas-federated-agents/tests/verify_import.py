import sys
import os

# Add current directory to path just in case, though installed package should work
sys.path.append(os.getcwd())

try:
    print("Attempting to import agent_hospital_a.agent...")
    from agent_hospital_a.agent import root_agent
    print("Successfully imported root_agent")
    print(f"Agent Name: {root_agent.name}")
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
