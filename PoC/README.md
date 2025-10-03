# HIMAS Federated Learning - Quick Start

Privacy-preserving medical ML where hospitals collaborate without sharing patient data.

## What You Need

**Docker Method:**
- Docker & Docker Compose

**Local Method:**
- Python 3.8+
- `pip install flwr==1.11.0 scikit-learn numpy pandas`

## Run It

### Docker (Real Distributed System)

docker-compose up --build

### Local Testing

python main_himas_federated.py

### What Happens 
1. Dataset split into 3 parts (one per hospital)
2. Each hospital trains locally on their data
3. Only model weights shared with server (31 numbers)
4. Server averages weights → global model
5. Repeat for 3 rounds

### Key Files

docker-compose.yml - Runs 4 containers
docker_client.py - Hospital client code
flower_server.py - Central coordinator

#### Privacy
🔒 Patient data never leaves the hospital - only 31 model weights are shared.