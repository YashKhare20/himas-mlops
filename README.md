# HIMAS - Healthcare Intelligence Multi-Agent System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flower Framework](https://img.shields.io/badge/Flower-1.5+-green.svg)](https://flower.ai/)
[![GCP](https://img.shields.io/badge/GCP-Ready-orange.svg)](https://cloud.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🏥 Overview

HIMAS (Healthcare Intelligence Multi-Agent System) is a privacy-preserving federated learning platform that enables healthcare institutions to collaboratively train machine learning models without sharing patient data. The project demonstrates how hospitals can leverage collective medical intelligence while maintaining strict HIPAA compliance and patient privacy.

### Key Features

- **Federated Learning**: Train global models across multiple hospitals without data sharing
- **Privacy-Preserving**: Differential privacy (ε=1.0) and secure aggregation
- **Multi-Agent System**: 5 specialized healthcare agents for disease surveillance, treatment optimization, and clinical decision support
- **Production-Ready**: Full deployment on Google Kubernetes Engine (GKE)
- **MIMIC-IV Integration**: Uses real ICU data (de-identified) for realistic healthcare scenarios

## 📋 Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Phase 1: Proof of Concept](#phase-1-proof-of-concept)
- [Phase 2: Production Deployment](#phase-2-production-deployment)
- [Agents](#agents)
- [Data Privacy](#data-privacy)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## 🏗 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         GKE Cluster                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │SuperLink │←→│SuperNode1│  │SuperNode2│  │SuperNode3│   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│        ↓                                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Global Model (Vertex AI Endpoint)          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Google ADK Agents                         │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │Orchestrator│ │Surveillance│ │ Treatment  │              │
│  └────────────┘ └────────────┘ └────────────┘              │
│  ┌────────────┐ ┌────────────┐                              │
│  │  Clinical  │ │  Privacy   │                              │
│  └────────────┘ └────────────┘                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌──────────────────┐
                    │  Healthcare UI    │
                    │  (Cloud Run)      │
                    └──────────────────┘
```

## ⚙️ Prerequisites

### Local Development (PoC)
- Python 3.9+
- Docker & Docker Compose
- 8GB RAM minimum
- Git

### Production Deployment
- Google Cloud Platform account
- GCP Credits ($300 budget)
- `gcloud` CLI installed
- `kubectl` configured
- PhysioNet credentialed access (for MIMIC-IV data)

## 🚀 Quick Start

### Clone the Repository

```bash
git clone https://github.com/[username]/himas-mlops.git
cd himas-mlops
```

### Run PoC (Local)

```bash
# Install dependencies
pip install -r requirements-poc.txt

# Navigate to PoC directory
cd poc

# Start federated learning with Docker Compose
docker-compose up --build

# Access MLflow UI
open http://localhost:5000
```

### Deploy to Production (GCP)

```bash
# Set up GCP environment
export PROJECT_ID="your-project-id"
gcloud config set project $PROJECT_ID

# Deploy to GKE
cd production/kubernetes
./k8s-deploy.sh
```

## 📁 Project Structure

```
himas-mlops/
├── README.md                        # This file
├── requirements-poc.txt             # PoC dependencies
├── requirements-prod.txt            # Production dependencies
├── .env.example                     # Environment variables template
├── LICENSE                          # MIT License
│
├── poc/                            # Proof of Concept (2 weeks)
│   ├── data/                       
│   │   ├── sample_hospital_a.csv  # Sample data (1K records)
│   │   ├── sample_hospital_b.csv
│   │   └── sample_hospital_c.csv
│   ├── federated/                  
│   │   ├── server/                
│   │   │   ├── app.py             # Flower server implementation
│   │   │   └── Dockerfile
│   │   ├── client/                
│   │   │   ├── app.py             # Hospital client logic
│   │   │   └── Dockerfile
│   │   └── strategies/            
│   │       └── fedavg_dp.py       # FedAvg with differential privacy
│   ├── models/                    
│   │   ├── mortality_predictor.py # Clinical ML model
│   │   └── utils.py               # Model utilities
│   └── docker-compose.yml         # Local orchestration
│
├── production/                     # Production deployment
│   ├── kubernetes/                
│   │   ├── namespace.yaml         # K8s namespace
│   │   ├── superlink.yaml         # Flower SuperLink config
│   │   ├── supernode.yaml         # Flower SuperNode config
│   │   ├── superexec.yaml         # Flower SuperExec config
│   │   ├── services.yaml          # K8s services
│   │   └── k8s-deploy.sh          # Deployment script
│   ├── docker/                    
│   │   ├── superexec.Dockerfile   # Custom SuperExec image
│   │   └── build-push.sh          # Build and push to registry
│   ├── agents/                    # Healthcare agents
│   │   ├── orchestrator/          
│   │   ├── surveillance/          
│   │   ├── treatment/             
│   │   ├── clinical/              
│   │   └── privacy/               
│   ├── infrastructure/            
│   │   ├── terraform/             # IaC for GKE
│   │   ├── artifact-registry/    # Docker registry setup
│   │   └── bigquery/              # Data warehouse schemas
│   └── ui/                        
│       ├── streamlit/             # Healthcare dashboard
│       └── api/                   # FastAPI backend
│
└── tests/                          # Test suite
    ├── unit/                       
    ├── integration/                
    └── e2e/                        
```

## 🧪 Phase 1: Proof of Concept

### Overview
The PoC demonstrates federated learning across 3 simulated hospitals using Docker containers. No agents are deployed in this phase - focus is purely on the MLOps pipeline.

### Running the PoC

1. **Prepare Sample Data**
```bash
cd poc/data
python generate_samples.py  # Creates sample CSV files from MIMIC-IV demo
```

2. **Start Federated Learning**
```bash
cd poc
docker-compose up --build
```

3. **Monitor Training**
```bash
# View logs
docker-compose logs -f flower-server

# Access MLflow
open http://localhost:5000
```

4. **Evaluate Results**
```bash
python evaluate_global_model.py
```

### PoC Components

- **Flower Server**: Coordinates federated learning rounds
- **Hospital Clients**: 3 Docker containers simulating hospitals
- **MLflow**: Tracks experiments and model metrics
- **FedAvg Strategy**: Aggregates model updates with differential privacy

## 🏭 Phase 2: Production Deployment

### GCP Setup

1. **Create GCP Project**
```bash
gcloud projects create himas-mlops-prod
gcloud config set project himas-mlops-prod
```

2. **Enable Required APIs**
```bash
gcloud services enable \
    container.googleapis.com \
    artifactregistry.googleapis.com \
    bigquery.googleapis.com \
    aiplatform.googleapis.com
```

3. **Create GKE Cluster**
```bash
gcloud container clusters create-auto himas-cluster \
    --region=us-central1 \
    --project=himas-mlops-prod
```

4. **Set Up Artifact Registry**
```bash
gcloud artifacts repositories create himas-docker \
    --repository-format=docker \
    --location=us-central1

gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Deploy Flower Framework

1. **Build and Push Docker Images**
```bash
cd production/docker
./build-push.sh
```

2. **Deploy to GKE**
```bash
cd production/kubernetes
kubectl apply -f namespace.yaml
kubectl apply -f superlink.yaml
kubectl apply -f supernode.yaml
kubectl apply -f superexec.yaml
kubectl apply -f services.yaml
```

3. **Verify Deployment**
```bash
kubectl get pods -n flower-system
kubectl get services -n flower-system
```

### Configure BigQuery

```bash
cd production/infrastructure/bigquery
bq mk --dataset himas_hospital_a
bq mk --dataset himas_hospital_b
bq mk --dataset himas_hospital_c

# Load MIMIC-IV data
bq load --source_format=CSV \
    himas_hospital_a.patients \
    gs://your-bucket/hospital_a_data.csv \
    schema.json
```

### Deploy Agents

```bash
cd production/agents
gcloud run deploy orchestrator --source orchestrator/
gcloud run deploy surveillance --source surveillance/
gcloud run deploy treatment --source treatment/
gcloud run deploy clinical --source clinical/
gcloud run deploy privacy --source privacy/
```

### Launch UI

```bash
cd production/ui/streamlit
gcloud run deploy himas-ui --source . --port 8501
```

## 🤖 Agents

### 1. Master Orchestrator Agent
- Coordinates federated learning workflows
- Manages cross-institutional communication
- Controls model synchronization

### 2. Disease Surveillance Agent
- Monitors epidemiological trends
- Detects outbreak patterns using differential privacy
- Provides early warning systems

### 3. Treatment Optimization Agent
- Analyzes treatment effectiveness
- Recommends optimal therapies
- Leverages collective medical experience

### 4. Clinical Decision Support Agent
- Provides diagnostic recommendations
- Integrates evidence-based guidelines
- RAG-enhanced responses using medical literature

### 5. Privacy Guardian Agent
- HIPAA compliance monitoring
- Differential privacy validation
- Real-time audit logging

## 🔒 Data Privacy

### Privacy Mechanisms

1. **Differential Privacy**: ε=1.0 noise injection during aggregation
2. **Secure Aggregation**: Encrypted model updates
3. **Data Isolation**: Raw data never leaves hospital premises
4. **HIPAA Compliance**: Full audit trails and de-identification

### Privacy Configuration

```python
# production/federated/strategies/fedavg_dp.py
privacy_config = {
    'epsilon': 1.0,
    'delta': 1e-5,
    'clip_norm': 1.0,
    'noise_multiplier': 0.5
}
```

## 📊 Monitoring

### MLflow Tracking
```bash
# Access production MLflow
kubectl port-forward -n flower-system svc/mlflow 5000:5000
open http://localhost:5000
```

### GKE Monitoring
```bash
# View pod metrics
kubectl top pods -n flower-system

# Check logs
kubectl logs -n flower-system -l app=superlink
```

### Cloud Operations
- Metrics: CPU, memory, network usage
- Logs: Centralized in Cloud Logging
- Alerts: Configured for critical events

## 🧑‍💻 Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# End-to-end tests
pytest tests/e2e/
```

### Code Quality

```bash
# Format code
black .

# Lint
flake8 .

# Type checking
mypy .
```

## 📝 Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# GCP Configuration
PROJECT_ID=your-project-id
REGION=us-central1
CLUSTER_NAME=himas-cluster

# Flower Configuration
FLOWER_SERVER_ADDRESS=localhost:8080
NUM_ROUNDS=10
MIN_CLIENTS=3

# Privacy Settings
DIFFERENTIAL_PRIVACY_EPSILON=1.0
DIFFERENTIAL_PRIVACY_DELTA=1e-5

# Model Configuration
MODEL_TYPE=xgboost
BATCH_SIZE=32
LEARNING_RATE=0.001
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MIMIC-IV**: PhysioNet for providing the clinical database
- **Flower Framework**: For excellent federated learning tools
- **Google Cloud**: For infrastructure and ADK platform
- **Course Instructor**: MLOps Fall 2025

## 🔗 Resources

- [Flower Documentation](https://flower.ai/docs/)
- [GKE Guide](https://cloud.google.com/kubernetes-engine/docs)
- [MIMIC-IV Access](https://physionet.org/content/mimic-iv-fhir/)
- [Google ADK](https://cloud.google.com/agent-development-kit)
- [Differential Privacy](https://developers.googleblog.com/2019/09/enabling-developers-and-organizations.html)

---

**Course**: MLOps | **Term**: Fall 2025 | **Budget**: $300 GCP Credits