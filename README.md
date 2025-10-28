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


# Data Bias Detection

## 1. Detecting Bias in Your Data

### 1.1 Dataset Overview

**Data Source:** MIMIC-IV Demo Dataset   
**Total Admissions:** 275 Deaths: 15 (5.5% mortality)  
**Survivors:** 260 (94.5%)   
**Class Imbalance:** 17:1 ratio   
**Task:** Binary classification - predict in-hospital mortality   
**Split:** 60% train (165), 20% validation (55), 20% test (55)  


![DatasetSummary](assets/DatasetSummary.png)


### 1.2 Demographic Features Identified
**Gender:** Male, Female   
**Age Group:** <40, 40-60, 60-75, 75+   
**Insurance:** Medicare, Medicaid, Other   
**Ethnicity:** White, Black, Hispanic, Other  

These features represent protected attributes and known sources of healthcare disparities.  

### 1.3 Baseline Model Performance
Model: Random Forest (100 trees, depth 5, balanced class weights) 

Overall Results:   

Accuracy: 89.1%   
Recall: 33.3% (caught 1 out of 3 deaths)   
Precision: 20.0%   
False Negatives: 2 (missed 2 deaths - CRITICAL)   
True Positives: 1    


![BaseLineModel](assets/BaselineModel.png)


### 1.4 Bias Detection Results

**GENDER BIAS (CRITICAL):**   
 Female: 0% recall (caught 0 deaths)  
 Male: 50% recall (caught 1/2 deaths)   
 Recall Gap: 50 percentage points

Impact: Model completely fails to predict female deaths.  

![ByGender](assets/Bygender.png)



**AGE BIAS (HIGH):**  
<40: 0% recall   
40-60: 0% recall   
60-75: 50% recall (ONLY functional group)     
75+: 0% recall  

Impact: 75% of age groups experience complete failure.

![ByAge](assets/ByAgeGroup.png)



**INSURANCE BIAS (HIGH):**  
 Medicaid: 33% accuracy, 0% recall   
 Medicare: 90% accuracy, 50% recall   
 Other: 94% accuracy, 0% recall  

 Impact: Low-income (Medicaid) patients worst served.

![ByInsurance](assets/ByInsurance.png)

### 1.5 Root Cause

Representation Bias:   
Training data: 1 female deaths vs 8 male deaths (1:8 ratio)   
Consequence: Model learned "males die 8× more often" 

Result: Systematic failure to recognize female mortality patterns

![ByInsurance](assets/DeathsByGender.png)


## 2. DATA SLICING USING FAIRLEARN

### **2.1 Tool Implementation**  

Tool: Fairlearn 0.13.0 (Microsoft Research)   
Method: MetricFrame class for automated demographic slicing  

**Why Fairlearn:**  
- Industry-standard fairness metrics  
- Seamless scikit-learn integration  
- FDA/regulatory alignment  
- Automated multi-dimensional slicing 

Code Efficiency: Reduced 50+ lines of manual code to 8 lines.  

### **2.2 Fairlearn Outputs**  

Gender Analysis Output:  

![ByGender](assets/Bygender.png)

![ByPerformance](assets/PerformanceRatios.png)


Differences (max-min):  
recall: 0.500 (MAXIMUM BIAS)  

Ratios (min/max):   
recall: 0.000 (COMPLETE FAILURE)  


### **2.3 Fairness Metrics**

Demographic Parity Difference:   
Result: 0.112   
Threshold: 0.1     
Status: BIAS DETECTED (exceeds threshold)   
Meaning: Males predicted to die 11.2% more often

Equalized Odds Difference:   
Result: 0.500   
Threshold: 0.1   
Status: SEVERE BIAS (5× threshold)      
Meaning: 50% difference in error rates between genders 


Automated Alert: "BIAS DETECTED" warnings triggered for both metrics


![FairMetrics](assets/FairMetrics.png)


### **2.4 Multi-Dimensional Slicing**

Intersectional Analysis: 

Gender (2) × Age (4) × Insurance (3) = 24 demographic slices analyzed  


Findings:   
Most disadvantaged: Female + Medicaid + <40 (0% accuracy)     
Least disadvantaged: Male + Medicare + 60-75 (90% accuracy)  
Gap: 90 percentage points    


Conclusion: Biases multiply at intersections, exceeding single-dimension effects.


## 3. MITIGATION OF BIAS

### **3.1 Techniques Applied**

Technique 1: SMOTE (Pre-processing)   
Purpose: Address class imbalance 

Technique 2: ThresholdOptimizer (Post-processing)   
Purpose: Enforce fairness constraints  

### **3.2 SMOTE Implementation**

Problem: 9 deaths vs 156 survivors (17:1 imbalance)  

Solution:   
Before: 165 samples (9 deaths, 156 survivors)   
After: 312 samples (156 deaths, 156 survivors)   
Created: 147 synthetic death samples

![SmotebeforeAfter](assets/BeforeAfterSmote.png)



Fairness Impact:   
Demographic Parity: 0.112 → 0.261 (WORSENED)   
Equalized Odds: 0.500 → 1.000 (WORSENED)   
Female Recall: 0% → 0% (NO CHANGE)

![ResultComparison](assets/smoteResultComparison.png)


Why SMOTE Failed on Fairness:  
Balanced overall classes but preserved gender ratio within death class.

### **3.3 ThresholdOptimizer Implementation**

Method: Adjusted decision thresholds per demographic group   
Constraint: Equalized Odds  
Optimization: On validation set (55 samples, 3 deaths)


Fairness Impact:   
Demographic Parity: 0.112 → 0.036 (IMPROVED)   
Equalized Odds: 0.500 → 0.037 (IMPROVED)  

Why ThresholdOptimizer Failed:   
Validation set too small (only 3 deaths), optimizer set conservative thresholds, predicted "everyone survives" (0% recall).


![ThresholdOptimizer](assets/ThresholdOptimizerResult.png)



### **3.4 Three-Way Comparison**

![3 way Comparison](assets/3WayComparison.png)

Selected Model: SMOTE (highest recall, clinically functional)


## 4. DOCUMENT BIAS MITIGATION PROCESS

### **4.1 Complete Process**  

Data Preparation: 275 admissions, identified 4 sensitive attributes, train/val/test split  
Baseline Training: Random Forest, 89.1% acc, 33.3% recall  
Bias Detection: Fairlearn identified gender (50% gap), age, insurance biases  
Root Cause: Only 2 female deaths in training (1:5 ratio)  
SMOTE Mitigation: Recall doubled (33% to 67%), fairness worsened  
ThresholdOpt Mitigation: Perfect fairness, 0% recall (unusable)  
Model Selection: SMOTE chosen (best recall)  

### **4.2 Types of Bias Found**

Representation Bias (CRITICAL): 
Evidence: 1 female vs 8 male deaths (1:8 ratio)   
Mitigation: SMOTE (partially) 
Status: Not resolved  

Gender Bias (CRITICAL): 
Evidence: 50% recall gap, 0.500 equalized odds 
Mitigation: SMOTE and ThresholdOpt attempted 
Status: Not resolved


Age Bias (HIGH): 
Evidence: Only 60-75 group functional 
Mitigation: Not addressed 
Status: Remains

Socioeconomic Bias (HIGH):
Evidence: Medicaid 33% accuracy 
Mitigation: Not addressed 
Status: Remains



### **4.3 Trade-Offs Documented**


**SMOTE Trade-Off (ACCEPTED):** 
Gains: +33% recall, caught 1 additional death   
Costs: -4% accuracy, fairness worsened (+133% demographic parity)   Decision: ACCEPT (healthcare prioritizes recall)  


**ThresholdOptimizer Trade-Off (REJECTED):**   
Gains: Perfect fairness (metrics <0.04)   
Costs: 0% recall (catches zero deaths)   
Decision: REJECT (unusable clinically)  


![PerformanceByGender](assets/BaselineVsThreshold1.png)

![PerformanceFairness](assets/BaselineVsThreshold2.png)

![Comparison](assets/BaselineVsThreshold3.png)



### **4.4 Limitations**

Data: Only 15 deaths, 2 female deaths insufficient   
Methods: SMOTE preserves demographic imbalance, ThresholdOpt needs larger validation sets   
Fairness: Cannot achieve perfect fairness with current data    
Evaluation: Small test set (3 deaths) limits statistical validity



### **4.5 Key Lessons**  

1. Pre-processing (SMOTE) balances classes but not demographics within classes  
2. Post-processing (ThresholdOpt) requires large validation sets (50+ samples per group)  
3. Healthcare priority: Recall > Fairness > Accuracy
4. Small datasets (<30 samples per demographic per class) limit all techniques
5. Honest failure documentation as valuable as success reporting


### 4.6 POC Dataset Limitation and Full-Scale Implementation Plan

**Important Note: Proof of Concept Study**

This analysis was conducted on the MIMIC-IV **DEMO dataset** containing only 275 admissions (100 unique patients) as a proof-of-concept implementation. The demo dataset is intentionally small and may not represent the full diversity of the complete MIMIC-IV database.

**Demo Dataset Constraints:**
- Only 15 total deaths (5.5% mortality rate)
- Only 2 female deaths in training set
- Limited demographic diversity (some ethnicity groups <5 samples)
- Small validation and test sets (3 deaths each)
- Insufficient statistical power for robust bias detection

**Full-Scale Implementation Plan:**

The methodologies, tools, and workflows developed in this POC would be applied to the **full MIMIC-IV dataset** for production deployment:

Expected Full Dataset Scale:
- 500,000+ ICU admissions
- 50,000+ in-hospital deaths (10% mortality rate)
- 5,000+ female deaths (adequate for learning)
- Balanced demographic representation across age, ethnicity, insurance
- Sufficient samples for train/validation/test (70/15/15 split)

Expected Improvements with Full Data:
- Gender bias resolvable: Sufficient female death samples (>1,000) for robust pattern learning
- SMOTE effectiveness: Demographic-aware sampling possible with large sample sizes
- ThresholdOptimizer viability: Validation set would contain 7,500+ deaths (adequate for threshold learning)
- Statistical significance: Confidence intervals narrow enough for clinical validation
- Intersectional analysis: 50+ samples per demographic intersection enables meaningful subgroup evaluation

**Conclusion:** This POC successfully demonstrated the bias detection and mitigation methodology using Fairlearn and SMOTE. The identified limitations (small sample size, insufficient female deaths, post-processing failures) would be resolved by transitioning to the full MIMIC-IV dataset containing 100× more samples. The workflows and findings from this study provide the foundation for full-scale clinical AI development.

**Note:** This POC study on demo data (275 admissions) establishes methodology for full-scale implementation on complete MIMIC-IV dataset (500,000+ admissions).