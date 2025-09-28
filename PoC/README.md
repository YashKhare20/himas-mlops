# HIMAS Layer 2 - Federated Learning Setup

Hey team! This is the Layer 2 implementation - federated learning for our healthcare AI project.

## What I Built

Built a federated learning system where 3 simulated hospital networks collaborate on AI model training without sharing patient data. Uses breast cancer dataset from sklearn and achieves around 98% accuracy when hospitals share model parameters instead of raw data.

## Quick Setup

```bash
git clone https://github.com/Manjusha-26/HIMAS-MLOps-POC.git
cd HIMAS-MLOps-POC
python -m venv himas-env
source himas-env/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

## How to Run

**Easy way (simulation on your laptop):**
```bash
python main_himas_federated.py
```

**Docker way (if you want to see containers):**
```bash
docker-compose up --build
```

The Docker version takes 2-3 minutes to build first time. After that it's fast.

## What You'll See

- Loads breast cancer data (569 samples)
- Splits it across 3 hospitals 
- Each hospital trains locally
- They share model weights (not patient data)
- Gets to ~98% accuracy through collaboration

## Key Files

- `main_himas_federated.py` - runs everything
- `src/federated/` - Flower federated learning stuff
- `src/storage/` - ChromaDB medical knowledge storage
- `docker-compose.yml` - if you want containerized version

## Technologies

- Flower for federated learning
- ChromaDB for storing medical knowledge
- sklearn datasets (breast cancer)
- Docker for deployment demo

## Issues?

If Docker fails with port 8080 error:
```bash
docker stop $(docker ps -q)
```

If Python imports fail, make sure you activated the virtual environment.

## Integration Notes

This Layer 2 gives you the federated learning foundation. You can plug in:
- Layer 1 data from your BigQuery setup
- Layer 3 deployment in your cloud infrastructure  
- Layer 4 monitoring dashboards
- Layer 5 frontend interfaces

The main functions you'll probably want to use are in `main_himas_federated.py` - the `CompleteFederatedHIMAS` class handles everything.

Let me know if anything breaks or you need changes for your layers.