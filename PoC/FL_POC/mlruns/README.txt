This folder is the MLflow BACKEND STORE for experiments and runs.

Typical contents (managed by MLflow):
- experiments/           → experiment metadata (IDs, names)
- <exp_id>/              → run folders (metrics, params, tags, artifacts pointers)
- meta.yaml              → MLflow tracking metadata files

Notes:
- The web UI reads from here (http://localhost:5000 in this POC).
- Structure is MLflow-internal; do not hand-edit files.
- This directory is mounted as a Docker volume and should be .gitignored.
