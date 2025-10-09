This folder stores MODEL ARTIFACTS produced by the federated runs.

Typical contents:
- global_model_round_*.pt / .pkl / .onnx  → aggregated models per round
- final_global_model.*                     → final checkpoint after FedAvg completes
- scaler_info.json / feature_list.json     → copies used at evaluation time
- plots/                                   → confusion matrices, ROC/PR curves, etc.

Notes:
- This directory is mounted as a Docker volume so files persist across container rebuilds.
- Large binaries are not intended for Git; keep this folder in .gitignore.
