# -*- coding: utf-8 -*-
"""
Global configuration for data export, splitting, and true federated learning.
Import values with:
    from config import *
"""

from pathlib import Path

SEED = 42

GBQ_PROJECT = "erudite-carving-472018-r5"
GBQ_DATASETS = ["curated_demo", "federated_demo"]
DATA_OUT_ROOT = Path(r"PoC\Data-Pipeline\data")

FEDERATED_DIR = DATA_OUT_ROOT / "federated_demo"
SPLITS_OUT_DIR = Path(r"PoC\Model-Development-Pipeline\artifacts\splits")

SPLIT_COL = "data_split"
VALID_SPLITS = {"train", "validation", "test"}
TARGET_COL = "icu_mortality_label"

ENFORCE_MIN_PER_CLASS = False
MIN_PER_CLASS = 1

FED_DATA_ROOT = DATA_OUT_ROOT / "federated_demo"
MODEL_DIR = Path(r"PoC\Model-Development-Pipeline\artifacts\models_federated_true")
REPORT_DIR = Path(r"PoC\Model-Development-Pipeline\artifacts\reports_federated_true")
CLIENT_CACHE_DIR = REPORT_DIR / "client_cache"
SAVE_GLOBAL_WEIGHTS = True
GLOBAL_WEIGHTS_DIR = MODEL_DIR / "global_weights"

FLOWER_ADDRESS_DEFAULT = "127.0.0.1:8080"
EXPECTED_CLIENTS = 3

MODELS = ["logreg", "mlp", "tabnet"]
ROUNDS_PER_MODEL = 3
FED_PROX_MU = 0.0

EVAL_THRESHOLD = 0.5
USE_VALIDATION_TUNED_THRESHOLD = False

METRICS_LIST = [
    "pr_auc", "roc_auc", "f1", "balanced_accuracy",
    "precision", "recall_sensitivity", "specificity",
    "accuracy", "kappa", "mcc", "brier", "log_loss"
]

LOGREG_CFG = dict(
    penalty="l2",
    alpha=1e-4,
    learning_rate="optimal",
)

MLP_CFG = dict(
    hidden_dim=64,
    lr=1e-3,
    local_epochs=1,
    batch_size=64,
)

TABNET_CFG = dict(
    n_d=8,
    n_a=8,
    n_steps=3,
    gamma=1.5,
    lr=1e-3,
    max_epochs=3,
    patience=2,
    batch_size=256,
    virtual_batch_size=128,
)

DEVICE = "cpu"

HOSPITALS = {"hospital_a": "a", "hospital_b": "b", "hospital_c": "c"}
