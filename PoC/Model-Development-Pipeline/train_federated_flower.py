"""
True cross-silo federated learning with Flower.
Models: Logistic Regression (SGD), MLP (PyTorch), TabNet (optional).
Aggregation: FedAvg, optionally FedProx via mu > 0.
Per-model FedAvg runs sequentially. After all models, an ensemble evaluation averages per-model probabilities.
Final client metrics and macro/micro summaries are saved to CSV/JSON.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns, EvaluateIns, Parameters

from config import (
    FED_DATA_ROOT as DATA_ROOT,
    MODEL_DIR, REPORT_DIR, CLIENT_CACHE_DIR,
    HOSPITALS, SPLIT_COL, VALID_SPLITS,
    TARGET_COL as CFG_TARGET_COL, SEED as RANDOM_STATE,
    MODELS, ROUNDS_PER_MODEL, FED_PROX_MU,
    EXPECTED_CLIENTS, EVAL_THRESHOLD,
    LOGREG_CFG, MLP_CFG, TABNET_CFG,
    SAVE_GLOBAL_WEIGHTS, GLOBAL_WEIGHTS_DIR,
)
from utils import ensure_dir, select_X_y, DEFAULT_TARGET

TARGET_COL = CFG_TARGET_COL or DEFAULT_TARGET


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_OK = True
except Exception:
    TABNET_OK = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_score, recall_score, balanced_accuracy_score, matthews_corrcoef,
    confusion_matrix, brier_score_loss, log_loss, cohen_kappa_score
)


def to_float64(X):
    """
    Convert array or sparse matrix to float64.

    Args:
        X: Array-like or sparse matrix.

    Returns:
        Float64 numpy array or sparse matrix.
    """
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.astype(np.float64)
    except Exception:
        pass
    return np.asarray(X, dtype=np.float64)


def to_float32(X):
    """
    Convert array or sparse matrix to float32.

    Args:
        X: Array-like or sparse matrix.

    Returns:
        Float32 numpy array or sparse matrix.
    """
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            return X.astype(np.float32)
    except Exception:
        pass
    return np.asarray(X, dtype=np.float32)


def to_int64(y):
    """
    Convert vector to int64.

    Args:
        y: Array-like labels.

    Returns:
        Numpy array of int64.
    """
    return np.asarray(y, dtype=np.int64)


def compute_metrics(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """
    Compute standard binary classification metrics from probabilities.

    Args:
        y_true: Ground truth labels as int64.
        p: Predicted probabilities for class 1.
        thr: Decision threshold for class assignment.

    Returns:
        Dict of metrics and counts.
    """
    yhat = (p >= thr).astype(int)
    has_two = np.unique(y_true).size == 2
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    out = {
        "threshold": float(thr),
        "prevalence": float(np.mean(y_true)),
        "pr_auc": float(average_precision_score(y_true, p)) if has_two else np.nan,
        "roc_auc": float(roc_auc_score(y_true, p)) if has_two else np.nan,
        "accuracy": float(accuracy_score(y_true, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, yhat)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, yhat, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, yhat)) if has_two else np.nan,
        "kappa": float(cohen_kappa_score(y_true, yhat)) if has_two else np.nan,
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p)) if has_two else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "n": int(len(y_true)),
    }
    return out


def macro_micro(df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute macro and sample-weighted micro averages of key metrics.

    Args:
        df: DataFrame with one row per client containing metrics.

    Returns:
        Tuple of (macro dict, micro dict).
    """
    metric_cols = [
        "pr_auc", "roc_auc", "f1", "balanced_accuracy", "precision",
        "recall_sensitivity", "specificity", "accuracy", "kappa", "mcc",
        "brier", "log_loss"
    ]
    macro = df[metric_cols].mean(numeric_only=True).to_dict()
    w = df["n"].values
    w = w / w.sum() if w.sum() > 0 else None

    def wavg(col):
        if (w is None) or (col not in df):
            return float("nan")
        v = df[col].values
        return float(np.average(v, weights=w))

    micro = {c: wavg(c) for c in metric_cols}
    return macro, micro


def load_hospital(letter: str) -> pd.DataFrame:
    """
    Load a hospital CSV and validate the split column values.

    Args:
        letter: Hospital letter suffix, such as 'a'.

    Returns:
        DataFrame with normalized split labels.
    """
    path = Path(DATA_ROOT) / f"hospital_{letter}_data.csv"
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    need = {"subject_id", SPLIT_COL, TARGET_COL}
    miss = need - set(df.columns)
    if miss:
        raise KeyError(f"{path} missing columns: {miss}")
    df = df.copy()
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    bad = set(df[SPLIT_COL].unique()) - VALID_SPLITS
    if bad:
        raise ValueError(f"{path} has unexpected split labels: {bad}")
    return df


@dataclass
class LogRegState:
    """
    Container for logistic regression state.
    """
    clf: SGDClassifier
    features: List[str]


def logreg_init(n_features: int) -> SGDClassifier:
    """
    Initialize an SGDClassifier configured for logistic regression.

    Args:
        n_features: Number of input features.

    Returns:
        Configured SGDClassifier instance.
    """
    return SGDClassifier(
        loss="log_loss",
        penalty=LOGREG_CFG["penalty"],
        alpha=LOGREG_CFG["alpha"],
        learning_rate=LOGREG_CFG["learning_rate"],
        random_state=RANDOM_STATE
    )


def logreg_get_params(clf: SGDClassifier) -> List[np.ndarray]:
    """
    Extract model parameters for federated aggregation.

    Args:
        clf: Trained or partially trained SGDClassifier.

    Returns:
        List of ndarrays [coef, intercept] as float32.
    """
    return [clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)]


def logreg_set_params(clf: SGDClassifier, params: List[np.ndarray]) -> None:
    """
    Set model parameters from federated aggregation.

    Args:
        clf: SGDClassifier instance.
        params: List [coef, intercept] arrays.

    Returns:
        None.
    """
    if len(params) == 0:
        return
    coef, intercept = params
    clf.coef_ = np.asarray(coef, dtype=np.float64)
    clf.intercept_ = np.asarray(intercept, dtype=np.float64)
    clf.classes_ = np.array([0, 1], dtype=np.int64)


class MLP(nn.Module):
    """
    Two-hidden-layer MLP producing a binary logit.
    """

    def __init__(self, d_in: int, d_hidden: Optional[int] = None):
        """
        Build the MLP.

        Args:
            d_in: Number of input features.
            d_hidden: Hidden dimension; defaults to config.
        """
        super().__init__()
        d_hidden = d_hidden or MLP_CFG["hidden_dim"]
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x):
        """
        Forward pass returning logits.

        Args:
            x: Input tensor of shape [N, d_in].

        Returns:
            1D tensor of logits.
        """
        return self.net(x).squeeze(1)


def mlp_get_params(model: nn.Module) -> List[np.ndarray]:
    """
    Extract MLP parameters as numpy arrays.

    Args:
        model: PyTorch module.

    Returns:
        List of ndarrays for each parameter tensor.
    """
    return [p.detach().cpu().numpy() for p in model.parameters()]


def mlp_set_params(model: nn.Module, nds: List[np.ndarray]) -> None:
    """
    Load MLP parameters from numpy arrays.

    Args:
        model: PyTorch module.
        nds: List of ndarrays with matching shapes.

    Returns:
        None.
    """
    with torch.no_grad():
        for p, w in zip(model.parameters(), nds):
            p.set_(torch.tensor(w, dtype=p.dtype))


@dataclass
class TabNetState:
    """
    Container for TabNet state.
    """
    model: "TabNetClassifier"
    features: List[str]


def tabnet_get_params(model: "TabNetClassifier") -> List[np.ndarray]:
    """
    Extract TabNet network parameters as numpy arrays.

    Args:
        model: Fitted TabNetClassifier.

    Returns:
        List of state-dict tensors as numpy arrays.
    """
    sd = model.network.state_dict()
    return [v.detach().cpu().numpy() for _, v in sd.items()]


def tabnet_set_params(model: "TabNetClassifier", nds: List[np.ndarray]) -> None:
    """
    Load TabNet network parameters from numpy arrays.

    Args:
        model: TabNetClassifier.
        nds: List of numpy arrays corresponding to state-dict order.

    Returns:
        None.
    """
    keys = list(model.network.state_dict().keys())
    new_sd = {k: torch.tensor(v) for k, v in zip(keys, nds)}
    model.network.load_state_dict(new_sd, strict=True)


class FederatedClient(fl.client.NumPyClient):
    """
    A Flower NumPyClient that can train and evaluate multiple model types on a single hospital's data.
    """

    def __init__(self, hospital_name: str):
        """
        Prepare splits, scalers, and model holders for a hospital.

        Args:
            hospital_name: One of HOSPITALS keys, for example 'hospital_a'.
        """
        self.hospital_name = hospital_name
        self.letter = HOSPITALS[hospital_name]
        self.df = load_hospital(self.letter)

        df_tr = self.df[self.df[SPLIT_COL] == "train"]
        df_va = self.df[self.df[SPLIT_COL] == "validation"]
        df_te = self.df[self.df[SPLIT_COL] == "test"]

        Xtr, ytr = select_X_y(df_tr, target=TARGET_COL)
        Xva, yva = select_X_y(df_va, target=TARGET_COL)
        Xte, yte = select_X_y(df_te, target=TARGET_COL)

        self.features = Xtr.columns.tolist()
        Xtr = Xtr.values
        Xva = Xva.reindex(columns=self.features, fill_value=0.0).values
        Xte = Xte.reindex(columns=self.features, fill_value=0.0).values
        ytr = ytr.values
        yva = yva.values
        yte = yte.values

        scaler = StandardScaler().fit(Xtr)
        Xtr_s64 = to_float64(scaler.transform(Xtr))
        Xva_s64 = to_float64(scaler.transform(Xva))
        Xte_s64 = to_float64(scaler.transform(Xte))

        Xtr_s32 = to_float32(Xtr_s64)
        Xva_s32 = to_float32(Xva_s64)
        Xte_s32 = to_float32(Xte_s64)

        ytr_i64 = to_int64(ytr)
        yva_i64 = to_int64(yva)
        yte_i64 = to_int64(yte)

        self.Xtr_s64, self.Xva_s64, self.Xte_s64 = Xtr_s64, Xva_s64, Xte_s64
        self.Xtr_s32, self.Xva_s32, self.Xte_s32 = Xtr_s32, Xva_s32, Xte_s32
        self.ytr, self.yva, self.yte = ytr_i64, yva_i64, yte_i64

        self.logreg: Optional[SGDClassifier] = None
        self.mlp: Optional[nn.Module] = None
        self.tabnet: Optional["TabNetClassifier"] = None

        ensure_dir(CLIENT_CACHE_DIR)
        self.cache_dir = CLIENT_CACHE_DIR / hospital_name
        ensure_dir(self.cache_dir)

    def _ensure_logreg(self) -> SGDClassifier:
        """
        Ensure a logistic regression instance exists and is initialized.

        Returns:
            SGDClassifier instance with classes allocated.
        """
        if self.logreg is None:
            self.logreg = logreg_init(self.Xtr_s64.shape[1])
            self.logreg.partial_fit(self.Xtr_s64, self.ytr, classes=np.array([0, 1], dtype=np.int64))
        return self.logreg

    def _ensure_mlp(self) -> nn.Module:
        """
        Ensure an MLP model exists.

        Returns:
            PyTorch module in training mode.
        """
        if not TORCH_OK:
            raise RuntimeError("PyTorch not available for MLP.")
        if self.mlp is None:
            self.mlp = MLP(self.Xtr_s32.shape[1]).train()
        return self.mlp

    def _ensure_tabnet(self) -> "TabNetClassifier":
        """
        Ensure a TabNet classifier exists and is initialized.

        Returns:
            TabNetClassifier initialized with a small warm-up fit.
        """
        if not (TORCH_OK and TABNET_OK):
            raise RuntimeError("pytorch and pytorch-tabnet required for TabNet.")
        if self.tabnet is None:
            self.tabnet = TabNetClassifier(
                n_d=TABNET_CFG["n_d"],
                n_a=TABNET_CFG["n_a"],
                n_steps=TABNET_CFG["n_steps"],
                gamma=TABNET_CFG["gamma"],
                seed=RANDOM_STATE,
                optimizer_params=dict(lr=TABNET_CFG["lr"])
            )
            self.tabnet.fit(
                self.Xtr_s32,
                self.ytr,
                eval_set=[(self.Xva_s32, self.yva)],
                max_epochs=1,
                patience=1,
                batch_size=128,
                virtual_batch_size=64
            )
        return self.tabnet

    def get_parameters(self, config):
        """
        Return model parameters for the requested model type.

        Args:
            config: Dict containing model_type.

        Returns:
            List of numpy arrays representing parameters.
        """
        mt = config.get("model_type", "logreg")
        if mt == "logreg":
            clf = self._ensure_logreg()
            return logreg_get_params(clf)
        if mt == "mlp":
            m = self._ensure_mlp()
            return mlp_get_params(m)
        if mt == "tabnet":
            m = self._ensure_tabnet()
            return tabnet_get_params(m)
        if mt == "fgb":
            return []
        raise ValueError(f"Unknown model_type: {mt}")

    def fit(self, parameters, config):
        """
        Perform a local training round for the requested model type.

        Args:
            parameters: Global parameters to initialize from.
            config: Dict with keys phase, model_type, prox_mu.

        Returns:
            Tuple of (updated parameters, num_examples, metrics dict).
        """
        mt = config.get("model_type", "logreg")
        prox_mu = float(config.get("prox_mu", 0.0))

        if mt == "logreg":
            clf = self._ensure_logreg()
            if len(parameters) > 0:
                logreg_set_params(clf, parameters)
            clf.partial_fit(self.Xtr_s64, self.ytr, classes=np.array([0, 1], dtype=np.int64))
            out_params = logreg_get_params(clf)
            n = int(self.Xtr_s64.shape[0])
            return out_params, n, {"hospital": self.hospital_name}

        if mt == "mlp":
            if not TORCH_OK:
                raise RuntimeError("MLP requires PyTorch.")
            model = self._ensure_mlp()
            if len(parameters) > 0:
                mlp_set_params(model, parameters)

            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=MLP_CFG["lr"])
            bce = nn.BCEWithLogitsLoss()

            X = torch.tensor(self.Xtr_s32)
            y = torch.tensor(self.ytr, dtype=torch.float32)
            dl = DataLoader(TensorDataset(X, y), batch_size=MLP_CFG["batch_size"], shuffle=True)

            for _ in range(MLP_CFG["local_epochs"]):
                global_params = [p.detach().clone() for p in model.parameters()] if prox_mu > 0 else None
                for xb, yb in dl:
                    opt.zero_grad()
                    logits = model(xb)
                    loss = bce(logits, yb)
                    if prox_mu > 0 and global_params is not None:
                        prox = 0.0
                        for p, g in zip(model.parameters(), global_params):
                            prox = prox + torch.sum((p - g) ** 2)
                        loss = loss + (prox_mu / 2.0) * prox
                    loss.backward()
                    opt.step()

            out_params = mlp_get_params(model)
            n = int(self.Xtr_s32.shape[0])
            return out_params, n, {"hospital": self.hospital_name}

    def evaluate(self, parameters, config):
        """
        Evaluate the requested model type on test data or perform ensemble evaluation.

        Args:
            parameters: Global parameters of the model or empty for ensemble.
            config: Dict with phase and model_type. For ensemble, contains all model parameters.

        Returns:
            Tuple of (loss, num_examples, metrics dict).
        """
        phase = config.get("phase", "eval")
        mt = config.get("model_type", "logreg")

        def predict_proba_from_model() -> np.ndarray:
            if mt == "logreg":
                clf = self._ensure_logreg()
                if len(parameters) > 0:
                    logreg_set_params(clf, parameters)
                return clf.predict_proba(self.Xte_s64)[:, 1]
            if mt == "mlp":
                if not TORCH_OK:
                    raise RuntimeError("MLP requires PyTorch.")
                model = self._ensure_mlp()
                if len(parameters) > 0:
                    mlp_set_params(model, parameters)
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(self.Xte_s32))
                    return torch.sigmoid(logits).cpu().numpy()
            if mt == "tabnet":
                if not (TORCH_OK and TABNET_OK):
                    raise RuntimeError("TabNet requires pytorch-tabnet + torch.")
                model = self._ensure_tabnet()
                if len(parameters) > 0:
                    tabnet_set_params(model, parameters)
                return model.predict_proba(self.Xte_s32)[:, 1]
            raise ValueError(f"Unknown model_type: {mt}")

        if phase == "ensemble_eval":
            gparams = config.get("global_params_by_model", {})
            probs = []
            if "logreg" in gparams:
                clf = self._ensure_logreg()
                logreg_set_params(clf, [np.array(a) for a in gparams["logreg"]])
                probs.append(clf.predict_proba(self.Xte_s64)[:, 1])
            if "mlp" in gparams and TORCH_OK:
                m = self._ensure_mlp()
                mlp_set_params(m, [np.array(a) for a in gparams["mlp"]])
                m.eval()
                with torch.no_grad():
                    logits = m(torch.tensor(self.Xte_s32))
                    probs.append(torch.sigmoid(logits).cpu().numpy())
            if "tabnet" in gparams and TORCH_OK and TABNET_OK:
                t = self._ensure_tabnet()
                tabnet_set_params(t, [np.array(a) for a in gparams["tabnet"]])
                probs.append(t.predict_proba(self.Xte_s32)[:, 1])

            if len(probs) == 0:
                return 0.5, int(len(self.yte)), {"hospital": self.hospital_name, "ensemble_used": 0}

            p_ens = np.mean(probs, axis=0)
            m = compute_metrics(self.yte, p_ens, thr=EVAL_THRESHOLD)
            loss = 1.0 - (m.get("roc_auc") if np.isfinite(m.get("roc_auc", np.nan)) else 0.0)
            m.update({"hospital": self.hospital_name, "ensemble_used": len(probs)})
            return float(loss), int(m["n"]), m

        p = predict_proba_from_model()
        m = compute_metrics(self.yte, p, thr=EVAL_THRESHOLD)
        loss = 1.0 - (m.get("roc_auc") if np.isfinite(m.get("roc_auc", np.nan)) else 0.0)
        m.update({"hospital": self.hospital_name, "model_type": mt})
        return float(loss), int(m["n"]), m


class MultiModelFedAvg(fl.server.strategy.Strategy):
    """
    A Flower Strategy that trains several model types in sequence using FedAvg,
    then runs a client-side ensemble evaluation across all trained models.
    """

    def __init__(self, models: List[str], rounds_per_model: int, expected_clients: int = 3, prox_mu: float = 0.0):
        """
        Initialize the multi-model FedAvg strategy.

        Args:
            models: Model type names in training order.
            rounds_per_model: Number of FedAvg rounds per model.
            expected_clients: Number of clients to sample per round.
            prox_mu: FedProx regularization coefficient.
        """
        self.models = models
        self.rounds_per_model = rounds_per_model
        self.expected_clients = expected_clients
        self.prox_mu = prox_mu

        self.current_model_idx = 0
        self.current_round_for_model = 0
        self.global_params_by_model: Dict[str, List[np.ndarray]] = {}
        self._empty_params: Parameters = ndarrays_to_parameters([])

    def _current_model(self) -> Optional[str]:
        """
        Return the current model type or None if finished.

        Returns:
            Model type string or None.
        """
        if self.current_model_idx >= len(self.models):
            return None
        return self.models[self.current_model_idx]

    def _advance_round(self) -> None:
        """
        Advance round counters and switch model if needed.
        """
        self.current_round_for_model += 1
        if self.current_round_for_model >= self.rounds_per_model:
            self.current_model_idx += 1
            self.current_round_for_model = 0

    def initialize_parameters(self, client_manager):
        """
        Let clients bootstrap parameters for the first model.

        Args:
            client_manager: Flower client manager.

        Returns:
            None to request initial params from clients.
        """
        return None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        """
        Configure fit instructions for the current model.

        Args:
            server_round: Current round number.
            parameters: Unused global parameters.
            client_manager: Flower client manager.

        Returns:
            List of (client, FitIns) pairs.
        """
        mt = self._current_model()
        if mt is None:
            return []
        clients = client_manager.sample(num_clients=self.expected_clients)
        if mt in self.global_params_by_model:
            params = ndarrays_to_parameters(self.global_params_by_model[mt])
        else:
            params = self._empty_params
        fit_ins = FitIns(parameters=params, config={"phase": "train", "model_type": mt, "prox_mu": self.prox_mu})
        return [(c, fit_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate client updates via weighted averaging and store global parameters.

        Args:
            server_round: Current round number.
            results: List of (client, FitRes).
            failures: List of failures.

        Returns:
            Tuple of (Parameters, metrics dict).
        """
        mt = self._current_model()
        if mt is None or len(results) == 0:
            return None, {}

        weights_results: List[Tuple[List[np.ndarray], int]] = []
        for _, fitres in results:
            nds = parameters_to_ndarrays(fitres.parameters)
            n = fitres.num_examples
            if len(nds) > 0 and n > 0:
                weights_results.append((nds, n))

        if not weights_results:
            return None, {}

        n_tot = sum(n for _, n in weights_results)
        avg = []
        for layer_idx in range(len(weights_results[0][0])):
            layer_sum = sum((nds[layer_idx] * (n / n_tot) for nds, n in weights_results))
            avg.append(layer_sum)

        self.global_params_by_model[mt] = avg

        if SAVE_GLOBAL_WEIGHTS:
            ensure_dir(GLOBAL_WEIGHTS_DIR)
            out = GLOBAL_WEIGHTS_DIR / f"global_{mt}.npz"
            np.savez(out, **{f"layer_{i}": w for i, w in enumerate(avg)})

        self._advance_round()
        return ndarrays_to_parameters(avg), {}

    def configure_evaluate(self, server_round, parameters: Parameters, client_manager):
        """
        Configure evaluation for the current model or trigger ensemble evaluation.

        Args:
            server_round: Current round number.
            parameters: Global parameters.
            client_manager: Flower client manager.

        Returns:
            List of (client, EvaluateIns) pairs.
        """
        mt = self._current_model()
        clients = client_manager.sample(num_clients=self.expected_clients)
        if mt is not None:
            params = ndarrays_to_parameters(self.global_params_by_model.get(mt, []))
            eval_ins = EvaluateIns(parameters=params, config={"phase": "eval", "model_type": mt})
            return [(c, eval_ins) for c in clients]

        cfg = {
            "phase": "ensemble_eval",
            "model_type": "ensemble",
            "global_params_by_model": {k: [a.tolist() for a in v] for k, v in self.global_params_by_model.items()},
        }
        eval_ins = EvaluateIns(parameters=ndarrays_to_parameters([]), config=cfg)
        return [(c, eval_ins) for c in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate evaluation metrics from clients and write final CSV/JSON summaries.

        Args:
            server_round: Round number.
            results: List of (client, EvaluateRes).
            failures: List of failures.

        Returns:
            Tuple of (loss, metrics dict).
        """
        ensure_dir(REPORT_DIR)
        recs = []
        for _, evr in results:
            m = evr.metrics or {}
            if "hospital" in m:
                recs.append(m)

        if not recs:
            return 0.0, {}

        df = pd.DataFrame(recs)

        if "ensemble_used" in df.columns:
            df.to_csv(REPORT_DIR / "ensemble_client_metrics.csv", index=False)
            macro, micro = macro_micro(df)
            with open(REPORT_DIR / "ensemble_summary.json", "w") as f:
                json.dump({"rows": int(df.shape[0]), "macro": macro, "micro": micro}, f, indent=2)
        else:
            if "model_type" in df.columns:
                mt_vals = df["model_type"].unique().tolist()
                if len(mt_vals) == 1:
                    mt = mt_vals[0]
                    df.to_csv(REPORT_DIR / f"{mt}_client_metrics.csv", index=False)
                    macro, micro = macro_micro(df)
                    with open(REPORT_DIR / f"{mt}_summary.json", "w") as f:
                        json.dump({"rows": int(df.shape[0]), "macro": macro, "micro": micro}, f, indent=2)

        mean_loss = float(np.mean([evr.loss for _, evr in results]))
        return mean_loss, {}

    def evaluate(self, server_round, parameters):
        """
        No server-side evaluation.

        Args:
            server_round: Round number.
            parameters: Global parameters.

        Returns:
            None to skip.
        """
        return None


def run_server(address: str, models: List[str], rounds_per_model: int, prox_mu: float) -> None:
    """
    Start the Flower server with the multi-model FedAvg strategy.

    Args:
        address: Host:port of the Flower server.
        models: Model names to train sequentially.
        rounds_per_model: Number of rounds per model.
        prox_mu: FedProx coefficient.

    Returns:
        None.
    """
    ensure_dir(REPORT_DIR)
    ensure_dir(MODEL_DIR)
    strategy = MultiModelFedAvg(models=models, rounds_per_model=rounds_per_model, expected_clients=EXPECTED_CLIENTS, prox_mu=prox_mu)
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=len(models) * rounds_per_model + 1),
        strategy=strategy,
    )


def run_client(address: str, hospital: str) -> None:
    """
    Start a Flower client for a given hospital.

    Args:
        address: Host:port of the Flower server.
        hospital: Hospital name, must be a key in HOSPITALS.

    Returns:
        None.
    """
    if hospital not in HOSPITALS:
        raise SystemExit(f"--hospital must be one of {list(HOSPITALS.keys())}")
    client = FederatedClient(hospital_name=hospital)
    fl.client.start_client(server_address=address, client=client.to_client())


def parse_models(s: str) -> List[str]:
    """
    Parse and validate a comma-separated model list.

    Args:
        s: String like "logreg,mlp,tabnet".

    Returns:
        List of valid model names with unknowns rejected.
    """
    raw = [x.strip().lower() for x in s.split(",") if x.strip()]
    allowed = {"logreg", "mlp", "tabnet", "fgb"}
    for m in raw:
        if m not in allowed:
            raise SystemExit(f"Unknown model '{m}'. Allowed: {sorted(list(allowed))}")
    return [m for m in raw if m != "fgb"]


def main() -> None:
    """
    CLI entrypoint to run server or client.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["server", "client"], required=True)
    p.add_argument("--address", default="127.0.0.1:8080")
    p.add_argument("--hospital", choices=list(HOSPITALS.keys()))
    p.add_argument("--models", default="logreg,mlp,tabnet", help="Comma-separated among: logreg,mlp,tabnet,fgb (fgb is ignored).")
    p.add_argument("--rounds_per_model", type=int, default=ROUNDS_PER_MODEL)
    p.add_argument("--prox_mu", type=float, default=FED_PROX_MU, help="FedProx mu (0.0 = FedAvg).")
    args = p.parse_args()

    if args.role == "server":
        run_server(args.address, parse_models(args.models), args.rounds_per_model, args.prox_mu)
    else:
        if not args.hospital:
            raise SystemExit("--hospital is required for role=client")
        run_client(args.address, args.hospital)


if __name__ == "__main__":
    main()
