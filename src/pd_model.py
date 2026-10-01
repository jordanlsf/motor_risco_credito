"""Modelagem experimental de Probability of Default (PD).

Extraído do Motor de Risco de Contraparte V3, preservando os métodos,
hiperparâmetros e métricas originais. A divisão aleatória estratificada
não equivale a uma validação temporal/out-of-time.
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
)
from .risk_metrics import ks_from_scores, ece_score, psi_score

SEED = 42


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita extremos por quantis ajustados somente aos dados de treino."""

    def __init__(self, low=0.005, high=0.995):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(arr, self.low, axis=0)
        self.hi_ = np.nanquantile(arr, self.high, axis=0)
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        return np.clip(arr, self.lo_, self.hi_)








@st.cache_resource(show_spinner="Treinando e calibrando modelos de PD...")
def train_models(df, drop_attr37=True):
    """Treina três modelos de PD e devolve métricas e predições da V3.

    IMPORTANTE: o conjunto de teste também orienta a escolha do modelo
    sugerido. Essa regra é mantida para equivalência com a V3; uma futura
    versão deverá reservar um teste independente após a seleção.
    """
    features = [c for c in df.columns if c.startswith("Attr")]
    if drop_attr37 and "Attr37" in features:
        features.remove("Attr37")

    X = df[features].copy()
    X["MissingCount"] = X.isna().sum(axis=1).astype(float)
    features2 = list(X.columns)
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    preprocess_basic = [
        ("winsor", Winsorizer(0.005, 0.995)),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    candidates = {
        "Regressão Logística": Pipeline(preprocess_basic + [
            ("scale", RobustScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", C=0.6,
                solver="liblinear", random_state=SEED
            )),
        ]),
        "Random Forest": Pipeline(preprocess_basic + [
            ("model", RandomForestClassifier(
                n_estimators=320, max_depth=9, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=SEED
            )),
        ]),
        "Gradient Boosting": Pipeline(preprocess_basic + [
            ("model", HistGradientBoostingClassifier(
                max_iter=220, learning_rate=0.06, max_leaf_nodes=25,
                min_samples_leaf=18, l2_regularization=0.5,
                class_weight="balanced", random_state=SEED
            )),
        ]),
    }

    models = {}
    rows = []
    preds = {}
    base_rate_train = float(y_train.mean())
    base_rate_test = float(y_test.mean())
    naive_test = np.repeat(base_rate_train, len(y_test))
    brier_naive = brier_score_loss(y_test, naive_test)

    for name, base in candidates.items():
        calibrated = CalibratedClassifierCV(
            estimator=base, method="sigmoid", cv=3, n_jobs=-1
        )
        calibrated.fit(X_train, y_train)
        p_test = calibrated.predict_proba(X_test)[:, 1]
        p_train = calibrated.predict_proba(X_train)[:, 1]

        auc = roc_auc_score(y_test, p_test)
        ap = average_precision_score(y_test, p_test)
        brier = brier_score_loss(y_test, p_test)
        bss = 1 - brier / brier_naive
        ks = ks_from_scores(y_test, p_test)
        ece = ece_score(y_test, p_test)
        cal_ratio = float(np.mean(p_test) / max(base_rate_test, 1e-12))
        psi = psi_score(p_train, p_test)
        ll = log_loss(y_test, np.clip(p_test, 1e-8, 1 - 1e-8))

        models[name] = calibrated
        preds[name] = {"train": p_train, "test": p_test}
        rows.append({
            "Modelo": name,
            "ROC-AUC": auc,
            "Gini": 2 * auc - 1,
            "PR-AUC": ap,
            "KS": ks,
            "Brier": brier,
            "Brier Skill": bss,
            "ECE": ece,
            "Calibração (Prev/Obs)": cal_ratio,
            "PSI treino-teste": psi,
            "LogLoss": ll,
        })

    metrics = pd.DataFrame(rows).set_index("Modelo")
    eligible = metrics[metrics["ROC-AUC"] >= 0.65]
    if len(eligible):
        suggested = eligible["Brier"].idxmin()
    else:
        suggested = metrics["ROC-AUC"].idxmax()

    return {
        "features": features2,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "preds": preds,
        "metrics": metrics,
        "suggested": suggested,
        "base_rate_train": base_rate_train,
        "base_rate_test": base_rate_test,
    }
