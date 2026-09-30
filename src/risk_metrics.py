"""Métricas de validação e critérios demonstrativos de Model Health.

Funções extraídas do Motor de Risco de Contraparte V3, com fórmulas e
thresholds preservados. Os semáforos são heurísticas experimentais;
não representam aprovação regulatória nem monitoramento temporal.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def ks_from_scores(y, p):
    """Estatística KS das distribuições de score por classe (via curva ROC)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece_score(y, p, bins=10):
    """Expected Calibration Error ponderado por frequência dos intervalos."""
    tmp = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    try:
        tmp["bin"] = pd.qcut(tmp["p"], q=bins, duplicates="drop")
    except Exception:
        return np.nan
    g = tmp.groupby("bin", observed=True).agg(
        n=("y", "size"), obs=("y", "mean"), pred=("p", "mean")
    )
    return float(np.sum((g["n"] / len(tmp)) * np.abs(g["obs"] - g["pred"])))


def psi_score(expected, actual, bins=10):
    """PSI com pontos de corte definidos pela distribuição de referência.

    Na V3, aplicado ao score treino versus teste aleatórios: é somente
    um proxy de estabilidade, não aferição de drift temporal em produção.
    """
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    eh, _ = np.histogram(e, bins=edges)
    ah, _ = np.histogram(a, bins=edges)
    ep = np.clip(eh / eh.sum(), 1e-6, None)
    ap = np.clip(ah / ah.sum(), 1e-6, None)
    return float(np.sum((ap - ep) * np.log(ap / ep)))


def metric_status_auc(x):
    return "Verde" if x >= .75 else ("Amarelo" if x >= .65 else "Vermelho")


def metric_status_bss(x):
    return "Verde" if x >= .10 else ("Amarelo" if x > 0 else "Vermelho")


def metric_status_calratio(x):
    return "Verde" if .80 <= x <= 1.20 else ("Amarelo" if .60 <= x <= 1.40 else "Vermelho")


def metric_status_psi(x):
    if not np.isfinite(x):
        return "Amarelo"
    return "Verde" if x < .10 else ("Amarelo" if x < .25 else "Vermelho")


def metric_status_missing(max_missing):
    return "Verde" if max_missing < .10 else ("Amarelo" if max_missing < .30 else "Vermelho")


def health_summary(statuses):
    if "Vermelho" in statuses:
        return "REQUER ATENÇÃO", "Vermelho"
    if "Amarelo" in statuses:
        return "ADEQUADO COM MONITORAMENTO", "Amarelo"
    return "VALIDAÇÃO FORA DA AMOSTRA FAVORÁVEL", "Verde"
