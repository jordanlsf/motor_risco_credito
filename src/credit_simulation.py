"""
Motor Quantitativo de Risco de Crédito e Contraparte.

Módulo de simulação de exposição futura, inadimplência,
Wrong-Way Risk, Credit VaR e CVaR.

Projeto experimental de portfólio.
"""

import numpy as np
import streamlit as st

from scipy.special import expit, logit


@st.cache_data(show_spinner="Simulando perda de crédito...")
def simulate_credit(
    pd0,
    lgd,
    ead_net,
    sigma_exp,
    beta_wwr,
    n=60000,
    seed=42
):
    """
    Simula exposição futura e distribuição de perdas de crédito.

    Parâmetros
    ----------
    pd0 : float
        Probabilidade inicial de default.

    lgd : float
        Severidade da perda em caso de default.

    ead_net : float
        Exposição líquida de garantias.

    sigma_exp : float
        Volatilidade utilizada na simulação da exposição.

    beta_wwr : float
        Intensidade da dependência entre exposição e PD.

    n : int
        Número de simulações de Monte Carlo.

    seed : int
        Semente para reprodutibilidade.
    """

    rng = np.random.default_rng(seed)

    z = rng.normal(size=n)

    exposure = (
        np.maximum(ead_net, 0)
        * np.exp(-0.5 * sigma_exp**2 + sigma_exp * z)
    )

    pd0 = float(np.clip(pd0, 1e-6, 1 - 1e-6))

    # Wrong-Way Risk:
    # preserva aproximadamente a PD média inicial,
    # incorporando dependência entre exposição e default.

    if abs(float(beta_wwr)) < 1e-12:

        scenario_pd = np.full(n, pd0, dtype=float)

    else:

        base_logit = logit(pd0)

        lo, hi = -20.0, 20.0

        for _ in range(70):

            mid = 0.5 * (lo + hi)

            mean_pd = float(
                np.mean(
                    expit(base_logit + mid + beta_wwr * z)
                )
            )

            if mean_pd > pd0:
                hi = mid
            else:
                lo = mid

        intercept_shift = 0.5 * (lo + hi)

        scenario_pd = expit(
            base_logit + intercept_shift + beta_wwr * z
        )

    # Simulação de default e perda financeira

    default = rng.uniform(size=n) < scenario_pd

    loss = default.astype(float) * float(lgd) * exposure

    # Métricas de risco de cauda

    def tail_metrics(x, alpha):

        x = np.asarray(x, float)

        var = float(
            np.quantile(x, alpha, method="higher")
        )

        k = max(
            1,
            int(np.ceil((1 - alpha) * len(x)))
        )

        cvar = float(
            np.mean(np.sort(x)[-k:])
        )

        return var, cvar

    v95, c95 = tail_metrics(loss, 0.95)
    v99, c99 = tail_metrics(loss, 0.99)

    return {
        "exposure": exposure,
        "loss": loss,
        "scenario_pd": scenario_pd,
        "PFE95": float(np.quantile(exposure, 0.95)),
        "PFE99": float(np.quantile(exposure, 0.99)),
        "EL": float(loss.mean()),
        "DefaultFreq": float(default.mean()),
        "CreditVaR95": v95,
        "CreditCVaR95": c95,
        "CreditVaR99": v99,
        "CreditCVaR99": c99,
    }
