# -*- coding: utf-8 -*-
"""Simulações ilustrativas de EAD/PFE, perda de crédito e Wrong-Way Risk.

Função extraída literalmente do Motor V3, preservando seed, fórmula e
quantis empíricos. Não configura EAD/LGD calibrados com dados reais.
"""
import numpy as np
import streamlit as st
from scipy.special import expit, logit

SEED = 42
N_CREDIT_SIM = 60000

@st.cache_data(show_spinner="Simulando perda de crédito...")
def simulate_credit(pd0, lgd, ead_net, sigma_exp, beta_wwr, n=N_CREDIT_SIM, seed=SEED):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    exposure = np.maximum(ead_net, 0) * np.exp(-.5*sigma_exp**2 + sigma_exp*z)

    pd0 = float(np.clip(pd0, 1e-6, 1-1e-6))

    # Wrong-Way Risk deve representar dependência entre exposição e crédito,
    # sem inflar mecanicamente a PD média-base. Ajustamos um intercepto para
    # manter E[PD_s] aproximadamente igual a pd0 na própria amostra simulada.
    if abs(float(beta_wwr)) < 1e-12:
        scenario_pd = np.full(n, pd0, dtype=float)
    else:
        base_logit = logit(pd0)
        lo, hi = -20.0, 20.0
        for _ in range(70):
            mid = 0.5*(lo+hi)
            mean_pd = float(np.mean(expit(base_logit + mid + beta_wwr*z)))
            if mean_pd > pd0:
                hi = mid
            else:
                lo = mid
        intercept_shift = 0.5*(lo+hi)
        scenario_pd = expit(base_logit + intercept_shift + beta_wwr*z)

    default = rng.uniform(size=n) < scenario_pd
    loss = default.astype(float) * float(lgd) * exposure

    def tail_metrics(x, alpha):
        x = np.asarray(x, float)
        var = float(np.quantile(x, alpha, method="higher"))
        k = max(1, int(np.ceil((1-alpha)*len(x))))
        cvar = float(np.mean(np.sort(x)[-k:]))
        return var, cvar

    v95, c95 = tail_metrics(loss, .95)
    v99, c99 = tail_metrics(loss, .99)
    return {
        "exposure": exposure,
        "loss": loss,
        "scenario_pd": scenario_pd,
        "PFE95": float(np.quantile(exposure,.95)),
        "PFE99": float(np.quantile(exposure,.99)),
        "EL": float(loss.mean()),
        "DefaultFreq": float(default.mean()),
        "CreditVaR95": v95,
        "CreditCVaR95": c95,
        "CreditVaR99": v99,
        "CreditCVaR99": c99,
    }
