# -*- coding: utf-8 -*-
"""Interface Streamlit extraída da V3. Cálculos preservados na Etapa 2."""

import os
import math
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from scipy.io import arff
from scipy.special import expit, logit
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, precision_recall_curve,
    brier_score_loss, log_loss, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score
)
from sklearn.inspection import permutation_importance

# ============================================================
# CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="Motor de Risco de Contraparte",
    page_icon="🏦",
    layout="wide"
)

DATA_DIR = str(__import__("pathlib").Path(__file__).resolve().parent / "data" / "polish_bankruptcy")
SEED = 42
N_DEMO_CP = 120
N_CREDIT_SIM = 60000

# ============================================================
# DICIONÁRIO DOS 64 ATRIBUTOS
# ============================================================

ATTR = {
    "Attr1": "Lucro líquido / Ativos totais",
    "Attr2": "Passivo total / Ativos totais",
    "Attr3": "Capital de giro / Ativos totais",
    "Attr4": "Ativo circulante / Passivo de curto prazo",
    "Attr5": "Liquidez operacional em dias (aprox.)",
    "Attr6": "Lucros acumulados / Ativos totais",
    "Attr7": "EBIT / Ativos totais",
    "Attr8": "Patrimônio líquido contábil / Passivo total",
    "Attr9": "Vendas / Ativos totais",
    "Attr10": "Patrimônio líquido / Ativos totais",
    "Attr11": "Resultado ampliado / Ativos totais",
    "Attr12": "Lucro bruto / Passivo de curto prazo",
    "Attr13": "(Lucro bruto + depreciação) / Vendas",
    "Attr14": "(Lucro bruto + juros) / Ativos totais",
    "Attr15": "Passivo total × 365 / (lucro bruto + depreciação)",
    "Attr16": "(Lucro bruto + depreciação) / Passivo total",
    "Attr17": "Ativos totais / Passivo total",
    "Attr18": "Lucro bruto / Ativos totais",
    "Attr19": "Lucro bruto / Vendas",
    "Attr20": "Estoque × 365 / Vendas",
    "Attr21": "Vendas do período / Vendas do período anterior",
    "Attr22": "Resultado operacional / Ativos totais",
    "Attr23": "Lucro líquido / Vendas",
    "Attr24": "Lucro bruto médio de 3 anos / Ativos totais",
    "Attr25": "(Patrimônio líquido - capital social) / Ativos totais",
    "Attr26": "(Lucro líquido + depreciação) / Passivo total",
    "Attr27": "Resultado operacional / Despesas financeiras",
    "Attr28": "Capital de giro / Ativo imobilizado",
    "Attr29": "Logaritmo dos Ativos totais",
    "Attr30": "(Passivo total - caixa) / Vendas",
    "Attr31": "(Lucro bruto + juros) / Vendas",
    "Attr32": "Passivo circulante × 365 / Custo dos produtos vendidos",
    "Attr33": "Despesas operacionais / Passivo de curto prazo",
    "Attr34": "Despesas operacionais / Passivo total",
    "Attr35": "Lucro sobre vendas / Ativos totais",
    "Attr36": "Vendas totais / Ativos totais",
    "Attr37": "(Ativo circulante - estoques) / Passivo de longo prazo",
    "Attr38": "Capital permanente / Ativos totais",
    "Attr39": "Lucro sobre vendas / Vendas",
    "Attr40": "(Ativo circulante - estoque - recebíveis) / Passivo curto",
    "Attr41": "Passivo total / capacidade operacional de pagamento",
    "Attr42": "Resultado operacional / Vendas",
    "Attr43": "Giro de recebíveis + giro de estoque (dias)",
    "Attr44": "Recebíveis × 365 / Vendas",
    "Attr45": "Lucro líquido / Estoque",
    "Attr46": "(Ativo circulante - estoque) / Passivo de curto prazo",
    "Attr47": "Estoque × 365 / Custo dos produtos vendidos",
    "Attr48": "Resultado operacional antes de depreciação / Ativos",
    "Attr49": "Resultado operacional antes de depreciação / Vendas",
    "Attr50": "Ativo circulante / Passivo total",
    "Attr51": "Passivo de curto prazo / Ativos totais",
    "Attr52": "Passivo de curto prazo × 365 / Custo dos produtos",
    "Attr53": "Patrimônio líquido / Ativo imobilizado",
    "Attr54": "Capital permanente / Ativo imobilizado",
    "Attr55": "Capital de giro",
    "Attr56": "(Vendas - custo dos produtos) / Vendas",
    "Attr57": "Liquidez operacional ajustada",
    "Attr58": "Custos totais / Vendas totais",
    "Attr59": "Passivo de longo prazo / Patrimônio líquido",
    "Attr60": "Vendas / Estoque",
    "Attr61": "Vendas / Contas a receber",
    "Attr62": "Passivo de curto prazo × 365 / Vendas",
    "Attr63": "Vendas / Passivo de curto prazo",
    "Attr64": "Vendas / Ativo imobilizado",
}

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem;}
.cp-callout {padding: 14px 16px; border-radius: 12px; background:#f5f8fb;
             border:1px solid #d8e2e8; color:#1f2937; margin:8px 0 14px 0;}
.cp-warn {padding:14px 16px; border-radius:12px; background:#fff6df;
          border:1px solid #e7c66d; color:#2c2617; margin:8px 0 14px 0;}
.cp-ok {padding:14px 16px; border-radius:12px; background:#edf8f2;
        border:1px solid #acd8be; color:#173326; margin:8px 0 14px 0;}
.cp-bad {padding:14px 16px; border-radius:12px; background:#fff0f0;
         border:1px solid #e4b4b4; color:#3b1e1e; margin:8px 0 14px 0;}
.small-note {font-size:.84rem; color:#64748b;}
</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# HELPERS
# ============================================================

def brl(x):
    """Formata valores monetários pela magnitude real."""
    x = float(x)
    a = abs(x)
    if a >= 1e9:
        return f"R$ {x/1e9:,.2f} bi".replace(",", "X").replace(".", ",").replace("X", ".")
    if a >= 1e6:
        return f"R$ {x/1e6:,.1f} mi".replace(",", "X").replace(".", ",").replace("X", ".")
    if a >= 1e3:
        return f"R$ {x/1e3:,.1f} mil".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")


def brl_el(x):
    """
    Formatter explícito para Perda Esperada.
    Evita qualquer conversão fixa para milhões: R$ 652.700 aparece como
    R$ 652,7 mil, e não como R$ 652,7 milhões.
    """
    return brl(float(x))


def pct(x, d=1):
    return f"{100*x:.{d}f}%".replace(".", ",")


def rating_from_pd(pd_value):
    # Faixas exclusivamente demonstrativas, não equivalentes a agência de rating.
    if pd_value < 0.01:
        return "A"
    if pd_value < 0.03:
        return "BBB"
    if pd_value < 0.07:
        return "BB"
    if pd_value < 0.15:
        return "B"
    return "CCC"


def color_status(s):
    return {"Verde":"🟢", "Amarelo":"🟠", "Vermelho":"🔴"}.get(s, "⚪")


# ============================================================
# LEITURA DA BASE
# ============================================================

@st.cache_data
def load_arff(horizon_months=12):
    # Nomenclatura do UCI é invertida em relação ao horizonte:
    # 5year.arff -> default em ~1 ano; 4year -> ~2 anos; ...
    file_map = {12:"5year.arff", 24:"4year.arff", 36:"3year.arff", 48:"2year.arff", 60:"1year.arff"}
    fp = os.path.join(DATA_DIR, file_map[horizon_months])
    data, _ = arff.loadarff(fp)
    df = pd.DataFrame(data)
    for c in [x for x in df.columns if x.startswith("Attr")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["class"] = df["class"].apply(lambda v: int(v.decode()) if isinstance(v, (bytes, bytearray)) else int(v))
    return df, file_map[horizon_months]


class Winsorizer(BaseEstimator, TransformerMixin):
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


def ks_from_scores(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece_score(y, p, bins=10):
    tmp = pd.DataFrame({"y":np.asarray(y), "p":np.asarray(p)})
    try:
        tmp["bin"] = pd.qcut(tmp["p"], q=bins, duplicates="drop")
    except Exception:
        return np.nan
    g = tmp.groupby("bin", observed=True).agg(n=("y","size"), obs=("y","mean"), pred=("p","mean"))
    return float(np.sum((g["n"]/len(tmp)) * np.abs(g["obs"]-g["pred"])))


def psi_score(expected, actual, bins=10):
    e = np.asarray(expected, float)
    a = np.asarray(actual, float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins+1)))
    if len(edges) < 3:
        return np.nan
    edges[0] = -np.inf
    edges[-1] = np.inf
    eh, _ = np.histogram(e, bins=edges)
    ah, _ = np.histogram(a, bins=edges)
    ep = np.clip(eh/eh.sum(), 1e-6, None)
    ap = np.clip(ah/ah.sum(), 1e-6, None)
    return float(np.sum((ap-ep)*np.log(ap/ep)))


def metric_status_auc(x):
    return "Verde" if x >= .75 else ("Amarelo" if x >= .65 else "Vermelho")


def metric_status_bss(x):
    return "Verde" if x >= .10 else ("Amarelo" if x > 0 else "Vermelho")


def metric_status_calratio(x):
    return "Verde" if .80 <= x <= 1.20 else ("Amarelo" if .60 <= x <= 1.40 else "Vermelho")


def metric_status_psi(x):
    if not np.isfinite(x): return "Amarelo"
    return "Verde" if x < .10 else ("Amarelo" if x < .25 else "Vermelho")


def metric_status_missing(max_missing):
    return "Verde" if max_missing < .10 else ("Amarelo" if max_missing < .30 else "Vermelho")


def health_summary(statuses):
    if "Vermelho" in statuses:
        return "REQUER ATENÇÃO", "Vermelho"
    if "Amarelo" in statuses:
        return "ADEQUADO COM MONITORAMENTO", "Amarelo"
    return "VALIDAÇÃO FORA DA AMOSTRA FAVORÁVEL", "Verde"


# ============================================================
# TREINO DOS MODELOS
# ============================================================

@st.cache_resource(show_spinner="Treinando e calibrando modelos de PD...")
def train_models(df, drop_attr37=True):
    features = [c for c in df.columns if c.startswith("Attr")]
    if drop_attr37 and "Attr37" in features:
        features.remove("Attr37")

    X = df[features].copy()
    X["MissingCount"] = X.isna().sum(axis=1).astype(float)
    features2 = list(X.columns)
    y = df["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.30, stratify=y, random_state=SEED
    )

    preprocess_basic = [
        ("winsor", Winsorizer(.005, .995)),
        ("imputer", SimpleImputer(strategy="median")),
    ]

    candidates = {
        "Regressão Logística": Pipeline(preprocess_basic + [
            ("scale", RobustScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced", C=0.6,
                solver="liblinear", random_state=SEED
            ))
        ]),
        "Random Forest": Pipeline(preprocess_basic + [
            ("model", RandomForestClassifier(
                n_estimators=320, max_depth=9, min_samples_leaf=5,
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=SEED
            ))
        ]),
        "Gradient Boosting": Pipeline(preprocess_basic + [
            ("model", HistGradientBoostingClassifier(
                max_iter=220, learning_rate=.06, max_leaf_nodes=25,
                min_samples_leaf=18, l2_regularization=.5,
                class_weight="balanced", random_state=SEED
            ))
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
        p_test = calibrated.predict_proba(X_test)[:,1]
        p_train = calibrated.predict_proba(X_train)[:,1]

        auc = roc_auc_score(y_test, p_test)
        ap = average_precision_score(y_test, p_test)
        brier = brier_score_loss(y_test, p_test)
        bss = 1 - brier/brier_naive
        ks = ks_from_scores(y_test, p_test)
        ece = ece_score(y_test, p_test)
        cal_ratio = float(np.mean(p_test) / max(base_rate_test, 1e-12))
        psi = psi_score(p_train, p_test)
        ll = log_loss(y_test, np.clip(p_test, 1e-8, 1-1e-8))

        models[name] = calibrated
        preds[name] = {"train":p_train, "test":p_test}
        rows.append({
            "Modelo": name,
            "ROC-AUC": auc,
            "Gini": 2*auc-1,
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

    eligible = metrics[metrics["ROC-AUC"] >= .65]
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


# ============================================================
# PORTFÓLIO ILUSTRATIVO DE CONTRAPARTES
# ============================================================

@st.cache_data
def make_demo_portfolio(X_test, y_test, pd_scores, n=N_DEMO_CP):
    rng = np.random.default_rng(SEED)
    n = min(n, len(X_test))
    chosen = X_test.sample(n=n, random_state=SEED).index
    pos = [X_test.index.get_loc(i) for i in chosen]
    pds = np.asarray(pd_scores)[pos]
    yobs = y_test.loc[chosen].values

    ead_gross = np.clip(rng.lognormal(mean=np.log(55e6), sigma=.65, size=n), 8e6, 300e6)
    collateral_pct = rng.uniform(0, .30, size=n)
    collateral = ead_gross * collateral_pct
    ead_net = np.maximum(ead_gross - collateral, 0)
    lgd = rng.uniform(.35, .60, size=n)
    sigma_exp = rng.uniform(.20, .40, size=n)
    pfe99 = ead_net * np.exp(-.5*sigma_exp**2 + sigma_exp*norm.ppf(.99))
    el = pds * lgd * ead_net

    out = pd.DataFrame({
        "Contraparte": [f"CP-{i+1:03d}" for i in range(n)],
        "row_index": chosen,
        "PD": pds,
        "Rating": [rating_from_pd(x) for x in pds],
        "LGD": lgd,
        "EAD Bruto": ead_gross,
        "Garantias": collateral,
        "EAD Líquido": ead_net,
        "PFE99": pfe99,
        "Expected Loss": el,
        "Default observado": yobs,
    })
    return out


def portfolio_concentration(port):
    w = port["EAD Líquido"] / max(port["EAD Líquido"].sum(), 1e-12)
    hhi = float(np.sum(w*w))
    top5 = float(port.nlargest(5, "EAD Líquido")["EAD Líquido"].sum() / max(port["EAD Líquido"].sum(),1e-12))
    return hhi, top5


# ============================================================
# SIMULAÇÃO EAD/PFE/CREDIT LOSS
# ============================================================

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


# ============================================================
# DADOS E TREINO
# ============================================================

st.sidebar.title("🏦 Counterparty Risk")

horizon = st.sidebar.selectbox(
    "Horizonte de PD",
    [12,24,36,48,60],
    format_func=lambda x: f"{x} meses",
    index=0
)

df, source_file = load_arff(horizon)

drop_attr37 = st.sidebar.toggle(
    "Excluir Attr37 (>40% missing no horizonte 12m)",
    value=True
)

T = train_models(df, drop_attr37=drop_attr37)

model_names = list(T["models"].keys())
suggested = T["suggested"]
model_choice = st.sidebar.selectbox(
    "Modelo de PD",
    model_names,
    index=model_names.index(suggested),
    help="O modelo sugerido prioriza menor Brier entre modelos com ROC-AUC ≥ 0,65."
)

model = T["models"][model_choice]
p_test = T["preds"][model_choice]["test"]
p_train = T["preds"][model_choice]["train"]
metrics_row = T["metrics"].loc[model_choice]

portfolio = make_demo_portfolio(T["X_test"], T["y_test"], p_test)
cp_options = portfolio["Contraparte"].tolist()
selected_cp = st.sidebar.selectbox("Contraparte demonstrativa", cp_options, index=0)
cp = portfolio.loc[portfolio["Contraparte"]==selected_cp].iloc[0]

page = st.sidebar.radio(
    "Navegação",
    [
        "Visão Executiva",
        "Carteira de Contrapartes",
        "Perfil da Contraparte",
        "EAD / PFE / Wrong-Way Risk",
        "Model Health",
        "Comparação de Modelos",
        "Dados & Qualidade",
        "Metodologia"
    ]
)

# ============================================================
# CABEÇALHO
# ============================================================

st.title("Motor de Risco de Contraparte")
st.caption(
    f"Protótipo de PD + exposição de crédito | Fonte: {source_file} | Horizonte selecionado: {horizon} meses. "
    "PD é estimada com dados reais do dataset UCI; LGD, EAD, PFE e carteira de contrapartes são ilustrativos."
)

# ============================================================
# HEALTH STATUS
# ============================================================

raw_features = [c for c in df.columns if c.startswith("Attr") and not (drop_attr37 and c=="Attr37")]
max_missing = float(df[raw_features].isna().mean().max())
health_parts = {
    "Discriminação (AUC)": metric_status_auc(metrics_row["ROC-AUC"]),
    "Brier Skill": metric_status_bss(metrics_row["Brier Skill"]),
    "Calibração média": metric_status_calratio(metrics_row["Calibração (Prev/Obs)"]),
    "Estabilidade (PSI proxy)": metric_status_psi(metrics_row["PSI treino-teste"]),
    "Qualidade de dados": metric_status_missing(max_missing),
}
health_text, health_color = health_summary(list(health_parts.values()))

# ============================================================
# 1) VISÃO EXECUTIVA
# ============================================================

if page == "Visão Executiva":
    st.subheader("Risco da contraparte + confiabilidade do modelo")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric(f"PD {horizon}m", pct(cp["PD"],2))
    c2.metric("Rating interno*", cp["Rating"])
    c3.metric("EAD líquido", brl(cp["EAD Líquido"]))
    c4.metric("PFE99 ilustrativo", brl(cp["PFE99"]))
    c5.metric("Perda esperada", brl_el(cp["Expected Loss"]))

    c6,c7,c8,c9 = st.columns(4)
    c6.metric("ROC-AUC", f"{metrics_row['ROC-AUC']:.3f}")
    c7.metric("Gini", f"{metrics_row['Gini']:.3f}")
    c8.metric("Brier", f"{metrics_row['Brier']:.4f}")
    c9.metric("Model Health", health_text)

    box_cls = "cp-bad" if health_color=="Vermelho" else ("cp-warn" if health_color=="Amarelo" else "cp-ok")
    st.markdown(
        f"""
<div class='{box_cls}'>
<b>Leitura executiva:</b><br>
A contraparte demonstrativa <b>{selected_cp}</b> possui PD estimada de <b>{pct(cp['PD'],2)}</b>.
O modelo selecionado é <b>{model_choice}</b> e o diagnóstico de validação fora da amostra é
<b>{health_text}</b>.<br><br>
<b>Importante:</b> esta base permite avaliar a qualidade da PD fora da amostra; afirmar que o modelo
“continua confiável” em produção exige monitoramento temporal com novas safras de dados.
</div>
""",
        unsafe_allow_html=True
    )

    hhi, top5 = portfolio_concentration(portfolio)
    a,b = st.columns([2,1])
    top = portfolio.nlargest(12,"Expected Loss").copy()
    fig = px.bar(
        top.sort_values("Expected Loss"), x="Expected Loss", y="Contraparte",
        orientation="h", title="Maiores perdas esperadas — carteira ilustrativa"
    )
    fig.update_xaxes(tickformat=".2s")
    a.plotly_chart(fig, use_container_width=True)

    health_df = pd.DataFrame({"Pilar":health_parts.keys(), "Status":health_parts.values()})
    health_df["Semáforo"] = health_df["Status"].map(color_status)
    b.dataframe(health_df[["Pilar","Semáforo","Status"]], hide_index=True, use_container_width=True)
    b.metric("Top 5 / EAD total", pct(top5))
    b.metric("HHI de concentração", f"{hhi:.3f}")
    st.caption("*Faixas de rating são exclusivamente ilustrativas e não correspondem a escalas de agência.")

# ============================================================
# 2) CARTEIRA DE CONTRAPARTES
# ============================================================

elif page == "Carteira de Contrapartes":
    st.subheader("Concentração, PD e perda esperada")
    hhi, top5 = portfolio_concentration(portfolio)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Contrapartes", f"{len(portfolio)}")
    c2.metric("EAD líquido total", brl(portfolio["EAD Líquido"].sum()))
    c3.metric("Perda esperada total", brl(portfolio["Expected Loss"].sum()))
    c4.metric("Top 5 / EAD", pct(top5))

    a,b = st.columns(2)
    fig = px.scatter(
        portfolio, x="EAD Líquido", y="PD", size="Expected Loss", color="Rating",
        hover_name="Contraparte", title="Mapa de risco: exposição × PD"
    )
    a.plotly_chart(fig, use_container_width=True)

    rating = portfolio.groupby("Rating", as_index=False).agg(
        Contrapartes=("Contraparte","count"),
        EAD=("EAD Líquido","sum"),
        EL=("Expected Loss","sum")
    )
    fig = px.bar(rating, x="Rating", y="EAD", text_auto=".2s", title="EAD por faixa de rating ilustrativa")
    b.plotly_chart(fig, use_container_width=True)

    show = portfolio.nlargest(25,"Expected Loss")[[
        "Contraparte","PD","Rating","LGD","EAD Líquido","PFE99","Expected Loss","Default observado"
    ]].copy()
    show["PD"] = show["PD"].map(lambda x:pct(x,2))
    show["LGD"] = show["LGD"].map(pct)
    for c in ["EAD Líquido","PFE99"]:
        show[c] = show[c].map(brl)
    show["Expected Loss"] = show["Expected Loss"].map(brl_el)
    show = show.rename(columns={"Expected Loss":"Perda esperada"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    st.info("EAD, LGD, garantias e PFE desta carteira são simulados para ensaio. A PD vem do modelo treinado no dataset.")

# ============================================================
# 3) PERFIL DA CONTRAPARTE
# ============================================================

elif page == "Perfil da Contraparte":
    st.subheader(f"Perfil — {selected_cp}")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("PD", pct(cp["PD"],2))
    c2.metric("Rating*", cp["Rating"])
    c3.metric("LGD ilustrativa", pct(cp["LGD"]))
    c4.metric("EAD líquido", brl(cp["EAD Líquido"]))
    c5.metric("Perda esperada", brl_el(cp["Expected Loss"]))

    row_idx = cp["row_index"]
    xrow = T["X_test"].loc[row_idx]

    # Indicadores com maior separação univariada robusta entre default/non-default
    raw = T["X_train"].copy()
    ytr = T["y_train"]
    scores = []
    for col in [c for c in raw.columns if c.startswith("Attr")]:
        a0 = raw.loc[ytr==0, col]
        a1 = raw.loc[ytr==1, col]
        iqr = np.nanquantile(raw[col],.75)-np.nanquantile(raw[col],.25)
        if not np.isfinite(iqr) or abs(iqr)<1e-12:
            continue
        s = abs(np.nanmedian(a1)-np.nanmedian(a0))/abs(iqr)
        scores.append((col,s))
    top_cols = [x[0] for x in sorted(scores,key=lambda z:z[1], reverse=True)[:10]]

    rows=[]
    for col in top_cols:
        train_col = raw[col].dropna()
        value = xrow[col]
        percentile = float((train_col <= value).mean()) if np.isfinite(value) and len(train_col) else np.nan
        rows.append({
            "Variável":col,
            "Descrição":ATTR.get(col,col),
            "Valor contraparte":value,
            "Percentil na amostra":percentile
        })
    prof = pd.DataFrame(rows)
    prof["Percentil na amostra"] = prof["Percentil na amostra"].map(lambda x:pct(x) if np.isfinite(x) else "N/D")
    st.dataframe(prof, hide_index=True, use_container_width=True)

    st.markdown(
        """
**Como ler:** estes são indicadores que, na amostra de treinamento, apresentam maior separação univariada
entre empresas default e não-default. Eles ajudam a contextualizar a contraparte, mas não são uma
explicação causal da PD nem substituem uma análise de crédito completa.
"""
    )
    st.caption("*Rating demonstrativo, construído apenas a partir de faixas de PD.")

# ============================================================
# 4) EAD / PFE / WWR
# ============================================================

elif page == "EAD / PFE / Wrong-Way Risk":
    st.subheader("Da PD à distribuição de perda de crédito")
    st.warning(
        "O dataset Polish Bankruptcy fornece informação para PD, mas não contém contratos de energia, EAD ou LGD. "
        "Nesta página, exposição, garantias, LGD e Wrong-Way Risk são simulados para demonstrar a arquitetura."
    )

    c1,c2,c3,c4 = st.columns(4)
    pd0 = c1.number_input("PD utilizada (%)", min_value=.01, max_value=99.0, value=float(cp["PD"]*100), step=.1)/100
    lgd = c2.number_input("LGD (%)", min_value=0.0, max_value=100.0, value=float(cp["LGD"]*100), step=1.0)/100
    ead_gross = c3.number_input("EAD bruto (R$ mi)", min_value=0.0, max_value=5000.0, value=float(cp["EAD Bruto"]/1e6), step=5.0)*1e6
    collateral = c4.number_input("Garantias (R$ mi)", min_value=0.0, max_value=5000.0, value=float(cp["Garantias"]/1e6), step=5.0)*1e6

    c5,c6 = st.columns(2)
    sigma_exp = c5.slider("Volatilidade da exposição futura", .05, 1.00, .30, .05)
    beta_wwr = c6.slider("Intensidade de Wrong-Way Risk", 0.0, 2.5, .70, .10,
                         help="0 = default independente da exposição simulada. Valores maiores aumentam a dependência entre PD e exposição, preservando aproximadamente a PD média-base.")

    ead_net = max(ead_gross-collateral,0)
    base = simulate_credit(pd0, lgd, ead_net, sigma_exp, 0.0, seed=101)
    wwr = simulate_credit(pd0, lgd, ead_net, sigma_exp, beta_wwr, seed=101)
    simple_el = pd0*lgd*ead_net

    st.caption(
        "Perda esperada simplificada = PD × LGD × EAD líquido, com "
        "EAD líquido = max(EAD bruto − Garantias, 0). "
        f"EAD líquido nesta configuração: {brl(ead_net)}."
    )

    a,b,c,d,e = st.columns(5)
    a.metric("Perda esperada simplificada", brl_el(simple_el))
    b.metric("PFE95", brl(wwr["PFE95"]))
    c.metric("PFE99", brl(wwr["PFE99"]))
    d.metric("Crédito VaR99", brl(wwr["CreditVaR99"]))
    e.metric("Crédito CVaR99", brl(wwr["CreditCVaR99"]))

    uplift = (wwr["CreditCVaR99"]/base["CreditCVaR99"]-1) if base["CreditCVaR99"]>0 else np.nan
    st.metric("Uplift de CVaR99 por Wrong-Way Risk", pct(uplift) if np.isfinite(uplift) else "N/D")

    a,b = st.columns(2)
    exp_sample = pd.DataFrame({
        "Exposição futura (R$ mi)": wwr["exposure"][:12000]/1e6,
        "Perda de crédito (R$ mi)": wwr["loss"][:12000]/1e6
    })
    fig = px.histogram(exp_sample, x="Exposição futura (R$ mi)", nbins=70, title="Distribuição simulada de exposição")
    a.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(exp_sample, x="Perda de crédito (R$ mi)", nbins=70, title="Distribuição simulada de perda")
    b.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**Fluxo conceitual:** PD estimada → cenários de exposição → EAD/PFE → default simulado → LGD → distribuição de perdas.
Quando o parâmetro de Wrong-Way Risk é positivo, cenários de maior exposição também recebem PD maior.
Na implantação real, a distribuição de EAD/PFE deve vir dos cenários do motor de mercado e dos contratos reais.
"""
    )

# ============================================================
# 5) MODEL HEALTH
# ============================================================

elif page == "Model Health":
    st.subheader("O modelo de PD é confiável?")
    st.caption(
        "Esta página mede validação fora da amostra. Monitoramento contínuo de produção exige novas safras temporais e comparação com defaults futuros."
    )

    box_cls = "cp-bad" if health_color=="Vermelho" else ("cp-warn" if health_color=="Amarelo" else "cp-ok")
    st.markdown(
        f"""
<div class='{box_cls}'>
<b>STATUS GERAL: {health_text}</b><br>
Modelo: <b>{model_choice}</b> | Teste holdout estratificado: {len(T['y_test'])} empresas.<br><br>
A conclusão deve ser entendida como <b>evidência de validação fora da amostra</b>, não como certificação para produção.
</div>
""",
        unsafe_allow_html=True
    )

    health_df = pd.DataFrame({"Pilar":health_parts.keys(), "Status":health_parts.values()})
    health_df["Semáforo"] = health_df["Status"].map(color_status)
    st.dataframe(health_df[["Pilar","Semáforo","Status"]], hide_index=True, use_container_width=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{metrics_row['ROC-AUC']:.3f}")
    c2.metric("Gini", f"{metrics_row['Gini']:.3f}")
    c3.metric("PR-AUC", f"{metrics_row['PR-AUC']:.3f}")
    c4.metric("Brier Skill", f"{metrics_row['Brier Skill']:.3f}")
    c5.metric("ECE", f"{metrics_row['ECE']:.3f}")

    a,b = st.columns(2)
    fpr,tpr,_ = roc_curve(T["y_test"], p_test)
    fig = go.Figure()
    fig.add_scatter(x=fpr,y=tpr,name=f"{model_choice} AUC={metrics_row['ROC-AUC']:.3f}")
    fig.add_scatter(x=[0,1],y=[0,1],mode="lines",name="Aleatório",line=dict(dash="dash"))
    fig.update_layout(title="ROC — discriminação", xaxis_title="Falso positivo", yaxis_title="Verdadeiro positivo")
    a.plotly_chart(fig,use_container_width=True)

    prob_true, prob_pred = calibration_curve(T["y_test"], p_test, n_bins=10, strategy="quantile")
    fig = go.Figure()
    fig.add_scatter(x=prob_pred,y=prob_true,mode="lines+markers",name=model_choice)
    fig.add_scatter(x=[0, max(.2,float(max(prob_pred.max(),prob_true.max())))],
                    y=[0, max(.2,float(max(prob_pred.max(),prob_true.max())))],
                    mode="lines",name="Calibração perfeita",line=dict(dash="dash"))
    fig.update_layout(title="Calibration plot", xaxis_title="PD prevista", yaxis_title="Default observado")
    b.plotly_chart(fig,use_container_width=True)

    # Decis de calibração
    cal = pd.DataFrame({"y":T["y_test"].values,"p":p_test})
    cal["Decil"] = pd.qcut(cal["p"],10,labels=False,duplicates="drop") + 1
    dec = cal.groupby("Decil",as_index=False).agg(N=("y","size"), PD_media=("p","mean"), Default_observado=("y","mean"))
    dec["PD_media"] = dec["PD_media"].map(lambda x:pct(x,2))
    dec["Default_observado"] = dec["Default_observado"].map(lambda x:pct(x,2))
    st.dataframe(dec,hide_index=True,use_container_width=True)

    threshold = st.slider("Threshold para classificação (não altera a PD)", .01, .50, .10, .01)
    pred_class = (p_test >= threshold).astype(int)
    cm = confusion_matrix(T["y_test"],pred_class)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Recall default", pct(recall_score(T["y_test"],pred_class,zero_division=0)))
    c2.metric("Precision default", pct(precision_score(T["y_test"],pred_class,zero_division=0)))
    c3.metric("F1", f"{f1_score(T['y_test'],pred_class,zero_division=0):.3f}")
    c4.metric("Accuracy", pct(accuracy_score(T["y_test"],pred_class)))
    st.write("Matriz de confusão:")
    st.dataframe(pd.DataFrame(cm,index=["Real 0","Real 1"],columns=["Prev 0","Prev 1"]),use_container_width=False)

    st.markdown(
        """
**Resposta à pergunta “o modelo ainda é confiável?”**

- Aqui medimos se o modelo **discrimina** default de não-default, se as probabilidades estão **calibradas** e se existe um sinal de **instabilidade de população** entre treino e teste.
- Como a divisão é aleatória/estratificada e não uma sequência de safras de produção, o PSI exibido é apenas um **proxy de estabilidade**, não monitoramento temporal real.
- Para produção, o dashboard deve receber mensalmente novas contrapartes, PDs emitidas e defaults observados para refazer calibração, AUC, Brier, PSI e vintage analysis.
"""
    )

# ============================================================
# 6) COMPARAÇÃO DE MODELOS
# ============================================================

elif page == "Comparação de Modelos":
    st.subheader("Logística × Random Forest × Gradient Boosting")
    comp = T["metrics"].copy().reset_index()
    show = comp.copy()
    for c in ["ROC-AUC","Gini","PR-AUC","KS","Brier","Brier Skill","ECE","Calibração (Prev/Obs)","PSI treino-teste","LogLoss"]:
        show[c] = show[c].map(lambda x:f"{x:.4f}")
    st.dataframe(show,hide_index=True,use_container_width=True)
    st.info(f"Candidato sugerido pelo critério demonstrativo (menor Brier entre AUC ≥ 0,65): {T['suggested']}")

    a,b = st.columns(2)
    fig = px.bar(comp,x="Modelo",y="ROC-AUC",text_auto=".3f",title="Discriminação — ROC-AUC")
    a.plotly_chart(fig,use_container_width=True)
    fig = px.bar(comp,x="Modelo",y="Brier",text_auto=".4f",title="Calibração probabilística — Brier (menor é melhor)")
    b.plotly_chart(fig,use_container_width=True)

    fig = go.Figure()
    for name in T["models"]:
        pt = T["preds"][name]["test"]
        qtrue,qpred = calibration_curve(T["y_test"],pt,n_bins=10,strategy="quantile")
        fig.add_scatter(x=qpred,y=qtrue,mode="lines+markers",name=name)
    fig.add_scatter(x=[0,.3],y=[0,.3],mode="lines",name="Perfeita",line=dict(dash="dash"))
    fig.update_layout(title="Comparação de calibração",xaxis_title="PD prevista",yaxis_title="Default observado")
    st.plotly_chart(fig,use_container_width=True)

# ============================================================
# 7) DADOS & QUALIDADE
# ============================================================

elif page == "Dados & Qualidade":
    st.subheader("Estrutura da base e qualidade de dados")
    attrs = [c for c in df.columns if c.startswith("Attr")]
    miss = df[attrs].isna().mean().sort_values(ascending=False)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Empresas", f"{len(df):,}".replace(",","."))
    c2.metric("Indicadores financeiros", f"{len(attrs)}")
    c3.metric("Taxa de default", pct(df["class"].mean(),2))
    c4.metric("Missing total", pct(df[attrs].isna().sum().sum()/(len(df)*len(attrs)),2))

    a,b = st.columns(2)
    target = df["class"].value_counts().rename(index={0:"Não-default",1:"Default"}).reset_index()
    target.columns=["Classe","N"]
    fig=px.bar(target,x="Classe",y="N",text_auto=True,title="Desbalanceamento do alvo")
    a.plotly_chart(fig,use_container_width=True)

    topmiss = miss.head(15).reset_index(); topmiss.columns=["Variável","Missing"]
    topmiss["Descrição"] = topmiss["Variável"].map(ATTR)
    fig=px.bar(topmiss.sort_values("Missing"),x="Missing",y="Variável",orientation="h",hover_data=["Descrição"],title="Top 15 variáveis com missing")
    b.plotly_chart(fig,use_container_width=True)

    st.markdown(
        f"""
**Horizonte selecionado:** {horizon} meses, carregado do arquivo **{source_file}**.

No dataset UCI, a nomenclatura dos arquivos é contraintuitiva: **5year.arff** representa a observação mais próxima do default e é usada aqui como horizonte aproximado de 12 meses; **1year.arff** corresponde ao horizonte mais distante.

**Tratamento do protótipo:** winsorização 0,5%–99,5% aprendida apenas no treino, imputação pela mediana do treino, indicador `MissingCount`, class weights e calibração sigmoid por cross-validation.
"""
    )

    if "Attr37" in df.columns:
        st.metric("Missing em Attr37", pct(df["Attr37"].isna().mean(),1))
        st.caption("O toggle da barra lateral permite excluir Attr37 antes do treinamento.")

    with st.expander("Dicionário dos 64 indicadores"):
        dct = pd.DataFrame({"Variável":list(ATTR.keys()),"Descrição":list(ATTR.values())})
        st.dataframe(dct,hide_index=True,use_container_width=True)

# ============================================================
# 8) METODOLOGIA
# ============================================================

elif page == "Metodologia":
    st.subheader("Arquitetura do motor de risco de contraparte")
    st.markdown(
        """
### 1. PD — Probability of Default

Indicadores financeiros → tratamento de dados → modelo → calibração → **PD**.

Foram implementados três candidatos:
- Regressão Logística;
- Random Forest;
- Gradient Boosting.

A avaliação fora da amostra usa **ROC-AUC, Gini, PR-AUC, KS, Brier, Brier Skill, ECE, calibration plot e PSI proxy**.

### 2. LGD — Loss Given Default

A base UCI não contém recuperação pós-default. Por isso, LGD é **parâmetro ilustrativo** neste protótipo.

Nesta versão, a convenção usada é:
`EAD líquido = max(EAD bruto − garantias, 0)` e depois `Perda = LGD × EAD líquido`.
Em uma implantação real, deve-se evitar dupla contagem caso a LGD já incorpore a recuperação das mesmas garantias.

### 3. EAD / PFE

A base UCI também não contém contratos nem Mark-to-Market. O dashboard simula uma distribuição de exposição para demonstrar:

`EAD atual → exposição futura → PFE95/PFE99`.

Na versão integrada, EAD/PFE deve vir dos cenários do **Motor de Risco de Mercado** e dos contratos reais.

### 4. Expected Loss

`EL = PD × LGD × EAD`

Quando há Wrong-Way Risk, PD e exposição passam a variar conjuntamente por cenário e a perda deve ser simulada, não apenas multiplicada de forma estática. Nesta versão, a PD cenário a cenário é recentrada para preservar aproximadamente a PD média-base, isolando melhor o efeito de dependência.

### 5. Model Health

O dashboard separa duas perguntas:

**Qual é o risco da contraparte?** → PD, LGD, EAD, PFE, perda esperada e Crédito VaR/CVaR.

**Quanto posso confiar na PD?** → discriminação, calibração, estabilidade e qualidade dos dados.

### Limitações essenciais

- Dataset polonês e não específico do mercado brasileiro de energia;
- Bankruptcy não é exatamente o mesmo evento de default contratual de uma comercializadora;
- Split holdout não substitui validação temporal/out-of-time;
- LGD, EAD, garantias, PFE, ratings e portfólio são demonstrativos;
- Produção exige dados reais de contrapartes, contratos, grupos econômicos, garantias, netting e histórico de defaults/recuperações;
- Os thresholds de semáforo de Model Health são heurísticas demonstrativas, não limites regulatórios.

### Próxima integração

`Motor de Mercado → cenários A+1...A+5 → MTM por contrato → EAD/PFE → PD/LGD → Credit Loss → Wrong-Way Risk → risco integrado.`
"""
    )
