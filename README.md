## Status de Validação da V3.1

A versão V3.1 foi validada funcionalmente em ambiente Google Colab, com execução a partir do repositório GitHub.

Foram testados com sucesso:

- carregamento dos dados públicos da UCI;
- treinamento dos modelos de PD;
- comparação entre Regressão Logística, Random Forest e Gradient Boosting;
- métricas de discriminação e calibração;
- simulação de EAD e PFE;
- Expected Loss;
- Credit VaR e CVaR;
- Wrong-Way Risk;
- dashboard Streamlit;
- navegação completa entre as páginas do aplicativo.

No horizonte experimental de 12 meses, o Gradient Boosting apresentou o melhor desempenho no holdout estratificado utilizado no protótipo, com:

- ROC-AUC: 0,9617
- Gini: 0,9235
- PR-AUC: 0,7796
- KS: 0,8141
- Brier Score: 0,0300
- ECE: 0,0096

Os resultados devem ser interpretados como evidência experimental fora da amostra e não como validação regulatória ou aprovação para uso em produção.
## Teste de Wrong-Way Risk

Foi realizado um teste de consistência com:

- PD = 5%
- LGD = 45%
- EAD = R$ 50 milhões
- volatilidade da exposição = 30%

Sem Wrong-Way Risk, a perda esperada simulada ficou próxima da perda esperada teórica:

EL = PD × LGD × EAD = R$ 1,125 milhão

Com dependência positiva entre exposição e risco de default, o motor apresentou aumento aproximado de:

- 21,7% na Expected Loss;
- 20,1% no Credit CVaR 99%.

O teste demonstra como a dependência entre exposição e deterioração de crédito pode aumentar as perdas mesmo quando a PD média permanece praticamente estável.

Dados UCI
   ↓
Data Loader
   ↓
Modelos de PD
   ↓
Calibração e Model Health
   ↓
PD da contraparte
   ↓
EAD / LGD / PFE
   ↓
Monte Carlo
   ↓
Wrong-Way Risk
   ↓
Expected Loss / VaR / CVaR
   ↓
Dashboard Streamlit

![Visão Executiva](assets/dashboard_visao_executiva.png)

