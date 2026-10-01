# Motor Quantitativo de Risco de Crédito e Contraparte

## Credit Risk Engine | PD · EAD · LGD

Motor quantitativo desenvolvido em Python para análise de risco de crédito bancário e de contraparte, integrando modelagem estatística, ciência de dados, simulação de cenários e visualização interativa em Streamlit.

O projeto foi estruturado para demonstrar uma arquitetura quantitativa aplicável a diferentes contextos de risco de crédito, incluindo instituições financeiras, empresas não financeiras, investimentos e gestão de contrapartes no setor de energia.

---

## Objetivo

Estimar e analisar o risco de crédito a partir da integração dos principais componentes:

- **PD (Probability of Default):** probabilidade de inadimplência ou falência.
- **EAD (Exposure at Default):** exposição financeira no momento do default.
- **LGD (Loss Given Default):** percentual de perda após o default, considerando possíveis recuperações.

A integração desses componentes permite estimar a perda esperada:

**EL = PD × EAD × LGD**

Além da perda esperada, o motor incorpora análises de exposição futura, risco de cauda e dependência entre exposição e deterioração de crédito.

---

## Metodologia e Aplicabilidade

O componente de PD utiliza dados públicos do **Polish Companies Bankruptcy Dataset**, disponibilizado pelo UCI Machine Learning Repository.

O evento observado na base é falência empresarial, utilizado neste projeto como aproximação experimental de default.

Foram implementados três modelos candidatos de PD:

- Regressão Logística;
- Random Forest;
- Gradient Boosting.

O processo inclui:

- tratamento de valores extremos;
- imputação de valores ausentes;
- tratamento do desbalanceamento das classes;
- calibração probabilística;
- comparação de desempenho;
- análise de qualidade do modelo.

Os componentes de EAD, LGD, garantias, PFE e carteira de contrapartes são simulados neste protótipo, com o objetivo de demonstrar a arquitetura integrada de mensuração de risco.

A estrutura possui aplicações potenciais em:

- instituições financeiras;
- análise de crédito corporativo;
- gestão de risco de contraparte;
- fundos e investimentos;
- empresas não financeiras;
- comercialização e gestão de energia.

---

## Arquitetura do Motor

```text
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
```

---

## Modelos de PD

O módulo de PD compara três abordagens:

### Regressão Logística

Modelo linear probabilístico utilizado como referência interpretável.

### Random Forest

Modelo baseado em conjunto de árvores de decisão, capaz de representar relações não lineares e interações entre variáveis.

### Gradient Boosting

Modelo baseado em aprendizado sequencial de árvores, utilizado para capturar relações complexas entre os indicadores financeiros e o risco de default.

As probabilidades são posteriormente calibradas para melhorar a qualidade da estimativa de PD.

---

## Métricas de Avaliação

O motor utiliza diferentes métricas para avaliar discriminação, calibração e estabilidade:

- ROC-AUC;
- Gini;
- PR-AUC;
- KS;
- Brier Score;
- Brier Skill Score;
- Expected Calibration Error (ECE);
- LogLoss;
- relação entre PD prevista e default observado;
- PSI treino-teste como proxy experimental de estabilidade.

O PSI apresentado neste projeto não representa monitoramento temporal de produção, pois treino e teste são derivados da mesma base histórica.

---

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

- **ROC-AUC:** 0,9617
- **Gini:** 0,9235
- **PR-AUC:** 0,7796
- **KS:** 0,8141
- **Brier Score:** 0,0300
- **Brier Skill:** 0,5358
- **ECE:** 0,0096
- **Calibração média Prev/Obs:** 1,0546
- **LogLoss:** 0,1084

Os resultados representam uma avaliação experimental em holdout aleatório estratificado.

Não constituem validação temporal (*out-of-time*), validação regulatória ou aprovação para utilização em produção.

---

## Teste de Wrong-Way Risk

Foi realizado um teste de consistência com os seguintes parâmetros:

- PD = 5%
- LGD = 45%
- EAD = R$ 50 milhões
- volatilidade da exposição = 30%
- intensidade de Wrong-Way Risk = 0,70

Sem Wrong-Way Risk, a perda esperada teórica é:

**EL = PD × LGD × EAD**

**EL = R$ 1,125 milhão**

A perda esperada simulada sem Wrong-Way Risk ficou próxima desse valor, em aproximadamente R$ 1,128 milhão.

No cenário com dependência positiva entre exposição e risco de default, o motor apresentou aproximadamente:

- **21,7% de aumento na Expected Loss**
- **20,1% de aumento no Credit CVaR 99%**

A frequência média de default permaneceu próxima de 5%.

Esse teste demonstra que a perda pode aumentar mesmo sem alteração significativa da PD média quando os eventos de default passam a ocorrer com maior frequência nos cenários de maior exposição.

Os resultados acima são específicos deste cenário simulado e não representam parâmetros universais de Wrong-Way Risk.

---

## Visão Executiva 1

![Visão Executiva](assets/visao_executiva1.jpeg)

---

## Visao Executiva 2

![Comparação dos Modelos de PD](assets/visao_executiva2.jpeg)

---

## Carteira de contrapartes

![Simulação de Wrong-Way Risk](assets/carteira_contraparte1.jpeg)

---



---

## Como executar

### Opção 1 — Execução local

#### 1. Clonar o repositório

```bash
git clone https://github.com/jordanlsf/motor_risco_credito.git
cd motor_risco_credito
```

#### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

#### 3. Preparar os dados

```bash
python prepare_data.py
```

O script prepara os arquivos `.arff` utilizados nos diferentes horizontes experimentais de PD.

#### 4. Executar o dashboard

```bash
streamlit run app.py
```

Após a inicialização, o Streamlit deverá disponibilizar a aplicação localmente, normalmente em:

```text
http://localhost:8501
```

---

### Opção 2 — Execução no Google Colab

#### 1. Clonar o repositório

```python
!git clone https://github.com/jordanlsf/motor_risco_credito.git
%cd motor_risco_credito
```

#### 2. Instalar as dependências

```python
!pip install -r requirements.txt
```

#### 3. Preparar os dados

```python
!python prepare_data.py
```

#### 4. Testar os principais módulos

```python
from src.data_loader import load_arff
from src.pd_model import train_models
from src.credit_simulation import simulate_credit

df, source_file = load_arff(12)
T = train_models(df, drop_attr37=True)

display(T["metrics"])
print("Modelo sugerido:", T["suggested"])
```

#### 5. Iniciar o Streamlit no Colab

```python
import subprocess
import time

log = open("/content/streamlit.log", "w")

streamlit_process = subprocess.Popen(
    [
        "streamlit", "run", "app.py",
        "--server.headless=true",
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ],
    stdout=log,
    stderr=subprocess.STDOUT
)

time.sleep(8)

print("Processo:", streamlit_process.poll())
```

#### 6. Verificar o status do servidor

```python
import urllib.request

url = "http://127.0.0.1:8501/_stcore/health"

with urllib.request.urlopen(url, timeout=5) as response:
    print("Status:", response.status)
    print("Resposta:", response.read().decode())
```

O resultado esperado é:

```text
Status: 200
Resposta: ok
```

#### 7. Abrir o dashboard pelo proxy do Colab

```python
from google.colab.output import eval_js

url = eval_js("google.colab.kernel.proxyPort(8501)")
print("Dashboard:", url)
```

Abra o endereço exibido pelo Colab em uma nova aba.

> **Observação sobre segurança:** as opções de CORS e XSRF utilizadas acima destinam-se exclusivamente à execução experimental e temporária no Google Colab com dados públicos ou simulados. Não representam configuração recomendada para aplicações em produção.

---

## Estrutura do Repositório

```text
motor_risco_credito/
│
├── assets/
│   ├── dashboard_visao_executiva.png
│   ├── dashboard_comparacao_modelos.png
│   ├── dashboard_wwr.png
│   └── dashboard_model_health.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── pd_model.py
│   ├── risk_metrics.py
│   └── credit_simulation.py
│
├── app.py
├── prepare_data.py
├── requirements.txt
├── risco_de_contraparte_v3.py
└── README.md
```

---

## Limitações

Este projeto possui caráter experimental e de portfólio.

As principais limitações são:

- o dataset utilizado contém empresas polonesas e não representa diretamente carteiras brasileiras;
- falência empresarial não é exatamente equivalente a todos os conceitos possíveis de default bancário ou contratual;
- a validação atual utiliza holdout aleatório estratificado e não validação temporal independente;
- EAD, LGD, garantias, ratings, carteira e PFE são simulados;
- os thresholds utilizados no Model Health são heurísticas demonstrativas;
- a implementação não representa modelo regulatório homologado;
- a utilização em produção exigiria dados reais, governança, monitoramento, validação independente, segurança e adequação às normas aplicáveis.

---

## Tecnologias

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Streamlit
- Plotly
- Estatística
- Machine Learning
- Monte Carlo
- Credit Risk Modeling
- Data Analytics

---

## Finalidade

Este projeto foi desenvolvido como demonstração prática de integração entre:

- Estatística;
- Ciência de Dados;
- Machine Learning;
- Risco de Crédito;
- Simulação;
- Engenharia de Software;
- Visualização de Dados.

O objetivo é demonstrar como técnicas quantitativas podem ser transformadas em uma ferramenta estruturada de apoio à análise de risco de crédito e contraparte.

> **Importante:** este projeto não deve ser utilizado para decisões reais de crédito sem calibração específica, validação independente e adequação regulatória.
