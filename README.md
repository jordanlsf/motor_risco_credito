# Motor Quantitativo de Risco de Crédito e Contraparte

### Credit Risk Engine | PD · EAD · LGD

Motor quantitativo desenvolvido em Python para análise de risco de crédito bancário e contraparte, integrando modelagem estatística, ciência de dados e simulação de cenários.

## Objetivo

Estimar a perda esperada a partir da integração dos componentes:

- **PD (Probability of Default):** probabilidade de inadimplência.
- **EAD (Exposure at Default):** exposição financeira no momento do default.
- **LGD (Loss Given Default):** percentual de perda após a inadimplência, considerando possíveis recuperações.

### Modelo de Perda Esperada

**EL = PD × EAD × LGD**

## ## Metodologia e Aplicabilidade

O motor foi desenvolvido a partir da integração de três componentes fundamentais da modelagem quantitativa de risco de crédito: PD (Probability of Default), EAD (Exposure at Default) e LGD (Loss Given Default).

O componente de PD utiliza dados públicos do *UCI Machine Learning Repository* para experimentação e desenvolvimento estatístico. Os componentes de EAD e LGD são explorados inicialmente por meio de hipóteses e cenários simulados.

Sua arquitetura modular permite adaptar as metodologias a diferentes segmentos econômicos, destacando-se:

- **Instituições financeiras:** avaliação do risco de crédito de clientes e empresas, estimativa de perdas esperadas e suporte à gestão de carteiras.
- **Setor elétrico:** avaliação de contrapartes em contratos de comercialização de energia e monitoramento de exposições financeiras.
- **Empresas não financeiras:** análise do risco de clientes corporativos, concentração de recebíveis e definição de políticas de crédito.
- **Fundos e instituições de investimento:** avaliação de exposição a emissores e contrapartes, análise de cenários e mensuração de perdas potenciais.

O objetivo é demonstrar como a integração entre Estatística, Ciência de Dados e modelagem financeira pode transformar informações em instrumentos quantitativos de suporte à tomada de decisão, independentemente do setor de aplicação.

*O projeto possui caráter experimental. Sua utilização em ambientes reais exige calibração, validação e adequação às características da carteira e aos requisitos regulatórios aplicáveis.*

## Tecnologias e Competências

- **Linguagem:** Python
- **Métodos quantitativos:** Estatística e Machine Learning
- **Modelagem financeira:** PD, EAD, LGD e Expected Loss
- **Aplicações:** Modelagem de Risco, Simulação e Ciência de Dados

