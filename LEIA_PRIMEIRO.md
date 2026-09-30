# Etapa 2 — Motor de Risco de Crédito V3.1

Esta é uma **reorganização técnica inicial**, não uma versão revalidada do modelo. O aplicativo `app.py` foi extraído literalmente da string `app_code` do arquivo original `risco_de_contraparte_v3.py`; a única modificação funcional foi trocar a pasta fixa `/content/polish_bankruptcy` por `data/polish_bankruptcy` junto ao app. A separação dos módulos PD, EAD e LGD acontecerá na próxima etapa, depois de confirmar esta versão.

## Arquivos
- `app.py`: dashboard Streamlit e cálculos V3 existentes.
- `prepare_data.py`: prepara os cinco ARFF a partir do ZIP oficial da UCI, via download ou ZIP local.
- `requirements.txt`: dependências declaradas no código original.
- `.gitignore`: impede envio acidental dos dados extraídos e ambientes locais.

## Execução no computador ou em ambiente Python/Colab
```bash
python -m pip install -r requirements.txt
python prepare_data.py
# ou: python prepare_data.py --zip /caminho/do/zip_original.zip
python -m streamlit run app.py
```

O comando de execução do Streamlit requer um ambiente capaz de expor uma interface web. Esta etapa **não publica uma URL pública automaticamente**.

## Entradas originais para reprodução
1. `risco_de_contraparte_v3.py` — origem da extração do aplicativo;
2. ZIP original *Polish Companies Bankruptcy*, UCI dataset 365, contendo `1year.arff` a `5year.arff` — entrada do motor.

## Antes de publicar no GitHub
- Não substitua seu backup V3; envie estes arquivos primeiro.
- Não publique dados de contrapartes reais, credenciais nem o ZIP se não for necessário.
- Ainda precisamos revisar seleção de modelo, validação temporal e interpretação de PD/EAD/LGD.
- Para reproduzir esta etapa no Google Colab, envie o **arquivo original V3** e rode `build_portfolio_v31.py` (script gerador entregue separadamente); para rodar a aplicação, obtenha novamente o ZIP UCI ou envie-o ao Colab.
