# Análise de Churn em Telecomunicações

Atividade prática da disciplina SISB110 - Ciência dos Dados e Inteligência Empresarial (UFAL).

Análise descritiva e inferencial da base Telco Customer Churn, para entender quais clientes cancelam o serviço e o que pode ser feito para reduzir os cancelamentos.

## Arquivos

- `analise_churn.py`: código da análise (limpeza, estatística descritiva, gráficos e testes de hipóteses)
- `Telco-Customer-Churn.csv`: base de dados usada

## Base de dados

Telco Customer Churn (IBM), disponível em: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Como rodar

1. Instalar as bibliotecas:

```
pip install pandas numpy matplotlib seaborn scipy
```

2. Deixar o arquivo `Telco-Customer-Churn.csv` na mesma pasta do código.

3. Rodar:

```
python analise_churn.py
```

O código mostra os resultados no terminal e salva os cinco gráficos na pasta `figuras`.

## Autor

José Pedro de Lira Barbosa
