# SISB110 - Atividade pratica: analise de churn (Telco Customer Churn)
# Para rodar: python analise_churn.py
# Precisa do arquivo Telco-Customer-Churn.csv na mesma pasta

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

os.makedirs("figuras", exist_ok=True)
sns.set_style("whitegrid")
cores = {"No": "#2E86AB", "Yes": "#D1495B"}

# ---------- 1. carregando e limpando os dados ----------
df = pd.read_csv("Telco-Customer-Churn.csv")
print("tamanho inicial:", df.shape)
print(df.dtypes)

# TotalCharges veio como texto, tem que converter
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("\nvalores ausentes:")
print(df.isna().sum()[df.isna().sum() > 0])

# os ausentes sao clientes com tenure 0 (ainda nao tem fatura), entao tirei essas linhas
print("ausentes com tenure 0:", df[df["TotalCharges"].isna()]["tenure"].eq(0).sum())
df = df.dropna(subset=["TotalCharges"])

# duplicatas
print("ids repetidos:", df["customerID"].duplicated().sum())
print("linhas iguais sem contar o id:", df.drop(columns="customerID").duplicated().sum())
df = df.drop(columns="customerID")  # o id nao serve pra analise

df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
print("tamanho depois da limpeza:", df.shape)

# outliers pelo IQR
numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in numericas:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    inf = q1 - 1.5 * iqr
    sup = q3 + 1.5 * iqr
    qtd = ((df[col] < inf) | (df[col] > sup)).sum()
    print(col, "limites:", round(inf, 2), round(sup, 2), "outliers:", qtd)
    df[col] = df[col].clip(inf, sup)  # se tivesse outlier seria limitado aqui

# ---------- 2. estatistica descritiva ----------
desc = pd.DataFrame({
    "media": df[numericas].mean(),
    "mediana": df[numericas].median(),
    "desvio": df[numericas].std(),
    "iqr": df[numericas].quantile(0.75) - df[numericas].quantile(0.25),
    "assimetria": df[numericas].skew(),
    "minimo": df[numericas].min(),
    "maximo": df[numericas].max(),
}).round(2)
print("\n", desc)

categoricas = ["Churn", "Contract", "InternetService", "PaymentMethod",
               "TechSupport", "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"]
for col in categoricas:
    tabela = pd.DataFrame({
        "absoluta": df[col].value_counts(),
        "relativa %": (df[col].value_counts(normalize=True) * 100).round(2),
    })
    print("\n", col)
    print(tabela)

print("\nmedias por churn:")
print(df.groupby("Churn")[numericas].agg(["mean", "median"]).round(2))

# ---------- 3. graficos ----------
# grafico 1: histograma da mensalidade por churn
plt.figure(figsize=(8, 4.8))
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True, stat="density",
             common_norm=False, palette=cores, bins=30, alpha=0.45)
plt.title("Distribuição da mensalidade por situação de Churn")
plt.xlabel("Mensalidade (US$)")
plt.ylabel("Densidade")
plt.tight_layout()
plt.savefig("figuras/g1_hist_monthly.png", dpi=170)
plt.close()

# grafico 2: boxplot do tenure por churn
plt.figure(figsize=(6.5, 4.8))
sns.boxplot(data=df, x="Churn", y="tenure", hue="Churn", palette=cores, legend=False)
plt.title("Tempo de contrato (tenure) por Churn")
plt.xlabel("Churn")
plt.ylabel("Meses de contrato")
plt.tight_layout()
plt.savefig("figuras/g2_box_tenure.png", dpi=170)
plt.close()

# grafico 3: barras empilhadas, proporcao de churn por contrato
prop = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
prop = prop.loc[["Month-to-month", "One year", "Two year"]]
ax = prop[["No", "Yes"]].plot(kind="bar", stacked=True, figsize=(7, 4.8),
                              color=[cores["No"], cores["Yes"]], width=0.6)
for i in range(3):
    sim = prop["Yes"].iloc[i]
    nao = prop["No"].iloc[i]
    if sim >= 8:
        ax.text(i, nao + sim / 2, f"{sim:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    else:
        ax.text(i, nao - 4, f"Churn: {sim:.1f}%", ha="center", va="center", color="white", fontweight="bold")
plt.title("Proporção de Churn por tipo de contrato")
plt.xlabel("Tipo de contrato")
plt.ylabel("% de clientes")
plt.xticks(rotation=0)
plt.legend(title="Churn", loc="lower right")
plt.tight_layout()
plt.savefig("figuras/g3_barras_contract.png", dpi=170)
plt.close()

# grafico 4: dispersao tenure x TotalCharges
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="tenure", y="TotalCharges", hue="Churn", palette=cores,
                alpha=0.45, s=18, edgecolor=None)
plt.title("Tenure vs. TotalCharges por Churn")
plt.xlabel("Meses de contrato")
plt.ylabel("Cobrança total (US$)")
plt.tight_layout()
plt.savefig("figuras/g4_scatter.png", dpi=170)
plt.close()

# grafico 5: correlacao de pearson (churn vira 0 e 1 so pra poder entrar na matriz)
df["churn_01"] = (df["Churn"] == "Yes").astype(int)
corr = df[numericas + ["churn_01"]].corr()
corr.index = corr.columns = ["tenure", "MonthlyCharges", "TotalCharges", "Churn (0/1)"]
plt.figure(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title("Matriz de correlação de Pearson")
plt.tight_layout()
plt.savefig("figuras/g5_heatmap.png", dpi=170)
plt.close()

# ---------- 4. inferencia ----------
# intervalo de confianca de 95% para a media da mensalidade
x = df["MonthlyCharges"]
n = len(x)
media = x.mean()
erro = stats.sem(x)
ic = stats.t.interval(0.95, df=n - 1, loc=media, scale=erro)
print("\nIC 95% media MonthlyCharges:", round(media, 2), "de", round(float(ic[0]), 2), "ate", round(float(ic[1]), 2))

# intervalo para a proporcao de churn (aproximacao normal)
k = df["churn_01"].sum()
p = k / n
ep = np.sqrt(p * (1 - p) / n)
print("proporcao de churn:", round(p, 4), "IC 95%:", round(p - 1.96 * ep, 4), round(p + 1.96 * ep, 4))

# teste t de welch: mensalidade de quem saiu x quem ficou
# H0: as medias sao iguais / H1: as medias sao diferentes / alfa = 0,05
saiu = df[df["Churn"] == "Yes"]["MonthlyCharges"]
ficou = df[df["Churn"] == "No"]["MonthlyCharges"]
print("\nlevene (variancias iguais?):", stats.levene(saiu, ficou))  # deu diferente, por isso welch
t, pvalor = stats.ttest_ind(saiu, ficou, equal_var=False)
print("welch: t =", round(t, 3), "p =", pvalor)
print("media saiu:", round(saiu.mean(), 2), "media ficou:", round(ficou.mean(), 2),
      "diferenca:", round(saiu.mean() - ficou.mean(), 2))
if pvalor < 0.05:
    print("rejeita H0")
else:
    print("nao rejeita H0")
print("conferindo com mann-whitney:", stats.mannwhitneyu(saiu, ficou).pvalue)

# tenure x churn
t2, p2 = stats.ttest_ind(df[df["Churn"] == "Yes"]["tenure"], df[df["Churn"] == "No"]["tenure"], equal_var=False)
r, pr = stats.pearsonr(df["tenure"], df["churn_01"])
print("\ntenure: welch t =", round(t2, 2), "p =", p2, "| correlacao com churn r =", round(r, 3))

# qui-quadrado: contrato x churn
# H0: contrato e churn sao independentes / H1: existe associacao
tab = pd.crosstab(df["Contract"], df["Churn"])
print("\n", tab)
chi2, p3, gl, esperado = stats.chi2_contingency(tab)
v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
print("qui-quadrado =", round(chi2, 1), "gl =", gl, "p =", p3, "V de Cramer =", round(v, 3))

# mesma coisa para os servicos e forma de pagamento
for col in ["InternetService", "TechSupport", "OnlineSecurity", "PaymentMethod"]:
    sub = df
    if col in ["TechSupport", "OnlineSecurity"]:
        sub = df[df[col] != "No internet service"]  # so quem tem internet
    tab = pd.crosstab(sub[col], sub["Churn"])
    chi2, p4, gl, esperado = stats.chi2_contingency(tab)
    print("\n", col, "- taxa de churn (%):")
    print((tab["Yes"] / tab.sum(axis=1) * 100).round(2))
    print("qui-quadrado =", round(chi2, 1), "gl =", gl, "p =", p4)

print("\nreceita mensal de quem saiu:", round(saiu.sum(), 2))
print("% dos cancelamentos no contrato mensal:",
      round(((df["Churn"] == "Yes") & (df["Contract"] == "Month-to-month")).sum() / k * 100, 2))
