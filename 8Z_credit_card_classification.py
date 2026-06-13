"""
Pipeline:
  1. Wczytanie i wstepna analiza danych
  2. Identyfikacja wartosci odstajacych (metoda IQR)
  3. Usuniecie wartosci odstajacych i ocena wplywu
  4. Statystyki podsumowujace i wizualizacje na oczyszczonym zbiorze
  5. Skalowanie danych (StandardScaler)
  6. Zbalansowanie klas (SMOTE)
  7. Model poczatkowy (Random Forest) + waznosc cech
  8. Wybor 10 najwazniejszych atrybutow
  9. Dostrajanie hiperparametrow (GridSearchCV) - patrz grid_tune.py
 10. Trenowanie modelu ostatecznego i ocena - patrz grid_tune.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, classification_report)
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")
OUT = "/sessions/optimistic-nifty-lovelace/mnt/outputs"
DATA = "/sessions/optimistic-nifty-lovelace/mnt/uploads/dataset.csv"

results = {}

# 1. WCZYTANIE I WSTEPNA ANALIZA
df = pd.read_csv(DATA)
print("1. WCZYTANIE DANYCH")
print("Ksztalt zbioru:", df.shape)
print("Braki danych:", int(df.isnull().sum().sum()))
print("Duplikaty ID:", int(df["ID"].duplicated().sum()))
target_counts = df["Target"].value_counts().sort_index()
print("Rozklad Target:", dict(target_counts), "udzial klasy 1: %.2f%%" % (100 * df["Target"].mean()))
results["n_rows_orig"] = int(df.shape[0])
results["n_cols"] = int(df.shape[1])
results["target0_orig"] = int(target_counts[0])
results["target1_orig"] = int(target_counts[1])
results["target1_pct_orig"] = round(100 * df["Target"].mean(), 2)

df = df.drop(columns=["ID"])
cat_cols = ["Income_type", "Education_type", "Family_status", "Housing_type", "Occupation_type"]
cont_cols = ["Num_children", "Num_family", "Account_length", "Total_income", "Age", "Years_employed"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(["Odrzucony (0)", "Zatwierdzony (1)"], target_counts.values, color=["#4C72B0", "#C44E52"])
for b, v in zip(bars, target_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 80, str(v), ha="center", fontweight="bold")
ax.set_ylabel("Liczba wnioskow")
ax.set_title("Rozklad zmiennej docelowej Target")
plt.tight_layout()
plt.savefig(OUT + "/fig1_target_dist.png", dpi=130)
plt.close()

# 2. IDENTYFIKACJA WARTOSCI ODSTAJACYCH
print("\n2. IDENTYFIKACJA WARTOSCI ODSTAJACYCH (IQR)")
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
outlier_info = {}
for ax, col in zip(axes.ravel(), cont_cols):
    sns.boxplot(y=df[col], ax=ax, color="#4C72B0")
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = int(((df[col] < low) | (df[col] > high)).sum())
    outlier_info[col] = (round(low, 2), round(high, 2), n_out)
    ax.set_title(col + "\n(odstajace: " + str(n_out) + ")")
    ax.set_ylabel("")
plt.suptitle("Wartosci odstajace w cechach ciaglych (przed czyszczeniem)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "/fig2_boxplots.png", dpi=130)
plt.close()
for col, (low, high, n_out) in outlier_info.items():
    print("  %-16s granice [%.2f, %.2f], odstajacych=%d" % (col, low, high, n_out))
results["outlier_info"] = {k: list(v) for k, v in outlier_info.items()}

# 3. USUNIECIE WARTOSCI ODSTAJACYCH
print("\n3. USUNIECIE WARTOSCI ODSTAJACYCH")
outlier_cols = ["Num_children", "Num_family", "Total_income", "Years_employed"]
mask = pd.Series(True, index=df.index)
for col in outlier_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask &= (df[col] >= low) & (df[col] <= high)
df_clean = df[mask].reset_index(drop=True)
removed = df.shape[0] - df_clean.shape[0]
print("Oryginalny:", df.shape[0], "Oczyszczony:", df_clean.shape[0], "Usunieto:", removed)
tc_clean = df_clean["Target"].value_counts().sort_index()
results["n_rows_clean"] = int(df_clean.shape[0])
results["n_removed"] = int(removed)
results["removed_pct"] = round(100 * removed / df.shape[0], 2)
results["target0_clean"] = int(tc_clean[0])
results["target1_clean"] = int(tc_clean[1])
results["target1_pct_clean"] = round(100 * df_clean["Target"].mean(), 2)

# 4. STATYSTYKI + WIZUALIZACJE
print("\n4. STATYSTYKI (oczyszczony zbior)")
desc = df_clean[cont_cols].describe().round(2)
print(desc)
results["desc_clean"] = desc.to_dict()

df_enc = df_clean.copy()
for col in cat_cols:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.ravel(), cont_cols):
    sns.histplot(df_clean[col], kde=True, ax=ax, color="#55A868", bins=30)
    ax.set_title(col)
    ax.set_ylabel("")
plt.suptitle("Rozklad cech ciaglych po usunieciu wartosci odstajacych", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "/fig3_distributions.png", dpi=130)
plt.close()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df_enc.corr(), annot=False, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Macierz korelacji atrybutow", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "/fig4_corr.png", dpi=130)
plt.close()

# 5. PODZIAL + SKALOWANIE
print("\n5. PODZIAL I SKALOWANIE")
X = df_enc.drop(columns=["Target"])
y = df_enc["Target"]
feature_names = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape[0], "Test:", X_test.shape[0])
results["n_train"] = int(X_train.shape[0])
results["n_test"] = int(X_test.shape[0])
scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=feature_names, index=X_test.index)

# 6. SMOTE
print("\n6. ZBALANSOWANIE KLAS (SMOTE)")
print("Przed SMOTE:", dict(y_train.value_counts().sort_index()))
X_train_bal, y_train_bal = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_s, y_train)
print("Po SMOTE:", dict(pd.Series(y_train_bal).value_counts().sort_index()))
results["train_before_smote"] = {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()}
results["train_after_smote"] = {str(k): int(v) for k, v in pd.Series(y_train_bal).value_counts().sort_index().items()}

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vb = y_train.value_counts().sort_index()
axes[0].bar(["0", "1"], vb.values, color=["#4C72B0", "#C44E52"])
axes[0].set_title("Przed SMOTE")
for i, v in enumerate(vb.values):
    axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")
va = pd.Series(y_train_bal).value_counts().sort_index()
axes[1].bar(["0", "1"], va.values, color=["#4C72B0", "#C44E52"])
axes[1].set_title("Po SMOTE")
for i, v in enumerate(va.values):
    axes[1].text(i, v + 30, str(v), ha="center", fontweight="bold")
for a in axes:
    a.set_ylabel("Liczba probek")
plt.suptitle("Balansowanie klas zbioru treningowego", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "/fig5_smote.png", dpi=130)
plt.close()

# 7. MODEL POCZATKOWY + WAZNOSC CECH
print("\n7. MODEL POCZATKOWY (Random Forest, wszystkie cechy)")
rf_init = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf_init.fit(X_train_bal, y_train_bal)
pred = rf_init.predict(X_test_s)
proba = rf_init.predict_proba(X_test_s)[:, 1]
m_init = {
    "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
    "AUC": roc_auc_score(y_test, proba), "MCC": matthews_corrcoef(y_test, pred),
}
for k, v in m_init.items():
    print("  %-10s %.4f" % (k, v))
print(classification_report(y_test, pred, digits=3))
results["metrics_init"] = {k: round(float(v), 4) for k, v in m_init.items()}

importances = pd.Series(rf_init.feature_importances_, index=feature_names).sort_values(ascending=False)
print("Waznosc cech:\n", importances.round(4))
results["importances"] = {str(k): round(float(v), 4) for k, v in importances.items()}

fig, ax = plt.subplots(figsize=(9, 7))
colors = ["#C44E52" if i < 10 else "#B0B0B0" for i in range(len(importances))]
ax.barh(importances.index[::-1], importances.values[::-1], color=colors[::-1])
ax.set_xlabel("Waznosc cechy (Gini importance)")
ax.set_title("Waznosc cech - model poczatkowy Random Forest\n(czerwone = 10 najwazniejszych)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "/fig6_importance.png", dpi=130)
plt.close()

# 8. TOP 10
top10 = list(importances.head(10).index)
print("\n8. TOP 10 ATRYBUTOW:", top10)
results["top10"] = top10
X_train_bal_10 = X_train_bal[top10]
X_test_10 = X_test_s[top10]

pickle.dump({
    "X_train_bal_10": X_train_bal_10, "y_train_bal": y_train_bal,
    "X_test_10": X_test_10, "y_test": y_test, "results": results,
}, open(OUT + "/ckpt.pkl", "wb"))
print("Zapisano checkpoint ckpt.pkl. Teraz uruchom grid_tune.py")
