"""Krok 9-10: dostrajanie hiperparametrow i model ostateczny (wczytuje checkpoint)."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)

RANDOM_STATE = 42
sns.set_style("whitegrid")
OUT = "/sessions/optimistic-nifty-lovelace/mnt/outputs"
ck = pickle.load(open(f"{OUT}/ckpt.pkl", "rb"))
X_train_bal_10, y_train_bal = ck["X_train_bal_10"], ck["y_train_bal"]
X_test_10, y_test = ck["X_test_10"], ck["y_test"]
results = ck["results"]

param_grid = {"n_estimators": [100, 200], "max_depth": [10, None], "min_samples_leaf": [2]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                    param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train_bal_10, y_train_bal)
print("Najlepsze parametry:", grid.best_params_)
print("Najlepsze AUC (CV):  %.4f" % grid.best_score_)
results["best_params"] = grid.best_params_
results["best_cv_auc"] = grid.best_score_

final = grid.best_estimator_
pred = final.predict(X_test_10)
proba = final.predict_proba(X_test_10)[:, 1]
m_final = {
    "Accuracy": accuracy_score(y_test, pred), "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred), "F1": f1_score(y_test, pred),
    "AUC": roc_auc_score(y_test, proba), "MCC": matthews_corrcoef(y_test, pred),
}
print("\nModel ostateczny:")
for k, v in m_final.items():
    print(f"  {k:10s}: {v:.4f}")
print(classification_report(y_test, pred, digits=3))
results["metrics_final"] = m_final

results["final_cv_auc_mean"] = grid.best_score_
print("CV AUC (model ostateczny, z GridSearch): %.4f" % grid.best_score_)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Odrzucony", "Zatwierdzony"], yticklabels=["Odrzucony", "Zatwierdzony"])
axes[0].set_xlabel("Predykcja"); axes[0].set_ylabel("Rzeczywista klasa")
axes[0].set_title("Macierz pomylek - model ostateczny")
fpr, tpr, _ = roc_curve(y_test, proba)
axes[1].plot(fpr, tpr, color="#C44E52", lw=2, label=f"ROC (AUC = {m_final['AUC']:.3f})")
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("Krzywa ROC - model ostateczny"); axes[1].legend(loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_final_eval.png", dpi=130); plt.close()

comp = pd.DataFrame({"Poczatkowy (20 cech)": results["metrics_init"],
                     "Ostateczny (10 cech)": m_final})
print("\nPOROWNANIE:\n", comp.round(4))

# serializacja
ser = {}
for k, v in results.items():
    if isinstance(v, pd.Series):
        ser[k] = {str(i): round(float(x), 4) for i, x in v.items()}
    elif isinstance(v, pd.DataFrame):
        ser[k] = v.round(4).to_dict()
    elif isinstance(v, dict):
        ser[k] = {str(i): (round(float(x), 4) if isinstance(x, (int, float, np.floating)) else x) for i, x in v.items()}
    else:
        ser[k] = (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
json.dump(ser, open(f"{OUT}/results.json", "w"), indent=2, default=str)
print("\nZapisano results.json + fig7. DONE")
