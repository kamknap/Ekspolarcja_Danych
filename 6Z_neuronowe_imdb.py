import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
df = pd.read_csv('imdb_top_1000_imputed.csv')

# Zmienna docelowa: 1 → ocena ≥ 8.0, 0 → ocena < 8.0
df['High_Rating'] = (df['IMDB_Rating'] >= 8.0).astype(int)
print(f"Rozkład klas: {df['High_Rating'].value_counts().to_dict()}")

# Cechy numeryczne
num_features = ['Meta_score', 'No_of_Votes', 'Gross', 'Runtime', 'Released_Year']

# Cechy kategoryczne
le_cert = LabelEncoder()
df['Certificate_enc'] = le_cert.fit_transform(df['Certificate'])

# Cecha z gatunków – liczba gatunków
df['Genre_count'] = df['Genre'].apply(lambda x: len(x.split(',')))

# Finalny zbiór cech
features = num_features + ['Certificate_enc', 'Genre_count']
X = df[features].values
y = df['High_Rating'].values

print(f"Rozmiar zbioru: {X.shape}, cechy: {features}")

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizacja cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Treningowych: {len(X_train)}, testowych: {len(X_test)}\n")

# 2. EKSPERYMENTY
results = []

def train_eval(label, clf, category):
    """Trenuje model i zwraca wyniki."""
    clf.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test  = accuracy_score(y_test,  clf.predict(X_test))
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    results.append({
        'Opis':          label,
        'Kategoria':     category,
        'Acc. treningowa': round(acc_train * 100, 2),
        'Acc. testowa':   round(acc_test  * 100, 2),
        'CV (śred.)':     round(cv_scores.mean() * 100, 2),
        'CV (std)':       round(cv_scores.std()  * 100, 2),
    })
    print(f"  [{label}]  train={acc_train:.4f}  test={acc_test:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    return clf

# Liczba warstw ukrytych
print("=== A. Liczba warstw ukrytych ===")
train_eval("1 warstwa (64)",        MLPClassifier(hidden_layer_sizes=(64,),       max_iter=500, random_state=42), "Liczba warstw")
train_eval("2 warstwy (64,32)",     MLPClassifier(hidden_layer_sizes=(64,32),     max_iter=500, random_state=42), "Liczba warstw")
train_eval("3 warstwy (64,32,16)",  MLPClassifier(hidden_layer_sizes=(64,32,16),  max_iter=500, random_state=42), "Liczba warstw")
train_eval("4 warstwy (64,64,32,16)",MLPClassifier(hidden_layer_sizes=(64,64,32,16), max_iter=500, random_state=42), "Liczba warstw")

# Liczba neuronów
print("\n=== B. Liczba neuronów (2 warstwy) ===")
train_eval("Małe (16,8)",           MLPClassifier(hidden_layer_sizes=(16,8),      max_iter=500, random_state=42), "Liczba neuronów")
train_eval("Średnie (64,32)",       MLPClassifier(hidden_layer_sizes=(64,32),     max_iter=500, random_state=42), "Liczba neuronów")
train_eval("Duże (128,64)",         MLPClassifier(hidden_layer_sizes=(128,64),    max_iter=500, random_state=42), "Liczba neuronów")
train_eval("Bardzo duże (256,128)", MLPClassifier(hidden_layer_sizes=(256,128),   max_iter=500, random_state=42), "Liczba neuronów")

# Funkcje aktywacji
print("\n=== C. Funkcje aktywacji (64,32) ===")
for act in ['relu', 'tanh', 'logistic']:
    train_eval(f"Aktywacja: {act}", MLPClassifier(hidden_layer_sizes=(64,32), activation=act, max_iter=500, random_state=42), "Funkcja aktywacji")

# Algorytm optymalizacji
print("\n=== D. Algorytm optymalizacji (64,32) ===")
train_eval("Solver: adam",   MLPClassifier(hidden_layer_sizes=(64,32), solver='adam',  max_iter=500, random_state=42), "Solver")
train_eval("Solver: sgd",    MLPClassifier(hidden_layer_sizes=(64,32), solver='sgd',   max_iter=500, random_state=42), "Solver")
train_eval("Solver: lbfgs",  MLPClassifier(hidden_layer_sizes=(64,32), solver='lbfgs', max_iter=500, random_state=42), "Solver")

# Regularyzacja (alpha)
print("\n=== E. Regularyzacja L2 (alfa) ===")
for alpha in [0.0001, 0.001, 0.01, 0.1]:
    train_eval(f"alpha={alpha}", MLPClassifier(hidden_layer_sizes=(64,32), alpha=alpha, max_iter=500, random_state=42), "Regularyzacja")

# 3. TABELARYCZNE PODSUMOWANIE
df_res = pd.DataFrame(results)
print("\n" + "="*80)
print(df_res.to_string(index=False))
print("="*80)

# Zapis CSV z wynikami
df_res.to_csv('wyniki_sieci.csv', index=False)

# 4. NAJLEPSZY MODEL RAPORT
best_idx = df_res['Acc. testowa'].idxmax()
best_row  = df_res.loc[best_idx]
print(f"\nNajlepszy model: {best_row['Opis']}  (acc. testowa = {best_row['Acc. testowa']}%)")

best_clf = MLPClassifier(hidden_layer_sizes=(64,32), solver='adam',
                         activation='relu', alpha=0.0001,
                         max_iter=500, random_state=42)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print("\nRaport klasyfikacji (najlepszy model):")
print(classification_report(y_test, y_pred,
                             target_names=['IMDB < 8.0', 'IMDB ≥ 8.0']))

# 5. WYKRESY
categories = df_res['Kategoria'].unique()
colors = {
    'Liczba warstw':    '#4C72B0',
    'Liczba neuronów':  '#DD8452',
    'Funkcja aktywacji':'#55A868',
    'Solver':           '#C44E52',
    'Regularyzacja':    '#8172B2',
}

fig, axes = plt.subplots(len(categories), 1, figsize=(10, 4 * len(categories)))
fig.suptitle('Wpływ parametrów sieci neuronowej na dokładność klasyfikacji\n(zbiór IMDB Top 1000)',
             fontsize=13, fontweight='bold', y=1.01)

for ax, cat in zip(axes, categories):
    sub = df_res[df_res['Kategoria'] == cat].reset_index(drop=True)
    x   = np.arange(len(sub))
    w   = 0.35
    c   = colors.get(cat, '#999')

    bars_tr = ax.bar(x - w/2, sub['Acc. treningowa'], w, label='Treningowa', color=c, alpha=0.9)
    bars_te = ax.bar(x + w/2, sub['Acc. testowa'],    w, label='Testowa',    color=c, alpha=0.5)

    # Errorbar dla cross-validation
    ax.errorbar(x + w/2, sub['CV (śred.)'],
                yerr=sub['CV (std)'], fmt='o', color='black',
                capsize=4, linewidth=1.5, label='CV (5-fold)')

    ax.set_title(cat, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['Opis'], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Dokładność [%]')
    ax.set_ylim(50, 100)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Etykiety wartości
    for bar in bars_te:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('wykres_sieci.png', dpi=150, bbox_inches='tight')
print("\nWykres zapisany: wykres_sieci.png")
print("Tabela wyników: wyniki_sieci.csv")

# 6. KRZYWA UCZENIA (najlepszy model)
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
    np.vstack([X_train, X_test]),
    np.hstack([y_train, y_test]),
    cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring='accuracy'
)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(train_sizes, train_scores.mean(axis=1)*100, 'o-', color='#4C72B0', label='Treningowa')
ax2.fill_between(train_sizes,
                 (train_scores.mean(axis=1) - train_scores.std(axis=1))*100,
                 (train_scores.mean(axis=1) + train_scores.std(axis=1))*100,
                 alpha=0.15, color='#4C72B0')
ax2.plot(train_sizes, test_scores.mean(axis=1)*100,  'o-', color='#DD8452', label='Walidacyjna')
ax2.fill_between(train_sizes,
                 (test_scores.mean(axis=1) - test_scores.std(axis=1))*100,
                 (test_scores.mean(axis=1) + test_scores.std(axis=1))*100,
                 alpha=0.15, color='#DD8452')
ax2.set_xlabel('Liczba przykładów treningowych')
ax2.set_ylabel('Dokładność [%]')
ax2.set_title('Krzywa uczenia – sieć (64, 32), relu, adam', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('krzywa_uczenia.png', dpi=150, bbox_inches='tight')
print("Krzywa uczenia: krzywa_uczenia.png")