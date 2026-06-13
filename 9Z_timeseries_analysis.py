"""
Zakres analizy:
  1. Wczytanie i wizualizacja szeregu
  2. Dekompozycja (model multiplikatywny)
  3. Badanie stacjonarnosci (test ADF) i roznicowanie
  4. Funkcje ACF i PACF
  5. Model SARIMA: dopasowanie, prognoza, ocena
  6. Prognoza na przyszle okresy
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set_style("whitegrid")
OUT = "."
DATA = "AirPassengers.csv"
res = {}

# 1. WCZYTANIE
df = pd.read_csv(DATA)
df.columns = ["Month", "Passengers"]
df["Month"] = pd.to_datetime(df["Month"])
df = df.set_index("Month").asfreq("MS")
y = df["Passengers"]
print("Liczba obserwacji:", len(y))
print("Zakres dat:", y.index.min().date(), "do", y.index.max().date())
print("Braki danych:", int(y.isnull().sum()))
print("Min:", int(y.min()), "Maks:", int(y.max()), "Srednia:", round(y.mean(), 1))
res["n"] = len(y)
res["start"] = str(y.index.min().date())
res["end"] = str(y.index.max().date())
res["min"] = int(y.min()); res["max"] = int(y.max())
res["mean"] = round(float(y.mean()), 1); res["std"] = round(float(y.std()), 1)

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(y.index, y.values, color="#C44E52", lw=1.5)
ax.set_title("Miesieczna liczba pasazerow linii lotniczych (1949-1960)", fontweight="bold")
ax.set_xlabel("Rok"); ax.set_ylabel("Liczba pasazerow (tys.)")
plt.tight_layout(); plt.savefig(OUT + "/ts_fig1_series.png", dpi=130); plt.close()

# srednia i odchylenie roczne (wzrost amplitudy)
yearly = y.groupby(y.index.year).agg(["mean", "std"])
res["amp_1949"] = round(float(yearly.loc[1949, "std"]), 1)
res["amp_1960"] = round(float(yearly.loc[1960, "std"]), 1)
print("Odch. std. 1949:", res["amp_1949"], "1960:", res["amp_1960"])

# 2. DEKOMPOZYCJA (multiplikatywna)
decomp = seasonal_decompose(y, model="multiplicative", period=12)
fig = decomp.plot()
fig.set_size_inches(11, 8)
for a in fig.axes:
    a.set_xlabel("")
fig.suptitle("Dekompozycja multiplikatywna szeregu", fontweight="bold", y=1.0)
plt.tight_layout(); plt.savefig(OUT + "/ts_fig2_decomp.png", dpi=130); plt.close()
seas = decomp.seasonal.groupby(decomp.seasonal.index.month).mean()
res["seas_peak_month"] = int(seas.idxmax()); res["seas_peak_val"] = round(float(seas.max()), 3)
res["seas_low_month"] = int(seas.idxmin()); res["seas_low_val"] = round(float(seas.min()), 3)
print("Szczyt sezonowy: miesiac", res["seas_peak_month"], "wsp.", res["seas_peak_val"])
print("Dolek sezonowy: miesiac", res["seas_low_month"], "wsp.", res["seas_low_val"])

# 3. STACJONARNOSC (ADF)
def adf(series, name):
    series = series.dropna()
    stat, p, *_ = adfuller(series)
    print(f"ADF [{name}]: stat={stat:.3f}, p={p:.4f}")
    return round(stat, 3), round(p, 4)

res["adf_raw"] = adf(y, "szereg surowy")
y_log = np.log(y)
res["adf_logdiff"] = adf(y_log.diff(), "log + roznicowanie (d=1)")
res["adf_logdiff_seas"] = adf(y_log.diff().diff(12), "log + roznicowanie d=1, D=1")

# 4. ACF / PACF (na log + d=1 + D=12)
y_stat = y_log.diff().diff(12).dropna()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
plot_acf(y_stat, ax=axes[0], lags=36, color="#4C72B0")
plot_pacf(y_stat, ax=axes[1], lags=36, method="ywm", color="#4C72B0")
axes[0].set_title("ACF (log, d=1, D=1)")
axes[1].set_title("PACF (log, d=1, D=1)")
plt.tight_layout(); plt.savefig(OUT + "/ts_fig3_acf_pacf.png", dpi=130); plt.close()

# 5. SARIMA: podzial train/test (ostatnie 12 miesiecy = test)
train, test = y_log.iloc[:-12], y_log.iloc[-12:]
configs = [
    ((1, 1, 1), (1, 1, 1, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((2, 1, 1), (1, 1, 1, 12)),
    ((1, 1, 0), (1, 1, 0, 12)),
]
best = None
for order, sorder in configs:
    try:
        m = SARIMAX(train, order=order, seasonal_order=sorder,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        if best is None or m.aic < best[2]:
            best = (order, sorder, m.aic, m)
    except Exception as e:
        print("blad", order, sorder, e)
order, sorder, aic, model = best
print("Najlepszy model: SARIMA", order, "x", sorder, "AIC=%.2f" % aic)
res["order"] = list(order); res["sorder"] = list(sorder); res["aic"] = round(aic, 2)

# prognoza na okres testowy
fc = model.get_forecast(steps=12)
pred_log = fc.predicted_mean
ci_log = fc.conf_int()
pred = np.exp(pred_log)
test_real = np.exp(test)
rmse = float(np.sqrt(np.mean((pred.values - test_real.values) ** 2)))
mae = float(np.mean(np.abs(pred.values - test_real.values)))
mape = float(np.mean(np.abs((pred.values - test_real.values) / test_real.values)) * 100)
print("RMSE=%.2f MAE=%.2f MAPE=%.2f%%" % (rmse, mae, mape))
res["rmse"] = round(rmse, 2); res["mae"] = round(mae, 2); res["mape"] = round(mape, 2)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(np.exp(y_log).index, np.exp(y_log).values, color="#4C72B0", lw=1.3, label="Dane rzeczywiste")
ax.plot(pred.index, pred.values, color="#C44E52", lw=2, label="Prognoza (test)")
ax.fill_between(pred.index, np.exp(ci_log.iloc[:, 0]), np.exp(ci_log.iloc[:, 1]),
                color="#C44E52", alpha=0.2, label="95% przedzial ufnosci")
ax.axvline(test.index[0], color="gray", ls="--", lw=1)
ax.set_title("Prognoza SARIMA na okres testowy (rok 1960)", fontweight="bold")
ax.set_xlabel("Rok"); ax.set_ylabel("Liczba pasazerow (tys.)")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT + "/ts_fig4_forecast.png", dpi=130); plt.close()

# 6. PROGNOZA NA PRZYSZLOSC (model na pelnych danych, 24 miesiace)
full = SARIMAX(y_log, order=order, seasonal_order=sorder,
               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
ffc = full.get_forecast(steps=24)
fpred = np.exp(ffc.predicted_mean)
fci = ffc.conf_int()
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(y.index, y.values, color="#4C72B0", lw=1.3, label="Dane historyczne")
ax.plot(fpred.index, fpred.values, color="#55A868", lw=2, label="Prognoza 24 mies.")
ax.fill_between(fpred.index, np.exp(fci.iloc[:, 0]), np.exp(fci.iloc[:, 1]),
                color="#55A868", alpha=0.2, label="95% przedzial ufnosci")
ax.set_title("Prognoza liczby pasazerow na 24 miesiace (1961-1962)", fontweight="bold")
ax.set_xlabel("Rok"); ax.set_ylabel("Liczba pasazerow (tys.)")
ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(OUT + "/ts_fig5_future.png", dpi=130); plt.close()
res["fc_1961_12"] = int(round(float(fpred.iloc[11])))
res["fc_last"] = int(round(float(fpred.iloc[-1])))

import json
json.dump(res, open(OUT + "/ts_results.json", "w"), indent=2, default=str)
print("\nWyniki:", json.dumps(res, indent=2, default=str))
print("DONE")
