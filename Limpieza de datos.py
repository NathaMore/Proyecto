#%%
#Librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# ---------------------------- Preparación de datos ----------------------------
incidentes = pd.read_csv("incident_event_log.csv")
incidentes["problem"] = incidentes["category"] + " - " + incidentes["subcategory"]
print(incidentes.shape[0])

#Eliminar faltantes en tiempos
incidentes = incidentes[incidentes["opened_at"].notna()]              # elimina NaN / None
incidentes = incidentes[incidentes["opened_at"].str.strip() != ""]    # elimina cadenas vacías
incidentes = incidentes[incidentes["opened_at"].str.strip() != "?"]   # elimina las que son "?"

incidentes = incidentes[incidentes["closed_at"].notna()]
incidentes = incidentes[incidentes["closed_at"].str.strip() != ""]
incidentes = incidentes[incidentes["closed_at"].str.strip() != "?"]

#Cambiar el formato a fecha
incidentes["opened_at"] = pd.to_datetime(incidentes["opened_at"], format="%d/%m/%Y %H:%M")
incidentes["closed_at"] = pd.to_datetime(incidentes["closed_at"], format="%d/%m/%Y %H:%M")

#Calcular el tiempo de resolución
incidentes["time"] = incidentes["closed_at"] - incidentes["opened_at"]
incidentes["time_min"] = incidentes["time"].dt.total_seconds() / 60

#%%
#-------------------------- Exploración de datos --------------------------

columnas = incidentes.shape[1]
filas = incidentes.shape[0]
n_total = len(incidentes)
print(filas)
#print(columnas)

#Recuento de valores nulos------------------
incidentes = incidentes.replace("?", np.nan)
nulos = incidentes.isna().sum()
nulos_pct = (nulos / n_total) * 100

#Agrupacion:
categoricas = incidentes.select_dtypes(include=["object", "category"]).columns
numericas = incidentes.select_dtypes(include=["int64", "float64"]).columns

print("\n--- Variables categóricas ---")
print(pd.DataFrame({
        "Nulos": nulos[categoricas],
        "% Nulos": nulos_pct[categoricas]
        }).sort_values("Nulos", ascending=False))

print("\n--- Variables numericas ---")
print(pd.DataFrame({
        "Nulos": nulos[numericas],
        "% Nulos": nulos_pct[numericas]
    }).sort_values("Nulos", ascending=False))

#%%
#filtrado
inc = incidentes[incidentes["incident_state"] == "Closed"]
print(len(inc))
inc = inc.drop_duplicates(subset=["number"])
print(len(inc))

# %%
#Exploración de variables numéricas-------------
print(inc.describe(include = ["int64", "float64"]))
num_cols = inc.select_dtypes(include = ["int64", "float64"]).columns
num_cols = [col for col in num_cols if col != "time_min"]

plt.figure(figsize=(30, 5 * 3))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, 3, i)
    plt.hist(inc[col], bins=30, edgecolor="black")
    plt.title(f"Histograma de {col}", fontsize= 40)
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

plt.hist(inc["time_min"], bins=50, edgecolor="black")
plt.xlabel("Minutos")
plt.ylabel("Frecuencia")
plt.title("Distribución de time_min")
plt.show()
