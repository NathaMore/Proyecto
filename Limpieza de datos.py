#%%
# ---------------------------- Preparación de datos ----------------------------
import numpy as np
import pandas as pd

incidentes = pd.read_csv("incident_event_log.csv")
incidentes["problem"] = incidentes["category"] + " - " + incidentes["subcategory"]
print(incidentes.shape[0])

#Eliminar faltantes en tiempos
incidentes = incidentes[incidentes["opened_at"].notna()]              # elimina NaN / None
incidentes = incidentes[incidentes["opened_at"].str.strip() != ""]    # elimina cadenas vacías
incidentes = incidentes[incidentes["opened_at"].str.strip() != "?"]   # elimina las que son "?"

incidentes = incidentes[incidentes["resolved_at"].notna()]
incidentes = incidentes[incidentes["resolved_at"].str.strip() != ""]
incidentes = incidentes[incidentes["resolved_at"].str.strip() != "?"]

#Cambiar el formato a fecha
incidentes["opened_at"] = pd.to_datetime(incidentes["opened_at"], format="%d/%m/%Y %H:%M")
incidentes["resolved_at"] = pd.to_datetime(incidentes["resolved_at"], format="%d/%m/%Y %H:%M")

#Calcular el tiempo de resolución
incidentes["time"] = incidentes["resolved_at"] - incidentes["opened_at"]
incidentes["time_min"] = incidentes["time"].dt.total_seconds() / 60

#%%
#-------------------------- Exploración de datos --------------------------
#filtro a casos relevantes
incidentes = incidentes[incidentes["incident_state"].isin(["Closed", "Resolved"])]

columnas = incidentes.shape[1]
filas = incidentes.shape[0]
n_total = len(incidentes)
#print(filas)
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

print("\n--- Variables categóricas ---")
print(pd.DataFrame({
        "Nulos": nulos[categoricas],
        "% Nulos": nulos_pct[categoricas]
    }).sort_values("Nulos", ascending=False))