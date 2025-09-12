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
# print(inc.describe(include = ["int64", "float64"]))
num_cols = inc.select_dtypes(include = ["int64", "float64"]).columns
num_cols = [col for col in num_cols if col != "time_min"]

plt.figure(figsize=(30, 5 * 3))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, 3, i)
    plt.hist(inc[col], bins=30, edgecolor="black")
    plt.title(f"Histograma de {col}", fontsize= 40)
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.xticks(fontsize=40)
plt.tight_layout()
plt.show()

plt.figure(figsize=(30, 5 * len(num_cols)))
for i, col in enumerate(num_cols, 1):
    plt.subplot(len(num_cols), 1, i)
    plt.scatter(inc[col], inc["time_min"], alpha=0.5)
    plt.title(f"{col} vs time_min", fontsize=20)
    plt.xlabel(col)
    plt.ylabel("time_min")
plt.tight_layout()
plt.show()


plt.hist(inc["time_min"], bins=50, edgecolor="black")
plt.xlabel("Minutos")
plt.ylabel("Frecuencia")
plt.title("Distribución de time_min")
plt.show()

#%%
# #Exploración de variables categoricas-------------
cat_cols = ["contact_type","urgency", "notify","priority", "impact", "knowledge"]
n = len(cat_cols)

plt.figure(figsize=(40, 50))
for i, col in enumerate(cat_cols, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=col, y=np.log1p(inc["time_min"]), data=inc, color="yellow")
    plt.title(col, fontsize=40)
    plt.xticks(fontsize=40)
plt.tight_layout()
plt.show()

# %%
#Correlaciones con la variable de interés
map_priority = {
    "1 - Critical": 1,
    "2 - High": 2,
    "3 - Moderate": 3,
    "4 - Low": 4
}
inc["priority_ord"] = inc["priority"].map(map_priority)

map_urgency = {
    "1 - High":1,
    "2 - Medium":2,
    "3 - Low":3
}
inc["urgency_ord"] = inc["urgency"].map(map_urgency)
inc["impact_ord"] = inc["impact"].map(map_urgency)

map_notify ={
    "Do Not Notify":0,
    "Send Email":1
}
inc["notify_ord"] = inc["notify"].map(map_notify)

map_sla ={
    "true":1,
    "false":0
}
inc["sla_ord"] = inc["made_sla"].map(map_sla)

inc["know_ord"] = inc["knowledge"].astype(int)

vars_interes = ["urgency_ord","impact_ord","priority_ord","notify_ord", "reassignment_count", "reopen_count", "sys_mod_count", "made_sla", "know_ord", "time_min"]
corr = inc[vars_interes].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Matriz de correlación", fontsize=14)
plt.show()

# %%
#-------------------------- Preparación de datos --------------------------
#Eliminacion de outliers en el reassignment_count:
Q1_rea= inc["reassignment_count"].quantile(0.25)
Q3_rea = inc["reassignment_count"].quantile(0.75)
IQR_rea = Q3_rea - Q1_rea

Limite_sup_rea = Q3_rea + 1.5 * IQR_rea
inc_final = inc[inc["reassignment_count"] <= Limite_sup_rea]
print(len(inc_final))

plt.figure(figsize=(8,5))
sns.histplot(inc_final["reassignment_count"])
plt.title("Numero de reasgnamientos", fontsize=14)
plt.ylabel("Frecuencia")
plt.show()


#Eliminacion de outliers en el sys_mod_count:
Q1_sys= inc["sys_mod_count"].quantile(0.25)
Q3_sys = inc["sys_mod_count"].quantile(0.75)
IQR_sys = Q3_sys - Q1_sys

Limite_sup_sys = Q3_sys + 1.5 * IQR_sys
inc_final = inc_final[inc_final["sys_mod_count"] <= Limite_sup_sys]
print(len(inc_final))

plt.figure(figsize=(8,5))
sns.histplot(inc_final["sys_mod_count"])
plt.title("Numero de actualizaciones", fontsize=14)
plt.ylabel("Frecuencia")
plt.show()


#Eliminacion de outliers en el tiempo:
Q1_time = inc["time_min"].quantile(0.25)
Q3_time = inc["time_min"].quantile(0.75)
IQR_time = Q3_time-Q1_time

Limite_sup_time = Q3_time + 1.5 * IQR_time
inc_final = inc_final[inc_final["time_min"] <= Limite_sup_time]
print(len(inc_final))

plt.figure(figsize=(8,5))
sns.histplot(inc_final["time_min"])
plt.title("Tiempos de resolución", fontsize=14)
plt.xlabel("Tiempo (minutos)")
plt.ylabel("Frecuencia")
plt.show()

# %%
#Reordenamiento de columnas inc_final
inc_final = inc_final.drop(["active", "made_sla","caller_id",
                            "opened_by","sys_created_by", "sys_created_at",
                            "sys_updated_by","sys_updated_at","contact_type",
                            "location","category", "subcategory","u_symptom",
                            "cmdb_ci","priority","assignment_group",
                            "assigned_to","u_priority_confirmation",
                            "notify","problem_id","rfc","vendor","caused_by",
                            "closed_code","resolved_by","resolved_at","problem",
                            "sla_ord","notify_ord"
                            ], axis=1)

cols = ["number", "opened_at", "closed_at", "incident_state",
        "time","time_min","reassignment_count","reopen_count",
        "sys_mod_count","impact","impact_ord",
        "urgency","urgency_ord","knowledge","know_ord"]
inc_final = inc_final[cols]
inc_final
# %%

# exportar dataset limpio
inc_final.to_csv("inc_final.csv", index=False)
print("Dataset limpio exportado como inc_final.csv") 