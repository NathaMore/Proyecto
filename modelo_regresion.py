import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

inc_final = pd.read_csv("inc_final.csv")
print(f"Dataset cargado: {inc_final.shape[0]} filas y {inc_final.shape[1]} columnas")

# ESTADISTICAS DESCRIPTIVAS -----------------------------------------------------------
# Calcular métricas básicas de time_min
mean_time = inc_final["time_min"].mean()
median_time = inc_final["time_min"].median()
std_time = inc_final["time_min"].std()
min_time = inc_final["time_min"].min()
max_time = inc_final["time_min"].max()

print("\nEstadísticas descriptivas de time_min (minutos)")
print(f"Media              : {mean_time:.2f}")
print(f"Mediana            : {median_time:.2f}")
print(f"Desviación estándar: {std_time:.2f}")
print(f"Mínimo             : {min_time:.2f}")
print(f"Máximo             : {max_time:.2f}")

# Seleccionar solo columnas numéricas
num_cols = inc_final.select_dtypes(include=["int64", "float64"])

# Calcular estadísticas de las variables numericas
stats = pd.DataFrame({
    "Media": num_cols.mean(),
    "Mediana": num_cols.median(),
    "Desviación estándar": num_cols.std(),
    "Mínimo": num_cols.min(),
    "Máximo": num_cols.max()
})

print("\n Estadísticas descriptivas de variables numéricas:")
print(stats)

# BOXPLOTS TODAS LAS VARIABLES (GRAFICAS)---------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Variables numéricas
num_cols = inc_final.select_dtypes(include=["int64", "float64"]).columns

# Separar time_min y el resto
num_cols_otros = [col for col in num_cols if col != "time_min"]

# Boxplot de time_min -------------------
plt.figure(figsize=(5, 8))
sns.boxplot(y=inc_final["time_min"], color="lightgreen")
plt.title("Boxplot de time_min (minutos)", fontsize=14)
plt.ylabel("time_min", fontsize=12)
plt.show()

#  Boxplots de las demás variables -------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=inc_final[num_cols_otros], palette="Set2")
plt.title("Boxplots de variables numéricas (excepto time_min)", fontsize=14)
plt.xticks(rotation=30, ha="right", fontsize=12)  # rotar y alinear nombres
plt.tight_layout()
plt.show()

# Variables numéricas
num_cols = inc_final.select_dtypes(include=["int64", "float64"]).columns
num_cols_otros = [col for col in num_cols if col != "time_min"]

# Histograma de time_min -------------------

plt.figure(figsize=(10, 6))
sns.histplot(inc_final["time_min"], bins=50, kde=False, color="lightgreen", edgecolor="black")
plt.title("Histograma de time_min (minutos)", fontsize=16)
plt.xlabel("time_min (minutos)", fontsize=12)
plt.ylabel("Frecuencia", fontsize=12)
plt.tight_layout()
plt.show()

# Histogramas de las demás variables -------------------
n = len(num_cols_otros)

plt.figure(figsize=(12, 4 * n))   # más espacio: 4 de alto por variable
for i, col in enumerate(num_cols_otros, 1):
    plt.subplot(n, 1, i)
    sns.histplot(inc_final[col], bins=30, kde=False, color="skyblue", edgecolor="black")
    plt.title(f"Histograma de {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)

plt.tight_layout()
plt.show()

# MODELO DE REGRESION LINEAL SIMPLE ---------------------------------------------------------------------

# definir variables 
X = inc_final[["reassignment_count", "reopen_count", "sys_mod_count",
               "impact_ord", "urgency_ord", "know_ord"]]
y = np.log1p(inc_final["time_min"])   # log para estabilizar colas

# train/Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# modelo
reglineal = LinearRegression()
reglineal.fit(X_train, y_train)

# predicciones
y_pred = reglineal.predict(X_test)

# re-transformación a minutos
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

# métricas
mae = mean_absolute_error(y_test_exp, y_pred_exp)
rmse = mean_squared_error(y_test_exp, y_pred_exp) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n Resultados del modelo de Regresión Lineal")
# MAE y RMSE : dicen cuántos minutos se equivoca el modelo en promedio (error típico)
# Coeficientes : dicen cómo influye cada variable (positivo = aumenta tiempo, negativo = reduce tiempo)
# Importancia de variables : dice cuáles son las variables más determinantes para predecir el tiempo

print(f"MAE : {mae:.2f} minutos")
print(f"RMSE: {rmse:.2f} minutos")

# coeficientes 
coef = pd.DataFrame({
    "Variable": X.columns,
    "Coeficiente": reglineal.coef_
}).sort_values(by="Coeficiente", ascending=False)

print("\n Coeficientes del modelo (escala log):")
print(coef)

print(f"\nIntercepto: {reglineal.intercept_:.4f}")

# MODELO DE RANDON FOREST REGRESSOR ------------------------------------------------
from sklearn.ensemble import RandomForestRegressor

# variables predictoras y target
X = inc_final[["reassignment_count", "reopen_count", "sys_mod_count",
               "impact_ord", "urgency_ord", "know_ord"]]
y = np.log1p(inc_final["time_min"])  # seguimos usando log para estabilizar

# train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Modelo Random Forest
rf = RandomForestRegressor(
    n_estimators=200,      # número de árboles
    max_depth=None,        # sin límite de profundidad
    random_state=42,
    n_jobs=-1              # usa todos los núcleos del PC
)
rf.fit(X_train, y_train)

# predicciones
y_pred = rf.predict(X_test)

# re-transformación
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

# métricas
mae_rf = mean_absolute_error(y_test_exp, y_pred_exp)
rmse_rf = mean_squared_error(y_test_exp, y_pred_exp) ** 0.5

print("\n Resultados del modelo Random Forest Regressor")
print(f"MAE : {mae_rf:.2f} minutos")
print(f"RMSE: {rmse_rf:.2f} minutos")

# importancia de variables
importances = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": rf.feature_importances_
}).sort_values(by="Importancia", ascending=False)

print("\n Importancia de variables según Random Forest:")
print(importances)

# MODELO GLM GAMMA ----------------------------------------------------------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf

# dataset (sin log-transformar, porque GLM Gamma ya lo maneja)
df = inc_final[["time_min", "reassignment_count", "reopen_count", 
                "sys_mod_count", "impact_ord", "urgency_ord", "know_ord"]]

# fórmula: variable dependiente ~ variables independientes
formula = "time_min ~ reassignment_count + reopen_count + sys_mod_count + impact_ord + urgency_ord + know_ord"

# ajuste del modelo
glm_gamma = smf.glm(formula=formula, data=df,
                    family=sm.families.Gamma(sm.families.links.log())).fit()

# resumen del modelo
print(glm_gamma.summary())

# predicciones
y_pred = glm_gamma.predict(df)
y_true = df["time_min"]

# métricas
mae_glm = mean_absolute_error(y_true, y_pred)
rmse_glm = mean_squared_error(y_true, y_pred) ** 0.5

print("\n Resultados del modelo GLM Gamma")
print(f"MAE : {mae_glm:.2f} minutos")
print(f"RMSE: {rmse_glm:.2f} minutos")
