import numpy as np
import pandas as pd

incidentes = pd.read_csv("incident_event_log.csv")
incidentes["problem"] = incidentes["category"] + " - " + incidentes["subcategory"]

#Selección de columnas
columnas = ["number","incident_state", "active",
             "reassignment_count", "reopen_count","sys_mod_count",
             "contact_type", "location", "problem", "priority",
             "assignment_group", "knowledge","opened_at", "resolved_at"             
             ]

df =incidentes[columnas]

df = df[df["opened_at"].notna()]              # elimina NaN / None
df = df[df["opened_at"].str.strip() != ""]    # elimina cadenas vacías
df = df[df["opened_at"].str.strip() != "?"]   # elimina las que son "?"

df = df[df["resolved_at"].notna()]
df = df[df["resolved_at"].str.strip() != ""]
df = df[df["resolved_at"].str.strip() != "?"]

df["opened_at"] = pd.to_datetime(df["opened_at"], format="%d/%m/%Y %H:%M")
df["resolved_at"] = pd.to_datetime(df["resolved_at"], format="%d/%m/%Y %H:%M")

df["time"] = df["resolved_at"] - df["opened_at"]