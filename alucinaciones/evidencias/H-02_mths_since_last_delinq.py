"""Evidencia H-02: qué dice el diccionario de Kaggle sobre mths_since_last_delinq (y qué no dice).

Uso (desde la raíz del repo, con el crudo en data/raw/): python alucinaciones/evidencias/H-02_mths_since_last_delinq.py
"""
import pandas as pd

dic = pd.read_excel("data/raw/credit-risk-dataset/LCDataDictionary.xlsx", sheet_name="LoanStats")
fila = dic[dic.iloc[:, 0].astype(str).str.strip() == "mths_since_last_delinq"]
print("Definición textual en LCDataDictionary.xlsx (hoja LoanStats):")
print("  ", repr(fila.iloc[0, 1]))
texto = " ".join(dic.iloc[:, 1].dropna().map(str)).lower()
for palabra in ("missing", "blank", "null", "never", "empty"):
    print(f"   ¿el diccionario menciona '{palabra}' en alguna definición? {palabra in texto}")
crudo = pd.read_csv("data/raw/credit-risk-dataset/loan/loan.csv", usecols=["mths_since_last_delinq", "delinq_2yrs"])
vacio = crudo["mths_since_last_delinq"].isna()
print(f"Vacíos: {vacio.mean():.1%} de {len(crudo):,}")
print("delinq_2yrs > 0 entre los vacíos:", f"{(crudo.loc[vacio, 'delinq_2yrs'] > 0).mean():.2%}",
      "(si vacío fuera 'nunca hubo mora', esta proporción debería ser 0)")
