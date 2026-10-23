"""Lectura del CSV crudo de Kaggle y variables derivadas de la originación (Parte A §3.1 y §3.4).

Solo lee archivos locales: el crudo se descarga a mano (data/fuentes.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

FORMATO_FECHA = "%b-%Y"  # p. ej. "Dec-2011" (verificado en el CSV el 23-oct-2026)
COLUMNAS_FECHA = ("issue_d", "earliest_cr_line")


def sha256(ruta: str | Path, bloque: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while trozo := f.read(bloque):
            h.update(trozo)
    return h.hexdigest()


def leer_crudo(ruta: str | Path, columnas: list[str] | None = None) -> pd.DataFrame:
    """Lee el CSV de préstamos y convierte las fechas "Mon-YYYY" a datetime. Una fecha ilegible es un error."""
    df = pd.read_csv(ruta, usecols=columnas, low_memory=False)
    for col in COLUMNAS_FECHA:
        if col in df.columns:
            texto = df[col]
            fechas = pd.to_datetime(texto, format=FORMATO_FECHA, errors="coerce")
            ilegibles = fechas.isna() & texto.notna()
            if ilegibles.any():
                raise ValueError(f"{col}: {int(ilegibles.sum())} fechas con formato distinto de {FORMATO_FECHA}")
            df[col] = fechas
    return df


def agregar_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables derivadas que solo usan información de la originación.

    antiguedad_credito_anios = (issue_d − earliest_cr_line) en años: la fecha de la primera línea de crédito se
    conoce al solicitar. Una antigüedad negativa (primera línea posterior al crédito) se deja vacía.
    """
    out = df.copy()
    if {"issue_d", "earliest_cr_line"} <= set(out.columns):
        anios = (out["issue_d"] - out["earliest_cr_line"]).dt.days / 365.25
        out["antiguedad_credito_anios"] = anios.where(anios >= 0)
    return out
