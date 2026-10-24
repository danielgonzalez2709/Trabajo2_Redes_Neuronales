"""Partición temporal por fecha de originación y muestra para el modo demo (Parte A §3.5).

train: issue_d <= train_hasta · val: train_hasta < issue_d <= val_hasta · test: issue_d > val_hasta.
Los cortes vienen de `configs/datos.yaml` (`particion.train_hasta`, `particion.val_hasta`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def asignar_split(issue_d: pd.Series, train_hasta: str, val_hasta: str) -> pd.Series:
    """Etiqueta cada crédito con su split según la fecha de originación. Una fecha faltante es un error."""
    corte_train, corte_val = pd.Timestamp(train_hasta), pd.Timestamp(val_hasta)
    if corte_train >= corte_val:
        raise ValueError(f"train_hasta ({train_hasta}) debe ser anterior a val_hasta ({val_hasta})")
    if issue_d.isna().any():
        raise ValueError(f"{int(issue_d.isna().sum())} créditos sin fecha de originación")
    split = np.where(issue_d <= corte_train, "train", np.where(issue_d <= corte_val, "val", "test"))
    return pd.Series(split, index=issue_d.index, name="split")


def muestra_estratificada(df: pd.DataFrame, n: int, semilla: int, estratos=("split", "target")) -> pd.DataFrame:
    """Muestra de ~n filas que conserva la proporción de cada combinación de estratos (modo demo)."""
    if n >= len(df):
        return df.copy()
    frac = n / len(df)
    rng = np.random.default_rng(semilla)
    partes = [
        grupo.sample(n=max(1, round(len(grupo) * frac)), random_state=rng)
        for _, grupo in df.groupby(list(estratos), observed=True)
    ]
    return pd.concat(partes).sort_index()
