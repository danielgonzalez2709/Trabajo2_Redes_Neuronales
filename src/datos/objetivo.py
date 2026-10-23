"""Variable objetivo de PD a partir de `loan_status` (Parte A §3.3; tabla literal del Anexo del núcleo, Parte 1).

1 = malo (incumple), 0 = bueno, NA = sin certeza sobre el comportamiento final (no entra al entrenamiento).
Un estado que no esté en la tabla es un ERROR, no un NA silencioso: si el CSV trae un texto distinto, hay que
mirarlo y decidir, no adivinar.
"""

from __future__ import annotations

import pandas as pd

# Copia literal de la tabla del enunciado (Anexo del núcleo, Parte 1, "Variable objetivo").
MAPEO_LOAN_STATUS: dict[str, int | None] = {
    "Fully Paid": 0,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Does not meet the credit policy. Status:Charged Off": 1,
    "Current": None,
    "Issued": None,
    "In Grace Period": None,
    "Late (16-30 days)": None,
}


def construir_objetivo(loan_status: pd.Series) -> pd.Series:
    """Devuelve una serie ``Int8`` con 0/1 y ``<NA>`` según ``MAPEO_LOAN_STATUS``.

    Lanza ``ValueError`` si aparece un estado fuera de la tabla (incluidos los vacíos).
    """
    desconocidos = sorted(set(loan_status.dropna().unique()) - set(MAPEO_LOAN_STATUS), key=str)
    if loan_status.isna().any():
        desconocidos.append("<vacío>")
    if desconocidos:
        raise ValueError(f"loan_status fuera de la tabla del enunciado: {desconocidos}")
    return loan_status.map(MAPEO_LOAN_STATUS).astype("Int8")
