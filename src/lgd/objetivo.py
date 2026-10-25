"""Población, EAD y variable objetivo de la LGD (tarea D1; Parte D §3.1 y §3.2; núcleo §5.1).

Fórmula ([PROPUESTO] en D §3.2, pendiente de verificar contra el diccionario — V-09):

    EAD en el incumplimiento = funded_amnt − total_rec_prncp  (principal pendiente al castigo)
    LGD_sin_recorte = 1 − (recoveries − collection_recovery_fee) / EAD
    LGD = min(1, max(0, LGD_sin_recorte))

La fórmula supone que lo cobrado después del castigo va a ``recoveries`` y no a ``total_rec_prncp``; no es verificable
con estos datos, que no tienen fechas de los pagos.

Población ([PROPUESTO] en D §3.1): solo los créditos castigados, con los dos valores de ``loan_status`` que existen en
``data/processed/montos.parquet``: ``"Charged Off"`` y ``"Does not meet the credit policy. Status:Charged Off"``.
``"Default"`` y ``"Late (31-120 days)"`` se excluyen (su cobranza no ha terminado) y se cuentan en el resumen.

Tratamiento de EAD ≤ 0 — **[DECIDIDO, Camila, 24-oct]**: los créditos con EAD ≤ 0 se **excluyen** de la población
de LGD (no se puede dividir) y se **cuentan** en ``resumen_objetivo`` (``n_ead_no_positiva``). En ``calcular_lgd`` quedan
con ``lgd_sin_recorte = lgd = NaN`` y ``lgd_valida = False``; ``mascara_poblacion_lgd`` los deja fuera. No se imputa.

Tratamiento de un campo faltante (NA) — **[DECIDIDO, Camila, 24-oct]**: igual que EAD ≤ 0 (excluir y contar en
``na_por_campo``). En los datos reales medidos el 24-oct no hay ningún NA en los cuatro campos.

Campos (cita literal de ``LCDataDictionary.xlsx``, hoja ``LoanStats``):
- ``loan_status``: TODO(dato-faltante): cita literal de LCDataDictionary.xlsx, hoja LoanStats, campo loan_status
- ``funded_amnt``: TODO(dato-faltante): cita literal de LCDataDictionary.xlsx, hoja LoanStats, campo funded_amnt
- ``total_rec_prncp``: TODO(dato-faltante): cita literal de LCDataDictionary.xlsx, hoja LoanStats, campo total_rec_prncp
- ``recoveries``: TODO(dato-faltante): cita literal de LCDataDictionary.xlsx, hoja LoanStats, campo recoveries
- ``collection_recovery_fee``: TODO(dato-faltante): cita literal de LCDataDictionary.xlsx, hoja LoanStats, campo
  collection_recovery_fee

(El diccionario no está en ``data/raw/credit-risk-dataset/`` de esta copia del repo; no se parafrasea de memoria.)

Supuesto de tipos: los montos llegan como numéricos (``float64`` en ``montos.parquet``, contrato núcleo §5.1). No se
admiten columnas ``object`` con ``pd.NA``.

Estos campos son **posteriores al desembolso** (núcleo §5.7): solo construyen el objetivo, nunca son variables de un modelo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ESTADOS_LGD = ("Charged Off", "Does not meet the credit policy. Status:Charged Off")
ESTADOS_EXCLUIDOS = ("Default", "Late (31-120 days)")
CAMPOS = ("funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee")
RUTA_MONTOS = Path("data/processed/montos.parquet")


def cargar_montos(ruta: str | Path = RUTA_MONTOS) -> pd.DataFrame:
    """Lee ``montos.parquet`` (única función con E/S del módulo)."""
    return pd.read_parquet(ruta)


def es_poblacion_lgd(loan_status) -> np.ndarray:
    """Máscara booleana: ``True`` exactamente para los dos estados Charged Off (comparación exacta)."""
    return pd.Series(loan_status, dtype=object).isin(ESTADOS_LGD).to_numpy(dtype=bool)


def mascara_poblacion_lgd(montos: pd.DataFrame) -> np.ndarray:
    """Población final de la LGD: estado Charged Off **y** ``lgd_valida`` (EAD > 0 y sin NA).

    Aplica la regla [DECIDIDO, Camila, 24-oct]: EAD ≤ 0 se excluye (y se cuenta en ``resumen_objetivo``).
    """
    return es_poblacion_lgd(montos["loan_status"]) & calcular_lgd(montos)["lgd_valida"].to_numpy(dtype=bool)


def calcular_ead(montos: pd.DataFrame) -> pd.Series:
    """EAD en el incumplimiento = ``funded_amnt − total_rec_prncp`` (principal pendiente al castigo; float, NaN si falta
    un campo)."""
    return (montos["funded_amnt"].astype(float) - montos["total_rec_prncp"].astype(float)).rename("ead")


def calcular_lgd(montos: pd.DataFrame) -> pd.DataFrame:
    """Columnas ``ead``, ``lgd_sin_recorte``, ``lgd`` (recortada a [0, 1]) y ``lgd_valida`` con el índice de ``montos``.

    ``lgd_valida`` es ``False`` (y ``lgd``/``lgd_sin_recorte`` son NaN) si la EAD es ≤ 0 ([DECIDIDO, Camila, 24-oct]:
    excluir y contar) o si falta algún campo de la fórmula ([DECIDIDO, Camila, 24-oct]: ídem). No filtra por estado: eso es
    ``es_poblacion_lgd``; la población final es ``mascara_poblacion_lgd``.
    """
    ead = calcular_ead(montos)
    neto = montos["recoveries"].astype(float) - montos["collection_recovery_fee"].astype(float)
    valida = (ead > 0) & neto.notna()  # NaN > 0 es False: una EAD con NA ya queda fuera
    lgd_sin_recorte = pd.Series(np.nan, index=montos.index, dtype=float)
    lgd_sin_recorte[valida] = 1.0 - neto[valida] / ead[valida]
    return pd.DataFrame({"ead": ead, "lgd_sin_recorte": lgd_sin_recorte, "lgd": lgd_sin_recorte.clip(0.0, 1.0),
                         "lgd_valida": valida.to_numpy(dtype=bool)}, index=montos.index)


def _conteos_poblacion(montos: pd.DataFrame, en_poblacion: np.ndarray, lgd_pob: pd.DataFrame) -> dict:
    """Conteos por estado, EAD ≤ 0, NA y relación entre ``recoveries`` y ``collection_recovery_fee``."""
    estados = montos["loan_status"]
    conocidos = set(ESTADOS_LGD) | set(ESTADOS_EXCLUIDOS)
    otros = estados[~estados.isin(conocidos)].value_counts(dropna=False)
    poblacion = montos[en_poblacion]
    rec = montos["recoveries"].astype(float)
    fee = montos["collection_recovery_fee"].astype(float)
    rec_pob, fee_pob = rec[en_poblacion], fee[en_poblacion]
    return {
        "n_total": int(len(montos)),
        "n_poblacion": int(en_poblacion.sum()),
        "poblacion_por_estado": {e: int((poblacion["loan_status"] == e).sum()) for e in ESTADOS_LGD},
        "excluidos_por_estado": {e: int((estados == e).sum()) for e in ESTADOS_EXCLUIDOS},
        "otros_estados": {str(e): int(n) for e, n in otros.items()},  # ningún estado desaparece sin contarse
        "n_ead_no_positiva": int((lgd_pob["ead"] <= 0).sum()),
        "na_por_campo": {c: int(poblacion[c].isna().sum()) for c in CAMPOS},
        "n_fee_mayor_que_recoveries": int((fee_pob > rec_pob).sum()),
        "n_recoveries_cero_con_fee": int(((rec_pob == 0) & (fee_pob > 0)).sum()),
        "n_recoveries_positivas_fuera_poblacion": int((rec[~en_poblacion] > 0).sum()),
    }


def _distribucion_lgd(validas: pd.DataFrame) -> dict:
    """Recortes y forma de la LGD medida sobre las filas válidas de la población."""
    hay = len(validas) > 0
    return {
        "n_lgd_valida": int(len(validas)),  # población final tras excluir EAD ≤ 0 y NA
        "n_lgd_sin_recorte_menor_0": int((validas["lgd_sin_recorte"] < 0).sum()),
        "n_lgd_sin_recorte_mayor_1": int((validas["lgd_sin_recorte"] > 1).sum()),
        "n_lgd_igual_0": int((validas["lgd"] == 0.0).sum()),
        "n_lgd_igual_1": int((validas["lgd"] == 1.0).sum()),
        "lgd_mediana": float(validas["lgd"].median()) if hay else float("nan"),
        "lgd_media": float(validas["lgd"].mean()) if hay else float("nan"),
    }


def resumen_objetivo(montos: pd.DataFrame) -> dict:
    """Conteos medidos de la población y del objetivo (para verificar V-09 y para el blog)."""
    en_poblacion = es_poblacion_lgd(montos["loan_status"])
    lgd_pob = calcular_lgd(montos[en_poblacion])
    validas = lgd_pob[lgd_pob["lgd_valida"]]
    return {**_conteos_poblacion(montos, en_poblacion, lgd_pob), **_distribucion_lgd(validas)}
