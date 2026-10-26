"""Tablas de la pérdida esperada para el notebook ``05_perdida_esperada`` y el blog (tarea E5; núcleo §5.8: cero lógica
en el notebook).

Solo ordenan lo que ya calculó `run.py el` en ``results/data/el/summary.json`` (o el resumen en memoria de una corrida);
no recalculan nada. El contraste con lo observado es solo de los créditos resueltos (E §3.2); los en mora van aparte con
su pérdida provisional. Montos en USD; tasas, PD y LGD como fracciones. Lo que no aplica a una fila (p. ej. la pérdida
realizada de los en mora) o no está definido (razón sin pérdida realizada, ``None`` en el JSON) aparece como NaN.
"""

from __future__ import annotations

import pandas as pd

# columna de la tabla ← clave del bloque del resumen (src/el/perdida_esperada.py: total, resueltos, en_mora)
COLUMNAS_TOTALES = {"n": "n", "E[L] (USD)": "el", "Pérdida realizada (USD)": "perdida_observada",
                    "E[L] / realizada": "razon_el_sobre_perdida_observada",
                    "Pérdida provisional (USD)": "perdida_provisional", "Tasa de malos": "tasa_malos",
                    "PD media": "pd_media", "LGD media": "lgd_media"}
FILAS_TOTALES = {"Todo el test": "total", "Resueltos (contraste)": "resueltos",
                 "En mora (cobranza abierta)": "en_mora"}
# columna de la tabla por banda ← (bloque, clave); bloque None = la banda misma
COLUMNAS_BANDAS = {
    "Puntaje mín.": (None, "score_min"), "Puntaje máx.": (None, "score_max"),
    "n": ("total", "n"), "E[L] (USD)": ("total", "el"),
    "n resueltos": ("resueltos", "n"), "E[L] resueltos (USD)": ("resueltos", "el"),
    "Pérdida realizada resueltos (USD)": ("resueltos", "perdida_observada"),
    "E[L] / realizada (resueltos)": ("resueltos", "razon_el_sobre_perdida_observada"),
    "n en mora": ("en_mora", "n"), "Pérdida provisional en mora (USD)": ("en_mora", "perdida_provisional"),
    "Tasa de malos": ("total", "tasa_malos"), "PD media": ("total", "pd_media"), "LGD media": ("total", "lgd_media"),
}


def _valor(valor) -> float:
    # None (no aplica o no definido) → NaN para que la columna quede numérica
    return float("nan") if valor is None else valor


def tabla_totales(resumen: dict) -> pd.DataFrame:
    """Filas: todo el test, los resueltos (contraste con la pérdida realizada) y los en mora (pérdida provisional)."""
    filas = {fila: {col: _valor(resumen[bloque].get(clave)) for col, clave in COLUMNAS_TOTALES.items()}
             for fila, bloque in FILAS_TOTALES.items()}
    return pd.DataFrame.from_dict(filas, orient="index")


def tabla_bandas(resumen: dict) -> pd.DataFrame:
    """Una fila por banda de puntaje (índice = número de banda; 1 = menor puntaje, más riesgo): rango de puntaje, n y
    E[L] de la banda, contraste de sus resueltos, sus créditos en mora y la tasa de malos, PD y LGD medias."""
    filas = {b["banda"]: {col: _valor((b if bloque is None else b[bloque])[clave])
                          for col, (bloque, clave) in COLUMNAS_BANDAS.items()}
             for b in resumen["por_banda"]}
    tabla = pd.DataFrame.from_dict(filas, orient="index")
    tabla.index.name = "Banda"
    return tabla
