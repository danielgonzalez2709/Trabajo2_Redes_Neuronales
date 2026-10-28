"""Diagnóstico de la sección 10.1 del blog: cuánto de la ventaja del MLP de PD en Brier y log-loss es nivel.

Desplaza el intercepto de cada modelo (suma una constante c a su logit, p' = σ(logit(p) + c)) hasta que su PD media en
prueba iguala la tasa de malos de prueba, y vuelve a medir Brier y log-loss. Usa las etiquetas de prueba para elegir c,
así que es solo un diagnóstico: no es un modelo ni una métrica de prueba válida.

Lee `results/data/pd/pred_test.parquet` (columnas y, pred_logistica, pred_mlp; el MLP es la semilla mediana, 20) y no
escribe nada:

    python scripts/diagnostico_nivel_pd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import log_loss as logloss  # la misma log-loss que usa la Parte B

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.common.metricas import brier  # noqa: E402  (Brier del proyecto, Parte B)

MODELOS = ("logistica", "mlp")


def desplazar(p: np.ndarray, tasa: float) -> np.ndarray:
    """PD con el logit desplazado por la constante c que hace la PD media igual a `tasa`."""
    z = logit(np.clip(p, 1e-15, 1 - 1e-15))
    c = brentq(lambda c: expit(z + c).mean() - tasa, -10.0, 10.0)
    return expit(z + c)


def diagnostico(pred: pd.DataFrame) -> pd.DataFrame:
    y = pred["y"].to_numpy(dtype=float)
    tasa = float(y.mean())
    filas = {}
    for m in MODELOS:
        p = pred[f"pred_{m}"].to_numpy(dtype=float)
        q = desplazar(p, tasa)
        filas[m] = {"pd_media": p.mean(), "brier": brier(y, p), "logloss": logloss(y, p),
                    "brier_nivel_igualado": brier(y, q), "logloss_nivel_igualado": logloss(y, q)}
    tabla = pd.DataFrame(filas).T
    tabla.loc["brecha (logistica - MLP)"] = tabla.loc["logistica"] - tabla.loc["mlp"]
    tabla.attrs["tasa_malos"] = tasa
    return tabla


if __name__ == "__main__":
    tabla = diagnostico(pd.read_parquet(RAIZ / "results" / "data" / "pd" / "pred_test.parquet"))
    print(f"Tasa de malos de prueba: {tabla.attrs['tasa_malos']:.4f}")
    print(tabla.round(4).to_string())
