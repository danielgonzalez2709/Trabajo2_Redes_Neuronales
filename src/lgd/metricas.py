"""Métricas de la LGD (Parte D §3.4; enunciado: "Reporten MAE y RMSE").

    MAE  = (1/n)·Σ|yᵢ − ŷᵢ|
    RMSE = sqrt((1/n)·Σ(yᵢ − ŷᵢ)²)

Ambas exigen dos vectores 1D del mismo largo, no vacíos y finitos (``ValueError`` si no).
"""

from __future__ import annotations

import numpy as np


def _pareja(y, p) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1:
        raise ValueError(f"y y la predicción deben ser vectores 1D; shapes {y.shape} y {p.shape}")
    if y.size != p.size or y.size == 0:
        raise ValueError(f"y y la predicción deben tener el mismo largo > 0; se recibió {y.size} y {p.size}")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        raise ValueError("y y la predicción deben ser finitos (sin NaN ni inf)")
    return y, p


def mae(y, p) -> float:
    y, p = _pareja(y, p)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p) -> float:
    y, p = _pareja(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2)))
