"""Validación común de entradas para los modelos de LGD (lineal y MLP): mismo comportamiento en ambos."""

from __future__ import annotations

import numpy as np


def a_matriz(X, n_columnas: int | None = None, nombre: str = "X", dtype=np.float32) -> np.ndarray:
    """``X`` como matriz 2D finita de tipo ``dtype`` (float32 para el MLP, float64 para la lineal); si se da
    ``n_columnas``, exige ese número de columnas."""
    X = np.asarray(X, dtype=dtype)
    if X.ndim != 2:
        raise ValueError(f"{nombre} debe ser una matriz 2D; se recibió shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{nombre} tiene valores no finitos (NaN o inf); el preprocesamiento debe imputarlos")
    if n_columnas is not None and X.shape[1] != n_columnas:
        raise ValueError(f"{nombre} tiene {X.shape[1]} columnas y el modelo espera {n_columnas}")
    return X


def a_objetivo(y, n_filas: int, nombre: str = "y", dtype=np.float32) -> np.ndarray:
    """``y`` como vector de tipo ``dtype`` y largo ``n_filas`` con valores finitos en [0, 1] (LGD)."""
    y = np.asarray(y, dtype=dtype).reshape(-1)
    if y.size != n_filas:
        raise ValueError(f"{nombre} tiene {y.size} valores y X tiene {n_filas} filas")
    if not np.all(np.isfinite(y)) or np.any(y < 0.0) or np.any(y > 1.0):
        raise ValueError(f"{nombre} debe estar en [0, 1] (LGD) y sin faltantes")
    return y
