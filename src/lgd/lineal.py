"""Referencia de baja complejidad para la LGD: regresión lineal (Spec, Parte D §3.3; enunciado Parte 2, punto 2).

``LinearRegression`` de scikit-learn sobre el mismo preprocesamiento que el MLP. Puede predecir fuera de [0, 1]:
``predict`` **no recorta**; ``contar_fuera_de_rango`` cuenta cuántas veces ocurre para reportarlo.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from src.lgd.validacion import a_matriz, a_objetivo


class LinealLGD:
    """Implementa ``ModeloLGD`` (núcleo §5.3) con mínimos cuadrados ordinarios."""

    def __init__(self) -> None:
        self.modelo_: LinearRegression | None = None

    def fit(self, X_tr, y_tr, X_val=None, y_val=None, cfg: dict | None = None, seed: int = 0) -> "LinealLGD":
        """Ajusta con train. ``X_val``, ``y_val``, ``cfg`` y ``seed`` no se usan (MCO es determinista y sin
        hiperparámetros); se aceptan para cumplir el contrato común."""
        X_tr = a_matriz(X_tr, nombre="X_tr", dtype=np.float64)
        y_tr = a_objetivo(y_tr, X_tr.shape[0], nombre="y_tr", dtype=np.float64)
        self.modelo_ = LinearRegression().fit(X_tr, y_tr)
        return self

    def _ajustado(self) -> LinearRegression:
        if self.modelo_ is None:
            raise RuntimeError("LinealLGD no está ajustado: llama a fit primero")
        return self.modelo_

    def predict(self, X) -> np.ndarray:
        """Predicción lineal sin recortar, shape (n,)."""
        modelo = self._ajustado()
        X = a_matriz(X, n_columnas=modelo.n_features_in_, dtype=np.float64)
        return np.asarray(modelo.predict(X), dtype=np.float64).reshape(-1)

    def contar_fuera_de_rango(self, X) -> dict:
        """Cuántas predicciones caen fuera de [0, 1]: ``{debajo_de_0, encima_de_1, total, n}``."""
        p = self.predict(X)
        debajo = int(np.sum(p < 0.0))
        encima = int(np.sum(p > 1.0))
        return {"debajo_de_0": debajo, "encima_de_1": encima, "total": debajo + encima, "n": int(p.size)}

    def n_parametros(self) -> int:
        """Coeficientes + intercepto."""
        m = self._ajustado()
        return int(np.size(m.coef_)) + int(np.size(m.intercept_))

    def guardar(self, ruta) -> None:
        ruta = Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, ruta)

    @classmethod
    def cargar(cls, ruta) -> "LinealLGD":
        obj = joblib.load(Path(ruta))
        if not isinstance(obj, cls):
            raise TypeError(f"{ruta} no contiene un {cls.__name__} (contiene {type(obj).__name__})")
        return obj
