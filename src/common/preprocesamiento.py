"""Preprocesador común (contrato del núcleo §5.2). Se ajusta SOLO con train y lo usan todos los modelos de una tarea.

Por variable numérica, en este orden (todo con estadísticos de train):
    1. imputación por la mediana
    2. recorte a los percentiles [p_inf, p_sup]
    3. log1p, si la variable lo declara (`transformacion: log1p`)
    4. estandarización (x − media) / desv
Por variable categórica: vacío → "DESCONOCIDO"; categoría no vista en train → "OTRA"; one-hot con las categorías de
train + "OTRA" (columna `nombre=categoria`).

Todas las operaciones son reimplementables en TypeScript con `to_dict()` (la app las replica; prueba de paridad §5.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DESCONOCIDO = "DESCONOCIDO"
OTRA = "OTRA"
VERSION = 1


class Preprocesador:
    """Se ajusta SOLO con train. Mismo objeto para todos los modelos de una tarea (regla de comparación justa)."""

    def __init__(self, percentiles_recorte: tuple[float, float] = (1.0, 99.0)):
        self.percentiles_recorte = tuple(float(p) for p in percentiles_recorte)
        self.variables: list[dict] = []

    # ------------------------------------------------------------------ ajuste
    def fit(self, df: pd.DataFrame, features: list[dict]) -> "Preprocesador":
        p_inf, p_sup = self.percentiles_recorte
        self.variables = []
        for f in features:
            nombre, tipo = f["nombre"], f["tipo"]
            if tipo == "num":
                x = pd.to_numeric(df[nombre], errors="raise").astype(float)
                mediana = float(x.median())
                x = x.fillna(mediana)
                lo, hi = (float(v) for v in np.percentile(x, [p_inf, p_sup]))
                x = x.clip(lo, hi)
                log1p = f.get("transformacion") == "log1p"
                if log1p:
                    if lo <= -1:
                        raise ValueError(f"{nombre}: log1p requiere valores > -1 y el recorte inferior es {lo}")
                    x = np.log1p(x)
                desv = float(x.std(ddof=0))
                self.variables.append({
                    "nombre": nombre, "tipo": "num", "mediana": mediana, "recorte": [lo, hi], "log1p": log1p,
                    "media": float(x.mean()), "desv": desv if desv > 0 else 1.0,
                })
            elif tipo == "cat":
                vistas = sorted(self._limpiar_cat(df[nombre]).unique())
                self.variables.append({"nombre": nombre, "tipo": "cat",
                                       "categorias": [c for c in vistas if c != OTRA] + [OTRA]})
            else:
                raise ValueError(f"{nombre}: tipo desconocido {tipo!r} (num|cat)")
        return self

    # ------------------------------------------------------------- transformar
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.variables:
            raise RuntimeError("Preprocesador sin ajustar: llama fit() con train primero")
        columnas = []
        for v in self.variables:
            if v["tipo"] == "num":
                x = pd.to_numeric(df[v["nombre"]], errors="raise").astype(float).fillna(v["mediana"])
                x = x.clip(*v["recorte"])
                if v["log1p"]:
                    x = np.log1p(x)
                columnas.append(((x - v["media"]) / v["desv"]).to_numpy()[:, None])
            else:
                cats = self._limpiar_cat(df[v["nombre"]])
                cats = cats.where(cats.isin(v["categorias"]), OTRA)
                columnas.append((cats.to_numpy()[:, None] == np.array(v["categorias"])[None, :]).astype(float))
        return np.hstack(columnas).astype(np.float32)

    def nombres_salida(self) -> list[str]:
        nombres = []
        for v in self.variables:
            if v["tipo"] == "num":
                nombres.append(v["nombre"])
            else:
                nombres.extend(f"{v['nombre']}={c}" for c in v["categorias"])
        return nombres

    # ------------------------------------------------------------ serializar
    def to_dict(self) -> dict:
        return {"version": VERSION, "percentiles_recorte": list(self.percentiles_recorte),
                "columnas_salida": self.nombres_salida(), "variables": [dict(v) for v in self.variables]}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocesador":
        if d.get("version") != VERSION:
            raise ValueError(f"versión de preprocesador {d.get('version')!r} no soportada (se espera {VERSION})")
        pre = cls(tuple(d["percentiles_recorte"]))
        pre.variables = [dict(v) for v in d["variables"]]
        return pre

    @staticmethod
    def _limpiar_cat(serie: pd.Series) -> pd.Series:
        return serie.astype("string").str.strip().fillna(DESCONOCIDO).replace("", DESCONOCIDO).astype(object)
