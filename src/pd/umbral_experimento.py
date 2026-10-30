# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Orquestación de B11: de la PD a la decisión (B-pd §3.9)
#
# Valida la config (`configs/pd/umbral.yaml`) y las rutas de escritura **antes** de leer nada, carga los datos de A
# con el preprocesador ya ajustado (sin reajustar), los dos modelos guardados de B5 (`models/pd_logistica.joblib` y
# `models/pd_mlp.pt`, la semilla mediana de `summary.json`), reentrena las `semillas.n` redes de la configuración
# final **sin guardarlas** (solo sus PD), controla la semilla mediana, analiza (trivial, matriz con 0,5, sensibilidad
# al umbral, escenarios $L/G$, curva de costo, metas de sensibilidad y variación entre semillas) y escribe
# `umbral.json` y `pred_umbral.parquet` en `salida` y las 2 figuras en `figuras`.
#
# Complemento que **solo agrega**: nunca escribe en `results/data/pd/` ni en `models/` (guarda de rutas, lección
# H-26) y no fija una política de corte. Todo lo que lee, entrena o escribe se inyecta (`Dependencias`).

# %%
"""Orquestación de B11 (B-pd §3.9): config, guarda de rutas, predicciones, análisis, control y escritura."""

import copy  # copia de la config que va al JSON
import functools  # entrenador por defecto con MLPPD fijado
import math  # √2, √n y +inf
import sys  # stderr de los errores esperados
import time  # reloj por defecto
from collections.abc import Callable  # tipos de las dependencias
from dataclasses import dataclass  # dependencias inyectables
from pathlib import Path  # rutas

import numpy as np  # PD, medias y comparaciones
import pandas as pd  # pred_umbral.parquet

from src.common.metricas import brier  # Brier de los modelos (B1)
from src.common.results import load_json, save_json  # JSON sin NaN y escritura atómica
from src.common.seeds import seeds_for_runs, validar_entero  # semillas efectivas y enteros de la config
from src.pd import experimento as ex  # piezas de B5 (datos, evidencia, entrenar_semilla, resumir, errores)
from src.pd import umbral as u  # lógica pura de B11
from src.pd.logistica import LogisticaPD  # carga de pd_logistica.joblib
from src.pd.mlp import MLPPD  # carga de pd_mlp.pt y reentrenamiento
from src.pd.tablas_umbral import ARCHIVO_UMBRAL, MODELOS  # contrato de umbral.json (un solo lugar)

__all__ = [  # interfaz pública (B-pd §3.9)
    "RAIZ_REPOSITORIO", "ARCHIVO_UMBRAL", "ARCHIVO_PRED", "BLOQUES_UMBRAL", "CLAVES_MEDIDA", "PREFIJO_ERROR",  # nombres
    "TOLERANCIA_SIGMAS", "Dependencias", "validar_rutas", "validar_cfg_umbral", "predicciones",  # config y datos
    "predicciones_semillas", "tolerancia", "dentro_de_tolerancia", "control_semilla_mediana",  # semillas y control
    "analizar", "tabla_pred",  # análisis
    "correr_umbral", "manejador",  # corrida
]


# %% [markdown]
# ## Constantes
#
# - Raíz **fija** del repositorio (no el directorio actual); las rutas de la config son relativas a ella.
# - Lista blanca de escritura (B-pd §3.9): `salida` = `results/data/pd_umbral` o su hermana `pd_umbral_<algo>`;
#   `figuras` = `blog/docs/figures` o una carpeta dentro de `salida`.
# - Claves exactas de la raíz (las de `base.yaml` heredadas + `figuras` y `umbral`), del bloque `umbral` y de las
#   rejillas `{min, max, n}`.
# - `TOLERANCIA_SIGMAS` $= 3$ (convención de tres sigmas de B-pd §8; la tolerancia completa está más abajo).
# - `ARCHIVO_UMBRAL` y `MODELOS` vienen de `src/pd/tablas_umbral.py`, el único lugar del contrato de `umbral.json`.
# - `SENTIDO_DECISION`: en TPR, TNR y tasa de aprobación, mayor es mejor (para `experimento.resumir`).

# %%
RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]  # src/pd/ → raíz
CARPETA_SALIDA = Path("results/data/pd_umbral")  # salida de B11 (y su hermana pd_umbral_<algo>)
CARPETA_FIGURAS_BLOG = Path("blog/docs/figures")  # figuras versionadas del blog
SEPARADOR_HERMANA = "_"  # results/data/pd_umbral_<algo>
ARCHIVO_PRED = "pred_umbral.parquet"  # PD de val y test de los dos modelos
CLAVES_RAIZ = ("base_seed", "datos", "logistica", "mlp", "busqueda", "semillas", "salida", "modelos",  # de base.yaml
               "figuras", "umbral")  # de umbral.yaml
CLAVES_UMBRAL = ("umbral_inicial", "escenarios_razon", "curva_razon", "rejilla_cuotas", "rejilla_umbrales",  # bloque
                 "metas_sensibilidad")  # metas
CLAVES_REJILLA = ("min", "max", "n")  # rejillas y curva de razones
BLOQUES_UMBRAL = ("trivial", "matriz_inicial", "sensibilidad_umbral", "escenarios", "curva_costo", "metas",  # análisis
                  "semillas", "control_semilla_mediana", "config", "tiempo_s", "evidencia")  # orden del JSON
CLAVES_MEDIDA = (*u.CLAVES_MATRIZ, "tpr", "tnr", "tasa_rechazo", "tasa_aprobacion", "costo",  # matriz y tasas
                 "costo_por_credito")  # costo
CLAVES_TEST_SEMILLA = ("tpr_test", "tnr_test", "tasa_aprobacion_test")  # lo que se resume entre semillas
SENTIDO_DECISION = {"tpr": "max", "tnr": "max", "tasa_aprobacion": "max"}  # mayor es mejor
TOLERANCIA_SIGMAS = 3  # tres sigmas (B-pd §8 y §3.9)
CLAVES_REFERENCIA = ("media", "desv", "n")  # campos de summary.json que usa el control
SPLITS = ex.SPLITS_MEDIDOS  # val elige, test se mide
PREFIJO_ERROR = "scripts/umbral_pd.py: error: "  # inicio de la línea de error en stderr


# %% [markdown]
# ## Guarda de rutas: una ruta relativa segura
#
# Texto no vacío, **relativa** (sin raíz ni unidad), sin componentes `..`; se resuelve contra la raíz (siguiendo
# enlaces) y debe quedar dentro de ella. Toda falla → `ValueError` que nombra la clave.

# %%
def _relativa_segura(clave: str, valor, raiz: Path) -> Path:
    """Ruta absoluta y resuelta de ``valor`` dentro de ``raiz``; si no, ValueError que nombra ``clave``."""
    if not isinstance(valor, str) or not valor.strip():  # '' o un número no son rutas
        raise ValueError(f"{clave} debe ser una ruta relativa (texto no vacío); se recibió {valor!r}")  # tipo
    ruta = Path(valor)  # ruta relativa
    if ruta.is_absolute() or ruta.anchor:  # absoluta (o con unidad en Windows)
        raise ValueError(f"{clave}={valor!r} debe ser relativa a la raíz del repositorio")  # absoluta
    if ".." in ruta.parts:  # subir de carpeta
        raise ValueError(f"{clave}={valor!r} no puede contener '..'")  # ..
    absoluta = (raiz / ruta).resolve()  # sigue los enlaces simbólicos
    if absoluta != raiz and raiz not in absoluta.parents:  # un enlace que sale de la raíz
        raise ValueError(f"{clave}={valor!r} sale de la raíz del repositorio {raiz}")  # fuera
    return absoluta  # dentro de la raíz


# %% [markdown]
# ## Guarda de rutas: `salida` y `figuras` (lección H-26)
#
# `salida` resuelta debe ser `results/data/pd_umbral` o una hermana `results/data/pd_umbral_<algo>` (sufijo no vacío);
# `figuras`, `blog/docs/figures` o una carpeta **dentro** de `salida` que no sea `umbral.json` ni
# `pred_umbral.parquet`. Así nunca se escribe en `results/data/pd/` ni en `models/`. La raíz es inyectable.

# %%
def _es_salida_permitida(relativa: Path) -> bool:
    """True si ``relativa`` es results/data/pd_umbral o results/data/pd_umbral_<algo>."""
    # No reusa sensibilidad._en_lista_blanca a propósito: allí se admiten subcarpetas y rutas fuera del repositorio;
    # B11 es más estricto (solo la carpeta o su hermana, siempre dentro de la raíz; B-pd §3.9).
    base = CARPETA_SALIDA.name  # pd_umbral
    nombre = relativa.name  # último componente
    hermana = nombre.startswith(base + SEPARADOR_HERMANA) and len(nombre) > len(base + SEPARADOR_HERMANA)  # sufijo
    return relativa.parent == CARPETA_SALIDA.parent and (nombre == base or hermana)  # ella o su hermana


def validar_rutas(cfg: dict, raiz=None) -> tuple[Path, Path]:
    """(salida, figuras) absolutas tras la lista blanca de escritura de B11; si no, ValueError."""
    raiz = Path(RAIZ_REPOSITORIO if raiz is None else raiz).resolve()  # raíz fija o inyectada
    salida = _relativa_segura("salida", cfg.get("salida"), raiz)  # resultados
    if not _es_salida_permitida(salida.relative_to(raiz)):  # fuera de la lista blanca
        raise ValueError(f"salida={cfg.get('salida')!r}: solo se admite {CARPETA_SALIDA} o "
                         f"{CARPETA_SALIDA}{SEPARADOR_HERMANA}<algo> (B-pd §3.9)")  # nombra la clave
    figuras = _relativa_segura("figuras", cfg.get("figuras"), raiz)  # figuras
    if figuras.relative_to(raiz) == CARPETA_FIGURAS_BLOG:  # las del blog
        return salida, figuras  # válidas
    if salida not in figuras.parents:  # ni el blog ni dentro de salida
        raise ValueError(f"figuras={cfg.get('figuras')!r}: solo se admite {CARPETA_FIGURAS_BLOG} o una carpeta "
                         f"dentro de salida (B-pd §3.9)")  # nombra la clave
    if figuras.relative_to(salida).parts[0] in (ARCHIVO_UMBRAL, ARCHIVO_PRED):  # pisaría un resultado
        raise ValueError(f"figuras={cfg.get('figuras')!r} pisaría un archivo de resultados de salida")  # choque
    return salida, figuras  # válidas


# %% [markdown]
# ## Validación: listas de reales y enteros
#
# Una lista no vacía de reales en un intervalo (con `umbral.validar_real`) y un entero $\ge$ mínimo (con
# `seeds.validar_entero`, su `TypeError` como `ValueError`).

# %%
def _lista_reales(valor, nombre: str, minimo: float, maximo: float, abierto_abajo: bool = False) -> list[float]:
    """Lista no vacía de reales validados; si no, ValueError."""
    if not isinstance(valor, list) or not valor:  # lista no vacía
        raise ValueError(f"umbral.{nombre} debe ser una lista no vacía; se recibió {valor!r}")  # forma
    return [u.validar_real(v, f"umbral.{nombre}[{i}]", minimo, maximo, abierto_abajo)  # cada valor
            for i, v in enumerate(valor)]  # en orden


def _entero(valor, nombre: str, minimo: int) -> int:
    """Entero ≥ minimo (seeds.validar_entero) con TypeError → ValueError."""
    try:  # regla única de enteros del proyecto
        return validar_entero(valor, nombre, minimo=minimo)  # ValueError si es < minimo
    except TypeError as exc:  # flotante, bool, texto…
        raise ValueError(str(exc)) from exc  # mismo mensaje


# %% [markdown]
# ## Validación: rejillas `{min, max, n}`
#
# Claves exactas. **Lineal** (cuotas y umbrales): $0 \le \min < \max \le 1$, $n \ge 2$, valores
# `linspace(min, max, n)` redondeados a 9 decimales (los de `umbral.DECIMALES_K`, para que 0,07 sea 0,07).
# **Logarítmica** (razones $L/G$): $0 < \min < \max < \infty$, `geomspace(min, max, n)`.

# %%
def _bloque_rejilla(bloque, nombre: str) -> dict:
    """{min, max, n} con claves exactas y n entero ≥ 2."""
    if not isinstance(bloque, dict) or set(bloque) != set(CLAVES_REJILLA):  # claves exactas
        raise ValueError(f"umbral.{nombre} debe ser un mapeo con exactamente {list(CLAVES_REJILLA)}; "
                         f"se recibió {bloque!r}")  # forma
    return {**bloque, "n": _entero(bloque["n"], f"umbral.{nombre}.n", minimo=2)}  # n validado


def _rejilla_lineal(bloque, nombre: str) -> np.ndarray:
    """linspace(min, max, n) en [0, 1] redondeado a 9 decimales."""
    b = _bloque_rejilla(bloque, nombre)  # claves y n
    lo = u.validar_real(b["min"], f"umbral.{nombre}.min", 0.0, 1.0)  # mínimo
    hi = u.validar_real(b["max"], f"umbral.{nombre}.max", 0.0, 1.0)  # máximo
    if lo >= hi:  # rejilla degenerada
        raise ValueError(f"umbral.{nombre}: min ({lo}) debe ser menor que max ({hi})")  # orden
    return u.validar_rejilla(np.round(np.linspace(lo, hi, b["n"]), u.DECIMALES_K), f"umbral.{nombre}")  # rejilla


def _rejilla_log(bloque, nombre: str) -> np.ndarray:
    """geomspace(min, max, n) con 0 < min < max < ∞."""
    b = _bloque_rejilla(bloque, nombre)  # claves y n
    lo = u.validar_real(b["min"], f"umbral.{nombre}.min", 0.0, math.inf, abierto_abajo=True)  # mínimo > 0
    hi = u.validar_real(b["max"], f"umbral.{nombre}.max", 0.0, math.inf, abierto_abajo=True)  # máximo > 0
    if lo >= hi:  # rejilla degenerada
        raise ValueError(f"umbral.{nombre}: min ({lo}) debe ser menor que max ({hi})")  # orden
    return np.geomspace(lo, hi, b["n"])  # rejilla logarítmica


# %% [markdown]
# ## Validación: bloque `umbral`
#
# Claves exactas; umbral inicial en $[0, 1]$; razones $L/G > 0$ finitas; rejillas y metas en $[0, 1]$.

# %%
def _validar_bloque_umbral(b) -> dict:
    """Bloque umbral validado (rejillas como arreglos)."""
    if not isinstance(b, dict):  # p. ej. umbral: 0.5
        raise ValueError(f"El bloque umbral debe ser un mapeo; se recibió {b!r}")  # tipo
    if set(b) != set(CLAVES_UMBRAL):  # claves exactas
        faltan, sobran = set(CLAVES_UMBRAL) - set(b), set(b) - set(CLAVES_UMBRAL)  # diferencias
        raise ValueError(f"El bloque umbral es inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # cuáles
    return {  # en el orden de CLAVES_UMBRAL
        "umbral_inicial": u.validar_real(b["umbral_inicial"], "umbral.umbral_inicial", 0.0, 1.0),  # 0,5
        "escenarios_razon": _lista_reales(b["escenarios_razon"], "escenarios_razon", 0.0, math.inf, True),  # L/G
        "curva_razon": _rejilla_log(b["curva_razon"], "curva_razon"),  # L/G de la curva
        "rejilla_cuotas": _rejilla_lineal(b["rejilla_cuotas"], "rejilla_cuotas"),  # cuotas
        "rejilla_umbrales": _rejilla_lineal(b["rejilla_umbrales"], "rejilla_umbrales"),  # umbrales
        "metas_sensibilidad": _lista_reales(b["metas_sensibilidad"], "metas_sensibilidad", 0.0, 1.0),  # metas
    }


# %% [markdown]
# ## Validación de la config completa
#
# Antes de leer datos: raíz con claves exactas, `datos.muestra` null (B11 usa todos los datos), los bloques de B5
# (`semillas`, `base_seed`, `mlp`, `busqueda`) con `experimento.validar_cfg_mlp_final`, el bloque `umbral`, la guarda
# de `salida` y `figuras` y `modelos` (solo lectura) relativa y dentro de la raíz.

# %%
def validar_cfg_umbral(cfg: dict, raiz=None) -> dict:
    """{n, base_seed, mlp, umbral, salida, figuras, modelos} validados; cualquier falla → ValueError."""
    if not isinstance(cfg, dict):  # sin config no hay corrida
        raise ValueError(f"La config de B11 debe ser un mapeo; se recibió {type(cfg).__name__}")  # tipo
    faltan, sobran = set(CLAVES_RAIZ) - set(cfg), set(cfg) - set(CLAVES_RAIZ)  # claves exactas
    if faltan or sobran:  # de más o de menos
        raise ValueError(f"La config de B11 es inválida: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # cuáles
    if ex.validar_datos(cfg) is not None:  # muestra
        raise ValueError("datos.muestra debe ser null: B11 usa todos los datos (B-pd §3.9)")  # muestra
    n, base_seed, mlp, _ = ex.validar_cfg_mlp_final(cfg)  # bloques de B5
    umbral = _validar_bloque_umbral(cfg["umbral"])  # bloque umbral
    raiz = Path(RAIZ_REPOSITORIO if raiz is None else raiz).resolve()  # raíz fija o inyectada
    salida, figuras = validar_rutas(cfg, raiz)  # lista blanca de escritura
    modelos = _relativa_segura("modelos", cfg["modelos"], raiz)  # solo lectura
    return {"n": n, "base_seed": base_seed, "mlp": mlp, "umbral": umbral, "salida": salida, "figuras": figuras,
            "modelos": modelos}  # todo validado


# %% [markdown]
# ## Dependencias inyectables
#
# Por defecto, las reales: `experimento.cargar_datos` (partición y preprocesador de A, sin reajustar),
# `LogisticaPD.cargar`, `MLPPD.cargar`, `experimento.entrenar_semilla` con `MLPPD`, `load_json`,
# `experimento.evidencia`, `experimento.huella_archivo`, `save_json`, `experimento.escribir_parquet_atomico`, las
# figuras de `src/viz/figs_umbral_pd.py` y `time.perf_counter`. Las pruebas pasan versiones falsas.

# %%
def _generar_figuras(destino: Path, datos: dict) -> list[Path]:
    """figs_umbral_pd.generar(destino=…, datos=…) (import diferido: matplotlib solo al dibujar)."""
    from src.viz import figs_umbral_pd as figs_umbral  # figuras de B11
    return figs_umbral.generar(destino=destino, datos=datos)  # las dos figuras


@dataclass(frozen=True)
class Dependencias:
    """Todo lo que la corrida lee, entrena o escribe."""

    cargar_datos: Callable = ex.cargar_datos  # rutas → (df, {split: (X, y, ids)})
    cargar_logistica: Callable = LogisticaPD.cargar  # ruta → modelo con predict_proba
    cargar_mlp: Callable = MLPPD.cargar  # ruta → modelo con predict_proba
    entrenar: Callable = functools.partial(ex.entrenar_semilla, MLPPD)  # (datos, cfg_mlp, seed, reloj) → (m, t)
    leer_summary: Callable = load_json  # ruta → summary.json
    evidencia: Callable = ex.evidencia  # (df, ruta_preprocesador) → huellas
    huella: Callable = ex.huella_archivo  # ruta → SHA-256
    escribir_json: Callable = save_json  # (ruta, obj) → ruta
    escribir_parquet: Callable = ex.escribir_parquet_atomico  # (ruta, df) → None
    generar_figuras: Callable = _generar_figuras  # (destino, datos) → rutas
    reloj: Callable = time.perf_counter  # segundos


# %% [markdown]
# ## Semilla mediana de `summary.json`
#
# `summary["mlp"]["semilla_mediana"]` entero $\ge 0$ (la de `models/pd_mlp.pt`, B-pd §3.5); nunca se escribe a mano.

# %%
def _semilla_mediana(summary) -> int:
    """Semilla mediana del MLP en summary.json; si falta o no es entera, ValueError."""
    mlp = summary.get("mlp") if isinstance(summary, dict) else None  # entrada mlp
    seed = mlp.get("semilla_mediana") if isinstance(mlp, dict) else None  # semilla
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:  # entero ≥ 0
        raise ValueError(f"summary.json debe tener mlp.semilla_mediana entera ≥ 0; se leyó {seed!r} "
                         "(corre run.py pd)")  # cómo arreglarlo
    return seed  # semilla de pd_mlp.pt


# %% [markdown]
# ## PD de los modelos guardados
#
# `{split: {y, ids, logistica, mlp}}` en val y test, con las matrices de A en el orden del parquet.

# %%
def predicciones(matrices: dict, modelos: dict) -> dict:
    """{split: {"y", "ids", modelo: PD}} de val y test."""
    resultado = {}  # por split
    for split in SPLITS:  # val y test
        X, y, ids = matrices[split]  # matrices del split
        resultado[split] = {"y": np.asarray(y), "ids": np.asarray(ids, dtype=object),  # objetivo e ids
                            **{m: np.asarray(modelos[m].predict_proba(X), dtype=float) for m in MODELOS}}  # PD
    return resultado  # val y test


# %% [markdown]
# ## PD de las semillas reentrenadas
#
# Para cada semilla efectiva $s$ = `base_seed + k`: `entrenar(matrices, cfg_mlp, s, reloj)` (por defecto
# `experimento.entrenar_semilla` con un `MLPPD` nuevo) y sus PD de val y test. **No se guardan las redes**.

# %%
def predicciones_semillas(matrices: dict, cfg_mlp: dict, semillas: list[int], entrenar, reloj) -> list[dict]:
    """[{seed, val, test, tiempo_s}] con las PD de cada red reentrenada."""
    resultado = []  # una entrada por semilla
    for seed in semillas:  # base_seed + 0…n−1
        modelo, tiempo_s = entrenar(matrices, cfg_mlp, seed, reloj)  # red nueva, entrenada
        resultado.append({"seed": seed, "tiempo_s": float(tiempo_s),  # semilla y fit
                          **{s: np.asarray(modelo.predict_proba(matrices[s][0]), dtype=float) for s in SPLITS}})  # PD
    return resultado  # la red se descarta


# %% [markdown]
# ## Tolerancia entre máquinas (B-pd §3.9 y §8)
#
# Para cada métrica de prueba $m$, con $\bar x_m$ la media de las $k$ semillas reentrenadas y $(\bar x^{\text{ref}}_m,
# s_m, n_m)$ la media, desviación y número de corridas de `summary.json`:
#
# $$|\bar x_m - \bar x^{\text{ref}}_m| \le 3\sqrt{\frac{s_m^2}{k} + \frac{s_m^2}{n_m}}$$
#
# (error estándar de la diferencia de dos medias independientes con la misma varianza). Con $k = n_m$ es el
# $3\sqrt2\, s_m/\sqrt{n_m}$ de §8; con $k < n_m$ la tolerancia crece. Con $s_m = 0$ solo pasa $\Delta = 0$.

# %%
def tolerancia(desv: float, n: int, k: int) -> float:
    """3·√(desv²/k + desv²/n)."""
    return TOLERANCIA_SIGMAS * math.sqrt(desv ** 2 / k + desv ** 2 / n)  # tres errores estándar de la diferencia


# %% [markdown]
# La entrada de referencia de cada métrica se valida antes de usarla: media y desv reales finitos (desv $\ge 0$) y
# $n$ entero $\ge 1$. Un `summary.json` dañado (p. ej. $n = 0$ o media `null`) es un error **esperado**
# (`ValueError`, código 2 en el script), no un `ZeroDivisionError` ni un `TypeError` con traza.

# %%
def _referencia(referencia, clave: str) -> dict:
    """{media, desv, n} de summary.json para ``clave`` validados; si no, ValueError."""
    ref = referencia.get(clave) if isinstance(referencia, dict) else None  # resumen de la métrica
    if not isinstance(ref, dict) or not set(CLAVES_REFERENCIA) <= set(ref):  # sin referencia
        raise ValueError(f"summary.json no tiene mlp.metricas.{clave} con {list(CLAVES_REFERENCIA)}")  # falta
    nombre = f"summary.json mlp.metricas.{clave}"  # prefijo de los mensajes
    media = u.validar_real(ref["media"], f"{nombre}.media", -math.inf, math.inf)  # real finito
    desv = u.validar_real(ref["desv"], f"{nombre}.desv", 0.0, math.inf)  # real finito ≥ 0
    n = _entero(ref["n"], f"{nombre}.n", minimo=1)  # entero ≥ 1
    return {"media": media, "desv": desv, "n": n}  # validados


def dentro_de_tolerancia(medias: dict, referencia: dict, k: int) -> bool:
    """True si cada media de k semillas queda a ≤ 3·√(s²/k + s²/n) de la de referencia."""
    k = _entero(k, "k (semillas reentrenadas)", minimo=1)  # al menos una semilla
    refs = {clave: _referencia(referencia, clave) for clave in medias}  # todas validadas antes de comparar
    return all(abs(media - refs[c]["media"]) <= tolerancia(refs[c]["desv"], refs[c]["n"], k)  # dentro
               for c, media in medias.items())  # cada métrica de prueba


# %% [markdown]
# ## Control de la semilla mediana
#
# Si la semilla mediana está entre las reentrenadas, sus PD (val y test) se comparan con las de `models/pd_mlp.pt`:
# `identica_bit_a_bit`, `max_abs_dif` y `dentro_tolerancia` = idéntica, o bien medias de las métricas de prueba
# (AUC, Gini, KS, Brier, log-loss) de las $k$ semillas reentrenadas dentro de la tolerancia de arriba frente a
# `summary.json`. La referencia se valida siempre (también si es idéntica). Fuera de tolerancia → `RuntimeError`.
# Si no está (demo con pocas semillas) → `None`.

# %%
def control_semilla_mediana(sem_preds: list[dict], seed_med: int, preds: dict, referencia: dict) -> dict | None:
    """{identica_bit_a_bit, max_abs_dif, dentro_tolerancia} o None; fuera de tolerancia → RuntimeError."""
    reentrenada = next((s for s in sem_preds if s["seed"] == seed_med), None)  # la mediana, si está
    if reentrenada is None:  # demo con pocas semillas
        return None  # sin control
    identica = all(np.array_equal(reentrenada[s], preds[s][ex.MODELO_MLP]) for s in SPLITS)  # bit a bit
    dif = max(float(np.max(np.abs(reentrenada[s] - preds[s][ex.MODELO_MLP]))) for s in SPLITS)  # máx |Δp|
    y_te = preds["test"]["y"]  # objetivo de prueba
    por_semilla = [ex.evaluar(y_te, s["test"]) for s in sem_preds]  # métricas de prueba por semilla
    medias = {f"{k}_test": float(np.mean([m[k] for m in por_semilla])) for k in ex.METRICAS}  # medias
    for clave in medias:  # summary.json dañado → error esperado aunque sea idéntica
        _referencia(referencia, clave)  # valida media, desv y n
    dentro = identica or dentro_de_tolerancia(medias, referencia, k=len(sem_preds))  # control
    if not dentro:  # no reproduce el modelo final
        raise RuntimeError(f"La semilla mediana {seed_med} reentrenada no reproduce pd_mlp.pt: medias de prueba "
                           f"{medias} fuera de la tolerancia 3·√(s²/k + s²/n) de summary.json "  # tolerancia
                           f"(máx |Δp| = {dif:.3g})")  # error
    return {"identica_bit_a_bit": bool(identica), "max_abs_dif": dif, "dentro_tolerancia": bool(dentro)}  # control


# %% [markdown]
# ## Medida de una política
#
# Matriz de confusión + tasas (+ costo si hay razón $L/G$) de una máscara de rechazo.

# %%
def _medida(y, rechazo, razon=None) -> dict:
    """{vp, fn, fp, vn, tpr, tnr, tasa_rechazo, tasa_aprobacion[, costo, costo_por_credito]}."""
    m = u.matriz_confusion(y, rechazo)  # conteos
    return {**m, **u.tasas(m), **(u.costo(m, razon) if razon is not None else {})}  # todo junto


# %% [markdown]
# ## Una política de cuota: de validación a prueba
#
# Con la cuota $r$ elegida en val: medida en val, en test **como cuota** (`rechazos_por_cuota(p_test, r)`, política
# por lotes) y en test con el **umbral de val** $t_{\text{val}}$ = `umbral_de_cuota(p_val, r)` tal cual
# (`p_test ≥ t_val`, contraste de PD absoluta). $t_{\text{val}} = +\infty$ (cuota 0) se guarda como `null`.

# %%
def _politica(pv: dict, pt: dict, modelo: str, cuota: float, razon=None) -> dict:
    """{cuota, umbral_val, val, test_cuota, test_umbral} de la cuota ``cuota``."""
    p_val, p_te = pv[modelo], pt[modelo]  # PD de val y test
    t_val = u.umbral_de_cuota(p_val, cuota)  # PD absoluta equivalente en val
    return {  # política y su contraste
        "cuota": cuota,  # cuota elegida en val
        "umbral_val": None if math.isinf(t_val) else t_val,  # +inf → null (JSON)
        "val": _medida(pv["y"], u.rechazos_por_cuota(p_val, cuota), razon),  # en val
        "test_cuota": _medida(pt["y"], u.rechazos_por_cuota(p_te, cuota), razon),  # en test, como cuota
        "test_umbral": _medida(pt["y"], u.rechazos_por_umbral(p_te, t_val), razon),  # en test, PD absoluta
    }


# %% [markdown]
# ## Bloque `trivial`
#
# PD constante $q$: principal $q$ = tasa de **train**; piso teórico $q$ = tasa de **test** (usa etiquetas de
# prueba). Brier en val y test con `umbral.brier_trivial` y, para comparar, el Brier de los dos modelos.

# %%
def _trivial(preds: dict, tasa_train: float) -> dict:
    """{tasas, principal, piso_teorico, modelos}."""
    tasas = {"train": tasa_train, **{s: float(np.mean(preds[s]["y"])) for s in SPLITS}}  # π por split
    def con_q(q: float) -> dict:  # Brier del trivial con PD q
        return {"q": q, **{f"brier_{s}": u.brier_trivial(preds[s]["y"], q) for s in SPLITS}}  # val y test
    return {"tasas": tasas, "principal": con_q(tasa_train), "piso_teorico": con_q(tasas["test"]),  # triviales
            "modelos": {m: {f"brier_{s}": brier(preds[s]["y"], preds[s][m]) for s in SPLITS}  # Brier del modelo
                        for m in MODELOS}}  # los dos modelos


# %% [markdown]
# ## Bloques `matriz_inicial` y `sensibilidad_umbral`
#
# Matriz con el umbral inicial (0,5) en val y test; TPR, TNR y tasa de rechazo en la rejilla de umbrales.

# %%
def _matriz_inicial(preds: dict, t: float) -> dict:
    """{umbral, modelos: {m: {val, test}}} con p ≥ t."""
    return {"umbral": t, "modelos": {m: {s: _medida(preds[s]["y"], u.rechazos_por_umbral(preds[s][m], t))  # medida
                                         for s in SPLITS} for m in MODELOS}}  # val y test


def _sensibilidad(preds: dict, umbrales: np.ndarray) -> dict:
    """{umbrales, modelos: {m: {split: {tpr, tnr, tasa_rechazo}}}}."""
    modelos = {}  # por modelo
    for m in MODELOS:  # logística y MLP
        modelos[m] = {}  # por split
        for s in SPLITS:  # val y test
            curva = u.curva_sensibilidad(preds[s]["y"], preds[s][m], umbrales)  # curva
            modelos[m][s] = {k: curva[k] for k in ("tpr", "tnr", "tasa_rechazo")}  # sin el eje repetido
    return {"umbrales": [float(t) for t in umbrales], "modelos": modelos}  # eje y curvas


# %% [markdown]
# ## Conteos de validación por modelo
#
# `umbral.conteos_por_cuota` sobre las PD de val de cada modelo, **una sola vez** (un ordenamiento por modelo): los
# reutilizan todas las razones (escenarios y curva) y todas las metas con `cuota_optima_conteos` y
# `cuota_para_meta_conteos`, que dan lo mismo que las versiones directas (lo exige una prueba).

# %%
def _conteos_val(preds: dict, cuotas: np.ndarray) -> dict:
    """{modelo: conteos_por_cuota en val}."""
    pv = preds["val"]  # validación
    return {m: u.conteos_por_cuota(pv["y"], pv[m], cuotas) for m in MODELOS}  # uno por modelo


# %% [markdown]
# ## Bloques `escenarios` y `curva_costo`
#
# Para cada $L/G$: $p^\* = 1/(1 + L/G)$ y, por modelo, la cuota $r^\*$ que minimiza el costo observado en **val**
# (empate → la menor) con su política (`_politica`). La curva repite lo mismo en la rejilla logarítmica y guarda
# listas para la figura: cada columna de `COLUMNAS_CURVA` es una ruta dentro de la política (una clave o una clave y
# una subclave).

# %%
COLUMNAS_CURVA = {  # columna de curva_costo → ruta dentro de la política del escenario
    "cuota": ("cuota",),  # r*
    "umbral_val": ("umbral_val",),  # t_val
    "tasa_rechazo_test_cuota": ("test_cuota", "tasa_rechazo"),  # rechazo en prueba por cuota
    "tasa_rechazo_test_umbral": ("test_umbral", "tasa_rechazo"),  # rechazo en prueba con el umbral de val
    "costo_por_credito_val": ("val", "costo_por_credito"),  # costo en val
    "costo_por_credito_test_cuota": ("test_cuota", "costo_por_credito"),  # costo en prueba por cuota
    "costo_por_credito_test_umbral": ("test_umbral", "costo_por_credito"),  # costo en prueba con el umbral de val
}


def _escenario(preds: dict, razon: float, cuotas: np.ndarray, conteos: dict) -> dict:
    """{razon, p_estrella, modelos: {m: política con costo}} con los conteos de val ya calculados."""
    pv, pt = preds["val"], preds["test"]  # atajos
    return {"razon": razon, "p_estrella": u.umbral_teorico(razon),  # referencia teórica
            "modelos": {m: _politica(pv, pt, m, u.cuota_optima_conteos(conteos[m], cuotas, razon), razon)  # política
                        for m in MODELOS}}  # dos modelos


def _valor(politica: dict, ruta: tuple[str, ...]):
    """El valor de ``politica`` en la ruta de claves ``ruta``."""
    valor = politica  # empieza en la política
    for clave in ruta:  # baja por la ruta
        valor = valor[clave]  # siguiente nivel
    return valor  # hoja


def _curva_costo(preds: dict, razones: np.ndarray, cuotas: np.ndarray, conteos: dict) -> dict:
    """{razones, p_estrella, modelos: {m: listas}} para la figura de costo."""
    escenarios = [_escenario(preds, float(r), cuotas, conteos) for r in razones]  # uno por razón
    return {"razones": [e["razon"] for e in escenarios], "p_estrella": [e["p_estrella"] for e in escenarios],  # eje
            "modelos": {m: {col: [_valor(e["modelos"][m], ruta) for e in escenarios]  # una lista por columna
                            for col, ruta in COLUMNAS_CURVA.items()} for m in MODELOS}}  # dos modelos


# %% [markdown]
# ## Bloque `metas`
#
# Para cada meta $m$: la menor cuota de la rejilla con $\mathrm{TPR}_{\text{val}} \ge m$ y su efecto en prueba; si
# ninguna la alcanza, la política queda en `null`.

# %%
def _metas(preds: dict, metas: list[float], cuotas: np.ndarray, conteos: dict) -> list[dict]:
    """[{meta, modelos: {m: política sin costo o nulos}}]."""
    pv, pt = preds["val"], preds["test"]  # atajos
    resultado = []  # una por meta
    for meta in metas:  # 0,5 / 0,7 / 0,8 / 0,9
        modelos = {}  # por modelo
        for m in MODELOS:  # logística y MLP
            r = u.cuota_para_meta_conteos(conteos[m], cuotas, meta)  # menor cuota que alcanza la meta
            modelos[m] = _politica(pv, pt, m, r) if r is not None else dict.fromkeys(  # política o nulos
                ("cuota", "umbral_val", "val", "test_cuota", "test_umbral"))  # inalcanzable
        resultado.append({"meta": meta, "modelos": modelos})  # meta
    return resultado  # metas


# %% [markdown]
# ## Bloque `semillas`
#
# Cada red reentrenada elige **su** cuota en val (por escenario y por meta) y se mide en test como cuota; se resumen
# TPR, TNR y tasa de aprobación de prueba con `experimento.resumir` (media, desv. muestral, mejor, peor, n; mayor es
# mejor). Una meta que alguna semilla no alcanza → resumen `null`.

# %%
def _decision_semilla(yv, pv, yt, pt, cuota) -> dict:
    """{cuota, tpr_test, tnr_test, tasa_aprobacion_test} (cuota None → valores None)."""
    if cuota is None:  # meta inalcanzable
        return {"cuota": None, **dict.fromkeys(CLAVES_TEST_SEMILLA)}  # nulos
    t = u.tasas(u.matriz_confusion(yt, u.rechazos_por_cuota(pt, cuota)))  # tasas de test
    return {"cuota": cuota, "tpr_test": t["tpr"], "tnr_test": t["tnr"],  # cuota, TPR y TNR
            "tasa_aprobacion_test": t["tasa_aprobacion"]}  # aprobación


def _resumen(decisiones: list[dict]) -> dict | None:
    """experimento.resumir de CLAVES_TEST_SEMILLA o None si falta alguna."""
    if any(d["cuota"] is None for d in decisiones):  # alguna semilla no alcanzó
        return None  # sin resumen
    return ex.resumir([{k: d[k] for k in CLAVES_TEST_SEMILLA} for d in decisiones],  # una corrida por semilla
                      sentido=SENTIDO_DECISION)  # mayor es mejor


def _semillas(preds: dict, sem_preds: list[dict], ucfg: dict) -> dict:
    """{n, semillas, tiempo_semillas_s, escenarios, metas, por_semilla}."""
    yv, yt, cuotas = preds["val"]["y"], preds["test"]["y"], ucfg["rejilla_cuotas"]  # atajos
    por_semilla = []  # detalle
    for s in sem_preds:  # cada red
        c = u.conteos_por_cuota(yv, s["val"], cuotas)  # una sola vez por semilla (val)
        esc = [{"razon": r, **_decision_semilla(yv, s["val"], yt, s["test"],  # escenario
                                                u.cuota_optima_conteos(c, cuotas, r))}  # cuota de val
               for r in ucfg["escenarios_razon"]]  # escenarios
        met = [{"meta": m, **_decision_semilla(yv, s["val"], yt, s["test"],  # meta
                                               u.cuota_para_meta_conteos(c, cuotas, m))}  # cuota de val
               for m in ucfg["metas_sensibilidad"]]  # metas
        por_semilla.append({"seed": s["seed"], "escenarios": esc, "metas": met})  # semilla
    return {"n": len(sem_preds), "semillas": [s["seed"] for s in sem_preds],  # semillas
            "tiempo_semillas_s": float(sum(s["tiempo_s"] for s in sem_preds)),  # Σ fit
            "escenarios": [{"razon": r, "resumen": _resumen([p["escenarios"][i] for p in por_semilla])}  # resumen
                           for i, r in enumerate(ucfg["escenarios_razon"])],  # por escenario
            "metas": [{"meta": m, "resumen": _resumen([p["metas"][i] for p in por_semilla])}  # resumen
                      for i, m in enumerate(ucfg["metas_sensibilidad"])],  # por meta
            "por_semilla": por_semilla}  # detalle


# %% [markdown]
# ## Análisis completo
#
# Los bloques de B-pd §3.9 a partir de las PD (sin E/S).

# %%
def analizar(preds: dict, tasa_train: float, sem_preds: list[dict], ucfg: dict) -> dict:
    """{trivial, matriz_inicial, sensibilidad_umbral, escenarios, curva_costo, metas, semillas}."""
    cuotas = ucfg["rejilla_cuotas"]  # rejilla de cuotas
    conteos = _conteos_val(preds, cuotas)  # un ordenamiento por modelo, para todas las razones y metas
    return {  # en el orden de BLOQUES_UMBRAL
        "trivial": _trivial(preds, tasa_train),  # PD constante
        "matriz_inicial": _matriz_inicial(preds, ucfg["umbral_inicial"]),  # umbral 0,5
        "sensibilidad_umbral": _sensibilidad(preds, ucfg["rejilla_umbrales"]),  # curva
        "escenarios": [_escenario(preds, r, cuotas, conteos) for r in ucfg["escenarios_razon"]],  # L/G de la tabla
        "curva_costo": _curva_costo(preds, ucfg["curva_razon"], cuotas, conteos),  # L/G de la figura
        "metas": _metas(preds, ucfg["metas_sensibilidad"], cuotas, conteos),  # metas de sensibilidad
        "semillas": _semillas(preds, sem_preds, ucfg),  # variación entre semillas
    }


# %% [markdown]
# ## `pred_umbral.parquet`
#
# `id`, `split` ∈ {val, test}, `y`, `pred_logistica`, `pred_mlp` (modelos guardados): val y luego test, en el orden
# del parquet de A.

# %%
def tabla_pred(preds: dict) -> pd.DataFrame:
    """DataFrame (id, split, y, pred_logistica, pred_mlp) de val y test."""
    partes = [pd.DataFrame({"id": preds[s]["ids"], "split": s, "y": preds[s]["y"],  # id, split e y
                            **{f"{ex.PREFIJO_PRED}{m}": preds[s][m] for m in MODELOS}}) for s in SPLITS]  # PD
    return pd.concat(partes, ignore_index=True)  # val y test


# %% [markdown]
# ## Corrida
#
# Valida (antes de leer) → lee el summary (semilla mediana) → carga datos y modelos → PD → reentrena las semillas →
# control (puede fallar sin escribir nada) → analiza → escribe JSON y parquet → figuras. `tiempo_s` = desde el inicio
# hasta antes de escribir. Devuelve `{umbral, rutas}`.

# %%
def correr_umbral(cfg: dict, raiz=None, dep: Dependencias | None = None) -> dict:
    """Corrida completa de B11; devuelve {"umbral": resultado, "rutas": archivos escritos}."""
    dep = dep if dep is not None else Dependencias()  # reales por defecto
    raiz = Path(RAIZ_REPOSITORIO if raiz is None else raiz).resolve()  # raíz fija o inyectada
    inicio = dep.reloj()  # comienzo
    v = validar_cfg_umbral(cfg, raiz)  # antes de leer nada
    rutas_pd = ex.RutasPD(raiz / ex.RUTAS_PD.parquet, raiz / ex.RUTAS_PD.preprocesador,  # entradas de A
                          raiz / ex.RUTAS_PD.resultados, v["modelos"])  # summary y modelos (solo lectura)
    summary = dep.leer_summary(rutas_pd.resultados / ex.ARCHIVO_SUMMARY)  # summary.json de B5
    seed_med = _semilla_mediana(summary)  # la de pd_mlp.pt
    ruta_log, ruta_mlp = v["modelos"] / ex.ARCHIVO_MODELO_LOGISTICA, v["modelos"] / ex.ARCHIVO_MODELO_MLP  # modelos
    df, matrices = dep.cargar_datos(rutas_pd)  # datos de A (todos)
    preds = predicciones(matrices, {ex.MODELO_LOGISTICA: dep.cargar_logistica(ruta_log),  # logística
                                    ex.MODELO_MLP: dep.cargar_mlp(ruta_mlp)})  # MLP de la app
    semillas = seeds_for_runs(v["base_seed"], v["n"])  # base_seed + 0…n−1
    sem_preds = predicciones_semillas(matrices, v["mlp"], semillas, dep.entrenar, dep.reloj)  # PD por semilla
    control = control_semilla_mediana(sem_preds, seed_med, preds, summary["mlp"].get("metricas"))  # control
    resultado = analizar(preds, float(np.mean(matrices["train"][1])), sem_preds, v["umbral"])  # bloques
    resultado["control_semilla_mediana"] = control  # control
    resultado["config"] = copy.deepcopy(cfg)  # config resuelta
    evidencia = {**dep.evidencia(df, rutas_pd.preprocesador), "sha256_modelo_logistica": dep.huella(ruta_log),  # A
                 "sha256_modelo_mlp": dep.huella(ruta_mlp)}  # y los dos modelos leídos
    resultado["tiempo_s"] = dep.reloj() - inicio  # tiempo del análisis
    resultado["evidencia"] = evidencia  # huellas
    resultado = {k: resultado[k] for k in BLOQUES_UMBRAL}  # orden fijo
    ruta_json, ruta_pred = v["salida"] / ARCHIVO_UMBRAL, v["salida"] / ARCHIVO_PRED  # destinos
    dep.escribir_json(ruta_json, resultado)  # umbral.json (sin NaN)
    dep.escribir_parquet(ruta_pred, tabla_pred(preds))  # pred_umbral.parquet
    figuras = dep.generar_figuras(v["figuras"], resultado)  # las dos figuras
    return {"umbral": resultado, "rutas": [ruta_json, ruta_pred, *figuras]}  # resultado y archivos


# %% [markdown]
# ## Informe y errores
#
# Imprime las rutas escritas y el tiempo total. Los errores esperados (`experimento.ERRORES_ESPERADOS`: config, datos,
# control, archivo) salen como una línea `scripts/umbral_pd.py: error: <Tipo>: <mensaje>` en stderr y código 2; los de
# programación (`NotImplementedError`, `RecursionError` y cualquier otro) se propagan con su traza (como `run.py pd`).

# %%
def manejador(cfg: dict | None, raiz=None, dep: Dependencias | None = None) -> int:
    """Corre B11 e informa; error esperado → una línea en stderr y código 2."""
    inicio = time.perf_counter()  # tiempo total
    try:  # solo los errores esperados se informan sin traza
        res = correr_umbral(cfg, raiz=raiz, dep=dep)  # corrida
    except ex.ERRORES_DE_PROGRAMACION:  # bugs que heredan de RuntimeError
        raise  # con su traza
    except ex.ERRORES_ESPERADOS as exc:  # config, datos, control o archivo
        print(f"{PREFIJO_ERROR}{type(exc).__name__}: {ex.una_linea(exc)}", file=sys.stderr)  # una línea
        return ex.CODIGO_ERROR  # 2
    for ruta in res["rutas"]:  # archivos escritos
        print(Path(ruta).as_posix())  # una por línea
    control = res["umbral"]["control_semilla_mediana"]  # control
    print(f"Control de la semilla mediana: {control if control is not None else 'no aplica (no reentrenada)'}")  # info
    print(f"Tiempo de ejecución: {time.perf_counter() - inicio:.1f} s")  # tiempo
    return 0  # éxito
