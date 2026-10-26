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
# # Scorecard de la logística: puntos por variable (Parte C §3.2, tarea C3)
#
# Convierte la regresión logística de la PD (Parte B, tarea B3) en una scorecard: unos **puntos base** más unos
# **puntos por variable** cuya suma es exactamente el puntaje de C1 (aceptación de C3: "suma de puntos = score").
# También resume, con estadísticos de **train**, una tabla de puntos por valor de cada variable, el rango de puntos de
# cada una y la distribución del puntaje en train, val y test (bandas e histograma).
#
# **Convención:** $y = 1$ malo, $p = \text{PD} = P(y = 1)$; odds de **buen** pagador $o = (1 - p)/p$; mayor puntaje =
# menor riesgo (C §3.1).
#
# **Comando** (núcleo §4.1): `python run.py scorecard --config configs/scorecard/base.yaml --set scorecard.pdo=40`.
# Escribe `scorecard.json` (núcleo §5.4) en la carpeta `salida` de la config.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable sin
# efectos al importar.

# %%
"""Scorecard de la logística: puntos base, puntos por variable y distribución del puntaje (Parte C §3.2, C3)."""

import copy  # copias profundas: la config de entrada no se muta
import math  # isfinite: validación de reales finitos
from decimal import Decimal  # decimales del ancho del histograma
from numbers import Real  # comprobación de que un valor es numérico
from pathlib import Path  # rutas de la config y de la salida

import numpy as np  # cuentas vectorizadas en float64
import pandas as pd  # lectura del parquet de datos

from src.common.config import load_config  # features.yaml con el cargador común
from src.common.preprocesamiento import Preprocesador  # transformación de train/val/test (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON del preprocesador y escritura de scorecard.json
from src.pd.experimento import huella_archivo, huella_particion  # evidencia del núcleo §6
from src.pd.logistica import LogisticaPD  # modelo de B3 (coeficientes, logit y PD)
from src.scorecard.scorecard import (  # transformación PD ↔ puntaje de C1
    CLAVES_PARAMETROS,  # claves exactas del bloque de parámetros (base, odds_base, pdo, factor, offset)
    bloque_scorecard,  # valida el bloque scorecard de la config
    parametros_scorecard,  # añade factor y offset
    score_desde_logit,  # score = offset − factor · logit
)

__all__ = [  # interfaz pública acordada (tarea C3); lo demás es interno
    "CLAVES_RAIZ", "CLAVES_RUTAS", "CLAVES_DISTRIBUCION", "MODELOS", "ARCHIVO_SCORECARD", "SPLITS",  # constantes
    "TOL_ADITIVIDAD",  # tolerancia de la suma de puntos
    "validar_cfg_puntos",  # config
    "puntos_base", "puntos_columnas", "grupos_columnas", "puntos_por_variable", "error_aditividad",  # puntos
    "puntos_valor_crudo", "entrada_numerica", "entrada_categorica",  # tablas por variable
    "cortes_bandas", "tabla_bandas", "histograma", "distribucion_split",  # distribución del puntaje
    "preparar_splits", "construir_scorecard", "manejador",  # orquestación
]

CLAVES_RAIZ = ("scorecard", "modelo", "rutas", "distribucion", "salida")  # claves exactas de la raíz de la config
CLAVES_RUTAS = ("modelo", "preprocesador", "datos", "features")  # claves exactas del bloque rutas
CLAVES_DISTRIBUCION = ("percentiles_tabla", "percentiles_rango", "fraccion_min_categoria", "ancho_bin",
                       "n_bandas")  # claves exactas del bloque distribucion
MODELOS = ("logistica",)  # modelos admitidos (el MLP es C4)
ARCHIVO_SCORECARD = "scorecard.json"  # nombre del resultado (núcleo §5.4)
SPLITS = ("train", "val", "test")  # particiones (núcleo §5.1)
TOL_ADITIVIDAD = 1e-9  # máximo |pb + Σ puntos − score| admitido (puntajes del orden de 10², float64)

_OTRA = "OTRA"  # categoría interna del preprocesador para lo no visto en train (núcleo §5.2)
_SEPARADOR = "="  # columnas one-hot: "nombre=categoria" (Preprocesador.nombres_salida)
_COLUMNA_ID = "id"  # identificador del crédito (núcleo §5.1)
_COLUMNA_SPLIT = "split"  # partición (núcleo §5.1)
_COLUMNA_OBJETIVO = "target"  # objetivo 0/1 (núcleo §5.1)
_N_PERCENTILES = 101  # p0 … p100 del puntaje por split
_N_TOP_RESUMEN = 5  # variables con mayor rango que imprime el manejador
_MAX_INTERVALOS = 10_000  # tope de intervalos del histograma (un ancho mal puesto no agota la memoria)


# %% [markdown]
# ## Validación de la configuración
#
# La config (tras la herencia y los `--set`) tiene exactamente las claves de raíz `CLAVES_RAIZ`:
#
# | Clave | Regla |
# |---|---|
# | `scorecard` | `{base, odds_base, pdo}`, validado por `bloque_scorecard` de C1 |
# | `modelo` | uno de `MODELOS` (`logistica`) |
# | `rutas` | exactamente `CLAVES_RUTAS`, cada una texto no vacío |
# | `distribucion.percentiles_tabla` | lista no vacía de reales en $[0, 100]$, estrictamente creciente |
# | `distribucion.percentiles_rango` | $[a, b]$ con $0 \le a < b \le 100$ |
# | `distribucion.fraccion_min_categoria` | real en $[0, 1)$ |
# | `distribucion.ancho_bin` | real $> 0$ |
# | `distribucion.n_bandas` | entero $\ge 2$ (no bool) |
# | `salida` | texto no vacío (no solo espacios) |

# %%
# TODO(refactor-post-B5): _es_real, _exigir_claves, _factor y _exigir_texto duplican validadores de
# src/scorecard/scorecard.py (C1) y src/export/exportar_app.py (C2). Se copian porque no se importa nada privado
# de otro módulo, como hacen C1 y C2; unificarlos en un módulo común cuando termine B5.
def _es_real(valor) -> bool:
    """True si ``valor`` es un real finito que no es bool.

    Parámetros:
        valor: cualquier objeto.
    Retorna:
        bool: si es un número real finito (los bool no cuentan).
    """
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, Real):  # bool es subclase de int
        return False  # texto, None, bool…: no es un número
    return math.isfinite(valor)  # NaN y ±inf tampoco valen


def _exigir_claves(bloque, claves: tuple, nombre: str) -> None:
    """ValueError si ``bloque`` no es un diccionario con exactamente ``claves``.

    Parámetros:
        bloque: diccionario a revisar.
        claves: claves exactas esperadas.
        nombre: nombre del bloque para el mensaje.
    """
    if not isinstance(bloque, dict):  # lista, None, escalar…
        raise ValueError(f"{nombre} debe ser un diccionario; se recibió {type(bloque).__name__}")  # tipo
    faltan = sorted(map(repr, set(claves) - set(bloque)))  # claves sin valor por defecto
    sobran = sorted(map(repr, set(bloque) - set(claves)))  # erratas que se ignorarían en silencio
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"{nombre} es inválido: faltan [{', '.join(faltan)}], sobran [{', '.join(sobran)}]")  # claves


def _exigir_texto(valor, nombre: str) -> None:
    """ValueError si ``valor`` no es texto con algún carácter distinto de espacio."""
    if not isinstance(valor, str) or not valor.strip():  # None, número, "" o "   "
        raise ValueError(f"{nombre} debe ser un texto no vacío; se recibió {valor!r}")  # tipo o vacío


def _validar_percentiles_tabla(valor) -> None:
    """ValueError si no es una lista no vacía de reales en [0, 100] estrictamente creciente."""
    nombre = "distribucion.percentiles_tabla"  # para los mensajes
    if not isinstance(valor, list) or not valor:  # lista no vacía
        raise ValueError(f"{nombre} debe ser una lista no vacía; se recibió {valor!r}")  # forma
    if not all(_es_real(p) and 0 <= p <= 100 for p in valor):  # cada percentil en [0, 100]
        raise ValueError(f"{nombre} debe tener reales en [0, 100]; se recibió {valor!r}")  # rango
    if any(b <= a for a, b in zip(valor[:-1], valor[1:])):  # orden estricto: sin repetidos
        raise ValueError(f"{nombre} debe ser estrictamente creciente; se recibió {valor!r}")  # orden


def _validar_percentiles_rango(valor) -> None:
    """ValueError si no es [a, b] con 0 ≤ a < b ≤ 100."""
    nombre = "distribucion.percentiles_rango"  # para los mensajes
    if not isinstance(valor, list) or len(valor) != 2 or not all(_es_real(p) for p in valor):  # par de reales
        raise ValueError(f"{nombre} debe ser una lista [a, b] de reales; se recibió {valor!r}")  # forma
    if not 0 <= valor[0] < valor[1] <= 100:  # intervalo no vacío dentro de [0, 100]
        raise ValueError(f"{nombre} debe cumplir 0 ≤ a < b ≤ 100; se recibió {valor!r}")  # rango


def _validar_distribucion(dist) -> None:
    """ValueError si el bloque distribucion no cumple la tabla de arriba."""
    _exigir_claves(dist, CLAVES_DISTRIBUCION, "El bloque distribucion")  # claves exactas primero
    _validar_percentiles_tabla(dist["percentiles_tabla"])  # tabla por variable numérica
    _validar_percentiles_rango(dist["percentiles_rango"])  # rango de puntos de las numéricas
    fraccion = dist["fraccion_min_categoria"]  # umbral del rango de las categóricas
    if not _es_real(fraccion) or not 0 <= fraccion < 1:  # con 1 ninguna categoría contaría
        raise ValueError(f"distribucion.fraccion_min_categoria debe ser real en [0, 1); se recibió {fraccion!r}")
    ancho = dist["ancho_bin"]  # ancho del histograma
    if not _es_real(ancho) or ancho <= 0:  # ancho 0 no define intervalos
        raise ValueError(f"distribucion.ancho_bin debe ser un real > 0; se recibió {ancho!r}")  # rango
    n_bandas = dist["n_bandas"]  # número de bandas de puntaje
    es_entero = isinstance(n_bandas, (int, np.integer)) and not isinstance(n_bandas, (bool, np.bool_))  # sin bool
    if not es_entero or n_bandas < 2:  # una sola banda no separa nada
        raise ValueError(f"distribucion.n_bandas debe ser un entero ≥ 2; se recibió {n_bandas!r}")  # tipo o rango


def validar_cfg_puntos(cfg: dict) -> dict:
    """Valida la config completa de la scorecard (tabla de arriba); cualquier falla → ValueError. No muta ``cfg``.

    Parámetros:
        cfg: config resuelta (herencia y ``--set`` ya aplicados).
    Retorna:
        dict: copia profunda con el bloque ``scorecard`` en float (``bloque_scorecard`` de C1).
    """
    _exigir_claves(cfg, CLAVES_RAIZ, "La config de la scorecard")  # raíz exacta
    sc = bloque_scorecard(cfg)  # base, odds_base y pdo validados por C1 (copia en float)
    if cfg["modelo"] not in MODELOS:  # el MLP llega con C4
        raise ValueError(f"modelo debe ser uno de {MODELOS}; se recibió {cfg['modelo']!r}")  # valor
    _exigir_claves(cfg["rutas"], CLAVES_RUTAS, "El bloque rutas")  # rutas exactas
    for clave in CLAVES_RUTAS:  # cada ruta es un texto
        _exigir_texto(cfg["rutas"][clave], f"rutas.{clave}")  # no vacía
    _validar_distribucion(cfg["distribucion"])  # percentiles, fracción, ancho y bandas
    _exigir_texto(cfg["salida"], "salida")  # carpeta de scorecard.json
    copia = copy.deepcopy(cfg)  # la entrada queda intacta
    copia["scorecard"] = sc  # bloque en float
    return copia  # config validada


# %% [markdown]
# ## Puntos base y puntos por columna
#
# C1 (C §3.1) da $\text{score} = \text{offset} - \text{factor}\cdot\operatorname{logit}(p)$ con
# $\text{factor} = \text{PDO}/\ln 2 > 0$ y $\text{offset} = \text{base} - \text{factor}\cdot\ln(\text{odds\_base})$.
# La logística de B3 es lineal en el logit sobre las columnas $x_j$ de `Preprocesador.transform`:
# $$\operatorname{logit}(p) = \beta_0 + \sum_j \beta_j x_j .$$
# Sustituyendo,
# $$\text{score} = \underbrace{\text{offset} - \text{factor}\,\beta_0}_{\text{puntos base}}
#   + \sum_j \underbrace{\left(-\text{factor}\,\beta_j\,x_j\right)}_{\text{puntos}_j} .$$
#
# - **La suma es exacta** porque el puntaje es una función afín del logit y el logit es lineal en $x$: no hay
#   aproximación (solo el redondeo de float64). **No** sería aditiva en probabilidad: $p = \sigma(\cdot)$ no es lineal.
# - **Signo:** $\text{factor} > 0$, así que $\beta_j > 0$ (más riesgo, más log-odds de malo) da **menos** puntos; las
#   odds de la scorecard son bueno:malo y el puntaje crece con ellas.
# - **PDO:** con `base` y `odds_base` fijos, `--set scorecard.pdo=40` duplica el factor; todos los puntos
#   $-\text{factor}\,\beta_j x_j$ se duplican y $\text{puntos base} - \text{base} =
#   -\text{factor}\,(\ln(\text{odds\_base}) + \beta_0)$ también.
# - **Qué son los puntos base:** con las numéricas estandarizadas con la media de train y las categóricas con
#   $\sum_k \beta_{c_k} \approx 0$ (celda siguiente), los puntos base son el puntaje de un **solicitante de
#   referencia**: cada numérica en su media de train ($x_j = 0$) y cada categórica en el promedio **no ponderado** de
#   los puntos de sus categorías. **No** son el puntaje medio de la población: en train, 525,96 frente a 538,45.

# %%
def _factor(params: dict) -> float:
    """Valida el bloque de parámetros de C1 y devuelve ``factor`` (> 0).

    Parámetros:
        params: diccionario con exactamente ``CLAVES_PARAMETROS``.
    Retorna:
        float: el factor.
    """
    _exigir_claves(params, CLAVES_PARAMETROS, "El bloque de parámetros de la scorecard")  # cinco claves
    factor = params["factor"]  # PDO / ln 2
    if not _es_real(factor) or factor <= 0:  # factor ≤ 0 invertiría el puntaje
        raise ValueError(f"factor debe ser un real finito > 0; se recibió {factor!r}")  # rango
    return float(factor)  # float de Python


def puntos_base(intercepto, params: dict) -> float:
    """Puntos base = offset − factor · β₀.

    Parámetros:
        intercepto: β₀ de la logística.
        params: parámetros de C1 (``parametros_scorecard``).
    Retorna:
        float: puntos base.
    """
    return float(score_desde_logit(float(intercepto), params))  # score del logit β₀ = offset − factor · β₀


def puntos_columnas(X, coef, params: dict) -> np.ndarray:
    """Puntos de cada columna: −factor · βⱼ · xⱼ.

    Parámetros:
        X: matriz (n, d) de ``Preprocesador.transform``.
        coef: β (d,) de la logística.
        params: parámetros de C1.
    Retorna:
        np.ndarray: float64 (n, d).
    """
    factor = _factor(params)  # factor validado
    X64 = np.asarray(X, dtype=np.float64)  # float32 del preprocesador → float64
    b = np.asarray(coef, dtype=np.float64).ravel()  # β como vector
    if X64.ndim != 2 or X64.shape[1] != b.size:  # una columna por coeficiente
        raise ValueError(f"X tiene forma {X64.shape} y hay {b.size} coeficientes: las columnas no coinciden")  # forma
    return -factor * b[None, :] * X64  # −factor · βⱼ · xⱼ por celda


def error_aditividad(pb, P, score) -> float:
    """máx |pb + Σⱼ Pⱼ − score| sobre las filas.

    Parámetros:
        pb: puntos base.
        P: puntos (n, k) por columna o por variable.
        score: puntaje de referencia (n,).
    Retorna:
        float: el error máximo.
    """
    suma = float(pb) + np.asarray(P, dtype=np.float64).sum(axis=1)  # puntaje reconstruido
    return float(np.max(np.abs(suma - np.asarray(score, dtype=np.float64))))  # peor fila


# %% [markdown]
# ## Puntos por variable
#
# Una numérica ocupa una columna. Una categórica con categorías $c_1, \dots, c_K$ (incluida `OTRA`) ocupa $K$
# columnas one-hot y en cada fila **una sola** vale 1, así que
# $$\sum_{k} -\text{factor}\,\beta_{c_k}\,x_{c_k} = -\text{factor}\,\beta_{c^\ast}$$
# con $c^\ast$ la categoría activa: los puntos de la variable son los de su categoría.
#
# Las columnas one-hot de una variable suman 1, colineales con el intercepto: $\beta_0$ y los $\beta_{c_k}$ no están
# identificados por separado (sumar $\delta$ a todos los $\beta_{c_k}$ y restarlo a $\beta_0$ da el mismo logit).
# La descomposición vale para **cualquier** $\beta$ (la suma usa los $\beta$ ajustados y reproduce su logit), pero
# el reparto entre puntos base y categorías no es único como representación. Con la regresión de B3 sí queda fijado:
# la penalización L2 actúa sobre los $\beta_j$ y no sobre el intercepto (lbfgs no penaliza el intercepto;
# scikit-learn User Guide §1.1.11 "Logistic regression", https://scikit-learn.org/stable/modules/linear_model.html,
# consultada el 2026-10-26), así que el desplazamiento $\delta$ que minimiza $\sum_k (\beta_{c_k} + \delta)^2$ deja
# $\sum_k \beta_{c_k} \approx 0$ en cada categórica. `OTRA` cuenta con $\beta = 0$: su columna vale 0 en todo train
# y solo la penalización la mueve. (Medido con el modelo real: $|\sum_k \beta_{c_k}| \approx 7.8\cdot 10^{-5}$ en
# las cuatro categóricas y $\beta_{\text{OTRA}} = 0$.) Aun así, **solo las diferencias** de puntos entre categorías
# de una misma variable se interpretan; el nivel de cada una depende de esa convención.

# %%
def _nombres_esperados(variables: list) -> list[tuple[str, list[str]]]:
    """Por variable, los nombres de sus columnas según la convención de ``Preprocesador.nombres_salida``."""
    salida = []  # [(nombre, [columnas])]
    for v in variables:  # orden de to_dict
        if v["tipo"] == "num":  # una columna con el nombre de la variable
            salida.append((v["nombre"], [v["nombre"]]))  # num
        else:  # una columna por categoría, incluida OTRA
            salida.append((v["nombre"], [f"{v['nombre']}{_SEPARADOR}{c}" for c in v["categorias"]]))  # cat
    return salida  # en el orden del preprocesador


def grupos_columnas(pre_json: dict) -> list[tuple[str, np.ndarray]]:
    """Índices de las columnas de cada variable en ``columnas_salida``.

    Parámetros:
        pre_json: ``Preprocesador.to_dict()``.
    Retorna:
        list[tuple[str, np.ndarray]]: (nombre, índices) en el orden de ``variables``.
    """
    esperados = _nombres_esperados(pre_json["variables"])  # nombres generados por variable
    plano = [c for _, cols in esperados for c in cols]  # todas las columnas en orden
    if plano != list(pre_json["columnas_salida"]):  # deben ser exactamente las del preprocesador
        raise ValueError("Los nombres de columna generados no coinciden con columnas_salida del preprocesador")  # dato
    grupos, inicio = [], 0  # resultado y primera columna de la variable
    for nombre, cols in esperados:  # bloques consecutivos
        grupos.append((nombre, np.arange(inicio, inicio + len(cols))))  # índices de la variable
        inicio += len(cols)  # siguiente bloque
    return grupos  # [(nombre, índices)]


def puntos_por_variable(P, grupos) -> np.ndarray:
    """Suma de los puntos de las columnas de cada variable.

    Parámetros:
        P: puntos por columna (n, d).
        grupos: salida de ``grupos_columnas``.
    Retorna:
        np.ndarray: float64 (n, número de variables).
    """
    P64 = np.asarray(P, dtype=np.float64)  # float64
    return np.column_stack([P64[:, idx].sum(axis=1) for _, idx in grupos])  # num: su columna; cat: suma one-hot


# %% [markdown]
# ## Puntos de una numérica en función de su valor crudo
#
# Con $t(v) = \text{log1p}^{\,[\text{si aplica}]}\big(\operatorname{recorte}(\operatorname{imputación}(v))\big)$ (el
# orden del preprocesador: imputar con la mediana → recortar a $[l, h]$ → `log1p` → estandarizar),
# $$\text{puntos}(v) = -\text{factor}\,\beta\;\frac{t(v) - \mu}{\sigma}.$$
# Es lineal en el valor **estandarizado**, pero no en el crudo: es plana fuera de $[l, h]$ y logarítmica si hay
# `log1p`. En la media transformada ($t(v) = \mu$) vale 0. `puntos_por_unidad_estandarizada` $= -\text{factor}\,\beta$.
#
# La tabla de cada numérica evalúa esta función en los percentiles de `percentiles_tabla` de los valores crudos **no
# vacíos de train**; el **rango de puntos** es $|\text{puntos}(v_{p_b}) - \text{puntos}(v_{p_a})|$ con
# `percentiles_rango` $= [p_a, p_b]$. Es una medida simple del peso de la variable en la población de train: no es
# SHAP (C4) y depende de la escala y la dispersión de la variable. Por defecto $[p_a, p_b] = [1, 99]$ (decisión de
# Juan José, 26-oct): el preprocesador recorta cada numérica en $[p_1, p_{99}]$ de train, así que fuera de ese tramo los
# puntos son planos y p1–p99 es **todo** el efecto que el modelo puede producir con esa variable. Con p5–p95, una
# numérica discreta y concentrada daba rango 0 aunque $\beta \ne 0$: `pub_rec` tiene p5 = p95 = 0 en train y, sin
# embargo, resta 9,9 puntos con 1 o más registros. Advertencias (`percentiles_rango` sigue siendo configurable con
# `--set distribucion.percentiles_rango=[a,b]`):
# - el rango de una numérica (entre dos percentiles) y el de una categórica (entre categorías con fracción mínima)
#   **no son comparables** entre sí: miden cosas distintas.

# %%
def puntos_valor_crudo(valores, v_pre: dict, puntos_unidad) -> np.ndarray:
    """Puntos de una numérica para valores crudos: imputar → recortar → log1p → estandarizar → × puntos_unidad.

    Parámetros:
        valores: valores crudos (NaN = vacío).
        v_pre: la variable en ``pre_json["variables"]``.
        puntos_unidad: −factor · β.
    Retorna:
        np.ndarray: float64 con la forma de ``valores``.
    """
    x = np.asarray(valores, dtype=np.float64)  # copia en float64
    x = np.where(np.isnan(x), float(v_pre["mediana"]), x)  # vacío → mediana de train
    x = np.clip(x, *v_pre["recorte"])  # recorte a los percentiles de train
    if v_pre["log1p"]:  # transformación declarada en features.yaml
        x = np.log1p(x)  # ln(1 + x)
    return float(puntos_unidad) * (x - float(v_pre["media"])) / float(v_pre["desv"])  # estandarizar y escalar


def entrada_numerica(f: dict, v_pre: dict, beta, crudos_train, params: dict, cfg_dist: dict) -> dict:
    """Entrada de ``puntos_por_variable`` de una numérica: tabla por percentiles de train y rango.

    Parámetros:
        f: la variable en ``features.yaml``.
        v_pre: la variable en ``pre_json["variables"]``.
        beta: β de su columna.
        crudos_train: valores crudos de train (los NaN se descartan).
        params: parámetros de C1.
        cfg_dist: bloque ``distribucion`` de la config.
    Retorna:
        dict: {nombre, tipo, etiqueta_es, beta, puntos_por_unidad_estandarizada, transformacion, tabla, rango_puntos}.
    """
    unidad = -_factor(params) * float(beta)  # puntos por unidad estandarizada
    crudos = np.asarray(crudos_train, dtype=np.float64)  # float64
    crudos = crudos[~np.isnan(crudos)]  # solo los no vacíos de train
    if crudos.size == 0:  # sin datos no hay percentiles
        raise ValueError(f"{f['nombre']}: no hay valores no vacíos en train")  # dato
    pct = list(cfg_dist["percentiles_tabla"])  # percentiles de la tabla
    valores = np.percentile(crudos, pct)  # interpolación lineal (por defecto de numpy)
    puntos = puntos_valor_crudo(valores, v_pre, unidad)  # puntos en cada percentil
    extremos = puntos_valor_crudo(np.percentile(crudos, cfg_dist["percentiles_rango"]), v_pre, unidad)  # p_a y p_b
    return {  # claves exactas de una numérica
        "nombre": f["nombre"], "tipo": "num", "etiqueta_es": f["etiqueta_es"],  # identidad
        "beta": float(beta), "puntos_por_unidad_estandarizada": float(unidad) + 0.0,  # + 0.0: −0.0 → 0.0
        "transformacion": {"mediana": float(v_pre["mediana"]), "recorte": [float(r) for r in v_pre["recorte"]],
                           "log1p": bool(v_pre["log1p"]), "media": float(v_pre["media"]),
                           "desv": float(v_pre["desv"])},  # receta del preprocesador
        "tabla": [{"percentil": p, "valor": float(v), "puntos": float(q) + 0.0}  # + 0.0: −0.0 → 0.0
                  for p, v, q in zip(pct, valores, puntos, strict=True)],  # una fila por percentil
        "rango_puntos": float(abs(extremos[1] - extremos[0])) + 0.0,  # |puntos(p_b) − puntos(p_a)|
    }


# %% [markdown]
# ## Puntos de una categórica
#
# Cada categoría vale $-\text{factor}\,\beta_c$ (su columna one-hot vale 1). Se listan las categorías de train sin
# `OTRA`, en el orden de `categorias_es` (el de la app); `OTRA` va aparte, marcada como interna (la app no la ofrece).
# `categorias_es` debe tener exactamente las categorías del preprocesador sin `OTRA` (como en C2; si no, error).
#
# `fraccion_train` es la media de la columna one-hot en `pre.transform(train)` (decisión de Juan José: misma limpieza
# del preprocesador). El **rango de puntos** es $\max - \min$ de los puntos entre las categorías con
# `fraccion_train` $\ge$ `fraccion_min_categoria`: una categoría casi vacía no debe definir el peso de la variable.

# %%
def entrada_categorica(f: dict, v_pre: dict, betas: dict, fracciones_train: dict, params: dict,
                       cfg_dist: dict) -> dict:
    """Entrada de ``puntos_por_variable`` de una categórica.

    Parámetros:
        f: la variable en ``features.yaml`` (con ``categorias_es``).
        v_pre: la variable en ``pre_json["variables"]``.
        betas: categoría → β (incluida OTRA).
        fracciones_train: categoría → fracción en train.
        params: parámetros de C1.
        cfg_dist: bloque ``distribucion`` de la config.
    Retorna:
        dict: {nombre, tipo, etiqueta_es, categorias, otra, rango_puntos}.
    """
    factor = _factor(params)  # factor validado
    etiquetas = f.get("categorias_es") or {}  # código → etiqueta en español
    de_train = [c for c in v_pre["categorias"] if c != _OTRA]  # categorías vistas en train
    if set(etiquetas) != set(de_train):  # la app ofrecería una categoría inexistente o perdería una
        raise ValueError(f"{f['nombre']}: categorias_es {sorted(etiquetas)} no coincide con las categorías del "
                         f"preprocesador sin OTRA {sorted(de_train)}")  # dato
    categorias = [{  # una fila por categoría, en el orden de categorias_es
        "categoria": c, "etiqueta": etiquetas[c], "beta": float(betas[c]),  # identidad y coeficiente
        "puntos": -factor * float(betas[c]) + 0.0, "fraccion_train": float(fracciones_train[c]),  # −0.0 → 0.0
    } for c in etiquetas]  # orden de categorias_es
    frecuentes = [c["puntos"] for c in categorias if c["fraccion_train"] >= cfg_dist["fraccion_min_categoria"]]
    if not frecuentes:  # ninguna categoría llega al umbral: el rango no está definido
        raise ValueError(f"{f['nombre']}: ninguna categoría tiene fracción de train ≥ "
                         f"{cfg_dist['fraccion_min_categoria']}")  # dato
    return {  # claves exactas de una categórica
        "nombre": f["nombre"], "tipo": "cat", "etiqueta_es": f["etiqueta_es"],  # identidad
        "categorias": categorias,  # categorías de train sin OTRA
        "otra": {"categoria": _OTRA, "interna": True, "beta": float(betas[_OTRA]),
                 "puntos": -factor * float(betas[_OTRA]) + 0.0},  # categoría interna; −0.0 → 0.0
        "rango_puntos": float(max(frecuentes) - min(frecuentes)) + 0.0,  # máx − mín entre las frecuentes
    }


# %% [markdown]
# ## Bandas e histograma del puntaje
#
# Los cortes de las bandas son los cuantiles internos $k/n$ ($k = 1, \dots, n - 1$) del puntaje de **train**
# (`np.quantile` lineal) y quedan **fijos**: val y test solo se evalúan con ellos (fuga, núcleo §5.7). La banda $k$ es
# $[\text{corte}_{k-1}, \text{corte}_k)$, la primera desde $-\infty$ y la última hasta $+\infty$; la banda 0 es la de
# menor puntaje (más riesgo). Un puntaje exactamente en un corte va a la banda superior. Por banda se resumen el
# puntaje medio, la PD media (`prob`, salida de `predict_proba`) y la tasa de malos observada.
#
# El histograma usa intervalos $[b_i, b_{i+1})$ de ancho fijo anclados en 0, con índice entero
# $k = \lfloor s/\text{ancho} \rfloor$ y bordes $b_i = (k_0 + i)\cdot\text{ancho}$, $k_0 = \min k$. En coma flotante
# $0.3/0.1 = 2.9999999999999996$: por eso los bordes se redondean a los decimales del ancho y el índice se corrige
# para que $b_k \le s < b_{k+1}$ se cumpla con los bordes que se escriben. Un tope (`_MAX_INTERVALOS`) evita pedir
# millones de intervalos por un ancho mal puesto.

# %%
def cortes_bandas(score_train, n_bandas: int) -> list[float]:
    """Cuantiles internos k/n_bandas (k = 1…n−1) del puntaje de train; ValueError si no son estrictamente crecientes.

    Parámetros:
        score_train: puntaje de train.
        n_bandas: número de bandas (≥ 2).
    Retorna:
        list[float]: n_bandas − 1 cortes.
    """
    q = np.quantile(np.asarray(score_train, dtype=np.float64), np.arange(1, n_bandas) / n_bandas)  # lineal
    if np.any(np.diff(q) <= 0):  # empates colapsan bandas
        raise ValueError(f"Los cortes de las bandas no son estrictamente crecientes: {q.tolist()}")  # dato
    return [float(c) for c in q]  # float de Python


def _media_o_none(x: np.ndarray):
    """Media en float, o None si ``x`` está vacío."""
    return float(np.mean(x)) if x.size else None  # banda vacía → None


def tabla_bandas(score, prob, y, cortes) -> list[dict]:
    """Por banda: límites, n, puntaje medio, PD media y tasa de malos.

    Parámetros:
        score: puntaje (n,).
        prob: PD = P(malo) (n,).
        y: objetivo 0/1 (n,).
        cortes: cortes de ``cortes_bandas``.
    Retorna:
        list[dict]: una fila por banda (len(cortes) + 1).
    """
    s = np.asarray(score, dtype=np.float64)  # puntaje
    p = np.asarray(prob, dtype=np.float64)  # PD
    yy = np.asarray(y, dtype=np.float64)  # objetivo
    banda = np.searchsorted(np.asarray(cortes, dtype=np.float64), s, side="right")  # en un corte → banda superior
    filas = []  # resultado
    for k in range(len(cortes) + 1):  # banda 0 … n − 1
        m = banda == k  # filas de la banda
        filas.append({
            "banda": k,  # 0 = menor puntaje
            "desde": float(cortes[k - 1]) if k > 0 else None,  # −∞ en la primera
            "hasta": float(cortes[k]) if k < len(cortes) else None,  # +∞ en la última
            "n": int(m.sum()),  # créditos
            "puntaje_medio": _media_o_none(s[m]), "pd_media": _media_o_none(p[m]),  # promedios
            "tasa_malos": _media_o_none(yy[m]),  # fracción de malos observada
        })
    return filas  # una fila por banda


def _decimales(ancho) -> int:
    """Decimales con que se escribe ``ancho`` (10 → 1 por "10.0", 0.1 → 1, 1e-05 → 5): los de los bordes."""
    return max(0, -Decimal(repr(float(ancho))).as_tuple().exponent)  # exponente de la representación más corta


def histograma(score, ancho) -> dict:
    """Histograma de intervalos [b_i, b_{i+1}) de ancho fijo anclados en 0 (índices enteros).

    Parámetros:
        score: puntaje (n ≥ 1).
        ancho: ancho de los intervalos (> 0).
    Retorna:
        dict: {ancho, bordes, conteos}.
    """
    s = np.asarray(score, dtype=np.float64)  # puntaje
    dec = _decimales(ancho)  # redondeo de los bordes
    k = np.floor(s / ancho).astype(np.int64)  # índice entero del intervalo (puede quedar 1 abajo por redondeo)
    k = k + (s >= np.round((k + 1) * ancho, dec))  # s ya alcanzó el borde siguiente: sube un intervalo
    k = k - (s < np.round(k * ancho, dec))  # s quedó bajo su borde: baja un intervalo
    k0 = int(k.min())  # primer intervalo
    n = int(k.max()) - k0 + 1  # número de intervalos
    if n > _MAX_INTERVALOS:  # ancho mal puesto o puntajes extremos
        raise ValueError(f"El histograma tendría {n} intervalos (> {_MAX_INTERVALOS}); revise distribucion.ancho_bin")
    bordes = np.round((k0 + np.arange(n + 1)) * ancho, dec)  # bordes múltiplos de ancho
    conteos = np.bincount(k - k0, minlength=n)  # créditos por intervalo
    return {"ancho": ancho, "bordes": [float(b) for b in bordes], "conteos": [int(c) for c in conteos]}  # JSON


def distribucion_split(score, prob, y, cortes, ancho) -> dict:
    """Resumen del puntaje de un split: n, media, desv muestral, 101 percentiles, histograma y bandas.

    Parámetros:
        score: puntaje (n,).
        prob: PD = P(malo) (n,).
        y: objetivo (n,).
        cortes: cortes de train.
        ancho: ancho del histograma.
    Retorna:
        dict: {n, media, desv, percentiles, histograma, bandas}.
    """
    s = np.asarray(score, dtype=np.float64)  # puntaje
    n = int(s.size)  # créditos
    return {
        "n": n,  # créditos del split
        "media": float(np.mean(s)),  # media
        "desv": float(np.std(s, ddof=1)) if n > 1 else 0.0,  # muestral; n = 1 → 0 (núcleo §5.4)
        "percentiles": [float(v) for v in np.percentile(s, np.arange(_N_PERCENTILES))],  # p0 … p100, lineal
        "histograma": histograma(s, ancho),  # conteos por intervalo
        "bandas": tabla_bandas(s, prob, y, cortes),  # con los cortes de train
    }


# %% [markdown]
# ## Construcción de `scorecard.json`
#
# 1. `preparar_splits(cfg)` carga `features.yaml`, el preprocesador (ajustado solo con train por A), los datos y la
#    logística de B3, y por split calcula $X$ = `pre.transform`, el **logit** (`predict_logit`), la PD
#    (`predict_proba`) y $y$; guarda también los crudos de train de las numéricas y la `evidencia` (huellas de la
#    partición, a partir de `id` y `split`, y del preprocesador; núcleo §6). No depende de `scorecard.*` ni de
#    `distribucion.*`: el mismo preparado sirve para cualquier `--set scorecard.*` (guarda el logit, no el puntaje).
# 2. `construir_scorecard(cfg, preparado)` pasa el logit a puntaje con C1, calcula los puntos por columna y **por
#    variable** y comprueba en los tres splits que su suma más los puntos base es el puntaje (error si supera
#    `TOL_ADITIVIDAD`): `run.py scorecard` verifica la aceptación de C3 en cada corrida.
# 3. Tablas de las numéricas, fracciones de las categóricas y cortes de las bandas: **solo con train**.

# %%
def preparar_splits(cfg: dict) -> dict:
    """Carga y transforma los datos de la scorecard (lo que no depende de scorecard.* ni de distribucion.*).

    Parámetros:
        cfg: config con el bloque ``rutas`` (se valida la config completa).
    Retorna:
        dict: {rutas, features, pre_json, coef, intercepto, crudos_train {nombre: float64}, evidencia,
        splits {split: {X, logit, prob, y}}}; los arrays son de solo lectura.
    """
    rutas = validar_cfg_puntos(cfg)["rutas"]  # config validada (no se muta)
    features = load_config(rutas["features"])["features"]  # variables (orden de la scorecard)
    pre_json = load_json(rutas["preprocesador"])  # receta del preprocesador
    pre = Preprocesador.from_dict(pre_json)  # preprocesador ajustado con train (A)
    datos = pd.read_parquet(rutas["datos"])  # todas las filas con su split
    modelo = LogisticaPD.cargar(rutas["modelo"])  # logística de B3
    coef, b0 = modelo.coeficientes()  # β (d,) y β₀
    splits = {}  # split → matrices y predicciones
    for s in SPLITS:  # train, val, test
        df = datos[datos[_COLUMNA_SPLIT] == s].reset_index(drop=True)  # filas del split
        if df.empty:  # sin filas no hay distribución
            raise ValueError(f"El split {s!r} no tiene filas")  # dato
        X = pre.transform(df)  # float32, columnas en orden fijo
        splits[s] = {"X": X, "logit": modelo.predict_logit(X), "prob": modelo.predict_proba(X),
                     "y": df[_COLUMNA_OBJETIVO].to_numpy()}  # logit, PD y objetivo
    train = datos[datos[_COLUMNA_SPLIT] == SPLITS[0]]  # crudos de train
    crudos = {f["nombre"]: pd.to_numeric(train[f["nombre"]], errors="raise").to_numpy(dtype=np.float64)
              for f in features if f["tipo"] == "num"}  # valores antes de transformar
    evidencia = {"sha256_particion": huella_particion(datos[_COLUMNA_ID].tolist(),  # listas: iterar la serie
                                                      datos[_COLUMNA_SPLIT].tolist()),  # Arrow tarda ~3 s
                 "sha256_preprocesador": huella_archivo(rutas["preprocesador"])}  # núcleo §6
    for arr in [coef, *crudos.values(), *(a for d in splits.values() for a in d.values())]:  # arrays compartidos
        arr.setflags(write=False)  # solo lectura: varios consumidores comparten el mismo preparado
    return {"rutas": dict(rutas), "features": features, "pre_json": pre_json, "coef": coef, "intercepto": b0,
            "crudos_train": crudos, "evidencia": evidencia, "splits": splits}  # preparado


def _puntajes(preparado: dict, params: dict, grupos: list, pb: float) -> dict:
    """Por split: puntaje de C1 y error de aditividad máximo (por columna y por variable); ValueError si se rompe."""
    salida = {}  # split → puntaje y error
    for s, d in preparado["splits"].items():  # train, val, test
        score = np.asarray(score_desde_logit(d["logit"], params), dtype=np.float64)  # referencia C1
        P = puntos_columnas(d["X"], preparado["coef"], params)  # puntos por columna
        error = max(error_aditividad(pb, P, score),  # pb + Σ columnas vs score
                    error_aditividad(pb, puntos_por_variable(P, grupos), score))  # pb + Σ variables vs score
        if not error <= TOL_ADITIVIDAD:  # también atrapa NaN
            raise ValueError(f"Aditividad rota en {s}: error {error:.3e} > {TOL_ADITIVIDAD:.0e}")  # numérico
        salida[s] = {"score": score, "error": error}  # resultados del split
    return salida  # split → puntaje y error


def _entradas(preparado: dict, grupos: list, params: dict, cfg_dist: dict) -> list[dict]:
    """Entradas de ``puntos_por_variable`` en el orden de features.yaml (estadísticos solo de train)."""
    features, pre_json, coef = preparado["features"], preparado["pre_json"], preparado["coef"]  # entradas
    if [f["nombre"] for f in features] != [nombre for nombre, _ in grupos]:  # mismas variables y orden
        raise ValueError("features.yaml y el preprocesador no tienen las mismas variables en el mismo orden")  # dato
    medias = np.mean(preparado["splits"][SPLITS[0]]["X"], axis=0, dtype=np.float64)  # fracción one-hot en train
    entradas = []  # resultado
    for f, v, (_, idx) in zip(features, pre_json["variables"], grupos, strict=True):  # orden de features.yaml
        if v["tipo"] == "num":  # una columna
            entradas.append(entrada_numerica(f, v, coef[idx[0]], preparado["crudos_train"][f["nombre"]], params,
                                             cfg_dist))  # tabla y rango
        else:  # una columna por categoría, en el orden de v["categorias"] (grupos_columnas)
            betas = {c: float(coef[j]) for c, j in zip(v["categorias"], idx, strict=True)}  # categoría → β
            fracciones = {c: float(medias[j]) for c, j in zip(v["categorias"], idx, strict=True)}  # → fracción
            entradas.append(entrada_categorica(f, v, betas, fracciones, params, cfg_dist))  # puntos y rango
    return entradas  # orden de features.yaml


def construir_scorecard(cfg: dict, preparado: dict | None = None) -> dict:
    """Construye el contenido de ``scorecard.json`` (núcleo §5.4) para la logística.

    Parámetros:
        cfg: config resuelta (``configs/scorecard/base.yaml`` con ``--set``).
        preparado: salida de ``preparar_splits`` con las mismas rutas; None → se prepara aquí.
    Retorna:
        dict: {base, odds_base, pdo, factor, offset, modelo, puntos_base, aditividad, puntos_por_variable,
        distribucion_scores, evidencia}.
    """
    cfg = validar_cfg_puntos(cfg)  # copia validada
    if preparado is None:  # sin preparación compartida
        preparado = preparar_splits(cfg)  # carga y transforma
    if preparado["rutas"] != cfg["rutas"]:  # un preparado de otros archivos daría una scorecard incoherente
        raise ValueError(f"El preparado se hizo con las rutas {preparado['rutas']} y la config pide {cfg['rutas']}")
    params = parametros_scorecard(cfg["scorecard"])  # base, odds_base, pdo, factor, offset
    dist = cfg["distribucion"]  # bloque de la distribución
    grupos = grupos_columnas(preparado["pre_json"])  # columnas de cada variable (valida los nombres)
    pb = puntos_base(preparado["intercepto"], params)  # offset − factor · β₀
    puntajes = _puntajes(preparado, params, grupos, pb)  # puntajes y aditividad por columna y por variable
    cortes = cortes_bandas(puntajes["train"]["score"], dist["n_bandas"])  # SOLO train
    return {
        **params,  # base, odds_base, pdo, factor, offset (C1)
        "modelo": cfg["modelo"],  # logistica
        "puntos_base": pb,  # offset − factor · β₀
        "aditividad": {"error_max": max(d["error"] for d in puntajes.values()), "tolerancia": TOL_ADITIVIDAD},
        "puntos_por_variable": _entradas(preparado, grupos, params, dist),  # estadísticos de train
        "distribucion_scores": {
            "cortes_bandas": cortes, "n_bandas": dist["n_bandas"], "ancho_bin": dist["ancho_bin"],  # comunes
            **{s: distribucion_split(puntajes[s]["score"], d["prob"], d["y"], cortes, dist["ancho_bin"])
               for s, d in preparado["splits"].items()},  # train, val, test
        },
        "evidencia": dict(preparado["evidencia"]),  # núcleo §6
    }


# %% [markdown]
# ## Manejador de `run.py scorecard`
#
# Construye la scorecard, la escribe en `salida/scorecard.json` e imprime un resumen: puntos base, error de
# aditividad, las 5 variables con mayor rango de puntos (nombre técnico y etiqueta) y, por banda de test, su rango de
# puntaje $[\text{desde}, \text{hasta})$, n, PD media y tasa de malos. El tiempo total lo imprime `run.py`
# (`src/common/cli.py`).

# %%
def _texto_media(valor) -> str:
    """Promedio de una banda para el resumen: None (banda vacía) → raya."""
    return "—" if valor is None else f"{valor:.4f}"  # banda vacía


def _imprimir_resumen(sc: dict, ruta: Path) -> None:
    """Resumen legible de la scorecard en la salida estándar."""
    print(f"Escrito: {ruta.as_posix()}")  # dónde quedó
    print(f"Puntos base: {sc['puntos_base']:.4f} (factor {sc['factor']:.4f}, offset {sc['offset']:.4f})")  # base
    ad = sc["aditividad"]  # error de la suma
    print(f"Error máximo de aditividad: {ad['error_max']:.3e} (tolerancia {ad['tolerancia']:.0e})")  # aceptación
    top = sorted(sc["puntos_por_variable"], key=lambda e: e["rango_puntos"], reverse=True)[:_N_TOP_RESUMEN]  # top
    print(f"Variables con mayor rango de puntos (top {_N_TOP_RESUMEN}):")  # cabecera
    for e in top:  # una línea por variable
        print(f"  {e['nombre']:<26} {e['etiqueta_es']:<40} {e['rango_puntos']:8.2f}")  # nombre, etiqueta y rango
    print("Bandas de test (banda 0 = menor puntaje = más riesgo):")  # cabecera
    for b in sc["distribucion_scores"]["test"]["bandas"]:  # una línea por banda
        desde = "-∞" if b["desde"] is None else f"{b['desde']:.1f}"  # primera banda sin límite inferior
        hasta = "+∞" if b["hasta"] is None else f"{b['hasta']:.1f}"  # última banda sin límite superior
        print(f"  banda {b['banda']}: [{desde}, {hasta})  n = {b['n']:>6}  pd_media = {_texto_media(b['pd_media'])}"
              f"  tasa_malos = {_texto_media(b['tasa_malos'])}")  # rango, n, PD media y tasa


def manejador(cfg: dict) -> int:
    """``run.py scorecard``: escribe ``salida/scorecard.json`` e imprime el resumen; devuelve 0.

    Parámetros:
        cfg: config resuelta.
    Retorna:
        int: 0 si todo salió bien.
    """
    sc = construir_scorecard(cfg)  # valida la config y construye
    ruta = save_json(Path(cfg["salida"]) / ARCHIVO_SCORECARD, sc)  # UTF-8, sangría 2, sin NaN, atómico
    _imprimir_resumen(sc, ruta)  # resumen
    return 0  # éxito
