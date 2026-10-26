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
# # Probabilidad → puntaje de la scorecard (Parte C §3.1, tarea C1)
#
# Transforma la PD (o su logit) en un puntaje de scorecard y vuelve del puntaje a la PD, con tres parámetros
# configurables con `--set scorecard.*` (núcleo §4.1): el puntaje base `base`, las odds de referencia `odds_base` y los
# puntos para duplicar las odds `pdo` (enunciado, Parte 1, tarea 6).
#
# **Convención:** $y = 1$ malo (incumplió), $y = 0$ bueno; $p = \text{PD} = P(y = 1)$. **Mayor puntaje = menor riesgo.**
#
# **Quién lo consume:** el exportador de la app (C2) y las scorecards de la logística (C3) y del MLP (C4). El bloque
# de cinco claves que produce `parametros_scorecard` (`base`, `odds_base`, `pdo`, `factor`, `offset`) es el bloque
# `"scorecard"` de `pd_model.json` (núcleo §5.5) y la cabecera de `scorecard.json` (núcleo §5.4).
#
# Referencia de la construcción (puntaje lineal en el logaritmo de las odds con base, odds de referencia y PDO):
# Siddiqi, N. (2017). *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2.ª ed.).
# Wiley. TODO(cita): capítulo y página de la fórmula.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable sin
# efectos al importar.

# %%
"""Transformación PD ↔ puntaje de la scorecard con base, odds de referencia y PDO (Parte C §3.1)."""

import math  # log, isfinite: factor y offset en float de Python
from numbers import Real  # comprobación de que un valor es numérico

import numpy as np  # entradas y salidas vectorizadas en float64
from scipy.special import expit, logit  # sigmoide y logit estables en float64

__all__ = [  # interfaz pública acordada (tarea C1); lo demás es interno
    "CLAVES_SCORECARD", "CLAVES_PARAMETROS",  # claves exactas
    "validar_cfg_scorecard", "bloque_scorecard", "parametros_scorecard",  # config y parámetros
    "score_desde_logit", "score_desde_pd", "logit_desde_score", "pd_desde_score",  # transformaciones
]

CLAVES_SCORECARD = ("base", "odds_base", "pdo")  # claves exactas del bloque scorecard de la config
CLAVES_PARAMETROS = ("base", "odds_base", "pdo", "factor", "offset")  # bloque "scorecard" de pd_model.json (§5.5)


# %% [markdown]
# ## Validación: números reales finitos
#
# Un valor de la config o de los parámetros es válido si es un número real **finito** y no es un bool
# (`bool` es subclase de `int` en Python y `float(True) = 1.0` se colaría como factor u offset).
#
# TODO(refactor-post-B5): `_es_real` duplica el privado de `src/pd/logistica.py` y `src/pd/mlp.py` (no se importa nada
# privado de otro módulo); se unifica en el mismo refactor que esos validadores.

# %%
def _es_real(valor) -> bool:
    """True si ``valor`` es un número real finito (los bool no cuentan).

    Parámetros:
        valor: cualquier objeto.

    Retorna:
        bool: si es un real finito que no es bool.
    """
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, Real):  # bool es subclase de int: se excluye
        return False  # texto, None, bool…: no es un número
    return math.isfinite(valor)  # ±inf y NaN tampoco valen


def _validar_claves(bloque, claves: tuple, nombre: str) -> None:
    """ValueError si ``bloque`` no es un diccionario con exactamente ``claves``.

    Parámetros:
        bloque: el diccionario a revisar.
        claves: claves exactas esperadas.
        nombre: nombre del bloque para el mensaje de error.
    """
    if not isinstance(bloque, dict):  # p. ej. una lista o None pasados por error
        raise ValueError(f"{nombre} debe ser un diccionario; se recibió {type(bloque).__name__}")  # tipo
    faltan = set(claves) - set(bloque)  # sin valor por defecto: deben venir
    sobran = set(bloque) - set(claves)  # una errata (p. ej. "pod") se ignoraría en silencio
    if faltan or sobran:  # el bloque debe ser exacto
        faltan_txt = ", ".join(sorted(map(repr, faltan)))  # repr: ordena claves de tipos mezclados (1 y "x")
        sobran_txt = ", ".join(sorted(map(repr, sobran)))  # ídem para las sobrantes, sin TypeError
        raise ValueError(f"{nombre} es inválido: faltan [{faltan_txt}], sobran [{sobran_txt}]")  # ambas listas


# %% [markdown]
# ## Validación del bloque `scorecard` de la config
#
# | Clave | Regla |
# |---|---|
# | (todas) | exactamente `base`, `odds_base`, `pdo` |
# | `base` | real finito (cualquier signo: es solo un desplazamiento del puntaje) |
# | `odds_base` | real finito $> 0$ (unas odds; se toma su logaritmo) |
# | `pdo` | real finito $> 0$ (con $\text{pdo} \le 0$ el puntaje dejaría de crecer al bajar el riesgo) |
#
# Son reglas de validez de la config, no constantes del dominio. Devuelve una copia con los tres valores como `float`;
# nunca muta la config recibida.

# %%
def validar_cfg_scorecard(cfg_sc: dict) -> dict:
    """Valida el bloque ``scorecard`` (tabla de arriba); cualquier falla → ValueError.

    Parámetros:
        cfg_sc: bloque ``scorecard`` de la config, ``{base, odds_base, pdo}``.

    Retorna:
        dict: copia nueva con ``base``, ``odds_base`` y ``pdo`` como float.
    """
    _validar_claves(cfg_sc, CLAVES_SCORECARD, "El bloque scorecard")  # primero las claves: lo demás las indexa
    for clave in CLAVES_SCORECARD:  # cada valor por separado, para nombrarlo en el error
        if not _es_real(cfg_sc[clave]):  # bool, texto, None, NaN, ±inf
            raise ValueError(f"scorecard.{clave} debe ser un real finito; se recibió {cfg_sc[clave]!r}")  # tipo
    for clave in ("odds_base", "pdo"):  # las dos que deben ser positivas
        if cfg_sc[clave] <= 0:  # ln(odds_base) exige odds_base > 0; pdo > 0 mantiene el sentido del puntaje
            raise ValueError(f"scorecard.{clave} debe ser > 0; se recibió {cfg_sc[clave]!r}")  # rango
    return {clave: float(cfg_sc[clave]) for clave in CLAVES_SCORECARD}  # copia nueva en float, sin mutar


def bloque_scorecard(cfg: dict) -> dict:
    """Extrae y valida el bloque ``scorecard`` de una config completa.

    Parámetros:
        cfg: config completa (p. ej. la de ``configs/scorecard/base.yaml`` con los overrides aplicados).

    Retorna:
        dict: ``validar_cfg_scorecard(cfg["scorecard"])``.
    """
    if not isinstance(cfg, dict) or "scorecard" not in cfg:  # sin bloque no hay parámetros por defecto
        raise ValueError("La config debe ser un diccionario con el bloque 'scorecard'")  # ValueError, no KeyError
    return validar_cfg_scorecard(cfg["scorecard"])  # copia validada


# %% [markdown]
# ## Derivación: factor y offset
#
# Con $p$ = PD, las **odds de buen pagador** (bueno:malo) son
# $$o = \frac{1 - p}{p}, \qquad \ln o = \ln\frac{1 - p}{p} = -\ln\frac{p}{1 - p} = -\operatorname{logit}(p).$$
#
# El puntaje se define **lineal en** $\ln o$:
# $$\text{score}(o) = \text{offset} + \text{factor}\cdot\ln o.$$
#
# Se imponen dos condiciones:
#
# 1. $\text{score}(\text{odds}_{\text{base}}) = \text{base}$;
# 2. $\text{score}(2o) - \text{score}(o) = \text{pdo}$.
#
# De (2): $\text{factor}\,[\ln(2o) - \ln o] = \text{factor}\cdot\ln 2 = \text{pdo}$, luego
# $$\text{factor} = \frac{\text{pdo}}{\ln 2}.$$
# De (1): $\text{offset} + \text{factor}\cdot\ln(\text{odds}_{\text{base}}) = \text{base}$, luego
# $$\text{offset} = \text{base} - \text{factor}\cdot\ln(\text{odds}_{\text{base}}).$$
#
# Por qué se cumplen las dos condiciones **para cualquier** $o$: sustituyendo el offset,
# $\text{score}(o) = \text{base} + \text{factor}\cdot\ln(o / \text{odds}_{\text{base}})$, que vale $\text{base}$ en
# $o = \text{odds}_{\text{base}}$; y la diferencia al duplicar es $\text{factor}\cdot\ln 2 = \text{pdo}$, sin
# depender de $o$.
#
# Con los valores por defecto (600, 50, 20): $\text{factor} = 20/\ln 2 \approx 28{,}8539$ y
# $\text{offset} = 600 - 20\log_2 50 \approx 487{,}1229$.
#
# **Puntos calientes (C §6):**
# - usar odds malo:bueno $p/(1-p)$ cambia el **signo**: el puntaje crecería con el riesgo;
# - usar $\log_{10}$ en vez de $\ln$ cambia el factor: duplicar las odds ya no sumaría pdo;
# - el PDO **no** es "puntos por cada 1 % de PD": el puntaje es lineal en $\ln o$, no en $p$.

# %%
def parametros_scorecard(cfg_sc: dict) -> dict:
    """Valida el bloque ``scorecard`` y añade ``factor`` y ``offset`` (derivación de arriba).

    Parámetros:
        cfg_sc: bloque ``scorecard`` de la config, ``{base, odds_base, pdo}``.

    Retorna:
        dict: exactamente las claves de ``CLAVES_PARAMETROS``, todas float.
    """
    sc = validar_cfg_scorecard(cfg_sc)  # copia validada en float: no muta la entrada
    factor = sc["pdo"] / math.log(2)  # factor = pdo / ln 2 (logaritmo natural)
    offset = sc["base"] - factor * math.log(sc["odds_base"])  # offset = base − factor · ln(odds_base)
    return {**sc, "factor": factor, "offset": offset}  # orden de CLAVES_PARAMETROS


# %% [markdown]
# ## Validación de los parámetros y de las entradas
#
# Las transformaciones reciben el bloque de **cinco** claves (`CLAVES_PARAMETROS`): pasar el bloque de tres claves de
# la config es un error (faltarían `factor` y `offset`). Se exige $\text{factor}$ real finito $> 0$ y $\text{offset}$
# real finito; **no** se comprueba que coincidan con `base`, `odds_base` y `pdo` (se usan tal como vienen).
#
# Las entradas deben ser **numéricas** (enteros o reales): texto (aunque sea `"0.3"`), bool, `None`, objetos y complejos
# son un error, porque `np.asarray(..., dtype=float64)` los convertiría en silencio (`True` → 1.0, `"1.5"` → 1.5). Se
# convierten a `float64`; un escalar devuelve `np.float64` (no un array 0-D) y un array conserva su forma (también el
# vacío, que da un array vacío).
#
# **Resultado no representable:** con entradas finitas pero enormes (p. ej. logit $= 10^{308}$) la cuenta puede
# desbordar a $\pm\infty$ en float64. Eso es un error (`ValueError`), no un puntaje: se calcula con
# `np.errstate(over="ignore", invalid="ignore")` para que no salga el `RuntimeWarning` de numpy y se comprueba que el
# resultado sea finito. `pd_desde_score` es la excepción **si el logit es finito**: la sigmoide de un logit enorme
# **satura** a 0 o 1, que sí es representable (decisión aprobada); si el logit mismo desborda (p. ej. con
# $\text{factor} < 1$ y un puntaje cercano a $\pm 10^{308}$), `ValueError`, como en `logit_desde_score`.

# %%
def _factor_offset(params: dict) -> tuple[float, float]:
    """Valida ``params`` y devuelve ``(factor, offset)``; cualquier falla → ValueError.

    Parámetros:
        params: diccionario con exactamente ``CLAVES_PARAMETROS``.

    Retorna:
        tuple[float, float]: factor (> 0) y offset.
    """
    _validar_claves(params, CLAVES_PARAMETROS, "El bloque de parámetros de la scorecard")  # cinco claves exactas
    factor, offset = params["factor"], params["offset"]  # los dos que usan las fórmulas
    if not _es_real(factor) or factor <= 0:  # factor ≤ 0 invertiría o anularía el puntaje
        raise ValueError(f"factor debe ser un real finito > 0; se recibió {factor!r}")  # tipo o rango
    if not _es_real(offset):  # bool, texto, None, NaN, ±inf
        raise ValueError(f"offset debe ser un real finito; se recibió {offset!r}")  # tipo
    return float(factor), float(offset)  # float de Python para la aritmética


_TIPOS_NUMERICOS = "iuf"  # dtype.kind admitidos: entero con signo, entero sin signo y real


def _a_float64(x, nombre: str) -> np.ndarray:
    """Convierte ``x`` (escalar, lista o array numéricos) a un array float64 de la misma forma.

    Parámetros:
        x: entrada numérica (enteros o reales; no bool, texto, None, objetos ni complejos).
        nombre: nombre de la entrada para el mensaje de error.

    Retorna:
        np.ndarray: array float64 (0-D si ``x`` es escalar).
    """
    try:  # una lista irregular ([[0.1], [0.2, 0.3]]) no forma un array
        a = np.asarray(x)  # sin dtype: conserva el tipo real para poder rechazarlo ([] ya sale float64 vacío)
    except (TypeError, ValueError) as exc:  # unifica el tipo de error del contrato
        raise ValueError(f"{nombre} no forma un array numérico (¿lista irregular?): {x!r}") from exc  # nombra x
    if a.dtype.kind not in _TIPOS_NUMERICOS:  # b bool, U/S texto, O objeto (incluye None), c complejo
        raise ValueError(  # tipo de error del contrato
            f"{nombre} debe ser numérico (entero o real; no bool ni texto); se recibió dtype {a.dtype} ({x!r})"
        )  # mensaje con el dtype detectado
    return a.astype(np.float64, copy=False)  # float32 o enteros → float64, sin copiar si ya lo es


def _exigir_finito(a: np.ndarray, nombre: str) -> None:
    """ValueError si algún elemento de ``a`` es NaN o ±inf."""
    if not np.all(np.isfinite(a)):  # un NaN daría un puntaje NaN en silencio
        raise ValueError(f"{nombre} debe ser finito (sin NaN ni ±inf)")  # dominio


def _exigir_representable(r: np.ndarray, nombre: str) -> None:
    """ValueError si una cuenta con entradas finitas desbordó a ±inf (o dio NaN) en float64."""
    if not np.all(np.isfinite(r)):  # entrada finita, salida infinita: no hay puntaje que devolver
        raise ValueError(f"El {nombre} no es representable en float64 (desborda a ±inf)")  # resultado


def _salida(r: np.ndarray):
    """Escalar (array 0-D) → ``np.float64``; array → el mismo array float64."""
    return np.float64(r) if r.ndim == 0 else r  # np.float64 escalar, no array 0-D


# %% [markdown]
# ## Puntaje desde el logit y desde la PD
#
# Como $\ln o = -\operatorname{logit}(p)$:
# $$\text{score} = \text{offset} - \text{factor}\cdot\operatorname{logit}(p).$$
#
# El **camino principal es el logit**: los modelos lo dan directamente en float64 (`predict_logit`, núcleo §5.3) y así
# no se pierde precisión con $p$ cerca de 1: al representarse en float64, un $p$ cercano a 1 ya perdió precisión
# relativa en su distancia a 1, y $1 - p$ conserva pocas cifras significativas. Desde la PD, el
# logit se calcula con `scipy.special.logit`, que exige $p \in (0, 1)$ abierto: en $p = 0$ o $p = 1$ las odds son
# infinitas o nulas y el puntaje no existe.
#
# El puntaje **no se redondea**: la prueba de paridad de la app exige
# $|\text{score}_{TS} - \text{score}_{Py}| < 10^{-3}$ (núcleo §5.5).

# %%
def score_desde_logit(logit_p, params: dict):
    """Puntaje desde el logit de la PD: score = offset − factor · logit.

    Parámetros:
        logit_p: logit(p) = ln(p/(1 − p)); escalar, lista o array finitos.
        params: diccionario con exactamente ``CLAVES_PARAMETROS``.

    Retorna:
        np.float64 si la entrada es escalar; si no, np.ndarray float64 de la misma forma.
    """
    factor, offset = _factor_offset(params)  # parámetros validados
    z = _a_float64(logit_p, "logit")  # entrada en float64
    _exigir_finito(z, "logit")  # NaN o ±inf no tienen puntaje
    with np.errstate(over="ignore", invalid="ignore"):  # el desbordamiento se reporta abajo como ValueError
        s = offset - factor * z  # score = offset − factor · logit(p)
    _exigir_representable(s, "puntaje")  # −factor · 1e308 sería −inf
    return _salida(s)  # escalar → np.float64


def score_desde_pd(p, params: dict):
    """Puntaje desde la PD: score = offset − factor · logit(p), con p en (0, 1) abierto.

    Parámetros:
        p: PD = P(malo); escalar, lista o array con valores en (0, 1).
        params: diccionario con exactamente ``CLAVES_PARAMETROS``.

    Retorna:
        np.float64 si la entrada es escalar; si no, np.ndarray float64 de la misma forma.
    """
    factor, offset = _factor_offset(params)  # parámetros validados
    q = _a_float64(p, "p")  # entrada en float64
    if not np.all((q > 0) & (q < 1)):  # NaN falla ambas comparaciones; 0 y 1 dan odds 0 o ∞
        raise ValueError("p debe estar en el intervalo abierto (0, 1) (sin NaN)")  # dominio
    with np.errstate(over="ignore", invalid="ignore"):  # el desbordamiento se reporta abajo como ValueError
        s = offset - factor * logit(q)  # logit estable de scipy; misma fórmula que score_desde_logit
    _exigir_representable(s, "puntaje")  # con un factor enorme el producto desborda
    return _salida(s)  # escalar → np.float64


# %% [markdown]
# ## Inversa: del puntaje al logit y a la PD
#
# Despejando $\operatorname{logit}(p)$ de $\text{score} = \text{offset} - \text{factor}\cdot\operatorname{logit}(p)$
# (con $\text{factor} > 0$):
# $$\operatorname{logit}(p) = \frac{\text{offset} - \text{score}}{\text{factor}}, \qquad
# p = \sigma(\operatorname{logit}(p)) = \frac{1}{1 + e^{-\operatorname{logit}(p)}}.$$
#
# La sigmoide se calcula con `scipy.special.expit`, que no desborda: con puntajes extremos la PD **satura** a 0 o 1
# en float64 sin avisos (no es un error; es el límite de la representación).

# %%
def logit_desde_score(score, params: dict):
    """Logit de la PD desde el puntaje: logit = (offset − score) / factor.

    Parámetros:
        score: puntaje; escalar, lista o array finitos.
        params: diccionario con exactamente ``CLAVES_PARAMETROS``.

    Retorna:
        np.float64 si la entrada es escalar; si no, np.ndarray float64 de la misma forma.
    """
    factor, offset = _factor_offset(params)  # parámetros validados (factor > 0: la división existe)
    s = _a_float64(score, "score")  # entrada en float64
    _exigir_finito(s, "score")  # NaN o ±inf no tienen logit
    with np.errstate(over="ignore", invalid="ignore"):  # el desbordamiento se reporta abajo como ValueError
        z = (offset - s) / factor  # inversa de score = offset − factor · logit
    _exigir_representable(z, "logit")  # con un factor diminuto la división desborda
    return _salida(z)  # escalar → np.float64


def pd_desde_score(score, params: dict):
    """PD desde el puntaje: p = σ((offset − score) / factor); puede saturar a 0 o 1.

    Parámetros:
        score: puntaje; escalar, lista o array finitos.
        params: diccionario con exactamente ``CLAVES_PARAMETROS``.

    Retorna:
        np.float64 si la entrada es escalar; si no, np.ndarray float64 de la misma forma.
    """
    z = logit_desde_score(score, params)  # valida params y score y despeja el logit
    return _salida(expit(z))  # sigmoide estable de scipy (satura a 0 o 1 sin avisos); z ya es float64
