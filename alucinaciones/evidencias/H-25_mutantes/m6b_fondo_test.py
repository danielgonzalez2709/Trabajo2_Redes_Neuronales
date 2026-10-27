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
# # SHAP del MLP, importancia por permutación y scorecard del MLP (Parte C §3.2 y §3.3, tarea C4)
#
# Convierte el perceptrón multicapa de la PD (Parte B, B5: `models/pd_mlp.pt`) en una scorecard: unos **puntos base**
# más unos **puntos por variable** cuya suma es exactamente el puntaje de C1 (aceptación de C4: "aditividad comprobada
# numéricamente"). Los puntos salen de los **valores de Shapley del logit** (SHAP; Lundberg y Lee, 2017
# [lundberg2017]). Como contraste se mide la **importancia por permutación** en test de los dos modelos y la
# **estabilidad** de la importancia SHAP frente a otro fondo y otras permutaciones.
#
# **Convención:** $y = 1$ malo, $p = \text{PD} = P(y = 1)$; odds de **buen** pagador $o = (1 - p)/p$; mayor puntaje =
# menor riesgo (C §3.1).
#
# **Comando** (núcleo §4.1): `python run.py scorecard --config configs/scorecard/mlp.yaml` (`--set scorecard.pdo=40`,
# `--set shap.n_explicar=200`, …). Escribe `scorecard_mlp.json` y `shap_mlp.json` en la carpeta `salida` de la config.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable sin
# efectos al importar.

# %%
"""SHAP del MLP sobre el logit, importancia por permutación y scorecard del MLP (Parte C §3.2 y §3.3, C4)."""

import time  # tiempo de cada etapa (tiempos_s)
from pathlib import Path  # carpeta de la salida

import numpy as np  # cuentas vectorizadas en float64
import pandas as pd  # lectura del parquet de datos
import shap  # explainer de permutaciones y masker Independent (shap 0.54.1, requirements.txt)
from scipy.special import expit  # σ estable: PD = σ(logit)
from scipy.stats import spearmanr  # correlación de rangos (dirección y estabilidad)

from src.common.config import load_config  # features.yaml con el cargador común
from src.common.metricas import auc  # AUC de la Parte B (src/common/metricas.py)
from src.common.preprocesamiento import Preprocesador  # transformación de train/val/test (núcleo §5.2)
from src.common.results import load_json, save_json  # JSON del preprocesador y escritura de los resultados
from src.common.seeds import make_rng  # semillas base_seed + run_id (núcleo §4.1)
from src.export.exportar_app import SALIDA_RED, capas_mlp, forward_desde_json  # forward float64 de la app (C2)
from src.pd.experimento import huella_archivo, huella_particion  # evidencia del núcleo §6
from src.pd.logistica import LogisticaPD, logloss  # logística de B3 y la log-loss de la Parte B
from src.scorecard.puntos import (  # piezas de la scorecard de C3
    CLAVE_RUTA_LOGISTICA,  # rutas.logistica (solo con modelo mlp)
    ConfigInvalida,  # error de config que el manejador convierte en código 2
    MODELO_MLP,  # "mlp"
    SPLITS,  # train, val, test
    cortes_bandas,  # deciles del puntaje de train
    distribucion_split,  # n, media, desv, percentiles, histograma y bandas
    error_aditividad,  # máx |pb + Σ puntos − score|
    grupos_columnas,  # columnas de cada variable en la salida del preprocesador
    validar_cfg_puntos,  # config de la scorecard (raíz + bloque shap con mlp)
    TOL_ADITIVIDAD,  # tolerancia de la suma de puntos (1e-9)
)
from src.scorecard.scorecard import parametros_scorecard, score_desde_logit  # C1: factor, offset y puntaje

__all__ = [  # interfaz pública acordada (tarea C4); lo demás es interno
    "CLAVES_SHAP", "ARCHIVO_SCORECARD_MLP", "ARCHIVO_SHAP", "TOL_EFICIENCIA", "TOL_FORMA_CERRADA",  # constantes
    "TOL_PHI0", "MAX_CODIGOS", "N_VARIABLES", "MIN_MAX_EVALS",  # φ₀, codificación, jugadores y mínimo de max_evals
    "validar_cfg_shap",  # config
    "a_variables", "a_columnas", "funcion_logit",  # jugadores
    "valores_shap", "error_eficiencia", "shap_cerrado_lineal",  # SHAP
    "importancia_global", "direccion", "estabilidad", "importancia_permutacion",  # resúmenes
    "preparar_mlp", "construir_scorecard_mlp", "manejador_mlp",  # orquestación
]

CLAVES_SHAP = ("base_seed", "n_fondo", "n_explicar", "n_estabilidad", "max_evals", "k_permutacion")  # bloque shap
ARCHIVO_SCORECARD_MLP = "scorecard_mlp.json"  # scorecard del MLP (mismas claves de raíz que scorecard.json)
ARCHIVO_SHAP = "shap_mlp.json"  # SHAP, permutación y estabilidad
TOL_EFICIENCIA = 1e-9  # máx |φ₀ + Σφ − logit| admitido (logits del orden de 1, float64)
TOL_FORMA_CERRADA = 1e-9  # máx |φ por permutación − forma cerrada| con la logística
TOL_PHI0 = 1e-12  # máx diferencia entre los φ₀ por fila de shap (otra BLAS puede cambiar el último bit)
RTOL_ISCLOSE, ATOL_ISCLOSE = 1e-5, 1e-8  # valores por defecto de np.isclose, que usa Tabular.invariants de shap
MAX_CODIGOS = int((1 - ATOL_ISCLOSE) / RTOL_ISCLOSE)  # 99 999: con códigos < 99 999, k y k + 1 nunca son isclose
N_VARIABLES = 15  # jugadores: las 15 variables de configs/features.yaml (decisión de Daniel, 23-oct; A §3.4)
MIN_MAX_EVALS = 2 * N_VARIABLES + 1  # 31: una permutación de ida y vuelta (shap: max_evals // (2·p + 1) ≥ 1)
EXPLAINER = "shap.explainers.Permutation"  # método (metodo.explainer)
MASKER = "shap.maskers.Independent"  # masker intervencional (metodo.masker)
SALIDA_EXPLICADA = "logit"  # se explica el logit, no la PD
CONJUNTO_FONDO, CONJUNTO_EXPLICADOS = "train", "test"  # de dónde salen el fondo y los explicados
_TIPO_NUM = "num"  # tipo de una variable numérica (features.yaml)
_OTRA = "OTRA"  # categoría interna del preprocesador para lo no visto en train (núcleo §5.2)
_N_TOP = 5  # filas de cada ranking en el resumen impreso


# %% [markdown]
# ## Valores de Shapley del logit
#
# Sea $f(x) = \operatorname{logit} p(x) = \ln\frac{p(x)}{1 - p(x)}$ la salida del MLP **antes** de la sigmoide y
# $N = \{1, \dots, M\}$ los jugadores ($M = 15$ variables). Para una coalición $S \subseteq N$, el valor del juego
# intervencional con un **fondo** $\{b^{(1)}, \dots, b^{(K)}\}$ es
# $$v_x(S) = \frac{1}{K}\sum_{k=1}^{K} f\big(x_S,\, b^{(k)}_{\bar S}\big),$$
# es decir, las variables de $S$ toman el valor del cliente y las demás el de cada fila del fondo (la **misma** fila
# para todas las ocultas). El valor de Shapley de $j$ es el aporte marginal medio sobre los órdenes de llegada:
# $$\varphi_j(x) = \sum_{S \subseteq N\setminus\{j\}} \frac{|S|!\,(M - |S| - 1)!}{M!}
#   \big[v_x(S \cup \{j\}) - v_x(S)\big].$$
# **Eficiencia:** $\sum_j \varphi_j = v_x(N) - v_x(\varnothing)$, así que con
# $\varphi_0 = v_x(\varnothing) = \mathbb{E}_{\text{fondo}}[f]$,
# $$f(x) = \varphi_0 + \sum_{j=1}^{M} \varphi_j(x).$$
#
# - **Por qué el logit.** El puntaje de C1 es afín en el logit: $\text{score} = \text{offset} - \text{factor}\cdot
#   \operatorname{logit}(p)$. Sustituyendo la eficiencia, $\text{score} = (\text{offset} - \text{factor}\,\varphi_0) +
#   \sum_j (-\text{factor}\,\varphi_j)$: la suma de puntos es el puntaje **exacto**. Si se explicara la PD,
#   $p = \varphi_0' + \sum_j \varphi_j'$ y el puntaje $\text{offset} - \text{factor}\,\ln\frac{p}{1-p}$ **no** es lineal
#   en $p$: los $-\text{factor}\,\varphi_j'$ no sumarían el puntaje (C §6, punto caliente).
# - **$\varphi_0$ es una esperanza, no el logit del cliente medio.** $\varphi_0 = \mathbb{E}_{\text{fondo}}[f(b)]$ y en
#   general $\ne f(\mathbb{E}[b])$: con $f(z) = z^2$ y fondo $\{-1, 1\}$, $\mathbb{E}[f] = 1$ pero $f(\mathbb{E}[z]) =
#   f(0) = 0$. El MLP (ReLU) no es lineal, así que los puntos base **no** son el puntaje de un "cliente promedio".

# %% [markdown]
# ## Jugadores: las 15 variables (no las 48 columnas)
#
# El preprocesador entrega 48 columnas: una por numérica y una one-hot por categoría (incluida `OTRA`). Si cada
# columna fuera un jugador, ocultar `term=60 months` sin ocultar `term=36 months` crearía clientes con **dos** plazos
# activos (o ninguno), que el modelo nunca vio. Por eso el jugador es la **variable**: la matriz $Z$ $(n, 15)$ tiene
# la numérica transformada (la misma columna del preprocesador) y, en cada categórica, el **índice** de la categoría
# activa en `pre_json["variables"][j]["categorias"]` (incluye `DESCONOCIDO` y `OTRA`). Ocultar la categórica reemplaza
# el índice entero, así que `a_columnas` siempre reconstruye un one-hot válido. Es la versión agrupada de SHAP: la
# eficiencia sigue valiendo (solo cambia qué es un jugador).
#
# - `a_variables(X48, grupos)`: numérica = su columna; categórica = posición del 1 (error si el bloque no es one-hot).
# - `a_columnas(Z, grupos, d)`: la inversa exacta; un índice no entero, negativo, NaN o fuera de rango → `ValueError`
#   (el masker de shap trabaja en float64: $2.0$ vale como $2$).
# - Un grupo de una columna es numérico; uno de dos o más, categórico (el preprocesador siempre añade `OTRA`).
# - `funcion_logit(logit_48, grupos, d)`: $f(Z) = \texttt{logit\_48}(\texttt{a\_columnas}(Z))$ en float64 (si shap
#   recibe float32, promedia en float32: `_build_fixed_output` de `shap/utils/_masked_model.py`).

# %%
def _es_entero(valor) -> bool:
    """True si ``valor`` es un entero de Python o de numpy que no es bool."""
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))  # bool es subclase de int


def _matriz(A, n_columnas: int, nombre: str) -> np.ndarray:
    """``A`` como matriz float64 (n, n_columnas); ValueError si la forma no coincide."""
    M = np.asarray(A, dtype=np.float64)  # copia o vista en float64
    if M.ndim != 2 or M.shape[1] != n_columnas:  # matriz con las columnas esperadas
        raise ValueError(f"{nombre} debe tener forma (n, {n_columnas}); se recibió {M.shape}")  # forma
    return M  # matriz validada


def _total_columnas(grupos) -> int:
    """Número de columnas del preprocesador que cubren los grupos (deben ser 0…d−1 consecutivas)."""
    indices = np.concatenate([np.asarray(idx) for _, idx in grupos])  # todas las columnas en orden
    if not np.array_equal(indices, np.arange(indices.size)):  # grupos_columnas da bloques consecutivos
        raise ValueError("Los grupos deben cubrir las columnas 0…d−1 en orden")  # dato
    return int(indices.size)  # d


def a_variables(X48, grupos) -> np.ndarray:
    """De las columnas del preprocesador a la matriz de jugadores Z.

    Parámetros:
        X48: matriz (n, d) de ``Preprocesador.transform``.
        grupos: salida de ``puntos.grupos_columnas`` [(nombre, índices)].
    Retorna:
        np.ndarray: float64 (n, número de variables); categóricas = índice de la categoría activa.
    """
    X = _matriz(X48, _total_columnas(grupos), "X48")  # forma validada
    Z = np.empty((X.shape[0], len(grupos)), dtype=np.float64)  # una columna por variable
    for j, (nombre, idx) in enumerate(grupos):  # orden del preprocesador
        if len(idx) == 1:  # numérica: su columna
            Z[:, j] = X[:, idx[0]]  # valor transformado
            continue  # siguiente variable
        bloque = X[:, idx]  # one-hot de la categórica
        if not (np.all((bloque == 0) | (bloque == 1)) and np.all(bloque.sum(axis=1) == 1)):  # un solo 1 por fila
            raise ValueError(f"{nombre}: las columnas one-hot no tienen exactamente un 1 por fila")  # dato
        Z[:, j] = np.argmax(bloque, axis=1)  # posición del 1 = índice de la categoría
    return Z  # jugadores


def a_columnas(Z, grupos, d) -> np.ndarray:
    """De la matriz de jugadores Z a las columnas del preprocesador (inversa exacta de ``a_variables``).

    Parámetros:
        Z: matriz (n, número de variables); categóricas = índice entero (puede venir en float).
        grupos: salida de ``puntos.grupos_columnas``.
        d: número de columnas del preprocesador.
    Retorna:
        np.ndarray: float64 (n, d).
    """
    if not _es_entero(d) or int(d) != _total_columnas(grupos):  # d debe ser el de los grupos
        raise ValueError(f"d = {d!r} no coincide con las {_total_columnas(grupos)} columnas de los grupos")  # dato
    V = _matriz(Z, len(grupos), "Z")  # forma validada
    X = np.zeros((V.shape[0], int(d)), dtype=np.float64)  # one-hot en cero
    filas = np.arange(V.shape[0])  # índice de fila para la asignación one-hot
    for j, (nombre, idx) in enumerate(grupos):  # orden del preprocesador
        if len(idx) == 1:  # numérica: su columna
            X[:, idx[0]] = V[:, j]  # valor transformado
            continue  # siguiente variable
        k = V[:, j]  # índices de la categoría
        valido = np.isfinite(k) & (k == np.round(k)) & (k >= 0) & (k < len(idx))  # entero en [0, K)
        if not np.all(valido):  # NaN, ±inf, no entero o fuera de rango
            raise ValueError(f"{nombre}: índice de categoría inválido {k[~valido][0]!r} (válidos 0…{len(idx) - 1})")
        X[filas, np.asarray(idx)[k.astype(np.int64)]] = 1.0  # una sola columna activa
    return X  # columnas del preprocesador


def funcion_logit(logit_48, grupos, d):
    """Función de los jugadores Z → logit (n,) float64, a partir de una función de las columnas.

    Parámetros:
        logit_48: función X (n, d) → logit (n,).
        grupos: salida de ``puntos.grupos_columnas``.
        d: número de columnas del preprocesador.
    Retorna:
        callable: f(Z) = logit_48(a_columnas(Z)) en float64.
    """
    def f(Z) -> np.ndarray:  # la función que explica shap
        return np.asarray(logit_48(a_columnas(Z, grupos, d)), dtype=np.float64).ravel()  # float64 (n,)

    return f  # jugadores → logit


# %% [markdown]
# ## SHAP intervencional con el explainer de permutaciones
#
# **Masker.** `shap.maskers.Independent(fondo, max_samples=n_fondo)` implementa $v_x(S)$ de arriba: cada variable
# oculta toma el valor de cada fila del fondo (`Tabular.__call__` en `shap/maskers/_tabular.py`). El fondo son
# `n_fondo` filas de **train** sin reemplazo (`make_rng(base_seed, 0)`): no se usa nada de val ni de test para
# definir la referencia. **Limitación:** es SHAP *intervencional*, que trata las variables como independientes. Al
# reemplazar `loan_amnt` por el de otra fila sin cambiar `annual_inc` o `revol_bal` se crean combinaciones poco
# realistas (un crédito grande con un ingreso pequeño) y el modelo se evalúa fuera de los datos que vio; con variables
# correlacionadas, el crédito se reparte entre ellas de forma que depende de esa extrapolación.
# TODO(cita): Aas, Jullum y Løland (2021, dependencia entre variables en SHAP); Janzing, Minorics y Blöbaum (2020,
# SHAP intervencional frente a observacional).
#
# **Explainer.** `shap.explainers.Permutation` (shap 0.54.1, `explainers/_permutation.py`) recorre una permutación
# $\pi$ de las variables **de ida y de vuelta** (muestreo antitético). Por fila, `explain_row` evalúa
# $v_x(\varnothing), v_x(\{\pi_1\}), \dots, v_x(N)$ y de vuelta hasta $v_x(\varnothing)$ (máscaras $2M + 1$) y suma a
# cada variable su aporte marginal en los dos sentidos; al final divide entre $2\,n_\pi$. Para **cada** recorrido
# $$\sum_j \big[v_x(\{\pi_1..\pi_j\}) - v_x(\{\pi_1..\pi_{j-1}\})\big] = v_x(N) - v_x(\varnothing) = f(x) - \varphi_0$$
# (suma telescópica). El docstring lo dice: "guarantees local accuracy (additivity) by iterating completely through
# an entire permutation of the features in both forward and reverse directions (antithetic sampling). If we do this
# once, then we get the exact SHAP values for models with up to second order interaction effects."
#
# **La telescopia es exacta solo si shap reevalúa toda variable que difiere del fondo** (hallazgo H-24). shap no
# reevalúa las filas del fondo en las que la variable "no cambia", y decide eso con `np.isclose(x, fondo)`
# (`Tabular.invariants` en `shap/maskers/_tabular.py`; `rtol` 1e-5, `atol` 1e-8), **no** con igualdad. Si el valor del
# cliente y el de una fila del fondo difieren menos que esa tolerancia, esa fila conserva el valor del fondo cuando la
# variable "se enciende", el último paso no llega a $f(x)$ y la suma no telescopia. Pasa con los float32 del
# preprocesador: `revol_bal` = 81 830 frente a una fila recortada en el p99 de train (81 831,62) son `isclose`, y con
# esa fila en el fondo el error de eficiencia fue 2,2e-6; con los tamaños aprobados y `base_seed` 34, 2,7e-9.
# **Arreglo:** `valores_shap` codifica cada jugador como el **índice** de su valor en la tabla de valores únicos de esa
# columna en fondo ∪ explicados, y explica $g(C) = f(\text{decodificar}(C))$. Valores iguales tienen el mismo código y
# valores distintos, códigos enteros distintos; como todos son menores que `MAX_CODIGOS` $= (1 - 10^{-8})/10^{-5}$,
# $|k - (k+1)| = 1 > 10^{-8} + 10^{-5}(k+1)$ y dos códigos distintos nunca son `isclose`. El juego de Shapley es el
# mismo ($g$ y $f$ dan lo mismo en cada coalición): solo cambia que shap ahora compara por igualdad exacta.
#
# Con eso la eficiencia es **exacta con cualquier número de permutaciones**; lo que cambia con más permutaciones es la
# varianza del reparto entre variables. Con $M = 15$, `max_evals` $= 500$ da $n_\pi = 500 // 31 = 16$ permutaciones
# por fila (`npermutations = max_evals // (2 * len(inds) + 1)`). `inds` son las variables que **varían** frente al
# fondo (`MaskedModel.varying_inputs`): una variable igual (ahora, exactamente igual) a la del cliente en todas las
# filas del fondo no entra, tiene $\varphi = 0$ y la fila hace más permutaciones; `n_permutaciones` del JSON es el
# número por fila cuando las 15 varían. TODO(cita): Štrumbelj y Kononenko (2014, estimación de Shapley por muestreo de
# permutaciones).
#
# **Por qué no los otros explainers** (docstrings y código de shap 0.54.1 instalado):
# - `DeepExplainer` (`explainers/_deep/__init__.py`): "Meant to approximate SHAP values for deep learning models"
#   (reglas de DeepLIFT por nodo); aproxima los valores aunque sumen $f(x) - \mathbb{E}[f(x)]$.
# - `GradientExplainer` (`explainers/_gradient.py`): gradientes esperados, "Integrated gradients values are a bit
#   different from SHAP values" y la estimación "converges to attributions that sum to the difference": la suma solo
#   se cumple en el límite del muestreo, no en cada corrida.
# - `KernelExplainer` (`explainers/_kernel.py`): "uses a special weighted linear regression"; impone la suma como
#   restricción ("eliminate one variable with the constraint that all features sum to the output") pero los $\varphi$
#   son coeficientes de una regresión sobre coaliciones muestreadas (`nsamples = 2 * X.shape[1] + 2048` por defecto) y,
#   desde 0.47.0, `l1_reg = "num_features(10)"` selecciona por defecto **10** variables: con 15 jugadores dejaría 5 en
#   cero. Además es mucho más costoso por fila.
# - El de permutaciones solo necesita evaluar $f$ (sirve para el forward de la app, que no es un módulo de PyTorch),
#   garantiza la eficiencia por construcción y es exacto con interacciones de orden 2.
#
# **Estado global de numpy.** `PermutationExplainer.__init__` llama a `np.random.seed(seed)` y baraja con
# `np.random.shuffle` (estado **global**, que el proyecto no usa: `src/common/seeds.py`). `valores_shap` guarda el
# estado con `np.random.get_state()` y lo restaura al salir, también si falla: llamar a SHAP no altera lo aleatorio del
# resto del programa, y la misma `semilla` da los mismos $\varphi$.
#
# **$\varphi_0$.** shap devuelve un $\varphi_0$ por fila; todos son la misma media del fondo. Se exige que sean finitos
# (si $f$ da NaN o ±inf, mensaje propio) y que no difieran más de `TOL_PHI0` (otra BLAS podría cambiar el último bit);
# se usa su media.

# %%
def _codificar(B: np.ndarray, E: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Códigos enteros (en float64) de cada jugador en la tabla de valores únicos de fondo ∪ explicados (H-24)."""
    if not (np.all(np.isfinite(B)) and np.all(np.isfinite(E))):  # NaN no tiene un código único
        raise ValueError("El fondo y los explicados deben ser finitos")  # dato
    tablas = [np.unique(np.concatenate([B[:, j], E[:, j]])) for j in range(B.shape[1])]  # valores por jugador
    if max(t.size for t in tablas) > MAX_CODIGOS:  # códigos grandes vecinos serían isclose
        raise ValueError(f"Demasiados valores distintos por variable (> {MAX_CODIGOS}) para codificar sin isclose")
    def codigos(M: np.ndarray) -> np.ndarray:  # índice de cada valor en su tabla
        return np.column_stack([np.searchsorted(t, M[:, j]) for j, t in enumerate(tablas)]).astype(np.float64)
    return tablas, codigos(B), codigos(E)  # tablas, fondo y explicados codificados


def _decodificar(C, tablas: list[np.ndarray]) -> np.ndarray:
    """Valores de los jugadores a partir de sus códigos (inversa exacta de ``_codificar``)."""
    C = np.asarray(C, dtype=np.float64)  # el masker trabaja en float64
    return np.column_stack([t[np.rint(C[:, j]).astype(np.int64)] for j, t in enumerate(tablas)])  # código → valor


def valores_shap(f, fondo, explicados, max_evals, semilla) -> tuple[np.ndarray, float]:
    """Valores de Shapley de ``f`` con el explainer de permutaciones y el masker Independent.

    Parámetros:
        f: función Z (m, p) → salida (m,) float64 (el logit).
        fondo: matriz (K, p) del fondo.
        explicados: matriz (n, p) de los casos que se explican.
        max_evals: evaluaciones por fila (≥ 2p + 1).
        semilla: semilla de las permutaciones (entero ≥ 0).
    Retorna:
        tuple[np.ndarray, float]: (φ float64 (n, p), φ₀ = E_fondo[f]).
    """
    B = np.asarray(fondo, dtype=np.float64)  # fondo en float64
    E = _matriz(explicados, B.shape[1], "explicados")  # mismas variables que el fondo
    tablas, C_fondo, C_expl = _codificar(B, E)  # códigos: shap compara por igualdad exacta (H-24)

    def g(C) -> np.ndarray:  # el mismo juego: f de los valores decodificados
        return np.asarray(f(_decodificar(C, tablas)), dtype=np.float64)  # float64: shap promedia en ese tipo

    estado = np.random.get_state()  # shap siembra y baraja el generador global
    try:  # restaurar el estado global pase lo que pase
        masker = shap.maskers.Independent(C_fondo, max_samples=C_fondo.shape[0])  # todo el fondo, sin submuestreo
        explainer = shap.explainers.Permutation(g, masker, seed=int(semilla))  # np.random.seed(semilla)
        explicacion = explainer(C_expl, max_evals=int(max_evals), silent=True)  # φ por fila (sin barra de progreso)
    finally:  # también si shap falla
        np.random.set_state(estado)  # el resto del programa no ve el cambio
    phi = np.asarray(explicacion.values, dtype=np.float64)  # (n, p)
    base = np.asarray(explicacion.base_values, dtype=np.float64).ravel()  # φ₀ de cada fila
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(phi))):  # f dio NaN o ±inf
        raise ValueError("φ₀ o φ no finitos: la función explicada devolvió NaN o ±inf")  # dato
    if np.ptp(base) > TOL_PHI0:  # φ₀ = E_fondo[f] no depende del cliente
        raise ValueError(f"shap devolvió φ₀ distintos por fila (rango {np.ptp(base):.3e} > {TOL_PHI0:.0e})")
    return phi, float(np.mean(base))  # φ y φ₀


def error_eficiencia(phi, phi0, logit) -> float:
    """máx |φ₀ + Σⱼ φⱼ − logit| sobre las filas.

    Parámetros:
        phi: (n, p).
        phi0: φ₀.
        logit: salida explicada (n,).
    Retorna:
        float: el error máximo.
    """
    suma = float(phi0) + np.asarray(phi, dtype=np.float64).sum(axis=1)  # salida reconstruida
    return float(np.max(np.abs(suma - np.asarray(logit, dtype=np.float64))))  # peor fila


# %% [markdown]
# ## Verificación con la logística: forma cerrada
#
# La logística es lineal en las columnas: $f(x) = \beta_0 + \sum_c \beta_c x_c$. En un juego aditivo
# $f = \beta_0 + \sum_j g_j(z_j)$ el aporte marginal de $j$ no depende de la coalición, así que
# $\varphi_j = g_j(z_j) - \mathbb{E}_{\text{fondo}}[g_j]$:
# - numérica: $\varphi_j = \beta_j\,(z_j - \bar z_j)$, con $\bar z_j$ la media del fondo;
# - categórica (agrupada): $g_j(z) = \sum_k \beta_k\,\mathbb{1}[z = k]$, luego
#   $\varphi_j = \sum_k \beta_k\,\big(\mathbb{1}[z_j = k] - \hat\pi_k\big)$, con $\hat\pi_k$ la frecuencia de la
#   categoría $k$ en el fondo.
#
# Las dos son $\sum_{c \in j} \beta_c\,(x_c - \bar x_c)$ sobre las columnas de la variable. `construir_scorecard_mlp`
# corre **el mismo** `valores_shap` con la logística sobre los primeros `n_estabilidad` explicados (para que el comando
# quepa en la demo) y exige $|\varphi^{\text{perm}} - \varphi^{\text{cerrada}}| \le$ `TOL_FORMA_CERRADA` y
# $\varphi_0 = $ media del logit del fondo: si el código de SHAP (jugadores, masker, fondo) estuviera mal, la
# logística lo delataría.

# %%
def shap_cerrado_lineal(Z, fondo, coef, grupos) -> np.ndarray:
    """SHAP intervencional exacto de un modelo lineal en las columnas, agrupado por variable.

    Parámetros:
        Z: explicados (n, p) en jugadores.
        fondo: fondo (K, p) en jugadores.
        coef: β (d,) del modelo lineal.
        grupos: salida de ``puntos.grupos_columnas``.
    Retorna:
        np.ndarray: float64 (n, p).
    """
    b = np.asarray(coef, dtype=np.float64).ravel()  # β como vector
    X = a_columnas(Z, grupos, b.size)  # columnas de los explicados
    X_fondo = a_columnas(fondo, grupos, b.size)  # columnas del fondo
    D = (X - X_fondo.mean(axis=0)) * b  # βc · (xc − media del fondo) por columna
    return np.column_stack([D[:, idx].sum(axis=1) for _, idx in grupos])  # suma por variable


# %% [markdown]
# ## Importancia global, dirección y estabilidad
#
# - **Importancia global:** $I_j = \frac{1}{n}\sum_i |\varphi_j(x_i)|$ sobre los explicados de test, de mayor a menor
#   (`puesto` 1 = la más importante). Mide cuánto mueve cada variable el logit, sin signo.
# - **Dirección:** numérica → $\rho$ de Spearman entre el valor **crudo** (no vacío) y $\varphi_j$ ($\rho > 0$: a mayor
#   valor, más log-odds de malo); `None` si el crudo o $\varphi$ son constantes (no definida; el JSON no admite NaN).
#   Categórica → $\varphi_j$ medio y número de explicados por categoría, nombrada con el **código limpio** del
#   preprocesador (el índice de $Z$; incluye `DESCONOCIDO` y `OTRA`), no con el texto crudo (`" 36 months"`).
# - **Estabilidad:** los primeros `n_estabilidad` explicados se vuelven a explicar con `base_seed + 1` (otro fondo de
#   train y otras permutaciones) y se comparan con sus $\varphi$ de la semilla base: rankings de $I_j$ en las dos
#   corridas y $\rho$ de Spearman entre los dos vectores de importancia. Mide cuánto depende el ranking del azar del
#   método (fondo y permutaciones), no del modelo.

# %%
def _spearman(a, b):
    """ρ de Spearman, o None si no está definido (menos de 2 valores o alguno constante)."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)  # float64
    if x.size < 2 or np.all(x == x[0]) or np.all(y == y[0]):  # sin variación no hay rangos que comparar
        return None  # no definido
    return float(spearmanr(x, y).statistic)  # correlación de Pearson de los rangos (empates promediados)


def importancia_global(phi, nombres) -> list[dict]:
    """Media de |φ| por variable, de mayor a menor.

    Parámetros:
        phi: (n, p).
        nombres: p nombres en el orden de las columnas de ``phi``.
    Retorna:
        list[dict]: [{nombre, media_abs_phi, puesto}] con puesto 1 = la más importante.
    """
    media = np.abs(np.asarray(phi, dtype=np.float64)).mean(axis=0)  # media |φ|
    orden = np.argsort(-media, kind="stable")  # de mayor a menor
    return [{"nombre": nombres[j], "media_abs_phi": float(media[j]), "puesto": i + 1}  # puesto desde 1
            for i, j in enumerate(orden)]  # una entrada por variable


def direccion(phi, crudos, features) -> dict:
    """Dirección del efecto de cada variable sobre el logit.

    Parámetros:
        phi: (n, p) en el orden de ``features``.
        crudos: DataFrame (n filas) con las numéricas crudas y las categóricas con el código limpio.
        features: variables de ``features.yaml`` (nombre, tipo y, en las categóricas, categorias_es).
    Retorna:
        dict: {nombre: {tipo: "num", spearman, n} | {tipo: "cat", phi_medio {cat: …}, n {cat: …}}}.
    """
    P = np.asarray(phi, dtype=np.float64)  # φ en float64
    salida = {}  # nombre → dirección
    for j, f in enumerate(features):  # orden de features.yaml
        if f["tipo"] == _TIPO_NUM:  # ρ de Spearman con el valor crudo
            x = pd.to_numeric(crudos[f["nombre"]], errors="raise").to_numpy(dtype=np.float64)  # crudo
            con = ~np.isnan(x)  # solo los no vacíos
            salida[f["nombre"]] = {"tipo": "num", "spearman": _spearman(x[con], P[con, j]), "n": int(con.sum())}
            continue  # siguiente variable
        valores = crudos[f["nombre"]].astype(str).to_numpy()  # códigos limpios
        orden = [c for c in f.get("categorias_es", {}) if c in set(valores)]  # orden de la app
        orden += sorted(set(valores) - set(orden))  # luego las demás (p. ej. OTRA)
        salida[f["nombre"]] = {  # φ medio y conteo por categoría
            "tipo": "cat",  # categórica
            "phi_medio": {c: float(P[valores == c, j].mean()) for c in orden},  # φ medio
            "n": {c: int((valores == c).sum()) for c in orden},  # explicados con esa categoría
        }
    return salida  # una entrada por variable


def estabilidad(phi_a, phi_b, nombres) -> dict:
    """Compara la importancia global de dos corridas de SHAP sobre los mismos clientes.

    Parámetros:
        phi_a: φ (n, p) de la semilla base.
        phi_b: φ (n, p) de la semilla base + 1.
        nombres: p nombres.
    Retorna:
        dict: {ranking_a, ranking_b, spearman_importancia}.
    """
    ig_a, ig_b = importancia_global(phi_a, nombres), importancia_global(phi_b, nombres)  # una vez cada corrida
    media_a = {e["nombre"]: e["media_abs_phi"] for e in ig_a}  # nombre → importancia (corrida a)
    media_b = {e["nombre"]: e["media_abs_phi"] for e in ig_b}  # nombre → importancia (corrida b)
    return {"ranking_a": [e["nombre"] for e in ig_a], "ranking_b": [e["nombre"] for e in ig_b],  # rankings
            "spearman_importancia": _spearman([media_a[n] for n in nombres], [media_b[n] for n in nombres])}  # ρ


# %% [markdown]
# ## Importancia por permutación (contraste)
#
# Para el modelo $\hat p$ y el test $(Z, y)$: $\text{AUC}_0 = \text{AUC}(y, \hat p(Z))$ y $\text{LL}_0$ la log-loss de
# la Parte B (`src.pd.logistica.logloss`). Para cada variable $j$ y repetición $r = 1..k$ se permutan las filas de la
# **columna $j$ de $Z$** (la categórica entera, por índice: la fila sigue siendo un one-hot válido) y
# $$\Delta\text{AUC}_{j,r} = \text{AUC}_0 - \text{AUC}(y, \hat p(Z^{\pi_r}_j)),\qquad
#   \Delta\text{LL}_{j,r} = \text{LL}(y, \hat p(Z^{\pi_r}_j)) - \text{LL}_0;$$
# se informan la media y la desviación **muestral** de las $k$ repeticiones ($k = 1$ → 0, núcleo §5.4). Las
# permutaciones salen de `make_rng(semilla, 0)` y, con la misma semilla, son las mismas para los dos modelos.
#
# **Diferencia con SHAP:** la permutación es **global y sobre el desempeño** (cuánto empeora la discriminación o la
# calibración si se rompe el vínculo de la variable con $y$); SHAP es **local y sobre la salida** (cuánto mueve la
# variable el logit de cada cliente, sin mirar $y$). Una variable puede mover mucho el logit y aportar poco a la AUC.
# Además, permutar también crea combinaciones poco realistas con variables correlacionadas. TODO(cita): Breiman (2001,
# importancia por permutación en bosques aleatorios); Fisher, Rudin y Dominici (2019, *model reliance*).

# %%
def _media_desv(x: list[float]) -> tuple[float, float]:
    """Media y desviación muestral (n = 1 → 0, núcleo §5.4)."""
    a = np.asarray(x, dtype=np.float64)  # valores de las repeticiones
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)  # media y s


def importancia_permutacion(prob_fn, Z, y, grupos, d, k, semilla) -> dict:
    """Caída de AUC y aumento de log-loss al permutar cada variable de Z.

    Parámetros:
        prob_fn: función X (n, d) → PD (n,).
        Z: jugadores (n, p) del conjunto evaluado.
        y: objetivo 0/1 (n,).
        grupos: salida de ``puntos.grupos_columnas``.
        d: número de columnas del preprocesador.
        k: repeticiones por variable (≥ 1).
        semilla: semilla de las permutaciones.
    Retorna:
        dict: {auc_base, logloss_base, variables: [{nombre, caida_auc_media, caida_auc_desv, aumento_logloss_media,
        aumento_logloss_desv}]}.
    """
    V = _matriz(Z, len(grupos), "Z")  # jugadores en float64
    yy = np.asarray(y)  # objetivo
    p0 = np.asarray(prob_fn(a_columnas(V, grupos, d)), dtype=np.float64)  # PD sin permutar
    auc0, ll0 = auc(yy, p0), logloss(yy, p0)  # referencias
    rng = make_rng(int(semilla), 0)  # generador de las permutaciones
    variables = []  # resultado por variable
    for j, (nombre, _) in enumerate(grupos):  # orden del preprocesador
        caidas, aumentos = [], []  # una entrada por repetición
        for _ in range(int(k)):  # k permutaciones de la columna j
            Vp = V.copy()  # las demás variables quedan igual
            Vp[:, j] = V[rng.permutation(V.shape[0]), j]  # variable entera (índice si es categórica)
            p = np.asarray(prob_fn(a_columnas(Vp, grupos, d)), dtype=np.float64)  # PD permutada
            caidas.append(auc0 - auc(yy, p))  # caída de AUC
            aumentos.append(logloss(yy, p) - ll0)  # aumento de log-loss
        (ca_m, ca_s), (al_m, al_s) = _media_desv(caidas), _media_desv(aumentos)  # resúmenes
        variables.append({"nombre": nombre, "caida_auc_media": ca_m, "caida_auc_desv": ca_s,
                          "aumento_logloss_media": al_m, "aumento_logloss_desv": al_s})  # una fila
    return {"auc_base": float(auc0), "logloss_base": float(ll0), "variables": variables}  # resultado


# %% [markdown]
# ## Validación del bloque `shap`
#
# Claves exactas `CLAVES_SHAP`, todas enteras (no bool, no real): `base_seed` $\ge 0$; `n_fondo`, `n_explicar`,
# `n_estabilidad`, `k_permutacion` $\ge 1$; `n_estabilidad` $\le$ `n_explicar` (se re-explican los primeros);
# `max_evals` $\ge 2\cdot 15 + 1 = 31$: con menos, shap no completa una permutación de ida y vuelta (`npermutations
# == 0` → error en `explain_row`).

# %%
def validar_cfg_shap(shap_cfg) -> dict:
    """Valida el bloque ``shap`` de la config; cualquier falla → ValueError. No muta la entrada.

    Parámetros:
        shap_cfg: bloque shap.
    Retorna:
        dict: copia con los valores como int.
    """
    if not isinstance(shap_cfg, dict):  # lista, None, escalar…
        raise ValueError(f"El bloque shap debe ser un diccionario; se recibió {type(shap_cfg).__name__}")  # tipo
    faltan = sorted(set(CLAVES_SHAP) - set(shap_cfg))  # claves sin valor por defecto
    sobran = sorted(map(str, set(shap_cfg) - set(CLAVES_SHAP)))  # erratas
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"El bloque shap es inválido: faltan {faltan}, sobran {sobran}")  # claves
    minimos = {"base_seed": 0, "n_fondo": 1, "n_explicar": 1, "n_estabilidad": 1, "max_evals": MIN_MAX_EVALS,
               "k_permutacion": 1}  # mínimo de cada clave
    for clave in CLAVES_SHAP:  # cada valor
        valor = shap_cfg[clave]  # valor configurado
        if not _es_entero(valor) or valor < minimos[clave]:  # entero (no bool) y en rango
            raise ValueError(f"shap.{clave} debe ser un entero ≥ {minimos[clave]}; se recibió {valor!r}")  # valor
    if shap_cfg["n_estabilidad"] > shap_cfg["n_explicar"]:  # se re-explican los primeros n_estabilidad
        raise ValueError(f"shap.n_estabilidad ({shap_cfg['n_estabilidad']}) no puede superar shap.n_explicar "
                         f"({shap_cfg['n_explicar']})")  # rango
    return {clave: int(shap_cfg[clave]) for clave in CLAVES_SHAP}  # copia en int


# %% [markdown]
# ## Preparación (carga y transformación)
#
# `preparar_mlp(cfg)` carga `features.yaml`, el preprocesador (ajustado solo con train por A), los datos, el MLP y la
# logística, y por split calcula $X$ = `pre.transform`, el **logit del forward float64** de las capas del MLP
# (`exportar_app.capas_mlp` + `forward_desde_json`, la **misma función que calcula la app**), la PD $= \sigma(\cdot)$
# e $y$. No depende de `scorecard.*`, `distribucion.*` ni `shap.*`: el mismo preparado sirve para `--set
# scorecard.pdo=40`. La logística (verificación y permutación) se lee de `rutas.logistica`, clave que solo existe con
# `modelo: mlp`. Guarda también `tiempo_s` (lo que tardó la preparación; `tiempos_s.preparacion` del JSON).
#
# **Forward float64 frente a `MLPPD`.** `MLPPD.predict_logit` corre en torch con float32; el forward de la app repite
# las mismas capas en float64. Explicar el forward float64 tiene una ventaja: los puntos explican **exactamente** la
# función que la persona ve en la app. La diferencia con `MLPPD` en los explicados se informa en
# `diferencia_mlppd {max_abs_pd, max_abs_logit}` (del orden del redondeo de float32).

# %%
def preparar_mlp(cfg: dict) -> dict:
    """Carga y transforma los datos de la scorecard del MLP (no depende de scorecard.*, distribucion.* ni shap.*).

    Parámetros:
        cfg: config con ``modelo: mlp`` (se valida completa).
    Retorna:
        dict: {rutas, features, pre_json, grupos, d, red, mlppd, logistica, crudos_train, crudos_test, ids_test,
        fracciones_train, evidencia, splits {split: {X, logit, prob, y}}, tiempo_s}; los arrays son de solo lectura.
    """
    inicio = time.perf_counter()  # tiempo de la preparación
    cfg = validar_cfg_puntos(cfg)  # copia validada
    if cfg["modelo"] != MODELO_MLP:  # la logística es puntos.preparar_splits
        raise ValueError(f"preparar_mlp necesita modelo {MODELO_MLP!r}; se recibió {cfg['modelo']!r}")  # valor
    rutas = cfg["rutas"]  # archivos
    features = load_config(rutas["features"])["features"]  # variables (orden de los jugadores)
    pre_json = load_json(rutas["preprocesador"])  # receta del preprocesador
    pre = Preprocesador.from_dict(pre_json)  # preprocesador ajustado con train (A)
    grupos = grupos_columnas(pre_json)  # columnas de cada variable (valida los nombres)
    if [f["nombre"] for f in features] != [nombre for nombre, _ in grupos]:  # mismas variables y orden
        raise ValueError("features.yaml y el preprocesador no tienen las mismas variables en el mismo orden")  # dato
    if len(features) != N_VARIABLES:  # MIN_MAX_EVALS de validar_cfg_shap supone 15 jugadores
        raise ValueError(f"Se esperan {N_VARIABLES} variables en features.yaml; hay {len(features)}")  # dato
    capas, mlppd = capas_mlp(rutas["modelo"])  # capas en float64 y el MLPPD cargado
    red = {"capas": capas, "salida": SALIDA_RED}  # bloque red del contrato de la app (núcleo §5.5)
    logistica = LogisticaPD.cargar(rutas[CLAVE_RUTA_LOGISTICA])  # logística de B3 (rutas.logistica)
    datos = pd.read_parquet(rutas["datos"])  # todas las filas con su split
    splits = {}  # split → matrices y predicciones
    for s in SPLITS:  # train, val, test
        df = datos[datos["split"] == s].reset_index(drop=True)  # filas del split
        if df.empty:  # sin filas no hay distribución
            raise ValueError(f"El split {s!r} no tiene filas")  # dato
        X = pre.transform(df)  # float32, columnas en orden fijo
        logit = forward_desde_json(red, X)  # forward float64 de la app
        splits[s] = {"X": X, "logit": logit, "prob": expit(logit), "y": df["target"].to_numpy()}  # por split
    train = datos[datos["split"] == "train"]  # crudos de train
    test = datos[datos["split"] == "test"].reset_index(drop=True)  # crudos de test (mismo orden que X de test)
    crudos_train = {f["nombre"]: pd.to_numeric(train[f["nombre"]], errors="raise").to_numpy(dtype=np.float64)
                    for f in features if f["tipo"] == _TIPO_NUM}  # numéricas antes de transformar
    preparado = {
        "rutas": dict(rutas), "features": features, "pre_json": pre_json, "grupos": grupos,  # receta
        "d": len(pre_json["columnas_salida"]), "red": red, "mlppd": mlppd, "logistica": logistica,  # modelos
        "crudos_train": crudos_train,  # para los bordes de los tramos (solo train)
        "crudos_test": test[[f["nombre"] for f in features]],  # para la dirección y los tramos de los explicados
        "ids_test": test["id"].astype(str).to_numpy(),  # identificadores de test
        "fracciones_train": np.mean(splits["train"]["X"], axis=0, dtype=np.float64),  # media one-hot de train
        "evidencia": {"sha256_particion": huella_particion(datos["id"].tolist(), datos["split"].tolist()),
                      "sha256_preprocesador": huella_archivo(rutas["preprocesador"])},  # núcleo §6
        "splits": splits,  # matrices y predicciones
    }
    arrays = [*crudos_train.values(), preparado["ids_test"], preparado["fracciones_train"],
              *(a for d in splits.values() for a in d.values())]  # arrays compartidos
    for arr in arrays:  # solo lectura: varios consumidores comparten el preparado
        arr.setflags(write=False)  # un error de escritura falla en vez de corromper
    preparado["tiempo_s"] = time.perf_counter() - inicio  # segundos de la preparación
    return preparado  # preparado


# %% [markdown]
# ## Puntos del MLP por variable
#
# De la eficiencia y C1: $\text{score}(x) = \underbrace{\text{offset} - \text{factor}\,\varphi_0}_{\text{puntos base}}
# + \sum_j \underbrace{(-\text{factor}\,\varphi_j(x))}_{\text{puntos}_j}$, exacto (aceptación de C4; se comprueba en
# cada corrida con `TOL_ADITIVIDAD`). **Signo:** $\text{factor} > 0$; $\varphi_j > 0$ (la variable sube el
# log-odds de malo respecto del fondo) ⇒ **menos** puntos. Con `--set scorecard.pdo=40` el factor se duplica y,
# como $\varphi$ no depende de la scorecard, todos los puntos y $\text{puntos base} - \text{base}$ también.
#
# A diferencia de la logística, los puntos de una variable **no** son una función de su valor: dependen del resto del
# cliente (interacciones del MLP). La tabla de cada variable resume los puntos de los **explicados de test**:
# - numérica: tramos $[\text{desde}, \text{hasta})$ con bordes = `percentiles_tabla` de los crudos **no vacíos de
#   train** y colas abiertas (`None`); un valor en un borde va al tramo superior (como las bandas de C3). Por tramo,
#   `n` y `puntos_medios` (`None` si $n = 0$); `n_sin_valor` = explicados con el crudo vacío (imputados por el
#   preprocesador, fuera de los tramos). `rango_puntos` = máx − mín de `puntos_medios` entre los tramos interiores a
#   `percentiles_rango` (p1–p99) con $n > 0$ (`None` si no hay ninguno). Con bordes repetidos los tramos
#   $[b, b)$ quedan vacíos: `pub_rec` tiene bordes de train $[0, 0, 0, 0, 0, 0, 1]$, así que el 0 cae en $[0, 1)$, el
#   1 o más en la cola $[1, \infty)$ y $[0, 1)$ es el **único** tramo interior con datos: su `rango_puntos` en el MLP es
#   0 (en la logística de C3, que evalúa la función en p1 y p99, no lo es);
# - categórica: por categoría (orden de `categorias_es`, sin `OTRA`), `n`, `puntos_medios` y `fraccion_train` (media de
#   la columna one-hot en train, como C3); `rango_puntos` entre las categorías con `fraccion_train` $\ge$
#   `fraccion_min_categoria` y $n > 0$.
#
# Los bordes y las fracciones salen **solo de train**; los puntos medios, de los explicados de test.

# %%
def _media_o_none(x: np.ndarray):
    """Media en float (−0.0 → 0.0), o None si ``x`` está vacío."""
    return float(np.mean(x)) + 0.0 if x.size else None  # vacío → None


def _rango(valores: list):
    """máx − mín (−0.0 → 0.0), o None si no hay valores."""
    return float(max(valores) - min(valores)) + 0.0 if valores else None  # sin valores no hay rango


def _entrada_numerica(f: dict, crudos_train, crudos_e, P_j, dist: dict) -> dict:
    """Tramos de una numérica con bordes de train y puntos medios de los explicados."""
    tr = np.asarray(crudos_train, dtype=np.float64)  # crudos de train
    tr = tr[~np.isnan(tr)]  # solo los no vacíos
    if tr.size == 0:  # sin datos no hay percentiles
        raise ValueError(f"{f['nombre']}: no hay valores no vacíos en train")  # dato
    pct = list(dist["percentiles_tabla"])  # percentiles de los bordes
    bordes = np.percentile(tr, pct)  # interpolación lineal (por defecto de numpy)
    x = np.asarray(crudos_e, dtype=np.float64)  # crudos de los explicados
    con = ~np.isnan(x)  # explicados con valor
    tramo = np.searchsorted(bordes, x[con], side="right")  # [desde, hasta): en un borde → tramo superior
    P = np.asarray(P_j, dtype=np.float64)[con]  # puntos de los explicados con valor
    pa, pb = dist["percentiles_rango"]  # tramos interiores para el rango
    tramos, interiores = [], []  # resultado y puntos medios interiores
    for k in range(len(bordes) + 1):  # cola inferior, tramos internos, cola superior
        m = tramo == k  # explicados del tramo
        medio = _media_o_none(P[m])  # puntos medios
        tramos.append({"desde": float(bordes[k - 1]) if k > 0 else None,  # cola inferior abierta
                       "hasta": float(bordes[k]) if k < len(bordes) else None,  # cola superior abierta
                       "n": int(m.sum()), "puntos_medios": medio})  # conteo y media
        if 0 < k < len(bordes) and pa <= pct[k - 1] and pct[k] <= pb and medio is not None:  # interior con datos
            interiores.append(medio)  # entra en el rango
    return {"nombre": f["nombre"], "tipo": "num", "etiqueta_es": f["etiqueta_es"], "tramos": tramos,  # tabla
            "n_sin_valor": int((~con).sum()), "rango_puntos": _rango(interiores)}  # vacíos y rango


def _entrada_categorica(f: dict, v_pre: dict, z_j, P_j, fracciones: dict, dist: dict) -> dict:
    """Puntos medios de los explicados por categoría (orden de categorias_es, sin OTRA)."""
    etiquetas = f.get("categorias_es") or {}  # código → etiqueta en español
    de_train = [c for c in v_pre["categorias"] if c != _OTRA]  # categorías vistas en train
    if set(etiquetas) != set(de_train):  # como C2 y C3
        raise ValueError(f"{f['nombre']}: categorias_es {sorted(etiquetas)} no coincide con las categorías del "
                         f"preprocesador sin OTRA {sorted(de_train)}")  # dato
    z = np.asarray(z_j, dtype=np.float64)  # índices de los explicados
    P = np.asarray(P_j, dtype=np.float64)  # puntos de los explicados
    categorias, frecuentes = [], []  # resultado y puntos para el rango
    for c in etiquetas:  # orden de categorias_es
        m = z == v_pre["categorias"].index(c)  # explicados con esa categoría
        medio = _media_o_none(P[m])  # puntos medios
        categorias.append({"categoria": c, "etiqueta": etiquetas[c], "n": int(m.sum()), "puntos_medios": medio,
                           "fraccion_train": float(fracciones[c])})  # una fila
        if fracciones[c] >= dist["fraccion_min_categoria"] and medio is not None:  # frecuente y con explicados
            frecuentes.append(medio)  # entra en el rango
    return {"nombre": f["nombre"], "tipo": "cat", "etiqueta_es": f["etiqueta_es"], "categorias": categorias,
            "rango_puntos": _rango(frecuentes)}  # tabla y rango


# %% [markdown]
# ## Construcción de `scorecard_mlp.json` y `shap_mlp.json`
#
# `construir_scorecard_mlp` sigue estos pasos, uno por ayudante (cada etapa se cronometra en `tiempos_s`):
# 1. `_shap_y_estabilidad` — fondo: `n_fondo` filas de **train** y explicados: `n_explicar` filas de **test**, sin
#    reemplazo con el mismo generador `make_rng(base_seed, 0)` (primero el fondo, después los explicados); SHAP del
#    forward float64 del MLP con `semilla = base_seed` (`shap_base`). Estabilidad: fondo de train con
#    `make_rng(base_seed + 1, 0)` y permutaciones con `base_seed + 1` sobre los primeros `n_estabilidad` explicados
#    (`estabilidad`; su eficiencia también se exige y entra en `eficiencia.error_max`). Eficiencia $\le$
#    `TOL_EFICIENCIA` o error.
# 2. `_verificar_logistica` — forma cerrada de la logística sobre los primeros `n_estabilidad` explicados y el mismo
#    fondo (`verificacion_logistica`).
# 3. `_scorecard_json` — puntos, tablas por variable, puntaje del MLP (forward float64) en train/val/test completos con
#    cortes = deciles de train (como C3) y aditividad pb + Σ puntos = score de C1 $\le$ `TOL_ADITIVIDAD` o error.
# 4. `_permutacion_dos_modelos` — importancia por permutación en **test** de la logística y del MLP
#    (PD = σ(forward float64)), semilla `base_seed` (`permutacion_logistica`, `permutacion_mlp`).
# 5. `_shap_json` — ensambla `shap_mlp.json`; `tiempos_s.preparacion` es el de `preparar_mlp` (aunque el preparado
#    se reutilice).
#
# Pedir más filas de las que hay (`n_fondo` > train o `n_explicar` > test) es un error de config que solo se ve con
# los datos cargados: `ConfigInvalida` (subclase de `ValueError`), que `puntos.manejador` convierte en código 2.

# %%
def _elegir_filas(n_train: int, n_test: int, sh: dict) -> tuple[np.ndarray, np.ndarray]:
    """Índices del fondo (train) y de los explicados (test), sin reemplazo, con make_rng(base_seed, 0)."""
    if sh["n_fondo"] > n_train or sh["n_explicar"] > n_test:  # sin reemplazo no se pueden pedir más filas
        raise ConfigInvalida(f"shap pide {sh['n_fondo']} filas de train y {sh['n_explicar']} de test; hay "
                             f"{n_train} y {n_test}")  # config
    rng = make_rng(sh["base_seed"], 0)  # generador de la corrida base
    i_fondo = rng.choice(n_train, size=sh["n_fondo"], replace=False)  # fondo de train
    i_expl = rng.choice(n_test, size=sh["n_explicar"], replace=False)  # explicados de test
    return i_fondo, i_expl  # índices


def _forward(red: dict):
    """Función X → logit con el forward float64 de la app (la que se explica y la de la permutación)."""
    return lambda X: forward_desde_json(red, X)  # misma función que la app


def _medir(tiempos: dict, etapa: str, funcion, *args):
    """Ejecuta ``funcion(*args)`` y guarda en ``tiempos[etapa]`` los segundos que tardó."""
    inicio = time.perf_counter()  # reloj de pared de alta resolución
    resultado = funcion(*args)  # la etapa
    tiempos[etapa] = time.perf_counter() - inicio  # segundos
    return resultado  # lo que devuelve la etapa


def _crudos_explicados(preparado: dict, i_expl, Z_e) -> pd.DataFrame:
    """Crudos de los explicados: numéricas crudas y categóricas con el código limpio del índice de Z."""
    crudos = preparado["crudos_test"].iloc[i_expl].reset_index(drop=True).copy()  # filas de los explicados
    for j, v in enumerate(preparado["pre_json"]["variables"]):  # orden de los jugadores
        if v["tipo"] != _TIPO_NUM:  # categórica: código limpio (incluye DESCONOCIDO y OTRA)
            crudos[v["nombre"]] = [v["categorias"][int(k)] for k in Z_e[:, j]]  # índice → código
    return crudos  # DataFrame de los explicados


def _shap_y_estabilidad(preparado: dict, sh: dict, tiempos: dict) -> dict:
    """Paso 1: SHAP del MLP (semilla base) y corrida de estabilidad (semilla base + 1); exige la eficiencia."""
    grupos, sp = preparado["grupos"], preparado["splits"]  # piezas
    n_train = len(sp["train"]["X"])  # filas de train
    i_fondo, i_expl = _elegir_filas(n_train, len(sp["test"]["X"]), sh)  # fondo y explicados
    fondo = a_variables(sp["test"]["X"][i_fondo % len(sp["test"]["X"])], grupos)  # MUTANTE
    expl = a_variables(sp["test"]["X"][i_expl], grupos)  # explicados de test en jugadores
    f_mlp = funcion_logit(_forward(preparado["red"]), grupos, preparado["d"])  # forward float64 en jugadores
    phi, phi0 = _medir(tiempos, "shap_base", valores_shap, f_mlp, fondo, expl, sh["max_evals"], sh["base_seed"])
    logit = f_mlp(expl)  # f(x) de los explicados
    ne, semilla_b = sh["n_estabilidad"], sh["base_seed"] + 1  # estabilidad
    i_fondo_b = make_rng(semilla_b, 0).choice(n_train, size=sh["n_fondo"], replace=False)  # otro fondo de train
    fondo_b = a_variables(sp["train"]["X"][i_fondo_b], grupos)  # fondo de la semilla base + 1
    phi_b, phi0_b = _medir(tiempos, "estabilidad", valores_shap, f_mlp, fondo_b, expl[:ne], sh["max_evals"],
                           semilla_b)  # segunda corrida sobre los primeros n_estabilidad
    eficiencia = max(error_eficiencia(phi, phi0, logit), error_eficiencia(phi_b, phi0_b, logit[:ne]))  # peor
    if not eficiencia <= TOL_EFICIENCIA:  # también atrapa NaN
        raise ValueError(f"Eficiencia rota: {eficiencia:.3e} > {TOL_EFICIENCIA:.0e}")  # aceptación de C4
    return {"i_expl": i_expl, "fondo": fondo, "expl": expl, "phi": phi, "phi0": phi0, "logit": logit,
            "phi_b": phi_b, "semilla_b": semilla_b, "eficiencia": eficiencia}  # resultado de la etapa


def _verificar_logistica(preparado: dict, fondo, expl, max_evals: int, semilla: int) -> dict:
    """Paso 2: SHAP por permutación de la logística frente a la forma cerrada; ValueError si difieren."""
    grupos, d, logistica = preparado["grupos"], preparado["d"], preparado["logistica"]  # piezas
    f_log = funcion_logit(logistica.predict_logit, grupos, d)  # logit de la logística en jugadores
    phi, phi0 = valores_shap(f_log, fondo, expl, max_evals, semilla)  # mismo código que el MLP
    coef, _ = logistica.coeficientes()  # β (d,)
    error = max(float(np.max(np.abs(phi - shap_cerrado_lineal(expl, fondo, coef, grupos)))),  # φ
                abs(phi0 - float(np.mean(f_log(fondo)))))  # φ₀ = media del logit del fondo
    if not error <= TOL_FORMA_CERRADA:  # también atrapa NaN
        raise ValueError(f"SHAP de la logística difiere de la forma cerrada: {error:.3e} > {TOL_FORMA_CERRADA:.0e}")
    return {"error_max": error, "tolerancia": TOL_FORMA_CERRADA, "n": int(expl.shape[0])}  # verificación


def _tablas_por_variable(preparado: dict, corrida: dict, P: np.ndarray, crudos_e: pd.DataFrame, dist: dict) -> list:
    """Entradas de ``puntos_por_variable`` en el orden de features.yaml (bordes y fracciones solo de train)."""
    grupos, entradas = preparado["grupos"], []  # columnas por variable y resultado
    for j, (f, v) in enumerate(zip(preparado["features"], preparado["pre_json"]["variables"], strict=True)):  # orden
        if f["tipo"] == _TIPO_NUM:  # tramos
            entradas.append(_entrada_numerica(f, preparado["crudos_train"][f["nombre"]], crudos_e[f["nombre"]],
                                              P[:, j], dist))  # tabla numérica
        else:  # categorías
            fr = {c: preparado["fracciones_train"][k] for c, k in zip(v["categorias"], grupos[j][1], strict=True)}
            entradas.append(_entrada_categorica(f, v, corrida["expl"][:, j], P[:, j], fr, dist))  # tabla categórica
    return entradas  # una entrada por variable


def _scorecard_json(preparado: dict, corrida: dict, params: dict, dist: dict, crudos_e: pd.DataFrame) -> dict:
    """Paso 3: puntos, aditividad, tablas por variable y distribución del puntaje (scorecard_mlp.json)."""
    sp = preparado["splits"]  # matrices y predicciones
    P = -params["factor"] * corrida["phi"]  # puntos de cada explicado y variable
    pb = float(score_desde_logit(corrida["phi0"], params))  # offset − factor · φ₀
    ad = error_aditividad(pb, P, np.asarray(score_desde_logit(corrida["logit"], params), dtype=np.float64))
    if not ad <= TOL_ADITIVIDAD:  # también atrapa NaN
        raise ValueError(f"Aditividad rota: error {ad:.3e} > {TOL_ADITIVIDAD:.0e}")  # aceptación de C4
    scores = {s: np.asarray(score_desde_logit(sp[s]["logit"], params), dtype=np.float64) for s in SPLITS}  # C1
    cortes = cortes_bandas(scores["train"], dist["n_bandas"])  # SOLO train
    return {
        **params,  # base, odds_base, pdo, factor, offset (C1)
        "modelo": MODELO_MLP,  # mlp
        "puntos_base": pb,  # offset − factor · φ₀
        "aditividad": {"error_max": ad, "tolerancia": TOL_ADITIVIDAD},  # pb + Σ puntos = score
        "puntos_por_variable": _tablas_por_variable(preparado, corrida, P, crudos_e, dist),  # tablas
        "distribucion_scores": {
            "cortes_bandas": cortes, "n_bandas": dist["n_bandas"], "ancho_bin": dist["ancho_bin"],  # comunes
            **{s: distribucion_split(scores[s], sp[s]["prob"], sp[s]["y"], cortes, dist["ancho_bin"])
               for s in SPLITS},  # train, val, test
        },
        "evidencia": dict(preparado["evidencia"]),  # núcleo §6
    }


def _permutacion_dos_modelos(preparado: dict, sh: dict, tiempos: dict) -> dict:
    """Paso 4: importancia por permutación en test de la logística y del MLP, con las mismas permutaciones."""
    grupos, d, test = preparado["grupos"], preparado["d"], preparado["splits"]["test"]  # piezas
    Z_test = a_variables(test["X"], grupos)  # test completo en jugadores
    f_mlp = _forward(preparado["red"])  # forward float64
    modelos = {"logistica": preparado["logistica"].predict_proba,  # PD de la logística
               "mlp": lambda X: expit(f_mlp(X))}  # PD del MLP = σ(forward float64)
    salida = {"k": sh["k_permutacion"], "semilla": sh["base_seed"], "conjunto": CONJUNTO_EXPLICADOS}  # parámetros
    for nombre, prob_fn in modelos.items():  # logística y MLP
        salida[nombre] = _medir(tiempos, f"permutacion_{nombre}", importancia_permutacion, prob_fn, Z_test,
                                test["y"], grupos, d, sh["k_permutacion"], sh["base_seed"])  # una corrida
    return salida  # bloque permutacion


def _shap_json(preparado: dict, corrida: dict, sh: dict, verificacion: dict, permutacion: dict,
               crudos_e: pd.DataFrame, tiempos: dict) -> dict:
    """Paso 5: ensambla shap_mlp.json."""
    nombres = [f["nombre"] for f in preparado["features"]]  # jugadores
    ne, logit, phi = sh["n_estabilidad"], corrida["logit"], corrida["phi"]  # piezas
    l32 = preparado["mlppd"].predict_logit(preparado["splits"]["test"]["X"][corrida["i_expl"]])  # torch float32
    return {
        "modelo": MODELO_MLP,  # mlp
        "metodo": {
            "explainer": EXPLAINER, "masker": MASKER, "salida": SALIDA_EXPLICADA,  # método
            "version_shap": shap.__version__, "jugadores": nombres,  # versión y jugadores
            "fondo": CONJUNTO_FONDO, "explicados": CONJUNTO_EXPLICADOS,  # de dónde salen
            "n_fondo": sh["n_fondo"], "n_explicar": sh["n_explicar"], "n_estabilidad": ne,  # tamaños
            "max_evals": sh["max_evals"], "n_permutaciones": sh["max_evals"] // MIN_MAX_EVALS,  # por fila (15 varían)
            "semillas": [sh["base_seed"], corrida["semilla_b"]],  # corrida base y de estabilidad
        },
        "phi0": corrida["phi0"],  # E_fondo[f]
        "eficiencia": {"error_max": corrida["eficiencia"], "tolerancia": TOL_EFICIENCIA},  # aceptación de C4
        "verificacion_logistica": verificacion,  # forma cerrada
        "diferencia_mlppd": {"max_abs_pd": float(np.max(np.abs(expit(logit) - expit(l32)))),  # PD
                             "max_abs_logit": float(np.max(np.abs(logit - l32)))},  # logit
        "importancia_global": importancia_global(phi, nombres),  # media |φ|
        "direccion": direccion(phi, crudos_e, preparado["features"]),  # signo del efecto
        "estabilidad": estabilidad(phi[:ne], corrida["phi_b"], nombres),  # base frente a base + 1
        "permutacion": permutacion,  # contraste
        "explicados": {"ids": [str(i) for i in preparado["ids_test"][corrida["i_expl"]]],  # créditos de test
                       "phi": phi.tolist(), "logit": logit.tolist()},  # listas: el dict es el JSON
        "evidencia": dict(preparado["evidencia"]),  # núcleo §6
        "tiempos_s": {k: float(v) for k, v in tiempos.items()},  # segundos por etapa
    }


def construir_scorecard_mlp(cfg: dict, preparado: dict | None = None) -> tuple[dict, dict]:
    """Construye ``scorecard_mlp.json`` y ``shap_mlp.json`` (pasos de la celda de arriba).

    Parámetros:
        cfg: config resuelta (``configs/scorecard/mlp.yaml`` con ``--set``).
        preparado: salida de ``preparar_mlp`` con las mismas rutas; None → se prepara aquí.
    Retorna:
        tuple[dict, dict]: (scorecard_mlp, shap_mlp).
    """
    cfg = validar_cfg_puntos(cfg)  # copia validada (incluye el bloque shap)
    if cfg["modelo"] != MODELO_MLP:  # la de la logística es puntos.construir_scorecard
        raise ValueError(f"construir_scorecard_mlp necesita modelo {MODELO_MLP!r}; se recibió {cfg['modelo']!r}")
    if preparado is None:  # sin preparación compartida
        preparado = preparar_mlp(cfg)  # carga y transforma
    if preparado["rutas"] != cfg["rutas"]:  # un preparado de otros archivos daría resultados incoherentes
        raise ValueError(f"El preparado se hizo con las rutas {preparado['rutas']} y la config pide {cfg['rutas']}")
    sh = cfg["shap"]  # bloque shap validado
    tiempos = {"preparacion": preparado["tiempo_s"]}  # segundos por etapa
    corrida = _shap_y_estabilidad(preparado, sh, tiempos)  # paso 1
    verificacion = _medir(tiempos, "verificacion_logistica", _verificar_logistica, preparado, corrida["fondo"],
                          corrida["expl"][:sh["n_estabilidad"]], sh["max_evals"], sh["base_seed"])  # paso 2
    crudos_e = _crudos_explicados(preparado, corrida["i_expl"], corrida["expl"])  # crudos de los explicados
    sc_mlp = _scorecard_json(preparado, corrida, parametros_scorecard(cfg["scorecard"]), cfg["distribucion"],
                             crudos_e)  # paso 3
    permutacion = _permutacion_dos_modelos(preparado, sh, tiempos)  # paso 4
    return sc_mlp, _shap_json(preparado, corrida, sh, verificacion, permutacion, crudos_e, tiempos)  # paso 5


# %% [markdown]
# ## Manejador de `run.py scorecard` con `modelo: mlp`
#
# `puntos.manejador` deriva aquí cuando la config tiene `modelo: mlp`. Escribe los dos JSON en `salida` e imprime un
# resumen: puntos base, errores de aditividad y de eficiencia, error de la verificación con la logística, las 5
# variables con mayor rango de puntos (con su etiqueta), las 5 más importantes por SHAP con su dirección, las 5 de mayor
# caída de AUC por permutación en cada modelo, el $\rho$ de estabilidad, la tasa de malos por banda de test, el tiempo
# de cada etapa y las rutas escritas. El tiempo total lo imprime `run.py`.

# %%
def _texto(valor, formato: str = ".4f") -> str:
    """Número para el resumen: None → raya."""
    return "—" if valor is None else format(valor, formato)  # sin valor


def _texto_direccion(dir_j: dict) -> str:
    """Dirección legible: ρ de una numérica; categoría de mayor y menor φ medio de una categórica."""
    if dir_j["tipo"] == "num":  # numérica
        return f"ρ(valor, φ) = {_texto(dir_j['spearman'], '+.2f')}"  # + : a mayor valor, más riesgo
    medios = dir_j["phi_medio"]  # categoría → φ medio
    mas, menos = max(medios, key=medios.get), min(medios, key=medios.get)  # extremos
    return f"más riesgo: {mas} ({medios[mas]:+.3f}); menos: {menos} ({medios[menos]:+.3f})"  # categórica


def _imprimir_resumen(sc: dict, sh: dict, rutas: list[Path]) -> None:
    """Resumen legible de la scorecard del MLP y de SHAP en la salida estándar."""
    print(f"Puntos base: {sc['puntos_base']:.4f} (φ₀ = {sh['phi0']:.6f}, factor {sc['factor']:.4f})")  # base
    print(f"Error de aditividad (pb + Σ puntos − score): {sc['aditividad']['error_max']:.3e} "
          f"(tolerancia {sc['aditividad']['tolerancia']:.0e})")  # aceptación de C4
    print(f"Error de eficiencia (φ₀ + Σφ − logit): {sh['eficiencia']['error_max']:.3e} "
          f"(tolerancia {sh['eficiencia']['tolerancia']:.0e})")  # aceptación de C4
    vl = sh["verificacion_logistica"]  # forma cerrada
    print(f"Verificación con la logística (forma cerrada, n = {vl['n']}): {vl['error_max']:.3e} "
          f"(tolerancia {vl['tolerancia']:.0e})")  # control del método
    print(f"Diferencia forward float64 − MLPPD: |ΔPD| ≤ {sh['diferencia_mlppd']['max_abs_pd']:.2e}")  # reportada
    con_rango = [e for e in sc["puntos_por_variable"] if e["rango_puntos"] is not None]  # rango definido
    print(f"Variables con mayor rango de puntos (top {_N_TOP}):")  # cabecera
    for e in sorted(con_rango, key=lambda e: e["rango_puntos"], reverse=True)[:_N_TOP]:  # una línea por variable
        print(f"  {e['nombre']:<26} {e['etiqueta_es']:<40} {e['rango_puntos']:8.2f}")  # nombre, etiqueta y rango
    print(f"Importancia global SHAP (top {_N_TOP}):")  # cabecera
    for e in sh["importancia_global"][:_N_TOP]:  # una línea por variable
        print(f"  {e['puesto']}. {e['nombre']:<26} media|φ| = {e['media_abs_phi']:.4f}  "
              f"{_texto_direccion(sh['direccion'][e['nombre']])}")  # nombre, importancia y dirección
    for modelo in ("logistica", "mlp"):  # contraste de los dos modelos
        res = sh["permutacion"][modelo]  # resultado del modelo
        top = sorted(res["variables"], key=lambda v: v["caida_auc_media"], reverse=True)[:_N_TOP]  # mayor caída
        print(f"Permutación en test, {modelo} (AUC base {res['auc_base']:.4f}; top {_N_TOP} por caída de AUC):")
        for v in top:  # una línea por variable
            print(f"  {v['nombre']:<26} ΔAUC = {v['caida_auc_media']:.4f} ± {v['caida_auc_desv']:.4f}  "
                  f"Δlog-loss = {v['aumento_logloss_media']:.4f}")  # caída y aumento
    print(f"Estabilidad (semillas {sh['metodo']['semillas']}, n = {sh['metodo']['n_estabilidad']}): ρ de las "
          f"importancias = {_texto(sh['estabilidad']['spearman_importancia'])}")  # estabilidad
    print("Bandas de test (banda 0 = menor puntaje = más riesgo):")  # cabecera
    for b in sc["distribucion_scores"]["test"]["bandas"]:  # una línea por banda
        desde = "-∞" if b["desde"] is None else f"{b['desde']:.1f}"  # primera banda sin límite inferior
        hasta = "+∞" if b["hasta"] is None else f"{b['hasta']:.1f}"  # última banda sin límite superior
        print(f"  banda {b['banda']}: [{desde}, {hasta})  n = {b['n']:>6}  "
              f"tasa_malos = {_texto(b['tasa_malos'])}")  # rango, n y tasa de malos
    print("Tiempos por etapa (s): " + ", ".join(f"{k} {v:.1f}" for k, v in sh["tiempos_s"].items()))  # etapas
    for ruta in rutas:  # archivos escritos
        print(f"Escrito: {ruta.as_posix()}")  # dónde quedó


def manejador_mlp(cfg: dict) -> int:
    """``run.py scorecard`` con ``modelo: mlp``: escribe ``scorecard_mlp.json`` y ``shap_mlp.json``; devuelve 0.

    Parámetros:
        cfg: config resuelta.
    Retorna:
        int: 0 si todo salió bien.
    """
    sc, sh = construir_scorecard_mlp(cfg)  # valida la config y construye
    carpeta = Path(cfg["salida"])  # carpeta de salida
    rutas = [save_json(carpeta / ARCHIVO_SCORECARD_MLP, sc),  # UTF-8, sangría 2, sin NaN, atómico
             save_json(carpeta / ARCHIVO_SHAP, sh)]  # SHAP, permutación y estabilidad
    _imprimir_resumen(sc, sh, rutas)  # resumen
    return 0  # éxito
