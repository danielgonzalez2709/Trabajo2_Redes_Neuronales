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
# # Métricas de la probabilidad de incumplimiento (Parte B §3.4)
#
# Este módulo reúne las métricas con que se evalúa un modelo de PD:
#
# - **Discriminación:** curva ROC, AUC, índice de Gini y estadístico KS (con su curva).
# - **Calibración:** puntaje de Brier y curva de calibración con bins por cuantiles.
#
# **Convención:** $y_i \in \{0, 1\}$ es el desenlace real del crédito $i$ ($y_i = 1$ **malo**, incumplió;
# $y_i = 0$ **bueno**) y $p_i \in [0, 1]$ es la PD predicha. Un puntaje más alto significa más riesgo, así que
# "clasificar como malo" es $p_i \ge t$ para un umbral $t$. Notación: $n$ créditos, $P = \sum_i y_i$ malos y
# $N = n - P$ buenos.
#
# **Quién lo consume:** la Parte B (evaluación de la logística y el MLP, figuras ROC/KS/calibración), la Parte C
# (scorecard), la Parte D (si evalúa clasificadores) y la Parte E (métricas del JSON de la app).
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# (`from src.common.metricas import auc`) sin efectos al importar.

# %%
"""Métricas de discriminación y calibración de la PD (Parte B §3.4).

Convención: ``y`` = 1 malo (incumplió), 0 bueno; ``p`` = PD predicha. Las métricas devuelven ``float`` y las
curvas devuelven un ``NamedTuple`` de arreglos listos para graficar.
"""

from __future__ import annotations  # anotaciones como texto: permite usar los alias de tipo antes de definirlos

from typing import Callable, NamedTuple  # Callable: tipo de los colaboradores inyectados; NamedTuple: curvas

import numpy as np  # todo el cálculo es vectorizado con numpy (sin bucles por crédito)

__all__ = [  # interfaz pública acordada (núcleo de la tarea B1); lo demás es interno
    "CurvaROC",
    "CurvaKS",
    "CurvaCalibracion",
    "FuncionConteos",
    "FuncionTasa",
    "FuncionCurvaROC",
    "FuncionIntegrador",
    "FuncionMetrica",
    "FuncionCurvaKS",
    "FuncionBinning",
    "validar_entradas",
    "conteos_acumulados",
    "tasa_verdaderos_positivos",
    "tasa_falsos_positivos",
    "curva_roc",
    "area_trapecio",
    "auc",
    "gini",
    "curva_ks",
    "ks",
    "brier",
    "bordes_cuantiles",
    "curva_calibracion",
]


# %% [markdown]
# ## Tipos de las curvas
#
# Cada curva es una tupla con nombre de arreglos de numpy, para poder graficarla directamente
# (`plt.plot(curva.fpr, curva.tpr)`):
#
# - `CurvaROC(fpr, tpr, umbrales)`: puntos $(\mathrm{FPR}(t), \mathrm{TPR}(t))$ con su umbral $t$.
# - `CurvaKS(umbrales, f_malos, f_buenos)`: las acumuladas $F_1(t)$ y $F_0(t)$ en cada umbral.
# - `CurvaCalibracion(pd_media, tasa_observada, n_por_bin)`: PD media vs. tasa observada de malos por bin.

# %%
class CurvaROC(NamedTuple):
    """Curva ROC: tasa de falsos positivos, tasa de verdaderos positivos y umbral de cada punto."""

    fpr: np.ndarray  # FPR(t) = FP(t) / N, de 0 a 1
    tpr: np.ndarray  # TPR(t) = VP(t) / P, de 0 a 1
    umbrales: np.ndarray  # t de cada punto, descendente; el primero es +inf (nadie clasificado como malo)


class CurvaKS(NamedTuple):
    """Acumuladas de la PD en malos y en buenos, evaluadas en cada valor único de la PD."""

    umbrales: np.ndarray  # valores únicos de p en orden ascendente
    f_malos: np.ndarray  # F₁(t): fracción de malos con p <= t
    f_buenos: np.ndarray  # F₀(t): fracción de buenos con p <= t


class CurvaCalibracion(NamedTuple):
    """Curva de calibración: por cada bin no vacío, PD media, tasa observada de malos y tamaño."""

    pd_media: np.ndarray  # media de p dentro del bin (eje x de la gráfica)
    tasa_observada: np.ndarray  # media de y dentro del bin = fracción de malos (eje y)
    n_por_bin: np.ndarray  # número de créditos del bin


# %% [markdown]
# ## Tipos de los colaboradores (inversión de dependencias)
#
# Las funciones compuestas no llaman directamente a sus piezas: las reciben como parámetros con un valor por
# defecto. Estos alias declaran la forma que debe tener cada pieza, de modo que la abstracción queda explícita
# y en las pruebas se puede inyectar una versión falsa.

# %%
FuncionConteos = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
"""(y, p) -> (umbrales descendentes, vp acumulados, fp acumulados)."""

FuncionTasa = Callable[[np.ndarray, int], np.ndarray]
"""(conteos acumulados, total de la clase) -> tasa (TPR o FPR)."""

FuncionCurvaROC = Callable[[np.ndarray, np.ndarray], CurvaROC]
"""(y, p) -> CurvaROC."""

FuncionIntegrador = Callable[[np.ndarray, np.ndarray], float]
"""(x, y) -> área bajo la poligonal."""

FuncionMetrica = Callable[[np.ndarray, np.ndarray], float]
"""(y, p) -> métrica escalar (por ejemplo, la AUC)."""

FuncionCurvaKS = Callable[[np.ndarray, np.ndarray], CurvaKS]
"""(y, p) -> CurvaKS."""

FuncionBinning = Callable[[np.ndarray, int], np.ndarray]
"""(p, n_bins) -> n_bins + 1 bordes crecientes."""


# %% [markdown]
# ## Validación de entradas
#
# Todas las funciones públicas que reciben $(y, p)$ pasan primero por aquí. Se exige:
#
# $$
# |y| = |p| = n > 0, \qquad y_i \in \{0, 1\}, \qquad p_i \in \mathbb{R} \text{ (sin NaN)}, \qquad 0 < P < n .
# $$
#
# La última condición (ambas clases presentes) es necesaria porque TPR divide entre $P$, FPR entre $N$ y
# $F_1$, $F_0$ también; con una sola clase la AUC y el KS no están definidos. El rango $p_i \in [0, 1]$ solo se
# exige donde $p$ se usa como probabilidad (Brier y calibración); la AUC y el KS dependen solo del orden de $p$
# y aceptan cualquier puntaje real.

# %%
def validar_entradas(y, p) -> tuple[np.ndarray, np.ndarray]:
    """Convierte y valida el par (y real, predicción).

    Parámetros
    ----------
    y : array-like de forma (n,)
        Desenlace real: 1 malo, 0 bueno.
    p : array-like de forma (n,)
        Predicción (PD o cualquier puntaje con más riesgo = valor más alto).

    Retorna
    -------
    (y, p) : tuple[np.ndarray, np.ndarray]
        ``y`` como enteros 0/1 y ``p`` como flotantes, ambos unidimensionales.

    Lanza
    -----
    ValueError
        Si las longitudes difieren, están vacíos, no son unidimensionales, hay NaN, ``y`` tiene valores fuera
        de {0, 1} o solo hay una clase.
    """
    y_arr = np.asarray(y, dtype=float)  # flotante para poder detectar NaN antes de pasar a enteros
    p_arr = np.asarray(p, dtype=float)  # la predicción siempre se trabaja como flotante
    if y_arr.ndim != 1 or p_arr.ndim != 1:  # las métricas se definen sobre vectores de créditos
        raise ValueError("y y p deben ser unidimensionales.")
    if y_arr.shape[0] != p_arr.shape[0]:  # cada predicción debe corresponder a un desenlace
        raise ValueError(f"Longitudes distintas: len(y)={y_arr.shape[0]}, len(p)={p_arr.shape[0]}.")
    if y_arr.shape[0] == 0:  # sin créditos no hay métrica que calcular
        raise ValueError("y y p están vacíos.")
    if np.isnan(y_arr).any() or np.isnan(p_arr).any():  # un NaN contaminaría sumas acumuladas y medias
        raise ValueError("y o p contienen NaN.")
    if not np.isin(y_arr, (0.0, 1.0)).all():  # convención binaria: 1 malo, 0 bueno
        raise ValueError("y solo puede tomar los valores 0 y 1.")
    n_malos = int(y_arr.sum())  # P = número de malos
    if n_malos == 0 or n_malos == y_arr.shape[0]:  # 0 < P < n: hacen falta las dos clases
        raise ValueError("y debe contener las dos clases (0 y 1).")
    return y_arr.astype(np.int64), p_arr  # y como enteros para que las sumas sean conteos exactos


def _validar_probabilidades(p: np.ndarray) -> None:
    """Exige p ∈ [0, 1] (Brier y calibración interpretan p como probabilidad). Uso interno."""
    if (p < 0.0).any() or (p > 1.0).any():  # una probabilidad fuera de [0, 1] no tiene sentido aquí
        raise ValueError("p debe estar en [0, 1].")


# %% [markdown]
# ## Conteos acumulados por umbral
#
# Para cada umbral $t$ (cada valor único de $p$, de mayor a menor) se clasifica como malo a todo crédito con
# $p_i \ge t$ y se cuentan
#
# $$
# \mathrm{VP}(t) = \sum_{i} \mathbb{1}[p_i \ge t]\, y_i, \qquad
# \mathrm{FP}(t) = \sum_{i} \mathbb{1}[p_i \ge t]\, (1 - y_i).
# $$
#
# **Cómo se lee:** al bajar el umbral se "aprueba como malo" a más gente; $\mathrm{VP}$ cuenta los malos
# atrapados y $\mathrm{FP}$ los buenos señalados por error. Los créditos empatados en $p$ entran todos a la vez,
# así que **un empate forma un solo punto** (la curva sube en diagonal en ese tramo).
#
# Cálculo: se ordena $p$ de forma descendente, se acumulan $y$ y $1 - y$ y se toma el valor acumulado en la
# **última** posición de cada grupo de empates.

# %%
def conteos_acumulados(y, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Verdaderos y falsos positivos acumulados para cada umbral único de p (descendente).

    Parámetros
    ----------
    y : array-like de forma (n,)
        Desenlace real (1 malo, 0 bueno).
    p : array-like de forma (n,)
        Predicción.

    Retorna
    -------
    (umbrales, vp, fp) : tuple[np.ndarray, np.ndarray, np.ndarray]
        ``umbrales``: valores únicos de p en orden descendente; ``vp[i]`` y ``fp[i]``: malos y buenos con
        ``p >= umbrales[i]``.
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    orden = np.argsort(-p, kind="mergesort")  # índices de p de mayor a menor (estable, determinista)
    p_ord = p[orden]  # PD ordenada descendentemente
    y_ord = y[orden]  # desenlaces en el mismo orden
    vp_acum = np.cumsum(y_ord)  # VP acumulado posición a posición (malos vistos hasta aquí)
    fp_acum = np.cumsum(1 - y_ord)  # FP acumulado posición a posición (buenos vistos hasta aquí)
    # Última posición de cada grupo de empates: donde el siguiente valor cambia, más la última posición.
    fin_grupo = np.r_[np.flatnonzero(np.diff(p_ord) != 0), p_ord.shape[0] - 1]
    return p_ord[fin_grupo], vp_acum[fin_grupo], fp_acum[fin_grupo]  # un punto por umbral único


# %% [markdown]
# ## Tasas de verdaderos y falsos positivos
#
# $$
# \mathrm{TPR}(t) = \frac{\mathrm{VP}(t)}{P}, \qquad \mathrm{FPR}(t) = \frac{\mathrm{FP}(t)}{N}.
# $$
#
# **Cómo se lee:** $\mathrm{TPR}(t)$ (sensibilidad) es la fracción de los malos que el umbral $t$ detecta;
# $\mathrm{FPR}(t)$ (1 − especificidad) es la fracción de los buenos que se señalan por error. Ambas van de 0
# (umbral $+\infty$) a 1 (umbral $\le \min p$).

# %%
def tasa_verdaderos_positivos(vp: np.ndarray, n_pos: int) -> np.ndarray:
    """TPR = vp / n_pos.

    Parámetros
    ----------
    vp : np.ndarray
        Verdaderos positivos acumulados.
    n_pos : int
        Número de malos P.

    Retorna
    -------
    np.ndarray
        Tasa de verdaderos positivos en cada umbral.
    """
    return np.asarray(vp, dtype=float) / n_pos  # fracción de malos detectados


def tasa_falsos_positivos(fp: np.ndarray, n_neg: int) -> np.ndarray:
    """FPR = fp / n_neg.

    Parámetros
    ----------
    fp : np.ndarray
        Falsos positivos acumulados.
    n_neg : int
        Número de buenos N.

    Retorna
    -------
    np.ndarray
        Tasa de falsos positivos en cada umbral.
    """
    return np.asarray(fp, dtype=float) / n_neg  # fracción de buenos señalados por error


# %% [markdown]
# ## Curva ROC
#
# Es el conjunto de puntos
#
# $$
# \bigl(\mathrm{FPR}(t),\ \mathrm{TPR}(t)\bigr), \qquad t \in \{+\infty\} \cup \{\text{valores únicos de } p\},
# $$
#
# recorridos de umbral alto a bajo. Con $t = +\infty$ nadie es clasificado como malo y el punto es $(0, 0)$;
# con el menor umbral todos lo son y el punto es $(1, 1)$. Es la misma convención que
# `sklearn.metrics.roc_curve(drop_intermediate=False)`.

# %%
def curva_roc(
    y,
    p,
    conteos: FuncionConteos = conteos_acumulados,
    tpr: FuncionTasa = tasa_verdaderos_positivos,
    fpr: FuncionTasa = tasa_falsos_positivos,
) -> CurvaROC:
    """Curva ROC de (0, 0) con umbral +inf a (1, 1).

    Parámetros
    ----------
    y, p : array-like de forma (n,)
        Desenlace real y predicción.
    conteos : FuncionConteos
        Calcula (umbrales, vp, fp); por defecto ``conteos_acumulados``.
    tpr, fpr : FuncionTasa
        Convierten los conteos en tasas; por defecto ``tasa_verdaderos_positivos`` y ``tasa_falsos_positivos``.

    Retorna
    -------
    CurvaROC
        ``fpr``, ``tpr`` y ``umbrales`` (descendentes, el primero +inf).
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    n_pos = int(y.sum())  # P: malos
    n_neg = int(y.shape[0] - n_pos)  # N: buenos
    umbrales, vp, fp = conteos(y, p)  # VP(t) y FP(t) por umbral único (colaborador inyectado)
    return CurvaROC(
        fpr=np.r_[0.0, fpr(fp, n_neg)],  # FPR(t), anteponiendo el punto t = +inf con FPR = 0
        tpr=np.r_[0.0, tpr(vp, n_pos)],  # TPR(t), anteponiendo el punto t = +inf con TPR = 0
        umbrales=np.r_[np.inf, umbrales],  # umbral +inf: nadie clasificado como malo
    )


# %% [markdown]
# ## Regla del trapecio
#
# Para una poligonal con vértices $(x_0, y_0), \dots, (x_m, y_m)$ y $x$ no decreciente,
#
# $$
# A = \sum_{k=1}^{m} (x_k - x_{k-1})\, \frac{y_k + y_{k-1}}{2}.
# $$
#
# **Cómo se lee:** cada tramo aporta el área del trapecio entre dos puntos consecutivos. Sobre la ROC es
# **exacta** (la ROC empírica es una poligonal), incluidos los tramos diagonales que generan los empates.

# %%
def area_trapecio(x: np.ndarray, y: np.ndarray) -> float:
    """Área bajo la poligonal (x, y) por la regla del trapecio.

    Parámetros
    ----------
    x, y : np.ndarray
        Coordenadas de los vértices, con ``x`` en orden no decreciente.

    Retorna
    -------
    float
        Área bajo la poligonal.
    """
    x = np.asarray(x, dtype=float)  # abscisas como flotantes
    y = np.asarray(y, dtype=float)  # ordenadas como flotantes
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))  # Σ ancho · (altura media) de cada tramo


# %% [markdown]
# ## AUC
#
# $$
# \mathrm{AUC} = \int_0^1 \mathrm{TPR}\; d(\mathrm{FPR}),
# $$
#
# calculada con la regla del trapecio sobre la curva ROC. Equivale a
#
# $$
# \mathrm{AUC} = \Pr(p_{\text{malo}} > p_{\text{bueno}}) + \tfrac{1}{2}\Pr(p_{\text{malo}} = p_{\text{bueno}})
# $$
#
# para un malo y un bueno tomados al azar (Hanley y McNeil, 1982) [VERIFICAR referencia].
#
# **Cómo se lee:** 0,5 es un modelo que ordena al azar, 1 es una separación perfecta y 0 un orden invertido.
# Solo depende del **orden** de $p$: no cambia ante transformaciones monótonas crecientes (por ejemplo, logit).

# %%
def auc(
    y,
    p,
    curva: FuncionCurvaROC = curva_roc,
    integrador: FuncionIntegrador = area_trapecio,
) -> float:
    """Área bajo la curva ROC.

    Parámetros
    ----------
    y, p : array-like de forma (n,)
        Desenlace real y predicción.
    curva : FuncionCurvaROC
        Construye la curva ROC; por defecto ``curva_roc``.
    integrador : FuncionIntegrador
        Integra TPR respecto a FPR; por defecto ``area_trapecio``.

    Retorna
    -------
    float
        AUC en [0, 1].
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    roc = curva(y, p)  # puntos (FPR, TPR) de la ROC (colaborador inyectado)
    return float(integrador(roc.fpr, roc.tpr))  # ∫ TPR d(FPR): x = FPR, y = TPR


# %% [markdown]
# ## Índice de Gini
#
# $$
# \mathrm{Gini} = 2\,\mathrm{AUC} - 1 .
# $$
#
# **Cómo se lee:** reescala la AUC a $[-1, 1]$: 0 es azar, 1 separación perfecta, −1 orden invertido. (Un error
# común es escribir $2\,\mathrm{AUC}$ sin restar 1, que daría 1 para un modelo al azar.)

# %%
def gini(y, p, auc_fn: FuncionMetrica = auc) -> float:
    """Índice de Gini = 2·AUC − 1.

    Parámetros
    ----------
    y, p : array-like de forma (n,)
        Desenlace real y predicción.
    auc_fn : FuncionMetrica
        Calcula la AUC; por defecto ``auc``.

    Retorna
    -------
    float
        Gini en [−1, 1].
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    return float(2.0 * auc_fn(y, p) - 1.0)  # 2·AUC − 1 (se resta 1: azar → 0)


# %% [markdown]
# ## Curva KS
#
# Las funciones de distribución empíricas de la PD en malos y en buenos son
#
# $$
# F_1(t) = \frac{1}{P} \sum_{i:\, y_i = 1} \mathbb{1}[p_i \le t], \qquad
# F_0(t) = \frac{1}{N} \sum_{i:\, y_i = 0} \mathbb{1}[p_i \le t],
# $$
#
# evaluadas en cada valor único de $p$ (ascendente); entre dos valores consecutivos ambas son constantes, así que
# esos puntos bastan para el máximo.
#
# **Cómo se lee:** si el modelo discrimina, los buenos se acumulan antes (PD bajas) y $F_0(t)$ va por encima de
# $F_1(t)$; la separación vertical entre ambas curvas es lo que mide el KS.

# %%
def curva_ks(y, p) -> CurvaKS:
    """Acumuladas F₁(t) (malos) y F₀(t) (buenos) en cada valor único de p.

    Parámetros
    ----------
    y, p : array-like de forma (n,)
        Desenlace real y predicción.

    Retorna
    -------
    CurvaKS
        ``umbrales`` ascendentes, ``f_malos`` = F₁(t) y ``f_buenos`` = F₀(t).
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    umbrales = np.unique(p)  # valores únicos de p, ya ordenados de menor a mayor
    p_malos = np.sort(p[y == 1])  # PD de los malos, ordenada para contar con búsqueda binaria
    p_buenos = np.sort(p[y == 0])  # PD de los buenos, ídem
    # searchsorted(..., side="right") da cuántos valores son <= t: el numerador de F₁(t) y F₀(t).
    f_malos = np.searchsorted(p_malos, umbrales, side="right") / p_malos.shape[0]  # F₁(t) = #{malos ≤ t}/P
    f_buenos = np.searchsorted(p_buenos, umbrales, side="right") / p_buenos.shape[0]  # F₀(t) = #{buenos ≤ t}/N
    return CurvaKS(umbrales=umbrales, f_malos=f_malos, f_buenos=f_buenos)


# %% [markdown]
# ## Estadístico KS
#
# $$
# \mathrm{KS} = \max_t \bigl| F_1(t) - F_0(t) \bigr| .
# $$
#
# **Cómo se lee:** la mayor distancia vertical entre las acumuladas de la PD en malos y en buenos; 0 es que no
# separa y 1 separación perfecta. Coincide con el estadístico de Kolmogórov-Smirnov de dos muestras
# (`scipy.stats.ks_2samp(p[y == 1], p[y == 0])`). Se calcula sobre las **distribuciones de la PD por clase**, no
# leyendo la curva ROC de forma ad hoc.

# %%
def ks(y, p, curva: FuncionCurvaKS = curva_ks) -> float:
    """Estadístico KS = máx_t |F₁(t) − F₀(t)|.

    Parámetros
    ----------
    y, p : array-like de forma (n,)
        Desenlace real y predicción.
    curva : FuncionCurvaKS
        Calcula las acumuladas; por defecto ``curva_ks``.

    Retorna
    -------
    float
        KS en [0, 1].
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    c = curva(y, p)  # F₁(t) y F₀(t) en cada umbral (colaborador inyectado)
    return float(np.max(np.abs(np.asarray(c.f_malos) - np.asarray(c.f_buenos))))  # máx_t |F₁(t) − F₀(t)|


# %% [markdown]
# ## Puntaje de Brier
#
# $$
# \mathrm{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2 .
# $$
#
# **Cómo se lee:** error cuadrático medio de la probabilidad; 0 es perfecto y menor es mejor. Premia a la vez la
# calibración y la discriminación (es una regla de puntuación propia). Nota: la definición original de Brier
# (1950) suma sobre las dos clases y en el caso binario da el doble; aquí se usa la versión binaria, con rango
# $[0, 1]$, la misma de `sklearn.metrics.brier_score_loss`.

# %%
def brier(y, p) -> float:
    """Puntaje de Brier binario = (1/n) Σ (pᵢ − yᵢ)².

    Parámetros
    ----------
    y : array-like de forma (n,)
        Desenlace real (1 malo, 0 bueno).
    p : array-like de forma (n,)
        PD predicha, en [0, 1].

    Retorna
    -------
    float
        Brier en [0, 1].

    Lanza
    -----
    ValueError
        Las de ``validar_entradas`` y, además, si algún p está fuera de [0, 1].
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    _validar_probabilidades(p)  # p se interpreta como probabilidad
    return float(np.mean((p - y) ** 2))  # (1/n) Σ (pᵢ − yᵢ)²


# %% [markdown]
# ## Bordes por cuantiles
#
# Para $B$ bins, los bordes son los cuantiles empíricos de $p$
#
# $$
# b_k = Q_p\!\left(\tfrac{k}{B}\right), \qquad k = 0, 1, \dots, B,
# $$
#
# con **interpolación lineal** entre estadísticos de orden: si $p_{(1)} \le \dots \le p_{(n)}$ y
# $h = (n - 1)\,q$, entonces $Q_p(q) = p_{(\lfloor h \rfloor + 1)} + (h - \lfloor h \rfloor)
# \bigl(p_{(\lfloor h \rfloor + 2)} - p_{(\lfloor h \rfloor + 1)}\bigr)$ (método por defecto de `np.percentile`,
# el que usa `sklearn.calibration.calibration_curve`). $b_0 = \min p$ y $b_B = \max p$.
#
# **Cómo se lee:** cada bin contiene aproximadamente $n / B$ créditos, así que todos los puntos de la curva de
# calibración se apoyan en el mismo volumen de datos.

# %%
def bordes_cuantiles(p, n_bins: int) -> np.ndarray:
    """Bordes de ``n_bins`` bins por cuantiles de p (interpolación lineal).

    Parámetros
    ----------
    p : array-like de forma (n,)
        Predicción.
    n_bins : int
        Número de bins B (>= 1).

    Retorna
    -------
    np.ndarray
        ``n_bins + 1`` bordes: los cuantiles 0, 1/B, …, 1 de p.
    """
    if int(n_bins) != n_bins or n_bins < 1:  # B debe ser un entero positivo
        raise ValueError("n_bins debe ser un entero >= 1.")
    niveles = np.linspace(0.0, 1.0, int(n_bins) + 1)  # q_k = k/B, k = 0..B
    return np.percentile(np.asarray(p, dtype=float), niveles * 100.0)  # Q_p(q_k), interpolación lineal


# %% [markdown]
# ## Curva de calibración
#
# Con bordes $b_0 \le \dots \le b_B$, el crédito $i$ va al bin
#
# $$
# k(i) = \#\{\, j \in \{1, \dots, B-1\} : b_j < p_i \,\},
# $$
#
# es decir, se compara solo con los bordes internos y **un dato igual a un borde va al bin inferior** (misma
# regla que `calibration_curve`, que usa `np.searchsorted(bordes[1:-1], p)` con el lado por defecto "left").
# Para cada bin $k$ con $n_k > 0$ créditos:
#
# $$
# \bar p_k = \frac{1}{n_k} \sum_{i:\,k(i)=k} p_i, \qquad \bar y_k = \frac{1}{n_k} \sum_{i:\,k(i)=k} y_i .
# $$
#
# Los bins vacíos (aparecen cuando hay empates y varios bordes coinciden) **se omiten**.
#
# **Cómo se lee:** se grafica $\bar y_k$ (tasa observada de malos) contra $\bar p_k$ (PD media); un modelo bien
# calibrado queda sobre la diagonal $\bar y_k = \bar p_k$.

# %%
def curva_calibracion(
    y,
    p,
    n_bins: int = 10,
    binning: FuncionBinning = bordes_cuantiles,
) -> CurvaCalibracion:
    """Curva de calibración por bins (por defecto, 10 bins por cuantiles; Parte B §3.4).

    Parámetros
    ----------
    y : array-like de forma (n,)
        Desenlace real (1 malo, 0 bueno).
    p : array-like de forma (n,)
        PD predicha, en [0, 1].
    n_bins : int
        Número de bins pedido al ``binning``.
    binning : FuncionBinning
        Calcula los bordes a partir de (p, n_bins); por defecto ``bordes_cuantiles``.

    Retorna
    -------
    CurvaCalibracion
        Por bin no vacío: ``pd_media``, ``tasa_observada`` y ``n_por_bin``.

    Lanza
    -----
    ValueError
        Las de ``validar_entradas`` y, además, si algún p está fuera de [0, 1].
    """
    y, p = validar_entradas(y, p)  # reutiliza la validación común
    _validar_probabilidades(p)  # p se interpreta como probabilidad
    bordes = np.asarray(binning(p, n_bins), dtype=float)  # b_0..b_B (colaborador inyectado)
    n_total_bins = bordes.shape[0] - 1  # B según los bordes recibidos
    bin_de = np.searchsorted(bordes[1:-1], p)  # k(i): lado "left" → un dato en un borde va al bin inferior
    suma_p = np.bincount(bin_de, weights=p, minlength=n_total_bins)  # Σ pᵢ por bin
    suma_y = np.bincount(bin_de, weights=y, minlength=n_total_bins)  # Σ yᵢ por bin (malos del bin)
    n_bin = np.bincount(bin_de, minlength=n_total_bins)  # n_k: créditos por bin
    no_vacio = n_bin > 0  # se omiten los bins vacíos (empates con bordes repetidos)
    return CurvaCalibracion(
        pd_media=suma_p[no_vacio] / n_bin[no_vacio],  # \bar p_k
        tasa_observada=suma_y[no_vacio] / n_bin[no_vacio],  # \bar y_k
        n_por_bin=n_bin[no_vacio],  # n_k
    )
