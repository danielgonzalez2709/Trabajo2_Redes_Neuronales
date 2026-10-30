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
# # De la PD a la decisión: umbral, matriz de confusión y costo (B11, B-pd §3.9)
#
# Lógica **pura** (sin E/S): convierte la PD en una decisión de aprobar o rechazar y mide sus consecuencias.
#
# **Convención:** positivo = malo ($y = 1$). Una política **rechaza** un subconjunto de créditos:
# VP = malos rechazados, FN = malos aprobados (el error caro), FP = buenos rechazados, VN = buenos aprobados.
#
# Dos formas de rechazar: **por umbral** $t$ (rechaza si $p \ge t$) y **por cuota** $r \in [0, 1]$ (rechaza los
# $k = \lceil r \cdot n \rceil$ créditos de mayor PD). El costo se mide en unidades de $G$ (lo que se pierde al
# rechazar a un bueno) con la razón $L/G$; sin dólares ni margen inventado (`int_rate` está excluida, núcleo §5.7).
#
# Las entradas $(y, p)$ se validan con `metricas.validar_entradas` (B1); las que solo reciben $p$, con la misma
# regla sin $y$ (`_validar_pd`).

# %%
"""De la PD a la decisión (B11): rechazos por cuota y por umbral, matriz de confusión, tasas, costo y cuotas."""

import math  # ceil, infinito e isfinite

import numpy as np  # máscaras, ordenamientos y sumas acumuladas

from src.common.metricas import brier, validar_entradas  # validación y Brier comunes (B1)

__all__ = [  # interfaz pública (B-pd §3.9)
    "DECIMALES_K", "k_de_cuota", "rechazos_por_cuota", "rechazos_por_umbral", "umbral_de_cuota",  # rechazos
    "matriz_confusion", "tasas", "costo", "umbral_teorico", "conteos_por_cuota", "cuota_optima",  # medidas
    "cuota_para_meta", "curva_sensibilidad", "brier_trivial", "validar_rejilla", "validar_real",  # metas, curva…
    "cuota_optima_conteos", "cuota_para_meta_conteos",  # cuotas desde conteos ya calculados (un solo orden)
]


# %% [markdown]
# ## Constantes
#
# `DECIMALES_K`: $r \cdot n$ se redondea a 9 decimales antes del techo (B-pd §3.9), para que $0{,}07 \cdot 100$
# ($= 7{,}000000000000001$ en coma flotante) dé 7 y no 8. `CLAVES_MATRIZ`: los cuatro conteos, en orden fijo.

# %%
DECIMALES_K = 9  # B-pd §3.9: k = ⌈round(r·n, 9)⌉
CLAVES_MATRIZ = ("vp", "fn", "fp", "vn")  # conteos de la matriz de confusión


# %% [markdown]
# ## Validación de un número real
#
# Real finito (bool y texto no cuentan como números) dentro de $[\text{mínimo}, \text{máximo}]$; `abierto_abajo`
# excluye el mínimo (p. ej. la razón $L/G > 0$). Devuelve el `float`.

# %%
def validar_real(valor, nombre: str, minimo: float, maximo: float, abierto_abajo: bool = False) -> float:
    """``valor`` como float finito en [minimo, maximo] (o (minimo, maximo]); si no, ValueError."""
    if isinstance(valor, bool) or not isinstance(valor, (int, float, np.integer, np.floating)):  # True o "0.5"
        raise ValueError(f"{nombre} debe ser un número real; se recibió {valor!r}")  # tipo
    x = float(valor)  # float nativo
    if not math.isfinite(x):  # NaN o ±inf
        raise ValueError(f"{nombre} debe ser finito; se recibió {valor!r}")  # no finito
    debajo = x <= minimo if abierto_abajo else x < minimo  # límite inferior abierto o cerrado
    if debajo or x > maximo:  # fuera del intervalo
        abre = "(" if abierto_abajo else "["  # notación del intervalo
        raise ValueError(f"{nombre} debe estar en {abre}{minimo}, {maximo}]; se recibió {valor!r}")  # rango
    return x  # válido


# %% [markdown]
# ## Validación de la PD sola
#
# Para las funciones que reciben solo $p$: vector 1-D no vacío de flotantes finitos en $[0, 1]$ (la misma regla de
# `metricas.validar_entradas` y de `metricas.brier`, sin $y$). Devuelve una copia en `float64`.

# %%
def _validar_pd(p) -> np.ndarray:
    """p como float64 1-D, no vacío, finito y en [0, 1]; si no, ValueError."""
    arr = np.array(p, dtype=float)  # copia: no se muta la entrada
    if arr.ndim != 1 or arr.shape[0] == 0:  # vector de créditos no vacío
        raise ValueError(f"p debe ser un vector 1-D no vacío; forma {arr.shape}")  # forma
    if not np.isfinite(arr).all():  # NaN o ±inf
        raise ValueError("p contiene valores no finitos (NaN o ±inf).")  # no finito
    if (arr < 0.0).any() or (arr > 1.0).any():  # una PD es una probabilidad
        raise ValueError("p debe estar en [0, 1].")  # rango
    return arr  # válida


# %% [markdown]
# ## Validación de $(y, p)$
#
# `metricas.validar_entradas` (largos iguales, no vacío, finito, $y \in \{0, 1\}$, las dos clases) más $p \in [0, 1]$.

# %%
def _validar_yp(y, p) -> tuple[np.ndarray, np.ndarray]:
    """(y int64, p float64) validados con metricas.validar_entradas y p ∈ [0, 1]."""
    y_arr, p_arr = validar_entradas(y, p)  # regla común de B1
    return y_arr, _validar_pd(p_arr)  # además, PD en [0, 1]


# %% [markdown]
# ## Número de rechazos de una cuota
#
# $$k = \left\lceil \mathrm{round}(r \cdot n,\ 9) \right\rceil, \qquad r \in [0, 1].$$

# %%
def k_de_cuota(cuota, n: int) -> int:
    """k = ⌈round(cuota·n, 9)⌉ con cuota ∈ [0, 1]."""
    r = validar_real(cuota, "cuota", 0.0, 1.0)  # cuota válida
    return int(math.ceil(round(r * n, DECIMALES_K)))  # techo tras redondear a 9 decimales


# %% [markdown]
# ## Orden de rechazo
#
# Índices de mayor a menor PD con orden **estable**: a igual PD, primero la fila que aparece antes
# (`argsort` estable de $-p$).

# %%
def _orden_descendente(p: np.ndarray) -> np.ndarray:
    """Índices de p de mayor a menor, estables ante empates."""
    return np.argsort(-p, kind="stable")  # estable: los empates conservan el orden de aparición


# %% [markdown]
# ## Rechazo por cuota
#
# Máscara booleana: `True` en los $k$ créditos de mayor PD (empates en el borde: orden estable).

# %%
def rechazos_por_cuota(p, cuota) -> np.ndarray:
    """Máscara de rechazo de los k = ⌈round(cuota·n, 9)⌉ créditos de mayor PD (orden estable)."""
    arr = _validar_pd(p)  # PD válida
    k = k_de_cuota(cuota, arr.shape[0])  # rechazos
    mascara = np.zeros(arr.shape[0], dtype=bool)  # nadie rechazado
    mascara[_orden_descendente(arr)[:k]] = True  # los k de mayor PD
    return mascara  # máscara booleana


# %% [markdown]
# ## Rechazo por umbral
#
# Máscara $p \ge t$ (el borde se rechaza). $t \in [0, 1]$ o $t = +\infty$ (no rechaza a nadie: es el umbral
# equivalente a una cuota con $k = 0$).

# %%
def _umbral_valido(t) -> float:
    """t ∈ [0, 1] o +inf; si no, ValueError."""
    if isinstance(t, (int, float, np.integer, np.floating)) and not isinstance(t, bool) and t == math.inf:  # +inf
        return math.inf  # no rechaza a nadie
    return validar_real(t, "umbral", 0.0, 1.0)  # umbral finito en [0, 1]


def rechazos_por_umbral(p, t) -> np.ndarray:
    """Máscara de rechazo p ≥ t."""
    arr = _validar_pd(p)  # PD válida
    return arr >= _umbral_valido(t)  # el borde se rechaza


# %% [markdown]
# ## Umbral equivalente a una cuota
#
# $t(r) = \min\{p_i : i \text{ rechazado por la cuota } r\}$, la menor PD rechazada; $+\infty$ si $k = 0$. Aplicado a
# los mismos datos reproduce la cuota salvo empates en el borde (con empates, $p \ge t$ rechaza de más).

# %%
def umbral_de_cuota(p, cuota) -> float:
    """Menor PD rechazada por la cuota (+inf si k = 0)."""
    arr = _validar_pd(p)  # PD válida
    mascara = rechazos_por_cuota(arr, cuota)  # rechazados
    return float(arr[mascara].min()) if mascara.any() else math.inf  # menor PD rechazada o +inf


# %% [markdown]
# ## Matriz de confusión
#
# $$VP = \sum_i \rho_i y_i,\quad FN = \sum_i (1-\rho_i) y_i,\quad FP = \sum_i \rho_i (1-y_i),\quad
# VN = \sum_i (1-\rho_i)(1-y_i),$$
#
# con $\rho_i = 1$ si el crédito $i$ se rechaza. $y$ se valida con `validar_entradas` (la máscara hace de $p$);
# la máscara debe ser booleana y del mismo largo. Conteos como `int` nativos (van al JSON).

# %%
def matriz_confusion(y, rechazo) -> dict:
    """{vp, fn, fp, vn} de la política ``rechazo`` (máscara booleana) frente a ``y``."""
    mascara = np.asarray(rechazo)  # máscara
    if mascara.dtype != bool:  # 0/1 enteros serían ambiguos
        raise ValueError(f"rechazo debe ser una máscara booleana; dtype {mascara.dtype}")  # tipo
    y_arr, _ = validar_entradas(y, mascara.astype(float))  # largos, clases y finitud (regla común)
    malo = y_arr == 1  # positivos
    return {  # en el orden de CLAVES_MATRIZ
        "vp": int(np.sum(mascara & malo)),  # malos rechazados
        "fn": int(np.sum(~mascara & malo)),  # malos aprobados (el error caro)
        "fp": int(np.sum(mascara & ~malo)),  # buenos rechazados
        "vn": int(np.sum(~mascara & ~malo)),  # buenos aprobados
    }


# %% [markdown]
# ## Tasas
#
# $$\mathrm{TPR} = \frac{VP}{VP+FN},\quad \mathrm{TNR} = \frac{VN}{VN+FP},\quad
# \text{rechazo} = \frac{VP+FP}{n},\quad \text{aprobación} = 1 - \text{rechazo}.$$
#
# Exige las cuatro claves, conteos enteros $\ge 0$ y las dos clases (si no, TPR o TNR no están definidas).

# %%
def _validar_matriz(m: dict) -> dict:
    """Copia de m con sus cuatro conteos enteros ≥ 0 y las dos clases; si no, ValueError."""
    if not isinstance(m, dict) or set(m) != set(CLAVES_MATRIZ):  # claves exactas
        raise ValueError(f"La matriz debe tener exactamente {list(CLAVES_MATRIZ)}; se recibió {m!r}")  # forma
    for clave in CLAVES_MATRIZ:  # cada conteo
        v = m[clave]  # valor
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0:  # entero ≥ 0
            raise ValueError(f"{clave} debe ser un entero ≥ 0; se recibió {v!r}")  # conteo inválido
    if m["vp"] + m["fn"] == 0 or m["fp"] + m["vn"] == 0:  # sin malos o sin buenos
        raise ValueError("La matriz necesita malos y buenos (TPR y TNR no están definidas).")  # una clase
    return {k: int(m[k]) for k in CLAVES_MATRIZ}  # int nativos


def tasas(m: dict) -> dict:
    """{tpr, tnr, tasa_rechazo, tasa_aprobacion} de la matriz ``m``."""
    m = _validar_matriz(m)  # conteos válidos
    n = sum(m.values())  # créditos
    rechazo = (m["vp"] + m["fp"]) / n  # (VP + FP)/n
    return {  # en el orden de B-pd §3.9
        "tpr": m["vp"] / (m["vp"] + m["fn"]),  # sensibilidad
        "tnr": m["vn"] / (m["vn"] + m["fp"]),  # especificidad
        "tasa_rechazo": rechazo,  # rechazados
        "tasa_aprobacion": 1.0 - rechazo,  # aprobados
    }


# %% [markdown]
# ## Costo
#
# En unidades de $G$ (lo que se pierde al rechazar a un bueno):
#
# $$\text{costo} = \frac{L}{G}\, FN + FP, \qquad \text{costo por crédito} = \frac{\text{costo}}{n}.$$
#
# La razón $L/G$ debe ser real, finita y $> 0$.

# %%
def _razon(razon) -> float:
    """L/G real, finita y > 0; si no, ValueError."""
    return validar_real(razon, "razón L/G", 0.0, math.inf, abierto_abajo=True)  # (0, ∞) (isfinite excluye +inf)


def costo(m: dict, razon) -> dict:
    """{costo, costo_por_credito} = (L/G)·FN + FP y su valor por crédito."""
    r = _razon(razon)  # razón válida
    m = _validar_matriz(m)  # conteos válidos
    total = r * m["fn"] + m["fp"]  # en unidades de G
    return {"costo": total, "costo_por_credito": total / sum(m.values())}  # total y por crédito


# %% [markdown]
# ## Umbral teórico
#
# Rechazar si la pérdida esperada de aprobar supera la de rechazar: $p\,L > (1 - p)\,G \iff p > p^\*$ con
#
# $$p^\* = \frac{1}{1 + L/G}.$$
#
# Es el corte que minimiza el costo esperado **si la PD está calibrada**; es solo una referencia teórica.

# %%
def umbral_teorico(razon) -> float:
    """p* = 1/(1 + L/G)."""
    return 1.0 / (1.0 + _razon(razon))  # referencia con PD calibrada


# %% [markdown]
# ## Rejilla de cuotas o de umbrales
#
# Lista no vacía, estrictamente creciente, de reales finitos en $[0, 1]$. Devuelve `float64`.

# %%
def validar_rejilla(rejilla, nombre: str = "rejilla") -> np.ndarray:
    """rejilla como float64 1-D, no vacía, estrictamente creciente y en [0, 1]; si no, ValueError."""
    valores = [validar_real(v, f"{nombre}[{i}]", 0.0, 1.0) for i, v in enumerate(rejilla)]  # cada valor
    arr = np.asarray(valores, dtype=float)  # vector
    if arr.shape[0] == 0:  # vacía
        raise ValueError(f"{nombre} no puede estar vacía")  # vacía
    if (np.diff(arr) <= 0).any():  # repetidos o desordenados
        raise ValueError(f"{nombre} debe ser estrictamente creciente")  # orden
    return arr  # válida


# %% [markdown]
# ## Conteos por cuota (camino rápido)
#
# Con el orden estable descendente y la suma acumulada de $y$, para cada cuota con $k$ rechazos:
# $VP(k) = \sum_{j < k} y_{(j)}$, $FP = k - VP$, $FN = P - VP$, $VN = N - FP$. Da lo mismo que
# `matriz_confusion(y, rechazos_por_cuota(p, r))` (una prueba lo exige) con un solo ordenamiento.

# %%
def conteos_por_cuota(y, p, rejilla) -> dict:
    """{vp, fn, fp, vn}: arreglos int64 con un valor por cuota de ``rejilla``."""
    y_arr, p_arr = _validar_yp(y, p)  # entradas válidas
    cuotas = validar_rejilla(rejilla, "rejilla_cuotas")  # cuotas válidas
    n, malos = y_arr.shape[0], int(y_arr.sum())  # créditos y malos
    acumulado = np.concatenate(([0], np.cumsum(y_arr[_orden_descendente(p_arr)])))  # VP con k = 0…n
    k = np.array([k_de_cuota(r, n) for r in cuotas], dtype=np.int64)  # rechazos por cuota
    vp = acumulado[k]  # malos rechazados
    fp = k - vp  # buenos rechazados
    return {"vp": vp, "fn": malos - vp, "fp": fp, "vn": (n - malos) - fp}  # cuatro arreglos


# %% [markdown]
# ## Conteos de una rejilla
#
# Los conteos de `conteos_por_cuota` validados contra la rejilla: las claves que se usan y un valor por cuota. Así,
# un mismo cálculo (un solo ordenamiento de $p$) sirve para todas las razones y metas.

# %%
def _conteos_validos(conteos, cuotas: np.ndarray, claves: tuple[str, ...]) -> dict:
    """{clave: arreglo} con un valor por cuota; si no, ValueError."""
    if not isinstance(conteos, dict) or not set(claves) <= set(conteos):  # claves necesarias
        raise ValueError(f"conteos debe tener las claves {list(claves)}")  # forma
    arreglos = {k: np.asarray(conteos[k]) for k in claves}  # como arreglos
    if any(a.shape != cuotas.shape for a in arreglos.values()):  # un valor por cuota
        raise ValueError(f"conteos debe tener {cuotas.shape[0]} valores por clave (uno por cuota)")  # largo
    return arreglos  # válidos


# %% [markdown]
# ## Cuota óptima
#
# $$r^\* = \arg\min_{r \in \mathcal R}\ \Big(\frac{L}{G}\, FN(r) + FP(r)\Big)$$
#
# sobre la rejilla $\mathcal R$ (creciente), con las etiquetas del conjunto (validación en B11). Empate exacto → la
# **menor** cuota (`argmin` devuelve el primer mínimo). `cuota_optima_conteos` recibe los conteos ya calculados;
# `cuota_optima` los calcula y la llama.

# %%
def cuota_optima_conteos(conteos: dict, rejilla, razon) -> float:
    """Cuota de la rejilla que minimiza (L/G)·FN + FP desde conteos_por_cuota; empate → la menor."""
    r = _razon(razon)  # razón válida
    cuotas = validar_rejilla(rejilla, "rejilla_cuotas")  # rejilla creciente
    c = _conteos_validos(conteos, cuotas, ("fn", "fp"))  # FN y FP por cuota
    return float(cuotas[int(np.argmin(r * c["fn"] + c["fp"]))])  # primer mínimo = menor cuota


def cuota_optima(y, p, razon, rejilla) -> float:
    """Cuota de la rejilla que minimiza (L/G)·FN + FP; empate → la menor."""
    _razon(razon)  # razón válida antes de ordenar
    return cuota_optima_conteos(conteos_por_cuota(y, p, rejilla), rejilla, razon)  # un solo camino


# %% [markdown]
# ## Cuota para una meta de sensibilidad
#
# $$r_m = \min\{r \in \mathcal R : \mathrm{TPR}(r) \ge m\}$$
#
# con $m \in [0, 1]$; `None` si ninguna cuota de la rejilla la alcanza. Igual que arriba: versión desde conteos y
# versión directa.

# %%
def cuota_para_meta_conteos(conteos: dict, rejilla, meta) -> float | None:
    """Menor cuota de la rejilla con TPR ≥ meta desde conteos_por_cuota (None si ninguna)."""
    m = validar_real(meta, "meta", 0.0, 1.0)  # meta válida
    cuotas = validar_rejilla(rejilla, "rejilla_cuotas")  # rejilla creciente
    c = _conteos_validos(conteos, cuotas, ("vp", "fn"))  # VP y FN por cuota
    tpr = c["vp"] / (c["vp"] + c["fn"])  # TPR por cuota
    alcanzan = np.flatnonzero(tpr >= m)  # cuotas que alcanzan la meta
    return float(cuotas[alcanzan[0]]) if alcanzan.size else None  # la menor o None


def cuota_para_meta(y, p, meta, rejilla) -> float | None:
    """Menor cuota de la rejilla con TPR ≥ meta (None si ninguna)."""
    validar_real(meta, "meta", 0.0, 1.0)  # meta válida antes de ordenar
    return cuota_para_meta_conteos(conteos_por_cuota(y, p, rejilla), rejilla, meta)  # un solo camino


# %% [markdown]
# ## Curva de sensibilidad al umbral
#
# Para cada umbral $t$: $VP(t) = \#\{\text{malos con } p \ge t\}$ y $FP(t) = \#\{\text{buenos con } p \ge t\}$
# (búsqueda binaria sobre las PD ordenadas de cada clase), y de ahí TPR, TNR y tasa de rechazo. Da lo mismo que
# `tasas(matriz_confusion(y, rechazos_por_umbral(p, t)))` en cada $t$.

# %%
def curva_sensibilidad(y, p, umbrales) -> dict:
    """{umbrales, tpr, tnr, tasa_rechazo}: listas de float, una entrada por umbral."""
    y_arr, p_arr = _validar_yp(y, p)  # entradas válidas
    t = validar_rejilla(umbrales, "rejilla_umbrales")  # umbrales válidos
    p_malos, p_buenos = np.sort(p_arr[y_arr == 1]), np.sort(p_arr[y_arr == 0])  # PD por clase
    vp = p_malos.shape[0] - np.searchsorted(p_malos, t, side="left")  # malos con p ≥ t
    fp = p_buenos.shape[0] - np.searchsorted(p_buenos, t, side="left")  # buenos con p ≥ t
    return {  # listas de float nativos (van al JSON)
        "umbrales": [float(x) for x in t],  # eje
        "tpr": [float(x) for x in vp / p_malos.shape[0]],  # sensibilidad
        "tnr": [float(x) for x in 1.0 - fp / p_buenos.shape[0]],  # especificidad
        "tasa_rechazo": [float(x) for x in (vp + fp) / y_arr.shape[0]],  # rechazados
    }


# %% [markdown]
# ## Brier del modelo trivial
#
# PD constante $q$: $\mathrm{Brier} = \frac1n \sum_i (q - y_i)^2 = \pi(1-\pi) + (q-\pi)^2$ con $\pi$ la tasa del
# conjunto (las pruebas comprueban la forma cerrada). Se calcula con `metricas.brier`.

# %%
def brier_trivial(y, q) -> float:
    """Brier de la PD constante q (metricas.brier)."""
    y_arr = np.asarray(y)  # objetivo
    return brier(y_arr, np.full(y_arr.shape, q, dtype=float))  # misma función de B1 (valida q ∈ [0, 1])
