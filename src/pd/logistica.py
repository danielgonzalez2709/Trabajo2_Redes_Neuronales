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
# # Regresión logística de referencia con búsqueda de $C$ (Parte B §3.1, núcleo §5.3)
#
# Modelo de baja complejidad del enunciado (Parte 1, tarea 3) sobre el **mismo** preprocesamiento común que el MLP
# (núcleo §5.2 y regla de comparación justa, núcleo §6). Cumple el protocolo `ModeloPD` del núcleo §5.3: `fit`,
# `predict_proba`, `predict_logit`, `n_parametros`, `guardar` y `cargar`; además expone `coeficientes()` (los usa la
# scorecard de la Parte C), la tabla de la búsqueda y el $C$ elegido.
#
# **Convención:** $y_i = 1$ malo (incumplió), $y_i = 0$ bueno; el modelo da el logit $z = w^\top x + b$ y la PD es
# $p = \sigma(z) = 1/(1 + e^{-z})$.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""Logística de PD con búsqueda de C en una rejilla declarada usando la log-loss de validación (Parte B §3.1)."""

import copy  # copias de la config y de la tabla de la búsqueda
import math  # isfinite para rechazar ±inf y NaN en la config
import time  # reloj por defecto de la búsqueda y del fit
import warnings  # avisos de sklearn convertidos en error, solo localmente
from numbers import Real  # comprobación de que un valor es numérico
from pathlib import Path  # rutas de guardado

import joblib  # persistencia del estimador de sklearn
import numpy as np  # entradas y salidas del contrato ModeloPD
import sklearn  # versión instalada (el archivo guardado queda atado a ella)
from sklearn.exceptions import ConvergenceWarning  # aviso de lbfgs sin converger
from sklearn.linear_model import LogisticRegression  # el estimador
from sklearn.metrics import log_loss  # criterio de la búsqueda

from src.common.metricas import auc  # AUC de validación, registrada al lado del criterio
from src.common.seeds import validar_entero  # regla única de enteros del proyecto (sin modificar)

__all__ = [  # interfaz pública acordada (tarea B3); lo demás es interno
    "CLAVES_LOGISTICA", "validar_cfg_logistica",  # config
    "crear_estimador", "logloss", "ajustar_sin_avisos",  # piezas inyectables
    "elegir_indice", "esta_en_borde", "buscar_C",  # búsqueda
    "LogisticaPD",  # el modelo que cumple ModeloPD (núcleo §5.3)
]

CLAVES_LOGISTICA = ("rejilla_C", "max_iter", "tol")  # claves exactas del bloque logistica
SOLVER = "lbfgs"  # optimizador determinista de sklearn (Parte B §3.1)
ETIQUETAS = [0, 1]  # clases de la PD: bueno y malo
CLASES_ESPERADAS = (0, 1)  # classes_ del estimador: la columna 1 de predict_proba es P(malo)
CLAVES_ARCHIVO = ("estimador", "C_elegido", "busqueda", "n_entradas", "version_sklearn")  # contenido del joblib


# %% [markdown]
# ## Validación de la config: tipos numéricos
#
# - **entero:** `seeds.validar_entero` (int de Python o de numpy, nunca bool); su `TypeError` se convierte en
#   `ValueError`, porque toda falla de `validar_cfg_logistica` es `ValueError`;
# - **real:** número real y finito (nunca bool ni texto); $\pm\infty$ y NaN no son valores válidos.
#
# TODO(refactor-post-B5): estos dos validadores duplican los privados de `src/pd/mlp.py` (no se importa nada
# privado de otro módulo); al cerrar B5 se mueven a un módulo común.

# %%
def _entero_config(valor, nombre: str, minimo: int) -> int:
    """``seeds.validar_entero`` con su TypeError convertido en ValueError."""
    try:  # reutiliza la regla única de enteros del proyecto
        return validar_entero(valor, nombre, minimo=minimo)  # ValueError si es < minimo
    except TypeError as exc:  # flotante, bool, texto…
        raise ValueError(str(exc)) from exc  # mismo mensaje, tipo de error del contrato


def _es_real(valor) -> bool:
    """True si ``valor`` es un número real finito (los bool no cuentan)."""
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, Real):  # bool es subclase de int: se excluye
        return False  # no es un número de la config
    return math.isfinite(valor)  # ±inf y NaN tampoco valen


# %% [markdown]
# ## Validación de la config: claves
#
# El bloque `logistica` es un diccionario con **exactamente** las claves de `CLAVES_LOGISTICA`: una que falte no
# tiene valor por defecto y una que sobre suele ser un error de escritura.

# %%
def _validar_claves(cfg_log) -> None:
    """ValueError si ``cfg_log`` no es un diccionario con exactamente CLAVES_LOGISTICA."""
    if not isinstance(cfg_log, dict):  # p. ej. una lista pasada por error
        raise ValueError(f"El bloque logistica debe ser un diccionario; se recibió {type(cfg_log).__name__}")  # tipo
    faltan = set(CLAVES_LOGISTICA) - set(cfg_log)  # sin valor por defecto: deben venir de la config
    sobran = set(cfg_log) - set(CLAVES_LOGISTICA)  # se ignorarían en silencio si no se rechazaran
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"Bloque logistica inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # ambas


# %% [markdown]
# ## Validación de la config: rejilla de $C$
#
# `rejilla_C` es una lista (o tupla) **no vacía** de reales finitos $C > 0$ **sin duplicados**: $C$ es el inverso de
# la fuerza de la regularización, así que $C \le 0$ no define un problema; un duplicado ajustaría dos veces lo mismo
# e inflaría el presupuesto `n_configuraciones_busqueda`. No se exige orden: la elección no depende de la posición.

# %%
def _validar_rejilla(rejilla) -> None:
    """ValueError si ``rejilla`` no es una lista o tupla no vacía de reales finitos > 0 sin duplicados."""
    if not isinstance(rejilla, (list, tuple)) or len(rejilla) == 0:  # un escalar o una lista vacía no son rejilla
        raise ValueError(f"logistica.rejilla_C debe ser una lista no vacía; se recibió {rejilla!r}")  # forma
    for i, C in enumerate(rejilla):  # cada valor por separado, para nombrarlo en el error
        if not _es_real(C) or C <= 0:  # real finito y positivo
            raise ValueError(f"logistica.rejilla_C[{i}] debe ser un real finito > 0; se recibió {C!r}")  # rango
    if len(set(rejilla)) != len(rejilla):  # 0.1 dos veces
        raise ValueError(f"logistica.rejilla_C no admite duplicados; se recibió {rejilla!r}")  # duplicado


# %% [markdown]
# ## Validación del bloque `logistica` completo
#
# | Clave | Regla |
# |---|---|
# | (todas) | exactamente `rejilla_C`, `max_iter`, `tol` |
# | `rejilla_C` | lista o tupla no vacía de reales finitos $> 0$, sin duplicados |
# | `max_iter` | entero $\ge 1$ (ni `1.0` ni `True`) |
# | `tol` | real finito $> 0$ |
#
# Son reglas de validez de nuestra config, no constantes del dominio. Devuelve una copia profunda con la rejilla
# como lista; nunca muta la config recibida (B5 la reutiliza).

# %%
def validar_cfg_logistica(cfg_log: dict) -> dict:
    """Valida el bloque ``logistica`` (tabla de arriba) y devuelve una copia profunda; cualquier falla → ValueError."""
    _validar_claves(cfg_log)  # primero las claves: las demás reglas las indexan
    _validar_rejilla(cfg_log["rejilla_C"])  # valores de C
    _entero_config(cfg_log["max_iter"], "logistica.max_iter", minimo=1)  # al menos una iteración
    if not _es_real(cfg_log["tol"]) or cfg_log["tol"] <= 0:  # tolerancia positiva y finita
        raise ValueError(f"logistica.tol debe ser un real finito > 0; se recibió {cfg_log['tol']!r}")  # rango
    copia = copy.deepcopy(cfg_log)  # copia: quien la use no altera la del llamador
    copia["rejilla_C"] = list(copia["rejilla_C"])  # una tupla se devuelve como lista
    return copia  # bloque validado


# %% [markdown]
# ## Bloque `logistica` de la config completa
#
# `fit` y `correr_logistica` reciben la config **completa** (núcleo §5.3) y leen `cfg["logistica"]`. Una config que
# no es diccionario o que no trae el bloque es una falla de config → `ValueError` (no `KeyError` ni `TypeError`),
# igual que cualquier otra regla de `validar_cfg_logistica`.

# %%
def _bloque_logistica(cfg) -> dict:
    """Copia validada de ``cfg["logistica"]``; ValueError si cfg no es dict o no trae el bloque."""
    if not isinstance(cfg, dict) or "logistica" not in cfg:  # config sin el bloque de la logística
        raise ValueError("La config debe ser un diccionario con el bloque 'logistica'")  # falla de config
    return validar_cfg_logistica(cfg["logistica"])  # reglas del bloque y copia profunda


# %% [markdown]
# ## Estimador: objetivo de sklearn con `lbfgs` y papel de $C$
#
# Con $n$ filas de train, `LogisticRegression(C, solver="lbfgs")` minimiza (penalización L2 por defecto; el intercepto
# $b$ **no** se penaliza)
#
# $$f(w, b) = \frac{1}{n}\sum_{i=1}^{n} \Big[-y_i \log p_i - (1 - y_i)\log(1 - p_i)\Big] + \frac{1}{2Cn}\lVert w
# \rVert_2^2, \qquad p_i = \sigma(w^\top x_i + b).$$
#
# (Lo verificamos en `sklearn/linear_model/_logistic.py` de la versión fijada: `l2_reg_strength = 1/(C·n)`.) Es
# decir, $C$ es el **inverso** de la fuerza: con $\lambda = 1/(C n)$ el término es $(\lambda/2)\lVert w\rVert^2$, la
# misma forma que la L2 del MLP (Parte B §3.2). $C$ grande ⇒ casi sin regularización; $C$ pequeño ⇒ pesos encogidos
# hacia 0.
#
# **Condiciones de optimalidad** (gradiente nulo; las verifica `test_05_optimalidad`):
#
# $$\nabla_w f = \frac{1}{n} X^\top (p - y) + \frac{w}{C n} = 0, \qquad \frac{\partial f}{\partial b} = \frac{1}{n}
# \sum_{i}(p_i - y_i) = 0.$$
#
# `lbfgs` se detiene cuando $\max_j |\nabla f|_j \le$ `tol` (o por la reducción relativa de $f$) o al agotar
# `max_iter`. No se pasa `penalty` (deprecado en sklearn 1.8: FutureWarning), ni `multi_class` (ya no existe en la
# versión fijada: TypeError), ni `class_weight` (reponderar las clases descalibraría la PD, que alimenta la
# scorecard y la E[L]). `random_state` solo lo usan `sag`, `saga` y `liblinear` (docstring de sklearn).

# %%
def crear_estimador(C: float, max_iter: int, tol: float) -> LogisticRegression:
    """``LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, tol=tol)``, el resto por defecto."""
    return LogisticRegression(C=C, solver=SOLVER, max_iter=max_iter, tol=tol)  # L2 por defecto, sin class_weight


# %% [markdown]
# ## Criterio: log-loss de validación
#
# $$\mathrm{LL} = -\frac{1}{n_{\text{val}}}\sum_{i \in \text{val}}\Big[y_i \log p_i + (1 - y_i)\log(1 - p_i)\Big].$$
#
# Es una regla de puntuación propia: premia a la vez discriminación y calibración (Parte B §3.3). Se calcula con
# `sklearn.metrics.log_loss(y, p, labels=[0, 1])`: `labels` evita el error con una sola clase presente.

# %%
def logloss(y, p) -> float:
    """Log-loss binaria con ``labels=[0, 1]``."""
    return float(log_loss(y, p, labels=ETIQUETAS))  # float nativo (va al JSON)


# %% [markdown]
# ## Ajuste sin avisos silenciosos
#
# Un `ConvergenceWarning` (lbfgs agotó `max_iter`) o un `FutureWarning` (parámetro deprecado) se convierten en
# **error** dentro de un `warnings.catch_warnings` local: el ajuste no se acepta en silencio y los filtros globales
# quedan iguales al salir.

# %%
def ajustar_sin_avisos(estimador, X, y):
    """Ajusta ``estimador`` con (X, y); ConvergenceWarning y FutureWarning son error; devuelve el estimador ajustado."""
    with warnings.catch_warnings():  # restaura los filtros globales al salir
        warnings.simplefilter("error", ConvergenceWarning)  # lbfgs sin converger = error
        warnings.simplefilter("error", FutureWarning)  # parámetro deprecado = error
        return estimador.fit(X, y)  # sklearn devuelve el propio estimador


# %% [markdown]
# ## Regla de elección
#
# $$i^\star = \arg\min_i \mathrm{LL}^{\text{val}}_i,$$
#
# con argmin **estricto** (sin tolerancia) y, si hay empate exacto, gana el **menor** $C$ (el modelo más
# regularizado), sin importar su posición en la rejilla. La AUC de validación se registra pero no decide.

# %%
def elegir_indice(filas: list[dict]) -> int:
    """Índice de la fila con menor ``logloss_val``; empate exacto → la de menor ``C``."""
    if not filas:  # sin filas no hay elección
        raise ValueError("No hay filas de búsqueda entre las que elegir")  # rejilla vacía
    return min(range(len(filas)), key=lambda i: (filas[i]["logloss_val"], filas[i]["C"]))  # (pérdida, C) lexicográfico


# %% [markdown]
# ## ¿$C$ en el borde de la rejilla?
#
# $C^\star$ está en el borde si la rejilla tiene más de un valor y $C^\star = \min$ o $\max$ de la rejilla (por
# valor, no por posición): el óptimo podría estar fuera y convendría ampliarla. Con un solo valor no hay búsqueda
# ni borde.

# %%
def esta_en_borde(rejilla, C: float) -> bool:
    """True si la rejilla tiene más de un valor y ``C`` es su mínimo o su máximo."""
    return len(rejilla) > 1 and C in (min(rejilla), max(rejilla))  # bool nativo


# %% [markdown]
# ## PD a partir del estimador
#
# $p = $ `predict_proba(X)[:, 1]` solo si `classes_` $= [0, 1]$: así la columna 1 es $P(y = 1) = P(\text{malo})$.

# %%
def _proba_malo(estimador, X) -> np.ndarray:
    """P(malo) en float64 (n,), comprobando que ``classes_`` sea [0, 1]."""
    if tuple(np.asarray(estimador.classes_).tolist()) != CLASES_ESPERADAS:  # otra codificación de clases
        raise ValueError(f"classes_ del estimador debe ser {list(CLASES_ESPERADAS)}; es {estimador.classes_!r}")  # error
    return np.asarray(estimador.predict_proba(X)[:, 1], dtype=np.float64)  # columna de la clase 1


# %% [markdown]
# ## Una fila de la búsqueda
#
# Para cada $C$: se ajusta **solo con train**, se mide en validación $\mathrm{LL}^{\text{val}}$ y
# $\mathrm{AUC}^{\text{val}}$, y se registran las iteraciones de lbfgs (`n_iter_`) y el tiempo del ajuste.

# %%
def _probar_C(C, X_tr, y_tr, X_val, y_val, cfg_log, crear, criterio, auc_fn, reloj) -> tuple:
    """Ajusta un C solo con train y devuelve (estimador ajustado, fila con sus 5 claves)."""
    estimador = crear(C, cfg_log["max_iter"], cfg_log["tol"])  # estimador sin ajustar
    inicio = reloj()  # comienzo del ajuste
    estimador = ajustar_sin_avisos(estimador, X_tr, y_tr)  # solo train: val nunca entrena
    tiempo = reloj() - inicio  # duración del ajuste de este C
    p_val = _proba_malo(estimador, X_val)  # PD de validación
    fila = {  # registro de este C
        "C": float(C),  # valor de la rejilla
        "logloss_val": float(criterio(y_val, p_val)),  # criterio que decide
        "auc_val": float(auc_fn(y_val, p_val)),  # al lado, no decide
        "n_iter": int(np.max(estimador.n_iter_)),  # iteraciones de lbfgs (n_iter_ es un arreglo de un valor)
        "tiempo_s": float(tiempo),  # segundos del ajuste
    }
    return estimador, fila  # estimador ajustado y su fila


# %% [markdown]
# ## Búsqueda de $C$
#
# Recorre la rejilla en su orden, ajusta cada $C$ con train, mide en validación y elige con la regla de arriba. El
# modelo final es el del $C$ elegido **ajustado solo con train** (sin reajustar con train + val): validación solo
# elige y prueba se mide una vez.
#
# **Por qué no `LogisticRegressionCV` ni $k$-fold:** la partición es temporal (train ≤ 2012, val 2013, test 2014);
# mezclar años en pliegues aleatorios usaría información del futuro para elegir $C$ y rompería la comparación con
# el MLP, que se elige con la misma validación.

# %%
def buscar_C(X_tr, y_tr, X_val, y_val, cfg_log, crear=crear_estimador, criterio=logloss, auc_fn=auc,
             reloj=time.perf_counter) -> tuple:
    """Devuelve (estimador del C elegido, filas en el orden de la rejilla, índice elegido)."""
    estimadores, filas = [], []  # uno por C
    for C in cfg_log["rejilla_C"]:  # orden de la rejilla
        estimador, fila = _probar_C(C, X_tr, y_tr, X_val, y_val, cfg_log, crear, criterio, auc_fn, reloj)  # un C
        estimadores.append(estimador)  # se conserva para no reajustar el elegido
        filas.append(fila)  # tabla de la búsqueda
    indice = elegir_indice(filas)  # argmin de la log-loss de val (empate → menor C)
    return estimadores[indice], filas, indice  # el elegido ya está ajustado con train


# %% [markdown]
# ## Validación de los datos
#
# `X_tr` y `X_val` matrices 2-D finitas con las mismas columnas; `y_tr` e `y_val` vectores 0/1 con una etiqueta por
# fila; `y_tr` con las dos clases (sin ellas no hay logística que ajustar).
#
# TODO(refactor-post-B5): duplica la validación privada de `MLPPD` (no se importa nada privado de `mlp.py`).

# %%
def _verificar_matriz(X, n_columnas: int, nombre: str) -> None:
    """ValueError si X no es una matriz 2-D finita con ``n_columnas`` columnas."""
    forma = np.shape(X)  # sin copiar los datos
    if len(forma) != 2 or forma[1] != n_columnas:  # mismas columnas que en fit
        raise ValueError(f"{nombre} debe tener forma (n, {n_columnas}); se recibió {forma}")  # forma
    if not np.all(np.isfinite(X)):  # un NaN indica un preprocesamiento incompleto
        raise ValueError(f"{nombre} contiene valores no finitos (NaN o ±inf)")  # dato


def _verificar_objetivo(y, n: int, nombre: str) -> None:
    """ValueError si y no es un vector 0/1 de longitud ``n``."""
    y = np.asarray(y)  # acepta listas o Series
    if y.shape != (n,):  # una etiqueta por fila; (n, 1) también se rechaza
        raise ValueError(f"{nombre} debe tener forma ({n},); se recibió {y.shape}")  # forma
    if not np.all((y == 0) | (y == 1)):  # un NaN tampoco es 0 ni 1
        raise ValueError(f"{nombre} debe contener solo 0 y 1")  # valores


def _validar_datos(X_tr, y_tr, X_val, y_val) -> int:
    """Valida los cuatro arreglos y devuelve el número de columnas."""
    if np.ndim(X_tr) != 2:  # X_tr fija las columnas
        raise ValueError(f"X_tr debe ser 2-D; se recibió la forma {np.shape(X_tr)}")  # forma
    n_entradas = int(np.shape(X_tr)[1])  # columnas del modelo
    _verificar_matriz(X_tr, n_entradas, "X_tr")  # finita
    _verificar_matriz(X_val, n_entradas, "X_val")  # mismas columnas que train
    _verificar_objetivo(y_tr, np.shape(X_tr)[0], "y_tr")  # etiquetas de train
    _verificar_objetivo(y_val, np.shape(X_val)[0], "y_val")  # etiquetas de val
    if np.unique(np.asarray(y_tr)).size < len(ETIQUETAS):  # una sola clase en train
        raise ValueError("y_tr debe contener las dos clases (0 y 1)")  # sin malos o sin buenos
    return n_entradas  # fit lo asigna solo si todo sale bien


# %% [markdown]
# ## Modelo de PD (`LogisticaPD`)
#
# Cumple el protocolo `ModeloPD` del núcleo §5.3. La búsqueda y el reloj se reciben por argumento (`buscar` y
# `reloj`, inversión de dependencias): las pruebas pueden sustituirlos sin tocar la clase.
#
# ### `fit`
#
# 1. Valida la config (`_bloque_logistica(cfg)`, copia sin mutar) y los datos.
# 2. Convierte X a `float64` (la precisión del ajuste; float32 y float64 con los mismos valores dan lo mismo).
# 3. Llama a `buscar` con la copia validada y asigna el estado **solo al final**: si algo falla, un modelo ya
#    entrenado sigue igual.
#
# **Por qué se ignora `seed`:** `lbfgs` es determinista (no usa azar); en `LogisticRegression` el `random_state`
# solo lo usan los solvers `sag`, `saga` y `liblinear`. Por eso se reporta **una** corrida y su desviación entre
# semillas es 0 (Parte B §3.1). `seed` se acepta para cumplir el protocolo `ModeloPD`.
#
# `tiempo_total_s` es el reloj de **todo** el `fit` (los ajustes de toda la rejilla): es el presupuesto de cómputo
# de la familia logística (núcleo §6). El reloj se recibe por argumento (`reloj`, por defecto `time.perf_counter`)
# para que las pruebas fijen el tiempo exacto.
#
# ### Predicción y parámetros
#
# `predict_logit` $= w^\top x + b$ (`decision_function`) y `predict_proba` $= \sigma(w^\top x + b)$, ambos `float64`
# $(n,)$. `n_parametros` $= d + 1$ (pesos + intercepto). Antes de `fit`/`cargar` → `RuntimeError`.
#
# ### `guardar` y `cargar`
#
# `joblib` guarda `{"estimador", "C_elegido", "busqueda", "n_entradas", "version_sklearn"}`. joblib es **pickle**:
# cargar ejecuta código, así que solo se cargan archivos propios; y un estimador serializado solo es fiable con la
# misma versión de sklearn, por eso `cargar` exige que coincida.

# %%
class LogisticaPD:
    """Regresión logística de PD con búsqueda de C que cumple el protocolo ``ModeloPD`` del núcleo §5.3."""

    def __init__(self, buscar=buscar_C, reloj=time.perf_counter):
        self._buscar = buscar  # inyectable: las pruebas pasan una búsqueda falsa
        self._reloj = reloj  # inyectable: mide todo el fit (tiempo_total_s)
        self._estimador = None  # None = sin entrenar ni cargar
        self._busqueda = None  # filas de la búsqueda
        self._C_elegido = None  # C del estimador
        self._n_entradas = None  # columnas que espera predict
        self._tiempo_total_s = None  # reloj de todo el fit (no se guarda en disco)

    # --- validaciones ------------------------------------------------------------------------------------------
    def _verificar_entrenado(self) -> None:
        """RuntimeError si el modelo aún no tiene estimador (ni fit ni cargar)."""
        if self._estimador is None:  # predecir sin coeficientes sería un error de uso
            raise RuntimeError("El modelo no está entrenado: llame a fit() o a cargar() antes")  # uso

    def _matriz_float64(self, X) -> np.ndarray:
        """X validada contra las columnas del modelo y convertida a float64."""
        self._verificar_entrenado()  # antes que las columnas: sin fit no hay n_entradas
        _verificar_matriz(X, self._n_entradas, "X")  # mismas columnas que en fit
        return np.asarray(X, dtype=np.float64)  # misma precisión que el ajuste

    # --- entrenamiento -----------------------------------------------------------------------------------------
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "LogisticaPD":
        """Busca C con ``cfg["logistica"]`` (``seed`` se ignora: lbfgs es determinista); devuelve el modelo."""
        inicio = self._reloj()  # reloj de todo el fit
        cfg_log = _bloque_logistica(cfg)  # copia validada: cfg queda intacta
        n_entradas = _validar_datos(X_tr, y_tr, X_val, y_val)  # falla antes de tocar el estado
        X_tr64 = np.asarray(X_tr, dtype=np.float64)  # precisión del ajuste
        X_val64 = np.asarray(X_val, dtype=np.float64)  # misma precisión en val
        estimador, filas, indice = self._buscar(X_tr64, y_tr, X_val64, y_val, cfg_log)  # búsqueda de C
        tiempo = self._reloj() - inicio  # presupuesto de tiempo de la familia
        self._estimador, self._busqueda = estimador, copy.deepcopy(filas)  # estado, solo al final
        self._C_elegido, self._n_entradas = float(filas[indice]["C"]), n_entradas  # C de la fila elegida
        self._tiempo_total_s = float(tiempo)  # segundos de todo el fit
        return self  # encadenable

    # --- predicción --------------------------------------------------------------------------------------------
    def predict_logit(self, X) -> np.ndarray:
        """Logit w·x + b de cada fila, numpy float64 (n,)."""
        X64 = self._matriz_float64(X)  # validada y en float64
        return np.asarray(self._estimador.decision_function(X64), dtype=np.float64)  # (n,)

    def predict_proba(self, X) -> np.ndarray:
        """PD = P(malo), numpy float64 (n,)."""
        return _proba_malo(self._estimador, self._matriz_float64(X))  # columna de la clase 1

    # --- información -------------------------------------------------------------------------------------------
    def n_parametros(self) -> int:
        """Parámetros entrenables: d pesos + 1 intercepto."""
        self._verificar_entrenado()  # sin estimador no hay coeficientes
        return int(self._estimador.coef_.size + self._estimador.intercept_.size)  # int nativo (va al JSON)

    def coeficientes(self) -> tuple[np.ndarray, float]:
        """Copias de (w float64 (d,), b float)."""
        self._verificar_entrenado()  # sin estimador no hay coeficientes
        w = np.array(self._estimador.coef_[0], dtype=np.float64, copy=True)  # copia: mutarla no altera el modelo
        return w, float(self._estimador.intercept_[0])  # b como float nativo

    @property
    def busqueda(self) -> list[dict]:
        """Copia de las filas de la búsqueda (una por C, en el orden de la rejilla)."""
        self._verificar_entrenado()  # sin fit ni cargar no hay búsqueda
        return copy.deepcopy(self._busqueda)  # copia profunda

    @property
    def C_elegido(self) -> float:
        """C del estimador elegido."""
        self._verificar_entrenado()  # sin fit ni cargar no hay C
        return self._C_elegido  # float

    @property
    def tiempo_total_s(self) -> float:
        """Segundos de todo el último fit (no se guarda en disco)."""
        if self._tiempo_total_s is None:  # sin fit en esta sesión (p. ej. recién cargado)
            raise RuntimeError("No hay tiempo de entrenamiento: el modelo no se entrenó con fit()")  # no None
        return self._tiempo_total_s  # float

    # --- persistencia ------------------------------------------------------------------------------------------
    def guardar(self, ruta) -> None:
        """Guarda con joblib el diccionario de CLAVES_ARCHIVO."""
        self._verificar_entrenado()  # un archivo sin estimador no serviría
        contenido = {  # las 5 claves que exige cargar
            "estimador": self._estimador,  # LogisticRegression ajustada
            "C_elegido": self._C_elegido,  # C elegido
            "busqueda": copy.deepcopy(self._busqueda),  # tabla de la búsqueda
            "n_entradas": self._n_entradas,  # columnas que espera predict
            "version_sklearn": sklearn.__version__,  # el pickle queda atado a esta versión
        }
        joblib.dump(contenido, Path(ruta))  # pickle: solo para archivos propios

    @classmethod
    def cargar(cls, ruta) -> "LogisticaPD":
        """Reconstruye el modelo desde un archivo propio de ``guardar``."""
        contenido = joblib.load(Path(ruta))  # pickle: solo archivos propios
        _verificar_archivo(contenido)  # claves, versión y columnas coherentes
        modelo = cls()  # búsqueda por defecto: el archivo no guarda código
        modelo._estimador = contenido["estimador"]  # estimador ajustado
        modelo._busqueda = copy.deepcopy(contenido["busqueda"])  # tabla de la búsqueda
        modelo._C_elegido = float(contenido["C_elegido"])  # C elegido
        modelo._n_entradas = int(contenido["n_entradas"])  # columnas que espera predict
        return modelo  # listo para predecir (sin tiempo_total_s)


# %% [markdown]
# ## Verificación del archivo cargado
#
# El archivo debe ser un diccionario con las 5 claves de `CLAVES_ARCHIVO`, la misma versión de sklearn instalada
# y `n_entradas` igual al número de columnas de `coef_`.

# %%
def _verificar_archivo(contenido) -> None:
    """ValueError si el contenido del joblib es inconsistente."""
    if not isinstance(contenido, dict):  # otro objeto serializado
        raise ValueError(f"El archivo debe contener un diccionario; contiene {type(contenido).__name__}")  # tipo
    faltan = set(CLAVES_ARCHIVO) - set(contenido)  # claves ausentes
    if faltan:  # archivo incompleto
        raise ValueError(f"Al archivo le faltan las claves {sorted(faltan)}")  # claves
    if contenido["version_sklearn"] != sklearn.__version__:  # pickle de otra versión
        raise ValueError(f"Modelo guardado con sklearn {contenido['version_sklearn']}; "
                         f"instalado {sklearn.__version__}")  # versión
    if contenido["n_entradas"] != contenido["estimador"].coef_.shape[1]:  # columnas incoherentes
        raise ValueError(f"n_entradas = {contenido['n_entradas']} no coincide con coef_ "
                         f"{contenido['estimador'].coef_.shape}")  # columnas
