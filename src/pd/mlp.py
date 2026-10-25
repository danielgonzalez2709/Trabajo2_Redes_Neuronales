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
# # MLP de la probabilidad de incumplimiento (Parte B §3.2, núcleo §5.3)
#
# Red neuronal de alimentación hacia adelante que estima la PD de un crédito a partir de las variables de
# originación ya preprocesadas (matriz `float32` del preprocesador común, núcleo §5.2). Cumple el protocolo
# `ModeloPD` del núcleo §5.3: `fit`, `predict_proba`, `predict_logit`, `n_parametros`, `guardar` y `cargar`.
#
# **Convención:** $y_i = 1$ malo (incumplió), $y_i = 0$ bueno; la red devuelve el **logit** $z = \log\frac{p}{1-p}$
# y la PD es $p = \sigma(z)$. La scorecard (Parte C) consume el logit.
#
# Toda la arquitectura y el entrenamiento salen del bloque `mlp` de la config (núcleo §5.3), así que se cambian con
# `--set` sin tocar código (núcleo §4.1).
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""MLP de PD: Linear → activación → Dropout por capa oculta y salida Linear(·, 1) con el logit (núcleo §5.3)."""

import copy  # copias de la config para no mutar la recibida
import math  # isfinite para rechazar ±inf en la config
from numbers import Real  # comprobación de que un valor es numérico
from pathlib import Path  # rutas de guardado

import numpy as np  # entradas y salidas del contrato ModeloPD
import torch  # tensores, guardado y carga
from scipy.special import expit  # σ(z) estable en float64
from torch import nn  # capas de la red y pérdida

from src.common.seeds import fijar_semillas, validar_entero  # semillas y validación de enteros (sin modificar)
from src.common.torch_utils import (  # utilidades compartidas de PyTorch
    DEVICES_VALIDOS,  # auto | cpu | cuda
    a_numpy,  # tensor → numpy float64 (n,)
    a_tensor,  # numpy → tensor float32
    activar_determinismo,  # modo determinista en CPU
    resolver_device,  # mlp.device → torch.device
)
from src.pd.entrenar import OPTIMIZADORES, construir_optimizador, entrenar  # bucle genérico y optimizadores

__all__ = [  # interfaz pública acordada (tarea B2); lo demás es interno
    "ACTIVACIONES", "CLAVES_MLP", "validar_cfg_mlp",  # lo que B4 necesita para armar el espacio de búsqueda
    "construir_red", "contar_parametros", "construir_perdida",  # fábricas reutilizables e inyectables
    "MLPPD",  # el modelo que cumple ModeloPD (núcleo §5.3)
]


# %% [markdown]
# ## Neurona, capa y activaciones
#
# Una capa oculta con $k$ neuronas transforma su entrada $x \in \mathbb{R}^{d}$ en
#
# $$h = \varphi(Wx + b), \qquad W \in \mathbb{R}^{k \times d},\; b \in \mathbb{R}^{k},$$
#
# donde $\varphi$ se aplica elemento a elemento. Activaciones admitidas (`mlp.activacion`):
#
# - **ReLU:** $\varphi(a) = \max(0, a)$.
# - **LeakyReLU:** $\varphi(a) = a$ si $a \ge 0$ y $\alpha a$ si $a < 0$, con la pendiente por defecto de PyTorch
#   $\alpha = 0.01$ (`nn.LeakyReLU(negative_slope=0.01)`; no se cambia).
# - **tanh:** $\varphi(a) = \tanh(a) = \dfrac{e^{a} - e^{-a}}{e^{a} + e^{-a}}$.

# %%
ACTIVACIONES = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}  # nombre de la config → clase

CLAVES_MLP = (  # claves obligatorias del bloque mlp de PD: las 11 de §5.3 + pos_weight
    "capas",  # anchuras de las capas ocultas
    "activacion",  # relu | leaky_relu | tanh
    "dropout",  # probabilidad de apagar una neurona al entrenar, en [0, 1)
    "l2",  # λ de la penalización L2 (weight_decay)
    "optimizador",  # adam | sgd
    "lr",  # tasa de aprendizaje
    "momento",  # momento de SGD
    "batch",  # tamaño de minilote
    "max_epocas",  # tope de épocas
    "paciencia",  # épocas sin mejora antes de detener
    "device",  # auto | cpu | cuda
    "pos_weight",  # peso de la clase positiva en la BCE (PD)
)


# %% [markdown]
# ## Validación del bloque `mlp`: tipos numéricos
#
# Un validador de enteros y un predicado de reales para los valores de la config:
#
# - **entero:** se reutiliza `seeds.validar_entero` (int de Python o de numpy, nunca bool); su `TypeError` se
#   convierte en `ValueError`, porque toda falla de `validar_cfg_mlp` es `ValueError`;
# - **real:** número real y finito (nunca bool); $\pm\infty$ y NaN no son valores válidos de la config.

# %%
def _entero_config(valor, nombre: str, minimo: int) -> int:
    """``seeds.validar_entero`` con su TypeError convertido en ValueError (contrato de validar_cfg_mlp)."""
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
# ## Validación del bloque `mlp`: claves
#
# El bloque debe tener **exactamente** las claves de `CLAVES_MLP`: una que falte no tiene valor por defecto
# (todo sale de la config) y una que sobre suele ser un error de escritura o una clave de otra tarea (p. ej.
# `perdida`, que es de la LGD).

# %%
def _validar_claves(cfg_mlp: dict) -> None:
    """ValueError si falta o sobra alguna clave respecto de CLAVES_MLP."""
    faltan = set(CLAVES_MLP) - set(cfg_mlp)  # sin valor por defecto: deben venir de la config
    sobran = set(cfg_mlp) - set(CLAVES_MLP)  # se ignorarían en silencio si no se rechazaran
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"Bloque mlp inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # ambas listas


# %% [markdown]
# ## Validación del bloque `mlp`: capas
#
# `capas` es una lista no vacía (al menos una capa oculta) de enteros $\ge 1$: una anchura 0 dejaría la red sin
# camino hasta la salida, y un flotante o un bool no es una anchura.

# %%
def _validar_capas(capas) -> None:
    """ValueError si ``capas`` no es una lista no vacía de enteros ≥ 1."""
    if not isinstance(capas, (list, tuple)) or len(capas) == 0:  # al menos una capa oculta
        raise ValueError(f"mlp.capas debe ser una lista no vacía de enteros; se recibió {capas!r}")  # vacía
    for i, ancho in enumerate(capas):  # cada anchura por separado, para nombrarla en el error
        _entero_config(ancho, f"mlp.capas[{i}]", minimo=1)  # entero ≥ 1 (ni flotante ni bool)


# %% [markdown]
# ## Validación del bloque `mlp`: rangos numéricos
#
# Un valor real debe ser finito y cumplir su condición (tabla de `validar_cfg_mlp`). Los contadores enteros
# (`batch`, `max_epocas`, `paciencia`) se validan directamente con `_entero_config`.

# %%
def _validar_rango(cfg_mlp: dict, clave: str, condicion, descripcion: str) -> None:
    """ValueError si ``cfg_mlp[clave]`` no es un real finito que cumpla ``condicion``."""
    valor = cfg_mlp[clave]  # valor a revisar
    if not _es_real(valor) or not condicion(valor):  # finito y dentro del rango
        raise ValueError(f"mlp.{clave} debe ser {descripcion}; se recibió {valor!r}")  # fuera de rango



# %% [markdown]
# ## Validación del bloque `mlp`: opciones
#
# Las claves categóricas solo admiten sus valores declarados (`activacion`, `optimizador`, `device`).

# %%
def _validar_opcion(cfg_mlp: dict, clave: str, opciones) -> None:
    """ValueError si ``cfg_mlp[clave]`` no está en ``opciones``."""
    if cfg_mlp[clave] not in opciones:  # p. ej. 'adamw' o 'sigmoid' no están
        raise ValueError(f"mlp.{clave} debe ser uno de {tuple(opciones)}; se recibió {cfg_mlp[clave]!r}")  # opciones


# %% [markdown]
# ## Validación del bloque `mlp` completo
#
# | Clave | Regla |
# |---|---|
# | (todas) | exactamente las 12 de `CLAVES_MLP` |
# | `capas` | lista no vacía de enteros $\ge 1$ |
# | `activacion` | `relu`, `leaky_relu` o `tanh` |
# | `dropout` | real finito en $[0, 1)$ |
# | `l2` | real finito $\ge 0$ |
# | `optimizador` | `adam` o `sgd` |
# | `lr` | real finito $> 0$ |
# | `momento` | real finito $\ge 0$ |
# | `batch`, `max_epocas`, `paciencia` | enteros $\ge 1$ |
# | `device` | `auto`, `cpu` o `cuda` |
# | `pos_weight` | real finito $> 0$ |
#
# Son reglas de validez de nuestra config, no constantes del dominio. Devuelve una copia profunda: `fit` nunca
# muta la config que recibe (B4 y B5 la reutilizan entre ensayos y semillas).

# %%
def validar_cfg_mlp(cfg_mlp: dict) -> dict:
    """Valida el bloque ``mlp`` de PD (tabla de arriba) y devuelve una copia profunda; cualquier falla → ValueError."""
    _validar_claves(cfg_mlp)  # primero las claves: las demás reglas las indexan
    _validar_capas(cfg_mlp["capas"])  # una anchura inválida rompería construir_red con un error poco claro
    _validar_rango(cfg_mlp, "dropout", lambda v: 0 <= v < 1, "un número en [0, 1)")  # p = 1 apagaría todo
    _validar_rango(cfg_mlp, "l2", lambda v: v >= 0, "un número >= 0")  # λ = 0 = sin L2
    _validar_rango(cfg_mlp, "lr", lambda v: v > 0, "un número > 0")  # lr = 0 no movería los pesos
    _validar_rango(cfg_mlp, "momento", lambda v: v >= 0, "un número >= 0")  # 0 = SGD sin momento
    _validar_rango(cfg_mlp, "pos_weight", lambda v: v > 0, "un número > 0")  # 0 anularía a los malos
    for clave in ("batch", "max_epocas", "paciencia"):  # contadores: un 0 o un 2.5 no tienen sentido
        _entero_config(cfg_mlp[clave], f"mlp.{clave}", minimo=1)  # misma regla que los enteros del proyecto
    _validar_opcion(cfg_mlp, "activacion", ACTIVACIONES)  # si no, KeyError tardío en construir_red
    _validar_opcion(cfg_mlp, "optimizador", OPTIMIZADORES)  # 'adamw' se rechaza: no es L2 (Parte B §6)
    _validar_opcion(cfg_mlp, "device", DEVICES_VALIDOS)  # se valida aquí aunque resolver_device repita la regla
    return copy.deepcopy(cfg_mlp)  # copia: quien la use no altera la del llamador


# %% [markdown]
# ## Arquitectura y dropout
#
# Con capas ocultas de anchuras $k_1, \dots, k_L$ y $d$ entradas, la red es
#
# $$h_0 = x, \qquad h_\ell = \mathrm{Dropout}_p\big(\varphi(W_\ell h_{\ell-1} + b_\ell)\big), \qquad
# z = w_{\text{sal}}^\top h_L + b_{\text{sal}},$$
#
# es decir `[Linear → activación → Dropout]` por capa oculta y una salida `Linear(k_L, 1)` **sin sigmoide**: devuelve
# el logit $z$ (núcleo §5.3).
#
# **Dropout invertido** (docstring de `nn.Dropout` de PyTorch 2.14.1): en entrenamiento cada elemento se multiplica
# por $m_j/(1-p)$ con $m_j \sim \mathrm{Bernoulli}(1-p)$, de modo que la esperanza no cambia; en evaluación
# (`eval()`) es la identidad. Por eso **al predecir no se aplica dropout** (Parte B §6).

# %%
def construir_red(n_entradas: int, capas, activacion: str, dropout: float) -> nn.Sequential:
    """[Linear → activación → Dropout] por capa oculta + Linear(·, 1) final que devuelve el logit."""
    modulos = []  # módulos en orden
    entradas = n_entradas  # anchura que entra a la capa actual
    for ancho in capas:  # una capa oculta por anchura
        modulos.append(nn.Linear(entradas, ancho))  # W h + b
        modulos.append(ACTIVACIONES[activacion]())  # φ (parámetros por defecto de PyTorch)
        modulos.append(nn.Dropout(dropout))  # dropout invertido
        entradas = ancho  # la siguiente capa recibe este ancho
    modulos.append(nn.Linear(entradas, 1))  # salida: el logit (sin Sigmoid)
    return nn.Sequential(*modulos)  # pila secuencial


# %% [markdown]
# ## Número de parámetros
#
# Cada `Linear(a, b)` tiene $a\,b$ pesos y $b$ sesgos, así que con anchuras $k_0 = d, k_1, \dots, k_L, k_{L+1} = 1$
#
# $$n_{\text{par}} = \sum_{\ell=1}^{L+1} \big(k_{\ell-1}\,k_\ell + k_\ell\big).$$
#
# Ejemplo: $d = 48$ y capas $[48, 24]$ dan $(48\cdot48+48) + (48\cdot24+24) + (24\cdot1+1) = 3553$. Se cuentan solo
# los parámetros **entrenables** (núcleo §5.3).

# %%
def contar_parametros(red: nn.Module) -> int:
    """Número de parámetros entrenables de ``red``."""
    return sum(p.numel() for p in red.parameters() if p.requires_grad)  # Σ elementos con gradiente


# %% [markdown]
# ## Pérdida: entropía cruzada binaria sobre logits
#
# Con $z_i$ el logit, $\sigma(z) = 1/(1 + e^{-z})$ y $w_+$ = `mlp.pos_weight`, la pérdida de un minilote es
# (docstring de `nn.BCEWithLogitsLoss`, `reduction='mean'`)
#
# $$\ell = -\frac{1}{n}\sum_{i=1}^{n}\Big[w_+\,y_i \log \sigma(z_i) + (1-y_i)\log\big(1-\sigma(z_i)\big)\Big].$$
#
# `BCEWithLogitsLoss` recibe el **logit** y aplica la sigmoide por dentro de forma numéricamente estable; usar
# `BCELoss` sobre logits sería un error (Parte B §6). Con $w_+ = 1$ es la log-verosimilitud negativa de Bernoulli
# promedio; $w_+ = 1$ es el valor por defecto para no descalibrar la PD (Parte B §3.2).

# %%
def construir_perdida(pos_weight: float, device: torch.device) -> nn.Module:
    """``BCEWithLogitsLoss`` con ``pos_weight`` en ``device``."""
    peso = torch.tensor([float(pos_weight)], dtype=torch.float32, device=device)  # w₊ como tensor (1,)
    return nn.BCEWithLogitsLoss(pos_weight=peso)  # BCE estable sobre logits


# %% [markdown]
# ## Modelo de PD (`ModeloPD`)
#
# Cumple el protocolo `ModeloPD` del núcleo §5.3. Las fábricas (`construir_red`, `construir_optimizador`,
# `construir_perdida`, `entrenar`) se reciben por argumento con valor por defecto (inversión de dependencias): las
# pruebas, B4 y la Parte D pueden sustituir cualquiera sin tocar la clase.
#
# ### `fit`
#
# 1. Valida la config (`validar_cfg_mlp`, copia sin mutar) y los datos (`_validar_datos`): `X_tr` y `X_val` 2-D con
#    las mismas columnas; `y_tr` e `y_val` vectores 0/1 con una etiqueta por fila (un NaN o un 2 → `ValueError`).
# 2. Resuelve el dispositivo y activa el determinismo en CPU (núcleo §7.1).
# 3. Fija las semillas con `fijar_semillas(seed)` **antes** de construir la red, de modo que la inicialización
#    sale de `seed`; el `Generator` de numpy que devuelve es el que baraja los minilotes.
# 4. Construye la red, el optimizador y la pérdida, y llama al bucle genérico con `y` como columna $(n, 1)$ (la
#    forma de la salida `Linear(·, 1)`).
# 5. **Solo al final** asigna el estado del modelo: si algo falla antes, un modelo ya entrenado sigue igual.
#
# ### `predict_logit` y `predict_proba`
#
# `predict_logit` hace una sola pasada en `eval()` (dropout = identidad) y `no_grad`; `predict_proba`
# $= \sigma(\text{logit})$ en `float64`. Ambas devuelven numpy de forma $(n,)$. Antes de `fit`/`cargar` →
# `RuntimeError`; `X` con otro número de columnas → `ValueError`.
#
# ### `n_parametros` y `resultado`
#
# `n_parametros` cuenta los parámetros entrenables (celda de arriba). `resultado` es el `ResultadoEntrenamiento` del
# último `fit` (historial, mejor época); no se guarda en disco, así que un modelo cargado no lo tiene.
#
# ### `guardar` y `cargar`
#
# `guardar` escribe con `torch.save` el diccionario `{"state_dict": pesos en CPU, "config": bloque mlp,
# "n_entradas": columnas}`. `config` es la **de entrenamiento** (incluido el `device` con que se entrenó) y no se
# modifica al cargar. `cargar` lo lee con `torch.load(weights_only=True)` (solo tensores y tipos simples) y
# reconstruye la red en CPU: `_device` es el dispositivo **de inferencia**, que puede diferir del de entrenamiento.

# %%
class MLPPD:
    """MLP de PD que cumple el protocolo ``ModeloPD`` del núcleo §5.3."""

    def __init__(
        self,
        construir_red=construir_red,
        construir_optimizador=construir_optimizador,
        construir_perdida=construir_perdida,
        entrenar=entrenar,
    ):
        self._construir_red = construir_red  # inyectable: las pruebas pasan _Captura para inspeccionar la red
        self._construir_optimizador = construir_optimizador  # inyectable: una prueba puede sustituir el optimizador
        self._construir_perdida = construir_perdida  # inyectable: las pruebas pueden pasar una pérdida guionizada
        self._entrenar = entrenar  # inyectable: el bucle genérico se puede sustituir en pruebas
        self._red = None  # None = sin entrenar ni cargar
        self._cfg_mlp = None  # bloque mlp con que se entrenó (se guarda con el modelo)
        self._n_entradas = None  # columnas que espera predict
        self._device = None  # dispositivo de inferencia
        self._resultado = None  # solo existe tras fit (no se guarda en disco)

    # --- validaciones ------------------------------------------------------------------------------------------
    def _verificar_entrenado(self) -> None:
        """RuntimeError si el modelo aún no tiene red (ni fit ni cargar)."""
        if self._red is None:  # predecir con pesos aleatorios sería un error silencioso
            raise RuntimeError("El modelo no está entrenado: llame a fit() o a cargar() antes")  # uso, no datos

    @staticmethod
    def _verificar_matriz(X, n_columnas: int, nombre: str) -> None:
        """ValueError si X no es 2-D con ``n_columnas`` columnas."""
        forma = np.shape(X)  # sin copiar los datos
        if len(forma) != 2 or forma[1] != n_columnas:  # evita un error críptico de matmul más adelante
            raise ValueError(f"{nombre} debe tener forma (n, {n_columnas}); se recibió {forma}")  # dato, no uso

    @staticmethod
    def _verificar_objetivo(y, n: int, nombre: str) -> None:
        """ValueError si y no es un vector 0/1 de longitud ``n``."""
        y = np.asarray(y)  # acepta listas o Series: se comparan como arreglo
        if y.shape != (n,):  # una etiqueta por fila de X
            raise ValueError(f"{nombre} debe tener forma ({n},); se recibió {y.shape}")  # (n, 1) también se rechaza
        if not np.all((y == 0) | (y == 1)):  # un NaN tampoco es 0 ni 1
            raise ValueError(f"{nombre} debe contener solo 0 y 1")  # la BCE aceptaría un 2 sin quejarse

    def _validar_datos(self, X_tr, y_tr, X_val, y_val) -> int:
        """Valida los cuatro arreglos sin tocar el estado del modelo; devuelve el número de columnas."""
        if np.ndim(X_tr) != 2:  # X_tr fija las columnas: debe ser una matriz
            raise ValueError(f"X_tr debe ser 2-D; se recibió la forma {np.shape(X_tr)}")  # sin columnas no hay red
        n_entradas = int(np.shape(X_tr)[1])  # variable local: el modelo previo no cambia si algo falla
        self._verificar_matriz(X_val, n_entradas, "X_val")  # mismas columnas que train
        self._verificar_objetivo(y_tr, np.shape(X_tr)[0], "y_tr")  # etiquetas de train contra las filas de X_tr
        self._verificar_objetivo(y_val, np.shape(X_val)[0], "y_val")  # y las de val contra X_val
        return n_entradas  # fit lo asigna al modelo solo si todo sale bien

    # --- entrenamiento -----------------------------------------------------------------------------------------
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "MLPPD":
        """Entrena con ``cfg["mlp"]`` y la semilla ``seed``; devuelve el propio modelo."""
        cfg_mlp = validar_cfg_mlp(cfg["mlp"])  # copia validada: cfg queda intacta
        n_entradas = self._validar_datos(X_tr, y_tr, X_val, y_val)  # falla antes de tocar el estado
        device = resolver_device(cfg_mlp["device"])  # 'auto' se resuelve aquí, una vez por fit
        activar_determinismo(device)  # misma semilla y dispositivo → mismas métricas (núcleo §7.1)
        rng = fijar_semillas(seed)  # antes de construir la red: la inicialización sale de seed
        red = self._construir_red(  # la arquitectura sale entera de la config (--set mlp.capas=...)
            n_entradas, cfg_mlp["capas"], cfg_mlp["activacion"], cfg_mlp["dropout"]  # n_entradas viene de los datos
        ).to(device)  # los parámetros viven donde están los datos
        optimizador = self._construir_optimizador(red, cfg_mlp)  # L2 solo en los pesos
        perdida = self._construir_perdida(cfg_mlp["pos_weight"], device)  # BCE sobre logits
        resultado = self._entrenar(  # bucle genérico: no sabe que es PD
            red,  # al volver, con los pesos de la mejor época
            perdida,  # BCE con logits: la red no lleva Sigmoid
            optimizador,  # ya trae la L2 en sus grupos de parámetros
            a_tensor(X_tr, device),  # float32; a_tensor rechaza NaN por si el preprocesador no imputó (núcleo §5.2)
            a_tensor(y_tr, device, columna=True),  # (n, 1): la forma de la salida Linear(·, 1)
            a_tensor(X_val, device),  # val solo decide la parada: nunca entrena
            a_tensor(y_val, device, columna=True),  # (n, 1), igual que en train
            batch=cfg_mlp["batch"],  # lo valida de nuevo entrenar (genérico, lo usa D)
            max_epocas=cfg_mlp["max_epocas"],  # tope; la paciencia puede parar antes
            paciencia=cfg_mlp["paciencia"],  # épocas sin mejora estricta antes de detener
            rng=rng,  # el mismo rng que sembró fijar_semillas: baraja los minilotes
        )
        self._red, self._cfg_mlp, self._n_entradas = red, cfg_mlp, n_entradas  # todo el estado, solo al final
        self._device, self._resultado = device, resultado  # predict usará el mismo dispositivo del fit
        return self  # encadenable: MLPPD().fit(...)

    # --- predicción --------------------------------------------------------------------------------------------
    def predict_logit(self, X) -> np.ndarray:
        """Logit log(p / (1 − p)) de cada fila, numpy float64 (n,); una sola pasada en eval() y no_grad."""
        self._verificar_entrenado()  # antes que las columnas: sin fit no hay n_entradas
        self._verificar_matriz(X, self._n_entradas, "X")  # mismas columnas que en fit
        self._red.eval()  # dropout = identidad al predecir
        with torch.no_grad():  # inferencia: sin grafo
            return a_numpy(self._red(a_tensor(X, self._device)))  # (n, 1) → float64 (n,)

    def predict_proba(self, X) -> np.ndarray:
        """PD = σ(logit), numpy float64 (n,)."""
        return expit(self.predict_logit(X))  # σ estable: no desborda con logits muy negativos

    # --- información -------------------------------------------------------------------------------------------
    def n_parametros(self) -> int:
        """Parámetros entrenables de la red."""
        self._verificar_entrenado()  # sin red no hay arquitectura que contar
        return contar_parametros(self._red)  # la regla de comparación justa reporta este número

    @property
    def resultado(self):
        """ResultadoEntrenamiento del último fit (no se guarda en disco)."""
        if self._resultado is None:  # modelo sin fit en esta sesión (p. ej. recién cargado)
            raise RuntimeError("No hay resultado de entrenamiento: el modelo no se entrenó con fit()")  # no None
        return self._resultado  # frozen: sus campos no se reasignan, pero el historial es una lista: no modificarla

    # --- persistencia ------------------------------------------------------------------------------------------
    def guardar(self, ruta) -> None:
        """Guarda ``{"state_dict" en CPU, "config" (bloque mlp), "n_entradas"}`` con ``torch.save``."""
        self._verificar_entrenado()  # un archivo sin pesos no serviría a la app ni a figures
        estado = {k: v.detach().cpu() for k, v in self._red.state_dict().items()}  # en CPU: cargable sin GPU
        paquete = {  # solo tensores y tipos simples: así lo admite torch.load(weights_only=True)
            "state_dict": estado,  # pesos de la mejor época
            "config": copy.deepcopy(self._cfg_mlp),  # la de entrenamiento, con su device original
            "n_entradas": self._n_entradas,  # columnas: cargar reconstruye la red sin los datos
        }
        torch.save(paquete, Path(ruta))  # pickle restringido; se lee con weights_only=True

    @classmethod
    def cargar(cls, ruta) -> "MLPPD":
        """Reconstruye el modelo desde ``ruta`` (inferencia en CPU) sin necesitar la config."""
        paquete = torch.load(Path(ruta), map_location="cpu", weights_only=True)  # no ejecuta código arbitrario
        n_entradas = int(paquete["n_entradas"])  # una sola conversión, usada abajo
        cfg_mlp = validar_cfg_mlp(paquete["config"])  # config de entrenamiento: no se sobrescribe su device
        modelo = cls()  # fábricas por defecto: el archivo no guarda código
        red = modelo._construir_red(  # misma arquitectura que en el entrenamiento
            n_entradas, cfg_mlp["capas"], cfg_mlp["activacion"], cfg_mlp["dropout"]  # todo sale del archivo
        )
        red.load_state_dict(paquete["state_dict"])  # falla si la arquitectura no coincide con los pesos
        modelo._red, modelo._cfg_mlp, modelo._n_entradas = red, cfg_mlp, n_entradas  # mismo estado que tras fit
        modelo._device = torch.device("cpu")  # dispositivo de inferencia (puede diferir del de entrenamiento)
        return modelo  # listo para predecir; sin resultado (no se guarda en disco)
