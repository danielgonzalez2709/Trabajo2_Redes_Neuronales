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
from numbers import Real  # comprobación de que un valor es numérico
from pathlib import Path  # rutas de guardado

import numpy as np  # entradas y salidas del contrato ModeloPD
import torch  # tensores, guardado y carga
from scipy.special import expit  # σ(z) estable en float64
from torch import nn  # capas de la red y pérdida

from src.common.seeds import fijar_semillas  # semillas de torch, random y numpy (reutilizada sin modificar)
from src.common.torch_utils import (  # utilidades compartidas de PyTorch
    DEVICES_VALIDOS,  # auto | cpu | cuda
    a_numpy,  # tensor → numpy float64 (n,)
    a_tensor,  # numpy → tensor float32
    activar_determinismo,  # modo determinista en CPU
    resolver_device,  # mlp.device → torch.device
)
from src.pd.entrenar import OPTIMIZADORES, construir_optimizador, entrenar  # bucle genérico y optimizadores

__all__ = [  # interfaz pública acordada (tarea B2)
    "ACTIVACIONES",
    "CLAVES_MLP",
    "validar_cfg_mlp",
    "construir_red",
    "contar_parametros",
    "construir_perdida",
    "MLPPD",
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
# ## Validación del bloque `mlp`
#
# Reglas de validez de nuestra config (no son constantes del dominio): exactamente las claves de `CLAVES_MLP`;
# `capas` lista no vacía de enteros $\ge 1$ (ni flotantes ni bool); `dropout` $\in [0, 1)$; `l2` $\ge 0$;
# `lr` $> 0$; `momento` $\ge 0$; `pos_weight` $> 0$; `batch`, `max_epocas` y `paciencia` enteros $\ge 1$;
# `activacion`, `optimizador` y `device` dentro de sus valores admitidos. Cualquier falla → `ValueError`.

# %%
def _es_entero(valor) -> bool:
    """True si ``valor`` es un entero de Python o de numpy (los bool no cuentan)."""
    return isinstance(valor, (int, np.integer)) and not isinstance(valor, (bool, np.bool_))  # int sin bool


def _es_real(valor) -> bool:
    """True si ``valor`` es un número real (los bool no cuentan)."""
    return isinstance(valor, Real) and not isinstance(valor, (bool, np.bool_))  # número sin bool


def _validar_claves(cfg_mlp: dict) -> None:
    """ValueError si falta o sobra alguna clave respecto de CLAVES_MLP."""
    faltan = set(CLAVES_MLP) - set(cfg_mlp)  # claves obligatorias ausentes
    sobran = set(cfg_mlp) - set(CLAVES_MLP)  # claves desconocidas
    if faltan or sobran:  # el bloque debe ser exacto
        raise ValueError(f"Bloque mlp inválido: faltan {sorted(faltan)}, sobran {sorted(sobran)}")  # claves


def _validar_capas(capas) -> None:
    """ValueError si ``capas`` no es una lista no vacía de enteros ≥ 1."""
    if not isinstance(capas, (list, tuple)) or len(capas) == 0:  # debe haber al menos una capa oculta
        raise ValueError(f"mlp.capas debe ser una lista no vacía de enteros; se recibió {capas!r}")  # vacía
    if not all(_es_entero(c) and c >= 1 for c in capas):  # cada anchura entera y positiva
        raise ValueError(f"mlp.capas debe contener enteros >= 1; se recibió {capas!r}")  # anchura inválida


def _validar_rango(cfg_mlp: dict, clave: str, condicion, descripcion: str) -> None:
    """ValueError si ``cfg_mlp[clave]`` no es un número real que cumpla ``condicion``."""
    valor = cfg_mlp[clave]  # valor a revisar
    if not _es_real(valor) or not condicion(valor):  # numérico y dentro del rango
        raise ValueError(f"mlp.{clave} debe ser {descripcion}; se recibió {valor!r}")  # fuera de rango


def _validar_entero_positivo(cfg_mlp: dict, clave: str) -> None:
    """ValueError si ``cfg_mlp[clave]`` no es un entero ≥ 1."""
    valor = cfg_mlp[clave]  # valor a revisar
    if not _es_entero(valor) or valor < 1:  # entero y positivo
        raise ValueError(f"mlp.{clave} debe ser un entero >= 1; se recibió {valor!r}")  # inválido


def _validar_opcion(cfg_mlp: dict, clave: str, opciones) -> None:
    """ValueError si ``cfg_mlp[clave]`` no está en ``opciones``."""
    if cfg_mlp[clave] not in opciones:  # valor categórico admitido
        raise ValueError(f"mlp.{clave} debe ser uno de {tuple(opciones)}; se recibió {cfg_mlp[clave]!r}")  # inválido


def validar_cfg_mlp(cfg_mlp: dict) -> dict:
    """Valida el bloque ``mlp`` de PD y devuelve una copia profunda; cualquier falla → ValueError."""
    _validar_claves(cfg_mlp)  # claves exactas
    _validar_capas(cfg_mlp["capas"])  # arquitectura
    _validar_rango(cfg_mlp, "dropout", lambda v: 0 <= v < 1, "un número en [0, 1)")  # p de dropout
    _validar_rango(cfg_mlp, "l2", lambda v: v >= 0, "un número >= 0")  # λ de la L2
    _validar_rango(cfg_mlp, "lr", lambda v: v > 0, "un número > 0")  # tasa de aprendizaje
    _validar_rango(cfg_mlp, "momento", lambda v: v >= 0, "un número >= 0")  # momento de SGD
    _validar_rango(cfg_mlp, "pos_weight", lambda v: v > 0, "un número > 0")  # peso de la clase positiva
    for clave in ("batch", "max_epocas", "paciencia"):  # enteros positivos
        _validar_entero_positivo(cfg_mlp, clave)  # cada uno ≥ 1
    _validar_opcion(cfg_mlp, "activacion", ACTIVACIONES)  # relu | leaky_relu | tanh
    _validar_opcion(cfg_mlp, "optimizador", OPTIMIZADORES)  # adam | sgd
    _validar_opcion(cfg_mlp, "device", DEVICES_VALIDOS)  # auto | cpu | cuda
    return copy.deepcopy(cfg_mlp)  # copia: nadie muta la config recibida


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
# `fit` lee `cfg["mlp"]`, lo valida (sin mutarlo), resuelve el dispositivo, activa el determinismo en CPU, fija las
# semillas con `fijar_semillas(seed)` **antes** de construir la red (el `Generator` de numpy que devuelve es el que
# baraja los minilotes) y llama al bucle genérico. Las fábricas (`construir_red`, `construir_optimizador`,
# `entrenar`) se reciben por argumento con valor por defecto (inversión de dependencias).
#
# `predict_logit` hace una sola pasada en `eval()` y `no_grad`; `predict_proba` $= \sigma(\text{logit})$. Ambas
# devuelven numpy `float64` de forma $(n,)$.
#
# `guardar` escribe `{"state_dict" (en CPU), "config" (bloque mlp), "n_entradas"}` con `torch.save`; `cargar` lo lee
# con `torch.load(weights_only=True)` y reconstruye la red en CPU, sin necesitar la config.

# %%
class MLPPD:
    """MLP de PD que cumple el protocolo ``ModeloPD`` del núcleo §5.3."""

    def __init__(
        self,
        construir_red=construir_red,  # fábrica de la red
        construir_optimizador=construir_optimizador,  # fábrica del optimizador
        entrenar=entrenar,  # bucle de entrenamiento
    ):
        self._construir_red = construir_red  # se guarda la fábrica de la red
        self._construir_optimizador = construir_optimizador  # la del optimizador
        self._entrenar = entrenar  # y el bucle
        self._red = None  # red entrenada (None = sin entrenar)
        self._cfg_mlp = None  # bloque mlp validado
        self._n_entradas = None  # número de columnas de X
        self._device = None  # dispositivo de la red
        self._resultado = None  # ResultadoEntrenamiento del fit

    # --- validaciones de datos -------------------------------------------------------------------------------
    def _verificar_entrenado(self) -> None:
        """RuntimeError si el modelo aún no tiene red (ni fit ni cargar)."""
        if self._red is None:  # sin red
            raise RuntimeError("El modelo no está entrenado: llame a fit() o a cargar() antes")  # uso inválido

    def _verificar_columnas(self, X) -> None:
        """ValueError si X no es 2-D con ``n_entradas`` columnas."""
        forma = np.shape(X)  # forma sin copiar
        if len(forma) != 2 or forma[1] != self._n_entradas:  # matriz con las columnas de fit
            raise ValueError(f"X debe tener forma (n, {self._n_entradas}); se recibió {forma}")  # columnas

    @staticmethod
    def _verificar_objetivo(y, n: int, nombre: str) -> None:
        """ValueError si y no es un vector 0/1 de longitud ``n``."""
        y = np.asarray(y)  # arreglo de numpy
        if y.shape != (n,):  # vector con una etiqueta por fila
            raise ValueError(f"{nombre} debe tener forma ({n},); se recibió {y.shape}")  # forma
        if not np.all((y == 0) | (y == 1)):  # solo etiquetas 0/1
            raise ValueError(f"{nombre} debe contener solo 0 y 1")  # etiquetas inválidas

    # --- entrenamiento -----------------------------------------------------------------------------------------
    def fit(self, X_tr, y_tr, X_val, y_val, cfg: dict, seed: int) -> "MLPPD":
        """Entrena con ``cfg["mlp"]`` y la semilla ``seed``; devuelve el propio modelo."""
        cfg_mlp = validar_cfg_mlp(cfg["mlp"])  # copia validada (no muta cfg)
        if np.ndim(X_tr) != 2:  # matriz de diseño
            raise ValueError(f"X_tr debe ser 2-D; se recibió la forma {np.shape(X_tr)}")  # forma
        self._n_entradas = int(np.shape(X_tr)[1])  # columnas del preprocesador
        self._verificar_columnas(X_val)  # val con las mismas columnas
        self._verificar_objetivo(y_tr, np.shape(X_tr)[0], "y_tr")  # etiquetas de train
        self._verificar_objetivo(y_val, np.shape(X_val)[0], "y_val")  # etiquetas de val
        device = resolver_device(cfg_mlp["device"])  # auto | cpu | cuda
        activar_determinismo(device)  # determinista en CPU (núcleo §7.1)
        rng = fijar_semillas(seed)  # torch y random sembrados; rng de numpy para los minilotes
        red = self._construir_red(  # la red se construye después de sembrar torch
            self._n_entradas, cfg_mlp["capas"], cfg_mlp["activacion"], cfg_mlp["dropout"]
        ).to(device)  # al dispositivo elegido
        optimizador = self._construir_optimizador(red, cfg_mlp)  # Adam o SGD con L2 en los pesos
        perdida = construir_perdida(cfg_mlp["pos_weight"], device)  # BCE sobre logits
        self._resultado = self._entrenar(  # bucle genérico con parada temprana
            red,  # red
            perdida,  # pérdida
            optimizador,  # optimizador
            a_tensor(X_tr, device),  # X de train (n, d)
            a_tensor(y_tr, device, columna=True),  # y de train (n, 1), como la salida
            a_tensor(X_val, device),  # X de val
            a_tensor(y_val, device, columna=True),  # y de val (n, 1)
            batch=cfg_mlp["batch"],  # minilote
            max_epocas=cfg_mlp["max_epocas"],  # tope de épocas
            paciencia=cfg_mlp["paciencia"],  # paciencia
            rng=rng,  # orden de los minilotes
        )
        self._red, self._cfg_mlp, self._device = red, cfg_mlp, device  # estado del modelo entrenado
        return self  # encadenable

    # --- predicción --------------------------------------------------------------------------------------------
    def predict_logit(self, X) -> np.ndarray:
        """Logit log(p / (1 − p)) de cada fila, numpy float64 (n,); una sola pasada en eval() y no_grad."""
        self._verificar_entrenado()  # requiere red
        self._verificar_columnas(X)  # columnas de fit
        self._red.eval()  # dropout apagado al predecir
        with torch.no_grad():  # sin grafo
            return a_numpy(self._red(a_tensor(X, self._device)))  # (n, 1) → float64 (n,)

    def predict_proba(self, X) -> np.ndarray:
        """PD = σ(logit), numpy float64 (n,)."""
        return expit(self.predict_logit(X))  # σ(z) estable en float64

    # --- información -------------------------------------------------------------------------------------------
    def n_parametros(self) -> int:
        """Parámetros entrenables de la red."""
        self._verificar_entrenado()  # requiere red
        return contar_parametros(self._red)  # Σ entrenables

    @property
    def resultado(self):
        """ResultadoEntrenamiento del último fit (no se guarda en disco)."""
        if self._resultado is None:  # sin fit en esta sesión
            raise RuntimeError("No hay resultado de entrenamiento: el modelo no se entrenó con fit()")  # uso inválido
        return self._resultado  # historial y mejor época

    # --- persistencia ------------------------------------------------------------------------------------------
    def guardar(self, ruta) -> None:
        """Guarda ``{"state_dict" en CPU, "config" (bloque mlp), "n_entradas"}`` con ``torch.save``."""
        self._verificar_entrenado()  # requiere red
        estado = {k: v.detach().cpu() for k, v in self._red.state_dict().items()}  # pesos en CPU
        paquete = {  # contenido del archivo (núcleo de la tarea B2)
            "state_dict": estado,  # pesos
            "config": copy.deepcopy(self._cfg_mlp),  # bloque mlp validado
            "n_entradas": self._n_entradas,  # columnas de X
        }
        torch.save(paquete, Path(ruta))  # serializa a disco

    @classmethod
    def cargar(cls, ruta) -> "MLPPD":
        """Reconstruye el modelo desde ``ruta`` (en CPU) sin necesitar la config."""
        paquete = torch.load(Path(ruta), map_location="cpu", weights_only=True)  # solo tensores y tipos simples
        cfg_mlp = validar_cfg_mlp(paquete["config"])  # bloque mlp guardado
        modelo = cls()  # fábricas por defecto
        red = modelo._construir_red(  # misma arquitectura
            paquete["n_entradas"], cfg_mlp["capas"], cfg_mlp["activacion"], cfg_mlp["dropout"]
        )
        red.load_state_dict(paquete["state_dict"])  # pesos guardados
        modelo._red, modelo._cfg_mlp = red, cfg_mlp  # estado del modelo
        modelo._n_entradas = int(paquete["n_entradas"])  # columnas esperadas
        modelo._device = torch.device("cpu")  # predicción en CPU
        return modelo  # listo para predecir
