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
# # Bucle de entrenamiento genérico: L2, optimizadores y parada temprana (Parte B §3.2)
#
# Este módulo entrena **cualquier** red de PyTorch con **cualquier** pérdida: no sabe que es PD. La Parte D lo
# reutiliza para la LGD con su propia pérdida (Parte B §3.2). Por eso todo lo que depende de la tarea (la red, la
# pérdida, el optimizador y el generador que baraja los minilotes) llega ya construido desde fuera (inversión de
# dependencias).
#
# Contenido:
#
# - **Regularización L2** sobre los pesos (no sobre los sesgos) mediante `weight_decay`.
# - **Optimizadores:** Adam o SGD con momento, construidos desde el bloque `mlp` de la config (núcleo §5.3).
# - **Minilotes** barajados con un `np.random.Generator` (semilla de `src/common/seeds.py`).
# - **Parada temprana** con la pérdida de validación y **restauración de los mejores pesos**.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""Entrenamiento genérico con minilotes, L2 por weight_decay, Adam/SGD y parada temprana (Parte B §3.2)."""

import copy  # copia profunda del state_dict de la mejor época
import math  # isfinite e inf para la parada temprana y la revisión de pérdidas
from dataclasses import dataclass  # resultado inmutable del entrenamiento
from typing import Iterator  # tipo del generador de minilotes

import numpy as np  # permutación de índices con el Generator de la corrida
import torch  # tensores, no_grad y optimizadores
from torch import nn  # tipo nn.Module de la red y de la pérdida

from src.common.seeds import validar_entero  # validación de enteros ≥ mínimo (reutilizada sin modificar)

__all__ = [  # interfaz pública acordada (tarea B2)
    "OPTIMIZADORES",
    "grupos_de_parametros",
    "construir_optimizador",
    "iterar_minilotes",
    "epoca_entrenamiento",
    "evaluar_perdida",
    "ParadaTemprana",
    "ResultadoEntrenamiento",
    "entrenar",
]

OPTIMIZADORES = ("adam", "sgd")  # valores admitidos de mlp.optimizador (núcleo §5.3)


# %% [markdown]
# ## Regularización L2 y grupos de parámetros
#
# La penalización L2 sobre los pesos $W$ (todas las matrices de las capas `Linear`) es
#
# $$\Omega(W) = \frac{\lambda}{2}\,\lVert W \rVert_2^2 = \frac{\lambda}{2}\sum_{j} w_j^2,
# \qquad \nabla_W\,\Omega = \lambda\,W .$$
#
# El `weight_decay` $\lambda$ de `torch.optim.Adam` y `torch.optim.SGD` suma justamente $\lambda\,\theta$ al gradiente
# ($g_t \leftarrow g_t + \lambda\,\theta_{t-1}$, docstrings de PyTorch 2.14.1), así que `weight_decay = mlp.l2`
# **es** la L2 $(\lambda/2)\lVert W\rVert^2$ sumada a la pérdida. No se usa `AdamW` (decaimiento desacoplado), que
# no es L2 con Adam (Parte B §3.2 y §6). TODO(cita): Loshchilov, I. y Hutter, F., "Decoupled Weight Decay
# Regularization" (ICLR 2019), para la diferencia entre L2 y weight decay desacoplado.
#
# Los sesgos ($b$, tensores de una dimensión) **no** se penalizan (decisión de Andrés): se arman dos grupos de
# parámetros, pesos (`ndim > 1`) con `weight_decay = λ` y sesgos (`ndim == 1`) con `weight_decay = 0`.

# %%
def grupos_de_parametros(red: nn.Module, l2: float) -> list[dict]:
    """Dos grupos para el optimizador: pesos (ndim > 1) con ``weight_decay = l2`` y sesgos (ndim == 1) con 0."""
    pesos = [p for p in red.parameters() if p.ndim > 1]  # matrices W de las capas
    sesgos = [p for p in red.parameters() if p.ndim == 1]  # vectores b de las capas
    return [  # un grupo por tratamiento de la L2
        {"params": pesos, "weight_decay": l2},  # L2 = (λ/2)‖W‖² → gradiente λW
        {"params": sesgos, "weight_decay": 0.0},  # sesgos sin penalizar
    ]


# %% [markdown]
# ## Optimizadores (tal como los aplica PyTorch 2.14.1)
#
# En ambos, $\gamma$ = `mlp.lr`, $g_t = \nabla_\theta f_t(\theta_{t-1}) + \lambda\,\theta_{t-1}$ (la L2 de arriba,
# con $\lambda = 0$ en los sesgos).
#
# **SGD con momento** ($\mu$ = `mlp.momento`; amortiguación $\tau = 0$ y sin Nesterov, valores por defecto de
# PyTorch):
#
# $$b_1 = g_1, \qquad b_t = \mu\,b_{t-1} + (1-\tau)\,g_t \;(t > 1), \qquad \theta_t = \theta_{t-1} - \gamma\,b_t .$$
#
# **Adam** con los valores por defecto de PyTorch, $\beta_1 = 0.9$, $\beta_2 = 0.999$ y $\epsilon = 10^{-8}$
# (firma de `torch.optim.Adam`; no se cambian), $m_0 = v_0 = 0$:
#
# $$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2,$$
#
# $$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad
# \theta_t = \theta_{t-1} - \gamma\,\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} .$$
#
# La división entre $1-\beta^t$ es la **corrección de sesgo**: como $m_0 = v_0 = 0$, en los primeros pasos $m_t$ y
# $v_t$ están sesgados hacia 0. Las ecuaciones son las de la docstring de `torch.optim.Adam` y `torch.optim.SGD`
# (`help(torch.optim.Adam)`). TODO(cita): Kingma, D. P. y Ba, J., "Adam: A Method for Stochastic Optimization"
# (ICLR 2015). `mlp.momento` solo lo usa SGD; Adam lo ignora.

# %%
def construir_optimizador(red: nn.Module, cfg_mlp: dict, agrupar=grupos_de_parametros) -> torch.optim.Optimizer:
    """'adam' → ``torch.optim.Adam(lr)`` (nunca AdamW); 'sgd' → ``torch.optim.SGD(lr, momentum=momento)``."""
    grupos = agrupar(red, cfg_mlp["l2"])  # pesos con L2 y sesgos sin L2
    nombre = cfg_mlp["optimizador"]  # adam | sgd
    if nombre == "adam":  # Adam con L2 acoplada (weight_decay suma λθ al gradiente)
        return torch.optim.Adam(grupos, lr=cfg_mlp["lr"])  # betas y eps por defecto de PyTorch
    if nombre == "sgd":  # SGD con momento clásico (sin Nesterov)
        return torch.optim.SGD(grupos, lr=cfg_mlp["lr"], momentum=cfg_mlp["momento"])  # τ = 0 por defecto
    raise ValueError(f"optimizador debe ser uno de {OPTIMIZADORES}; se recibió {nombre!r}")  # desconocido


# %% [markdown]
# ## Minilotes
#
# En cada época se baraja $\{0, \dots, n-1\}$ con el `np.random.Generator` de la corrida y se recorre en trozos de
# `batch` índices; el último trozo puede ser más corto (ningún crédito se omite ni se repite). Como el generador
# sale de `fijar_semillas(seed)`, el orden es reproducible.

# %%
def iterar_minilotes(n: int, batch: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Índices barajados de ``0..n-1`` en trozos de ``batch`` (el último puede ser más corto)."""
    orden = rng.permutation(n)  # permutación reproducible con el rng de la corrida
    for inicio in range(0, n, batch):  # inicio de cada minilote
        yield orden[inicio : inicio + batch]  # el último trozo incluye el resto


# %% [markdown]
# ## Una época de entrenamiento
#
# Para cada minilote $B$: se pone la red en modo entrenamiento (dropout activo), se calcula la pérdida
# $\ell_B$, se retropropaga y se da un paso del optimizador. La pérdida de entrenamiento de la época es el
# promedio ponderado por tamaño de minilote,
#
# $$\bar\ell_{\text{train}} = \frac{1}{n}\sum_{B} |B|\,\ell_B ,$$
#
# que coincide con la media por observación cuando la pérdida promedia dentro del minilote (`reduction='mean'`).

# %%
def epoca_entrenamiento(red, perdida, optimizador, X, y, batch, rng) -> float:
    """Una pasada por todos los minilotes con retropropagación; devuelve la pérdida media ponderada de la época."""
    red.train()  # dropout activo durante el entrenamiento
    total = 0.0  # suma de |B|·ℓ_B
    n = X.shape[0]  # número de observaciones
    for indices in iterar_minilotes(n, batch, rng):  # minilotes barajados
        idx = torch.as_tensor(indices, device=X.device)  # índices en el dispositivo de los datos
        optimizador.zero_grad()  # borra los gradientes del paso anterior
        valor = perdida(red(X[idx]), y[idx])  # pérdida del minilote
        valor.backward()  # retropropagación
        optimizador.step()  # actualización de los parámetros
        total += valor.item() * len(indices)  # acumula ponderando por tamaño
    return total / n  # pérdida media de la época


# %% [markdown]
# ## Pérdida de evaluación
#
# La pérdida de validación se mide en una sola pasada con la red en modo evaluación (`eval()`: el dropout es la
# identidad) y sin construir el grafo (`no_grad`).

# %%
def evaluar_perdida(red, perdida, X, y) -> float:
    """Pérdida sobre (X, y) en modo evaluación y sin gradientes."""
    red.eval()  # dropout apagado
    with torch.no_grad():  # sin grafo ni gradientes
        return float(perdida(red(X), y))  # pérdida como float de Python


# %% [markdown]
# ## Parada temprana
#
# Regla (decisión de Andrés): con $L_e$ la pérdida de validación de la época $e$ (épocas numeradas desde 1) y
# $L^{*} = \min_{e' < e} L_{e'}$ la mejor hasta entonces ($L^{*} = +\infty$ al inicio),
#
# - hay **mejora** si $L_e < L^{*}$ (estrictamente: `min_delta = 0`, un empate no cuenta); entonces
#   $L^{*} \leftarrow L_e$, se guarda una copia profunda de los pesos y la cuenta de épocas sin mejora vuelve a 0;
# - si no hay mejora, la cuenta sube en 1, y al llegar a `paciencia` se detiene el entrenamiento.
#
# Al terminar (por paciencia o por `max_epocas`) se **restauran los pesos de la mejor época**. TODO(cita): referencia
# para la parada temprana (p. ej. Prechelt, "Early Stopping — But When?"), por verificar.

# %%
class ParadaTemprana:
    """Parada temprana con mejora estricta (min_delta = 0) y copia profunda de los mejores pesos."""

    def __init__(self, paciencia: int):
        self.paciencia = validar_entero(paciencia, "paciencia", minimo=1)  # épocas sin mejora toleradas (≥ 1)
        self.mejor_perdida = math.inf  # L* inicial: cualquier pérdida finita es mejora
        self.mejor_epoca = 0  # 0 = todavía no hay mejor época
        self._sin_mejora = 0  # épocas seguidas sin mejora
        self._mejor_estado = None  # state_dict de la mejor época

    def actualizar(self, perdida_val: float, epoca: int, red: nn.Module) -> bool:
        """Registra la pérdida de la ``epoca``; devuelve True si hay que detener el entrenamiento."""
        if perdida_val < self.mejor_perdida:  # mejora estricta (el empate no cuenta)
            self.mejor_perdida = perdida_val  # nueva mejor pérdida
            self.mejor_epoca = epoca  # y su época
            self._sin_mejora = 0  # reinicia la cuenta de paciencia
            self._mejor_estado = copy.deepcopy(red.state_dict())  # copia (el optimizador modifica en el sitio)
            return False  # tras una mejora nunca se detiene
        self._sin_mejora += 1  # una época más sin mejora
        return self._sin_mejora >= self.paciencia  # detener al agotar la paciencia

    def restaurar(self, red: nn.Module) -> None:
        """Carga en ``red`` los pesos de la mejor época."""
        if self._mejor_estado is None:  # nunca hubo una época registrada
            raise RuntimeError("No hay pesos que restaurar: actualizar() no registró ninguna época")  # uso inválido
        red.load_state_dict(self._mejor_estado)  # vuelve a los mejores pesos


# %% [markdown]
# ## Resultado del entrenamiento
#
# `historial` tiene una entrada `{"epoca", "perdida_train", "perdida_val"}` por época corrida; `mejor_epoca` y
# `mejor_perdida_val` son los de la parada temprana; `epocas_corridas = len(historial)`.

# %%
@dataclass(frozen=True)
class ResultadoEntrenamiento:
    """Historial por época, mejor época, épocas corridas y mejor pérdida de validación."""

    historial: list  # [{"epoca", "perdida_train", "perdida_val"}, ...]
    mejor_epoca: int  # época (desde 1) con la menor pérdida de validación
    epocas_corridas: int  # épocas efectivamente entrenadas
    mejor_perdida_val: float  # pérdida de validación de la mejor época


# %% [markdown]
# ## Comprobaciones del bucle
#
# - **Formas:** la salida de la red y el objetivo deben tener la misma forma. Si no, pérdidas como `MSELoss`
#   difundirían $(n, 1)$ contra $(n,)$ a una matriz $(n, n)$ sin error. Se comprueba con una fila en modo
#   evaluación (no consume aleatoriedad del dropout).
# - **Finitud:** una pérdida NaN o $\pm\infty$ en train o en val detiene todo con `FloatingPointError`, nombrando la
#   época (decisión de Andrés).

# %%
def validar_formas(red: nn.Module, X: torch.Tensor, y: torch.Tensor, nombre: str) -> None:
    """ValueError si X e y tienen distinto número de filas o si la salida de la red no tiene la forma de y."""
    if X.shape[0] != y.shape[0]:  # mismas observaciones en X e y
        raise ValueError(f"{nombre}: X tiene {X.shape[0]} filas e y tiene {y.shape[0]}")  # filas distintas
    red.eval()  # sin dropout: la comprobación no consume aleatoriedad
    with torch.no_grad():  # sin grafo
        forma_salida = tuple(red(X[:1]).shape[1:])  # forma de la salida por observación
    if forma_salida != tuple(y.shape[1:]):  # debe coincidir con la del objetivo
        raise ValueError(  # formas incompatibles
            f"{nombre}: la salida de la red tiene forma por fila {forma_salida} y el objetivo {tuple(y.shape[1:])}"
        )


def verificar_finita(valor: float, nombre: str, epoca: int) -> None:
    """FloatingPointError si ``valor`` es NaN o ±inf, indicando la época."""
    if not math.isfinite(valor):  # NaN o ±inf
        raise FloatingPointError(f"Pérdida de {nombre} no finita ({valor}) en la época {epoca}")  # aborta


# %% [markdown]
# ## Bucle completo
#
# Para $e = 1, \dots,$ `max_epocas`: una época de entrenamiento, la pérdida de validación, la comprobación de
# finitud, el registro en el historial y la consulta a la parada temprana. Al salir, la red queda con los mejores
# pesos. `X` e `y` son tensores ya en el dispositivo de la red.

# %%
def entrenar(
    red,
    perdida,
    optimizador,
    X_tr,
    y_tr,
    X_val,
    y_val,
    *,
    batch,
    max_epocas,
    paciencia,
    rng,
    parada_cls=ParadaTemprana,
) -> ResultadoEntrenamiento:
    """Entrena ``red`` con minilotes y parada temprana; al terminar deja en ``red`` los pesos de la mejor época."""
    batch = validar_entero(batch, "batch", minimo=1)  # tamaño de minilote ≥ 1
    max_epocas = validar_entero(max_epocas, "max_epocas", minimo=1)  # al menos una época
    validar_formas(red, X_tr, y_tr, "train")  # salida y objetivo compatibles en train
    validar_formas(red, X_val, y_val, "val")  # y en validación
    parada = parada_cls(paciencia)  # estado de la parada temprana
    historial = []  # una entrada por época
    for epoca in range(1, max_epocas + 1):  # épocas numeradas desde 1
        perdida_train = epoca_entrenamiento(red, perdida, optimizador, X_tr, y_tr, batch, rng)  # entrena
        perdida_val = evaluar_perdida(red, perdida, X_val, y_val)  # mide en validación
        verificar_finita(perdida_train, "train", epoca)  # NaN/inf en train → error
        verificar_finita(perdida_val, "val", epoca)  # NaN/inf en val → error
        historial.append({"epoca": epoca, "perdida_train": perdida_train, "perdida_val": perdida_val})  # registra
        if parada.actualizar(perdida_val, epoca, red):  # ¿se agotó la paciencia?
            break  # detener
    parada.restaurar(red)  # la red queda con los mejores pesos
    return ResultadoEntrenamiento(  # resumen inmutable
        historial=historial,  # historial completo
        mejor_epoca=parada.mejor_epoca,  # mejor época
        epocas_corridas=len(historial),  # épocas corridas
        mejor_perdida_val=parada.mejor_perdida,  # mejor pérdida de validación
    )
