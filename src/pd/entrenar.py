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
from dataclasses import dataclass  # frozen: los campos del resultado no se reasignan

import numpy as np  # permutación de índices con el Generator de la corrida
import torch  # tensores, no_grad y optimizadores
from torch import nn  # tipo nn.Module de la red y de la pérdida

from src.common.seeds import validar_entero  # validación de enteros ≥ mínimo (reutilizada sin modificar)

__all__ = [  # interfaz pública acordada (tarea B2); lo demás es interno
    "OPTIMIZADORES", "grupos_de_parametros", "construir_optimizador",  # config → optimizador
    "iterar_minilotes", "epoca_entrenamiento", "evaluar_perdida",  # piezas del bucle (la Parte D las reutiliza)
    "ParadaTemprana", "ResultadoEntrenamiento", "entrenar",  # parada temprana y bucle completo
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
# no es L2 con Adam (Parte B §3.2 y §6). Referencia: Loshchilov, I. y Hutter, F., "Decoupled Weight Decay
# Regularization" (`loshchilov2017` en blog/references.bib, verificada en A8), para la diferencia entre L2 y weight decay
# desacoplado.
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
# (`help(torch.optim.Adam)`). Referencia: Kingma, D. P. y Ba, J., "Adam: A Method for Stochastic Optimization"
# (`kingma2014` en blog/references.bib, verificada en A8). `mlp.momento` solo lo usa SGD; Adam lo ignora.

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
# En cada época se baraja $\{0, \dots, n-1\}$ con el `np.random.Generator` de la corrida y se parte en trozos de
# `batch` índices; el último trozo puede ser más corto (ningún crédito se omite ni se repite). Como el generador
# sale de `fijar_semillas(seed)`, el orden es reproducible. La permutación se copia al dispositivo **una vez por
# época** y los trozos son vistas de ese tensor (`torch.split`).

# %%
def iterar_minilotes(n: int, batch: int, rng: np.random.Generator, device) -> tuple[torch.Tensor, ...]:
    """Índices barajados de ``0..n-1`` en ``device``, en trozos de ``batch`` (el último puede ser más corto)."""
    orden = torch.as_tensor(rng.permutation(n), device=torch.device(device))  # el orden sale del rng de numpy
    return torch.split(orden, batch)  # vistas sin copia; el último trozo lleva el resto


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
#
# El orden de los minilotes sale de `iterar_minilotes` (el rng de numpy de la corrida). La suma se acumula como
# tensor, con una sola conversión a `float` al final (sin sincronizar con el dispositivo en cada minilote).

# %%
def epoca_entrenamiento(red, perdida, optimizador, X, y, batch, rng) -> float:
    """Una pasada por todos los minilotes con retropropagación; devuelve la pérdida media ponderada de la época."""
    red.train()  # dropout activo durante el entrenamiento
    n = X.shape[0]  # número de observaciones
    total = torch.zeros((), device=X.device)  # Σ |B|·ℓ_B, acumulada sin salir del dispositivo
    for idx in iterar_minilotes(n, batch, rng, X.device):  # índices ya en el dispositivo de los datos
        optimizador.zero_grad()  # sin esto, los gradientes de minilotes anteriores se sumarían
        valor = perdida(red(X[idx]), y[idx])  # ℓ_B: media dentro del minilote
        valor.backward()  # gradiente de ℓ_B (más la L2, que añade el optimizador)
        optimizador.step()  # un paso de Adam o SGD
        total = total + valor.detach() * len(idx)  # pondera por |B|; detach para no retener el grafo
    return float(total) / n  # única conversión a float: media por observación


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
# Al terminar (por paciencia o por `max_epocas`) se **restauran los pesos de la mejor época**. Referencia para la parada
# temprana: Prechelt, L. (1998). Early Stopping - But When? En *Neural Networks: Tricks of the Trade* (LNCS 1524,
# pp. 55-69). Springer. https://doi.org/10.1007/3-540-49430-8_3 (verificada en Crossref el 28-oct).

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
# ## Comprobación de formas
#
# Antes de entrenar, en cada conjunto (train y val):
#
# - debe haber al menos una fila (con $n = 0$ la media de la época sería $0/0$);
# - $X$ e $y$ deben tener el mismo número de filas;
# - la salida de la red y el objetivo deben tener la misma forma por fila. Si no, pérdidas como `MSELoss`
#   difundirían $(n, 1)$ contra $(n,)$ a una matriz $(n, n)$ sin error.
#
# La forma de la salida se mira con una sola fila en modo evaluación, para no consumir aleatoriedad del dropout.
# Todo error es `ValueError` y nombra el conjunto.

# %%
def validar_formas(red: nn.Module, X: torch.Tensor, y: torch.Tensor, nombre: str) -> None:
    """ValueError nombrando el conjunto: sin filas, filas distintas en X e y, o salida con otra forma que y."""
    if X.shape[0] == 0:  # sin filas no hay pérdida que promediar
        raise ValueError(f"{nombre}: el conjunto no tiene filas")  # conjunto vacío
    if X.shape[0] != y.shape[0]:  # cada fila de X necesita su objetivo
        raise ValueError(f"{nombre}: X tiene {X.shape[0]} filas e y tiene {y.shape[0]}")  # filas distintas
    red.eval()  # sin dropout: la comprobación no altera el estado aleatorio del entrenamiento
    with torch.no_grad():  # es solo una inspección: no hace falta grafo
        forma_salida = tuple(red(X[:1]).shape[1:])  # con una fila basta para conocer la forma por observación
    if forma_salida != tuple(y.shape[1:]):  # la pérdida necesita formas idénticas para no difundir
        raise ValueError(  # formas incompatibles
            f"{nombre}: la salida de la red tiene forma por fila {forma_salida} "  # lo que produce la red
            f"y el objetivo {tuple(y.shape[1:])}"  # frente a lo que se le pide
        )


# %% [markdown]
# ## Comprobación de finitud
#
# Regla (decisión de Andrés): una pérdida NaN o $\pm\infty$ detiene el entrenamiento con `FloatingPointError`;
# el mensaje nombra la fase (`train` o `val`) y la época. Seguir entrenando con una pérdida no finita solo
# propagaría NaN a los pesos.

# %%
def verificar_finita(valor: float, nombre: str, epoca: int) -> None:
    """FloatingPointError si ``valor`` es NaN o ±inf, nombrando la fase y la época."""
    if not math.isfinite(valor):  # NaN o ±inf
        raise FloatingPointError(f"Pérdida de {nombre} no finita ({valor}) en la época {epoca}")  # fase y época


# %% [markdown]
# ## Bucle completo
#
# Para $e = 1, \dots,$ `max_epocas`: una época de entrenamiento y su comprobación de finitud (antes de medir val,
# para nombrar la fase donde nació el problema), la pérdida de validación y su comprobación, el registro en el
# historial y la consulta a la parada temprana. Al salir (por paciencia o por `max_epocas`), la red queda con los
# mejores pesos. `X` e `y` son tensores ya en el dispositivo de la red.

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
    batch = validar_entero(batch, "batch", minimo=1)  # un minilote vacío no daría un paso
    max_epocas = validar_entero(max_epocas, "max_epocas", minimo=1)  # al menos una época para tener mejores pesos
    validar_formas(red, X_tr, y_tr, "train")  # falla antes de gastar tiempo entrenando
    validar_formas(red, X_val, y_val, "val")  # la parada temprana depende de esta pérdida
    parada = parada_cls(paciencia)  # inyectable: la Parte D puede usar otra regla
    historial = []  # una entrada por época
    for epoca in range(1, max_epocas + 1):  # épocas numeradas desde 1
        perdida_train = epoca_entrenamiento(red, perdida, optimizador, X_tr, y_tr, batch, rng)  # entrena una época
        verificar_finita(perdida_train, "train", epoca)  # antes de medir val: nombra la fase de origen
        perdida_val = evaluar_perdida(red, perdida, X_val, y_val)  # criterio de la parada temprana
        verificar_finita(perdida_val, "val", epoca)  # una val no finita invalidaría la comparación
        historial.append({"epoca": epoca, "perdida_train": perdida_train, "perdida_val": perdida_val})  # registra
        if parada.actualizar(perdida_val, epoca, red):  # ¿se agotó la paciencia?
            break  # las épocas restantes ya no mejorarían según la regla
    parada.restaurar(red)  # también al llegar a max_epocas: la última época no tiene por qué ser la mejor
    return ResultadoEntrenamiento(  # frozen: no se reasignan los campos; la lista del historial sí es mutable
        historial=historial,  # una entrada por época: número de época, pérdida de train y de val
        mejor_epoca=parada.mejor_epoca,  # la de los pesos que quedaron en la red
        epocas_corridas=len(historial),  # puede ser < max_epocas si paró por paciencia
        mejor_perdida_val=parada.mejor_perdida,  # criterio con que se eligieron esos pesos
    )
