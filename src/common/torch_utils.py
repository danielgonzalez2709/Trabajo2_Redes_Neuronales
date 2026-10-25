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
# # Utilidades de PyTorch: dispositivo, determinismo y conversión de arreglos (núcleo §5.3 y §7.1)
#
# Piezas pequeñas que comparten los MLP de PD (Parte B) y de LGD (Parte D):
#
# - **Dispositivo:** la clave `mlp.device` de la config vale `auto`, `cpu` o `cuda` (núcleo §5.3).
# - **Determinismo:** misma semilla y mismo dispositivo → mismas métricas; en CPU se exige
#   `torch.use_deterministic_algorithms(True)` (núcleo §7.1).
# - **Conversión:** los modelos reciben y devuelven arreglos de numpy (contrato `ModeloPD`), pero la red trabaja con
#   tensores `float32`. Aquí se hace el paso de ida y de vuelta y se rechazan los valores no finitos.
#
# El archivo está en formato *percent* de Jupytext: se lee como notebook, pero sigue siendo un módulo importable
# sin efectos al importar.

# %%
"""Dispositivo (auto|cpu|cuda), modo determinista en CPU y conversión numpy ↔ tensor (núcleo §5.3, §7.1)."""

import numpy as np  # arreglos de entrada y de salida
import torch  # tensores y dispositivos

__all__ = ["DEVICES_VALIDOS", "resolver_device", "activar_determinismo", "a_tensor", "a_numpy"]  # interfaz pública

DEVICES_VALIDOS = ("auto", "cpu", "cuda")  # valores admitidos de mlp.device (núcleo §5.3)


# %% [markdown]
# ## Dispositivo
#
# Regla: `auto` → `cuda` si hay GPU y si no `cpu`; `cpu` y `cuda` se respetan tal cual. Pedir `cuda` sin GPU es un
# error (`RuntimeError`): no se degrada a CPU en silencio, porque los resultados registran el `device` (núcleo §7.1).

# %%
def resolver_device(nombre: str) -> torch.device:
    """Traduce ``mlp.device`` a un ``torch.device``; nombre desconocido → ValueError; 'cuda' sin GPU → RuntimeError."""
    if nombre not in DEVICES_VALIDOS:  # solo auto|cpu|cuda
        raise ValueError(f"device debe ser uno de {DEVICES_VALIDOS}; se recibió {nombre!r}")  # nombre inválido
    hay_gpu = torch.cuda.is_available()  # ¿hay una GPU utilizable?
    if nombre == "auto":  # elegir según el hardware
        return torch.device("cuda" if hay_gpu else "cpu")  # GPU si hay, si no CPU
    if nombre == "cuda" and not hay_gpu:  # se pidió GPU explícitamente y no hay
        raise RuntimeError("Se pidió device 'cuda' pero torch.cuda.is_available() es False")  # sin degradar
    return torch.device(nombre)  # 'cpu' o 'cuda' tal cual


# %% [markdown]
# ## Modo determinista
#
# En CPU se activa `torch.use_deterministic_algorithms(True)`: PyTorch usa solo algoritmos deterministas (o lanza un
# error si una operación no tiene versión determinista). En CUDA no se activa: entre CPU y GPU se acepta una
# tolerancia documentada (núcleo §7.1).

# %%
def activar_determinismo(device) -> None:
    """En CPU enciende ``torch.use_deterministic_algorithms(True)``; en CUDA no hace nada."""
    if torch.device(device).type == "cpu":  # acepta un torch.device o su nombre
        torch.use_deterministic_algorithms(True)  # algoritmos deterministas (núcleo §7.1)


# %% [markdown]
# ## numpy → tensor
#
# Regla: el resultado es `float32` en el dispositivo pedido. Se admiten un vector $(n,)$ o una matriz de diseño
# $(n, d)$. Con `columna=True` la entrada debe ser un vector $(n,)$ y sale como columna $(n, 1)$ (la forma del
# objetivo que espera la pérdida junto a una salida `Linear(·, 1)`). NaN o $\pm\infty$ → `ValueError`.

# %%
def a_tensor(x, device, columna: bool = False) -> torch.Tensor:
    """Convierte ``x`` a tensor float32 en ``device``; ``columna=True`` exige un vector y lo devuelve (n, 1)."""
    arreglo = np.asarray(x, dtype=np.float32)  # copia/convierte a float32
    if columna and arreglo.ndim != 1:  # una columna sale solo de un vector
        raise ValueError(f"Con columna=True se espera un vector (n,); se recibió la forma {arreglo.shape}")  # forma
    if arreglo.ndim not in (1, 2):  # vector o matriz de diseño
        raise ValueError(f"Se espera un arreglo 1-D o 2-D; se recibió la forma {arreglo.shape}")  # forma inválida
    if not np.all(np.isfinite(arreglo)):  # NaN o ±inf romperían el entrenamiento en silencio
        raise ValueError("El arreglo contiene valores no finitos (NaN o ±inf)")  # dato inválido
    tensor = torch.as_tensor(arreglo, device=torch.device(device))  # al dispositivo pedido
    return tensor.reshape(-1, 1) if columna else tensor  # (n, 1) si es columna; si no, la forma original


# %% [markdown]
# ## tensor → numpy
#
# Regla: un vector $(n,)$ o una columna $(n, 1)$ salen como `float64` de forma $(n,)$ (la forma de `predict_proba`
# y `predict_logit` del contrato `ModeloPD`). Cualquier otra forma → `ValueError` (no se aplana una matriz en
# silencio).

# %%
def a_numpy(t: torch.Tensor) -> np.ndarray:
    """Convierte un tensor (n,) o (n, 1) a numpy float64 de forma (n,)."""
    if not (t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1)):  # solo vector o columna
        raise ValueError(f"Se espera un tensor (n,) o (n, 1); se recibió la forma {tuple(t.shape)}")  # forma inválida
    return t.detach().cpu().numpy().astype(np.float64).reshape(-1)  # sin grafo, en CPU, float64 y (n,)
