"""Pérdidas del MLP de LGD (Spec, Parte D §3.3). Ambas reciben el **logit** z (salida lineal de la red) y el objetivo y.

Derivación de la pérdida [PROPUESTO] por máxima cuasi-verosimilitud (Papke & Wooldridge, 1996)
---------------------------------------------------------------------------------------------
La LGD es una fracción: y ∈ [0, 1]. Se modela su media condicional como μ(x) = E[y | x] = σ(z(x)), con
σ(z) = 1 / (1 + e^(−z)) y z(x) el logit que produce la red.

Por qué sigmoide en la salida: su rango es (0, 1), el mismo de la LGD, así que toda predicción es una LGD
admisible sin recortes; y es la **inversa** del enlace logit (logit(μ) = log(μ / (1 − μ)) = z) de la
cuasi-verosimilitud de abajo.

Cuasi-log-verosimilitud de Bernoulli de una observación (se usa aunque y no sea 0/1):

    ℓ(μ; y) = y·log μ + (1 − y)·log(1 − μ),      y ∈ [0, 1].

La pérdida es la cuasi-log-verosimilitud media cambiada de signo:

    L = −(1/n)·Σᵢ ℓ(μᵢ; yᵢ),      μᵢ = σ(zᵢ).

Gradiente respecto al logit (usando dσ/dz = μ(1 − μ)):

    ∂ℓ/∂μ = y/μ − (1 − y)/(1 − μ) = (y − μ) / (μ(1 − μ))
    ∂ℓ/∂z = ∂ℓ/∂μ · μ(1 − μ) = y − μ      ⇒      ∂(−ℓ)/∂z = μ − y.

Su esperanza condicional es E[μ − y | x] = μ − E[y | x], que se anula cuando μ = E[y | x]. Como −ℓ es convexa en
μ para y ∈ [0, 1], el minimizador **poblacional** de la pérdida esperada E[−ℓ(μ; y) | x] es μ = E[y | x]: lo
que aprende el MLP con esta pérdida apunta a la media condicional sin suponer que y sea Bernoulli. Para un modelo
**paramétrico** con la media bien especificada, el estimador de cuasi-máxima verosimilitud de Bernoulli es
consistente (Papke & Wooldridge, 1996). Por eso es falso que "la entropía cruzada solo sirve con objetivos 0/1".

Casos en el borde: si una LGD es exactamente 0 o 1, el logit óptimo para esa observación diverge (z → −∞ o +∞);
lo contienen la penalización L2 y la parada temprana (``src/lgd/mlp.py``).

Implementación: ``binary_cross_entropy_with_logits`` calcula exactamente −ℓ a partir de z con la forma
numéricamente estable (acepta objetivos en [0, 1]).

Contraste opcional ``mse``: verosimilitud gaussiana con varianza constante sobre μ = σ(z), L = (1/n)·Σ(μᵢ − yᵢ)².

Referencia: Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with
an application to 401(k) plan participation rates. *Journal of Applied Econometrics, 11*(6), 619–632.
TODO(cita): enunciado del teorema verificado solo en fuente secundaria (Oberhofer & Pfaffermayr, 2009).
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def perdida_bce_fraccional(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Entropía cruzada con objetivo fraccional: −(1/n)·Σ[y·log σ(z) + (1 − y)·log(1 − σ(z))]."""
    return F.binary_cross_entropy_with_logits(logits, y, reduction="mean")


def perdida_mse(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Error cuadrático medio sobre la LGD estimada σ(z)."""
    return torch.mean((torch.sigmoid(logits) - y) ** 2)


_PERDIDAS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "bce_fraccional": perdida_bce_fraccional,
    "mse": perdida_mse,
}
PERDIDAS_VALIDAS = tuple(_PERDIDAS)


def obtener_perdida(nombre: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Función de pérdida para ``mlp.perdida``; valor desconocido → ``ValueError``."""
    if nombre not in _PERDIDAS:
        raise ValueError(f"mlp.perdida debe ser una de {list(PERDIDAS_VALIDAS)}; se recibió {nombre!r}")
    return _PERDIDAS[nombre]
