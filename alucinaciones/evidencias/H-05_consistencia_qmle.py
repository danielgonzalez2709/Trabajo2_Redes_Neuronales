"""Evidencia H-05: la consistencia del QMLE de Bernoulli es una propiedad del modelo paramétrico bien especificado.

Datos con E[y|x] = σ(−0,5 + 1,2·x) e y ~ Beta (no Bernoulli). El QMLE paramétrico (logit lineal) con la BCE fraccional
de ``src/lgd/perdidas.py`` recupera los coeficientes verdaderos. Esto respalda la frase para el modelo paramétrico; para un
MLP (no identificable, con L2 y parada temprana) lo que se sostiene es que E[y|x] minimiza la pérdida esperada, no la
"consistencia" del estimador.

Uso: .venv/bin/python alucinaciones/evidencias/H-05_consistencia_qmle.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.lgd.perdidas import perdida_bce_fraccional  # noqa: E402

torch.set_default_dtype(torch.float64)

# 1. Minimizador puntual: para y fija, argmin_μ de la BCE está en μ = y (con y ∈ {0, 1} el logit diverge).
zs = torch.linspace(-20, 20, 400001)
for y in [0.0, 0.17, 0.5, 0.93, 1.0]:
    perdidas = torch.nn.functional.binary_cross_entropy_with_logits(zs, torch.full_like(zs, y), reduction="none")
    print(f"y = {y:.2f} -> argmin μ = {float(torch.sigmoid(zs[perdidas.argmin()])):.5f}")

# 2. QMLE paramétrico con y ~ Beta de media σ(a + b·x).
rng = np.random.default_rng(1)
n, phi = 200_000, 5
x = rng.normal(size=n)
media = 1 / (1 + np.exp(-(-0.5 + 1.2 * x)))
y = rng.beta(media * phi, (1 - media) * phi)
X = torch.tensor(np.c_[np.ones(n), x])
Y = torch.tensor(y)
coef = torch.zeros(2, requires_grad=True)
opt = torch.optim.LBFGS([coef], max_iter=200)


def cierre():
    opt.zero_grad()
    perdida = perdida_bce_fraccional(X @ coef, Y)
    perdida.backward()
    return perdida


opt.step(cierre)
print(f"QMLE con y ~ Beta: coeficientes {[round(c, 4) for c in coef.detach().tolist()]} (verdaderos [-0.5, 1.2])")
