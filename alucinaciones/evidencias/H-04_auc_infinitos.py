"""H-04: auc() del commit a11a56c con puntajes ±inf (logit de una PD saturada en 0 o 1).

Ejecuta: .venv/bin/python alucinaciones/evidencias/H-04_auc_infinitos.py
"""
import numpy as np

from src.common.metricas import auc

y = np.array([0, 1, 0, 1, 0, 1])
p = np.array([-np.inf, -np.inf, 0.2, 0.5, np.inf, np.inf])

# AUC por definición (Mann-Whitney): pares (malo, bueno) con p_malo > p_bueno, empates valen 1/2.
malos, buenos = p[y == 1], p[y == 0]
pares = [(1.0 if a > b else 0.5 if a == b else 0.0) for a in malos for b in buenos]
print("AUC por pares (correcta):", np.mean(pares))
try:
    print("auc() del módulo:", auc(y, p))
except ValueError as e:
    print("auc() rechaza la entrada:", e)
