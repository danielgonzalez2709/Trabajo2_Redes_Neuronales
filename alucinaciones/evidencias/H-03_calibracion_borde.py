"""H-03: ¿a qué bin va un valor que cae exactamente en un borde de cuantil? Contraste con sklearn.

Ejecuta: .venv/bin/python alucinaciones/evidencias/H-03_calibracion_borde.py
"""
import inspect

import numpy as np
from sklearn import calibration
from sklearn.calibration import calibration_curve

# 1) El código fuente de sklearn usa searchsorted con el lado por defecto ("left").
fuente = inspect.getsource(calibration.calibration_curve)
print("Línea de sklearn:", [l.strip() for l in fuente.splitlines() if "searchsorted" in l])

# 2) Datos con empates (como los de la prueba): p redondeado a 2 decimales.
rng = np.random.default_rng(20261024)
y = rng.binomial(1, 0.15, 2000)
p = np.round(1 / (1 + np.exp(-rng.normal(-1.8 + y, 1))), 2)
bordes = np.percentile(p, np.linspace(0, 100, 11))
print("Bordes internos que coinciden con algún dato:", int(np.isin(bordes[1:-1], p).sum()), "de 9")

tasa_ref, pd_ref = calibration_curve(y, p, n_bins=10, strategy="quantile")
for lado in ("left", "right"):
    ids = np.searchsorted(bordes[1:-1], p, side=lado)
    pd_media = np.array([p[ids == k].mean() for k in np.unique(ids)])
    coincide = pd_media.shape == pd_ref.shape and np.allclose(pd_media, pd_ref, atol=1e-12)
    print(f"side={lado!r}: coincide con calibration_curve -> {coincide}")
