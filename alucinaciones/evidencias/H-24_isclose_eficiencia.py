import json, sys, numpy as np, pandas as pd, shap
sys.path.insert(0, ".")
from src.common.preprocesamiento import Preprocesador
from src.scorecard.puntos import grupos_columnas
from src.scorecard.shap_pd import a_variables, valores_shap, error_eficiencia
pre_json = json.load(open("models/preprocesador_pd.json", encoding="utf-8")); pre = Preprocesador.from_dict(pre_json)
df = pd.read_parquet("data/processed/loans_pd.parquet"); g = grupos_columnas(pre_json)
Z = a_variables(pre.transform(df[df.split == "train"]), g)
j = [v["nombre"] for v in pre_json["variables"]].index("revol_bal")
col = np.unique(Z[:, j]); d = np.diff(col); k = int(np.argmin(np.where(d > 0, d / np.abs(col[1:]), np.inf)))
a, b = col[k], col[k + 1]
print(f"revol_bal transformado: a = {a!r}, b = {b!r}, b - a = {b - a:.3e}, np.isclose(a, b) = {np.isclose(a, b)} (rtol 1e-5 por defecto)")
fondo = np.tile(Z[:1], (4, 1)); fondo[:, j] = a; x = fondo[:1].copy(); x[0, j] = b
f = lambda M: 1e6 * M[:, j]          # f muy sensible a revol_bal
phi, phi0 = valores_shap(f, fondo, x, 62, 0)
print(f"f(x) - E_fondo f = {f(x)[0] - phi0:.6e}; phi(revol_bal) = {phi[0, j]:.6e}; error de eficiencia = {error_eficiencia(phi, phi0, f(x)):.3e}")
