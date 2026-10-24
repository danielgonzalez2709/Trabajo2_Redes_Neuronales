"""Evidencia H-01: PSI con cortes por cuantiles (versión original generada por la IA) vs. por valor, sobre pub_rec real.

Uso (desde la raíz del repo): python alucinaciones/evidencias/H-01_psi_discreto.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.preprocesamiento import Preprocesador  # noqa: E402
from src.common.results import load_json  # noqa: E402
from src.datos.eda import psi  # noqa: E402


def psi_cuantiles(ref, comp, n_bins=10):
    """Versión original (commit 3b1d137 en adelante, antes de la corrección)."""
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    e = np.bincount(np.searchsorted(cortes, ref, side="right"), minlength=len(cortes) + 1) / len(ref)
    a = np.bincount(np.searchsorted(cortes, comp, side="right"), minlength=len(cortes) + 1) / len(comp)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


df = pd.read_parquet("data/processed/loans_pd.parquet")
pre = Preprocesador.from_dict(load_json("models/preprocesador_pd.json"))
i = pre.nombres_salida().index("pub_rec")
X = {s: pre.transform(df[df.split == s])[:, i] for s in ("train", "val", "test")}
print("valores distintos de pub_rec transformada en train:", np.unique(X["train"]).round(4).tolist())
print("cortes por cuantiles:", np.unique(np.quantile(X["train"], np.linspace(0, 1, 11)[1:-1])).round(4).tolist())
for s in ("val", "test"):
    print(f"{s}: proporción con pub_rec >= 1 = {np.mean(X[s] > X['train'].min()):.4f} "
          f"(train {np.mean(X['train'] > X['train'].min()):.4f}) | PSI original = {psi_cuantiles(X['train'], X[s]):.4f}"
          f" | PSI corregido = {psi(X['train'], X[s]):.4f}")
