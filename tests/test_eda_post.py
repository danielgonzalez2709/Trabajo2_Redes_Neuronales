"""A6b — Descriptivo después del preprocesamiento (Parte A §3.8) y EDA en memoria para el notebook."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json
from src.datos.eda import descriptivo_post, psi
from src.datos.experimento_eda import eda_en_memoria

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = [{"nombre": "dti", "tipo": "num", "transformacion": "ninguna"},
            {"nombre": "monto", "tipo": "num", "transformacion": "log1p"},
            {"nombre": "plazo", "tipo": "cat", "transformacion": "ninguna"}]


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 6000
    dti = rng.uniform(0, 40, n)
    plazo = rng.choice(["36", "60"], n)
    p = 0.05 + 0.005 * dti + 0.1 * (plazo == "60")
    return pd.DataFrame({"dti": dti, "monto": rng.lognormal(9, 1, n), "plazo": plazo,
                         "target": (rng.random(n) < p).astype(int),
                         "split": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2])})


def test_estandariza_train_y_mide_psi(df):
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    out = descriptivo_post(df, pre)
    assert out["estandarizacion"]["train"]["dti"]["media"] == pytest.approx(0, abs=1e-5)
    assert out["estandarizacion"]["train"]["dti"]["desv"] == pytest.approx(1, abs=1e-4)
    assert out["psi_vs_train"]["val"]["dti"] < 0.02  # mismo proceso generador
    assert out["relacion_con_objetivo_train"]["dti"]["spearman_grupos"] > 0.8


def test_distribucion_antes_y_despues(df):
    """A §3.8.1: el log1p reduce la asimetría de una lognormal; el histograma suma todos los créditos de train."""
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    d = descriptivo_post(df, pre)["distribucion_antes_despues_train"]["monto"]
    assert d["antes"]["asimetria"] > 2 and abs(d["despues"]["asimetria"]) < 0.5
    n_train = int((df["split"] == "train").sum())
    assert sum(d["antes"]["histograma"]["conteos"]) == sum(d["despues"]["histograma"]["conteos"]) == n_train


def test_psi_es_kl_simetrizada():
    rng = np.random.default_rng(3)
    ref, comp = rng.normal(0, 1, 5000), rng.normal(0.4, 1.2, 5000)
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
    e = np.bincount(np.searchsorted(cortes, ref, side="right"), minlength=10) / len(ref)
    a = np.bincount(np.searchsorted(cortes, comp, side="right"), minlength=10) / len(comp)
    kl = lambda p, q: float(np.sum(p * np.log(p / q)))  # noqa: E731
    assert psi(ref, comp) == pytest.approx(kl(a, e) + kl(e, a))


def test_eda_en_memoria_reproduce_el_preprocesador_guardado_y_no_escribe(tmp_path, monkeypatch):
    """Con la config del repositorio, el preprocesador ajustado en memoria es el de models/ (mismo train)."""
    monkeypatch.chdir(RAIZ)
    cfg = load_config(RAIZ / "configs" / "datos.yaml")
    antes = {p: p.stat().st_mtime_ns for p in (RAIZ / "results" / "data" / "eda").glob("*.json")}
    r = eda_en_memoria(cfg, RAIZ)
    guardado = load_json(RAIZ / cfg["salida"]["preprocesador"])
    assert json.dumps(r["preprocesador"].to_dict(), sort_keys=True) == json.dumps(guardado, sort_keys=True)
    assert {p: p.stat().st_mtime_ns for p in (RAIZ / "results" / "data" / "eda").glob("*.json")} == antes


def test_eda_en_memoria_respeta_otra_particion():
    cfg = load_config(RAIZ / "configs" / "datos.yaml")
    cfg["particion"] = {**cfg["particion"], "train_hasta": "2011-12-31", "val_hasta": "2012-12-31"}
    r = eda_en_memoria(cfg, RAIZ)
    resumen = load_json(RAIZ / "data" / "processed" / "resumen_datos.json")
    assert r["particion"]["n_por_split"]["train"] < resumen["n_por_split"]["train"]
    assert sum(r["particion"]["n_por_split"].values()) == resumen["n_modelado"]
