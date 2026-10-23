"""A6/A6b — Análisis descriptivo (Parte A §3.7 y §3.8)."""

import numpy as np
import pandas as pd
import pytest

from src.common.preprocesamiento import Preprocesador
from src.datos.eda import descriptivo_post, eda_inicial, evaluar_hipotesis, psi, tasa_por_grupo

FEATURES = [{"nombre": "dti", "tipo": "num", "transformacion": "ninguna"},
            {"nombre": "plazo", "tipo": "cat", "transformacion": "ninguna"}]


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 6000
    dti = rng.uniform(0, 40, n)
    plazo = rng.choice(["36", "60"], n)
    p = 0.05 + 0.005 * dti + 0.1 * (plazo == "60")
    return pd.DataFrame({"dti": dti, "plazo": plazo, "target": (rng.random(n) < p).astype(int),
                         "split": rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2])})


def test_psi_cero_si_misma_distribucion_y_crece_con_el_cambio():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 20000)
    assert psi(ref, ref) == pytest.approx(0, abs=1e-12)
    assert psi(ref, rng.normal(0, 1, 20000)) < 0.01
    assert psi(ref, rng.normal(1, 1, 20000)) > psi(ref, rng.normal(0.3, 1, 20000)) > 0.05


def test_psi_a_mano():
    # referencia uniforme en 2 bins (mediana = 0.5); comparada 80/20
    ref = np.array([0.1, 0.2, 0.8, 0.9])
    comp = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    esperado = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(ref, comp, n_bins=2) == pytest.approx(esperado)


def test_psi_variable_discreta_no_colapsa():
    """Regresión: con pub_rec ∈ {0, 1} los cortes por cuantiles colapsaban y el PSI daba 0 siempre."""
    ref = np.array([0] * 900 + [1] * 100, dtype=float)
    comp = np.array([0] * 700 + [1] * 300, dtype=float)
    esperado = (0.7 - 0.9) * np.log(0.7 / 0.9) + (0.3 - 0.1) * np.log(0.3 / 0.1)
    assert psi(ref, comp) == pytest.approx(esperado)


def test_hipotesis_no_monotona_se_marca():
    x = pd.Series(np.repeat(np.arange(10.0), 100))
    tasas = np.array([0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    y = pd.Series((np.tile(np.arange(100), 10) < np.repeat(tasas * 100, 100)).astype(int))
    h = evaluar_hipotesis(pd.DataFrame({"x": x, "target": y}), [{"id": "H", "variable": "x", "direccion": "+"}])[0]
    assert h["veredicto_eda"] == "la apoya, pero no es monótona"


def test_tasa_por_grupo_deciles_crecientes(df):
    tabla = tasa_por_grupo(df["dti"], df["target"], "num")
    assert len(tabla) == 10 and tabla["n"].sum() == len(df)
    assert tabla["tasa"].iloc[-1] > tabla["tasa"].iloc[0]


def test_hipotesis_numerica_y_categorica(df):
    hs = [{"id": "H1", "variable": "dti", "direccion": "+"},
          {"id": "H2", "variable": "dti", "direccion": "-"},
          {"id": "H3", "variable": "plazo", "mas_riesgosa": "60", "referencia": "36"}]
    v = {h["id"]: h["veredicto_eda"] for h in evaluar_hipotesis(df[df["split"] == "train"], hs)}
    assert v == {"H1": "la apoya", "H2": "no la apoya", "H3": "la apoya"}


def test_eda_inicial_solo_train(df):
    out = eda_inicial(df, FEATURES)
    assert out["n_train"] == (df["split"] == "train").sum()
    assert set(out["variables"]) == {"dti", "plazo"}


def test_descriptivo_post_estandariza_train_y_mide_psi(df):
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    out = descriptivo_post(df, pre)
    assert out["estandarizacion"]["train"]["dti"]["media"] == pytest.approx(0, abs=1e-5)
    assert out["estandarizacion"]["train"]["dti"]["desv"] == pytest.approx(1, abs=1e-4)
    assert out["psi_vs_train"]["val"]["dti"] < 0.02  # mismo proceso generador
    assert out["relacion_con_objetivo_train"]["dti"]["spearman_grupos"] > 0.8
