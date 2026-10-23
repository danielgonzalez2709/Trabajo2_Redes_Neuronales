"""A4 — Partición temporal (Parte A §3.5)."""

import pandas as pd
import pytest

from src.datos.particion import asignar_split, muestra_estratificada


def _fechas(*textos):
    return pd.Series(pd.to_datetime(list(textos)))


def test_cortes_y_bordes():
    fechas = _fechas("2012-06-01", "2013-12-01", "2014-01-01", "2014-12-01", "2015-01-01")
    split = asignar_split(fechas, "2013-12-31", "2014-12-31")
    assert list(split) == ["train", "train", "val", "val", "test"]


def test_sin_solapamiento_y_orden_temporal():
    fechas = pd.Series(pd.date_range("2008-01-01", "2015-12-01", freq="MS"))
    split = asignar_split(fechas, "2013-12-31", "2014-12-31")
    assert fechas[split == "train"].max() < fechas[split == "val"].min()
    assert fechas[split == "val"].max() < fechas[split == "test"].min()
    assert set(split) == {"train", "val", "test"}


def test_fecha_faltante_es_error():
    with pytest.raises(ValueError, match="sin fecha"):
        asignar_split(_fechas("2012-01-01", None), "2013-12-31", "2014-12-31")


def test_cortes_invertidos_es_error():
    with pytest.raises(ValueError, match="anterior"):
        asignar_split(_fechas("2012-01-01"), "2014-12-31", "2013-12-31")


def test_muestra_estratificada_conserva_proporciones_y_es_reproducible():
    df = pd.DataFrame({"split": ["train"] * 800 + ["test"] * 200, "target": ([0] * 9 + [1]) * 100})
    a = muestra_estratificada(df, 100, semilla=1)
    b = muestra_estratificada(df, 100, semilla=1)
    assert a.index.equals(b.index)
    assert abs(len(a) - 100) <= 4
    assert (a["split"] == "train").mean() == pytest.approx(0.8, abs=0.03)
    assert a["target"].mean() == pytest.approx(0.1, abs=0.03)
