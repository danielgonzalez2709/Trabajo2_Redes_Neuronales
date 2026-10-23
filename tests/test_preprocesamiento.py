"""A5 — Preprocesador común (núcleo §5.2)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.common.preprocesamiento import DESCONOCIDO, OTRA, Preprocesador

FEATURES = [
    {"nombre": "ingreso", "tipo": "num", "transformacion": "log1p"},
    {"nombre": "dti", "tipo": "num", "transformacion": "ninguna"},
    {"nombre": "vivienda", "tipo": "cat"},
]


@pytest.fixture
def train():
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame({
        "ingreso": np.where(rng.random(n) < 0.05, np.nan, rng.lognormal(10, 1, n)),
        "dti": rng.normal(15, 8, n),
        "vivienda": rng.choice(["RENT", "OWN", "MORTGAGE", None], n, p=[0.4, 0.1, 0.45, 0.05]),
    })


def test_ajuste_solo_con_train_estandariza_train(train):
    X = Preprocesador().fit(train, FEATURES).transform(train)
    assert X.dtype == np.float32
    assert np.allclose(X[:, :2].mean(axis=0), 0, atol=1e-5)
    assert np.allclose(X[:, :2].std(axis=0), 1, atol=1e-4)


def test_columnas_y_categorias(train):
    pre = Preprocesador().fit(train, FEATURES)
    assert pre.nombres_salida()[:2] == ["ingreso", "dti"]
    cats = pre.variables[2]["categorias"]
    assert cats[-1] == OTRA and DESCONOCIDO in cats and cats[:-1] == sorted(cats[:-1])
    X = pre.transform(train)
    assert X.shape == (len(train), 2 + len(cats))
    assert np.all(X[:, 2:].sum(axis=1) == 1)  # exactamente una categoría por fila


def test_valores_no_vistos_van_a_otra_y_vacios_a_desconocido(train):
    pre = Preprocesador().fit(train, FEATURES)
    nuevo = pd.DataFrame({"ingreso": [np.nan], "dti": [1e9], "vivienda": ["BARCO"]})
    fila = pre.transform(nuevo)[0]
    nombres = pre.nombres_salida()
    assert fila[nombres.index(f"vivienda={OTRA}")] == 1
    v_ing, v_dti = pre.variables[0], pre.variables[1]
    assert fila[0] == pytest.approx((np.log1p(v_ing["mediana"]) - v_ing["media"]) / v_ing["desv"], rel=1e-5)
    assert fila[1] == pytest.approx((v_dti["recorte"][1] - v_dti["media"]) / v_dti["desv"], rel=1e-5)  # recortado


def test_val_no_cambia_los_estadisticos(train):
    pre = Preprocesador().fit(train, FEATURES)
    antes = json.dumps(pre.to_dict())
    pre.transform(train.assign(dti=train["dti"] + 100))
    assert json.dumps(pre.to_dict()) == antes


def test_ida_y_vuelta_json_bit_a_bit(train):
    pre = Preprocesador().fit(train, FEATURES)
    copia = Preprocesador.from_dict(json.loads(json.dumps(pre.to_dict())))
    assert np.array_equal(pre.transform(train), copia.transform(train))


def test_sin_ajustar_es_error(train):
    with pytest.raises(RuntimeError):
        Preprocesador().transform(train)


def test_tipo_desconocido_es_error(train):
    with pytest.raises(ValueError, match="tipo desconocido"):
        Preprocesador().fit(train, [{"nombre": "dti", "tipo": "fecha"}])
