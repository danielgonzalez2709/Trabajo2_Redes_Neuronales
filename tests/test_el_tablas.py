"""E5 — tablas de la E[L] para el notebook 05 y el blog (núcleo §5.8: cero lógica en el notebook).

El contraste con lo observado es solo de los créditos resueltos (E §3.2); los en mora van aparte con su pérdida
provisional.
"""

import math

import pytest

from src.el.tablas import tabla_bandas, tabla_totales


def _base(n, el, tasa, pd_, lgd):
    return {"n": n, "el": el, "ead": 10 * el, "tasa_malos": tasa, "pd_media": pd_, "lgd_media": lgd}


def _resueltos(n, el, obs, tasa, pd_, lgd):
    return {**_base(n, el, tasa, pd_, lgd), "perdida_observada": obs,
            "razon_el_sobre_perdida_observada": el / obs if obs else None}


def _mora(n, el, prov, pd_, lgd):
    return {**_base(n, el, 1.0, pd_, lgd), "perdida_provisional": prov}


def _resumen():
    """Resumen pequeño con las mismas claves que results/data/el/summary.json."""
    return {
        "total": _base(100, 900.0, 0.25, 0.2, 0.9),
        "resueltos": _resueltos(90, 800.0, 1_000.0, 0.17, 0.19, 0.9),
        "en_mora": _mora(10, 100.0, 400.0, 0.29, 0.91),
        "por_banda": [
            {"banda": 1, "score_min": 480.0, "score_max": 540.5, "total": _base(50, 600.0, 0.4, 0.3, 0.92),
             "resueltos": _resueltos(42, 520.0, 900.0, 0.3, 0.29, 0.92), "en_mora": _mora(8, 80.0, 300.0, 0.33, 0.9)},
            {"banda": 2, "score_min": 540.6, "score_max": 650.0, "total": _base(50, 300.0, 0.1, 0.1, 0.88),
             "resueltos": _resueltos(48, 280.0, 0.0, 0.06, 0.1, 0.88), "en_mora": _mora(2, 20.0, 100.0, 0.12, 0.9)},
        ],
    }


def test_tabla_totales_todo_el_test_resueltos_y_en_mora():
    t = tabla_totales(_resumen())
    assert t.index.tolist() == ["Todo el test", "Resueltos (contraste)", "En mora (cobranza abierta)"]
    assert t["n"].tolist() == [100, 90, 10]
    assert t["E[L] (USD)"].tolist() == [900.0, 800.0, 100.0]
    # la pérdida realizada y la razón solo existen para los resueltos
    realizada = t["Pérdida realizada (USD)"].tolist()
    assert math.isnan(realizada[0]) and realizada[1] == 1_000.0 and math.isnan(realizada[2])
    razon = t["E[L] / realizada"].tolist()
    assert math.isnan(razon[0]) and razon[1] == pytest.approx(0.8) and math.isnan(razon[2])
    provisional = t["Pérdida provisional (USD)"].tolist()
    assert math.isnan(provisional[0]) and math.isnan(provisional[1]) and provisional[2] == 400.0
    assert t["Tasa de malos"].tolist() == [0.25, 0.17, 1.0]
    assert t["PD media"].tolist() == [0.2, 0.19, 0.29]
    assert t["LGD media"].tolist() == [0.9, 0.9, 0.91]


def test_tabla_bandas_contrasta_los_resueltos_de_cada_banda():
    t = tabla_bandas(_resumen())
    assert t.index.tolist() == [1, 2] and t.index.name == "Banda"
    assert t["Puntaje mín."].tolist() == [480.0, 540.6] and t["Puntaje máx."].tolist() == [540.5, 650.0]
    assert t["n"].tolist() == [50, 50] and t["E[L] (USD)"].tolist() == [600.0, 300.0]
    assert t["n resueltos"].tolist() == [42, 48]
    assert t["E[L] resueltos (USD)"].tolist() == [520.0, 280.0]
    assert t["Pérdida realizada resueltos (USD)"].tolist() == [900.0, 0.0]
    assert t["E[L] / realizada (resueltos)"].iloc[0] == pytest.approx(520 / 900)
    assert math.isnan(t["E[L] / realizada (resueltos)"].iloc[1])  # sin pérdida realizada la razón no está definida
    assert t["n en mora"].tolist() == [8, 2]
    assert t["Pérdida provisional en mora (USD)"].tolist() == [300.0, 100.0]
    assert t["Tasa de malos"].tolist() == [0.4, 0.1]
    assert t["PD media"].tolist() == [0.3, 0.1] and t["LGD media"].tolist() == [0.92, 0.88]


def test_las_tablas_no_mutan_el_resumen():
    r = _resumen()
    antes = repr(r)
    tabla_totales(r), tabla_bandas(r)
    assert repr(r) == antes
