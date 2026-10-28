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


# --- Descomposición de la razón E[L] / realizada de los resueltos (blog 9.5): factor PD · factor LGD · factor EAD ---

def _el_test():
    """Créditos de juguete con las columnas de results/data/el/el_test.parquet que usa la descomposición.

    Banda 1: dos castigados y un pagado; banda 2: un castigado; además un crédito en mora, que no entra (no está resuelto).
    ead = F (monto desembolsado), ead_incumplimiento = F − P (principal pendiente al castigo), perdida_observada = L.
    """
    import pandas as pd

    return pd.DataFrame({
        "estado": ["castigado", "castigado", "pagado", "castigado", "en_mora"],
        "banda": [1, 1, 1, 2, 2],
        "pd": [0.5, 0.2, 0.1, 0.4, 0.9],
        "lgd": [0.8, 0.9, 0.7, 0.5, 0.9],
        "ead": [100.0, 200.0, 50.0, 100.0, 1_000.0],
        "ead_incumplimiento": [80.0, 100.0, 0.0, 50.0, 800.0],
        "perdida_observada": [40.0, 90.0, 0.0, 25.0, float("nan")],
    })


def test_descomposicion_razon_a_mano_por_banda_y_total():
    from src.el.tablas import descomposicion_razon

    d = descomposicion_razon(_el_test())
    # Banda 1 (a mano): Σ_R pd·lgd·F = 40 + 36 + 3,5 = 79,5; Σ_C lgd·F = 80 + 180 = 260; LGD observada = L/(F−P) = 0,5 y 0,9,
    # Σ_C lgd_obs·F = 50 + 180 = 230; Σ_C L = 130.
    b1 = d.loc[1]
    assert b1["Factor PD"] == pytest.approx(79.5 / 260)
    assert b1["Factor LGD"] == pytest.approx(260 / 230)
    assert b1["Factor EAD"] == pytest.approx(230 / 130)
    assert b1["E[L] / realizada"] == pytest.approx(79.5 / 130)
    # Banda 2: solo el castigado (el en mora no entra): 20 / 25; factores 20/50, 50/50 y 50/25.
    b2 = d.loc[2]
    assert (b2["Factor PD"], b2["Factor LGD"], b2["Factor EAD"]) == pytest.approx((20 / 50, 50 / 50, 50 / 25))
    # Total: suma de las dos bandas sin el en mora.
    t = d.loc["Total"]
    assert t["E[L] / realizada"] == pytest.approx(99.5 / 155)
    assert t["Factor PD"] == pytest.approx(99.5 / 310)
    assert t["Factor LGD"] == pytest.approx(310 / 280)
    assert t["Factor EAD"] == pytest.approx(280 / 155)


def test_descomposicion_es_una_identidad():
    """El producto de los tres factores reproduce la razón en cada fila (los términos intermedios se cancelan)."""
    from src.el.tablas import descomposicion_razon

    d = descomposicion_razon(_el_test())
    producto = d["Factor PD"] * d["Factor LGD"] * d["Factor EAD"]
    assert producto.tolist() == pytest.approx(d["E[L] / realizada"].tolist())
    assert list(d.index) == [1, 2, "Total"]


def test_descomposicion_rechaza_castigados_sin_exposicion_al_castigo():
    """La LGD observada es L/(F − P): con F − P ≤ 0 no está definida y la función no debe dividir en silencio."""
    from src.el.tablas import descomposicion_razon

    df = _el_test()
    df.loc[0, "ead_incumplimiento"] = 0.0
    with pytest.raises(ValueError, match="ead_incumplimiento"):
        descomposicion_razon(df)
