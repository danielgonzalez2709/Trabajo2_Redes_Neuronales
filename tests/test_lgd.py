"""Pruebas de la variable objetivo de la LGD (tarea D1; Parte D §3.1 y §3.2).

Casos hechos a mano: cada número esperado se calcula en el comentario a partir de la fórmula
    EAD = funded_amnt − total_rec_prncp
    LGD = 1 − (recoveries − collection_recovery_fee) / EAD, recortada a [0, 1].
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lgd.objetivo import (
    ESTADOS_EXCLUIDOS,
    ESTADOS_LGD,
    calcular_ead,
    calcular_lgd,
    es_poblacion_lgd,
    mascara_poblacion_lgd,
    resumen_objetivo,
)

CO = "Charged Off"
CO_POL = "Does not meet the credit policy. Status:Charged Off"


def _montos(filas):
    return pd.DataFrame(filas, columns=["id", "loan_status", "funded_amnt", "total_rec_prncp", "recoveries",
                                        "collection_recovery_fee"])


# --- población ---------------------------------------------------------------------------------------------------

def test_estados_lgd_son_los_dos_charged_off():
    assert set(ESTADOS_LGD) == {CO, CO_POL}
    assert set(ESTADOS_EXCLUIDOS) == {"Default", "Late (31-120 days)"}


def test_es_poblacion_lgd_filtra_exactamente_los_charged_off():
    estados = pd.Series([CO, CO_POL, "Fully Paid", "Does not meet the credit policy. Status:Fully Paid", "Default",
                         "Late (31-120 days)", "Current", "charged off", None])
    mascara = es_poblacion_lgd(estados)
    assert mascara.dtype == bool
    assert mascara.tolist() == [True, True, False, False, False, False, False, False, False]


# --- EAD ---------------------------------------------------------------------------------------------------------

def test_calcular_ead_es_principal_pendiente():
    m = _montos([["1", CO, 10000.0, 2500.0, 0.0, 0.0], ["2", CO, 5000.0, 5000.0, 0.0, 0.0]])
    assert calcular_ead(m).tolist() == [7500.0, 0.0]


# --- LGD: casos a mano -------------------------------------------------------------------------------------------

def test_recuperacion_parcial():
    # EAD = 10000 − 2000 = 8000; neto = 1000 − 200 = 800; LGD = 1 − 800/8000 = 0.9
    r = calcular_lgd(_montos([["1", CO, 10000.0, 2000.0, 1000.0, 200.0]]))
    assert r.loc[0, "ead"] == pytest.approx(8000.0)
    assert r.loc[0, "lgd_sin_recorte"] == pytest.approx(0.9)
    assert r.loc[0, "lgd"] == pytest.approx(0.9)
    assert bool(r.loc[0, "lgd_valida"])


def test_recuperacion_cero_da_lgd_uno():
    # EAD = 4000; neto = 0; LGD = 1
    r = calcular_lgd(_montos([["1", CO, 5000.0, 1000.0, 0.0, 0.0]]))
    assert r.loc[0, "lgd_sin_recorte"] == pytest.approx(1.0)
    assert r.loc[0, "lgd"] == pytest.approx(1.0)


def test_recuperacion_mayor_que_ead_se_recorta_a_cero():
    # EAD = 1000; neto = 1500 − 0 = 1500; LGD sin recorte = 1 − 1.5 = −0.5 → 0
    r = calcular_lgd(_montos([["1", CO, 3000.0, 2000.0, 1500.0, 0.0]]))
    assert r.loc[0, "lgd_sin_recorte"] == pytest.approx(-0.5)
    assert r.loc[0, "lgd"] == 0.0


def test_tarifa_mayor_que_recuperacion_se_recorta_a_uno():
    # EAD = 2000; neto = 100 − 300 = −200; LGD sin recorte = 1 + 0.1 = 1.1 → 1
    r = calcular_lgd(_montos([["1", CO, 2000.0, 0.0, 100.0, 300.0]]))
    assert r.loc[0, "lgd_sin_recorte"] == pytest.approx(1.1)
    assert r.loc[0, "lgd"] == 1.0


@pytest.mark.parametrize("prncp", [5000.0, 5000.5])
def test_ead_no_positiva_queda_sin_lgd(prncp):
    # EAD = 0 o −0.5: no se puede dividir → NaN y lgd_valida = False ([DECIDIDO, Camila, 24-oct]: excluir y contar)
    r = calcular_lgd(_montos([["1", CO, 5000.0, prncp, 10.0, 1.0]]))
    assert r.loc[0, "ead"] <= 0
    assert math.isnan(r.loc[0, "lgd_sin_recorte"])
    assert math.isnan(r.loc[0, "lgd"])
    assert not bool(r.loc[0, "lgd_valida"])


@pytest.mark.parametrize("col", ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"])
def test_na_en_un_campo_queda_sin_lgd(col):
    m = _montos([["1", CO, 5000.0, 1000.0, 100.0, 10.0]])
    m[col] = np.nan
    r = calcular_lgd(m)
    assert math.isnan(r.loc[0, "lgd"])
    assert not bool(r.loc[0, "lgd_valida"])


def test_calcular_lgd_conserva_indice_y_no_modifica_la_entrada():
    m = _montos([["1", CO, 10000.0, 2000.0, 1000.0, 200.0], ["2", CO, 5000.0, 1000.0, 0.0, 0.0]])
    m.index = [7, 3]
    copia = m.copy()
    r = calcular_lgd(m)
    assert list(r.index) == [7, 3]
    pd.testing.assert_frame_equal(m, copia)
    validas = r.loc[r["lgd_valida"], "lgd"]
    assert ((validas >= 0) & (validas <= 1)).all()


def test_ead_no_positiva_se_excluye_de_la_poblacion_final():
    # [DECIDIDO, Camila, 24-oct]: EAD ≤ 0 se excluye de la población de LGD y se cuenta en el resumen
    m = _montos([
        ["1", CO, 5000.0, 1000.0, 0.0, 0.0],            # EAD 4000 → entra
        ["2", CO, 5000.0, 5000.0, 0.0, 0.0],            # EAD 0 → fuera
        ["3", CO_POL, 5000.0, 5001.0, 0.0, 0.0],        # EAD −1 → fuera
        ["4", "Fully Paid", 5000.0, 1000.0, 0.0, 0.0],  # estado fuera
    ])
    assert mascara_poblacion_lgd(m).tolist() == [True, False, False, False]
    s = resumen_objetivo(m)
    assert s["n_poblacion"] == 3
    assert s["n_ead_no_positiva"] == 2
    assert s["n_lgd_valida"] == 1


# --- resumen -----------------------------------------------------------------------------------------------------

def test_resumen_objetivo_cuenta_bien():
    m = _montos([
        ["1", CO, 10000.0, 2000.0, 1000.0, 200.0],   # LGD 0.9
        ["2", CO, 5000.0, 1000.0, 0.0, 0.0],         # LGD 1 exacto
        ["3", CO_POL, 3000.0, 2000.0, 1500.0, 0.0],  # sin recorte −0.5 → 0
        ["4", CO, 2000.0, 0.0, 100.0, 300.0],        # sin recorte 1.1 → 1; fee > recoveries
        ["5", CO, 5000.0, 5000.0, 0.0, 0.0],         # EAD = 0
        ["6", CO, 5000.0, np.nan, 0.0, 0.0],         # NA
        ["7", CO, 1000.0, 0.0, 0.0, 5.0],            # recoveries = 0 con fee > 0; LGD sin recorte 1.005
        ["8", "Default", 1000.0, 0.0, 0.0, 0.0],
        ["9", "Late (31-120 days)", 1000.0, 0.0, 50.0, 0.0],  # recoveries > 0 fuera de la población
        ["10", "Fully Paid", 1000.0, 1000.0, 0.0, 0.0],
    ])
    s = resumen_objetivo(m)
    assert s["n_total"] == 10
    assert s["n_poblacion"] == 7
    assert s["poblacion_por_estado"] == {CO: 6, CO_POL: 1}
    assert s["excluidos_por_estado"] == {"Default": 1, "Late (31-120 days)": 1}
    assert s["n_ead_no_positiva"] == 1
    assert s["na_por_campo"] == {"funded_amnt": 0, "total_rec_prncp": 1, "recoveries": 0,
                                 "collection_recovery_fee": 0}
    assert s["n_lgd_valida"] == 5
    assert s["n_lgd_sin_recorte_menor_0"] == 1
    assert s["n_lgd_sin_recorte_mayor_1"] == 2
    assert s["n_lgd_igual_0"] == 1
    assert s["n_lgd_igual_1"] == 3
    assert s["n_fee_mayor_que_recoveries"] == 2
    assert s["n_recoveries_cero_con_fee"] == 1
    assert s["n_recoveries_positivas_fuera_poblacion"] == 1
    assert s["lgd_mediana"] == pytest.approx(1.0)
    assert s["lgd_media"] == pytest.approx((0.9 + 1 + 0 + 1 + 1) / 5)
    assert s["otros_estados"] == {"Fully Paid": 1}


def test_resumen_cuenta_estados_desconocidos_en_otros_estados():
    # " Charged Off" (con espacio) no es un estado de la población ni de los excluidos: debe contarse, no perderse
    m = _montos([["1", CO, 5000.0, 1000.0, 0.0, 0.0], ["2", " Charged Off", 5000.0, 1000.0, 0.0, 0.0],
                 ["3", " Charged Off", 5000.0, 1000.0, 0.0, 0.0], ["4", "Current", 5000.0, 1000.0, 0.0, 0.0]])
    s = resumen_objetivo(m)
    assert s["n_poblacion"] == 1
    assert s["otros_estados"] == {" Charged Off": 2, "Current": 1}


def test_mascara_poblacion_excluye_charged_off_con_na():
    m = _montos([["1", CO, 5000.0, 1000.0, np.nan, 0.0], ["2", CO, 5000.0, 1000.0, 0.0, 0.0]])
    assert mascara_poblacion_lgd(m).tolist() == [False, True]


@pytest.mark.lento
def test_estados_lgd_existen_en_montos_reales():
    ruta = Path(__file__).resolve().parents[1] / "data" / "processed" / "montos.parquet"
    if not ruta.exists():
        pytest.skip("falta data/processed/montos.parquet")
    estados = set(pd.read_parquet(ruta, columns=["loan_status"])["loan_status"].unique())
    assert set(ESTADOS_LGD) <= estados


def test_citas_del_docstring_coinciden_con_el_diccionario():
    """Las definiciones citadas en ``objetivo.py`` son literales de ``diccionario_loanstats.csv`` (regla: nunca parafrasear)."""
    import csv
    from pathlib import Path

    import src.lgd.objetivo as objetivo

    ruta = Path(__file__).resolve().parents[1] / "data" / "processed" / "diccionario_loanstats.csv"
    with ruta.open(encoding="utf-8") as f:
        diccionario = {fila["variable"]: fila["descripcion"] for fila in csv.DictReader(f)}
    for campo in ["loan_status", "funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]:
        assert f'``{campo}``: "{diccionario[campo]}"' in objetivo.__doc__
