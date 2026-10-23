"""A2 — Variable objetivo (Parte A §3.3): mapeo literal del enunciado y error ante estados desconocidos."""

import pandas as pd
import pytest

from src.datos.objetivo import MAPEO_LOAN_STATUS, construir_objetivo


def test_tabla_del_enunciado_completa():
    malos = {k for k, v in MAPEO_LOAN_STATUS.items() if v == 1}
    buenos = {k for k, v in MAPEO_LOAN_STATUS.items() if v == 0}
    na = {k for k, v in MAPEO_LOAN_STATUS.items() if v is None}
    assert buenos == {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
    assert malos == {"Charged Off", "Default", "Late (31-120 days)",
                     "Does not meet the credit policy. Status:Charged Off"}
    assert na == {"Current", "Issued", "In Grace Period", "Late (16-30 days)"}


def test_construir_objetivo_mapea_cada_estado():
    estados = pd.Series(list(MAPEO_LOAN_STATUS))
    y = construir_objetivo(estados)
    assert y.dtype == "Int8"
    for estado, valor in zip(estados, y):
        esperado = MAPEO_LOAN_STATUS[estado]
        assert (pd.isna(valor) and esperado is None) or valor == esperado


@pytest.mark.parametrize("raro", ["Fully paid", "Charged Off ", "Late (31-120 Days)", "Otro"])
def test_estado_fuera_de_la_tabla_es_error(raro):
    with pytest.raises(ValueError, match="fuera de la tabla"):
        construir_objetivo(pd.Series(["Fully Paid", raro]))


def test_estado_vacio_es_error():
    with pytest.raises(ValueError, match="vacío"):
        construir_objetivo(pd.Series(["Fully Paid", None]))
