"""El diccionario versionado (data/processed/diccionario_loanstats.csv) permite citar cada variable sin el crudo (núcleo §0.5)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
DIC = pd.read_csv(RAIZ / "data" / "processed" / "diccionario_loanstats.csv", dtype=str, keep_default_na=False)
DEF = dict(zip(DIC["variable"], DIC["descripcion"]))
CRUDO = RAIZ / "data" / "raw" / "credit-risk-dataset" / "LCDataDictionary.xlsx"


def test_estan_las_variables_del_modelo_y_los_montos():
    features = yaml.safe_load((RAIZ / "configs" / "features.yaml").read_text(encoding="utf-8"))["features"]
    montos = yaml.safe_load((RAIZ / "configs" / "datos.yaml").read_text(encoding="utf-8"))["montos"]["columnas"]
    nombres = [f["nombre"] for f in features if f["nombre"] != "antiguedad_credito_anios"]
    faltan = [n for n in [*nombres, "issue_d", "earliest_cr_line", *montos] if n not in DEF]
    assert not faltan


def test_las_citas_de_features_yaml_son_textuales():
    """Cada cita_diccionario de una variable directa coincide con el diccionario versionado, letra por letra."""
    for f in yaml.safe_load((RAIZ / "configs" / "features.yaml").read_text(encoding="utf-8"))["features"]:
        if f["nombre"] in DEF:
            assert f["cita_diccionario"] == DEF[f["nombre"]], f["nombre"]


@pytest.mark.skipif(not CRUDO.is_file(), reason="sin el diccionario crudo de Kaggle (data/raw no se versiona)")
def test_copia_literal_del_crudo():
    hoja = pd.read_excel(CRUDO, sheet_name="LoanStats", dtype=str)
    assert len(hoja) == len(DIC)
    assert hoja.iloc[:, 1].fillna("").tolist() == DIC["descripcion"].tolist()
