"""Pruebas de la construcción de ``loans_lgd.parquet`` y del preprocesador de la LGD (tarea D2; núcleo §5.1, §5.2).

Datos sintéticos pequeños hechos aquí; la prueba lenta revisa los archivos reales de ``data/processed/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json
from src.lgd.preparar import (CAMPOS_FORMULA, ajustar_preprocesador, columnas_contrato, construir_loans_lgd,
                               preparar)

RAIZ = Path(__file__).resolve().parents[1]
CO = "Charged Off"
CO_POL = "Does not meet the credit policy. Status:Charged Off"
FEATURES = [{"nombre": "loan_amnt", "tipo": "num", "transformacion": "ninguna"}, {"nombre": "term", "tipo": "cat"}]
NOMBRES = ["loan_amnt", "term"]
MONTOS = {"funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee", "ead", "lgd", "lgd_sin_recorte"}


def _sinteticos():
    """8 créditos: 5 Charged Off válidos (3 train, 1 val, 1 test), 1 con EAD = 0, 1 con NA, 1 Fully Paid y 1 Default."""
    filas = [
        # id, split, loan_amnt, term, loan_status, funded, prncp, recoveries, fee
        ("a", "train", 1000.0, " 36 months", CO, 1000.0, 200.0, 80.0, 0.0),        # LGD 0.9
        ("b", "train", 2000.0, " 60 months", CO_POL, 2000.0, 0.0, 0.0, 0.0),       # LGD 1
        ("c", "train", 3000.0, " 36 months", CO, 3000.0, 1000.0, 2500.0, 0.0),     # sin recorte −0.25 → 0
        ("d", "val", 100000.0, " 60 months", CO, 5000.0, 1000.0, 400.0, 0.0),      # LGD 0.9 (loan_amnt extremo)
        ("e", "test", 4000.0, " 36 months", CO, 4000.0, 2000.0, 1000.0, 0.0),      # LGD 0.5
        ("f", "train", 5000.0, " 36 months", CO, 5000.0, 5000.0, 0.0, 0.0),        # EAD = 0 → fuera y contado
        ("g", "train", 6000.0, " 36 months", CO, 6000.0, 1000.0, np.nan, 0.0),     # NA → fuera y contado
        ("h", "train", 7000.0, " 36 months", "Fully Paid", 7000.0, 7000.0, 0.0, 0.0),
        ("i", "val", 8000.0, " 60 months", "Default", 8000.0, 1000.0, 0.0, 0.0),
    ]
    loans_pd = pd.DataFrame({
        "id": [f[0] for f in filas],
        "issue_d": pd.to_datetime(["2011-01-01"] * len(filas)),
        "anio_origen": [2011] * len(filas),
        "target": [1] * len(filas),
        "split": [f[1] for f in filas],
        "loan_amnt": [f[2] for f in filas],
        "term": [f[3] for f in filas],
    })
    montos = pd.DataFrame({
        "id": [f[0] for f in filas], "loan_status": [f[4] for f in filas], "funded_amnt": [f[5] for f in filas],
        "total_rec_prncp": [f[6] for f in filas], "recoveries": [f[7] for f in filas],
        "collection_recovery_fee": [f[8] for f in filas],
    })
    return loans_pd, montos


# --- construir_loans_lgd -----------------------------------------------------------------------------------------

def test_columnas_del_contrato_en_orden():
    assert columnas_contrato(NOMBRES) == ["id", "issue_d", "split", "loan_amnt", "term", "funded_amnt",
                                          "total_rec_prncp", "recoveries", "collection_recovery_fee", "ead", "lgd",
                                          "lgd_sin_recorte"]
    assert CAMPOS_FORMULA == ("funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee")


def test_filtro_de_poblacion_y_columnas():
    loans_pd, montos = _sinteticos()
    df, resumen = construir_loans_lgd(loans_pd, montos, NOMBRES)
    assert df["id"].tolist() == ["a", "b", "c", "d", "e"]
    assert list(df.columns) == columnas_contrato(NOMBRES)
    for prohibida in ("lgd_valida", "loan_status", "target", "anio_origen"):
        assert prohibida not in df.columns
    assert df.index.tolist() == list(range(5))


def test_lgd_en_rango_y_valores_a_mano():
    loans_pd, montos = _sinteticos()
    df, _ = construir_loans_lgd(loans_pd, montos, NOMBRES)
    assert df["lgd"].between(0, 1).all()
    assert df["lgd"].tolist() == pytest.approx([0.9, 1.0, 0.0, 0.9, 0.5])
    assert df["lgd_sin_recorte"].tolist() == pytest.approx([0.9, 1.0, -0.25, 0.9, 0.5])
    assert df["ead"].tolist() == pytest.approx([800.0, 2000.0, 2000.0, 4000.0, 2000.0])
    assert df["ead"].dtype == float and df["lgd"].dtype == float and df["lgd_sin_recorte"].dtype == float


def test_split_y_variables_heredados_sin_cambios():
    loans_pd, montos = _sinteticos()
    df, _ = construir_loans_lgd(loans_pd, montos, NOMBRES)
    base = loans_pd.set_index("id").loc[df["id"]]
    assert df["split"].tolist() == base["split"].tolist()
    assert df["loan_amnt"].tolist() == base["loan_amnt"].tolist()
    assert df["term"].tolist() == base["term"].tolist()
    assert (df["issue_d"].to_numpy() == base["issue_d"].to_numpy()).all()


def test_id_desalineado_da_error():
    loans_pd, montos = _sinteticos()
    montos = montos.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="id"):
        construir_loans_lgd(loans_pd, montos, NOMBRES)


def test_largo_distinto_da_error():
    loans_pd, montos = _sinteticos()
    with pytest.raises(ValueError, match="id"):
        construir_loans_lgd(loans_pd, montos.iloc[:-1], NOMBRES)


def test_ead_no_positiva_y_na_excluidos_y_contados():
    loans_pd, montos = _sinteticos()
    df, resumen = construir_loans_lgd(loans_pd, montos, NOMBRES)
    assert "f" not in set(df["id"]) and "g" not in set(df["id"])
    assert resumen["n_ead_no_positiva"] == 1
    assert resumen["na_por_campo"]["recoveries"] == 1
    assert resumen["n_poblacion"] == 7
    assert resumen["n_lgd_valida"] == 5
    assert {s: v["n"] for s, v in resumen["lgd_por_split"].items()} == {"train": 3, "val": 1, "test": 1}
    assert resumen["lgd_por_split"]["train"]["fraccion_lgd_1"] == pytest.approx(1 / 3)
    assert "n_por_split" not in resumen  # una sola fuente del conteo por split: lgd_por_split[s]["n"]
    assert resumen["excluidos_por_estado"]["Default"] == 1


def test_split_vacio_queda_con_n_cero():
    loans_pd, montos = _sinteticos()
    loans_pd["split"] = loans_pd["split"].replace("test", "val")
    _, resumen = construir_loans_lgd(loans_pd, montos, NOMBRES)
    assert resumen["lgd_por_split"]["test"] == {"n": 0}
    assert resumen["lgd_por_split"]["val"]["n"] == 2


def test_features_sin_montos_en_construccion():
    loans_pd, montos = _sinteticos()
    with pytest.raises(ValueError, match="posterior"):
        construir_loans_lgd(loans_pd, montos, ["loan_amnt", "recoveries"])


def test_ajustar_preprocesador_rechaza_montos():
    loans_pd, montos = _sinteticos()
    df, _ = construir_loans_lgd(loans_pd, montos, NOMBRES)
    with pytest.raises(ValueError, match="posterior"):
        ajustar_preprocesador(df, [*FEATURES, {"nombre": "recoveries", "tipo": "num"}], (1.0, 99.0))


# --- preparar (artefactos) ---------------------------------------------------------------------------------------

def _escribir_entorno(tmp_path: Path) -> dict:
    loans_pd, montos = _sinteticos()
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    loans_pd.to_parquet(tmp_path / "data/processed/loans_pd.parquet", index=False)
    montos.to_parquet(tmp_path / "data/processed/montos.parquet", index=False)
    (tmp_path / "configs/features.yaml").write_text(yaml.safe_dump({"features": FEATURES}), encoding="utf-8")
    (tmp_path / "configs/datos.yaml").write_text(yaml.safe_dump({
        "features": "configs/features.yaml", "preprocesamiento": {"percentiles_recorte": [0.0, 100.0]}}),
        encoding="utf-8")
    return {"datos": {
        "config_datos": "configs/datos.yaml",
        "loans_pd": "data/processed/loans_pd.parquet",
        "montos": "data/processed/montos.parquet",
        "salida": {"loans_lgd": "data/processed/loans_lgd.parquet", "preprocesador": "models/preprocesador_lgd.json",
                   "resumen": "results/data/lgd/resumen_objetivo.json"},
    }}


def test_preparar_escribe_los_tres_artefactos(tmp_path):
    cfg = _escribir_entorno(tmp_path)
    resumen = preparar(cfg, tmp_path)
    df = pd.read_parquet(tmp_path / "data/processed/loans_lgd.parquet")
    assert list(df.columns) == columnas_contrato(NOMBRES)
    assert len(df) == 5
    assert load_json(tmp_path / "results/data/lgd/resumen_objetivo.json")["n_lgd_valida"] == 5
    assert {s: v["n"] for s, v in resumen["lgd_por_split"].items()} == {"train": 3, "val": 1, "test": 1}
    pre = Preprocesador.from_dict(load_json(tmp_path / "models/preprocesador_lgd.json"))
    assert not (set(pre.nombres_salida()) & MONTOS)
    assert {v["nombre"] for v in pre.variables} == set(NOMBRES)


def test_preprocesador_se_ajusta_solo_con_train(tmp_path):
    cfg = _escribir_entorno(tmp_path)
    preparar(cfg, tmp_path)
    pre = load_json(tmp_path / "models/preprocesador_lgd.json")
    loan = next(v for v in pre["variables"] if v["nombre"] == "loan_amnt")
    # train de la población LGD: a, b, c → loan_amnt 1000, 2000, 3000 (f y g quedan fuera; d = 100000 es val)
    assert loan["mediana"] == pytest.approx(2000.0)
    assert loan["media"] == pytest.approx(2000.0)
    assert loan["recorte"] == pytest.approx([1000.0, 3000.0])
    term = next(v for v in pre["variables"] if v["nombre"] == "term")
    assert term["categorias"][-1] == "OTRA"


# --- archivos reales ---------------------------------------------------------------------------------------------

@pytest.mark.lento
def test_loans_lgd_real_esquema_y_tamano():
    ruta = RAIZ / "data/processed/loans_lgd.parquet"
    if not ruta.exists():
        pytest.skip("falta data/processed/loans_lgd.parquet (python -m src.lgd.preparar)")
    features = yaml.safe_load((RAIZ / "configs/features.yaml").read_text())["features"]
    nombres = [f["nombre"] for f in features]
    df = pd.read_parquet(ruta)
    assert list(df.columns) == columnas_contrato(nombres)
    assert len(df) == 43_236
    assert df["lgd"].between(0, 1).all() and df["lgd"].notna().all()
    assert (df["ead"] > 0).all()
    assert set(df["split"]) == {"train", "val", "test"}
    pre = Preprocesador.from_dict(load_json(RAIZ / "models/preprocesador_lgd.json"))
    assert not (set(pre.nombres_salida()) & MONTOS)
    # misma partición, fecha y variables que loans_pd, por id
    base = pd.read_parquet(RAIZ / "data/processed/loans_pd.parquet").set_index("id").loc[df["id"]].reset_index()
    pd.testing.assert_frame_equal(df[["id", "issue_d", "split", *nombres]], base[["id", "issue_d", "split", *nombres]])
    # el preprocesador guardado es exactamente un reajuste sobre el train de loans_lgd
    percentiles = tuple(yaml.safe_load((RAIZ / "configs/datos.yaml").read_text())["preprocesamiento"]["percentiles_recorte"])
    reajuste = Preprocesador(percentiles).fit(df[df["split"] == "train"], features).to_dict()
    assert load_json(RAIZ / "models/preprocesador_lgd.json") == reajuste
