"""A5 — Preparación de datos sobre un CSV pequeño con el formato del crudo (esquema = núcleo §5.1)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json
from src.datos.preparar import preparar

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = RAIZ / "configs" / "features.yaml"
ESTADOS = ["Fully Paid", "Charged Off", "Current", "Late (31-120 days)", "Issued"]
CAMPOS_RESUMEN = ["archivo_crudo", "sha256", "fecha_descarga", "n_registros_crudo", "n_variables_crudo", "periodo",
                  "conteo_por_loan_status", "n_por_split", "tasa_malos_por_split", "tasa_malos_por_anio",
                  "fraccion_na_por_anio"]


@pytest.fixture
def entorno(tmp_path: Path) -> tuple[dict, Path]:
    rng = np.random.default_rng(0)
    n = 400
    meses = pd.date_range("2010-01-01", "2015-12-01", freq="MS")
    crudo = pd.DataFrame({
        "id": np.arange(n), "member_id": np.arange(n),
        "issue_d": meses[rng.integers(0, len(meses), n)].strftime("%b-%Y"),
        "loan_status": rng.choice(ESTADOS, n),
        "loan_amnt": rng.uniform(1000, 35000, n), "term": rng.choice([" 36 months", " 60 months"], n),
        "emp_length": rng.choice(["< 1 year", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n), "annual_inc": rng.lognormal(11, 0.5, n),
        "purpose": rng.choice(["credit_card", "debt_consolidation"], n), "dti": rng.uniform(0, 40, n),
        "delinq_2yrs": rng.integers(0, 3, n), "earliest_cr_line": ["Jan-1995"] * n,
        "inq_last_6mths": rng.integers(0, 5, n), "open_acc": rng.integers(1, 20, n),
        "pub_rec": rng.integers(0, 2, n), "revol_bal": rng.uniform(0, 30000, n),
        "revol_util": rng.uniform(0, 100, n), "total_acc": rng.integers(5, 40, n),
        "recoveries": 0.0,  # columna que no debe llegar al parquet
    })
    (tmp_path / "raw").mkdir()
    crudo.to_csv(tmp_path / "raw" / "loan.csv", index=False)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "features.yaml").write_text(FEATURES.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = {
        "crudo": {"csv": "raw/loan.csv", "url": "u", "fecha_descarga": "2026-10-23"},
        "features": "configs/features.yaml",
        "salida": {"carpeta": "processed", "preprocesador": "models/pre.json"},
        "particion": {"train_hasta": "2012-12-31", "val_hasta": "2013-12-31", "test_hasta": "2014-12-31"},
        "preprocesamiento": {"percentiles_recorte": [1.0, 99.0]},
        "base_seed": 0,
    }
    return cfg, tmp_path


def test_esquema_del_parquet_y_resumen(entorno):
    cfg, raiz = entorno
    resumen = preparar(cfg, raiz)
    df = pd.read_parquet(raiz / "processed" / "loans_pd.parquet")
    nombres = [f["nombre"] for f in yaml.safe_load(FEATURES.read_text(encoding="utf-8"))["features"]]
    assert list(df.columns) == ["id", "issue_d", "anio_origen", "target", "split", *nombres]
    assert set(df["target"]) <= {0, 1} and df["target"].notna().all()
    assert df["issue_d"].max() <= pd.Timestamp("2014-12-31")
    assert "recoveries" not in df.columns and "loan_status" not in df.columns
    assert resumen["n_modelado"] == len(df) == sum(resumen["n_por_split"].values())
    assert resumen["n_registros_crudo"] == 400 and resumen["n_variables_crudo"] == 20
    assert load_json(raiz / "processed" / "resumen_datos.json") == resumen
    # tipos del contrato §5.1
    assert df["id"].map(type).eq(str).all() and pd.api.types.is_datetime64_any_dtype(df["issue_d"])
    assert pd.api.types.is_integer_dtype(df["anio_origen"]) and str(df["target"].dtype) == "int8"
    assert set(df["split"]) <= {"train", "val", "test"}
    # campos del contrato §5.1 (más los adicionales del blog)
    assert set(CAMPOS_RESUMEN) <= set(resumen)
    # los resueltos posteriores a test_hasta (2015) se cuentan y no entran
    crudo = pd.read_csv(raiz / "raw" / "loan.csv")
    fechas = pd.to_datetime(crudo["issue_d"], format="%b-%Y")
    resueltos_2015 = ((fechas > "2014-12-31") & crudo["loan_status"].isin(["Fully Paid", "Charged Off",
                                                                            "Late (31-120 days)"])).sum()
    assert resueltos_2015 > 0 and resumen["n_resueltos_posteriores_a_test"] == resueltos_2015


def test_preprocesador_ajustado_solo_con_train(entorno):
    cfg, raiz = entorno
    preparar(cfg, raiz)
    df = pd.read_parquet(raiz / "processed" / "loans_pd.parquet")
    pre = Preprocesador.from_dict(load_json(raiz / "models" / "pre.json"))
    X = pre.transform(df[df["split"] == "train"])
    assert np.allclose(X[:, pre.nombres_salida().index("dti")].mean(), 0, atol=1e-5)
    assert X.shape[0] == (df["split"] == "train").sum()
