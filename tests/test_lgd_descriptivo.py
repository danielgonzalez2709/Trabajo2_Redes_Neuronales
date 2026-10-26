"""Pruebas del descriptivo de la LGD observada y después del preprocesamiento (tarea D1b; Parte D §3.2b).

Sintéticos pequeños hechos aquí (valores a mano); la prueba lenta usa los archivos reales de ``data/processed/``.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.common.config import load_config
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json, save_json
from src.lgd.descriptivo import CLAVES_CUADRE, descriptivo_observado, descriptivo_preprocesado, generar
from src.pd.experimento import huella_archivo, huella_particion

RAIZ = Path(__file__).resolve().parents[1]
FEATURES = [{"nombre": "loan_amnt", "tipo": "num", "transformacion": "log1p"},
            {"nombre": "dti", "tipo": "num", "transformacion": "ninguna"},
            {"nombre": "term", "tipo": "cat"}]


def _pequeno() -> pd.DataFrame:
    """8 créditos con LGD, EAD, año y plazo elegidos para calcular el resumen a mano."""
    return pd.DataFrame({
        "id": [str(i) for i in range(8)],
        "issue_d": pd.to_datetime(["2011-03-01", "2011-07-01", "2012-01-01", "2013-02-01", "2013-05-01",
                                   "2014-01-01", "2014-06-01", "2014-09-01"]),
        "split": ["train", "train", "train", "val", "val", "test", "test", "test"],
        "term": [" 36 months", " 60 months", " 36 months", " 60 months", " 36 months", " 36 months",
                 " 60 months", " 36 months"],
        "ead": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "lgd": [0.0, 1.0, 1.0, 0.5, 0.2, 1.0, 0.9, 0.4],
        "lgd_sin_recorte": [-0.1, 1.0, 1.0, 0.5, 0.2, 1.0, 0.9, 0.4],
    })


def _grande(n: int = 3000) -> pd.DataFrame:
    """Sintético con relación creciente LGD ~ dti para la parte posterior al preprocesamiento."""
    rng = np.random.default_rng(0)
    dti = rng.uniform(0, 40, n)
    lgd = np.clip(0.5 + 0.01 * dti + rng.normal(0, 0.05, n), 0, 1)
    return pd.DataFrame({
        "id": [str(i) for i in range(n)],
        "issue_d": pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 3 * 365, n), unit="D"),
        "split": rng.choice(["train", "val", "test"], n, p=[0.5, 0.25, 0.25]),
        "loan_amnt": rng.lognormal(9, 1, n), "dti": dti, "term": rng.choice([" 36 months", " 60 months"], n),
        "ead": rng.uniform(100, 1000, n), "lgd": lgd, "lgd_sin_recorte": lgd,
    })


# --- observado -----------------------------------------------------------------------------------------------------

def test_histograma_suma_n_total_y_por_split():
    df = _pequeno()
    h = descriptivo_observado(df, n_bins=20)["histograma_lgd"]
    assert len(h["bordes"]) == 21 and h["bordes"][0] == 0.0 and h["bordes"][-1] == 1.0
    assert sum(h["conteos"]) == 8
    assert {s: sum(c) for s, c in h["por_split"].items()} == {"train": 3, "val": 2, "test": 3}
    assert h["conteos"][0] == 1 and h["conteos"][-1] == 3  # LGD = 0 en el primer bin; las tres LGD = 1 en el último


def test_histogramas_por_split_con_los_mismos_bordes_que_el_total():
    df = _grande()
    h = descriptivo_observado(df, n_bins=20)["histograma_lgd"]
    assert all(len(c) == 20 for c in h["por_split"].values())
    assert (np.sum([h["por_split"][s] for s in h["por_split"]], axis=0) == np.array(h["conteos"])).all()


def test_masa_exacta_en_0_y_1_y_recortes():
    o = descriptivo_observado(_pequeno())
    assert o["n"] == 8
    assert o["masa"]["n_lgd_igual_0"] == 1 and o["masa"]["n_lgd_igual_1"] == 3
    assert o["masa"]["fraccion_lgd_0"] == pytest.approx(1 / 8)
    assert o["masa"]["fraccion_lgd_1"] == pytest.approx(3 / 8)
    assert o["recortes"] == {"n_lgd_sin_recorte_menor_0": 1, "n_lgd_sin_recorte_mayor_1": 0}


def test_resumen_por_anio_a_mano():
    o = descriptivo_observado(_pequeno())
    a = o["por_anio"]
    assert list(a) == ["2011", "2012", "2013", "2014"]
    assert a["2011"]["n"] == 2 and a["2011"]["lgd"]["media"] == pytest.approx(0.5)
    assert a["2011"]["ead"]["media"] == pytest.approx(150.0)
    assert a["2013"]["lgd"]["media"] == pytest.approx(0.35) and a["2013"]["ead"]["mediana"] == pytest.approx(450.0)
    assert a["2014"]["lgd"]["media"] == pytest.approx(2.3 / 3)
    assert a["2014"]["fraccion_lgd_1"] == pytest.approx(1 / 3)
    assert o["fraccion_lgd_1_por_anio"] == pytest.approx({"2011": 0.5, "2012": 1.0, "2013": 0.0, "2014": 1 / 3})
    assert {k: v["fraccion_lgd_0"] for k, v in a.items()} == pytest.approx(
        {"2011": 0.5, "2012": 0.0, "2013": 0.0, "2014": 0.0})


def test_resumen_por_plazo_limpia_espacios():
    p = descriptivo_observado(_pequeno())["por_plazo"]
    assert set(p) == {"36 months", "60 months"}
    assert p["36 months"]["n"] == 5 and p["36 months"]["lgd"]["media"] == pytest.approx(0.52)
    assert p["36 months"]["ead"]["media"] == pytest.approx(460.0)
    assert p["60 months"]["lgd"]["media"] == pytest.approx(0.8)
    assert p["60 months"]["ead"]["media"] == pytest.approx(1300 / 3)


def test_resumen_por_split_y_cuantiles():
    o = descriptivo_observado(_pequeno(), cuantiles=(0.5,))
    t = o["por_split"]["train"]
    assert t["n"] == 3 and t["lgd"]["mediana"] == pytest.approx(1.0) and t["lgd"]["cuantiles"]["p50"] == 1.0
    assert t["lgd"]["desv"] == pytest.approx(float(np.std([0.0, 1.0, 1.0])))
    assert o["por_split"]["test"]["fraccion_lgd_1"] == pytest.approx(1 / 3)


def test_observado_es_puro():
    df = _pequeno()
    copia = df.copy()
    descriptivo_observado(df)
    pd.testing.assert_frame_equal(df, copia)


# --- preprocesado -----------------------------------------------------------------------------------------------------

def test_preprocesado_usa_el_preprocesador_sin_reajustar(monkeypatch):
    df = _grande()
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    antes = json.dumps(pre.to_dict(), sort_keys=True)

    def _prohibido(*a, **k):
        raise AssertionError("el descriptivo no debe reajustar el preprocesador")

    monkeypatch.setattr(Preprocesador, "fit", _prohibido)
    out = descriptivo_preprocesado(df, pre, umbral=0.9)
    assert json.dumps(pre.to_dict(), sort_keys=True) == antes
    assert out["n_por_split"]["train"] == int((df["split"] == "train").sum())
    # objetivo continuo: la "tasa" es la LGD media
    assert out["tasa_objetivo_por_split"]["val"] == pytest.approx(df.loc[df["split"] == "val", "lgd"].mean())
    assert out["relacion_con_objetivo_train"]["dti"]["monotona"] is True
    assert "notas_objetivo_continuo" in out
    notas = " ".join(out["notas_objetivo_continuo"].values())
    assert "n_grupos == 2" in notas and "purpose" in notas


def test_preprocesado_propaga_el_umbral():
    df = _grande()
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    out = descriptivo_preprocesado(df, pre, umbral=1.01)
    assert out["relacion_con_objetivo_train"]["dti"]["monotona"] is False


# --- generar ---------------------------------------------------------------------------------------------------------

def _escribir_insumos(tmp_path: Path, df: pd.DataFrame, pre: Preprocesador) -> dict:
    (tmp_path / "data").mkdir()
    df.to_parquet(tmp_path / "data/loans_lgd.parquet", index=False)
    save_json(tmp_path / "models/pre.json", pre.to_dict())
    resumen = {"n_lgd_valida": len(df), "n_lgd_igual_0": int((df["lgd"] == 0).sum()),
               "n_lgd_igual_1": int((df["lgd"] == 1).sum()),
               "n_lgd_sin_recorte_menor_0": int((df["lgd_sin_recorte"] < 0).sum()),
               "n_lgd_sin_recorte_mayor_1": int((df["lgd_sin_recorte"] > 1).sum())}
    save_json(tmp_path / "results/resumen.json", resumen)
    (tmp_path / "datos.yaml").write_text(yaml.safe_dump({"eda": {"umbral_monotonia": 0.9}}))
    return {"datos": {"config_datos": "datos.yaml",
                      "salida": {"loans_lgd": "data/loans_lgd.parquet", "preprocesador": "models/pre.json",
                                 "resumen": "results/resumen.json"}},
            "descriptivo": {"salida": "results/descriptivo.json", "n_bins_histograma": 20,
                            "cuantiles": [0.05, 0.5, 0.95]}}


def test_generar_escribe_json_con_esquema(tmp_path, monkeypatch):
    df = _grande()
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    cfg = _escribir_insumos(tmp_path, df, pre)
    monkeypatch.setattr(Preprocesador, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("reajuste")))
    out = generar(cfg, tmp_path)
    leido = load_json(tmp_path / "results/descriptivo.json")
    assert set(leido) == {"observado", "preprocesado", "fuente", "evidencia"}
    assert leido["evidencia"] == {"sha256_particion": huella_particion(df["id"], df["split"]),
                                  "sha256_preprocesador": huella_archivo(tmp_path / cfg["datos"]["salida"]
                                                                         ["preprocesador"])}
    assert not any(k.startswith("hash_") for k in leido["fuente"])
    assert leido["fuente"]["resumen_objetivo"] == "results/resumen.json"
    assert set(leido["observado"]) >= {"n", "histograma_lgd", "masa", "recortes", "por_split", "por_anio",
                                       "por_plazo", "fraccion_lgd_1_por_anio"}
    assert set(leido["preprocesado"]) >= {"psi_vs_train", "relacion_con_objetivo_train", "efecto_transformaciones",
                                          "estandarizacion", "correlacion_train"}
    assert len(leido["observado"]["histograma_lgd"]["bordes"]) == 21
    assert list(leido["observado"]["por_split"]["train"]["lgd"]["cuantiles"]) == ["p5", "p50", "p95"]
    assert out["observado"]["n"] == len(df)


@pytest.mark.parametrize("clave", [*CLAVES_CUADRE, "n_lgd_valida"])
def test_generar_falla_si_no_cuadra_con_resumen_objetivo(tmp_path, clave):
    df = _grande()
    pre = Preprocesador().fit(df[df["split"] == "train"], FEATURES)
    cfg = _escribir_insumos(tmp_path, df, pre)
    r = load_json(tmp_path / "results/resumen.json")
    r[clave] += 1
    save_json(tmp_path / "results/resumen.json", r)
    with pytest.raises(ValueError, match="resumen_objetivo"):
        generar(cfg, tmp_path)
    assert not (tmp_path / "results/descriptivo.json").exists()


def test_importar_descriptivo_no_carga_torch():
    codigo = "import sys, src.lgd.descriptivo; print('torch' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "False"


# --- datos reales ----------------------------------------------------------------------------------------------------

@pytest.mark.lento
def test_datos_reales(tmp_path):
    cfg = load_config(RAIZ / "configs/lgd/base.yaml")
    if not (RAIZ / cfg["datos"]["salida"]["loans_lgd"]).exists():
        pytest.skip("falta data/processed/loans_lgd.parquet (python -m src.lgd.preparar)")
    cfg["descriptivo"]["salida"] = str(tmp_path / "descriptivo.json")
    out = generar(cfg, RAIZ)
    assert out["observado"]["n"] == 43236
    assert sum(out["observado"]["histograma_lgd"]["conteos"]) == 43236
    assert sum(sum(c) for c in out["observado"]["histograma_lgd"]["por_split"].values()) == 43236
    assert (tmp_path / "descriptivo.json").exists()
