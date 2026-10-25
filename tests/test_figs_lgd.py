"""D6 — las figuras de la Parte D se regeneran desde results/ sin reentrenar (núcleo §3, §7.1; Parte D §3.2b y §3.4)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.viz.figs_lgd import (BANDAS_LGD, _etiqueta_mae, cuantiles_qq, generar, mae_por_banda, mae_semilla_mediana,
                              tabla_error_por_banda)

RAIZ = Path(__file__).resolve().parents[1]
NOMBRES = ["lgd-distribucion-predicha.png", "lgd-ead-plazo.png", "lgd-error-por-banda.png", "lgd-histograma.png",
           "lgd-por-anio.png", "lgd-qq.png", "lgd-semillas.png"]


def _resumen(media: float) -> dict:
    return {"n": 3, "media": media, "mediana": media, "desv": 0.1,
            "cuantiles": {f"p{q}": media * (0.5 + q / 100) for q in (1, 5, 25, 50, 75, 95, 99)}}  # monótonos


def _escribir_insumos(raiz: Path) -> None:
    """Insumos sintéticos pequeños con las mismas claves que los reales de results/data/lgd/."""
    base = raiz / "results" / "data" / "lgd"
    (base / "semillas").mkdir(parents=True)
    rng = np.random.default_rng(0)
    n = 200
    y = np.clip(rng.uniform(0.3, 1.2, n), 0, 1)
    pd.DataFrame({"id": [str(i) for i in range(n)], "y": y, "pred_lineal": rng.uniform(0.8, 1.0, n),
                  "pred_mlp": rng.uniform(0.8, 1.0, n)}).to_parquet(base / "pred_test.parquet")
    metricas = {k: {"media": 0.08, "desv": 0.001, "mejor": 0.07, "peor": 0.09, "n": 3}
                for k in ("mae_val", "rmse_val", "mae_test", "rmse_test")}
    resumen = {
        "modelos": {"lineal": {"n_parametros": 5, "metricas": metricas},
                    "mlp": {"n_parametros": 50, "metricas": metricas, "semillas": [0, 1, 2], "semilla_mediana": 1}},
        "referencias": {
            "constante_media": {"valor": 0.9, "metricas": {"mae_val": 0.08, "rmse_val": 0.09, "mae_test": 0.081,
                                                          "rmse_test": 0.09}},
            "constante_mediana": {"valor": 0.95, "metricas": {"mae_val": 0.08, "rmse_val": 0.09, "mae_test": 0.07,
                                                             "rmse_test": 0.09}}},
    }
    (base / "summary.json").write_text(json.dumps(resumen))
    for s in range(3):
        (base / "semillas" / f"mlp_seed{s}.json").write_text(json.dumps(
            {"modelo": "mlp", "seed": s, "metricas": {"mae_val": 0.08, "rmse_val": 0.09, "mae_test": 0.075 + s / 1000,
                                                      "rmse_test": 0.09}}))
    bordes = np.linspace(0, 1, 5).tolist()
    anios = {"2011": "train", "2012": "train", "2013": "val", "2014": "test"}
    descriptivo = {"observado": {
        "n": 30,
        "histograma_lgd": {"bordes": bordes, "conteos": [1, 2, 3, 24],
                           "por_split": {"train": [1, 1, 1, 7], "val": [0, 1, 1, 8], "test": [0, 0, 1, 9]}},
        "masa": {"n_lgd_igual_0": 1, "n_lgd_igual_1": 12, "fraccion_lgd_0": 1 / 30, "fraccion_lgd_1": 0.4},
        "por_split": {s: {"n": 10, "lgd": _resumen(0.9), "ead": _resumen(5000.0), "fraccion_lgd_1": 0.4,
                          "fraccion_lgd_0": 0.0} for s in ("train", "val", "test")},
        "por_anio": {a: {"n": 7, "lgd": _resumen(0.85 + i / 100), "ead": _resumen(5000.0),
                         "fraccion_lgd_1": 0.2 + i / 10, "fraccion_lgd_0": 0.0} for i, a in enumerate(anios)},
        "split_por_anio": {a: [s] for a, s in anios.items()},
        "por_plazo": {t: {"n": 15, "lgd": _resumen(0.9 + i / 100), "ead": _resumen(5000.0 * (i + 1)),
                          "fraccion_lgd_1": 0.4, "fraccion_lgd_0": 0.0}
                      for i, t in enumerate(("36 months", "60 months"))},
        "fraccion_lgd_1_por_anio": {a: 0.2 + i / 10 for i, a in enumerate(anios)},
    }, "preprocesado": {}}
    (base / "descriptivo.json").write_text(json.dumps(descriptivo))


def _archivos(raiz: Path) -> set[Path]:
    return {p for p in raiz.rglob("*") if p.is_file()}


def test_generar_produce_las_siete_figuras_solo_en_destino(tmp_path):
    raiz = tmp_path / "repo"
    _escribir_insumos(raiz)
    destino = tmp_path / "figuras"
    antes = _archivos(tmp_path)
    rutas = generar(raiz=raiz, destino=destino)
    assert sorted(r.name for r in rutas) == NOMBRES
    assert all(r.parent == destino and r.stat().st_size > 0 for r in rutas)
    assert _archivos(tmp_path) - antes == set(rutas)


def test_mae_por_banda_a_mano():
    y = np.array([0.0, 0.2, 0.5, 0.6, 0.8, 0.9, 0.99, 1.0, 1.0])
    pred = np.array([0.3, 0.4, 0.5, 0.9, 0.9, 0.9, 0.9, 0.8, 0.9])
    bandas = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.99), (0.99, 1.0)]
    tabla = mae_por_banda(y, pred, bandas)
    assert [b["n"] for b in tabla] == [3, 2, 2, 2]
    assert sum(b["n"] for b in tabla) == len(y)  # conservación: cada y en [0, 1] cae en exactamente una banda
    # [0, 0.5]: 0.3, 0.2, 0 · (0.5, 0.8]: 0.3, 0.1 · (0.8, 0.99]: 0, 0.09 · (0.99, 1]: 0.2, 0.1
    np.testing.assert_allclose([b["mae"] for b in tabla], [0.5 / 3, 0.2, 0.045, 0.15])
    assert [(b["desde"], b["hasta"]) for b in tabla] == bandas


def test_mae_por_banda_vacia_es_nan():
    tabla = mae_por_banda(np.array([1.0]), np.array([0.9]), [(0.0, 0.5), (0.5, 1.0)])
    assert tabla[0]["n"] == 0 and np.isnan(tabla[0]["mae"])
    assert tabla[1]["n"] == 1 and tabla[1]["mae"] == pytest.approx(0.1)


def test_bandas_cubren_cero_a_uno_sin_huecos():
    assert BANDAS_LGD[0][0] == 0.0 and BANDAS_LGD[-1][1] == 1.0
    assert all(a[1] == b[0] for a, b in zip(BANDAS_LGD, BANDAS_LGD[1:]))


def test_cuantiles_qq_coinciden_con_np_quantile():
    rng = np.random.default_rng(1)
    y, pred = rng.uniform(size=101), rng.normal(size=57)
    probs = np.linspace(0, 1, 11)
    q_obs, q_pred = cuantiles_qq(y, pred, probs)
    np.testing.assert_array_equal(q_obs, np.quantile(y, probs))
    np.testing.assert_array_equal(q_pred, np.quantile(pred, probs))


def test_tabla_error_por_banda_usa_la_mediana():
    y = np.array([0.0, 0.6, 0.9, 1.0])
    pred = pd.DataFrame({"y": y, "pred_lineal": [0.5, 0.5, 0.5, 0.5], "pred_mlp": [0.9, 0.9, 0.9, 0.9]})
    referencias = {"constante_media": {"valor": 0.2}, "constante_mediana": {"valor": 0.7}}
    tabla = tabla_error_por_banda(pred, referencias)
    assert set(tabla) == {"lineal", "mlp", "constante_mediana"}
    assert tabla["constante_mediana"] == mae_por_banda(y, np.full(len(y), 0.7))
    assert tabla["lineal"] == mae_por_banda(y, pred["pred_lineal"].to_numpy())
    assert tabla["mlp"] == mae_por_banda(y, pred["pred_mlp"].to_numpy())


def test_mae_semilla_mediana():
    semillas = [{"seed": s, "metricas": {"mae_test": 0.07 + s / 100}} for s in (0, 1, 2)]
    assert mae_semilla_mediana(semillas, {"modelos": {"mlp": {"semilla_mediana": 2}}}) == (2, pytest.approx(0.09))
    with pytest.raises(ValueError, match="semilla mediana"):
        mae_semilla_mediana(semillas, {"modelos": {"mlp": {"semilla_mediana": 7}}})


def test_etiqueta_mae_de_banda_vacia():
    assert _etiqueta_mae(float("nan")) == "—"
    assert _etiqueta_mae(0.12345) == "0.123"


def _datos_en_memoria(raiz: Path) -> dict:
    base = raiz / "results" / "data" / "lgd"
    return {"obs": json.loads((base / "descriptivo.json").read_text())["observado"],
            "resumen": json.loads((base / "summary.json").read_text()),
            "pred": pd.read_parquet(base / "pred_test.parquet"),
            "semillas": [json.loads(r.read_text()) for r in (base / "semillas").glob("*.json")]}


def test_generar_con_datos_en_memoria_no_lee_results(tmp_path):
    origen = tmp_path / "origen"
    _escribir_insumos(origen)
    datos = _datos_en_memoria(origen)
    vacia = tmp_path / "vacia"  # sin results/: si generar leyera archivos, fallaría
    vacia.mkdir()
    rutas = generar(raiz=vacia, destino=tmp_path / "figs", datos=datos)
    assert sorted(r.name for r in rutas) == NOMBRES


@pytest.mark.parametrize("archivo, tarea", [("descriptivo.json", "D1b"), ("summary.json", "D4"),
                                            ("pred_test.parquet", "D4")])
def test_insumo_faltante_da_mensaje_claro(tmp_path, archivo, tarea):
    _escribir_insumos(tmp_path)
    (tmp_path / "results" / "data" / "lgd" / archivo).unlink()
    with pytest.raises(FileNotFoundError, match=f"falta .*{archivo}.*{tarea}"):
        generar(raiz=tmp_path, destino=tmp_path / "figs")


def test_sin_semillas_da_error(tmp_path):
    _escribir_insumos(tmp_path)
    for r in (tmp_path / "results" / "data" / "lgd" / "semillas").glob("*.json"):
        r.unlink()
    with pytest.raises(ValueError, match="semillas"):
        generar(raiz=tmp_path, destino=tmp_path / "figs")


@pytest.mark.lento
def test_generar_con_resultados_reales(tmp_path):
    rutas = generar(raiz=RAIZ, destino=tmp_path)
    assert sorted(r.name for r in rutas) == NOMBRES
    assert all(r.stat().st_size > 10_000 for r in rutas)


def test_figura_qq_usa_las_probs_recibidas(tmp_path, monkeypatch):
    from src.viz import figs_lgd

    probs_vistas = []
    original = figs_lgd.cuantiles_qq

    def espia(y, pred, probs=figs_lgd.PROBS_QQ):
        probs_vistas.append(probs)
        return original(y, pred, probs)

    monkeypatch.setattr(figs_lgd, "cuantiles_qq", espia)
    pred = pd.DataFrame({"y": [0.1, 0.5, 1.0], "pred_lineal": [0.4, 0.5, 0.6], "pred_mlp": [0.3, 0.5, 0.7]})
    probs = np.array([0.25, 0.5, 0.75])
    figs_lgd.figura_qq(pred, tmp_path, probs=probs)
    assert len(probs_vistas) == 2  # una llamada por modelo (lineal y MLP)
    assert all(np.array_equal(v, probs) for v in probs_vistas)


def test_mediana_y_rango_a_mano():
    from src.viz.figs_lgd import _mediana_y_rango

    mediana, rango = _mediana_y_rango([{"mediana": 10.0, "cuantiles": {"p25": 4.0, "p75": 30.0}}])
    np.testing.assert_array_equal(mediana, [10.0])
    np.testing.assert_array_equal(rango, [[6.0], [20.0]])  # [mediana − p25], [p75 − mediana]


def test_etiquetas_de_plazo_desde_features():
    from src.viz.figs_lgd import etiqueta_plazo, etiquetas_plazo

    etiquetas = etiquetas_plazo(RAIZ)  # configs/features.yaml real: term.categorias_es
    assert etiqueta_plazo("36 months", etiquetas) == "36 meses"
    assert etiqueta_plazo("60 months", etiquetas) == "60 meses"
    assert etiqueta_plazo("12 months", etiquetas) == "12 months"  # sin etiqueta: el código tal cual


def test_etiquetas_de_plazo_sin_config(tmp_path):
    from src.viz.figs_lgd import etiqueta_plazo, etiquetas_plazo

    etiquetas = etiquetas_plazo(tmp_path)
    assert etiquetas == {}
    assert etiqueta_plazo("36 months", etiquetas) == "36 months"
