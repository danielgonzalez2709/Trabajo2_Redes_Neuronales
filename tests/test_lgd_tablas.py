"""D8 — funciones del notebook 04_lgd (núcleo §5.8: cero lógica en el notebook): tablas desde los JSON de la Parte D,
salidas de la demo redirigidas y comparación de modelos."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.results import save_json
from src.lgd.tablas import (comparacion, describir_mlp, redirigir_salidas, tabla_efecto_transformaciones,
                            tabla_metricas, tabla_metricas_compacta, tabla_por_anio, tabla_por_plazo, tabla_por_split,
                            tabla_poblacion, tabla_psi, tabla_referencias, tabla_relacion_objetivo)
from src.viz.figs_lgd import insumos_de_corrida

RAIZ = Path(__file__).resolve().parents[1]


def _met(media, desv=0.0, n=1):
    return {"media": media, "desv": desv, "mejor": media, "peor": media, "n": n}


def _resumen() -> dict:
    lin = {k: _met(v) for k, v in {"mae_val": 0.10, "rmse_val": 0.20, "mae_test": 0.11, "rmse_test": 0.21}.items()}
    mlp = {k: _met(v, d, 3) for k, (v, d) in {"mae_val": (0.09, 0.002), "rmse_val": (0.19, 0.003),
                                               "mae_test": (0.10, 0.01), "rmse_test": (0.20, 0.004)}.items()}
    return {  # summary.json por modelo (núcleo §5.4, como B; D10) + referencias (D §3.4)
        "lineal": {"n_parametros": 49, "n_configuraciones_busqueda": 1, "tiempo_total_s": 0.5, "metricas": lin},
        "mlp": {"n_parametros": 900, "n_configuraciones_busqueda": 30, "tiempo_total_s": 12.0, "metricas": mlp,
                "tiempo_busqueda_s": 9.0, "tiempo_semillas_s": 3.0, "semillas": [0, 1, 2], "semilla_mediana": 1,
                "config": {"capas": [8, 4], "perdida": "bce_fraccional"}, "config_de_busqueda": True,
                "busqueda": {"tipo": "optuna", "origen": "lgd-hpo: mejor de 30 ensayos"}},
        "referencias": {
            "constante_media": {"valor": 0.9, "n_parametros": 1, "n_configuraciones_busqueda": 1,
                                "metricas": {"mae_val": 0.12, "rmse_val": 0.22, "mae_test": 0.13, "rmse_test": 0.23}},
            "constante_mediana": {"valor": 0.95, "n_parametros": 1, "n_configuraciones_busqueda": 1,
                                  "metricas": {"mae_val": 0.14, "rmse_val": 0.24, "mae_test": 0.08, "rmse_test": 0.25}},
        },
    }


def test_tabla_metricas_lineal_mlp_y_constantes():
    t = tabla_metricas(_resumen())
    assert list(t.index) == ["lineal", "mlp", "constante_media", "constante_mediana"]
    assert t.loc["mlp", "MAE test"] == pytest.approx(0.10)
    assert t.loc["mlp", "MAE test (desv)"] == pytest.approx(0.01)
    assert t.loc["mlp", "MAE val (desv)"] == pytest.approx(0.002)
    assert t.loc["mlp", "RMSE test (desv)"] == pytest.approx(0.004)
    assert t.loc["constante_mediana", "MAE test"] == pytest.approx(0.08)
    assert t.loc["constante_mediana", "MAE test (desv)"] == 0.0
    assert t.loc["mlp", "semillas"] == 3 and t.loc["lineal", "semillas"] == 1
    assert t.loc["mlp", "parámetros"] == 900 and t.loc["mlp", "configuraciones"] == 30
    assert t.loc["mlp", "tiempo (s)"] == pytest.approx(12.0)
    assert np.isnan(t.loc["constante_media", "tiempo (s)"])


def test_comparacion_diferencias_de_mae_test():
    c = comparacion(_resumen())
    assert c["metrica"] == "mae_test"
    assert c["valores"] == {"mlp": 0.10, "lineal": 0.11, "constante_media": 0.13, "constante_mediana": 0.08}
    assert c["mlp_menos"]["lineal"] == pytest.approx(-0.01)
    assert c["mlp_menos"]["constante_media"] == pytest.approx(-0.03)
    assert c["mlp_menos"]["constante_mediana"] == pytest.approx(0.02)
    assert "mlp" not in c["mlp_menos"]
    assert c["mejor"] == "constante_mediana"
    assert c["desv_mlp"] == pytest.approx(0.01)


def test_comparacion_con_otra_metrica():
    c = comparacion(_resumen(), metrica="rmse_test")
    assert c["valores"] == {"mlp": 0.20, "lineal": 0.21, "constante_media": 0.23, "constante_mediana": 0.25}
    assert c["mejor"] == "mlp" and c["desv_mlp"] == pytest.approx(0.004)
    assert c["mlp_menos"]["constante_mediana"] == pytest.approx(-0.05)


def test_comparacion_metrica_inexistente_es_error():
    with pytest.raises(ValueError):
        comparacion(_resumen(), metrica="r2_test")


def test_tabla_metricas_mlp_fuera_de_la_busqueda_sin_configuraciones():
    """Con ``config_de_busqueda`` false, ``n_configuraciones_busqueda`` es null (núcleo §5.4) y la tabla no lo
    inventa."""
    r = _resumen()
    r["mlp"].update(n_configuraciones_busqueda=None, tiempo_busqueda_s=None, busqueda=None, config_de_busqueda=False)
    t = tabla_metricas(r)
    assert pd.isna(t.loc["mlp", "configuraciones"])
    assert tabla_metricas_compacta(r).loc["mlp", "configuraciones"] == "—"


def test_describir_mlp_con_y_sin_busqueda():
    r = _resumen()
    texto = describir_mlp(r["mlp"])
    assert "lgd-hpo: mejor de 30 ensayos" in texto and "[8, 4]" in texto and "bce_fraccional" in texto
    assert "búsqueda 9.0 s" in texto and "semillas 3.0 s" in texto
    r["mlp"].update(tiempo_busqueda_s=None, busqueda=None, config_de_busqueda=False)
    texto = describir_mlp(r["mlp"])
    assert "config_de_busqueda: false" in texto and "búsqueda —" in texto and "semillas 3.0 s" in texto


def test_tabla_metricas_compacta():
    t = tabla_metricas_compacta(_resumen(), decimales=3)
    assert t.loc["mlp", "MAE test"] == "0.100 ± 0.010"
    assert t.loc["mlp", "MAE val"] == "0.090 ± 0.002"
    assert t.loc["constante_media", "MAE test"] == "0.130"
    assert t.loc["mlp", "parámetros"] == 900 and t.loc["mlp", "semillas"] == 3
    assert t.loc["constante_media", "tiempo (s)"] == "—"


def test_redirigir_salidas_no_toca_la_config_original(tmp_path):
    cfg = {"salida": "results/data/lgd_demo",
           "modelos": {"lineal": "models/lgd_lineal_demo.joblib", "mlp": "models/lgd_mlp_demo.pt"}, "mlp": {"x": 1}}
    nueva = redirigir_salidas(cfg, tmp_path)
    assert nueva["salida"] == str(tmp_path)
    assert nueva["modelos"] == {"lineal": str(tmp_path / "lgd_lineal_demo.joblib"),
                                "mlp": str(tmp_path / "lgd_mlp_demo.pt")}
    assert cfg["salida"] == "results/data/lgd_demo" and cfg["modelos"]["mlp"] == "models/lgd_mlp_demo.pt"
    nueva["mlp"]["x"] = 2
    assert cfg["mlp"]["x"] == 1  # copia profunda
    # ruta absoluta: entrenar_semillas hace raiz / cfg["salida"] y con una ruta absoluta queda la temporal
    assert Path(".") / nueva["salida"] == tmp_path


def _obs() -> dict:
    def grupo(n, media, f1, f0):
        return {"n": n, "lgd": {"n": n, "media": media, "mediana": media}, "ead": {"n": n, "media": 10.0,
                                                                                    "mediana": 9.0},
                "fraccion_lgd_1": f1, "fraccion_lgd_0": f0}
    return {"n": 6,
            "por_split": {"train": grupo(3, 0.9, 0.2, 0.1), "val": grupo(2, 0.8, 0.5, 0.0),
                          "test": grupo(1, 1.0, 1.0, 0.0)},
            "por_anio": {"2012": grupo(3, 0.9, 0.2, 0.1), "2013": grupo(2, 0.8, 0.5, 0.0),
                         "2014": grupo(1, 1.0, 1.0, 0.0)},
            "split_por_anio": {"2012": ["train"], "2013": ["val"], "2014": ["test"]}}


def test_tabla_por_split_y_por_anio():
    s = tabla_por_split(_obs())
    assert list(s.index) == ["train", "val", "test"]
    assert s.loc["val", "créditos"] == 2 and s.loc["val", "LGD media"] == pytest.approx(0.8)
    assert s.loc["test", "fracción LGD = 1"] == pytest.approx(1.0)
    assert s.loc["train", "fracción LGD = 0"] == pytest.approx(0.1)
    assert s.loc["train", "EAD mediana"] == pytest.approx(9.0)
    a = tabla_por_anio(_obs())
    assert list(a.index) == ["2012", "2013", "2014"]
    assert a.loc["2013", "split"] == "val" and a.loc["2014", "créditos"] == 1


def test_tabla_por_plazo():
    obs = _obs()
    obs["por_plazo"] = {"36 months": obs["por_split"]["train"], "60 months": obs["por_split"]["val"]}
    t = tabla_por_plazo(obs)
    assert list(t.index) == ["36 months", "60 months"]
    assert t.loc["60 months", "créditos"] == 2 and t.loc["36 months", "EAD mediana"] == pytest.approx(9.0)
    assert t.loc["60 months", "fracción LGD = 1"] == pytest.approx(0.5)


def _pre() -> dict:
    return {
        "efecto_transformaciones": {
            "train": {"x": {"imputado": 0.1, "recortado_abajo": 0.2, "recortado_arriba": 0.3},
                      "c": {"desconocido": 0.4, "a_otra": 0.5}},
            "val": {"x": {"imputado": 0.6, "recortado_abajo": 0.7, "recortado_arriba": 0.8},
                    "c": {"desconocido": 0.9, "a_otra": 0.05}}},
        "psi_vs_train": {"val": {"x": 0.5, "c": 0.3}, "test": {"x": 0.02, "c": 0.4}},
        "relacion_con_objetivo_train": {
            "x": {"spearman_grupos": -0.5, "tasa_grupo_bajo": 0.9, "tasa_grupo_alto": 0.8, "n_grupos": 10,
                  "monotona": False, "tasas_por_grupo": [0.9, 0.8]},
            "p": {"spearman_grupos": 1.0, "tasa_grupo_bajo": 0.7, "tasa_grupo_alto": 0.95, "n_grupos": 2,
                  "monotona": True, "tasas_por_grupo": [0.7, 0.95]}},
    }


def test_tabla_efecto_transformaciones():
    t = tabla_efecto_transformaciones(_pre())
    assert t.loc["x", ("train", "imputado")] == pytest.approx(0.1)
    assert t.loc["x", ("val", "recortado_arriba")] == pytest.approx(0.8)
    assert t.loc["c", ("val", "desconocido")] == pytest.approx(0.9)
    assert t.loc["c", ("train", "a_otra")] == pytest.approx(0.5)
    assert np.isnan(t.loc["c", ("train", "imputado")])  # las categóricas no se imputan ni recortan


def test_tabla_psi_ordenada_por_test():
    t = tabla_psi(_pre())  # val y test ordenan distinto: manda test
    assert list(t.index) == ["c", "x"]
    assert t.loc["x", "val"] == pytest.approx(0.5) and t.loc["c", "test"] == pytest.approx(0.4)


def test_tabla_psi_ordenada_por_val():
    assert list(tabla_psi(_pre(), por="val").index) == ["x", "c"]


def test_tabla_psi_split_inexistente_es_error():
    with pytest.raises(ValueError):
        tabla_psi(_pre(), por="train")


def test_tabla_referencias_valor_de_train():
    t = tabla_referencias(_resumen())
    assert list(t.index) == ["constante_media", "constante_mediana"]
    assert t.loc["constante_media", "valor (de train)"] == pytest.approx(0.9)
    assert t.loc["constante_mediana", "valor (de train)"] == pytest.approx(0.95)


def test_tabla_relacion_objetivo():
    t = tabla_relacion_objetivo(_pre())
    assert t.loc["x", "ρ (Spearman por grupos)"] == pytest.approx(-0.5)
    assert t.loc["p", "n_grupos"] == 2 and bool(t.loc["p", "monótona"]) is True
    assert t.loc["x", "LGD media grupo bajo"] == pytest.approx(0.9)
    assert t.loc["p", "LGD media grupo alto"] == pytest.approx(0.95)


def test_tabla_poblacion_cuadra_con_el_resumen():
    r = {"n_total": 100, "n_poblacion": 10, "poblacion_por_estado": {"Charged Off": 9, "X": 1},
         "excluidos_por_estado": {"Default": 2}, "otros_estados": {"Fully Paid": 88},
         "n_lgd_igual_0": 1, "n_lgd_igual_1": 4, "n_lgd_sin_recorte_menor_0": 5, "n_lgd_sin_recorte_mayor_1": 6,
         "n_ead_no_positiva": 7}
    t = tabla_poblacion(r)
    assert t.loc["créditos en loans_pd", "valor"] == 100
    assert t.loc["población LGD", "valor"] == 10
    assert t.loc["población: Charged Off", "valor"] == 9
    assert t.loc["excluidos: Default", "valor"] == 2
    assert t.loc["LGD = 1", "valor"] == 4 and t.loc["LGD = 0", "valor"] == 1
    assert t.loc["LGD < 0 antes del recorte", "valor"] == 5
    assert t.loc["LGD > 1 antes del recorte", "valor"] == 6
    assert t.loc["EAD ≤ 0 (excluidos)", "valor"] == 7
    assert t.loc["fuera de la población: Fully Paid", "valor"] == 88


def test_insumos_de_corrida_lee_la_carpeta_de_la_demo(tmp_path):
    (tmp_path / "semillas").mkdir()
    save_json(tmp_path / "summary.json", _resumen())
    pd.DataFrame({"id": ["a"], "y": [1.0], "pred_lineal": [0.9], "pred_mlp": [0.95]}).to_parquet(
        tmp_path / "pred_test.parquet", index=False)
    for k in (0, 1):
        save_json(tmp_path / "semillas" / f"mlp_seed{k}.json", {"seed": k})
    save_json(tmp_path / "semillas" / "lineal_seed0.json", {"seed": 0})
    d = insumos_de_corrida(tmp_path, obs={"n": 1})
    assert d["obs"] == {"n": 1}
    assert d["resumen"]["mlp"]["n_parametros"] == 900
    assert list(d["pred"].columns) == ["id", "y", "pred_lineal", "pred_mlp"]
    assert [s["seed"] for s in d["semillas"]] == [0, 1]


def test_insumos_de_corrida_sin_summary_da_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        insumos_de_corrida(tmp_path, obs={})


def _summary_real_con_formato_d10() -> bool:
    """TODO(retirar-tras-D4b): hasta que D4b regenere los resultados, el summary.json versionado tiene el formato
    anterior a D10 (raíz ``modelos``)."""
    from src.common.results import load_json

    ruta = RAIZ / "results/data/lgd/summary.json"
    return ruta.is_file() and "modelos" not in load_json(ruta)


@pytest.mark.skipif(not _summary_real_con_formato_d10(),
                    reason="faltan los resultados de D4 o tienen el formato anterior a D10 (los regenera D4b)")
def test_tablas_con_los_resultados_reales():
    from src.common.results import load_json

    base = RAIZ / "results/data/lgd"
    t = tabla_metricas(load_json(base / "summary.json"))
    assert t.loc["mlp", "semillas"] == 30
    obs = load_json(base / "descriptivo.json")["observado"]
    assert tabla_por_split(obs)["créditos"].sum() == obs["n"]
    assert tabla_poblacion(load_json(base / "resumen_objetivo.json")).loc["población LGD", "valor"] == obs["n"]
    assert tabla_por_plazo(obs)["créditos"].sum() == obs["n"]
    pre = load_json(base / "descriptivo.json")["preprocesado"]
    assert set(tabla_psi(pre).columns) == {"val", "test"}
    assert len(tabla_relacion_objetivo(pre)) == len(pre["relacion_con_objetivo_train"])
    assert len(tabla_efecto_transformaciones(pre)) == len(pre["efecto_transformaciones"]["train"])
