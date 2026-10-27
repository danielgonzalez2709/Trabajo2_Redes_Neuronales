"""Pruebas del cierre de hipótesis (Parte C §3.4, tarea C6; enunciado P1.2).

Por hipótesis se reúnen cuatro fuentes: EDA de train (A), SHAP del MLP (C4), tasas de malos observadas en test con
los grupos/cortes de TRAIN y la compuerta de relevancia (permutación del MLP en test). Veredicto:
confirmada (EDA, SHAP y test a favor y permutación relevante) / refutada (≥ 2 en contra y ninguna a favor) /
no concluyente (lo demás). Logística y PSI son contraste sin voto.

Bloques:
1. señales (EDA/test numéricas y categóricas, SHAP, compuerta);
2. veredicto: juguetes de cada rama y motivo;
3. sensibilidad (u = 0,3 / 0,7 y sin compuerta) y matiz;
4. fuga: grupos de test con los cortes de train (deciles y conteos con valores no vistos);
5. comprobación de que los grupos de train reproducen hipotesis_eda.json;
6. datos reales (lee los JSON ya guardados; no corre SHAP): orden, evidencia copiada, huellas, formato;
7. gancho de ``shap_pd.manejador_mlp`` (monkeypatch, sin correr SHAP).
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.common.results import load_json
from src.datos.eda import UMBRAL_MONOTONIA, grupos_ordenados, tendencia_numerica
from src.scorecard import hipotesis as hip
from src.scorecard.hipotesis import (
    ARCHIVO_HIPOTESIS,
    AVISO_CEGUERA,
    RUTAS_POR_DEFECTO,
    UMBRAL_RHO_SHAP,
    VEREDICTOS,
    Z_PERMUTACION,
    asignar_grupos,
    comprobar_train,
    construir_hipotesis,
    cortes_de_train,
    entrada_hipotesis,
    escribir,
    grupos_test_con_cortes_de_train,
    relevante,
    senal_categorica,
    senal_direccion,
    senal_shap,
    tendencia_por_grupos,
    veredicto,
)

RAIZ = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------------------------------------------
# Juguetes
# ------------------------------------------------------------------------------------------------------------------
H_NUM = {"id": "HX", "enunciado": "A mayor x, mayor riesgo.", "variable": "x", "direccion": "+"}
H_CAT = {"id": "HC", "enunciado": "A es más riesgosa que B.", "variable": "c", "mas_riesgosa": "A",
         "referencia": "B"}


def _eda_num(bajo=0.1, alto=0.2, rho=1.0, monotona=True):
    return {**H_NUM, "evidencia_eda": {"spearman_grupos": rho, "tasa_grupo_bajo": bajo, "tasa_grupo_alto": alto,
                                       "n_grupos": 3, "monotona": monotona,
                                       "tasas_por_grupo": [bajo, (bajo + alto) / 2, alto]},
            "veredicto_eda": "la apoya"}


def _shap(var="x", rho=0.9, phi=None, n=None, caida=(0.01, 0.001)):
    if phi is None:
        d = {"tipo": "num", "spearman": rho, "n": 1000}
    else:
        d = {"tipo": "cat", "phi_medio": phi, "n": n}
    return {"direccion": {var: d},
            "importancia_global": [{"nombre": var, "media_abs_phi": 0.1, "puesto": 1}],
            "permutacion": {"k": 5, "mlp": {"variables": [{"nombre": var, "caida_auc_media": caida[0],
                                                             "caida_auc_desv": caida[1]}]}}}


def _logistica_num(beta=0.3):
    return {"nombre": "x", "tipo": "num", "beta": beta,
            "tabla": [{"percentil": 1, "valor": 0.0, "puntos": 5.0}, {"percentil": 50, "valor": 1.0, "puntos": 0.0},
                      {"percentil": 99, "valor": 2.0, "puntos": -5.0}]}


def _test_num(senal="a_favor", monotona=True, rho=1.0):
    return {"tasas_por_grupo": [0.1, 0.2, 0.3], "n_por_grupo": [10, 10, 10], "tasa_grupo_bajo": 0.1,
            "tasa_grupo_alto": 0.3, "spearman_grupos": rho, "monotona": monotona, "senal": senal}


def _entrada(eda=None, shap=None, logistica=None, test=None, psi=0.05):
    return entrada_hipotesis(H_NUM, eda or _eda_num(), shap or _shap(), logistica or _logistica_num(),
                             test or _test_num(), psi)


# ------------------------------------------------------------------------------------------------------------------
# 1. Señales
# ------------------------------------------------------------------------------------------------------------------
def test_1a_constantes():
    assert UMBRAL_RHO_SHAP == 0.5
    assert Z_PERMUTACION == 2
    assert VEREDICTOS == ("confirmada", "refutada", "no concluyente")
    assert ARCHIVO_HIPOTESIS == "hipotesis.json"
    assert AVISO_CEGUERA == ("Las cifras del EDA, de SHAP y de la permutación ya se conocían (C4) cuando se fijó la "
                             "regla; la regla solo es ciega frente a las tasas de test, que no se habían calculado.")


@pytest.mark.parametrize("bajo,alto,rho,signo,esperada", [
    (0.1, 0.2, 0.8, 1, "a_favor"), (0.2, 0.1, -0.8, 1, "en_contra"), (0.1, 0.2, -0.3, 1, "nula"),
    (0.2, 0.1, -0.8, -1, "a_favor"), (0.1, 0.2, 0.8, -1, "en_contra"), (0.1, 0.1, 0.0, 1, "nula"),
    (0.2, 0.2, 0.8, 1, "nula")])  # a = 0, b > 0
def test_1b_senal_direccion(bajo, alto, rho, signo, esperada):
    assert senal_direccion(bajo, alto, rho, signo) == esperada


def test_1c_senal_categorica():
    assert senal_categorica(0.3, 0.2) == "a_favor"
    assert senal_categorica(0.2, 0.3) == "en_contra"
    assert senal_categorica(0.2, 0.2) == "nula"
    assert senal_categorica(None, 0.2) == "nula"
    assert senal_categorica(float("nan"), 0.2) == "nula"


def test_1d_senal_shap():
    d = _shap(rho=0.6)["direccion"]["x"]
    assert senal_shap(d, H_NUM) == "a_favor"
    assert senal_shap(d, {**H_NUM, "direccion": "-"}) == "en_contra"
    assert senal_shap(d, H_NUM, umbral=0.7) == "nula"
    assert senal_shap(_shap(rho=0.5)["direccion"]["x"], H_NUM) == "a_favor"  # s·ρ ≥ u
    assert senal_shap(_shap(rho=-0.5)["direccion"]["x"], H_NUM) == "en_contra"  # s·ρ = −u exacto
    dc = _shap("c", phi={"A": 0.2, "B": -0.1}, n={"A": 50, "B": 50})["direccion"]["c"]
    assert senal_shap(dc, H_CAT) == "a_favor"
    assert senal_shap(dc, {**H_CAT, "mas_riesgosa": "B", "referencia": "A"}) == "en_contra"
    sin_b = _shap("c", phi={"A": 0.2}, n={"A": 50})["direccion"]["c"]
    assert senal_shap(sin_b, H_CAT) == "nula"


def test_1e_relevante():
    assert relevante(0.01, 0.001)
    assert not relevante(0.001, 0.001)
    assert not relevante(0.002, 0.001)  # 0,002 − 2·0,001 = 0 no es > 0
    assert relevante(0.002, 0.001, z=1)


# ------------------------------------------------------------------------------------------------------------------
# 2. Veredicto
# ------------------------------------------------------------------------------------------------------------------
def test_2a_confirmada():
    v, motivo = veredicto({"eda": "a_favor", "shap": "a_favor", "test": "a_favor"}, True)
    assert v == "confirmada" and motivo
    e = _entrada()
    assert e["veredicto"] == "confirmada"


def test_2b_refutada():
    v, motivo = veredicto({"eda": "en_contra", "shap": "en_contra", "test": "nula"}, True)
    assert v == "refutada" and "en contra" in motivo
    e = _entrada(eda=_eda_num(0.2, 0.1, -0.9), shap=_shap(rho=-0.9), test=_test_num("en_contra", rho=-1.0))
    assert e["veredicto"] == "refutada"


def test_2c_no_concluyente_por_discrepancia():
    v, motivo = veredicto({"eda": "a_favor", "shap": "en_contra", "test": "a_favor"}, True)
    assert v == "no concluyente" and "discrepan" in motivo
    e = _entrada(shap=_shap(rho=-0.9))
    assert e["veredicto"] == "no concluyente"
    assert "condicional" in e["matiz"]  # marginal (EDA/test) frente a condicional (SHAP)


def test_2d_no_concluyente_por_senal_nula():
    v, motivo = veredicto({"eda": "a_favor", "shap": "nula", "test": "a_favor"}, True)
    assert v == "no concluyente" and "nula" in motivo and "SHAP" in motivo


def test_2e_no_concluyente_por_permutacion():
    v, motivo = veredicto({"eda": "a_favor", "shap": "a_favor", "test": "a_favor"}, False)
    assert v == "no concluyente" and "permutación" in motivo
    v2, _ = veredicto({"eda": "a_favor", "shap": "a_favor", "test": "a_favor"}, False, usar_compuerta=False)
    assert v2 == "confirmada"


def test_2g_una_en_contra_y_dos_nulas_no_es_refutada():
    v, motivo = veredicto({"eda": "en_contra", "shap": "nula", "test": "nula"}, True)
    assert v == "no concluyente" and "nula" in motivo


def test_2h_motivo_nombra_todas_las_condiciones():
    v, motivo = veredicto({"eda": "a_favor", "shap": "a_favor", "test": "en_contra"}, False)  # como H9
    assert v == "no concluyente"
    assert "discrepan" in motivo and "permutación" in motivo
    _, motivo2 = veredicto({"eda": "a_favor", "shap": "nula", "test": "en_contra"}, False)
    assert "discrepan" in motivo2 and "nula de SHAP" in motivo2 and "permutación" in motivo2
    _, motivo3 = veredicto({"eda": "a_favor", "shap": "nula", "test": "a_favor"}, True)
    assert "permutación" not in motivo3


def test_2f_dos_en_contra_con_una_a_favor_no_es_refutada():
    v, _ = veredicto({"eda": "en_contra", "shap": "en_contra", "test": "a_favor"}, True)
    assert v == "no concluyente"


# ------------------------------------------------------------------------------------------------------------------
# 3. Sensibilidad y matiz
# ------------------------------------------------------------------------------------------------------------------
def test_3a_sin_compuerta_cambia_el_veredicto():
    e = _entrada(shap=_shap(caida=(0.001, 0.001)))
    assert e["veredicto"] == "no concluyente"
    assert e["shap"]["relevante"] is False
    assert e["sensibilidad"]["sin_compuerta"] == "confirmada"
    assert e["cambia_con_sensibilidad"] is True
    assert "sin_compuerta" in e["matiz"] and "permutación" in e["matiz"]
    assert "PSI" in e["matiz"]


def test_3b_umbral_de_rho_cambia_el_veredicto():
    e = _entrada(shap=_shap(rho=0.4))
    assert e["veredicto"] == "no concluyente"
    assert e["sensibilidad"]["rho_0.3"] == "confirmada"
    assert e["sensibilidad"]["rho_0.7"] == "no concluyente"
    assert e["cambia_con_sensibilidad"] is True
    e2 = _entrada(shap=_shap(rho=0.6))
    assert e2["veredicto"] == "confirmada" and e2["sensibilidad"]["rho_0.7"] == "no concluyente"


def test_3c_sin_cambio():
    e = _entrada(shap=_shap(rho=0.95))
    assert e["sensibilidad"] == {"rho_0.3": "confirmada", "rho_0.7": "confirmada", "sin_compuerta": "confirmada"}
    assert e["cambia_con_sensibilidad"] is False


def test_3d_matiz_por_plantillas():
    e = _entrada(eda=_eda_num(rho=0.75, monotona=False), test=_test_num(monotona=False, rho=0.5),
                 logistica=_logistica_num(beta=-0.2))
    assert "no es monótona" in e["matiz"]
    assert "test" in e["matiz"]
    assert "logística" in e["matiz"]
    assert e["logistica"]["coincide"] is False
    assert e["matiz"].endswith(hip.NOTA_CAUSALIDAD)
    assert _entrada()["matiz"].endswith(hip.NOTA_CAUSALIDAD)


def test_3e_categorica_con_pocos_explicados():
    eda = {**H_CAT, "evidencia_eda": {"tasa_mas_riesgosa": 0.3, "tasa_referencia": 0.2}, "veredicto_eda": "la apoya"}
    sh = _shap("c", phi={"A": 0.4, "B": -0.1}, n={"A": 10, "B": 500})
    log = {"nombre": "c", "tipo": "cat", "categorias": [{"categoria": "A", "beta": 0.5, "puntos": -10.0},
                                                         {"categoria": "B", "beta": -0.1, "puntos": 3.0}]}
    test = {"mas_riesgosa": {"tasa": 0.4, "n": 100}, "referencia": {"tasa": 0.25, "n": 1000}, "senal": "a_favor"}
    e = entrada_hipotesis(H_CAT, eda, sh, log, test, 0.1)
    assert e["veredicto"] == "confirmada"
    assert e["shap"]["direccion"]["delta_phi"] == pytest.approx(0.5)
    assert e["logistica"]["coincide"] is True
    assert "n = 10" in e["matiz"]


# ------------------------------------------------------------------------------------------------------------------
# 4. Fuga: grupos de test con los cortes de train
# ------------------------------------------------------------------------------------------------------------------
def test_4a_deciles_de_train_en_test():
    rng = np.random.default_rng(0)
    x_tr = pd.Series(rng.uniform(0, 100, 2000))
    x_te = pd.Series(rng.uniform(50, 150, 2000))
    g_tr, g_te = grupos_test_con_cortes_de_train(x_tr, x_te)
    pd.testing.assert_series_equal(g_tr.astype(float), grupos_ordenados(x_tr).astype(float), check_names=False)
    _, bordes = pd.qcut(x_tr, 10, labels=False, retbins=True, duplicates="drop")
    bordes[0], bordes[-1] = -np.inf, np.inf
    esperado = pd.cut(x_te, bordes, labels=False)
    np.testing.assert_array_equal(g_te.to_numpy(dtype=float), esperado.to_numpy(dtype=float))
    assert (g_te == 9).mean() > 0.4  # test desplazado: cae sobre todo en el decil alto de train
    con_test = pd.qcut(x_te, 10, labels=False)  # recalcular con test daría otros grupos
    assert not np.array_equal(g_te.to_numpy(dtype=float), con_test.to_numpy(dtype=float))


def test_4b_conteos_con_valores_no_vistos():
    x_tr = pd.Series([0.0] * 900 + [1.0] * 80 + [2.0] * 15 + [7.0] * 5)
    x_te = pd.Series([0.0, 1.0, 2.0, 3.0, 9.0, np.nan])
    c = cortes_de_train(x_tr)
    assert c["tipo"] == "conteo"
    g_tr = asignar_grupos(x_tr, c)
    pd.testing.assert_series_equal(g_tr.astype(float), grupos_ordenados(x_tr).astype(float), check_names=False)
    g_te = asignar_grupos(x_te, c)
    tope = float(x_tr.quantile(0.99))
    ultimo = len(np.unique(x_tr.clip(upper=tope))) - 1
    assert g_te.iloc[0] == 0 and g_te.iloc[1] == 1
    assert g_te.iloc[3] == min(2, ultimo)  # 3 no visto → último valor de train ≤ 3
    assert g_te.iloc[4] == ultimo  # 9 > p99 de train → recortado
    assert pd.isna(g_te.iloc[5])  # vacíos excluidos


def test_4c_tendencia_por_grupos_igual_a_la_de_A():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=3000))
    y = pd.Series((rng.uniform(size=3000) < 1 / (1 + np.exp(-x))).astype(float))
    t = tendencia_por_grupos(grupos_ordenados(x), y)
    ref = tendencia_numerica(x, y)
    for k in ("spearman_grupos", "tasa_grupo_bajo", "tasa_grupo_alto", "n_grupos", "monotona", "tasas_por_grupo"):
        assert t[k] == ref[k], k
    assert sum(t["n_por_grupo"]) == 3000


# ------------------------------------------------------------------------------------------------------------------
# 5. Los grupos de train reproducen hipotesis_eda.json
# ------------------------------------------------------------------------------------------------------------------
def test_5a_comprobar_train():
    t = {"tasas_por_grupo": [0.1, 0.2]}
    comprobar_train(t, {"tasas_por_grupo": [0.1, 0.2 + 1e-13]}, "HX")
    with pytest.raises(ValueError, match="HX"):
        comprobar_train(t, {"tasas_por_grupo": [0.1, 0.21]}, "HX")
    with pytest.raises(ValueError, match="HX"):
        comprobar_train(t, {"tasas_por_grupo": [0.1]}, "HX")


def test_5b_eda_alterado_lanza(tmp_path):
    eda = load_json(RAIZ / RUTAS_POR_DEFECTO["hipotesis_eda"])
    eda[1]["evidencia_eda"]["tasas_por_grupo"][0] += 0.01
    ruta = tmp_path / "hipotesis_eda.json"
    ruta.write_text(json.dumps(eda), encoding="utf-8")
    with pytest.raises(ValueError, match="tasas_por_grupo"):
        construir_hipotesis({**_rutas(), "hipotesis_eda": ruta})


# ------------------------------------------------------------------------------------------------------------------
# 6. Datos reales (lee los archivos ya guardados)
# ------------------------------------------------------------------------------------------------------------------
def _rutas():
    return {k: RAIZ / v for k, v in RUTAS_POR_DEFECTO.items()}


@pytest.fixture(scope="module")
def real():
    return construir_hipotesis(_rutas())


def test_6a_once_hipotesis_en_orden(real):
    yaml = load_config(RAIZ / RUTAS_POR_DEFECTO["hipotesis_yaml"])["hipotesis"]
    assert [h["id"] for h in real["hipotesis"]] == [h["id"] for h in yaml]
    assert len(real["hipotesis"]) == 11
    for h, y in zip(real["hipotesis"], yaml):
        assert h["enunciado"] == y["enunciado"] and h["variable"] == y["variable"]


def test_6b_evidencia_copiada(real):
    eda = {h["id"]: h for h in load_json(RAIZ / RUTAS_POR_DEFECTO["hipotesis_eda"])}
    sh = load_json(RAIZ / RUTAS_POR_DEFECTO["shap"])
    sc = load_json(RAIZ / RUTAS_POR_DEFECTO["scorecard"])
    psi = load_json(RAIZ / RUTAS_POR_DEFECTO["post_preprocesamiento"])["psi_vs_train"]["test"]
    perm = {v["nombre"]: v for v in sh["permutacion"]["mlp"]["variables"]}
    imp = {v["nombre"]: v for v in sh["importancia_global"]}
    pv = {v["nombre"]: v for v in sc["puntos_por_variable"]}
    for h in real["hipotesis"]:
        v = h["variable"]
        assert {k: h["eda"][k] for k in eda[h["id"]]["evidencia_eda"]} == eda[h["id"]]["evidencia_eda"]
        assert h["eda"]["veredicto_eda"] == eda[h["id"]]["veredicto_eda"]
        assert h["shap"]["media_abs_phi"] == imp[v]["media_abs_phi"] and h["shap"]["puesto"] == imp[v]["puesto"]
        assert h["shap"]["n_variables"] == 15
        pm = h["shap"]["permutacion_mlp"]
        assert pm == {"caida_auc_media": perm[v]["caida_auc_media"], "caida_auc_desv": perm[v]["caida_auc_desv"],
                      "k": sh["permutacion"]["k"]}
        assert h["psi_test"] == psi[v]
        if "direccion" in h:
            assert h["shap"]["direccion"]["spearman"] == sh["direccion"][v]["spearman"]
            assert h["logistica"]["beta"] == pv[v]["beta"]
            tabla = {f["percentil"]: f["puntos"] for f in pv[v]["tabla"]}
            assert h["logistica"]["puntos_p1"] == tabla[1] and h["logistica"]["puntos_p99"] == tabla[99]
        else:
            cats = {c["categoria"]: c for c in pv[v]["categorias"]}
            assert h["logistica"]["beta_mas_riesgosa"] == cats[h["mas_riesgosa"]]["beta"]
            assert h["logistica"]["beta_referencia"] == cats[h["referencia"]]["beta"]
            phi = sh["direccion"][v]["phi_medio"]
            assert h["shap"]["direccion"]["delta_phi"] == phi[h["mas_riesgosa"]] - phi[h["referencia"]]


def test_6c_test_con_cortes_de_train_reales(real):
    datos = pd.read_parquet(RAIZ / RUTAS_POR_DEFECTO["datos"])
    tr, te = datos[datos["split"] == "train"], datos[datos["split"] == "test"]
    h = next(h for h in real["hipotesis"] if h["variable"] == "dti")
    x_tr = tr["dti"].dropna()
    _, bordes = pd.qcut(x_tr, 10, labels=False, retbins=True, duplicates="drop")
    bordes[0], bordes[-1] = -np.inf, np.inf
    m = te["dti"].notna()
    g = pd.cut(te.loc[m, "dti"], bordes, labels=False)
    tasas = te.loc[m, "target"].astype(float).groupby(g).mean()
    np.testing.assert_allclose(h["test"]["tasas_por_grupo"], tasas.to_numpy(), rtol=0, atol=1e-15)
    assert sum(h["test"]["n_por_grupo"]) == int(m.sum())
    t1 = next(h for h in real["hipotesis"] if h["variable"] == "term")
    tasa60 = te.loc[te["term"].astype("string").str.strip() == "60 months", "target"].astype(float).mean()
    assert t1["test"]["mas_riesgosa"]["tasa"] == pytest.approx(tasa60, abs=1e-15)


def test_6d_huellas(real, tmp_path):
    sh = load_json(RAIZ / RUTAS_POR_DEFECTO["shap"])
    assert real["evidencia"] == sh["evidencia"]
    otro = copy.deepcopy(sh)
    otro["evidencia"]["sha256_particion"] = "0" * 64
    ruta = tmp_path / "shap_mlp.json"
    ruta.write_text(json.dumps(otro), encoding="utf-8")
    with pytest.raises(ValueError, match="evidencia"):
        construir_hipotesis({**_rutas(), "shap": ruta})


def test_6e_formato(real):
    assert set(real) == {"criterio", "hipotesis", "evidencia"}
    assert AVISO_CEGUERA in json.dumps(real["criterio"], ensure_ascii=False)
    json.dumps(real, allow_nan=False)
    claves = {"id", "enunciado", "variable", "eda", "shap", "logistica", "test", "psi_test", "veredicto", "motivo",
              "sensibilidad", "cambia_con_sensibilidad", "matiz"}
    for h in real["hipotesis"]:
        assert claves <= set(h)
        assert ("direccion" in h) != ("mas_riesgosa" in h and "referencia" in h)
        assert h["veredicto"] in VEREDICTOS
        assert set(h["sensibilidad"]) == {"rho_0.3", "rho_0.7", "sin_compuerta"}
        assert all(v in VEREDICTOS for v in h["sensibilidad"].values())
        assert h["cambia_con_sensibilidad"] == any(v != h["veredicto"] for v in h["sensibilidad"].values())
        for bloque in ("eda", "shap", "test"):
            assert h[bloque]["senal"] in ("a_favor", "en_contra", "nula")
        assert h["matiz"].endswith(hip.NOTA_CAUSALIDAD)


def test_6f_escribir(tmp_path):
    ruta = escribir(_rutas(), tmp_path)
    assert ruta == tmp_path / ARCHIVO_HIPOTESIS
    leido = load_json(ruta)
    assert len(leido["hipotesis"]) == 11


# ------------------------------------------------------------------------------------------------------------------
# 7. Gancho de manejador_mlp
# ------------------------------------------------------------------------------------------------------------------
def _cfg(tmp_path):
    return {"salida": str(tmp_path), "rutas": {"datos": str(RAIZ / RUTAS_POR_DEFECTO["datos"]),
                                               "preprocesador": str(RAIZ / RUTAS_POR_DEFECTO["preprocesador"])}}


def test_7a_gancho_escribe_si_hay_scorecard(tmp_path, monkeypatch, capsys):
    from src.scorecard import shap_pd
    sc = load_json(RAIZ / RUTAS_POR_DEFECTO["scorecard"])
    sh = load_json(RAIZ / RUTAS_POR_DEFECTO["shap"])
    (tmp_path / "scorecard.json").write_text(json.dumps(sc), encoding="utf-8")
    monkeypatch.setattr(shap_pd, "construir_scorecard_mlp", lambda cfg: ({}, sh))
    monkeypatch.setattr(shap_pd, "_imprimir_resumen", lambda *a: None)
    assert shap_pd.manejador_mlp(_cfg(tmp_path)) == 0
    assert (tmp_path / ARCHIVO_HIPOTESIS).is_file()
    assert ARCHIVO_HIPOTESIS in capsys.readouterr().out


def test_7b_gancho_avisa_sin_scorecard(tmp_path, monkeypatch, capsys):
    from src.scorecard import shap_pd
    sh = load_json(RAIZ / RUTAS_POR_DEFECTO["shap"])
    monkeypatch.setattr(shap_pd, "construir_scorecard_mlp", lambda cfg: ({}, sh))
    monkeypatch.setattr(shap_pd, "_imprimir_resumen", lambda *a: None)
    assert shap_pd.manejador_mlp(_cfg(tmp_path)) == 0
    assert not (tmp_path / ARCHIVO_HIPOTESIS).exists()
    assert "Aviso" in capsys.readouterr().out


def test_3f_matiz_con_spearman_none():
    eda = _eda_num(rho=None, monotona=False)
    e = _entrada(eda=eda, test=_test_num("nula", monotona=False, rho=None))
    assert "n/d" in e["matiz"]
    assert e["eda"]["senal"] == "nula"


def test_3g_matiz_marginal_frente_a_condicional():
    e = _entrada(test=_test_num("en_contra", rho=-1.0))  # EDA y SHAP a favor, test en contra
    assert "cambia entre train y test" in e["matiz"]
    assert "condicional del MLP" not in e["matiz"]
    e2 = _entrada(shap=_shap(rho=-0.9))  # SHAP discrepa de EDA y de test
    assert "condicional del MLP" in e2["matiz"] and "cambia entre train y test" not in e2["matiz"]
    assert "cambia entre train y test" not in _entrada()["matiz"]
