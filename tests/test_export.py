"""Pruebas del exportador de la app: ``pd_model.json`` y ``casos_paridad.json`` (Parte C §3.5, tarea C2; núcleo §5.5).

Solo se usa la interfaz pública de ``src.export.exportar_app`` y, para construir las referencias independientes, la de
los módulos que el exportador consume (preprocesador, logística, MLP, scorecard de C1, métricas comunes y config).

Datos y modelo: los REALES del repositorio (``data/processed/loans_pd.parquet``, ``models/preprocesador_pd.json``,
``models/pd_logistica.joblib`` de B3 y ``results/data/pd/summary.json``). El MLP de B5 aún no existe: el camino del MLP
se prueba con un ``MLPPD`` pequeño entrenado aquí con una etiqueta SINTÉTICA de regla fija (no el objetivo real) sobre
una submuestra de train; solo sirve para comparar el JSON con el modelo, ninguna cifra sale de él (núcleo §0.8).

Toda escritura va a ``tmp_path``; nunca a ``app/public/model`` (se comprueba con SHA-256).

Qué se prueba (numeración de la tarea C2):

1. Contrato §5.5: claves y tipos de primer nivel y de cada bloque.
2. Entradas del formulario: solo ``en_app``, orden de ``features.yaml``, claves por tipo, rejilla del paso (redondeo
   hacia afuera) con casos a mano, categóricas sin OTRA en el orden de ``categorias_es`` y moda de train.
3. ``preprocesamiento`` = ``Preprocesador.to_dict()`` del archivo.
4. ``red``: W salida × entrada, dimensiones encadenadas, activaciones; logística = una capa identidad.
5. ``scorecard`` = parámetros de C1; ``poblacion``: 101 percentiles del puntaje en test.
6. ``metricas`` = las de ``results/data/pd/summary.json``.
7. Referencia JSON vs modelo en TODO el test (logística real y MLP pequeño con relu, leaky_relu y tanh).
8. Casos de paridad: 1 000, cobertura de los grupos forzados, reproducibles, entradas crudas.
9. ``exportar`` escribe los dos archivos sin NaN ni inf.
10. ``validar_cfg_export`` rechaza configs inválidas.
11. ``configs/scorecard/export.yaml`` hereda ``base.yaml`` y es válida.

Fuga de información (núcleo §5.7): los estadísticos del formulario (mediana, recorte, moda) salen solo de train; de test
solo salen los percentiles del puntaje y las métricas (pruebas 2 y 5).
"""

import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.common.config import load_config
from src.common.metricas import auc, brier, ks
from src.common.preprocesamiento import Preprocesador
from src.common.results import load_json
from src.common.seeds import make_rng
from src.export.exportar_app import (
    ARCHIVO_MODELO,
    ARCHIVO_PARIDAD,
    MODELOS,
    RUTA_CONFIG,
    TOLERANCIA_PARIDAD,
    VERSION_CONTRATO,
    capas_logistica,
    capas_mlp,
    commit_actual,
    construir_casos_paridad,
    construir_pd_model,
    elegir_casos,
    entradas_app,
    exportar,
    forward_desde_json,
    inferir_desde_json,
    metricas_test,
    percentiles_score,
    preprocesar_desde_json,
    validar_cfg_export,
)
from src.pd.logistica import LogisticaPD
from src.pd.mlp import MLPPD
from src.scorecard.scorecard import bloque_scorecard, parametros_scorecard, score_desde_logit, score_desde_pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_EXPORT_YAML = RAIZ / "configs" / "scorecard" / "export.yaml"
RUTA_SUMMARY = RAIZ / "results" / "data" / "pd" / "summary.json"
CARPETA_APP_MODELO = RAIZ / "app" / "public" / "model"

CLAVES_PRIMER_NIVEL = {"version_contrato", "generado", "commit", "modelo", "entradas", "preprocesamiento", "red",
                       "scorecard", "poblacion", "metricas"}
CLAVES_EXPORT = {"modelo", "rutas", "destino", "n_casos_paridad", "n_forzados_por_grupo", "base_seed", "pasos"}
CLAVES_RUTAS = {"logistica", "mlp", "preprocesador", "datos", "features"}
CLAVES_NUM = {"nombre", "tipo", "etiqueta_es", "ayuda_es", "min", "max", "paso", "valor_defecto"}
CLAVES_CAT = {"nombre", "tipo", "etiqueta_es", "ayuda_es", "categorias", "valor_defecto", "etiquetas_categorias"}
ACTIVACIONES_JSON = {"relu", "leaky_relu", "tanh", "identidad"}
OTRA = "OTRA"
DESCONOCIDO = "DESCONOCIDO"

# Pasos del formulario que fija la config de la tarea C2 (decisión de Juan José).
PASOS_ESPERADOS = {"loan_amnt": 25, "annual_inc": 1000, "dti": 0.1, "delinq_2yrs": 1, "antiguedad_credito_anios": 1,
                   "inq_last_6mths": 1, "open_acc": 1, "pub_rec": 1, "revol_bal": 100, "revol_util": 0.1,
                   "total_acc": 1}

# Semilla propia de las pruebas (submuestra y MLP pequeño); no es la de la config.
SEMILLA_PRUEBA = 20261025
N_SUBMUESTRA_MLP = 5000  # filas de train para el MLP pequeño (4 000 entrenan, 1 000 deciden la parada)
N_VAL_MLP = 1000
CFG_MLP_PEQUENO = {"capas": [8, 4], "dropout": 0.1, "l2": 0.0, "optimizador": "adam", "lr": 1e-2, "momento": 0.0,
                   "batch": 256, "max_epocas": 3, "paciencia": 2, "device": "cpu", "pos_weight": 1.0}
# Cota del MLP pequeño (JSON en float64 vs MLPPD en float32), medida en las 74 154 filas de test el 25-oct:
# máx |dPD| = 2.357e-07 (relu), 2.243e-07 (leaky_relu), 2.316e-07 (tanh); máx |dscore| = 4.915e-05, 5.525e-05,
# 2.695e-05. Cota ≈ 10 × el máximo medido, redondeada hacia arriba: 3e-6 en la PD y 6e-4 en el puntaje. Solo compara la
# receta con el modelo de Python; la paridad de la app (núcleo §5.5) se mide contra la receta en casos_paridad.json.
COTA_MLP = {"pd": 3e-6, "score": 6e-4}

N_CASOS_PEQUENO = 40  # casos para comparar dos semillas sin elegir 1 000 dos veces
TOL_PERCENTILES = 1e-9  # percentiles del puntaje: mismo camino de Python que el exportador
TOL_REL_MULTIPLO = 1e-9 # tolerancia relativa para "es múltiplo del paso" (0.1 no es exacto en binario)


# ------------------------------------------------------------------------------------------------------------------
# Fixtures de módulo (el parquet y los modelos se cargan una vez)
# ------------------------------------------------------------------------------------------------------------------


def _con_rutas_absolutas(cfg: dict) -> dict:
    """Copia de la config con las rutas de ``export.rutas`` absolutas respecto de la raíz del repo (sin depender del
    directorio de trabajo de pytest)."""
    cfg = copy.deepcopy(cfg)
    cfg["export"]["rutas"] = {k: str(RAIZ / v) for k, v in cfg["export"]["rutas"].items()}
    return cfg


@pytest.fixture(scope="module")
def cfg_real() -> dict:
    """Config de ``configs/scorecard/export.yaml`` (modelo logística) con rutas absolutas."""
    return _con_rutas_absolutas(load_config(RUTA_EXPORT_YAML))


@pytest.fixture(scope="module")
def features() -> list[dict]:
    return load_config(RAIZ / "configs" / "features.yaml")["features"]


@pytest.fixture(scope="module")
def pre_json() -> dict:
    return load_json(RAIZ / "models" / "preprocesador_pd.json")


@pytest.fixture(scope="module")
def pre(pre_json) -> Preprocesador:
    return Preprocesador.from_dict(pre_json)


@pytest.fixture(scope="module")
def datos() -> pd.DataFrame:
    return pd.read_parquet(RAIZ / "data" / "processed" / "loans_pd.parquet")


@pytest.fixture(scope="module")
def df_train(datos) -> pd.DataFrame:
    return datos[datos["split"] == "train"].reset_index(drop=True)


@pytest.fixture(scope="module")
def df_test(datos) -> pd.DataFrame:
    return datos[datos["split"] == "test"].reset_index(drop=True)


@pytest.fixture(scope="module")
def logistica() -> LogisticaPD:
    return LogisticaPD.cargar(RAIZ / "models" / "pd_logistica.joblib")


@pytest.fixture(scope="module")
def params_sc(cfg_real) -> dict:
    return parametros_scorecard(bloque_scorecard(cfg_real))


@pytest.fixture(scope="module")
def modelo_json(cfg_real) -> dict:
    """``pd_model.json`` de la logística real, construido en memoria."""
    return construir_pd_model(cfg_real)


@pytest.fixture(scope="module")
def nombres_en_app(features) -> list[str]:
    return [f["nombre"] for f in features if f["en_app"]]


@pytest.fixture(scope="module")
def casos(cfg_real, modelo_json, df_test, pre_json) -> dict:
    """``casos_paridad.json`` de la logística real con los parámetros de la config."""
    exp = cfg_real["export"]
    df_casos = elegir_casos(df_test, pre_json, exp["n_casos_paridad"], exp["n_forzados_por_grupo"], exp["base_seed"])
    return construir_casos_paridad(modelo_json, df_casos)


def _limpiar_independiente(serie: pd.Series) -> pd.Series:
    """Limpieza de categóricas del núcleo §5.2 escrita aquí con pandas: strip y vacío (o NA) → DESCONOCIDO."""
    s = serie.astype("string").str.strip()
    return s.fillna(DESCONOCIDO).replace("", DESCONOCIDO).astype(object)


def _variable(pre_json: dict, nombre: str) -> dict:
    return next(v for v in pre_json["variables"] if v["nombre"] == nombre)


# ------------------------------------------------------------------------------------------------------------------
# 0. Constantes públicas
# ------------------------------------------------------------------------------------------------------------------


def test_00_constantes_publicas():
    assert VERSION_CONTRATO == 1
    assert MODELOS == ("logistica", "mlp")
    assert Path(RUTA_CONFIG) == Path("configs/scorecard/export.yaml")
    assert TOLERANCIA_PARIDAD == {"pd": 1e-6, "score": 1e-3}  # núcleo §5.5
    assert ARCHIVO_MODELO == "pd_model.json"
    assert ARCHIVO_PARIDAD == "casos_paridad.json"


# ------------------------------------------------------------------------------------------------------------------
# 1. Contrato §5.5: primer nivel
# ------------------------------------------------------------------------------------------------------------------


def test_01_contrato_primer_nivel(modelo_json):
    assert set(modelo_json) == CLAVES_PRIMER_NIVEL
    assert modelo_json["version_contrato"] == 1 and isinstance(modelo_json["version_contrato"], int)
    assert isinstance(modelo_json["generado"], str)
    datetime.fromisoformat(modelo_json["generado"])  # ISO 8601 parseable (ValueError si no)
    assert isinstance(modelo_json["commit"], str) and modelo_json["commit"]
    assert modelo_json["modelo"] == "logistica"
    assert isinstance(modelo_json["entradas"], list)
    assert isinstance(modelo_json["preprocesamiento"], dict)
    assert set(modelo_json["red"]) == {"capas", "salida"}
    assert set(modelo_json["scorecard"]) == {"base", "odds_base", "pdo", "factor", "offset"}
    assert set(modelo_json["poblacion"]) == {"percentiles_score", "referencia"}
    assert set(modelo_json["metricas"]) == {"auc_test", "ks_test", "brier_test"}
    assert all(isinstance(v, float) for v in modelo_json["metricas"].values())


def test_01b_commit_actual_es_sha_o_desconocido():
    commit = commit_actual()
    assert isinstance(commit, str)
    assert commit == "desconocido" or re.fullmatch(r"[0-9a-f]{40}", commit)


# ------------------------------------------------------------------------------------------------------------------
# 2. Entradas del formulario
# ------------------------------------------------------------------------------------------------------------------


def test_02_entradas_solo_en_app_y_en_orden(modelo_json, nombres_en_app):
    assert [e["nombre"] for e in modelo_json["entradas"]] == nombres_en_app


def test_02b_entradas_iguales_a_entradas_app_con_train(modelo_json, features, pre_json, df_train, cfg_real):
    """construir_pd_model arma el formulario con train (no con todos los datos ni con test: núcleo §5.7)."""
    assert modelo_json["entradas"] == entradas_app(features, pre_json, df_train, cfg_real["export"]["pasos"])


def test_02c_entradas_claves_exactas_y_textos(modelo_json, features):
    por_nombre = {f["nombre"]: f for f in features}
    for e in modelo_json["entradas"]:
        f = por_nombre[e["nombre"]]
        assert e["tipo"] == f["tipo"]
        assert set(e) == (CLAVES_NUM if f["tipo"] == "num" else CLAVES_CAT), e["nombre"]
        assert e["etiqueta_es"] == f["etiqueta_es"]
        assert e["ayuda_es"] == f["ayuda_es"]


def _es_multiplo(x: float, paso: float) -> bool:
    q = x / paso
    return abs(q - round(q)) <= TOL_REL_MULTIPLO * max(1.0, abs(q))


def test_02d_numericas_rejilla_del_paso(modelo_json, pre_json):
    """Redondeo hacia afuera a la rejilla del paso anclada en 0: el rango del formulario cubre todo el recorte y se
    pasa en menos de un paso (fuera del recorte el modelo es plano)."""
    for e in (e for e in modelo_json["entradas"] if e["tipo"] == "num"):
        v = _variable(pre_json, e["nombre"])
        lo, hi = v["recorte"]
        paso = e["paso"]
        assert paso == PASOS_ESPERADOS[e["nombre"]]
        assert e["min"] <= lo and lo - e["min"] < paso, e["nombre"]
        assert e["max"] >= hi and e["max"] - hi < paso, e["nombre"]
        for clave in ("min", "max", "valor_defecto"):
            assert _es_multiplo(e[clave], paso), (e["nombre"], clave, e[clave])
        assert e["min"] <= e["valor_defecto"] <= e["max"]
        acotado = e["valor_defecto"] in (e["min"], e["max"])
        assert acotado or abs(e["valor_defecto"] - v["mediana"]) <= paso / 2 + 1e-9, e["nombre"]


# Casos a mano (recorte y mediana de models/preprocesador_pd.json; cuentas hechas con Python):
# - loan_amnt, paso 25: recorte [1350, 35000] → 1350/25 = 54 y 35000/25 = 1400 (exactos) → min 1350, max 35000;
#   mediana 10000 → 10000.
# - annual_inc, paso 1000: recorte [15058.54, 230000] → floor(15.05854) = 15 → 15000; ceil(230) = 230 → 230000;
#   mediana 60000 → 60000.
# - dti, paso 0.1: recorte [0.6346…, 32.5] → floor(6.346) = 6 → 0.6 (y no 0.6000000000000001); 32.5/0.1 = 325 → 32.5;
#   mediana 14.99 → 149.9 → 150 → 15.0.
# - delinq_2yrs, paso 1: recorte [0, 3], mediana 0 → 0, 3, 0.
# - antiguedad_credito_anios, paso 1: recorte [3.5044…, 36] → 3, 36; mediana 12.8378… → 13.
# - inq_last_6mths, paso 1: recorte [0, 5], mediana 1 → 0, 5, 1.
# - open_acc, paso 1: recorte [2, 23], mediana 9 → 2, 23, 9.
# - pub_rec, paso 1: recorte [0, 1], mediana 0 → 0, 1, 0.
# - revol_bal, paso 100: recorte [0, 81831.62] → 0; ceil(818.3162) = 819 → 81900; mediana 10392 → 103.92 → 104 → 10400.
# - revol_util, paso 0.1: recorte [0, 97.6] → 0; 97.6/0.1 = 975.9999999999999 → ceil 976 → 97.6 (y no
#   97.60000000000001); mediana 56.4 → 564 → 56.4 (y no 56.400000000000006).
# - total_acc, paso 1: recorte [4, 56], mediana 21 → 4, 56, 21.
CASOS_NUM_A_MANO = {
    "loan_amnt": (1350, 35000, 10000),
    "annual_inc": (15000, 230000, 60000),
    "dti": (0.6, 32.5, 15.0),
    "delinq_2yrs": (0, 3, 0),
    "antiguedad_credito_anios": (3, 36, 13),
    "inq_last_6mths": (0, 5, 1),
    "open_acc": (2, 23, 9),
    "pub_rec": (0, 1, 0),
    "revol_bal": (0, 81900, 10400),
    "revol_util": (0, 97.6, 56.4),
    "total_acc": (4, 56, 21),
}


@pytest.mark.parametrize("nombre", sorted(CASOS_NUM_A_MANO))
def test_02e_numericas_a_mano(modelo_json, nombre):
    """Igualdad EXACTA: el resultado se redondea a los decimales del paso (sin artefactos de coma flotante)."""
    e = next(e for e in modelo_json["entradas"] if e["nombre"] == nombre)
    assert (e["min"], e["max"], e["valor_defecto"]) == CASOS_NUM_A_MANO[nombre]


def test_02f_categoricas(modelo_json, features, pre_json, df_train):
    por_nombre = {f["nombre"]: f for f in features}
    for e in (e for e in modelo_json["entradas"] if e["tipo"] == "cat"):
        f = por_nombre[e["nombre"]]
        cats_pre = [c for c in _variable(pre_json, e["nombre"])["categorias"] if c != OTRA]
        assert OTRA not in e["categorias"]
        assert e["categorias"] == list(f["categorias_es"])  # orden de categorias_es
        assert set(e["categorias"]) == set(cats_pre)  # conjunto = preprocesador − OTRA
        assert e["etiquetas_categorias"] == f["categorias_es"]
        assert all(isinstance(e["etiquetas_categorias"][c], str) for c in e["categorias"])
        moda = _limpiar_independiente(df_train[e["nombre"]]).value_counts().idxmax()  # moda de train tras limpiar
        assert e["valor_defecto"] == moda, e["nombre"]
        assert e["valor_defecto"] in e["categorias"]


@pytest.mark.parametrize("cambio", ["falta", "sobra"])
def test_02g_categorias_es_incoherente_es_error(features, pre_json, df_train, cambio):
    features_mal = copy.deepcopy(features)
    term = next(f for f in features_mal if f["nombre"] == "term")
    if cambio == "falta":
        del term["categorias_es"]["60 months"]
    else:
        term["categorias_es"]["120 months"] = "120 meses"
    with pytest.raises(ValueError):
        entradas_app(features_mal, pre_json, df_train, PASOS_ESPERADOS)


def _formulario_de_juguete(mediana: float, filas_cat: list) -> list[dict]:
    """entradas_app con una numérica (recorte [0, 1], paso 0.1) y una categórica A/B (categorias_es en orden B, A)."""
    features = [
        {"nombre": "x", "tipo": "num", "en_app": True, "etiqueta_es": "X", "ayuda_es": "x"},
        {"nombre": "c", "tipo": "cat", "en_app": True, "etiqueta_es": "C", "ayuda_es": "c",
         "categorias_es": {"B": "Be", "A": "A"}},
    ]
    pre_json = {"categoricas": {"limpieza": "strip", "vacio": DESCONOCIDO, "no_vista": OTRA},
                "variables": [{"nombre": "x", "tipo": "num", "mediana": mediana, "recorte": [0.0, 1.0], "log1p": False,
                               "media": 0.0, "desv": 1.0},
                              {"nombre": "c", "tipo": "cat", "categorias": ["A", "B", OTRA]}]}
    df_train = pd.DataFrame({"x": [0.0] * len(filas_cat), "c": filas_cat})
    return entradas_app(features, pre_json, df_train, {"x": 0.1})


def test_02h_empate_de_la_mediana_hacia_arriba():
    # 0.15/0.1 = 1.4999999999999998 en binario: es el empate 1.5 salvo coma flotante → hacia arriba → 2 · 0.1 = 0.2.
    assert _formulario_de_juguete(0.15, ["A"])[0]["valor_defecto"] == 0.2
    # 0.14/0.1 = 1.4000000000000001 → 1 → 0.1 (no es empate).
    assert _formulario_de_juguete(0.14, ["A"])[0]["valor_defecto"] == 0.1


def test_02i_moda_con_empate_por_orden_de_categorias_es():
    # A y B salen una vez cada una: empate → la primera en el orden de categorias_es (B, A) → "B".
    assert _formulario_de_juguete(0.5, ["A", "B"])[1]["valor_defecto"] == "B"
    # Sin empate gana la más frecuente aunque vaya después en categorias_es.
    assert _formulario_de_juguete(0.5, ["A", "A", "B"])[1]["valor_defecto"] == "A"


# ------------------------------------------------------------------------------------------------------------------
# 3. Preprocesamiento
# ------------------------------------------------------------------------------------------------------------------


def test_03_preprocesamiento_es_to_dict(modelo_json, pre):
    assert modelo_json["preprocesamiento"] == pre.to_dict()


# ------------------------------------------------------------------------------------------------------------------
# 4. Red
# ------------------------------------------------------------------------------------------------------------------


def _comprobar_red(red: dict, n_entradas: int) -> None:
    assert red["salida"] == "sigmoid"
    capas = red["capas"]
    assert len(capas) >= 1
    entrada = n_entradas
    for i, capa in enumerate(capas):
        assert set(capa) == {"W", "b", "activacion"}
        W = np.asarray(capa["W"], dtype=float)
        assert W.ndim == 2 and W.shape[1] == entrada, f"capa {i}: W {W.shape} no recibe {entrada}"  # salida × entrada
        assert len(capa["b"]) == W.shape[0]
        assert capa["activacion"] in ACTIVACIONES_JSON
        entrada = W.shape[0]
    assert entrada == 1  # una sola salida: el logit
    assert capas[-1]["activacion"] == "identidad"


def test_04_red_logistica(modelo_json, pre_json, logistica):
    n = len(pre_json["columnas_salida"])
    assert n == 48
    _comprobar_red(modelo_json["red"], n)
    w, b = logistica.coeficientes()
    assert len(modelo_json["red"]["capas"]) == 1
    capa = modelo_json["red"]["capas"][0]
    np.testing.assert_array_equal(np.asarray(capa["W"]), w[None, :])
    assert capa["b"] == [b]
    assert capas_logistica(logistica) == [{"W": [w.tolist()], "b": [b], "activacion": "identidad"}]


# ------------------------------------------------------------------------------------------------------------------
# 5. Scorecard y población
# ------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def score_modelo_test(logistica, pre, df_test, params_sc) -> np.ndarray:
    """Puntaje de la logística real en TODO el test, por el camino de Python (preprocesador + modelo)."""
    return score_desde_logit(logistica.predict_logit(pre.transform(df_test)), params_sc)


def test_05_scorecard_es_la_de_c1(modelo_json, params_sc):
    assert modelo_json["scorecard"] == params_sc
    assert (modelo_json["scorecard"]["base"], modelo_json["scorecard"]["odds_base"],
            modelo_json["scorecard"]["pdo"]) == (600, 50, 20)


def test_05b_poblacion(modelo_json, score_modelo_test):
    pob = modelo_json["poblacion"]
    assert pob["referencia"] == "test"
    pct = np.asarray(pob["percentiles_score"], dtype=float)
    assert pct.shape == (101,)
    assert np.all(np.diff(pct) >= 0)
    # Camino de Python (predict_logit sobre pre.transform(test) + score_desde_logit de C1), el mismo de las métricas
    # (decisión de Juan José, 25-oct); 1e-9 solo absorbe el redondeo de la interpolación de np.percentile.
    tol = TOL_PERCENTILES
    assert pct[0] == pytest.approx(score_modelo_test.min(), abs=tol)
    assert pct[100] == pytest.approx(score_modelo_test.max(), abs=tol)
    np.testing.assert_allclose(pct, np.percentile(score_modelo_test, np.arange(101)), rtol=0, atol=tol)


def test_05c_percentiles_score_a_mano():
    # np.percentile(0..100, k) = k (interpolación lineal sobre 101 puntos equiespaciados).
    assert percentiles_score(np.arange(101.0)) == pytest.approx(list(range(101)))
    # Con [0, 10]: el percentil k es 10·k/100 = k/10 (lineal).
    r = percentiles_score(np.array([10.0, 0.0]))
    assert len(r) == 101 and r[0] == 0.0 and r[25] == pytest.approx(2.5) and r[100] == 10.0
    assert all(isinstance(x, float) for x in r)


# ------------------------------------------------------------------------------------------------------------------
# 6. Métricas
# ------------------------------------------------------------------------------------------------------------------


def test_06_metricas_iguales_al_summary(modelo_json):
    summary = load_json(RUTA_SUMMARY)["logistica"]["metricas"]
    for nombre in ("auc_test", "ks_test", "brier_test"):
        assert modelo_json["metricas"][nombre] == pytest.approx(summary[nombre]["media"], abs=1e-12), nombre


def test_06b_metricas_test_usa_las_comunes():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.7])
    m = metricas_test(y, p)
    assert set(m) == {"auc_test", "ks_test", "brier_test"}
    assert m["auc_test"] == auc(y, p) and m["ks_test"] == ks(y, p) and m["brier_test"] == brier(y, p)


# ------------------------------------------------------------------------------------------------------------------
# 7. Referencia: lo que calcula el JSON == lo que calcula el modelo de Python
# ------------------------------------------------------------------------------------------------------------------


def test_07_preprocesar_desde_json_igual_al_preprocesador(modelo_json, pre, df_test):
    X_json = preprocesar_desde_json(modelo_json["preprocesamiento"], df_test)
    X_py = pre.transform(df_test)  # float32 (núcleo §5.2)
    assert X_json.dtype == np.float64 and X_json.shape == X_py.shape
    print(f"\nmáx |dX| JSON vs Preprocesador = {np.max(np.abs(X_json - X_py)):.3e}")
    np.testing.assert_allclose(X_json, X_py, rtol=1e-6, atol=1e-6)  # diferencia = redondeo de float32


def test_07b_preprocesar_desde_json_a_mano(modelo_json, pre_json):
    """Una fila con recortes, vacíos, espacios y categorías no vistas; cuentas con models/preprocesador_pd.json."""
    fila = {v["nombre"]: v["mediana"] for v in pre_json["variables"] if v["tipo"] == "num"}
    fila.update({"loan_amnt": 50000.0, "dti": None, "revol_util": -5.0, "term": " 60 months", "emp_length": None,
                 "home_ownership": "ANY", "purpose": ""})
    X = preprocesar_desde_json(modelo_json["preprocesamiento"], [fila])
    col = {c: i for i, c in enumerate(pre_json["columnas_salida"])}
    assert X.shape == (1, 48)
    # loan_amnt 50000 → recorte 35000 → (log1p(35000) − 9.17214254609383) / 0.7057320968696751
    assert X[0, col["loan_amnt"]] == pytest.approx(1.8292909889240003, abs=1e-12)
    # dti vacío → mediana 14.99 → (14.99 − 15.093069702369752) / 7.352874379877304
    assert X[0, col["dti"]] == pytest.approx(-0.014017606862946464, abs=1e-12)
    # revol_util −5 → recorte 0 → (0 − 53.75247124973712) / 26.52542241432555
    assert X[0, col["revol_util"]] == pytest.approx(-2.0264510932239514, abs=1e-12)
    # " 60 months" → strip → "60 months"; emp_length vacío → DESCONOCIDO; "ANY" no visto → OTRA; "" → DESCONOCIDO,
    # que no es categoría de purpose en train → OTRA.
    unos = {"term=60 months", "emp_length=DESCONOCIDO", "home_ownership=OTRA", "purpose=OTRA"}
    for c, i in col.items():
        if "=" in c:
            assert X[0, i] == (1.0 if c in unos else 0.0), c


@pytest.mark.parametrize("activacion, esperado", [
    ("relu", 2.0),  # relu([-1, 2]) = [0, 2] → 0 + 2
    ("leaky_relu", 1.99),  # pendiente 0.01 de nn.LeakyReLU: [-0.01, 2] → 1.99
    ("tanh", 0.20243342412005205),  # tanh(−1) + tanh(2) = −0.7615941559557649 + 0.9640275800758169
])
def test_07c_forward_desde_json_a_mano(activacion, esperado):
    red = {"capas": [{"W": [[1.0, 0.0], [0.0, 1.0]], "b": [0.0, 0.0], "activacion": activacion},
                     {"W": [[1.0, 1.0]], "b": [0.0], "activacion": "identidad"}], "salida": "sigmoid"}
    z = forward_desde_json(red, np.array([[-1.0, 2.0]]))
    assert z.shape == (1,)
    assert z[0] == pytest.approx(esperado, abs=1e-12)


def test_07d_forward_identidad_a_mano():
    # 1·1 + (−2)·1 + 0.5 = −0.5; W de 1 × 2 (salida × entrada).
    red = {"capas": [{"W": [[1.0, -2.0]], "b": [0.5], "activacion": "identidad"}], "salida": "sigmoid"}
    np.testing.assert_allclose(forward_desde_json(red, np.array([[1.0, 1.0], [0.0, 0.0]])), [-0.5, 0.5])


def test_07d2_vacios_y_textos_raros_a_mano(modelo_json, pre_json):
    """Lo que los casos de paridad (datos reales) no cubren: vacíos escritos de otras formas y no textos."""
    fila = {v["nombre"]: v["mediana"] for v in pre_json["variables"] if v["tipo"] == "num"}
    fila.update({"dti": pd.NA, "revol_util": "   ", "annual_inc": "", "loan_amnt": " 10000 ",
                 "term": "nan", "emp_length": pd.NA, "home_ownership": 36, "purpose": pd.NA})
    X = preprocesar_desde_json(modelo_json["preprocesamiento"], [fila])
    col = {c: i for i, c in enumerate(pre_json["columnas_salida"])}
    # Numéricas vacías (pd.NA, "   ", "") → mediana de train:
    # dti: (14.99 − 15.093069702369752) / 7.352874379877304
    assert X[0, col["dti"]] == pytest.approx(-0.014017606862946464, abs=1e-12)
    # revol_util: (56.4 − 53.75247124973712) / 26.52542241432555
    assert X[0, col["revol_util"]] == pytest.approx(0.09981099297529118, abs=1e-12)
    # annual_inc: (log1p(60000) − 10.981100232922868) / 0.5234929956042197
    assert X[0, col["annual_inc"]] == pytest.approx(0.04014623879521442, abs=1e-12)
    # " 10000 " es un número con espacios → 10000, igual que el valor numérico 10000.
    X_num = preprocesar_desde_json(modelo_json["preprocesamiento"], [dict(fila, loan_amnt=10000.0)])
    assert X[0, col["loan_amnt"]] == X_num[0, col["loan_amnt"]]
    # Categóricas: "nan" es un texto real (no vacío) → OTRA; pd.NA → DESCONOCIDO, que en emp_length existe y en purpose
    # no se vio en train → OTRA; 36 (no texto) → "36" → OTRA.
    unos = {"term=OTRA", "emp_length=DESCONOCIDO", "home_ownership=OTRA", "purpose=OTRA"}
    for c, i in col.items():
        if "=" in c:
            assert X[0, i] == (1.0 if c in unos else 0.0), c


def test_07d3_numerica_con_texto_no_numerico_es_error(modelo_json, pre_json):
    fila = {v["nombre"]: v["mediana"] for v in pre_json["variables"] if v["tipo"] == "num"}
    fila.update({"dti": "abc", "term": "36 months", "emp_length": "1 year", "home_ownership": "RENT",
                 "purpose": "car"})
    with pytest.raises(ValueError):
        preprocesar_desde_json(modelo_json["preprocesamiento"], [fila])


@pytest.mark.parametrize("caso", ["activacion", "W_no_encaja", "b_no_encaja", "dos_salidas"])
def test_07d4_forward_errores(caso):
    capa = {"W": [[1.0, 1.0]], "b": [0.0], "activacion": "identidad"}
    if caso == "activacion":
        capa["activacion"] = "sigmoid"  # no es una activación de capa del contrato
    elif caso == "W_no_encaja":
        capa["W"] = [[1.0, 1.0, 1.0]]  # 3 entradas para una X de 2 columnas
    elif caso == "b_no_encaja":
        capa["b"] = [0.0, 0.0]  # 2 sesgos para 1 salida
    else:
        capa = {"W": [[1.0, 1.0], [1.0, 1.0]], "b": [0.0, 0.0], "activacion": "identidad"}  # termina en 2 salidas
    with pytest.raises(ValueError):
        forward_desde_json({"capas": [capa], "salida": "sigmoid"}, np.ones((3, 2)))


def test_07d5_inferir_rechaza_otra_salida(modelo_json, df_test):
    otro = {**modelo_json, "red": {**modelo_json["red"], "salida": "softmax"}}
    with pytest.raises(ValueError):
        inferir_desde_json(otro, df_test.iloc[:3])


def _pre_alterado(pre_json: dict, caso: str) -> dict:
    pre_mal = copy.deepcopy(pre_json)
    if caso == "version":
        pre_mal["version"] = 2
    elif caso == "operacion":
        pre_mal["orden_numericas"] = ["imputar_mediana", "raiz_cuadrada"]
    elif caso == "limpieza":
        pre_mal["categoricas"]["limpieza"] = "lower"
    elif caso == "nombres":
        a, b = pre_mal["columnas_salida"][1], pre_mal["columnas_salida"][2]  # term=36 months ↔ term=60 months
        pre_mal["columnas_salida"][1], pre_mal["columnas_salida"][2] = b, a  # mismo número, otro orden
    return pre_mal


@pytest.mark.parametrize("caso", ["version", "operacion", "limpieza", "nombres"])
def test_07d6_preprocesar_rechaza_json_alterado(pre_json, df_test, caso):
    with pytest.raises(ValueError):
        preprocesar_desde_json(_pre_alterado(pre_json, caso), df_test.iloc[:3])


def test_07d7_preprocesar_rechaza_entrada_faltante(pre_json, df_test, nombres_en_app):
    filas = df_test.iloc[:2][nombres_en_app].to_dict(orient="records")
    del filas[1]["dti"]  # el segundo diccionario no trae dti
    with pytest.raises(ValueError):
        preprocesar_desde_json(pre_json, filas)


def test_07e_logistica_json_vs_modelo_en_todo_el_test(modelo_json, logistica, pre, df_test, params_sc):
    pd_json, score_json = inferir_desde_json(modelo_json, df_test)
    X = pre.transform(df_test)
    pd_py = logistica.predict_proba(X)
    score_py = score_desde_logit(logistica.predict_logit(X), params_sc)
    d_pd, d_score = np.max(np.abs(pd_json - pd_py)), np.max(np.abs(score_json - score_py))
    print(f"\nlogística, {len(df_test)} filas de test: máx |dPD| = {d_pd:.3e}, máx |dscore| = {d_score:.3e}")
    assert pd_json.shape == score_json.shape == (len(df_test),)
    assert d_pd < TOLERANCIA_PARIDAD["pd"]
    assert d_score < TOLERANCIA_PARIDAD["score"]


@pytest.fixture(scope="module", params=["relu", "leaky_relu", "tanh"])
def mlp_pequeno(request, tmp_path_factory, pre, df_train):
    """MLPPD pequeño entrenado con una etiqueta SINTÉTICA (regla fija sobre dti y revol_util estandarizados) en una
    submuestra de train; guardado en un temporal. Devuelve (ruta .pt, activación)."""
    rng = make_rng(SEMILLA_PRUEBA, 0)
    idx = rng.choice(len(df_train), size=N_SUBMUESTRA_MLP, replace=False)
    X = pre.transform(df_train.iloc[idx])
    col = {c: i for i, c in enumerate(pre.nombres_salida())}
    y = (X[:, col["dti"]] + X[:, col["revol_util"]] > 0).astype(np.int64)  # regla fija, no el objetivo real
    cfg = {"mlp": {**CFG_MLP_PEQUENO, "activacion": request.param}}
    n_tr = N_SUBMUESTRA_MLP - N_VAL_MLP
    modelo = MLPPD().fit(X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:], cfg, seed=SEMILLA_PRUEBA)
    ruta = tmp_path_factory.mktemp(f"mlp_{request.param}") / "pd_mlp.pt"
    modelo.guardar(ruta)
    return ruta, request.param


def test_07f_capas_mlp(mlp_pequeno, pre_json):
    ruta, activacion = mlp_pequeno
    capas, modelo = capas_mlp(ruta)
    assert isinstance(modelo, MLPPD)
    estado = torch.load(ruta, map_location="cpu", weights_only=True)["state_dict"]
    pesos = [k for k in estado if k.endswith(".weight")]  # una por nn.Linear, en orden (Dropout no tiene pesos)
    assert len(capas) == len(pesos) == len(CFG_MLP_PEQUENO["capas"]) + 1
    for capa, k in zip(capas, pesos):
        np.testing.assert_array_equal(np.asarray(capa["W"]), estado[k].numpy())  # salida × entrada
        np.testing.assert_array_equal(np.asarray(capa["b"]), estado[k.replace(".weight", ".bias")].numpy())
    assert [c["activacion"] for c in capas] == [activacion] * len(CFG_MLP_PEQUENO["capas"]) + ["identidad"]
    _comprobar_red({"capas": capas, "salida": "sigmoid"}, len(pre_json["columnas_salida"]))


def test_07g_mlp_json_vs_modelo_en_todo_el_test(mlp_pequeno, modelo_json, pre, df_test, params_sc):
    """Receta del JSON con las capas del MLP (sin reconstruir todo pd_model.json) vs MLPPD en todo el test."""
    ruta, activacion = mlp_pequeno
    capas, modelo = capas_mlp(ruta)
    json_mlp = {**copy.deepcopy(modelo_json), "modelo": "mlp", "red": {"capas": capas, "salida": "sigmoid"}}
    X = pre.transform(df_test)
    pd_json, score_json = inferir_desde_json(json_mlp, df_test)
    pd_py = modelo.predict_proba(X)
    score_py = score_desde_logit(modelo.predict_logit(X), params_sc)
    d_pd, d_score = np.max(np.abs(pd_json - pd_py)), np.max(np.abs(score_json - score_py))
    print(f"\nMLP pequeño ({activacion}), {len(df_test)} filas de test: máx |dPD| = {d_pd:.3e}, "
          f"máx |dscore| = {d_score:.3e}")
    assert d_pd < COTA_MLP["pd"]
    assert d_score < COTA_MLP["score"]


@pytest.mark.parametrize("mlp_pequeno", ["relu"], indirect=True)
def test_07h_construir_pd_model_con_mlp(mlp_pequeno, cfg_real, pre, df_test, params_sc):
    """pd_model.json completo con modelo: mlp (una sola activación): red, métricas y percentiles del propio MLP."""
    ruta, _ = mlp_pequeno
    cfg = copy.deepcopy(cfg_real)
    cfg["export"]["modelo"] = "mlp"
    cfg["export"]["rutas"]["mlp"] = str(ruta)
    modelo_json = construir_pd_model(cfg)
    assert modelo_json["modelo"] == "mlp"
    _comprobar_red(modelo_json["red"], len(pre.nombres_salida()))
    assert modelo_json["red"]["capas"] == capas_mlp(ruta)[0]
    modelo = MLPPD.cargar(ruta)
    X = pre.transform(df_test)
    pd_py = modelo.predict_proba(X)
    m = modelo_json["metricas"]
    assert m["auc_test"] == pytest.approx(auc(df_test["target"].to_numpy(), pd_py), abs=1e-9)
    score_py = score_desde_logit(modelo.predict_logit(X), params_sc)
    np.testing.assert_allclose(modelo_json["poblacion"]["percentiles_score"],
                               np.percentile(score_py, np.arange(101)), rtol=0, atol=TOL_PERCENTILES)


# ------------------------------------------------------------------------------------------------------------------
# 8. Casos de paridad
# ------------------------------------------------------------------------------------------------------------------


def _grupos_forzados(df_test: pd.DataFrame, pre_json: dict, nombres_en_app: list[str]) -> dict[str, set]:
    """Grupos que deben quedar representados, calculados aquí de forma independiente: vacíos por variable, categorías
    no vistas en train (→ OTRA) y, por numérica, filas por debajo y por encima del recorte."""
    grupos = {}
    for nombre in nombres_en_app:
        v = _variable(pre_json, nombre)
        s = df_test[nombre]
        if v["tipo"] == "num":
            lo, hi = v["recorte"]
            vacio = s.isna()
            grupos[f"bajo:{nombre}"] = set(df_test.loc[s < lo, "id"])
            grupos[f"alto:{nombre}"] = set(df_test.loc[s > hi, "id"])
        else:
            limpia = s.astype("string").str.strip()
            vacio = limpia.isna() | (limpia == "")
            no_vista = ~_limpiar_independiente(s).isin(v["categorias"])
            grupos[f"otra:{nombre}"] = set(df_test.loc[no_vista, "id"])
        grupos[f"vacio:{nombre}"] = set(df_test.loc[vacio.to_numpy(), "id"])
    return {g: ids for g, ids in grupos.items() if ids}


def test_08_casos_forma_y_tolerancia(casos, cfg_real, df_test):
    assert set(casos) == {"version_contrato", "modelo", "tolerancia", "casos"}
    assert casos["version_contrato"] == 1
    assert casos["modelo"] == "logistica"
    assert casos["tolerancia"] == {"pd": 1e-6, "score": 1e-3}
    filas = casos["casos"]
    assert len(filas) == cfg_real["export"]["n_casos_paridad"] == 1000
    ids = [c["id"] for c in filas]
    assert len(set(ids)) == len(ids)
    assert set(ids) <= set(df_test["id"])
    assert all(set(c) == {"id", "entradas", "pd", "score"} for c in filas)


def test_08b_casos_cubren_los_grupos_forzados(casos, cfg_real, df_test, pre_json, nombres_en_app):
    ids = {c["id"] for c in casos["casos"]}
    n_g = cfg_real["export"]["n_forzados_por_grupo"]
    grupos = _grupos_forzados(df_test, pre_json, nombres_en_app)
    # Los que la tarea nombra explícitamente existen en el test real (si no, la prueba no probaría nada).
    for g in ("vacio:emp_length", "vacio:revol_util", "otra:home_ownership"):
        assert g in grupos, g
    otra_home = df_test.loc[df_test["home_ownership"].astype("string").str.strip() == "ANY", "id"]
    assert set(otra_home) <= ids  # la fila con home_ownership no visto (ANY → OTRA)
    faltan = {g: (len(ids & m), min(n_g, len(m))) for g, m in grupos.items() if len(ids & m) < min(n_g, len(m))}
    assert not faltan, f"grupos sin sus filas forzadas (tienen, deben): {faltan}"
    for nombre in (n for n in nombres_en_app if _variable(pre_json, n)["tipo"] == "num"):
        for lado in ("bajo", "alto"):
            if f"{lado}:{nombre}" in grupos:
                assert ids & grupos[f"{lado}:{nombre}"], f"{lado}:{nombre}"


def test_08c_casos_reproducibles(casos, df_test, pre_json, cfg_real):
    exp = cfg_real["export"]
    b = elegir_casos(df_test, pre_json, exp["n_casos_paridad"], exp["n_forzados_por_grupo"], exp["base_seed"])
    assert [c["id"] for c in casos["casos"]] == list(b["id"])  # misma semilla → mismas filas y mismo orden
    # Otra semilla con n pequeño (1 forzada por grupo: 19 grupos en el test real, caben en 40) → otras filas.
    c = elegir_casos(df_test, pre_json, N_CASOS_PEQUENO, 1, exp["base_seed"])
    d = elegir_casos(df_test, pre_json, N_CASOS_PEQUENO, 1, exp["base_seed"] + 1)
    assert len(c) == len(d) == N_CASOS_PEQUENO
    assert set(c["id"]) != set(d["id"])


def test_08c2_elegir_casos_errores(df_test, pre_json, cfg_real):
    exp = cfg_real["export"]
    with pytest.raises(ValueError):  # 19 grupos con hasta 10 forzadas cada uno no caben en 5 casos
        elegir_casos(df_test, pre_json, 5, exp["n_forzados_por_grupo"], exp["base_seed"])
    with pytest.raises(ValueError):  # 50 filas no alcanzan para 1 000 casos
        elegir_casos(df_test.iloc[:50], pre_json, exp["n_casos_paridad"], exp["n_forzados_por_grupo"],
                     exp["base_seed"])


def test_08d_entradas_crudas(casos, df_test, nombres_en_app):
    crudo = df_test.set_index("id")
    alguno_con_espacio = False
    for caso in casos["casos"]:
        assert set(caso["entradas"]) == set(nombres_en_app)  # exactamente las variables en_app
        fila = crudo.loc[caso["id"]]
        for nombre, valor in caso["entradas"].items():
            original = fila[nombre]
            if pd.isna(original):
                assert valor is None, (caso["id"], nombre)
            else:
                assert valor == original, (caso["id"], nombre)  # textos SIN limpiar
        alguno_con_espacio |= caso["entradas"]["term"] in (" 36 months", " 60 months")
    assert alguno_con_espacio
    assert any(c["entradas"]["emp_length"] is None for c in casos["casos"])
    assert any(c["entradas"]["revol_util"] is None for c in casos["casos"])


def test_08e_score_de_los_casos_es_el_de_c1(casos, modelo_json):
    pd_c = np.array([c["pd"] for c in casos["casos"]])
    score_c = np.array([c["score"] for c in casos["casos"]])
    np.testing.assert_allclose(score_c, score_desde_pd(pd_c, modelo_json["scorecard"]), rtol=0, atol=1e-9)


def test_08f_casos_reproducen_la_receta_desde_list_dict(casos, modelo_json):
    """La app recibe las entradas crudas del JSON (con None): la receta debe dar la misma PD y el mismo puntaje."""
    entradas = [c["entradas"] for c in casos["casos"]]
    pd_r, score_r = inferir_desde_json(modelo_json, entradas)
    np.testing.assert_allclose(pd_r, [c["pd"] for c in casos["casos"]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(score_r, [c["score"] for c in casos["casos"]], rtol=0, atol=1e-9)


# ------------------------------------------------------------------------------------------------------------------
# 9. Exportar escribe los dos archivos
# ------------------------------------------------------------------------------------------------------------------


def _rechazar_constante(nombre):
    raise ValueError(f"JSON con {nombre}")


def _huellas(carpeta: Path) -> dict[str, str]:
    import hashlib

    if not carpeta.exists():
        return {}
    return {r.relative_to(carpeta).as_posix(): hashlib.sha256(r.read_bytes()).hexdigest()
            for r in sorted(carpeta.rglob("*")) if r.is_file()}


def test_09_exportar_escribe_en_destino(cfg_real, tmp_path):
    cfg = copy.deepcopy(cfg_real)
    cfg["export"]["destino"] = str(tmp_path / "app")
    antes = _huellas(CARPETA_APP_MODELO)
    ruta_modelo, ruta_casos = exportar(cfg)
    assert Path(ruta_modelo) == tmp_path / "app" / ARCHIVO_MODELO
    assert Path(ruta_casos) == tmp_path / "app" / ARCHIVO_PARIDAD
    for ruta in (ruta_modelo, ruta_casos):
        texto = Path(ruta).read_text(encoding="utf-8")
        json.loads(texto, parse_constant=_rechazar_constante)  # sin NaN ni ±inf
    modelo = json.loads(Path(ruta_modelo).read_text(encoding="utf-8"))
    casos = json.loads(Path(ruta_casos).read_text(encoding="utf-8"))
    assert set(modelo) == CLAVES_PRIMER_NIVEL and modelo["modelo"] == "logistica"
    assert len(casos["casos"]) == 1000
    assert _huellas(CARPETA_APP_MODELO) == antes  # nada cambia en app/public/model


# ------------------------------------------------------------------------------------------------------------------
# 10. validar_cfg_export
# ------------------------------------------------------------------------------------------------------------------


def _export_valido() -> dict:
    return {
        "modelo": "logistica",
        "rutas": {"logistica": "models/pd_logistica.joblib", "mlp": "models/pd_mlp.pt",
                  "preprocesador": "models/preprocesador_pd.json", "datos": "data/processed/loans_pd.parquet",
                  "features": "configs/features.yaml"},
        "destino": "app/public/model",
        "n_casos_paridad": 1000,
        "n_forzados_por_grupo": 10,
        "base_seed": 0,
        "pasos": dict(PASOS_ESPERADOS),
    }


def test_10_validar_acepta_la_valida(features):
    exp = _export_valido()
    r = validar_cfg_export(exp, features)
    assert r == _export_valido()
    assert exp == _export_valido()  # no muta la entrada


def _mutar(clave_ruta, valor=None, borrar=False):
    def aplicar(exp):
        nodo = exp
        for k in clave_ruta[:-1]:
            nodo = nodo[k]
        if borrar:
            del nodo[clave_ruta[-1]]
        else:
            nodo[clave_ruta[-1]] = valor
        return exp
    return aplicar


CASOS_INVALIDOS = {
    "falta_pasos": _mutar(("pasos",), borrar=True),
    "falta_destino": _mutar(("destino",), borrar=True),
    "sobra_clave": _mutar(("otra",), 1),
    "modelo_desconocido": _mutar(("modelo",), "lightgbm"),
    "rutas_falta": _mutar(("rutas", "mlp"), borrar=True),
    "rutas_sobra": _mutar(("rutas", "lgd"), "models/lgd_mlp.pt"),
    "pasos_falta": _mutar(("pasos", "loan_amnt"), borrar=True),
    "pasos_sobra_cat": _mutar(("pasos", "term"), 1),
    "pasos_sobra": _mutar(("pasos", "grade"), 1),
    "paso_cero": _mutar(("pasos", "dti"), 0),
    "paso_negativo": _mutar(("pasos", "dti"), -0.1),
    "paso_bool": _mutar(("pasos", "pub_rec"), True),
    "paso_texto": _mutar(("pasos", "loan_amnt"), "25"),
    "paso_nan": _mutar(("pasos", "dti"), float("nan")),
    "paso_inf": _mutar(("pasos", "dti"), math.inf),
    "n_casos_cero": _mutar(("n_casos_paridad",), 0),
    "n_casos_flotante": _mutar(("n_casos_paridad",), 1000.0),
    "n_forzados_cero": _mutar(("n_forzados_por_grupo",), 0),
    "n_forzados_bool": _mutar(("n_forzados_por_grupo",), True),
    "base_seed_negativa": _mutar(("base_seed",), -1),
    "base_seed_texto": _mutar(("base_seed",), "0"),
}


@pytest.mark.parametrize("caso", sorted(CASOS_INVALIDOS))
def test_10b_validar_rechaza(features, caso):
    with pytest.raises(ValueError):
        validar_cfg_export(CASOS_INVALIDOS[caso](_export_valido()), features)


def test_10c_validar_rechaza_si_no_es_dict(features):
    with pytest.raises(ValueError):
        validar_cfg_export(["modelo", "logistica"], features)


# ------------------------------------------------------------------------------------------------------------------
# 11. configs/scorecard/export.yaml
# ------------------------------------------------------------------------------------------------------------------


def test_11_config_export_hereda_base_y_es_valida(features):
    cfg = load_config(RUTA_EXPORT_YAML)
    assert {"scorecard", "export"} <= set(cfg)  # C3 puede añadir bloques a base.yaml
    assert cfg["scorecard"] == {"base": 600, "odds_base": 50, "pdo": 20}
    assert validar_cfg_export(cfg["export"], features) == _export_valido()  # rutas, destino, n, semilla y pasos
